"""
ExtractionManager - Handles LLM extraction and result parsing.
"""

import logging
import json
import os
from typing import Any, Optional, Dict, List

import pandas as pd
import instructor
from pydantic import BaseModel
import tiktoken

from delm.utils.rate_limiter import RateLimiter

# Module-level logger
log = logging.getLogger(__name__)

from delm.schemas import ExtractionSchema
from delm.utils import RetryHandler, ConcurrentProcessor
from delm.config import LLMExtractionConfig
from delm.constants import (
    SYSTEM_CHUNK_COLUMN,
    SYSTEM_CHUNK_ID_COLUMN,
    SYSTEM_BATCH_ID_COLUMN,
    SYSTEM_ERRORS_COLUMN,
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN,
)
from delm.exceptions import InstructorError
from delm.utils.cost_tracker import CostTracker
from delm.core.experiment_manager import BaseExperimentManager
from delm.utils.type_checks import is_pydantic_model
from delm.utils.semantic_cache import SemanticCache, make_cache_key


class ExtractionManager:
    """Handles LLM extraction and result parsing."""

    def __init__(
        self,
        model_config: LLMExtractionConfig,
        extraction_schema: ExtractionSchema,
        cost_tracker: Optional[CostTracker],
        semantic_cache: "SemanticCache",
        rate_limiter: "RateLimiter",
    ):
        """Initialize the ExtractionManager.

        Args:
            model_config: The model configuration.
            extraction_schema: The extraction schema.
            cost_tracker: The cost tracker.
            semantic_cache: The semantic cache.
            rate_limiter: The rate limiter.
        """

        log.debug("Initializing ExtractionManager")
        self.model_config = model_config
        self.temperature = model_config.temperature

        log.debug(
            f"Model config: {self.model_config.model}, temperature: {self.temperature}"
        )

        # Create Instructor client using universal provider interface
        provider_string = self.model_config.get_provider_string()
        log.debug(f"Creating Instructor client with provider: {provider_string}")

        # Convert mode string to instructor.Mode enum if provided
        instructor_mode = None
        if model_config.mode:
            mode_map = {
                "tools": instructor.Mode.TOOLS,
                "json": instructor.Mode.JSON,
                "json_schema": instructor.Mode.JSON_SCHEMA,
                "md_json": instructor.Mode.MD_JSON,
            }
            instructor_mode = mode_map.get(model_config.mode.lower())
            if instructor_mode is None:
                raise ValueError(
                    f"Invalid mode '{model_config.mode}'. "
                    f"Supported modes: {list(mode_map.keys())}"
                )
            log.debug(f"Using Instructor mode: {instructor_mode}")

        if model_config.base_url:
            log.debug(f"Using custom base_url: {model_config.base_url}")
            if model_config.provider == "openai":
                import openai

                openai_client = openai.OpenAI(
                    base_url=model_config.base_url,
                    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                )
                if instructor_mode is not None:
                    self.client = instructor.from_openai(
                        client=openai_client,
                        model=model_config.model,
                        mode=instructor_mode,
                    )
                else:
                    self.client = instructor.from_openai(
                        client=openai_client,
                        model=model_config.model,
                    )
            else:
                if instructor_mode is not None:
                    self.client = instructor.from_provider(
                        model=provider_string,
                        base_url=model_config.base_url,
                        mode=instructor_mode,
                    )
                else:
                    self.client = instructor.from_provider(
                        model=provider_string,
                        base_url=model_config.base_url,
                    )
        else:
            if instructor_mode is not None:
                self.client = instructor.from_provider(
                    provider_string,
                    mode=instructor_mode,
                )
            else:
                self.client = instructor.from_provider(provider_string)

        self.extraction_schema = extraction_schema

        log.debug(
            f"Creating ConcurrentProcessor with max_workers: {model_config.max_workers}"
        )
        self.concurrent_processor = ConcurrentProcessor(
            max_workers=model_config.max_workers
        )

        log.debug(f"Creating RetryHandler with max_retries: {model_config.max_retries}")
        self.retry_handler = RetryHandler(max_retries=model_config.max_retries)

        self.prompt_template = model_config.prompt_template
        self.system_prompt = model_config.system_prompt

        self.cost_tracker = cost_tracker
        self.semantic_cache = semantic_cache
        self.rate_limiter = rate_limiter

        self.max_output_tokens = 0

        log.debug(f"Cost tracking enabled: {self.cost_tracker is not None}")
        log.debug(f"Rate limiter: {type(self.rate_limiter).__name__}")
        log.debug("ExtractionManager initialized successfully")

    def process_with_batching(
        self,
        text_chunks: List[str],
        text_chunk_ids: List[int],
        batch_size: int,
        experiment_manager: "BaseExperimentManager",
        auto_checkpoint: bool = True,
    ) -> pd.DataFrame:
        """
        Process text chunks with persistent batching and checkpointing.

        This method handles the complete extraction pipeline with:
        - Splitting text chunks into batches
        - Processing batches with concurrent execution
        - Saving batch checkpoints for resuming
        - Consolidating results into final DataFrame

        Args:
            text_chunks: The text chunks to process.
            text_chunk_ids: The IDs of the text chunks.
            batch_size: The size of each batch.
            experiment_manager: The experiment manager.
            auto_checkpoint: Whether to auto-checkpoint.

        Returns:
            A DataFrame containing the extracted data.
        """
        from tqdm.auto import tqdm

        log.debug(
            "Starting batch processing with %d chunks, batch_size=%d",
            len(text_chunks),
            batch_size,
        )

        # 1. Discover all unverified batch IDs (checkpoint files that exist)
        if auto_checkpoint:
            log.debug("Auto-checkpoint enabled, discovering existing batch IDs")
            unverified_batch_ids = experiment_manager.get_all_existing_batch_ids()
            log.debug(
                "Found %d unverified batch IDs: %s",
                len(unverified_batch_ids),
                sorted(unverified_batch_ids),
            )
        else:
            log.debug("Auto-checkpoint disabled, starting fresh")
            unverified_batch_ids = set()

        # 2. Attempt to load each batch, classify as verified or corrupted, and count chunks in verified batches
        verified_batch_ids = set()
        corrupted_batch_ids = set()
        already_processed_chunks = 0

        log.debug("Verifying %d existing batch checkpoints", len(unverified_batch_ids))
        for batch_id in sorted(unverified_batch_ids):
            try:
                log.debug("Loading batch checkpoint %d", batch_id)
                batch_df = experiment_manager.load_batch_checkpoint_by_id(batch_id)
                if batch_df is not None:
                    verified_batch_ids.add(batch_id)
                    chunk_count = len(batch_df)
                    already_processed_chunks += chunk_count
                    log.debug(
                        "Batch %d verified successfully with %d chunks",
                        batch_id,
                        chunk_count,
                    )
                else:
                    corrupted_batch_ids.add(batch_id)
                    log.debug("Batch %d is corrupted (None returned)", batch_id)
            except Exception as e:
                log.warning("Failed to load batch %d: %s", batch_id, e)
                corrupted_batch_ids.add(batch_id)

        log.debug(
            "Verified %d batches (%d chunks), corrupted %d batches",
            len(verified_batch_ids),
            already_processed_chunks,
            len(corrupted_batch_ids),
        )

        # 3. Delete corrupted batch files so they can be replaced
        if corrupted_batch_ids:
            log.debug(
                "Deleting %d corrupted batch checkpoints", len(corrupted_batch_ids)
            )
            for batch_id in corrupted_batch_ids:
                try:
                    log.debug("Deleting corrupted batch checkpoint %d", batch_id)
                    deleted = experiment_manager.delete_batch_checkpoint(batch_id)
                    if deleted:
                        log.debug(
                            "Successfully deleted corrupted batch checkpoint %d",
                            batch_id,
                        )
                    else:
                        log.warning(
                            "Failed to delete corrupted batch checkpoint %d (delete returned False)",
                            batch_id,
                        )
                except Exception as e:
                    log.warning("Failed to delete corrupted batch %d: %s", batch_id, e)
        else:
            log.debug("No corrupted batch checkpoints to delete")

        # 4. Determine which batches to process (not verified)
        total_batches = (len(text_chunks) + batch_size - 1) // batch_size
        all_batch_ids = list(range(total_batches))
        total_chunks = len(text_chunks)
        batches_to_process = [i for i in all_batch_ids if i not in verified_batch_ids]

        log.debug(
            "Batch calculation: total_chunks=%d, batch_size=%d, total_batches=%d",
            total_chunks,
            batch_size,
            total_batches,
        )
        log.debug("Batches to process: %s", batches_to_process)
        log.debug(
            "Processing %d batches out of %d total (already processed %d chunks)",
            len(batches_to_process),
            total_batches,
            already_processed_chunks,
        )

        if not auto_checkpoint:
            log.debug("Auto-checkpoint disabled, collecting batch DataFrames in memory")
            batch_dfs = []

        # 5. Set up progress bar
        log.debug(
            "Setting up progress bar with total=%d, initial=%d",
            total_chunks,
            already_processed_chunks,
        )
        with tqdm(
            total=total_chunks,
            desc="Processing chunks",
            initial=already_processed_chunks,
        ) as pbar:
            for batch_id in batches_to_process:
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, total_chunks)
                batch_chunks = text_chunks[start:end]
                batch_chunk_ids = text_chunk_ids[start:end]
                # Chunk id is the start index
                if not batch_chunks:
                    log.debug("Skipping empty batch %d", batch_id)
                    continue

                log.debug(
                    "Processing batch %d with %d chunks (range: %d-%d)",
                    batch_id,
                    len(batch_chunks),
                    start,
                    end,
                )

                # Check if we are over budget
                if self.cost_tracker is not None and self.cost_tracker.is_over_budget():
                    log.warning(
                        "Over budget, stopping extraction at batch %d", batch_id
                    )
                    break

                log.debug("Starting concurrent processing for batch %d", batch_id)

                def _on_chunk_complete():
                    pbar.update(1)
                    pbar.refresh()

                try:
                    results = self.concurrent_processor.process_concurrently(
                        batch_chunks,
                        lambda p: self._extract_from_text_chunk(p),
                        on_item_complete=_on_chunk_complete,
                    )
                except Exception as e:
                    log.error(
                        "Concurrent processing failed for batch %d: %s", batch_id, e
                    )
                    continue

                log.debug(
                    "Concurrent processing completed for batch %d, got %d results",
                    batch_id,
                    len(results),
                )
                log.debug("Parsing batch %d results.", batch_id)
                batch_df = self.parse_results_dataframe(
                    results=results,
                    text_chunks=batch_chunks,
                    text_chunk_ids=batch_chunk_ids,
                    batch_id=batch_id,
                )
                log.debug(
                    "Batch %d parsed to DataFrame with %d rows", batch_id, len(batch_df)
                )

                # Explicitly free results to prevent memory accumulation
                del results

                if auto_checkpoint:
                    log.debug("Saving batch checkpoint %d", batch_id)
                    experiment_manager.save_batch_checkpoint(batch_df, batch_id)
                    experiment_manager.save_state(self.cost_tracker)
                    log.debug(
                        "Successfully saved batch checkpoint %d and state", batch_id
                    )
                    # Explicitly free batch DataFrame after saving
                    del batch_df

                    # Periodic WAL checkpoint every 10 batches to reclaim memory
                    if batch_id > 0 and batch_id % 10 == 0:
                        if hasattr(self.semantic_cache, "checkpoint"):
                            self.semantic_cache.checkpoint()
                            log.debug(
                                "Periodic WAL checkpoint after batch %d", batch_id
                            )
                else:
                    log.debug(
                        "Adding batch %d DataFrame to memory collection", batch_id
                    )
                    batch_dfs.append(batch_df)

        if auto_checkpoint:
            # 6. Concatenate all results
            log.debug("Auto-checkpoint enabled, consolidating batch results")
            consolidated_df = experiment_manager.consolidate_batches()
            log.debug("Batch consolidation completed, cleaning up checkpoints")
            experiment_manager.cleanup_batch_checkpoints()
            log.debug("Batch checkpoints cleaned up")
        else:
            log.debug(
                "Auto-checkpoint disabled, concatenating batch DataFrames from memory"
            )
            consolidated_df = pd.concat(batch_dfs, ignore_index=True)
            log.debug("Memory concatenation completed")

        log.debug(
            "Batch processing completed. Final results: %d records",
            len(consolidated_df),
        )

        return consolidated_df

    def _extract_from_text_chunk(
        self,
        text_chunk: str,
    ) -> Dict[str, Any]:
        """Extract data from a single text chunk. Error safe extraction. Does not raise any errors.

        Args:
            text_chunk: The text chunk to extract data from.

        Returns:
            A dictionary containing the extracted data and errors.
        """
        # Hot path - minimal logging
        if self.cost_tracker is not None and self.cost_tracker.is_over_budget():
            log.debug("Over budget, skipping text chunk extraction")
            return {"extracted_data": None, "errors": "Over budget"}

        try:
            result = self._instructor_extract_with_retry(text_chunk)
            return {"extracted_data": result, "errors": []}
        except Exception as llm_error:
            log.error("Extraction failed for text chunk: %s", llm_error)
            return {"extracted_data": None, "errors": str(llm_error)}

    def _instructor_extract_with_retry(self, text_chunk: str) -> BaseModel:
        """Use Instructor + Pydantic schema for structured output.

        Args:
            text_chunk: The text chunk to extract data from.

        Returns:
            The extracted data as a Pydantic model.

        Raises:
            InstructorError: If the LLM API call fails.
            ValueError: If the response is not a Pydantic model.
        """
        # Hot path - minimal logging. Schema is cached internally.
        schema = self.extraction_schema.create_pydantic_schema()
        prompt = self.extraction_schema.create_prompt(text_chunk, self.prompt_template)
        system_prompt = self.system_prompt

        estimated_total_tokens = self._estimate_total_tokens(
            system_prompt, prompt, schema, tokenize=True
        )

        def _instructor_extract():
            self.rate_limiter.before_request(est_tokens=estimated_total_tokens)

            try:
                response, completion = (
                    self.client.chat.completions.create_with_completion(
                        model=self.model_config.model,
                        temperature=self.temperature,
                        response_model=schema,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        max_retries=0,
                    )
                )
            except Exception as e:
                log.error("LLM API call failed: %s", e)
                raise InstructorError(
                    f"Failed to extract data from text chunk with Instructor: {e}"
                ) from e

            if not is_pydantic_model(response):
                log.error("Invalid response type: %s", type(response))
                raise ValueError(f"Unsupported response type: {type(response)}")

            usage = getattr(completion, "usage", None)
            prompt_tokens = (
                getattr(usage, "prompt_tokens", None) if usage is not None else None
            )
            completion_tokens = (
                getattr(usage, "completion_tokens", None) if usage is not None else None
            )

            if completion_tokens is not None and prompt_tokens is not None:
                self.max_output_tokens = max(self.max_output_tokens, completion_tokens)
                log.debug("Updated max output tokens: %d", self.max_output_tokens)

                self.rate_limiter.after_request(
                    actual_tokens=prompt_tokens + completion_tokens
                )

            if self.cost_tracker is not None:
                if isinstance(prompt_tokens, int) and isinstance(
                    completion_tokens, int
                ):
                    self.cost_tracker.track_token_usage(
                        prompt_tokens, completion_tokens
                    )
                else:
                    # Fallback: system + user + JSON schema of Pydantic model
                    self.cost_tracker.track_input_text(
                        system_prompt + "\n" + prompt + "\n", schema
                    )
                    self.cost_tracker.track_output_pydantic(response)

            return response

        # Check semantic cache - hot path, minimal logging
        provider_and_model = self.model_config.get_provider_string()
        try:
            key = make_cache_key(
                prompt_text=prompt,
                system_prompt=system_prompt,
                model_name=provider_and_model,
                temperature=self.temperature,
            )
            cached = self.semantic_cache.get(key)
            if cached:
                loaded = json.loads(cached.decode("utf-8"))
                pydantic_result = schema(**loaded)
                if (
                    self.cost_tracker is not None
                    and self.cost_tracker.count_cache_hits_towards_cost
                ):
                    self.cost_tracker.track_input_text(
                        system_prompt + "\n" + prompt + "\n", schema
                    )
                    self.cost_tracker.track_output_pydantic(pydantic_result)
                return pydantic_result
        except Exception as e:
            log.error(f"Cache error {e}, performing new extraction")

        response = self.retry_handler.execute_with_retry(_instructor_extract)
        response_dict = response.model_dump(mode="json")
        # Convert to dict to save to semantic cache
        try:
            self.semantic_cache.set(key, json.dumps(response_dict).encode("utf-8"))
        except Exception as e:
            log.error(f"Cache error {e}, did not cache result")
        return response

    def _estimate_total_tokens(
        self,
        system_prompt: str,
        prompt: str,
        schema: BaseModel,
        tokenize: bool = True,
    ) -> int:
        """Estimate the total tokens for a given system prompt, prompt, and schema."""
        # Include schema JSON for estimation alongside system + user prompt
        complete_prompt = (
            f"{system_prompt}\n{prompt}\n{json.dumps(schema.model_json_schema())}"
        )
        if tokenize:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(tokenizer.encode(complete_prompt))
        else:
            input_tokens = len(complete_prompt) // 4

        total_tokens = input_tokens + self.max_output_tokens

        return total_tokens

    def parse_results_dataframe(
        self,
        results: List[Dict[str, Any]],
        text_chunks: List[str],
        text_chunk_ids: List[int],
        batch_id: int = 0,
    ) -> pd.DataFrame:
        """
        Parse extraction results into a DataFrame. Also cleans the results to remove any invalid items according to the schema.

        Args:
            results: The results to parse.
            text_chunks: The text chunks that were used to generate the results.
            text_chunk_ids: The IDs of the text chunks that were used to generate the results.
            batch_id: The ID of the batch that the results belong to.

        Returns:
            A DataFrame containing the parsed results.
        """
        log.debug(
            "Parsing results DataFrame: batch_id=%d, results_count=%d",
            batch_id,
            len(results),
        )

        # Hot path - process all results without per-item logging
        data: List[pd.DataFrame] = []
        for result, text_chunk, chunk_id in zip(results, text_chunks, text_chunk_ids):
            errors_json = json.dumps(result["errors"]) if result["errors"] else None
            extracted_data: Optional[BaseModel] = result["extracted_data"]

            if extracted_data is None:
                row_df = pd.DataFrame(
                    [
                        {
                            SYSTEM_CHUNK_ID_COLUMN: chunk_id,
                            SYSTEM_BATCH_ID_COLUMN: batch_id,
                            SYSTEM_CHUNK_COLUMN: text_chunk,
                            SYSTEM_EXTRACTED_DATA_JSON_COLUMN: None,
                            SYSTEM_ERRORS_COLUMN: errors_json,
                        }
                    ]
                )
                data.append(row_df)
            else:
                extracted_data_dict = (
                    self.extraction_schema.validate_and_parse_response_to_dict(
                        extracted_data, str(text_chunk)
                    )
                )
                row = {
                    SYSTEM_CHUNK_ID_COLUMN: chunk_id,
                    SYSTEM_BATCH_ID_COLUMN: batch_id,
                    SYSTEM_CHUNK_COLUMN: text_chunk,
                    SYSTEM_EXTRACTED_DATA_JSON_COLUMN: json.dumps(extracted_data_dict),
                    SYSTEM_ERRORS_COLUMN: errors_json,
                }
                data.append(pd.DataFrame([row]))

        # Outer join to preserve all columns in case there is a mismatch in the column sets.
        return (
            pd.concat(data, ignore_index=True, join="outer") if data else pd.DataFrame()
        )
