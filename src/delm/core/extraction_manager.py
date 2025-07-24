"""
ExtractionManager - Handles LLM extraction and result parsing.
"""

import re
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import instructor
from pydantic import BaseModel, Field

from delm.schemas import SchemaManager
from delm.utils import RetryHandler, ConcurrentProcessor
from delm.config import LLMExtractionConfig
from delm.constants import SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN, SYSTEM_BATCH_ID_COLUMN, SYSTEM_ERRORS_COLUMN, SYSTEM_EXTRACTED_DATA_JSON_COLUMN
from delm.exceptions import ProcessingError, ValidationError, APIError
from delm.utils.cost_tracker import CostTracker
from delm.core.experiment_manager import BaseExperimentManager
from delm.utils.type_checks import is_pydantic_model
from delm.utils.semantic_cache import SemanticCache, make_cache_key


class ExtractionManager:
    """Handles LLM extraction and result parsing."""
    
    def __init__(
        self,
        model_config: LLMExtractionConfig, 
        schema_manager: 'SchemaManager', 
        cost_tracker: 'CostTracker',
        semantic_cache: 'SemanticCache',
    ):
        self.model_config = model_config
        self.temperature = model_config.temperature
        self.extract_to_dataframe = model_config.extract_to_dataframe
        
        # Use Instructor's universal provider interface
        self.client = instructor.from_provider(self.model_config.get_provider_string())
        
        self.schema_manager = schema_manager
        self.extraction_schema = self.schema_manager.get_extraction_schema()
        
        self.concurrent_processor = ConcurrentProcessor(
            max_workers=model_config.max_workers
        )
        self.retry_handler = RetryHandler(max_retries=model_config.max_retries)

        self.track_cost = model_config.track_cost
        self.cost_tracker = cost_tracker
        self.semantic_cache = semantic_cache
    
    def extract_from_text_chunks(
        self, 
        text_chunks: List[str], 
    ) -> List[Dict[str, Any]]:
        """Process text chunks with concurrent execution."""
        return self.concurrent_processor.process_concurrently(
            text_chunks,
            lambda p: self._extract_from_text_chunk(p)
        )

    def process_with_persistent_batching(
        self, 
        text_chunks: List[str], 
        batch_size: int,
        experiment_manager: 'BaseExperimentManager',
        auto_checkpoint: bool = True,
    ) -> pd.DataFrame:
        """
        Process text chunks with persistent batching and checkpointing.
        
        This method handles the complete extraction pipeline with:
        - Splitting text chunks into batches
        - Processing batches with concurrent execution
        - Saving batch checkpoints for resuming
        - Consolidating results into final DataFrame
        """
        from ..constants import BATCH_FILE_DIGITS, SYSTEM_CHUNK_ID_COLUMN
        from tqdm.auto import tqdm
        import os

        # 1. Discover all unverified batch IDs (checkpoint files that exist)
        unverified_batch_ids = experiment_manager.get_all_batch_ids() if auto_checkpoint else set()

        # 2. Attempt to load each batch, classify as verified or corrupted, and count chunks in verified batches
        verified_batch_ids = set()
        corrupted_batch_ids = set()
        already_processed_chunks = 0
        for batch_id in sorted(unverified_batch_ids):
            try:
                batch_df = experiment_manager.load_batch_checkpoint_by_id(batch_id)
                if batch_df is not None:
                    verified_batch_ids.add(batch_id)
                    already_processed_chunks += len(batch_df)
                else:
                    corrupted_batch_ids.add(batch_id)
            except Exception as e:
                corrupted_batch_ids.add(batch_id)

        # 3. Delete corrupted batch files so they can be replaced
        for batch_id in corrupted_batch_ids:
            try:
                deleted = experiment_manager.delete_batch_checkpoint(batch_id)
            except Exception as e:
                pass

        # 4. Determine which batches to process (not verified)
        total_batches = (len(text_chunks) + batch_size - 1) // batch_size
        all_batch_ids = list(range(total_batches))
        total_chunks = len(text_chunks)
        batches_to_process = [i for i in all_batch_ids if i not in verified_batch_ids]

        if not auto_checkpoint:
            batch_dfs = []

        # 5. Set up progress bar
        with tqdm(
            total=total_chunks,
            desc="Processing chunks",
            initial=already_processed_chunks
        ) as pbar:
            for batch_id in batches_to_process:
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, total_chunks)
                batch_chunks = text_chunks[start:end]
                # Chunk id is the start index
                if not batch_chunks:
                    continue
                chunk_id_offset = start
                results = self.extract_from_text_chunks(batch_chunks)
                batch_df = self.parse_results_dataframe(
                    results=results,
                    text_chunks=batch_chunks,
                    chunk_id_offset=chunk_id_offset,
                    batch_id=batch_id,
                    output="exploded" if self.extract_to_dataframe else "json_string_column" # TODO: rename self.extract_to_dataframe to self.extract_to_exploded_dataframe
                )
                pbar.update(len(batch_chunks))
                if auto_checkpoint:
                    experiment_manager.save_batch_checkpoint(batch_df, batch_id)
                    experiment_manager.save_state(self.cost_tracker)
                else:
                    batch_dfs.append(batch_df)


        if auto_checkpoint:
            # 6. Concatenate all results
            consolidated_df = experiment_manager.consolidate_batches()
            experiment_manager.cleanup_batch_checkpoints()
        else:
            consolidated_df = pd.concat(batch_dfs, ignore_index=True)
        
        # save to extracted data path
        experiment_manager.save_extracted_data(consolidated_df)
        return consolidated_df
        
    
    def _extract_from_text_chunk(
        self, 
        text_chunk: str, 
    ) -> Dict[str, Any]:
        """Extract data from a single text chunk."""
        try:
            result = self._instructor_extract(text_chunk)
            return {"extracted_data": result, "errors": []}
        except Exception as llm_error:
            return {"extracted_data": None, "errors": str(llm_error)}
    
    def _instructor_extract(self, text_chunk: str) -> BaseModel:
        """Use Instructor + Pydantic schema for structured output."""
        schema = self.extraction_schema.create_pydantic_schema()
        prompt = self.extraction_schema.create_prompt(text_chunk)
        system_prompt = self.schema_manager.config.system_prompt
        provider_and_model = self.model_config.get_provider_string()


        def _extract_with_schema():
            if self.track_cost:
                self.cost_tracker.track_input_text(system_prompt + "\n" + prompt)
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config.name,
                    temperature=self.temperature,
                    response_model=schema, # type: ignore TODO: is there a way to get rid of this type error?
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    # max_retries=0,
                )
            except Exception as e:
                raise APIError(
                    f"Failed to extract data from text chunk with Instructor: {e}",
                    {"text_length": len(text_chunk), "model_name": self.model_config.name}
                ) from e
            if not is_pydantic_model(response):
                raise ProcessingError(
                    f"Unsupported response type: {type(response)}",
                    {"response_type": type(response), "text_chunk": text_chunk[:10]}
                )
            if self.track_cost:
                self.cost_tracker.track_output_pydantic(response)
            return response
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
                if self.track_cost and self.cost_tracker.count_cache_hits_towards_cost:
                    self.cost_tracker.track_input_text(system_prompt + "\n" + prompt)
                    self.cost_tracker.track_output_pydantic(pydantic_result)
                return pydantic_result
            response = self.retry_handler.execute_with_retry(_extract_with_schema)
            # Convert to dict to save to semantic cache
            response_dict = response.model_dump(mode="json")
            self.semantic_cache.set(key, json.dumps(response_dict).encode("utf-8"))
            return response
        except Exception as e:
            raise ProcessingError(
                f"Failed to extract data from text chunk: {e}",
                {"text_length": len(text_chunk), "model_name": self.model_config.name}
            )
    
    def parse_results_dataframe(
        self,
        results: List[Dict[str, Any]],
        text_chunks: List[str],
        chunk_id_offset: int = 0,
        batch_id: int = 0,
        output: str = "exploded"  # or "json_string_column"
    ) -> pd.DataFrame:
        """
        Parse extraction results into a DataFrame.

        output="exploded": Each extracted item becomes its own row (exploded/structured).
        output="json_column": All extracted data for a chunk is serialized into a single JSON column.
        """
        data = []
        for idx, (result, text_chunk) in enumerate(zip(results, text_chunks)):
            chunk_id = chunk_id_offset + idx
            errors_json = json.dumps(result["errors"]) if result["errors"] else None
            extracted_data: BaseModel | None = result["extracted_data"]
            if output == "exploded":
                if extracted_data is None:
                    row_df = pd.DataFrame([{ 
                        SYSTEM_CHUNK_ID_COLUMN: chunk_id,
                        SYSTEM_BATCH_ID_COLUMN: batch_id,
                        SYSTEM_CHUNK_COLUMN: text_chunk,
                        SYSTEM_ERRORS_COLUMN: errors_json
                    }])
                    data.append(row_df)
                else:
                    parsed_df = self.extraction_schema.validate_and_parse_response_to_exploded_dataframe(extracted_data, text_chunk)
                    if not parsed_df.empty:
                        parsed_df[SYSTEM_CHUNK_ID_COLUMN] = chunk_id
                        parsed_df[SYSTEM_BATCH_ID_COLUMN] = batch_id
                        parsed_df[SYSTEM_CHUNK_COLUMN] = text_chunk
                        parsed_df[SYSTEM_ERRORS_COLUMN] = errors_json
                        data.append(parsed_df)
            elif output == "json_string_column":
                if extracted_data is None:
                    row_df = pd.DataFrame([{
                        SYSTEM_CHUNK_ID_COLUMN: chunk_id,
                        SYSTEM_BATCH_ID_COLUMN: batch_id,
                        SYSTEM_CHUNK_COLUMN: text_chunk,
                        SYSTEM_EXTRACTED_DATA_JSON_COLUMN: None,
                        SYSTEM_ERRORS_COLUMN: errors_json
                    }])
                    data.append(row_df)
                else:
                    extracted_data_dict = self.extraction_schema.validate_and_parse_response_to_dict(extracted_data, str(text_chunk))
                    row = {
                        SYSTEM_CHUNK_ID_COLUMN: chunk_id,
                        SYSTEM_BATCH_ID_COLUMN: batch_id,
                        SYSTEM_CHUNK_COLUMN: text_chunk,
                        SYSTEM_EXTRACTED_DATA_JSON_COLUMN: json.dumps(extracted_data_dict), 
                        SYSTEM_ERRORS_COLUMN: errors_json
                    }
                    data.append(pd.DataFrame([row]))
            else:
                raise ValueError(f"Unknown output type: {output}")
        # Outer join to preserve all columns
        return pd.concat(data, ignore_index=True, join="outer") if data else pd.DataFrame()