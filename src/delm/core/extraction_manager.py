"""
ExtractionManager - Handles LLM extraction and result parsing.
"""

import re
from typing import Any, Dict, List, Optional

import pandas as pd
import instructor
from pydantic import BaseModel, Field

from ..schemas import SchemaManager
from ..utils import RetryHandler, ConcurrentProcessor
from ..config import LLMExtractionConfig
from ..constants import SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN, SYSTEM_EXTRACTED_DATA_COLUMN, SYSTEM_BATCH_ID_COLUMN, SYSTEM_ERRORS_COLUMN
from ..exceptions import ProcessingError, ValidationError
from ..utils.cost_tracker import CostTracker
from .experiment_manager import ExperimentManager


class ExtractionManager:
    """Handles LLM extraction and result parsing."""
    
    def __init__(
        self,
        model_config: LLMExtractionConfig, 
        schema_manager: 'SchemaManager', 
        cost_tracker: 'CostTracker',
    ):
        self.model_config = model_config
        self.temperature = model_config.temperature
        self.regex_fallback_pattern = model_config.regex_fallback_pattern
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
    

    def process_and_parse(
        self, 
        text_chunks: List[str], 
        verbose: bool = False,
        chunk_id_offset: int = 0,
        batch_id: int = 0
    ) -> pd.DataFrame:
        """Process text chunks and return structured DataFrame or Dataframe with JSON output."""
        # Process chunks
        results = self.process_text_chunks(text_chunks, verbose)
        
        # Return JSON output by default, or parse to DataFrame if configured
        if self.extract_to_dataframe:
            return self._parse_results_to_dataframe(results, text_chunks, chunk_id_offset, batch_id)
        else:
            return self._create_json_output_dataframe(results, text_chunks, chunk_id_offset, batch_id)
    
    def process_text_chunks(
        self, 
        text_chunks: List[str], 
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """Process text chunks with concurrent execution."""
        return self.concurrent_processor.process_concurrently(
            text_chunks,
            lambda p: self._extract_from_text_chunk(p, verbose)
        )

    def process_with_persistent_batching(
        self, 
        text_chunks: List[str], 
        batch_size: int,
        experiment_manager: 'ExperimentManager',
        auto_checkpoint: bool = True,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Process text chunks with persistent batching and checkpointing.
        
        This method handles the complete extraction pipeline with:
        - Splitting text chunks into batches
        - Processing batches with concurrent execution
        - Saving batch checkpoints for resuming
        - Consolidating results into final DataFrame
        """
        from ..constants import BATCH_FILE_DIGITS
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
                    if verbose:
                        print(f"Loaded existing batch {batch_id:0{BATCH_FILE_DIGITS}d}")
                else:
                    corrupted_batch_ids.add(batch_id)
            except Exception as e:
                corrupted_batch_ids.add(batch_id)
                if verbose:
                    print(f"Failed to load batch {batch_id}: {e}")

        # 3. Delete corrupted batch files so they can be replaced
        for batch_id in corrupted_batch_ids:
            try:
                deleted = experiment_manager.delete_batch_checkpoint(batch_id)
                if verbose and deleted:
                    print(f"Deleted corrupted batch file for batch {batch_id}")
            except Exception as e:
                if verbose:
                    print(f"Failed to delete corrupted batch file {batch_id}: {e}")

        # 4. Determine which batches to process (not verified)
        total_batches = (len(text_chunks) + batch_size - 1) // batch_size
        all_batch_ids = list(range(total_batches))
        total_chunks = len(text_chunks)
        batches_to_process = [i for i in all_batch_ids if i not in verified_batch_ids]

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
                if not batch_chunks:
                    continue
                chunk_id_offset = start
                batch_df = self.process_and_parse(
                    batch_chunks, verbose,
                    chunk_id_offset=chunk_id_offset, batch_id=batch_id
                )
                pbar.update(len(batch_chunks))
                if auto_checkpoint:
                    experiment_manager.save_batch_checkpoint(batch_df, batch_id)
                    experiment_manager.save_state(self.cost_tracker)

        # 6. Concatenate all results (load all batch DataFrames at the end to avoid memory issues)
        batch_files = experiment_manager.list_batch_checkpoints()
        dfs = [experiment_manager.load_batch_checkpoint(p) for p in batch_files]
        if dfs:
            consolidated_df = pd.concat(dfs, ignore_index=True)
            return consolidated_df
        else:
            return pd.DataFrame()
    
    def _extract_from_text_chunk(
        self, 
        text_chunk: str, 
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Extract data from a single text chunk with optional fallback."""
        # We assume API key is set in environment for instructor
        try:
            result = self._instructor_extract(text_chunk)
            return {"result": result, "errors": []}
        except Exception as llm_error:
            if verbose:
                print(f"Error processing text chunk: {llm_error}")
            if self.regex_fallback_pattern:
                if verbose:
                    print("Falling back to regex extraction")
                try:
                    regex_result = self._regex_extract(text_chunk)
                    # Even if regex succeeds, log the original LLM error
                    return {"result": regex_result, "errors": [str(llm_error)]}
                except Exception as regex_error:
                    if verbose:
                        print(f"Regex fallback also failed: {regex_error}")
                    return {"result": None, "errors": [str(llm_error), str(regex_error)]}
            else:
                if verbose:
                    print("No regex fallback enabled, returning null result with error")
                return {"result": None, "errors": [str(llm_error)]}
    
    def _regex_extract(self, text_chunk: str) -> Dict[str, List[str]]:
        """Extract data using the user-provided regex pattern."""
        if not self.regex_fallback_pattern:
            return {}
        
        try:
            pattern = re.compile(self.regex_fallback_pattern)
            matches = pattern.findall(text_chunk)
            return {"extracted": matches}
        except re.error as e:
            raise ValidationError(
                f"Invalid regex pattern '{self.regex_fallback_pattern}': {e}",
                {"regex_pattern": self.regex_fallback_pattern, "text_length": len(text_chunk)}
            )
    
    def _instructor_extract(self, text_chunk: str) -> Dict[str, Any]:
        """Use Instructor + Pydantic schema for structured output."""
        def _extract_with_schema():
            # Use configurable schema if available, otherwise raise an error
            if self.extraction_schema:
                schema = self.extraction_schema.create_pydantic_schema()
                prompt = self.extraction_schema.create_prompt(text_chunk)
            else:
                raise ProcessingError(
                    "No extraction schema provided. You must specify a schema for extraction.",
                    {"text_chunk": text_chunk[:100]}
                )

            # TODO: Let user specify system prompt
            system_prompt = "You are a precise data‑extraction assistant."
            if self.track_cost:
                self.cost_tracker.track_input_text(system_prompt + "\n" + prompt)
            
            
            # Use the model name directly
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=self.temperature,
                response_model=schema,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            if self.track_cost:
                ### Convert response to JSON string for token counting
                # Note: Instructor's response model is not Pydantic, so we need to convert it to a JSON string
                # Could not find a better way to do this
                if hasattr(response, "model_dump"):
                    output_json = response.model_dump(mode="json") # type: ignore
                elif hasattr(response, "dict"):
                    output_json = response.dict() # type: ignore
                else:
                    output_json = response
                import json
                output_text = json.dumps(output_json)
                self.cost_tracker.track_output_text(output_text)
            return response
        try:
            return self.retry_handler.execute_with_retry(_extract_with_schema)
        except Exception as e:
            raise ProcessingError(
                f"Failed to extract data from text chunk: {e}",
                {"text_length": len(text_chunk), "model_name": self.model_config.name}
            ) from e
    
    def _parse_results_to_dataframe(
        self, 
        results: List[Dict[str, Any]], 
        text_chunks: List[str],
        chunk_id_offset: int = 0,
        batch_id: int = 0
    ) -> pd.DataFrame:
        """Parse LLM responses into structured DataFrame."""
        all_results = []
        for idx, (result, text_chunk) in enumerate(zip(results, text_chunks)):
            # Handle None results by creating a row with null extracted data
            if result["result"] is None:
                import json
                row_df = pd.DataFrame([{
                    SYSTEM_CHUNK_COLUMN: text_chunk,
                    SYSTEM_CHUNK_ID_COLUMN: chunk_id_offset + idx,
                    SYSTEM_BATCH_ID_COLUMN: batch_id,
                    SYSTEM_EXTRACTED_DATA_COLUMN: None,
                    SYSTEM_ERRORS_COLUMN: json.dumps(result["errors"]) if result["errors"] else None
                }])
                all_results.append(row_df)
            else:
                parsed_df = self._parse_single_result(result["result"], text_chunk, chunk_id_offset + idx)
                if not parsed_df.empty:
                    parsed_df['batch_id'] = batch_id
                    # Always add errors column - None if no errors, JSON string if errors exist
                    import json
                    parsed_df[SYSTEM_ERRORS_COLUMN] = json.dumps(result["errors"]) if result["errors"] else None
                    all_results.append(parsed_df)
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    def _create_json_output_dataframe(
        self, 
        results: List[Any], 
        text_chunks: List[str],
        chunk_id_offset: int = 0,
        batch_id: int = 0
    ) -> pd.DataFrame:
        """Create DataFrame with JSON output in a column."""
        import json
        data = []
        for idx, (result, text_chunk) in enumerate(zip(results, text_chunks)):
            # Extract result and errors from the standardized structure
            actual_result = result["result"]
            errors = result["errors"]
            
            # Convert Pydantic model to dict if needed
            if actual_result is None:
                result_dict = None
            elif hasattr(actual_result, 'model_dump'):
                result_dict = actual_result.model_dump(mode='json')  # type: ignore
            elif hasattr(actual_result, 'dict'):
                result_dict = actual_result.dict()  # type: ignore
            elif isinstance(actual_result, dict):
                result_dict = actual_result
            else:
                result_dict = None
            
            row = {
                SYSTEM_CHUNK_COLUMN: text_chunk,
                SYSTEM_CHUNK_ID_COLUMN: chunk_id_offset + idx,
                SYSTEM_BATCH_ID_COLUMN: batch_id,
                SYSTEM_EXTRACTED_DATA_COLUMN: json.dumps(result_dict) if result_dict else None,
                SYSTEM_ERRORS_COLUMN: json.dumps(errors) if errors else None
            }
            data.append(row)
        return pd.DataFrame(data)
    
    def _parse_single_result(self, result: Any, text_chunk: str, chunk_id: int) -> pd.DataFrame:
        """Parse a single LLM response."""
        if not self.extraction_schema or result is None:
            return pd.DataFrame()
        
        row_metadata = {'chunk_id': chunk_id, 'text_chunk': text_chunk}
        
        if hasattr(result, 'dict'):
            return self.extraction_schema.parse_response(result, str(text_chunk), row_metadata)
        elif isinstance(result, dict):
            return self._parse_dict_response(result, text_chunk, row_metadata)
        else:
            raise ProcessingError(
                f"Unsupported response type: {type(result)}",
                {"response_type": type(result), "text_chunk": text_chunk[:100]}
            )
    
    def _parse_dict_response(self, response: Dict[str, Any], text_chunk: str, row_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse dictionary response with error handling."""
        if not self.extraction_schema:
            return pd.DataFrame()
            
        try:
            schema_class = self.extraction_schema.create_pydantic_schema()
            
            if hasattr(response, 'get') and self.extraction_schema.container_name in response:
                return self.extraction_schema.parse_response(response, str(text_chunk), row_metadata)
            else:
                model_instance = schema_class(**response)
                return self.extraction_schema.parse_response(model_instance, str(text_chunk), row_metadata)
        except Exception as e:
            print(f"Failed to parse response as Pydantic model: {e}")
            return pd.DataFrame()
    
 