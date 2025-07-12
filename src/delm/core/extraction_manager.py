"""
ExtractionManager - Handles LLM extraction and result parsing.
"""

import re
from typing import Any, Dict, List, Optional

import openai
import instructor
import pandas as pd
from pydantic import BaseModel, Field

from ..schemas import SchemaManager
from ..utils import RetryHandler, BatchProcessor
from ..config import ModelConfig
from ..constants import SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN, SYSTEM_EXTRACTED_DATA_COLUMN
from ..exceptions import ProcessingError, APIError, ValidationError


class ExtractionManager:
    """Handles LLM extraction and result parsing."""
    
    def __init__(
        self, 
        model_config: ModelConfig, 
        schema_manager: 'SchemaManager', 
        api_key: Optional[str] = None
    ):
        # API configuration
        self.api_key = api_key
        if self.api_key and openai is not None:
            openai.api_key = self.api_key
            
        # Model configuration
        self.model_name = model_config.name
        self.temperature = model_config.temperature
        self.regex_fallback_pattern = model_config.regex_fallback_pattern
        self.extract_to_dataframe = model_config.extract_to_dataframe
        
        # Schema system - use injected SchemaManager
        self.schema_manager = schema_manager
        self.extraction_schema = self.schema_manager.get_extraction_schema()
        
        # Processing components
        self.batch_processor = BatchProcessor(
            batch_size=model_config.batch_size,
            max_workers=model_config.max_workers
        )
        self.retry_handler = RetryHandler(max_retries=model_config.max_retries)
    

    def process_and_parse(
        self, 
        text_chunks: List[str], 
        verbose: bool = False, 
        use_regex_fallback: bool = False
    ) -> pd.DataFrame:
        """Process text chunks and return structured DataFrame or Dataframe with JSON output."""
        # Process chunks
        results = self.process_text_chunks(text_chunks, verbose, use_regex_fallback)
        
        # Return JSON output by default, or parse to DataFrame if configured
        if self.extract_to_dataframe:
            return self._parse_results_to_dataframe(results, text_chunks)
        else:
            return self._create_json_output_dataframe(results, text_chunks)
    
    def process_text_chunks(
        self, 
        text_chunks: List[str], 
        verbose: bool = False, 
        use_regex_fallback: bool = False
    ) -> List[Dict[str, Any]]:
        """Process text chunks with progress bar."""
        return self.batch_processor.process_batch(
            text_chunks,
            lambda p: self._extract_from_text_chunk(p, verbose, use_regex_fallback)
        )
    
    def _extract_from_text_chunk(
        self, 
        text_chunk: str, 
        verbose: bool = False, 
        use_regex_fallback: bool = False
    ) -> Dict[str, Any]:
        """Extract data from a single text chunk with optional fallback."""
        if not self.api_key:
            if verbose:
                print("No API key found, falling back to regex extraction")
            if self.regex_fallback_pattern:
                return self._regex_extract(text_chunk)
            else:
                return {}
        
        try:
            return self._instructor_extract(text_chunk)
        except Exception as e:
            if verbose:
                print(f"Error processing text chunk: {e}")
            if use_regex_fallback and self.regex_fallback_pattern:
                if verbose:
                    print("Falling back to regex extraction")
                return self._regex_extract(text_chunk)
            else:
                if verbose:
                    print("No regex fallback enabled, returning empty result")
                return {}
    
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
            client = instructor.patch(openai.OpenAI(api_key=self.api_key))
            
            # Use configurable schema if available, otherwise create a simple default schema
            if self.extraction_schema:
                schema = self.extraction_schema.create_pydantic_schema()
                prompt = self.extraction_schema.create_prompt(text_chunk)
            else:
                # Create a simple default schema for numeric extraction
                class DefaultExtractSchema(BaseModel):
                    numbers: List[str] = Field(
                        default_factory=list,
                        description="Numeric strings (keep punctuation), in order of appearance",
                    )
                schema = DefaultExtractSchema
                prompt = f"Extract all numeric strings from the following text chunk:\n\n{text_chunk}"
            
            response = client.chat.completions.create(  # type: ignore
                model=self.model_name,
                temperature=self.temperature,
                response_model=schema,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise data‑extraction assistant.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            return response
        
        try:
            return self.retry_handler.execute_with_retry(_extract_with_schema)
        except Exception as e:
            raise ProcessingError(
                f"Failed to extract data from text chunk: {e}",
                {"text_length": len(text_chunk), "model_name": self.model_name}
            ) from e
    
    def _parse_results_to_dataframe(
        self, 
        results: List[Dict[str, Any]], 
        text_chunks: List[str]
    ) -> pd.DataFrame:
        """Parse LLM responses into structured DataFrame."""
        all_results = []
        
        for idx, (result, text_chunk) in enumerate(zip(results, text_chunks)):
            parsed_df = self._parse_single_result(result, text_chunk, idx)
            if not parsed_df.empty:
                all_results.append(parsed_df)
        
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    def _create_json_output_dataframe(
        self, 
        results: List[Any], 
        text_chunks: List[str]
    ) -> pd.DataFrame:
        """Create DataFrame with JSON output in a column."""
        import json
        
        data = []
        for idx, (result, text_chunk) in enumerate(zip(results, text_chunks)):
            # Convert Pydantic model to dict if needed
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump(mode='json')
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {}
            
            row = {
                SYSTEM_CHUNK_COLUMN: text_chunk,
                SYSTEM_CHUNK_ID_COLUMN: idx,
                SYSTEM_EXTRACTED_DATA_COLUMN: json.dumps(result_dict) if result_dict else '{}'
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _parse_single_result(self, result: Dict[str, Any], text_chunk: str, chunk_id: int) -> pd.DataFrame:
        """Parse a single LLM response."""
        if not self.extraction_schema or result is None:
            return pd.DataFrame()
        
        row_metadata = {'chunk_id': chunk_id, 'text_chunk': text_chunk}
        
        if hasattr(result, 'dict'):
            return self.extraction_schema.parse_response(result, str(text_chunk), row_metadata)
        elif isinstance(result, dict):
            return self._parse_dict_response(result, text_chunk, row_metadata)
        
        return pd.DataFrame()
    
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
    
 