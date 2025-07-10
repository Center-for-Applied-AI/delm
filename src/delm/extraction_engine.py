"""
ExtractionEngine - Handles LLM-based data extraction with fallback strategies.
"""

import re
from typing import Any, Dict, List, Optional
from pathlib import Path

import openai
import instructor
from pydantic import BaseModel, Field

from .schemas import SchemaRegistry
from .retry_handler import RetryHandler
from .batch_processing import BatchProcessor


class ExtractionEngine:
    """Handles LLM extraction with configurable schemas and fallback strategies."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        batch_size: int = 10,
        max_workers: int = 1,
        regex_fallback_pattern: Optional[str] = None,
        schema_spec_path: Optional[str | Path] = None,
    ) -> None:
        # API configuration
        self.api_key = api_key
        if self.api_key and openai is not None:
            openai.api_key = self.api_key
            
        # Model configuration
        self.model_name = model_name
        self.temperature = temperature
        self.regex_fallback_pattern = regex_fallback_pattern
        
        # Schema system
        self.schema_registry = SchemaRegistry()
        self.extraction_schema = None
        
        if schema_spec_path:
            schema_config = self._load_schema_spec(schema_spec_path)
            # Handle both direct extraction config and nested extraction config
            if 'extraction' in schema_config:
                self.extraction_schema = self.schema_registry.create(schema_config['extraction'])
            else:
                # Assume the entire config is the extraction schema
                self.extraction_schema = self.schema_registry.create(schema_config)
        
        # Processing components
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            max_workers=max_workers
        )
        self.retry_handler = RetryHandler(max_retries=max_retries)
    
    def process_text_chunks(
        self, 
        text_chunks: List[str], 
        verbose: bool = False, 
        use_regex_fallback: bool = False
    ) -> List[Dict[str, Any]]:
        """Process text chunks with concurrency and progress bar."""
        return self.batch_processor.process_batch(
            text_chunks,
            lambda p: self.extract_from_text_chunk(p, verbose, use_regex_fallback)
        )
    
    def extract_from_text_chunk(
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
            print(f"Invalid regex pattern '{self.regex_fallback_pattern}': {e}")
            return {}
    
    def _instructor_extract(self, text_chunk: str) -> Dict[str, Any]:
        """Use Instructor + Pydantic schema for structured output."""
        def extract_with_schema():
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
            return self.retry_handler.execute_with_retry(extract_with_schema)
        except Exception as e:
            print(f"Failed to extract data from text chunk. Error: {e}.")
            raise
    
    @staticmethod
    def _load_schema_spec(path: str | Path) -> Dict[str, Any]:
        """Load schema specification from YAML or JSON file."""
        import yaml
        import json
        
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(p.read_text()) or {}
        if p.suffix.lower() == ".json":
            return json.loads(p.read_text())
        raise ValueError("Schema spec must be YAML or JSON") 