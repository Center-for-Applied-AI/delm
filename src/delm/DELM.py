from __future__ import annotations

"""
DELM v0.2 – "Phase‑2" implementation
-----------------------------------
Major upgrades from the Phase‑1 prototype:
• Multi‑format loaders (.txt, .html/.md, .docx, .pdf*)
• Pluggable split strategies (ParagraphSplit, FixedWindowSplit, RegexSplit)
• Relevance scoring abstraction (KeywordScorer, FuzzyScorer)
• Structured extraction via `Instructor` with a Pydantic schema fallback
• Built‑in stub fallback when either OPENAI_API_KEY or Instructor is absent
*PDF uses `marker` OCR if available; else raises NotImplementedError.

The public API and method signatures remain unchanged so downstream code
continues to work.  Future phases should require **only** new strategy
classes or loader helpers – no breaking changes.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dotenv
import pandas as pd

# Required deps -------------------------------------------------------------- #
import openai  # type: ignore

from .config import DELMConfig
from .core import DataProcessor, ExperimentManager, ExtractionManager
from .schemas import SchemaManager
from .constants import (
    SYSTEM_CHUNK_COLUMN, SYSTEM_SCORE_COLUMN, 
    SYSTEM_CHUNK_ID_COLUMN, SYSTEM_EXTRACTED_DATA_COLUMN
)
from .exceptions import DataError
from .utils.cost_tracker import CostTracker

# --------------------------------------------------------------------------- #
# Main class                                                                  #
# --------------------------------------------------------------------------- #


class DELM:
    """Extraction pipeline with pluggable strategies."""

    def __init__(
        self,
        config: DELMConfig,
    ) -> None:
        # Config
        self.config = config
        
        # Initialize components using composition
        self._initialize_components()

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "DELM":
        """
        Create a DELM instance from a YAML configuration file.
        
        This is the recommended way to create DELM instances for most use cases.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configured DELM instance
            
        Example:
            delm = DELM.from_yaml("config.yaml")
        """
        config = DELMConfig.from_yaml(Path(config_path))
        return cls(config=config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DELM":
        """
        Create a DELM instance from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configured DELM instance
            
        Example:
            config = {
                "model": {"name": "gpt-4o-mini"},
                "data": {"target_column": "text"},
                "schema": {"spec_path": "schema.yaml"},
                "experiment": {"name": "test"}
            }
            delm = DELM.from_dict(config)
            # Data is passed to methods, not stored in config.yaml
            output_df = delm.prep_data(my_dataframe)
            # This choice was made to allow for more flexibility: a dataframe could not be stored in config.yaml
        """
        config = DELMConfig.from_dict(config_dict)
        return cls(config=config)

    def _initialize_components(self) -> None:
        """Initialize all components using composition."""
        # Environment & secrets -------------------------------------------- #
        if self.config.model.dotenv_path:
            dotenv.load_dotenv(self.config.model.dotenv_path)
        
        # Initialize components
        self.data_processor = DataProcessor(self.config.data)
        self.schema_manager = SchemaManager(self.config.schema)
        self.experiment_manager = ExperimentManager(self.config.experiment)
        self.cost_tracker = CostTracker(
            provider=self.config.model.provider,
            model=self.config.model.name,
        )
        self.extraction_manager = ExtractionManager(
            self.config.model,
            schema_manager=self.schema_manager,  # Pass the instance
            cost_tracker=self.cost_tracker,
        )


    # ------------------------------ Public API --------------------------- #
    def prep_data(self, data_source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare data for processing using configuration from constructor.
        Saves preprocessed data as .feather file in experiment directory.
        
        Args:
            data_source: Data to process (file path, Path, or DataFrame)
            
        Returns:
            DataFrame with chunked and scored data ready for LLM processing
        """
        # Use DataProcessor to load and process data
        df = self.data_processor.load_and_process(data_source)
        
        # Save preprocessed data using ExperimentManager
        self.preprocessed_data_path = self.experiment_manager.save_preprocessed_data(
            df, self.config.experiment.name
        )
        
        return df
    


    def process_via_llm(self) -> pd.DataFrame:
        """Process data through LLM extraction using configuration from constructor."""
        # Load preprocessed data from feather file
        if not hasattr(self, 'preprocessed_data_path') or not self.preprocessed_data_path.exists():
            raise DataError(
                "No preprocessed data found. Run prep_data() first.",
                {"suggestion": "Call prep_data() with your data source before processing"}
            )
        
        data = self.experiment_manager.load_preprocessed_data(self.preprocessed_data_path)

        text_chunks = data[SYSTEM_CHUNK_COLUMN].tolist()
        
        # Use ExtractionManager to process and parse results
        use_regex_fallback = self.config.model.regex_fallback_pattern is not None
        output_df = self.extraction_manager.process_and_parse(
            text_chunks, self.config.experiment.verbose, use_regex_fallback
        )
        
        if self.config.experiment.verbose:
            if self.config.model.extract_to_dataframe:
                print(f"Processed {len(data)} chunks. Extracted to DataFrame with {len(output_df)} structured rows.")
            else:
                print(f"Processed {len(data)} chunks. JSON output saved to `extracted_data` column.")
        
        if self.config.model.track_cost:
            self.cost_tracker.print_cost_summary()
        
        return output_df
    

    






