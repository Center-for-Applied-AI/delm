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
    SYSTEM_CHUNK_COLUMN 
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
        *,
        config: DELMConfig,
        experiment_name: str,
        experiment_directory: Path,
        overwrite_experiment: bool = False,
        verbose: bool = False,
        auto_checkpoint_and_resume_experiment: bool = True,
    ) -> None:
        # Config
        self.config = config
        self.experiment_name = experiment_name
        self.experiment_directory = experiment_directory
        self.overwrite_experiment = overwrite_experiment
        self.verbose = verbose
        self.auto_checkpoint_and_resume_experiment = auto_checkpoint_and_resume_experiment
        self._initialize_components()

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path], experiment_name: str, experiment_directory: Path, **kwargs) -> "DELM":
        """
        Create a DELM instance from a YAML configuration file.
        
        This is the recommended way to create DELM instances for most use cases.
        
        Args:
            config_path: Path to YAML configuration file
            experiment_name: Name of the experiment (used for directory structure, file naming)
            experiment_directory: Base directory for experiment outputs
        Returns:
            Configured DELM instance
        Example:
            delm = DELM.from_yaml("config.yaml", experiment_name="my_exp", experiment_directory=Path("./experiments"))
        """
        config = DELMConfig.from_yaml(Path(config_path))
        return cls(config=config, experiment_name=experiment_name, experiment_directory=experiment_directory, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], experiment_name: str, experiment_directory: Path, **kwargs) -> "DELM":
        """
        Create a DELM instance from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            experiment_name: Name of the experiment (used for directory structure, file naming)
            experiment_directory: Base directory for experiment outputs
        Returns:
            Configured DELM instance
        Example:
            config = {
                "model": {"name": "gpt-4o-mini"},
                "data": {"target_column": "text"},
                "schema": {"spec_path": "schema.yaml"},
            }
            delm = DELM.from_dict(config, experiment_name="my_exp", experiment_directory=Path("./experiments"))
        """
        config = DELMConfig.from_dict(config_dict)
        return cls(config=config, experiment_name=experiment_name, experiment_directory=experiment_directory, **kwargs)

    def _initialize_components(self) -> None:
        """Initialize all components using composition."""
        # Environment & secrets -------------------------------------------- #
        if self.config.llm_extraction.dotenv_path:
            dotenv.load_dotenv(self.config.llm_extraction.dotenv_path)
        
        # Initialize components
        self.data_processor = DataProcessor(self.config.data_preprocessing)
        self.schema_manager = SchemaManager(self.config.schema)
        self.experiment_manager = ExperimentManager(
            experiment_name=self.experiment_name,
            experiment_directory=self.experiment_directory,
            overwrite_experiment=self.overwrite_experiment,
            verbose=self.verbose,
            auto_checkpoint_and_resume_experiment=self.auto_checkpoint_and_resume_experiment,
        )
        
        # Initialize experiment with config and schema
        # Note that config dict contains everything but the schema, which is in schema_dict
        config_dict = self.config.to_config_dict()
        schema_dict = self.config.to_schema_dict()
        self.experiment_manager.initialize_experiment(config_dict, schema_dict)
        
        # Initialize cost tracker (may be loaded from state if resuming)
        self.cost_tracker = CostTracker(
            provider=self.config.llm_extraction.provider,
            model=self.config.llm_extraction.name,
        )
        
        # Load state if resuming
        if self.auto_checkpoint_and_resume_experiment:
            state = self.experiment_manager.load_state()
            if state and "cost_tracker" in state:
                self.cost_tracker = CostTracker.from_dict(state["cost_tracker"])
        
        self.extraction_manager = ExtractionManager(
            self.config.llm_extraction,
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
        self.experiment_manager.save_preprocessed_data(df)
        return df
    


    def process_via_llm(self, preprocessed_file_path: Path | None = None) -> pd.DataFrame:
        """Process data through LLM extraction using configuration from constructor, with batch checkpointing and resuming."""
        # Load preprocessed data from feather file
        data = self.experiment_manager.load_preprocessed_data(preprocessed_file_path)
        text_chunks = data[SYSTEM_CHUNK_COLUMN].tolist()
        
        # Process with persistent batching
        final_df = self.extraction_manager.process_with_persistent_batching(
            text_chunks=text_chunks,
            batch_size=self.config.llm_extraction.batch_size,
            experiment_manager=self.experiment_manager,
            auto_checkpoint=self.auto_checkpoint_and_resume_experiment,
            verbose=self.verbose
        )
        
        # Handle final consolidation and cleanup
        if self.auto_checkpoint_and_resume_experiment:
            result_path = self.experiment_manager.consolidate_batches(self.experiment_name)
            self.experiment_manager.cleanup_batch_checkpoints()
            if self.verbose:
                print(f"Final consolidated result saved to: {result_path}")
            # Load and return the consolidated result
            final_df = pd.read_feather(result_path)
        
        # Print summary
        if self.verbose:
            if self.config.llm_extraction.extract_to_dataframe:
                print(f"Processed {len(data)} chunks. Extracted to DataFrame with {len(final_df)} structured rows.")
            else:
                print(f"Processed {len(data)} chunks. JSON output saved to `extracted_data` column.")
        
        if self.config.llm_extraction.track_cost:
            self.cost_tracker.print_cost_summary()
        
        return final_df
    

    






