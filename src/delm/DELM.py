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
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dotenv
import pandas as pd

# Required deps -------------------------------------------------------------- #
import openai  # type: ignore

# Import new schema system and processing utilities
from .schemas import SchemaRegistry
from .models import ExtractionVariable

# Import strategy classes
from .splitting_strategies import SplitStrategy, ParagraphSplit, FixedWindowSplit, RegexSplit
from .scoring_strategies import RelevanceScorer, KeywordScorer, FuzzyScorer

# Import data loader factory
from .data_loaders import loader_factory

# Import extraction engine
from .extraction_engine import ExtractionEngine

# Import new config system
from .config import DELMConfig, ConfigValidationError

# Import system constants
from .constants import (
    SYSTEM_CHUNK_COLUMN, SYSTEM_SCORE_COLUMN, 
    SYSTEM_LLM_JSON_COLUMN, SYSTEM_CHUNK_ID_COLUMN
)






# --------------------------------------------------------------------------- #
# Main class                                                                  #
# --------------------------------------------------------------------------- #


class DELM:
    """Extraction pipeline with pluggable strategies."""

    def __init__(
        self,
        config: DELMConfig,
    ) -> None:
        # Validate and store config
        config.validate()
        self.config = config
        
        # Load configuration and initialize components
        self._load_config()
        
        # Runtime artefacts
        self.raw_df: pd.DataFrame | None = None
        self.processed_df: pd.DataFrame | None = None

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
            # Data is passed to methods, not stored in config
            output_df = delm.prep_data(my_dataframe)
            # This choice was made to allow for more flexibility: a dataframe could not be stored in config.yaml
        """
        config = DELMConfig.from_dict(config_dict)
        return cls(config=config)

    def _load_config(self) -> None:
        """Load configuration and initialize all components."""
        # Constants ------------------------------------------------------- #
        self.TARGET_COLUMN_NAME = self.config.data.target_column
        self.CHUNK_COLUMN_NAME = SYSTEM_CHUNK_COLUMN  # This is internal, not configurable
        self.root_dir = Path.cwd()

        # Environment & secrets -------------------------------------------- #
        if self.config.model.dotenv_path:
            dotenv.load_dotenv(self.config.model.dotenv_path)
        self.api_key: str | None = os.getenv("OPENAI_API_KEY")
        if self.api_key and openai is not None:
            openai.api_key = self.api_key

        # Experiment configuration ----------------------------------------- #
        # Make sure experiments dir exists and create if not
        self.experiments_dir = self.config.experiment.directory
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Check if experiment name is already in use
        if (self.experiments_dir / self.config.experiment.name).exists() and not self.config.experiment.overwrite_experiment:
            raise ValueError(f"Experiment name '{self.config.experiment.name}' already exists in {self.experiments_dir}")
        elif (self.experiments_dir / self.config.experiment.name).exists() and self.config.experiment.overwrite_experiment:
            shutil.rmtree(self.experiments_dir / self.config.experiment.name)

        # Create experiment directory
        self.experiment_name = self.config.experiment.name
        self.experiment_dir = self.experiments_dir / self.experiment_name

        # TODO: Implement experiment resumption via an experiment_path argument

        # Model configuration ---------------------------------------------- #
        self.model_name = self.config.model.name
        self.temperature = self.config.model.temperature
        self.max_retries = self.config.model.max_retries
        self.regex_fallback_pattern = self.config.model.regex_fallback_pattern

        # Strategy objects ------------------------------------------------- #
        self.splitter: SplitStrategy = self.config.data.splitting.strategy
        self.scorer: RelevanceScorer = self.config.data.scoring.scorer

        # Schema system ---------------------------------------------------- #
        self.schema_registry = SchemaRegistry()
        self.extraction_schema = None

        # Enhanced processing components ------------------------------------ #
        self.extraction_engine = ExtractionEngine(
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            max_retries=self.max_retries,
            batch_size=self.config.model.batch_size,
            max_workers=self.config.model.max_workers,
            regex_fallback_pattern=self.regex_fallback_pattern,
            schema_spec_path=self.config.schema.spec_path
        )

        # Data processing configuration ------------------------------------ #
        self.target_column = self.config.data.target_column
        self.drop_target_column = self.config.data.drop_target_column
        self.verbose = self.config.experiment.verbose

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
        if isinstance(data_source, (str, Path)):
            # Handle file loading
            path = Path(data_source)
            
            try:
                data = loader_factory.load_file(path)
                
                # TODO: There has to be a better way to handle this.
                # Maybe return a tuple of (data, target_column) and handle it in the caller?
                file_suffix = path.suffix.lower()
                if file_suffix == ".csv":
                    if self.target_column == "":
                        raise ValueError("Target column is required for CSV files")
                elif file_suffix == ".txt":
                    # data is a string for text-based files
                    if isinstance(data, str):
                        df = pd.DataFrame({
                            self.TARGET_COLUMN_NAME: [data]
                        })
                    else:
                        raise ValueError("Text loader should return string")
                else:
                    raise
                        
            except ValueError:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        else:
            # Handle DataFrame input
            if self.target_column == "":
                raise ValueError("Target column is required for DataFrame input")
            df = data_source.copy()
        
        # Process the DataFrame (chunking and scoring)
        if self.target_column == "":
            # For text files, use the default target column
            target_column = self.TARGET_COLUMN_NAME
        else:
            target_column = self.target_column
            
        df[self.CHUNK_COLUMN_NAME] = df[target_column].apply(self.splitter.split)
        df = df.explode(self.CHUNK_COLUMN_NAME).reset_index(drop=True)
        df[SYSTEM_CHUNK_ID_COLUMN] = range(len(df))
        
        if self.drop_target_column and target_column != self.TARGET_COLUMN_NAME:
            df = df.drop(columns=[target_column])

        df[SYSTEM_SCORE_COLUMN] = df[self.CHUNK_COLUMN_NAME].apply(self.scorer.score)
        
        # Save preprocessed data to feather file
        self._save_preprocessed_data(df)
        
        return df
    
    def _save_preprocessed_data(self, df: pd.DataFrame) -> None:
        """Save preprocessed data as .feather file in experiment directory."""
        # Create preprocessed directory
        preprocessed_dir = self.experiment_dir / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on experiment name and timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = f"{self.experiment_name}_{timestamp}"
        
        # Save as feather file
        feather_path = preprocessed_dir / f"{source_name}_prepped.feather"
        df.to_feather(feather_path)
        
        # Store path for later use
        self.preprocessed_data_path = feather_path
        
        if self.verbose:
            print(f"Saved preprocessed data to: {feather_path}")

    def process_via_llm(self) -> pd.DataFrame:
        """Process data through LLM extraction using configuration from constructor."""
        # Load preprocessed data from feather file
        if not hasattr(self, 'preprocessed_data_path') or not self.preprocessed_data_path.exists():
            raise ValueError("No preprocessed data found. Run prep_data() first.")
        
        data = pd.read_feather(self.preprocessed_data_path)
        
        # TODO: add filtering for score

        text_chunks = data[self.CHUNK_COLUMN_NAME].tolist()
        
        # Use unified processing method - batching is controlled by BatchProcessor settings
        # Regex fallback is enabled if regex_fallback_pattern is provided
        use_regex_fallback = self.regex_fallback_pattern is not None
        results = self.extraction_engine.process_text_chunks(text_chunks, self.verbose, use_regex_fallback)
        
        out = data.copy()
        out[SYSTEM_LLM_JSON_COLUMN] = results
        if self.verbose:
            print(f"Processed {len(out)} rows. Saved json to `{SYSTEM_LLM_JSON_COLUMN}` column")
        self.processed_df = out
        return out
    
    def _process_text_chunks(self, text_chunks: List[str], verbose: bool = False, use_regex_fallback: bool = False) -> List[Dict[str, Any]]:
        """Process text chunks with configurable concurrency (defaults to sequential)."""
        return self.extraction_engine.process_text_chunks(text_chunks, verbose, use_regex_fallback)
    
    def parse_to_dataframe(self, data: pd.DataFrame, metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse LLM responses into a structured DataFrame."""
        if self.processed_df is None:
            raise ValueError("No processed data available. Run process_via_llm first.")
        
        all_results = []
        for idx, row in data.iterrows():
            response = row.get("llm_json")
            text_chunk = row.get(self.CHUNK_COLUMN_NAME, "")
            
            # Add row metadata - preserve all original columns except the processed columns
            row_metadata = {}
            
            # Preserve all original columns from the input DataFrame
            for col in row.index:
                if col not in [SYSTEM_CHUNK_COLUMN, SYSTEM_LLM_JSON_COLUMN]:  # Skip the processed columns
                    row_metadata[col] = row[col]
            
            # Add any additional metadata passed to the function
            if metadata:
                row_metadata.update(metadata)
            
            # Parse response using the schema
            if self.extraction_engine.extraction_schema and response is not None:
                # Handle both Pydantic models and dictionaries
                if hasattr(response, 'dict'):
                    # It's a Pydantic model
                    parsed_df = self.extraction_engine.extraction_schema.parse_response(response, str(text_chunk), row_metadata)
                elif isinstance(response, dict):
                    # It's a dictionary (fallback case)
                    # Try to convert to Pydantic model if possible
                    try:
                        schema_class = self.extraction_engine.extraction_schema.create_pydantic_schema()
                        if hasattr(response, 'get') and self.extraction_engine.extraction_schema.container_name in response:
                            # For nested schemas, we need to handle the container structure
                            parsed_df = self.extraction_engine.extraction_schema.parse_response(response, str(text_chunk), row_metadata)
                        else:
                            # For simple schemas, try to create a model instance
                            model_instance = schema_class(**response)
                            parsed_df = self.extraction_engine.extraction_schema.parse_response(model_instance, str(text_chunk), row_metadata)
                    except Exception as e:
                        print(f"Failed to parse response as Pydantic model: {e}")
                        parsed_df = pd.DataFrame()
                else:
                    parsed_df = pd.DataFrame()
                all_results.append(parsed_df)
        
        # Purpose: to future proof. Pandas update will not support empty DataFrame concatenation
        non_empty_results = [df for df in all_results if not df.empty]
        if non_empty_results:
            return pd.concat(non_empty_results, ignore_index=True)
        return pd.DataFrame()





