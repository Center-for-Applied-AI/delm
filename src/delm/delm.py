from __future__ import annotations

"""DELM extraction pipeline core module.

Major upgrades from the Phase‑1 prototype:
• Multi‑format loaders (.txt, .html/.md, .docx, .pdf*)
• Pluggable split strategies (ParagraphSplit, FixedWindowSplit, RegexSplit)
• Relevance scoring abstraction (KeywordScorer, FuzzyScorer)
• Structured extraction via `Instructor` 
*PDF uses `marker` OCR if available; else raises NotImplementedError.

The public API and method signatures remain unchanged so downstream code
continues to work.  Future phases should require **only** new strategy
classes or loader helpers – no breaking changes.
"""
from pathlib import Path
import dotenv
import pandas as pd

from delm.config import DELMConfig
from delm.core.data_processor import DataProcessor
from delm.core.experiment_manager import DiskExperimentManager, InMemoryExperimentManager
from delm.core.extraction_manager import ExtractionManager
from delm.schemas import SchemaManager
from delm.constants import (
    SYSTEM_RECORD_ID_COLUMN,
    SYSTEM_CHUNK_COLUMN,
    SYSTEM_RANDOM_SEED,
    SYSTEM_SCORE_COLUMN,
    SYSTEM_CHUNK_ID_COLUMN,
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN,
)
from delm.utils.cost_tracker import CostTracker
from delm.utils.semantic_cache import SemanticCacheFactory
from typing import Any, Dict, Union

# --------------------------------------------------------------------------- #
# Main class                                                                  #
# --------------------------------------------------------------------------- #


class DELM:
    """Extraction pipeline with pluggable strategies.

    Attributes:
        config: DELMConfig instance for this pipeline.
        experiment_name: Name of the experiment.
        experiment_directory: Directory for experiment outputs.
        overwrite_experiment: Whether to overwrite existing experiment data.
        auto_checkpoint_and_resume_experiment: Whether to auto-resume experiments.
    """

    def __init__(
        self,
        *,
        config: DELMConfig,
        experiment_name: str,
        experiment_directory: Path,
        overwrite_experiment: bool = False,
        auto_checkpoint_and_resume_experiment: bool = True,
        use_disk_storage: bool = True,
    ) -> None:
        """Initialize the DELM extraction pipeline.

        Args:
            config: DELMConfig instance for this pipeline.
            experiment_name: Name of the experiment.
            experiment_directory: Directory for experiment outputs.
            overwrite_experiment: Whether to overwrite existing experiment data.
            auto_checkpoint_and_resume_experiment: Whether to auto-resume experiments.
            use_disk_storage: If True, use disk-based experiment manager; if False, use in-memory manager.
        """
        self.config = config
        self.experiment_name = experiment_name
        self.experiment_directory = experiment_directory
        self.overwrite_experiment = overwrite_experiment
        self.auto_checkpoint_and_resume_experiment = auto_checkpoint_and_resume_experiment
        self.use_disk_storage = use_disk_storage
        self._initialize_components()

    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path],
        experiment_name: str,
        experiment_directory: Path,
        **kwargs
    ) -> "DELM":
        """Create a DELM instance from a YAML configuration file.

        Args:
            config_path: Path to YAML configuration file.
            experiment_name: Name of the experiment.
            experiment_directory: Base directory for experiment outputs.
            **kwargs: Additional keyword arguments for DELM constructor.

        Returns:
            Configured DELM instance.
        """
        config = DELMConfig.from_yaml(Path(config_path))
        return cls(config=config, experiment_name=experiment_name, experiment_directory=experiment_directory, **kwargs)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        experiment_name: str,
        experiment_directory: Path,
        **kwargs
    ) -> "DELM":
        """Create a DELM instance from a configuration dictionary.

        Args:
            config_dict: Configuration dictionary.
            experiment_name: Name of the experiment.
            experiment_directory: Base directory for experiment outputs.
            **kwargs: Additional keyword arguments for DELM constructor.

        Returns:
            Configured DELM instance.
        """
        config = DELMConfig.from_dict(config_dict)
        return cls(config=config, experiment_name=experiment_name, experiment_directory=experiment_directory, **kwargs)

    ## ------------------------------- Public API ------------------------------- ##


    def process_via_llm(
        self, preprocessed_file_path: Path | None = None
    ) -> pd.DataFrame:
        """Process data through LLM extraction using configuration from constructor, with batch checkpointing and resuming."""
        # Load preprocessed data from the experiment manager
        data = self.experiment_manager.load_preprocessed_data(preprocessed_file_path)
        meta_data = data.drop(columns=[SYSTEM_CHUNK_COLUMN])
        chunk_ids = data[SYSTEM_CHUNK_ID_COLUMN].tolist()
        text_chunks = data[SYSTEM_CHUNK_COLUMN].tolist()

        final_df = self.extraction_manager.process_with_persistent_batching(
            text_chunks=text_chunks,
            text_chunk_ids=chunk_ids,
            batch_size=self.config.llm_extraction.batch_size,
            experiment_manager=self.experiment_manager,
            auto_checkpoint=self.auto_checkpoint_and_resume_experiment,
        )

        # left join with meta_data on chunk id
        final_df = pd.merge(final_df, meta_data, on=SYSTEM_CHUNK_ID_COLUMN, how="left", )

        # get unique record ids
        record_ids = meta_data[SYSTEM_RECORD_ID_COLUMN].unique().tolist()

        # Print summary
        if self.config.llm_extraction.extract_to_dataframe:
            print(f"Processed {len(data)} chunks from {len(record_ids)} records. Extracted to DataFrame with {len(final_df)} exploded rows.")
        else:
            print(f"Processed {len(data)} chunks from {len(record_ids)} records. JSON output saved to `{SYSTEM_EXTRACTED_DATA_JSON_COLUMN}` column.")
        
        return final_df

    
    def prep_data(self, data: str | Path | pd.DataFrame, sample_size: int = -1) -> pd.DataFrame:
        """Preprocess data using the instance config and always save to the experiment manager.

        Args:
            data: Input data as a string, Path, or DataFrame.

        Returns:
            DataFrame of prepped (chunked) data.
        """
        df = self.data_processor.load_data(data)
        
        
        if sample_size > 0 and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=SYSTEM_RANDOM_SEED)

        df = self.data_processor.process_dataframe(df) # type: ignore
        self.experiment_manager.save_preprocessed_data(df)
        return df


    def get_extraction_results_df(self) -> pd.DataFrame:
        """Get the results from the experiment manager."""
        return self.experiment_manager.get_results()

    def get_extraction_results_json(self):
        cols = [SYSTEM_CHUNK_COLUMN, SYSTEM_EXTRACTED_DATA_JSON_COLUMN]
        results_df = self.experiment_manager.get_results()
        if not all(col in results_df.columns for col in cols):
            raise ValueError("Json extraction results are not available. Please set `extract_to_dataframe` to `True` in the configuration or use `get_extraction_results_df` instead.")
        return results_df[[SYSTEM_CHUNK_COLUMN, SYSTEM_EXTRACTED_DATA_JSON_COLUMN]]

    def get_cost_summary(self) -> dict[str, Any]:
        if not self.config.llm_extraction.track_cost:
            raise ValueError("Cost tracking is not enabled in the configuration. Please set `track_cost` to `True` in the configuration.")
        return self.cost_tracker.get_cost_summary_dict()

    
    ## ------------------------------ Private API ------------------------------- ##


    def _initialize_components(self) -> None:
        """Initialize all components using composition."""
        # Environment & secrets -------------------------------------------- #
        if self.config.llm_extraction.dotenv_path:
            dotenv.load_dotenv(self.config.llm_extraction.dotenv_path)
        
        # Initialize components
        self.data_processor = DataProcessor(self.config.data_preprocessing)
        self.schema_manager = SchemaManager(self.config.schema)
        if self.use_disk_storage:
            self.experiment_manager = DiskExperimentManager(
                experiment_name=self.experiment_name,
                experiment_directory=self.experiment_directory,
                overwrite_experiment=self.overwrite_experiment,
                auto_checkpoint_and_resume_experiment=self.auto_checkpoint_and_resume_experiment,
            )
        else:
            self.experiment_manager = InMemoryExperimentManager(
                experiment_name=self.experiment_name
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
        
        self.semantic_cache = SemanticCacheFactory.from_config(self.config.semantic_cache)
        self.extraction_manager = ExtractionManager(
            self.config.llm_extraction,
            schema_manager=self.schema_manager,
            cost_tracker=self.cost_tracker,
            semantic_cache=self.semantic_cache,
        )


