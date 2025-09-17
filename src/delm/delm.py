from __future__ import annotations

"""DELM extraction pipeline core module.
"""
from datetime import datetime
import logging
import time
from pathlib import Path
import dotenv
import pandas as pd

# Module-level logger
log = logging.getLogger(__name__)

from delm.config import DELMConfig
from delm.core.data_processor import DataProcessor
from delm.core.experiment_manager import (
    DiskExperimentManager,
    InMemoryExperimentManager,
)
from delm.core.extraction_manager import ExtractionManager
from delm.schemas import SchemaManager
from delm.logging import configure as _configure_logging
from delm.constants import (
    SYSTEM_RECORD_ID_COLUMN,
    SYSTEM_CHUNK_COLUMN,
    SYSTEM_RANDOM_SEED,
    SYSTEM_CHUNK_ID_COLUMN,
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN,
    SYSTEM_ERRORS_COLUMN,
    DEFAULT_CONSOLE_LOG_LEVEL,
    DEFAULT_FILE_LOG_LEVEL,
    SYSTEM_LOG_FILE_PREFIX,
    SYSTEM_LOG_FILE_SUFFIX,
    DEFAULT_LOG_DIR,
)
from delm.utils.cost_tracker import CostTracker
from delm.utils.semantic_cache import SemanticCacheFactory
from typing import Any, Dict, Union, Optional

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
        save_file_log: bool = True,
        log_dir: Union[str, Optional][Path] = None,
        console_log_level: str = DEFAULT_CONSOLE_LOG_LEVEL,
        file_log_level: str = DEFAULT_FILE_LOG_LEVEL,
        override_logging: bool = True,
    ) -> None:
        """Initialize the DELM extraction pipeline.

        Args:
            config: DELM configuration for this pipeline.
            experiment_name: Name of the experiment.
            experiment_directory: Base directory for experiment outputs.
            overwrite_experiment: Whether to overwrite existing experiment data.
            auto_checkpoint_and_resume_experiment: Whether to auto‑resume from checkpoints.
            use_disk_storage: If True, use disk‑based experiment manager; otherwise in‑memory.
            save_file_log: If True, write a rotating log file under ``log_dir``.
            log_dir: Directory for log files. If None and ``save_file_log`` is True, defaults
                to ``DEFAULT_LOG_DIR/<experiment_name>``.
            console_log_level: Log level for console output.
            file_log_level: Log level for file output.
            override_logging: If True, force reconfiguration of logging for the process.

        Raises:
            ValueError: If the provided ``config`` is invalid.
        """
        # Configure logging
        if save_file_log:
            if log_dir is None:
                log_dir = Path(DEFAULT_LOG_DIR) / experiment_name
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_name = f"{SYSTEM_LOG_FILE_PREFIX}{experiment_name}_{current_time}{SYSTEM_LOG_FILE_SUFFIX}"
        else:
            log_file_name = None

        _configure_logging(
            console_level=console_log_level,
            file_dir=log_dir,
            file_name=log_file_name,
            file_level=file_log_level,
            force=override_logging,
        )

        log = logging.getLogger(__name__)
        log.debug(
            "Initialising DELM…",
            extra={
                "experiment_name": experiment_name,
                "experiment_directory": str(experiment_directory),
                "use_disk_storage": use_disk_storage,
            },
        )

        # Validate configuration before proceeding
        config.validate()

        self.config = config
        self.experiment_name = experiment_name
        self.experiment_directory = experiment_directory
        self.overwrite_experiment = overwrite_experiment
        self.auto_checkpoint_and_resume_experiment = (
            auto_checkpoint_and_resume_experiment
        )
        self.use_disk_storage = use_disk_storage
        self._initialize_components()

        log.debug("DELM pipeline initialized successfully")

    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path],
        experiment_name: str,
        experiment_directory: Path,
        **kwargs: Any,
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
        log.debug("Creating DELM instance from YAML config: %s", config_path)
        config = DELMConfig.from_yaml(Path(config_path))
        log.debug(
            "Config loaded from YAML: %s",
            config.name if hasattr(config, "name") else "unknown",
        )
        return cls(
            config=config,
            experiment_name=experiment_name,
            experiment_directory=experiment_directory,
            **kwargs,
        )

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        experiment_name: str,
        experiment_directory: Path,
        **kwargs: Any,
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
        log.debug("Creating DELM instance from dict config")
        config = DELMConfig.from_dict(config_dict)
        log.debug(
            "Config loaded from dict: %s",
            config.name if hasattr(config, "name") else "unknown",
        )
        return cls(
            config=config,
            experiment_name=experiment_name,
            experiment_directory=experiment_directory,
            **kwargs,
        )

    ## ------------------------------- Public API ------------------------------- ##

    def process_via_llm(
        self, preprocessed_file_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Process data through LLM extraction using configuration from constructor, with batch checkpointing and resuming.

        Args:
            preprocessed_file_path: The path to the preprocessed data. If None, the preprocessed data will be loaded from the experiment manager.

        Returns:
            A DataFrame containing the extracted data.
        """
        log.debug("Starting LLM processing pipeline")

        # Load preprocessed data from the experiment manager
        log.debug("Loading preprocessed data from experiment manager")
        data = self.experiment_manager.load_preprocessed_data(preprocessed_file_path)
        log.debug("Loaded preprocessed data: %d rows", len(data))

        meta_data = data.drop(columns=[SYSTEM_CHUNK_COLUMN])
        chunk_ids = data[SYSTEM_CHUNK_ID_COLUMN].tolist()
        text_chunks = data[SYSTEM_CHUNK_COLUMN].tolist()
        log.debug("Prepared %d chunks for LLM processing", len(text_chunks))

        log.debug(
            "Starting batch processing with batch_size: %d",
            self.config.llm_extraction.batch_size,
        )
        final_df = self.extraction_manager.process_with_batching(
            text_chunks=text_chunks,
            text_chunk_ids=chunk_ids,
            batch_size=self.config.llm_extraction.batch_size,
            experiment_manager=self.experiment_manager,
            auto_checkpoint=self.auto_checkpoint_and_resume_experiment,
        )
        log.debug("Batch processing completed: %d results", len(final_df))

        log.debug("Saving extracted data to experiment manager")
        self.experiment_manager.save_extracted_data(final_df)

        # left join with meta_data on chunk id
        log.debug("Merging results with metadata")
        final_df = pd.merge(final_df, meta_data, on=SYSTEM_CHUNK_ID_COLUMN, how="left")
        log.debug("Merge completed: %d final rows", len(final_df))

        # get unique record ids
        num_records_processed = len(final_df[SYSTEM_RECORD_ID_COLUMN].unique())
        num_chunks_processed = len(final_df[SYSTEM_CHUNK_ID_COLUMN].unique())
        num_chunks_with_errors = len(final_df[final_df[SYSTEM_ERRORS_COLUMN].notna()])

        log.info(
            "LLM processing completed: %d chunks (%d with errors) from %d records",
            num_chunks_processed,
            num_chunks_with_errors,
            num_records_processed,
        )

        return final_df

    def prep_data(
        self, data: Union[str, Path] | pd.DataFrame, sample_size: int = -1
    ) -> pd.DataFrame:
        """Preprocess data using the instance config and always save to the experiment manager.

        Args:
            data: Input data as a string path, ``Path``, or ``DataFrame``.
            sample_size: Optional number of records to sample before processing. ``-1``
                (default) processes all rows; a positive value samples deterministically
                using ``SYSTEM_RANDOM_SEED``.

        Returns:
            A DataFrame containing chunked (and optionally scored) data ready for extraction.
        """
        log.debug("Starting data preprocessing")
        log.debug("Loading data from source: %s", data)

        df = self.data_processor.load_data(data)
        log.debug("Data loaded: %d rows", len(df))

        if sample_size > 0 and sample_size < len(df):
            log.debug("Sampling %d rows from %d total rows", sample_size, len(df))
            df = df.sample(n=sample_size, random_state=SYSTEM_RANDOM_SEED)
            log.debug("Sampling completed: %d rows", len(df))

        log.debug("Processing dataframe with data processor")
        df = self.data_processor.process_dataframe(df)  # type: ignore
        log.debug("Data processing completed: %d processed rows", len(df))

        log.debug("Saving preprocessed data to experiment manager")
        self.experiment_manager.save_preprocessed_data(df)
        log.info("Data preprocessing completed: %d processed rows saved", len(df))
        return df

    def get_extraction_results(self) -> pd.DataFrame:
        """Get the results from the experiment manager.

        Returns:
            A DataFrame containing the extraction results.
        """
        log.debug("Retrieving extraction results DataFrame from experiment manager")
        results_df = self.experiment_manager.get_results()
        log.debug("Retrieved results: %d rows", len(results_df))
        return results_df

    def get_cost_summary(self) -> dict[str, Any]:
        """Get the cost summary from the cost tracker.

        Returns:
            A dictionary containing the cost summary.

        Raises:
            ValueError: If cost tracking is not enabled in the configuration.
        """
        log.debug("Retrieving cost summary")
        if not self.config.llm_extraction.track_cost:
            log.error("Cost tracking not enabled in configuration")
            raise ValueError(
                "Cost tracking is not enabled in the configuration. Please set `track_cost` to `True` in the configuration."
            )

        cost_summary = self.cost_tracker.get_cost_summary_dict()
        log.debug("Cost summary retrieved: %s", cost_summary)
        return cost_summary

    ## ------------------------------ Private API ------------------------------- ##

    def _initialize_components(self) -> None:
        """Initialize all components using composition."""
        log.debug("Initializing DELM components")

        # Environment & secrets -------------------------------------------- #
        if self.config.llm_extraction.dotenv_path:
            log.debug(
                "Loading environment from %s", self.config.llm_extraction.dotenv_path
            )
            dotenv.load_dotenv(self.config.llm_extraction.dotenv_path)

        # Initialize components
        log.debug("Initializing data processor")
        self.data_processor = DataProcessor(self.config.data_preprocessing)

        log.debug("Initializing schema manager")
        self.schema_manager = SchemaManager(self.config.schema)

        if self.use_disk_storage:
            log.debug("Initializing disk-based experiment manager")
            self.experiment_manager = DiskExperimentManager(
                experiment_name=self.experiment_name,
                experiment_directory=self.experiment_directory,
                overwrite_experiment=self.overwrite_experiment,
                auto_checkpoint_and_resume_experiment=self.auto_checkpoint_and_resume_experiment,
            )
        else:
            log.debug("Initializing in-memory experiment manager")
            self.experiment_manager = InMemoryExperimentManager(
                experiment_name=self.experiment_name
            )

        # Initialize experiment with DELMConfig object
        log.debug("Initializing experiment")
        self.experiment_manager.initialize_experiment(self.config)  # type: ignore

        # Initialize cost tracker (may be loaded from state if resuming)
        log.debug("Initializing cost tracker")
        self.cost_tracker = CostTracker(
            provider=self.config.llm_extraction.provider,
            model=self.config.llm_extraction.name,
            max_budget=self.config.llm_extraction.max_budget,
        )

        # Load state if resuming
        if self.auto_checkpoint_and_resume_experiment:
            log.debug("Checking for existing state to resume")
            loaded_cost_tracker = self.experiment_manager.load_state()
            if loaded_cost_tracker:
                log.info("Resuming from previous state")
                self.cost_tracker = loaded_cost_tracker

        log.debug("Initializing semantic cache")
        self.semantic_cache = SemanticCacheFactory.from_config(
            self.config.semantic_cache
        )

        log.debug("Initializing extraction manager")
        self.extraction_manager = ExtractionManager(
            self.config.llm_extraction,
            schema_manager=self.schema_manager,
            cost_tracker=self.cost_tracker,
            semantic_cache=self.semantic_cache,
        )

        log.debug("All components initialized successfully")
