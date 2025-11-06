from __future__ import annotations

"""DELM extraction pipeline core module.
"""
from datetime import datetime
import logging
import time
from pathlib import Path
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
from delm.result import ExtractionResult
from typing import Any, Dict, Union, Optional, List
from delm.strategies import SplitStrategy, RelevanceScorer, ParagraphSplit, KeywordScorer

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
        # New API parameters (optional)
        provider: Optional[str] = None,
        model: Optional[str] = None,
        schema: Optional[Union[Dict[str, Any], str, Path]] = None,
        temperature: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        max_budget: Optional[float] = None,
        splitting: Optional[Union[str, SplitStrategy]] = None,
        scoring: Optional[Union[List[str], RelevanceScorer]] = None,
        score_filter: Optional[str] = None,
        target_column: Optional[str] = None,
        experiment: Optional[str] = None,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        # Old API parameters (for backward compatibility)
        config: Optional[DELMConfig] = None,
        experiment_name: Optional[str] = None,
        experiment_directory: Optional[Path] = None,
        overwrite_experiment: bool = False,
        auto_checkpoint_and_resume_experiment: bool = True,
        use_disk_storage: bool = True,
        save_file_log: bool = True,
        log_dir: Optional[Union[str, Path]] = None,
        console_log_level: str = DEFAULT_CONSOLE_LOG_LEVEL,
        file_log_level: str = DEFAULT_FILE_LOG_LEVEL,
        override_logging: bool = True,
    ) -> None:
        """Initialize the DELM extraction pipeline.

        Can be initialized in two ways:
        
        1. New API (direct parameters):
            DELM(provider="openai", model="gpt-4o-mini", schema=schema)
        
        2. Old API (config object):
            DELM(config=config, experiment_name="test", experiment_directory=Path("."))

        Args:
            provider: LLM provider (e.g., "openai", "anthropic").
            model: Model name (e.g., "gpt-4o-mini").
            schema: Schema definition (dict, path to YAML, or Schema object).
            temperature: Temperature for LLM responses.
            batch_size: Number of records per batch.
            max_workers: Number of concurrent workers.
            max_budget: Maximum budget for extraction.
            splitting: Splitting strategy (string or SplitStrategy object).
            scoring: Scoring strategy (list of keywords or RelevanceScorer object).
            score_filter: Pandas query string for filtering by score.
            target_column: Column name containing text to extract from.
            experiment: Experiment name (for new API).
            prompt_template: Custom prompt template.
            system_prompt: Custom system prompt.
            config: DELM configuration (old API).
            experiment_name: Name of the experiment (old API).
            experiment_directory: Base directory for experiment outputs (old API).
            overwrite_experiment: Whether to overwrite existing experiment data.
            auto_checkpoint_and_resume_experiment: Whether to auto‑resume from checkpoints.
            use_disk_storage: If True, use disk‑based experiment manager; otherwise in‑memory.
            save_file_log: If True, write a rotating log file under ``log_dir``.
            log_dir: Directory for log files.
            console_log_level: Log level for console output.
            file_log_level: Log level for file output.
            override_logging: If True, force reconfiguration of logging for the process.

        Raises:
            ValueError: If the provided parameters are invalid.
        """
        # Determine which API is being used
        using_new_api = config is None and (provider is not None or schema is not None)
        
        if using_new_api:
            # New API: build config from parameters
            config = self._build_config_from_params(
                provider=provider,
                model=model,
                schema=schema,
                temperature=temperature,
                batch_size=batch_size,
                max_workers=max_workers,
                max_budget=max_budget,
                splitting=splitting,
                scoring=scoring,
                score_filter=score_filter,
                target_column=target_column,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
            )
            # Use experiment parameter as experiment_name
            if experiment is None:
                experiment = "delm_extraction"
            experiment_name = experiment
            # Use default experiment directory
            if experiment_directory is None:
                from delm.constants import DEFAULT_EXPERIMENT_DIR
                experiment_directory = DEFAULT_EXPERIMENT_DIR
        else:
            # Old API: use provided config
            if config is None:
                raise ValueError(
                    "Must provide either 'config' (old API) or 'provider'+'schema' (new API)"
                )
            if experiment_name is None:
                raise ValueError("experiment_name is required when using config parameter")
            if experiment_directory is None:
                raise ValueError("experiment_directory is required when using config parameter")
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
    def from_config(
        cls,
        config_path: Union[str, Path, DELMConfig],
        **overrides: Any,
    ) -> "DELM":
        """Create a DELM instance from a config file with optional overrides.

        Args:
            config_path: Path to YAML config file or DELMConfig object.
            **overrides: Parameters to override from config (e.g., temperature=0.5).

        Returns:
            Configured DELM instance.

        Example:
            >>> delm = DELM.from_config("config.yaml", temperature=0.5)
        """
        log.debug("Creating DELM instance from config with overrides")
        
        # Load config if it's a path
        if isinstance(config_path, (str, Path)):
            base_config = DELMConfig.from_yaml(Path(config_path))
        else:
            base_config = config_path
        
        # Extract experiment-related overrides
        experiment = overrides.pop("experiment", None)
        experiment_directory = overrides.pop("experiment_directory", None)
        
        # Build parameters dict from config
        params = {
            "provider": base_config.llm_extraction.provider,
            "model": base_config.llm_extraction.name,
            "temperature": base_config.llm_extraction.temperature,
            "batch_size": base_config.llm_extraction.batch_size,
            "max_workers": base_config.llm_extraction.max_workers,
            "max_budget": base_config.llm_extraction.max_budget,
            "target_column": base_config.data_preprocessing.target_column,
            "score_filter": base_config.data_preprocessing.pandas_score_filter,
            "prompt_template": base_config.schema.prompt_template,
            "system_prompt": base_config.schema.system_prompt,
        }
        
        # Load schema from file
        if base_config.schema.spec_path:
            params["schema"] = base_config.schema.spec_path
        
        # Add splitting strategy
        if base_config.data_preprocessing.splitting.strategy:
            params["splitting"] = base_config.data_preprocessing.splitting.strategy
        
        # Add scoring strategy
        if base_config.data_preprocessing.scoring.scorer:
            params["scoring"] = base_config.data_preprocessing.scoring.scorer
        
        # Apply overrides
        params.update(overrides)
        
        # Add experiment info
        if experiment:
            params["experiment"] = experiment
        if experiment_directory:
            params["experiment_directory"] = Path(experiment_directory)
        
        return cls(**params)

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

    def extract(
        self,
        data: Union[str, Path, pd.DataFrame],
        sample_size: Optional[int] = None,
    ) -> ExtractionResult:
        """Extract structured data from text (single-step method).

        This is the recommended method for most use cases. It combines
        prep_data() and process_via_llm() into a single call.

        Args:
            data: Input data as DataFrame, file path, or directory path.
            sample_size: Optional number of records to sample (for testing).

        Returns:
            ExtractionResult object with data, cost, and statistics.

        Example:
            >>> delm = DELM(provider="openai", model="gpt-4o-mini", schema=schema)
            >>> result = delm.extract(df)
            >>> print(result.data)
        """
        log.debug("Starting extraction pipeline")
        
        # Step 1: Prep data
        sample = sample_size if sample_size else -1
        prepped_df = self.prep_data(data, sample_size=sample)
        log.debug(f"Data prep completed: {len(prepped_df)} chunks")
        
        # Step 2: Process via LLM
        result_df = self.process_via_llm()
        log.debug(f"LLM processing completed: {len(result_df)} results")
        
        # Step 3: Get statistics
        num_records = len(result_df[SYSTEM_RECORD_ID_COLUMN].unique())
        num_chunks = len(result_df[SYSTEM_CHUNK_ID_COLUMN].unique())
        num_errors = len(result_df[result_df[SYSTEM_ERRORS_COLUMN].notna()])
        
        # Step 4: Get cost summary (if tracking enabled)
        cost_summary = None
        if self.config.llm_extraction.track_cost:
            cost_summary = self.get_cost_summary()
        
        log.info(
            f"Extraction completed: {num_records} records, {num_chunks} chunks, "
            f"{num_errors} errors"
        )
        
        return ExtractionResult(
            data=result_df,
            cost=cost_summary,
            num_records=num_records,
            num_chunks=num_chunks,
            num_errors=num_errors,
        )

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

    def preview_prompt(
        self,
        text: Optional[str] = None,
    ) -> str:
        """Preview the compiled prompt for the extraction schema.

        Returns:
            A string containing the compiled prompt.
        """
        target_column_name = self.config.data_preprocessing.target_column
        if text is None:
            text = f"<{target_column_name}>"
        prompt = self.schema_manager.extraction_schema.create_prompt(
            text=text,
            prompt_template=self.schema_manager.prompt_template,
        )
        return prompt

    ## ------------------------------ Private API ------------------------------- ##

    @staticmethod
    def _build_config_from_params(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        schema: Optional[Union[Dict[str, Any], str, Path]] = None,
        temperature: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_workers: Optional[int] = None,
        max_budget: Optional[float] = None,
        splitting: Optional[Union[str, SplitStrategy]] = None,
        scoring: Optional[Union[List[str], RelevanceScorer]] = None,
        score_filter: Optional[str] = None,
        target_column: Optional[str] = None,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> DELMConfig:
        """Build a DELMConfig from individual parameters.

        Args:
            provider: LLM provider.
            model: Model name.
            schema: Schema definition.
            temperature: Temperature setting.
            batch_size: Batch size.
            max_workers: Max workers.
            max_budget: Max budget.
            splitting: Splitting strategy.
            scoring: Scoring strategy.
            score_filter: Score filter.
            target_column: Target column.
            prompt_template: Prompt template.
            system_prompt: System prompt.

        Returns:
            DELMConfig object.

        Raises:
            ValueError: If schema is not provided.
        """
        from delm.config import (
            DELMConfig,
            LLMExtractionConfig,
            DataPreprocessingConfig,
            SchemaConfig,
            SplittingConfig,
            ScoringConfig,
            SemanticCacheConfig,
        )
        from delm.constants import (
            DEFAULT_PROVIDER,
            DEFAULT_MODEL_NAME,
            DEFAULT_TEMPERATURE,
            DEFAULT_BATCH_SIZE,
            DEFAULT_MAX_WORKERS,
            DEFAULT_PROMPT_TEMPLATE,
            DEFAULT_SYSTEM_PROMPT,
            SYSTEM_RAW_DATA_COLUMN,
        )
        import tempfile
        import yaml
        
        if schema is None:
            raise ValueError("schema parameter is required")
        
        # Handle schema parameter
        schema_path = None
        if isinstance(schema, dict):
            # Write dict to temporary YAML file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False
            )
            yaml.dump(schema, temp_file)
            temp_file.close()
            schema_path = Path(temp_file.name)
        elif isinstance(schema, (str, Path)):
            schema_path = Path(schema)
        else:
            raise ValueError(
                f"schema must be dict, str, or Path, got {type(schema)}"
            )
        
        # Build LLM config
        llm_config = LLMExtractionConfig(
            provider=provider or DEFAULT_PROVIDER,
            name=model or DEFAULT_MODEL_NAME,
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            batch_size=batch_size or DEFAULT_BATCH_SIZE,
            max_workers=max_workers or DEFAULT_MAX_WORKERS,
            max_budget=max_budget,
        )
        
        # Handle splitting parameter
        split_strategy = None
        if splitting is not None:
            if isinstance(splitting, str):
                # String shortcut
                if splitting.lower() == "paragraph":
                    split_strategy = ParagraphSplit()
                elif splitting.lower() == "sentence":
                    from delm.strategies import RegexSplit
                    split_strategy = RegexSplit(r'\. ')
                elif splitting.lower() == "fixed-window":
                    from delm.strategies import FixedWindowSplit
                    split_strategy = FixedWindowSplit()
                else:
                    raise ValueError(f"Unknown splitting strategy: {splitting}")
            else:
                split_strategy = splitting
        
        # Handle scoring parameter
        scorer = None
        if scoring is not None:
            if isinstance(scoring, list):
                # List of keywords
                scorer = KeywordScorer(keywords=scoring)
            else:
                scorer = scoring
        
        # Build preprocessing config
        preprocessing_config = DataPreprocessingConfig(
            target_column=target_column or SYSTEM_RAW_DATA_COLUMN,
            splitting=SplittingConfig(strategy=split_strategy),
            scoring=ScoringConfig(scorer=scorer),
            pandas_score_filter=score_filter,
        )
        
        # Build schema config
        schema_config = SchemaConfig(
            spec_path=schema_path,
            prompt_template=prompt_template or DEFAULT_PROMPT_TEMPLATE,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
        
        # Build semantic cache config (use defaults)
        cache_config = SemanticCacheConfig()
        
        return DELMConfig(
            llm_extraction=llm_config,
            data_preprocessing=preprocessing_config,
            schema=schema_config,
            semantic_cache=cache_config,
        )

    def _initialize_components(self) -> None:
        """Initialize all components using composition."""
        log.debug("Initializing DELM components")

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
