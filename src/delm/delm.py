from __future__ import annotations

from delm.utils.rate_limiter import BucketRateLimiter, NoOpRateLimiter

"""DELM extraction pipeline core module.
"""
from datetime import datetime
import logging
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
from delm.schemas import Schema
from delm.logging import configure as _configure_logging
from delm.constants import (
    SYSTEM_RECORD_ID_COLUMN,
    SYSTEM_CHUNK_COLUMN,
    SYSTEM_RANDOM_SEED,
    SYSTEM_CHUNK_ID_COLUMN,
    SYSTEM_ERRORS_COLUMN,
    SYSTEM_LOG_FILE_SUFFIX,
)
from delm.strategies import SplitStrategy, RelevanceScorer
from delm.utils.cost_tracker import CostTracker
from delm.utils.semantic_cache import SemanticCacheFactory
from typing import Any, Union, Optional

# --------------------------------------------------------------------------- #
# Main class                                                                  #
# --------------------------------------------------------------------------- #


class DELM:
    """
    Data Extraction with Language Model (DELM) pipeline.
    """

    def __init__(
        self,
        schema: Union[str, Path, dict, Schema],
        *,
        # LLM Settings (flat)
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        mode: Optional[str] = None,
        temperature: float = 0.0,
        batch_size: int = 10,
        max_workers: int = 1,
        max_retries: int = 3,
        base_delay: float = 1.0,
        tokens_per_minute: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        track_cost: bool = True,
        max_budget: Optional[float] = None,
        model_input_cost_per_1M_tokens: Optional[float] = None,
        model_output_cost_per_1M_tokens: Optional[float] = None,
        # Data Preprocessing (flat)
        target_column: str = "text",
        drop_target_column: bool = False,
        splitting_strategy: Optional[Union[dict, SplitStrategy]] = None,
        relevance_scorer: Optional[Union[dict, RelevanceScorer]] = None,
        score_filter: Optional[str] = None,  # pandas query syntax
        # Prompt Settings
        prompt_template: Optional[
            str
        ] = "Extract the following information from the text:\n\n{variables}\n\nText to analyze:\n{text}",
        system_prompt: Optional[str] = "You are a precise data-extraction assistant.",
        # Semantic Cache Settings
        cache_backend: Optional[str] = "sqlite",
        cache_path: Union[str, Path] = ".delm/cache",
        cache_max_size_mb: int = 512,
        cache_synchronous: str = "normal",
        # =============================================
        # Non-DELMConfig Settings
        # Experiment Settings (if using disk storage)
        use_disk_storage: bool = False,
        experiment_path: Optional[
            Union[str, Path]
        ] = None,  # experiment directory and path
        overwrite_experiment: bool = False,
        auto_checkpoint_and_resume_experiment: bool = True,
        # Logging Settings
        save_log_file: bool = False,
        log_dir: Optional[Union[str, Path]] = ".delm/logs",
        log_file_prefix: str = "",
        console_log_level: str = "INFO",
        file_log_level: str = "DEBUG",
        override_logging: bool = True,
    ) -> None:
        """Initialize the DELM extraction pipeline.

        Args:
            schema: Extraction schema defining the variables to extract. Can be a path
                to a YAML file, a dictionary, or a Schema object.

            provider: LLM provider to use.
            model: Model name to use for extraction.
            base_url: Custom API base URL for the provider. Useful for proxies or
                self-hosted endpoints.
            mode: Instructor mode for structured output.
            temperature: Sampling temperature for LLM responses. Lower values produce
                more deterministic outputs.
            batch_size: Number of text chunks to process per API batch.
            max_workers: Maximum number of concurrent workers for parallel processing.
            max_retries: Maximum number of retry attempts for failed API calls.
            base_delay: Base delay in seconds for exponential backoff between retries.
            tokens_per_minute: Rate limit for tokens per minute.
            requests_per_minute: Rate limit for requests per minute.
            track_cost: Whether to track API costs during extraction.
            max_budget: Maximum budget in dollars. Extraction stops if exceeded.
            model_input_cost_per_1M_tokens: Override input token cost per 1M tokens.
                Uses built-in pricing if not specified.
            model_output_cost_per_1M_tokens: Override output token cost per 1M tokens.
                Uses built-in pricing if not specified.

            target_column: Name of the column containing text to extract from.
            drop_target_column: Whether to drop the original target column after
                splitting into chunks.
            splitting_strategy: Strategy for splitting text into chunks. Can be a
                dict config or a ``SplitStrategy`` instance.
            relevance_scorer: Strategy for scoring chunk relevance. Can be a dict
                config or a ``RelevanceScorer`` instance.
            score_filter: Pandas query string to filter chunks by score
                (e.g., ``"delm_score > 0.5"``).

            prompt_template: Template for the extraction prompt. Must contain
                ``{variables}`` and ``{text}`` placeholders.
            system_prompt: System prompt for the LLM.

            cache_backend: Backend for semantic caching. Options: ``"sqlite"``,
                ``"lmdb"``, ``"filesystem"``, ``"none"``, or ``None`` to disable.
            cache_path: Directory path for cache storage.
            cache_max_size_mb: Maximum cache size in megabytes.
            cache_synchronous: SQLite synchronous mode (``"normal"`` or ``"full"``).

            use_disk_storage: Whether to use disk-based storage for experiment data
                and checkpoints.
            experiment_path: Directory path for experiment data when using disk storage.
                Required if ``use_disk_storage=True``.
            overwrite_experiment: Whether to overwrite an existing experiment directory.
            auto_checkpoint_and_resume_experiment: Whether to automatically save
                checkpoints and resume from them on restart.

            save_log_file: Whether to save logs to a file.
            log_dir: Directory for log files.
            log_file_prefix: Prefix for log file names.
            console_log_level: Logging level for console output.
            file_log_level: Logging level for file output.
            override_logging: Whether to override existing logging configuration.
        """
        config = DELMConfig(
            schema=schema,
            provider=provider,
            model=model,
            base_url=base_url,
            mode=mode,
            temperature=temperature,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            base_delay=base_delay,
            tokens_per_minute=tokens_per_minute,
            requests_per_minute=requests_per_minute,
            track_cost=track_cost,
            max_budget=max_budget,
            model_input_cost_per_1M_tokens=model_input_cost_per_1M_tokens,
            model_output_cost_per_1M_tokens=model_output_cost_per_1M_tokens,
            target_column=target_column,
            drop_target_column=drop_target_column,
            splitting_strategy=splitting_strategy,
            relevance_scorer=relevance_scorer,
            score_filter=score_filter,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            cache_backend=cache_backend,
            cache_path=cache_path,
            cache_max_size_mb=cache_max_size_mb,
            cache_synchronous=cache_synchronous,
        )

        # Configure logging
        if save_log_file:
            if log_dir is None:
                log_dir = log_dir
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_name = f"{log_file_prefix}{current_time}{SYSTEM_LOG_FILE_SUFFIX}"
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
        log.debug("Initialising DELM…")

        # Validate configuration before proceeding
        config.validate()
        self.config = config

        self.experiment_path = experiment_path
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
        config: Union[str, Path, DELMConfig],
        *,
        # =============================================
        # Non-DELMConfig Settings
        # Experiment Settings (if using disk storage)
        use_disk_storage: bool = False,
        experiment_path: Optional[
            Union[str, Path]
        ] = None,  # experiment directory and path
        overwrite_experiment: bool = False,
        auto_checkpoint_and_resume_experiment: bool = True,
        # Logging Settings
        save_log_file: bool = False,
        log_dir: Optional[Union[str, Path]] = ".delm/logs",
        log_file_prefix: str = "",
        console_log_level: str = "INFO",
        file_log_level: str = "DEBUG",
        override_logging: bool = True,
    ) -> DELM:
        """
        Create a DELM instance from a DELMConfig object.
        """
        config = DELMConfig.from_any(config)

        return cls(
            schema=config.schema,
            provider=config.llm_extraction_cfg.provider,
            model=config.llm_extraction_cfg.model,
            base_url=config.llm_extraction_cfg.base_url,
            mode=config.llm_extraction_cfg.mode,
            temperature=config.llm_extraction_cfg.temperature,
            prompt_template=config.llm_extraction_cfg.prompt_template,
            system_prompt=config.llm_extraction_cfg.system_prompt,
            batch_size=config.llm_extraction_cfg.batch_size,
            max_workers=config.llm_extraction_cfg.max_workers,
            max_retries=config.llm_extraction_cfg.max_retries,
            base_delay=config.llm_extraction_cfg.base_delay,
            tokens_per_minute=config.llm_extraction_cfg.tokens_per_minute,
            requests_per_minute=config.llm_extraction_cfg.requests_per_minute,
            track_cost=config.llm_extraction_cfg.track_cost,
            max_budget=config.llm_extraction_cfg.max_budget,
            model_input_cost_per_1M_tokens=config.llm_extraction_cfg.model_input_cost_per_1M_tokens,
            model_output_cost_per_1M_tokens=config.llm_extraction_cfg.model_output_cost_per_1M_tokens,
            target_column=config.data_preprocessing_cfg.target_column,
            drop_target_column=config.data_preprocessing_cfg.drop_target_column,
            splitting_strategy=config.data_preprocessing_cfg.splitting_strategy,
            relevance_scorer=config.data_preprocessing_cfg.relevance_scorer,
            score_filter=config.data_preprocessing_cfg.score_filter,
            cache_backend=config.semantic_cache_cfg.backend,
            cache_path=config.semantic_cache_cfg.path,
            cache_max_size_mb=config.semantic_cache_cfg.max_size_mb,
            cache_synchronous=config.semantic_cache_cfg.synchronous,
            use_disk_storage=use_disk_storage,
            experiment_path=experiment_path,
            overwrite_experiment=overwrite_experiment,
            auto_checkpoint_and_resume_experiment=auto_checkpoint_and_resume_experiment,
            save_log_file=save_log_file,
            log_dir=log_dir,
            log_file_prefix=log_file_prefix,
            console_log_level=console_log_level,
            file_log_level=file_log_level,
            override_logging=override_logging,
        )

    ## ------------------------------- Public API ------------------------------- ##

    def extract(
        self, data: str | Path | pd.DataFrame, sample_size: int = -1
    ) -> pd.DataFrame:
        """Extract data from the given data source.

        Args:
            data: The data source to extract data from.
            sample_size: Optional number of records to sample before processing. ``-1``
                (default) processes all rows; a positive value samples deterministically
                using ``SYSTEM_RANDOM_SEED``.

        Returns:
            A DataFrame containing the extracted data.
        """
        self.prep_data(data, sample_size)
        return self.process_via_llm()

    def prep_data(
        self, data: str | Path | pd.DataFrame, sample_size: int = -1
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
        log.info("Data processing completed: %d processed rows", len(df))

        log.debug("Saving preprocessed data to experiment manager")
        self.experiment_manager.save_preprocessed_data(df)
        log.debug("Data preprocessing completed: %d processed rows saved", len(df))
        return df

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
            self.config.llm_extraction_cfg.batch_size,
        )
        final_df = self.extraction_manager.process_with_batching(
            text_chunks=text_chunks,
            text_chunk_ids=chunk_ids,
            batch_size=self.config.llm_extraction_cfg.batch_size,
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
        if self.cost_tracker is None:
            log.error("Cost tracking not enabled in configuration")
            raise ValueError(
                "Cost tracking is not enabled in the configuration. Please set `track_cost` to `True` in the configuration."
            )

        log.debug("Retrieving cost summary")
        if not self.config.llm_extraction_cfg.track_cost:
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
        target_column_name = self.config.data_preprocessing_cfg.target_column
        if text is None:
            text = f"<{target_column_name}>"
        prompt = self.config.schema.schema.create_prompt(
            text=text,
            prompt_template=self.config.llm_extraction_cfg.prompt_template,
        )
        return prompt

    ## ------------------------------ Private API ------------------------------- ##

    def _initialize_components(self) -> None:
        """Initialize all components using composition."""
        log.debug("Initializing DELM components")

        # Initialize components
        log.debug("Initializing data processor")
        self.data_processor = DataProcessor(self.config.data_preprocessing_cfg)

        if self.use_disk_storage:
            log.debug("Initializing disk-based experiment manager")
            self.experiment_manager = DiskExperimentManager(
                experiment_path=Path(self.experiment_path),
                overwrite_experiment=self.overwrite_experiment,
                auto_checkpoint_and_resume_experiment=self.auto_checkpoint_and_resume_experiment,
            )
        else:
            log.debug("Initializing in-memory experiment manager")
            self.experiment_manager = InMemoryExperimentManager()

        # Initialize experiment with DELMConfig object
        log.debug("Initializing experiment")
        self.experiment_manager.initialize_experiment(self.config)  # type: ignore

        # Initialize cost tracker (may be loaded from state if resuming)
        log.debug("Initializing cost tracker")
        if self.config.llm_extraction_cfg.track_cost:
            self.cost_tracker = CostTracker(
                provider=self.config.llm_extraction_cfg.provider,
                model=self.config.llm_extraction_cfg.model,
                max_budget=self.config.llm_extraction_cfg.max_budget,
                model_input_cost_per_1M_tokens=self.config.llm_extraction_cfg.model_input_cost_per_1M_tokens,
                model_output_cost_per_1M_tokens=self.config.llm_extraction_cfg.model_output_cost_per_1M_tokens,
            )
        else:
            self.cost_tracker = None

        # Load cost tracker from experiment manager if resuming
        if self.auto_checkpoint_and_resume_experiment:
            log.debug("Checking for existing state to resume")
            loaded_cost_tracker = self.experiment_manager.load_state()
            if loaded_cost_tracker is not None:
                self.cost_tracker = loaded_cost_tracker
                log.debug("Resumed cost tracker from saved state")
            else:
                log.debug("No saved state found, using fresh cost tracker")

        log.debug("Initializing semantic cache")
        self.semantic_cache = SemanticCacheFactory.from_config(
            self.config.semantic_cache_cfg
        )

        if (
            self.config.llm_extraction_cfg.tokens_per_minute
            or self.config.llm_extraction_cfg.requests_per_minute
        ):
            self.rate_limiter = BucketRateLimiter(
                tokens_per_minute=self.config.llm_extraction_cfg.tokens_per_minute,
                requests_per_minute=self.config.llm_extraction_cfg.requests_per_minute,
            )
        else:
            self.rate_limiter = NoOpRateLimiter()

        log.debug("Initializing extraction manager")
        self.extraction_manager = ExtractionManager(
            self.config.llm_extraction_cfg,
            extraction_schema=self.config.schema.schema,
            cost_tracker=self.cost_tracker,
            semantic_cache=self.semantic_cache,
            rate_limiter=self.rate_limiter,
        )

        log.debug("All components initialized successfully")
