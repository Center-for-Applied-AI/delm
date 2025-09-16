"""
DELM Experiment Managers
=======================
Defines the interface for experiment managers and provides disk-based and in-memory implementations.

- DiskExperimentManager: Handles experiment directories, file I/O, checkpointing, and state management on disk.
- InMemoryExperimentManager: Stores all data in memory.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Union, Optional, List
import pandas as pd
import json
import time
from abc import ABC, abstractmethod

import yaml

from delm.config import DELMConfig
from delm.constants import (
    DATA_DIR_NAME,
    CACHE_DIR_NAME,
    PROCESSING_CACHE_DIR_NAME,
    BATCH_FILE_PREFIX,
    BATCH_FILE_SUFFIX,
    BATCH_FILE_DIGITS,
    STATE_FILE_NAME,
    CONSOLIDATED_RESULT_PREFIX,
    CONSOLIDATED_RESULT_SUFFIX,
    PREPROCESSED_DATA_PREFIX,
    PREPROCESSED_DATA_SUFFIX,
)
from delm.utils.cost_tracker import CostTracker
from delm.exceptions import ExperimentManagementError

# Module-level logger
log = logging.getLogger(__name__)


class BaseExperimentManager(ABC):
    """Abstract base class for DELM experiment managers."""

    @abstractmethod
    def get_results(self) -> pd.DataFrame:
        """Get the results from the experiment directory.

        Returns:
            A DataFrame containing the results.
        """
        pass

    @abstractmethod
    def initialize_experiment(self, delm_config: DELMConfig):
        """Initialize the experiment.

        Args:
            delm_config: The DELM configuration.
        """
        pass

    @abstractmethod
    def save_preprocessed_data(self, df: pd.DataFrame) -> Path:
        """Save the preprocessed data to the experiment directory.

        Args:
            df: The DataFrame to save.

        Returns:
            The path to the saved data.
        """
        pass

    @abstractmethod
    def load_preprocessed_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load the preprocessed data from the experiment directory.

        Args:
            file_path: Optional explicit path to a feather file; when omitted,
                use the manager's default preprocessed data path.

        Returns:
            A DataFrame containing the preprocessed data.
        """
        pass

    @abstractmethod
    def save_batch_checkpoint(self, batch_df: pd.DataFrame, batch_id: int) -> Path:
        """Save a batch checkpoint to the experiment directory.

        Args:
            batch_df: The DataFrame to save.
            batch_id: The ID of the batch.

        Returns:
            The path to the saved data.
        """
        pass

    @abstractmethod
    def list_batch_checkpoints(self) -> List[Path]:
        """List all batch checkpoint files in the processing cache directory.

        Returns:
            A list of paths to the batch checkpoint files.
        """
        pass

    @abstractmethod
    def load_batch_checkpoint(self, batch_path: Path) -> pd.DataFrame:
        """Load a batch checkpoint from a feather file.

        Args:
            batch_path: The path to the batch checkpoint file.

        Returns:
            A DataFrame containing the batch checkpoint data.
        """
        pass

    @abstractmethod
    def load_batch_checkpoint_by_id(self, batch_id: int) -> pd.DataFrame:
        """Load a batch checkpoint by batch ID.

        Args:
            batch_id: The ID of the batch.

        Returns:
            A DataFrame containing the batch checkpoint data.
        """
        pass

    @abstractmethod
    def consolidate_batches(self) -> pd.DataFrame:
        """Consolidate all batch files into a single DataFrame and save as final result.

        Returns:
            A DataFrame containing the consolidated data.
        """
        pass

    @abstractmethod
    def cleanup_batch_checkpoints(self):
        """Remove all batch checkpoint files after consolidation."""
        pass

    @abstractmethod
    def get_all_existing_batch_ids(self) -> set:
        """Get all existing batch IDs.

        Returns:
            A set of all existing batch IDs.
        """
        pass

    @abstractmethod
    def get_batch_checkpoint_path(self, batch_id: int) -> Path:
        """Get the path to the batch checkpoint file for a given batch ID.

        Args:
            batch_id: The ID of the batch.

        Returns:
            The path to the batch checkpoint file.
        """
        pass

    @abstractmethod
    def delete_batch_checkpoint(self, batch_id: int) -> bool:
        """Delete the batch checkpoint file for a given batch ID.

        Args:
            batch_id: The ID of the batch.

        Returns:
            True if the batch checkpoint file was deleted, False otherwise.
        """
        pass

    @abstractmethod
    def save_state(self, cost_tracker: CostTracker):
        """Save the experiment state to the experiment directory.

        Args:
            cost_tracker: The cost tracker to save.
        """
        pass

    @abstractmethod
    def load_state(self) -> Optional[CostTracker]:
        """Load the experiment state from the experiment directory.

        Returns:
            The restored cost tracker, or None if not found.
        """
        pass

    @abstractmethod
    def save_extracted_data(self, df: pd.DataFrame) -> Path:
        """Save the extracted data to the experiment directory.

        Args:
            df: The DataFrame to save.

        Returns:
            The path to the saved data.
        """
        pass


class DiskExperimentManager(BaseExperimentManager):
    """Handles experiment directories, config/schema validation, batch checkpointing, and state management (disk-based)."""

    def __init__(
        self,
        experiment_name: str,
        experiment_directory: Path,
        overwrite_experiment: bool = False,
        auto_checkpoint_and_resume_experiment: bool = True,
    ):
        self.experiment_name = experiment_name
        self.experiment_directory = experiment_directory
        self.overwrite_experiment = overwrite_experiment
        self.auto_checkpoint_and_resume_experiment = (
            auto_checkpoint_and_resume_experiment
        )
        self.experiment_dir = self._get_experiment_dir()

    # --- Properties for common paths ---
    @property
    def config_dir(self) -> Path:
        d = self.experiment_dir / "config"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def data_dir(self) -> Path:
        d = self.experiment_dir / DATA_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache_dir(self) -> Path:
        d = self.experiment_dir / CACHE_DIR_NAME / PROCESSING_CACHE_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def is_experiment_completed(self) -> bool:
        """Check if the experiment is completed by checking if the consolidated result file exists."""
        result_file = (
            self.data_dir
            / f"{CONSOLIDATED_RESULT_PREFIX}{self.experiment_name}{CONSOLIDATED_RESULT_SUFFIX}"
        )
        return result_file.exists()

    def get_results(self) -> pd.DataFrame:
        """Get the consolidated results from the experiment directory.

        Returns:
            A DataFrame containing the results.

        Raises:
            FileNotFoundError: If the consolidated result file does not exist.
        """
        result_file = (
            self.data_dir
            / f"{CONSOLIDATED_RESULT_PREFIX}{self.experiment_name}{CONSOLIDATED_RESULT_SUFFIX}"
        )
        if not result_file.exists():
            log.debug(f"Consolidated result file not found: {result_file}")
            raise FileNotFoundError(
                f"Consolidated result file does not exist: {result_file}"
            )
        log.debug(f"Reading consolidated result file: {result_file}")
        return pd.read_feather(result_file)

    def initialize_experiment(self, delm_config: DELMConfig):
        """Validate and create experiment directory structure; write config and schema files.

        Raises:
            ExperimentManagementError: If the experiment directory exists and neither
                overwrite nor checkpoint/resume is allowed.
            FileNotFoundError: If attempting to resume without config files present.
            ValueError: If resume config or schema mismatches current configuration.
        """
        experiment_dir_path = self.experiment_dir
        if experiment_dir_path.exists():
            log.debug(f"Experiment directory already exists: {experiment_dir_path}")
            if self.overwrite_experiment:
                log.warning(
                    f"Overwriting experiment directory: {experiment_dir_path} in 3 seconds..."
                )
                time.sleep(3)
                shutil.rmtree(experiment_dir_path)
                log.info(f"Deleted and will recreate: {experiment_dir_path}")
            elif self.auto_checkpoint_and_resume_experiment:
                # Check if experiment already completed
                if self.is_experiment_completed():
                    log.error(f"Experiment already completed: {experiment_dir_path}")
                    raise ExperimentManagementError(
                        """Experiment exists and is already completed. 
                        To proceed, set overwrite_experiment=True to 
                        overwrite the existing experiment, or use a different 
                        experiment name."""
                    )
                # Verify config/schema match before resuming
                log.debug(
                    f"Verifying existing experiment config (from {experiment_dir_path}) matches current config"
                )
                self.verify_resume_config(delm_config)
                log.debug(f"Existing experiment config matches current config")
            else:
                log.error(f"Experiment directory already exists: {experiment_dir_path}")
                raise ExperimentManagementError(
                    (
                        f"\nExperiment directory already exists. To proceed, choose one of the following:\n"
                        f"  - Set overwrite_experiment=True to overwrite the existing experiment.\n"
                        f"  - Set auto_checkpoint_and_resume_experiment=True to resume (if config/schema match, previous experiment was checkpointed, and previous run did not complete).\n"
                    ),
                    {
                        "experiment_name": self.experiment_name,
                        "experiment_dir": str(self.experiment_directory),
                        "overwrite_experiment": self.overwrite_experiment,
                        "auto_checkpoint_and_resume_experiment": self.auto_checkpoint_and_resume_experiment,
                    },
                )
        log.debug(f"Creating experiment directory structure: {experiment_dir_path}")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Experiment directory structure created: {experiment_dir_path}")

        # Save pipeline config and schema spec files to experiment config directory
        log.debug(
            f"Saving pipeline config and schema spec files to experiment config directory: {experiment_dir_path}"
        )
        pipeline_config_path = self.config_dir / f"config_{self.experiment_name}.yaml"
        schema_spec_path = self.config_dir / f"schema_spec_{self.experiment_name}.yaml"
        serialized_config_dict = delm_config.to_serialized_config_dict()
        serialized_schema_spec_dict = delm_config.to_serialized_schema_spec_dict()
        with open(pipeline_config_path, "w") as f:
            yaml.dump(
                serialized_config_dict, f, default_flow_style=False, sort_keys=False
            )
        with open(schema_spec_path, "w") as f:
            yaml.dump(
                serialized_schema_spec_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        log.debug(
            f"Pipeline config and schema spec files saved to experiment config directory: {experiment_dir_path}"
        )

        self.preprocessed_data_path = (
            self.data_dir
            / f"{PREPROCESSED_DATA_PREFIX}{self.experiment_name}{PREPROCESSED_DATA_SUFFIX}"
        )
        log.debug(f"Experiment initialized: {experiment_dir_path}")

    def _find_config_differences(
        self, config1: dict, config2: dict, path: str = ""
    ) -> list:
        """Recursively find differences between two config dictionaries for error messages."""
        log.debug("Finding config differences...")
        differences = []

        # Get all keys from both configs
        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            # Check if key exists in both configs
            if key not in config1:
                differences.append(f"Missing in current config: {current_path}")
            elif key not in config2:
                differences.append(f"Missing in saved config: {current_path}")
            else:
                val1, val2 = config1[key], config2[key]

                # Recursively compare nested dictionaries
                if isinstance(val1, dict) and isinstance(val2, dict):
                    differences.extend(
                        self._find_config_differences(val1, val2, current_path)
                    )
                # Compare other values directly
                elif val1 != val2:
                    differences.append(f"{current_path}: {val1} != {val2}")
        log.debug(f"Config differences found: {differences}")
        return differences

    def verify_resume_config(self, delm_config: DELMConfig):
        """Compare config/schema in config/ folder to user-supplied DELMConfig. Abort if they differ."""
        config_yaml = self.config_dir / f"config_{self.experiment_name}.yaml"
        schema_yaml = self.config_dir / f"schema_spec_{self.experiment_name}.yaml"
        log.debug(f"Verifying resume configs from: {config_yaml} and {schema_yaml}")
        if not config_yaml.exists() or not schema_yaml.exists():
            log.error(
                f"Cannot resume experiment: config files not found: {config_yaml} and {schema_yaml}"
            )
            raise FileNotFoundError(
                f"Cannot resume experiment: config files not found: {config_yaml} and {schema_yaml}"
            )

        file_config = yaml.safe_load(config_yaml.read_text())
        file_schema = yaml.safe_load(schema_yaml.read_text())

        current_config_dict = delm_config.to_serialized_config_dict()
        current_schema_dict = delm_config.to_serialized_schema_spec_dict()

        if file_config != current_config_dict:
            differences = self._find_config_differences(
                current_config_dict, file_config
            )
            raise ValueError(
                f"Config mismatch: current config does not match the one used for this experiment. \nMismatched fields:\n"
                + "\n".join(f"  - {diff}" for diff in differences)
            )
        if file_schema != current_schema_dict:
            differences = self._find_config_differences(
                current_schema_dict, file_schema
            )
            raise ValueError(
                f"Schema mismatch: current schema does not match the one used for this experiment. \nMismatched fields:\n"
                + "\n".join(f"  - {diff}" for diff in differences)
            )

        log.debug(f"Resume config verified successfully")

    # --- Preprocessing Data ---
    def save_preprocessed_data(self, df: pd.DataFrame) -> Path:
        """Save preprocessed data as feather file."""
        log.debug(f"Saving preprocessed data to: {self.preprocessed_data_path}")
        df.to_feather(self.preprocessed_data_path)
        log.info(f"Preprocessed data saved to: {self.preprocessed_data_path}")
        return self.preprocessed_data_path

    def load_preprocessed_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load preprocessed data from feather file."""
        if file_path is None:
            if not hasattr(self, "preprocessed_data_path"):
                raise ValueError(
                    "Experiment not initialized. Call initialize_experiment() first."
                )
            file_path = self.preprocessed_data_path
        if not file_path.exists():
            log.error(f"Preprocessed data file does not exist: {file_path}")
            raise FileNotFoundError(
                f"Preprocessed data file does not exist: {file_path}"
            )
        if file_path.suffix.lower() != ".feather":
            raise ValueError(
                f"Preprocessed data file must be a feather file: {file_path}"
            )
        log.debug(f"Loading preprocessed data from: {file_path}")
        return pd.read_feather(file_path)

    # --- Batch Checkpointing ---
    def save_batch_checkpoint(self, batch_df: pd.DataFrame, batch_id: int) -> Path:
        """Save a batch checkpoint as a feather file."""
        batch_filename = (
            f"{BATCH_FILE_PREFIX}{batch_id:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        )
        batch_path = self.cache_dir / batch_filename
        log.debug(f"Saving batch checkpoint to: {batch_path}")
        batch_df.to_feather(batch_path)
        log.debug(f"Batch checkpoint saved to: {batch_path}")
        return batch_path

    def list_batch_checkpoints(self) -> List[Path]:
        """List all batch checkpoint files in the processing cache directory."""
        log.debug(f"Listing batch checkpoint files in: {self.cache_dir}")
        batch_files = sorted(
            [p for p in self.cache_dir.glob(f"{BATCH_FILE_PREFIX}*{BATCH_FILE_SUFFIX}")]
        )
        log.debug(f"Batch checkpoint files found: {batch_files}")
        return batch_files

    def load_batch_checkpoint(self, batch_path: Path) -> pd.DataFrame:
        """Load a batch checkpoint from a feather file.

        Args:
            batch_path: Path to the batch feather file.

        Returns:
            The loaded DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not ``.feather``.
        """
        log.debug(f"Loading batch checkpoint from: {batch_path}")
        if not batch_path.exists():
            log.error(f"Batch checkpoint file does not exist: {batch_path}")
            raise FileNotFoundError(
                f"Batch checkpoint file does not exist: {batch_path}"
            )
        if batch_path.suffix.lower() != ".feather":
            raise ValueError(
                f"Batch checkpoint file must be a feather file: {batch_path}"
            )
        return pd.read_feather(batch_path)

    def load_batch_checkpoint_by_id(self, batch_id: int) -> pd.DataFrame:
        """Load a batch checkpoint by batch ID.

        Args:
            batch_id: Batch ID to load.

        Returns:
            The loaded DataFrame.
        """
        log.debug(f"Loading batch checkpoint by ID: {batch_id}")
        batch_filename = (
            f"{BATCH_FILE_PREFIX}{batch_id:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        )
        batch_path = self.cache_dir / batch_filename
        log.debug(f"Batch checkpoint path: {batch_path}")
        df = self.load_batch_checkpoint(batch_path)
        log.debug(f"Batch checkpoint successfully loaded by ID: {batch_id}")
        return df

    def consolidate_batches(self) -> pd.DataFrame:
        """Consolidate all batch files into a single DataFrame and save as final result.

        Returns:
            The concatenated DataFrame across all batch files.

        Raises:
            FileNotFoundError: If no batch files are present.
        """
        batch_files = self.list_batch_checkpoints()
        if not batch_files:
            log.error(f"No batch files found for consolidation.")
            raise FileNotFoundError(f"No batch files found for consolidation.")

        log.debug(f"Consolidating {len(batch_files)} batch files")
        start_time = time.time()
        dfs = [self.load_batch_checkpoint(p) for p in batch_files]
        consolidated_df = pd.concat(dfs, ignore_index=True)
        elapsed_time = time.time() - start_time
        log.debug(
            f"Consolidated {len(batch_files)} batches ({len(consolidated_df)} rows) in {elapsed_time:.2f}s"
        )
        return consolidated_df

    def cleanup_batch_checkpoints(self):
        """Remove all batch checkpoint files after consolidation."""
        batch_files = self.list_batch_checkpoints()
        log.debug(f"Cleaning up {len(batch_files)} batch files")
        for p in batch_files:
            try:
                log.debug(f"Deleting batch file: {p}")
                p.unlink()
                log.debug(f"Batch file deleted: {p}")
            except Exception as e:
                log.error(f"Failed to delete batch file {p}: {e}")

    def get_all_existing_batch_ids(self) -> set:
        """Return a set of all batch IDs for which a checkpoint file exists."""
        log.debug(f"Getting all batch IDs")
        ids = set()
        for p in self.list_batch_checkpoints():
            stem = p.stem
            if stem.startswith(BATCH_FILE_PREFIX):
                try:
                    batch_id = int(stem.split("_")[1])
                    ids.add(batch_id)
                except Exception:
                    continue
        log.debug(f"All existing batch IDs: {ids}")
        return ids

    def get_batch_checkpoint_path(self, batch_id: int) -> Path:
        """Return the full path to the batch checkpoint file for a given batch ID."""
        log.debug(f"Getting batch checkpoint path for batch ID: {batch_id}")
        batch_filename = (
            f"{BATCH_FILE_PREFIX}{batch_id:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        )
        log.debug(f"Batch checkpoint path: {batch_filename}")
        return self.cache_dir / batch_filename

    def delete_batch_checkpoint(self, batch_id: int) -> bool:
        """Delete the batch checkpoint file for a given batch ID.

        Returns:
            True if the file was deleted; False if it did not exist.
        """
        log.debug(f"Deleting batch checkpoint for batch ID: {batch_id}")
        path = self.get_batch_checkpoint_path(batch_id)
        log.debug(f"Batch checkpoint path: {path}")
        if path.exists():
            path.unlink()
            log.debug(f"Batch checkpoint deleted: {path}")
            return True
        log.debug(f"Batch checkpoint not found for batch ID: {batch_id}")
        return False

    # --- State Management ---
    def save_state(self, cost_tracker: CostTracker):
        """Save experiment state (cost tracker only) to state file as JSON."""
        log.debug(f"Saving experiment state to: {self.cache_dir / STATE_FILE_NAME}")
        state_path = self.cache_dir / STATE_FILE_NAME
        state = {
            "cost_tracker": cost_tracker.to_dict(),
        }
        start_time = time.time()
        with open(state_path, "w") as f:
            json.dump(state, f)
        elapsed_time = time.time() - start_time
        log.debug(f"Experiment state saved to: {state_path} in {elapsed_time:.2f}s")
        return state_path

    def load_state(self) -> Optional[CostTracker]:
        """Load experiment state from state file as JSON. Returns dict or None if not found."""
        log.debug(f"Loading experiment state from: {self.cache_dir / STATE_FILE_NAME}")
        state_path = self.cache_dir / STATE_FILE_NAME
        if not state_path.exists():
            log.debug(f"Experiment state file not found: {state_path}")
            return None
        start_time = time.time()
        with open(state_path, "r") as f:
            state = json.load(f)
        elapsed_time = time.time() - start_time
        log.debug(f"Experiment state loaded from: {state_path} in {elapsed_time:.2f}s")
        return CostTracker.from_dict(state["cost_tracker"])

    def save_extracted_data(self, df: pd.DataFrame) -> Path:
        """Save extracted data as feather file."""
        log.debug(
            f"Saving extracted data to: {self.data_dir / CONSOLIDATED_RESULT_PREFIX}{self.experiment_name}{CONSOLIDATED_RESULT_SUFFIX}"
        )
        result_filename = f"{CONSOLIDATED_RESULT_PREFIX}{self.experiment_name}{CONSOLIDATED_RESULT_SUFFIX}"
        result_path = self.data_dir / result_filename
        df.to_feather(result_path)
        log.info(f"Saved extracted data to: {result_path}")
        return result_path

    # --- Private helpers ---
    def _get_experiment_dir(self) -> Path:
        """Return the experiment directory path (does not create it)."""
        log.debug(
            f"Getting experiment directory path: {self.experiment_directory / self.experiment_name}"
        )
        return self.experiment_directory / self.experiment_name


class InMemoryExperimentManager(BaseExperimentManager):
    """Stores all experiment data in memory. Disk-specific features are not supported."""

    def __init__(self, experiment_name: str, **kwargs):
        log.debug(f"Initializing InMemoryExperimentManager: {experiment_name}")
        if kwargs.get("overwrite_experiment", False):
            raise ValueError(
                "overwrite_experiment is not supported for InMemoryExperimentManager."
            )
        if kwargs.get("auto_checkpoint_and_resume_experiment", False):
            raise ValueError(
                "auto_checkpoint_and_resume_experiment is not supported for InMemoryExperimentManager."
            )
        self.experiment_name = experiment_name
        self._preprocessed_data = None
        self._batches = {}  # batch_id -> DataFrame
        self._state = None
        self._extracted_data = None
        self._config_dict = None
        self._schema_dict = None

    def get_results(self) -> pd.DataFrame:
        """Return extracted results held in memory.

        Returns:
            The extracted results DataFrame.

        Raises:
            ValueError: If results have not been saved.
        """
        log.debug(f"Getting results from InMemoryExperimentManager")
        if self._extracted_data is None:
            log.error("Attempted to get results but no extracted data is present.")
            raise ValueError("No extracted data available in memory.")
        log.debug(f"Results available in memory.")
        return self._extracted_data

    def initialize_experiment(self, delm_config: DELMConfig):
        """Initialize in-memory experiment by storing config and schema dicts."""
        log.debug(f"Initializing experiment in InMemoryExperimentManager")
        self._config_dict = delm_config.to_serialized_config_dict()
        self._schema_dict = delm_config.to_serialized_schema_spec_dict()

    def save_preprocessed_data(self, df: pd.DataFrame) -> str:
        """Save preprocessed data in memory.

        Returns:
            The literal string "in-memory".
        """
        self._preprocessed_data = df.copy()
        return "in-memory"

    def load_preprocessed_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        if file_path is not None:
            log.error(
                "Loading preprocessed data from a file path is not supported for InMemoryExperimentManager YET."
            )
            raise NotImplementedError(
                "Loading preprocessed data from a file path is not supported for InMemoryExperimentManager YET."
            )
        if self._preprocessed_data is None:
            log.error(
                "Attempted to load preprocessed data but none is available in memory."
            )
            raise ValueError("No preprocessed data available in memory.")
        return self._preprocessed_data.copy()

    def save_batch_checkpoint(self, batch_df: pd.DataFrame, batch_id: int) -> str:
        """Save a batch checkpoint in memory.

        Returns:
            A synthetic identifier string for the in-memory batch (e.g., "in-memory-batch-3").
        """
        self._batches[batch_id] = batch_df.copy()
        return f"in-memory-batch-{batch_id}"

    def list_batch_checkpoints(self) -> list:
        """List all batch checkpoint IDs in memory."""
        return sorted(self._batches.keys())

    def load_batch_checkpoint(self, batch_path: str) -> pd.DataFrame:
        """Load a batch checkpoint by a synthetic path string.

        Args:
            batch_path: Path string in the form "in-memory-batch-{id}".

        Returns:
            The stored batch DataFrame.

        Raises:
            ValueError: If the path is malformed.
        """
        if not batch_path.startswith("in-memory-batch-"):
            raise ValueError(
                f"Invalid batch path format: {batch_path}. Expected format: 'in-memory-batch-{{id}}'"
            )
        try:
            batch_id = int(batch_path.split("-")[-1])
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid batch path format: {batch_path}. Expected format: 'in-memory-batch-{{id}}'"
            ) from e
        return self.load_batch_checkpoint_by_id(batch_id)

    def load_batch_checkpoint_by_id(self, batch_id: int) -> pd.DataFrame:
        """Load a batch checkpoint by batch ID.

        Args:
            batch_id: The batch identifier previously saved.

        Returns:
            The stored batch DataFrame.

        Raises:
            ValueError: If the batch is not present in memory.
        """
        if batch_id not in self._batches:
            log.error(
                f"Attempted to load batch checkpoint {batch_id} but it's not available in memory."
            )
            raise ValueError(f"No batch checkpoint with id {batch_id} in memory.")
        return self._batches[batch_id].copy()

    def consolidate_batches(self) -> pd.DataFrame:
        """Concatenate all batch DataFrames in memory.

        Returns:
            Concatenated DataFrame across all in-memory batches.

        Raises:
            ValueError: If no batches have been saved.
        """
        if not self._batches:
            log.error(
                "Attempted to consolidate batches but no batch checkpoints are available in memory."
            )
            raise ValueError("No batch checkpoints in memory to consolidate.")
        log.debug(f"Consolidating {len(self._batches)} batches in memory")
        start_time = time.time()
        dfs = [self._batches[bid] for bid in sorted(self._batches.keys())]
        consolidated_df = pd.concat(dfs, ignore_index=True)
        elapsed_time = time.time() - start_time
        log.debug(
            f"Consolidated {len(self._batches)} batches ({len(consolidated_df)} rows) in {elapsed_time:.2f}s"
        )
        return consolidated_df

    def cleanup_batch_checkpoints(self):
        """Remove all batch checkpoints from memory."""
        self._batches.clear()

    def get_all_existing_batch_ids(self) -> set:
        """Return all batch IDs stored in memory."""
        return set(self._batches.keys())

    def get_batch_checkpoint_path(self, batch_id: int) -> str:
        """Return the synthetic path string for a batch ID."""
        return f"in-memory-batch-{batch_id}"

    def delete_batch_checkpoint(self, batch_id: int) -> bool:
        """Delete a batch checkpoint by ID.

        Returns:
            True if the checkpoint existed and was removed; False otherwise.
        """
        if batch_id in self._batches:
            del self._batches[batch_id]
            return True
        return False

    def save_state(self, cost_tracker: CostTracker):
        """Save the cost tracker in memory."""
        self._state = cost_tracker

    def load_state(self) -> Optional[CostTracker]:
        return self._state

    def save_extracted_data(self, df: pd.DataFrame) -> str:
        """Save extracted data in memory.

        Returns:
            The literal string "in-memory".
        """
        self._extracted_data = df.copy()
        return "in-memory"
