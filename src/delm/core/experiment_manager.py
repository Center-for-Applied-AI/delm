"""
DELM Experiment Manager
======================
Handles experiment directories and file I/O.
"""

import datetime
import shutil
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import json
import hashlib

from ..config import ExperimentConfig
from ..constants import DATA_DIR_NAME, CACHE_DIR_NAME, PROCESSING_CACHE_DIR_NAME, BATCH_FILE_PREFIX, BATCH_FILE_SUFFIX, BATCH_FILE_DIGITS, STATE_FILE_NAME, CONSOLIDATED_RESULT_PREFIX, CONSOLIDATED_RESULT_SUFFIX, PREPROCESSED_DATA_PREFIX, PREPROCESSED_DATA_SUFFIX, SYSTEM_EXTRACTED_DATA_JSON_COLUMN
from ..exceptions import ExperimentError, FileError

class ExperimentManager:
    """Handles experiment directories, config/schema validation, batch checkpointing, and state management."""

    def __init__(self, experiment_name: str, experiment_directory: Path, overwrite_experiment: bool = False, auto_checkpoint_and_resume_experiment: bool = True):
        self.experiment_name = experiment_name
        self.experiment_directory = experiment_directory
        self.overwrite_experiment = overwrite_experiment
        self.auto_checkpoint_and_resume_experiment = auto_checkpoint_and_resume_experiment
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

    def get_results(self) -> pd.DataFrame:
        """
        Get the results from the experiment directory.
        """
        return pd.read_feather(self.data_dir / f"{CONSOLIDATED_RESULT_PREFIX}{self.experiment_name}{CONSOLIDATED_RESULT_SUFFIX}")

    def initialize_experiment(self, config_dict: dict, schema_dict: dict):
        """Validate and create experiment directory structure, write config and schema files."""
        experiment_dir_path = self.experiment_dir
        if experiment_dir_path.exists():
            if self.overwrite_experiment:
                shutil.rmtree(experiment_dir_path)
            elif self.auto_checkpoint_and_resume_experiment:
                # Verify config/schema match before resuming
                self.verify_resume_config(config_dict, schema_dict)
            else:
                raise ExperimentError(
                    (
                        f"\nExperiment directory already exists. To proceed, choose one of the following:\n"
                        f"  - Set overwrite_experiment=True to overwrite the existing experiment.\n"
                        f"  - Set auto_checkpoint_and_resume_experiment=True to resume (if config/schema match, previous experiment was checkpointed, and previous run did not complete).\n"
                    ),
                    {
                        "experiment_name": self.experiment_name,
                        "experiment_dir": str(self.experiment_directory)
                    }
                )
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Save config and schema files to experiment config directory
        import yaml
        config_yaml = self.config_dir / f"config_{self.experiment_name}.yaml"
        schema_yaml = self.config_dir / f"schema_spec_{self.experiment_name}.yaml"
        config_yaml.write_text(yaml.safe_dump(config_dict))
        schema_yaml.write_text(yaml.safe_dump(schema_dict))
        self.preprocessed_data_path = self.data_dir / f"{PREPROCESSED_DATA_PREFIX}{self.experiment_name}{PREPROCESSED_DATA_SUFFIX}"

    def _find_config_differences(self, config1: dict, config2: dict, path: str = "") -> list:
        """Recursively find differences between two config dictionaries for error messages."""
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
                    differences.extend(self._find_config_differences(val1, val2, current_path))
                # Compare other values directly
                elif val1 != val2:
                    differences.append(f"{current_path}: {val1} != {val2}")
        
        return differences

    def verify_resume_config(self, config_dict: dict, schema_dict: dict):
        """Compare config/schema in config/ folder to user-supplied config/schema. Abort if they differ."""
        import yaml
        config_yaml = self.config_dir / f"config_{self.experiment_name}.yaml"
        schema_yaml = self.config_dir / f"schema_spec_{self.experiment_name}.yaml"
        file_config = yaml.safe_load(config_yaml.read_text())
        file_schema = yaml.safe_load(schema_yaml.read_text())
        
        if file_config != config_dict:
            differences = self._find_config_differences(config_dict, file_config)
            raise ExperimentError(
                f"Config mismatch: current config does not match the one used for this experiment.\n\nMismatched fields:\n" + "\n".join(f"  - {diff}" for diff in differences),
                {"mismatched_fields": differences}
            )
        if file_schema != schema_dict:
            differences = self._find_config_differences(schema_dict, file_schema)
            raise ExperimentError(
                f"Schema mismatch: current schema does not match the one used for this experiment.\n\nMismatched fields:\n" + "\n".join(f"  - {diff}" for diff in differences),
                {"mismatched_fields": differences}
            )

    # --- Preprocessing Data ---
    def save_preprocessed_data(self, df: pd.DataFrame) -> Path:
        """Save preprocessed data as feather file."""
        df.to_feather(self.preprocessed_data_path)
        return self.preprocessed_data_path

    def load_preprocessed_data(self, file_path: Path | None = None) -> pd.DataFrame:
        """Load preprocessed data from feather file."""
        if file_path is None:
            file_path = self.preprocessed_data_path
        if not file_path.exists():
            raise FileError(
                f"Preprocessed data file does not exist: {file_path}",
                {"file_path": str(file_path), "suggestion": "Run preprocessing first, or specify preprocessed_file_path to the preprocessed data path"}
            )
        try:
            df = pd.read_feather(file_path)
            return df
        except Exception as e:
            raise FileError(
                f"Failed to load preprocessed data from {file_path}",
                {"file_path": str(file_path)}
            ) from e

    # --- Batch Checkpointing ---
    def save_batch_checkpoint(self, batch_df: pd.DataFrame, batch_id: int) -> Path:
        """Save a batch checkpoint as a feather file."""
        batch_filename = f"{BATCH_FILE_PREFIX}{batch_id:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        batch_path = self.cache_dir / batch_filename
        batch_df.to_feather(batch_path)
        return batch_path

    def list_batch_checkpoints(self) -> List[Path]:
        """List all batch checkpoint files in the processing cache directory."""
        return sorted([p for p in self.cache_dir.glob(f"{BATCH_FILE_PREFIX}*{BATCH_FILE_SUFFIX}")])

    def load_batch_checkpoint(self, batch_path: Path) -> pd.DataFrame:
        """Load a batch checkpoint from a feather file."""
        if not batch_path.exists():
            raise FileError(
                f"Batch checkpoint file does not exist: {batch_path}",
                {"file_path": str(batch_path), "suggestion": "Check batch processing or rerun experiment."}
            )
        try:
            df = pd.read_feather(batch_path)
            return df
        except Exception as e:
            raise FileError(
                f"Failed to load batch checkpoint: {batch_path}",
                {"file_path": str(batch_path)}
            ) from e

    def load_batch_checkpoint_by_id(self, batch_id: int) -> pd.DataFrame:
        """Load a batch checkpoint by batch ID."""
        batch_filename = f"{BATCH_FILE_PREFIX}{batch_id:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        batch_path = self.cache_dir / batch_filename
        return self.load_batch_checkpoint(batch_path)

    def consolidate_batches(self) -> pd.DataFrame:
        """Consolidate all batch files into a single DataFrame and save as final result."""
        batch_files = self.list_batch_checkpoints()
        if not batch_files:
            raise FileError(
                "No batch files found for consolidation.",
                {"suggestion": "Check that batch processing completed successfully."}
            )
        dfs = [self.load_batch_checkpoint(p) for p in batch_files]
        consolidated_df = pd.concat(dfs, ignore_index=True)
        return consolidated_df

    def cleanup_batch_checkpoints(self):
        """Remove all batch checkpoint files after consolidation."""
        batch_files = self.list_batch_checkpoints()
        for p in batch_files:
            try:
                p.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete batch file {p}: {e}")

    def get_all_batch_ids(self) -> set:
        """Return a set of all batch IDs for which a checkpoint file exists."""
        ids = set()
        for p in self.list_batch_checkpoints():
            stem = p.stem
            if stem.startswith(BATCH_FILE_PREFIX):
                try:
                    batch_id = int(stem.split('_')[1])
                    ids.add(batch_id)
                except Exception:
                    continue
        return ids

    def get_batch_checkpoint_path(self, batch_id: int) -> Path:
        """Return the full path to the batch checkpoint file for a given batch ID."""
        batch_filename = f"{BATCH_FILE_PREFIX}{batch_id:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        return self.cache_dir / batch_filename

    def delete_batch_checkpoint(self, batch_id: int) -> bool:
        """Delete the batch checkpoint file for a given batch ID. Returns True if deleted."""
        path = self.get_batch_checkpoint_path(batch_id)
        if path.exists():
            path.unlink()
            return True
        return False

    # --- State Management ---
    def save_state(self, cost_tracker: Any):
        """Save experiment state (cost tracker only) to state file as JSON."""
        state_path = self.cache_dir / STATE_FILE_NAME
        state = {
            "cost_tracker": cost_tracker.to_dict(),
        }
        with open(state_path, "w") as f:
            json.dump(state, f)
        return state_path

    def load_state(self):
        """Load experiment state from state file as JSON. Returns dict or None if not found."""
        state_path = self.cache_dir / STATE_FILE_NAME
        if not state_path.exists():
            return None
        with open(state_path, "r") as f:
            state = json.load(f)
        return state


    def save_extracted_data(self, df: pd.DataFrame) -> Path:
        result_filename = f"{CONSOLIDATED_RESULT_PREFIX}{self.experiment_name}{CONSOLIDATED_RESULT_SUFFIX}"
        result_path = self.data_dir / result_filename
        df.to_feather(result_path)
        return result_path


    # --- Private helpers ---
    def _get_experiment_dir(self) -> Path:
        """Return the experiment directory path (does not create it)."""
        return self.experiment_directory / self.experiment_name 