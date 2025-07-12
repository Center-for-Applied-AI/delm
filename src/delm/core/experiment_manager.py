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

from ..config import ExperimentConfig
from ..constants import PREPROCESSED_DIR_NAME
from ..exceptions import ExperimentError, FileError


class ExperimentManager:
    """Handles experiment directories and file I/O."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_dir = self._setup_experiment_directory()
        self.verbose = config.verbose
    
    def _setup_experiment_directory(self) -> Path:
        """Create and validate experiment directory."""
        # Make sure experiments dir exists and create if not
        experiments_dir = self.config.directory
        experiments_dir.mkdir(parents=True, exist_ok=True)

        # Check if experiment name is already in use
        if (experiments_dir / self.config.name).exists() and not self.config.overwrite_experiment:
            raise ExperimentError(
                f"Experiment name '{self.config.name}' already exists in {experiments_dir}",
                {
                    "experiment_name": self.config.name,
                    "experiment_dir": str(experiments_dir),
                    "suggestion": "Use overwrite_experiment=True or choose a different name"
                }
            )
        elif (experiments_dir / self.config.name).exists() and self.config.overwrite_experiment:
            shutil.rmtree(experiments_dir / self.config.name)

        # Create experiment directory
        experiment_dir = experiments_dir / self.config.name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        return experiment_dir
    
    def save_preprocessed_data(self, df: pd.DataFrame, experiment_name: str) -> Path:
        """Save preprocessed data as feather file."""
        # Create preprocessed directory
        preprocessed_dir = self.experiment_dir / PREPROCESSED_DIR_NAME
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on experiment name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = f"{experiment_name}_{timestamp}"
        
        # Save as feather file
        feather_path = preprocessed_dir / f"{source_name}_prepped.feather"
        df.to_feather(feather_path)
        
        if self.verbose:
            print(f"Saved preprocessed data to: {feather_path}")
        
        return feather_path
    
    def load_preprocessed_data(self, file_path: Path) -> pd.DataFrame:
        """Load preprocessed data from feather file."""
        if not file_path.exists():
            raise FileError(
                f"Preprocessed data file does not exist: {file_path}",
                {"file_path": str(file_path), "suggestion": "Run preprocessing first"}
            )
        
        try:
            return pd.read_feather(file_path)
        except Exception as e:
            raise FileError(
                f"Failed to load preprocessed data from {file_path}",
                {"file_path": str(file_path)}
            ) from e 