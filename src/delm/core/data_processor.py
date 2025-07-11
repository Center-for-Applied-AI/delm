"""
DELM Data Processor
==================
Handles data loading, preprocessing, chunking, and scoring.
"""

from pathlib import Path
from typing import Union
import pandas as pd

from ..strategies import loader_factory
from ..config import DataConfig
from ..constants import (
    SYSTEM_CHUNK_COLUMN, SYSTEM_SCORE_COLUMN, SYSTEM_CHUNK_ID_COLUMN
)


class DataProcessor:
    """Handles data loading, preprocessing, chunking, and scoring."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.splitter = config.splitting.strategy
        self.scorer = config.scoring.scorer
        self.target_column = config.target_column
        self.drop_target_column = config.drop_target_column
        
        # Constants
        self.TARGET_COLUMN_NAME = self.target_column
        self.CHUNK_COLUMN_NAME = SYSTEM_CHUNK_COLUMN  # This is internal, not configurable
    
    def load_and_process(self, data_source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data and apply preprocessing pipeline."""
        df = self._load_data(data_source)
        return self._process_dataframe(df)
    
    def _load_data(self, data_source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data from various sources."""
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
                    if isinstance(data, pd.DataFrame):
                        df = data
                    else:
                        raise ValueError("CSV loader should return DataFrame")
                elif file_suffix == ".txt":
                    # data is a string for text-based files
                    if isinstance(data, str):
                        df = pd.DataFrame({
                            self.TARGET_COLUMN_NAME: [data]
                        })
                    else:
                        raise ValueError("Text loader should return string")
                else:
                    raise ValueError(f"Unsupported file type: {path.suffix}")
                        
            except ValueError:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        else:
            # Handle DataFrame input
            if self.target_column == "":
                raise ValueError("Target column is required for DataFrame input")
            df = data_source.copy()
        
        return df
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply chunking and scoring to DataFrame."""
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
        
        return df 