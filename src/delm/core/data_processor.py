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
    SYSTEM_CHUNK_COLUMN, SYSTEM_SCORE_COLUMN, SYSTEM_CHUNK_ID_COLUMN,
    DEFAULT_TARGET_COLUMN
)
from ..exceptions import DataError, ValidationError


class DataProcessor:
    """Handles data loading, preprocessing, chunking, and scoring."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.splitter = config.splitting.strategy
        self.scorer = config.scoring.scorer
        self.target_column = config.target_column
        self.drop_target_column = config.drop_target_column
        
        # Constants
        self.TARGET_COLUMN_NAME = self.target_column if self.target_column else DEFAULT_TARGET_COLUMN
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
                self.extension_requires_target_column = loader_factory.requires_target_column(path.suffix)
                
                if self.extension_requires_target_column:
                    if self.target_column == "":
                        raise ValidationError(
                            f"Target column is required for {path.suffix} files",
                            {"file_path": str(path), "file_type": path.suffix, "suggestion": "Specify target_column in config"}
                        )
                    if isinstance(data, pd.DataFrame):
                        df = data
                    else:
                        raise DataError(
                            f"{path.suffix} loader should return DataFrame",
                            {"file_path": str(path), "actual_type": type(data).__name__}
                        )
                else:
                    # data is a string for text-based files
                    if isinstance(data, str):
                        df = pd.DataFrame({
                            self.TARGET_COLUMN_NAME: [data]
                        })
                    else:
                        raise DataError(
                            f"{path.suffix} loader should return string",
                            {"file_path": str(path), "actual_type": type(data).__name__}
                        )
                        
            except ValueError:
                raise DataError(
                    f"Unsupported file type: {path.suffix}",
                    {"file_path": str(path), "file_extension": path.suffix, "suggestion": "Use supported file types"}
                )
        else:
            # Handle DataFrame input
            if self.target_column == "":
                raise ValidationError(
                    "Target column is required for DataFrame input",
                    {"data_type": "DataFrame", "suggestion": "Specify target_column in config"}
                )
            df = data_source.copy()
        
        return df
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply chunking and scoring to DataFrame."""
        # Process the DataFrame (chunking and scoring)
        target_column = self.TARGET_COLUMN_NAME
            
        # 1. Chunk the data
        df[self.CHUNK_COLUMN_NAME] = df[target_column].apply(self.splitter.split)
        df = df.explode(self.CHUNK_COLUMN_NAME).reset_index(drop=True)
        df[SYSTEM_CHUNK_ID_COLUMN] = range(len(df))
        # Drop target column if requested or if it's a text file
        if self.drop_target_column or not self.extension_requires_target_column:
            df = df.drop(columns=[target_column])

        # 2. Score the chunks
        df[SYSTEM_SCORE_COLUMN] = df[self.CHUNK_COLUMN_NAME].apply(self.scorer.score)

        # 3. TODO: Implement filtering by score
        
        return df 