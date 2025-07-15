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
        if not self.target_column:
            self.target_column = DEFAULT_TARGET_COLUMN

        self.drop_target_column = config.drop_target_column
        self.pandas_score_filter = config.pandas_score_filter
    

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
                            self.target_column: [data]
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
        
        # Check for invalid configuration: dropping target column without splitting
        if self.drop_target_column and self.splitter is None:
            raise DataError(
                "Cannot drop target column when no splitting strategy is specified",
                {
                    "target_column": self.target_column,
                    "drop_target_column": self.drop_target_column,
                    "suggestion": "Either specify a splitting strategy or set drop_target_column=False"
                }
            )
            
        # 1. Chunk the data (or use target column if no splitting)
        if self.splitter is not None:
            # Apply splitting strategy - use system chunk column name
            df[SYSTEM_CHUNK_COLUMN] = df[self.target_column].apply(self.splitter.split)
            df = df.explode(SYSTEM_CHUNK_COLUMN).reset_index(drop=True)
            self.chunk_column = SYSTEM_CHUNK_COLUMN
        else:
            # No splitting - use target column name as chunk column (no duplication)
            self.chunk_column = self.target_column
        
        df[SYSTEM_CHUNK_ID_COLUMN] = range(len(df))
        
        # Drop target column if requested (only when splitting was done)
        if self.drop_target_column and self.splitter is not None:
            df = df.drop(columns=[self.target_column])
        elif self.drop_target_column and self.splitter is None:
            # This case is handled by the error above, but just in case
            pass

        # 2. Score and filter the chunks (only if scorer is provided)
        if self.scorer is not None:
            df[SYSTEM_SCORE_COLUMN] = df[self.chunk_column].apply(self.scorer.score)
            # TODO: Give warning if scorer is used but filter is none.
            if self.pandas_score_filter is not None:
                df = df.query(self.pandas_score_filter)

        return df 