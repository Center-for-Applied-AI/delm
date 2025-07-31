"""
DELM Data Processor
==================
Handles data loading, preprocessing, chunking, and scoring.
"""

import logging
from pathlib import Path
from typing import Union
import pandas as pd

# Module-level logger
log = logging.getLogger(__name__)

from delm.strategies import loader_factory
from delm.config import DataPreprocessingConfig
from delm.constants import (
    SYSTEM_CHUNK_COLUMN, 
    SYSTEM_SCORE_COLUMN, 
    SYSTEM_CHUNK_ID_COLUMN,
    SYSTEM_RECORD_ID_COLUMN,
    SYSTEM_RAW_DATA_COLUMN
)

class DataProcessor:
    """Handles data loading, preprocessing, chunking, and scoring."""
    
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.splitter = config.splitting.strategy
        self.scorer = config.scoring.scorer
        self.target_column = config.target_column
        self.drop_target_column = config.drop_target_column
        self.pandas_score_filter = config.pandas_score_filter
    
    def load_data(self, data_source: Union[str,Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data from various sources.
        
        Args:
            data_source: The data source to load. Can be a path to a file or directory (str or Path), or a DataFrame.

        Returns:
            A DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data source path does not exist.
            ValueError: If the target column is not found in the data source or if the data source is a directory and contains multiple file types.
        """

        if isinstance(data_source, (str, Path)):
            # Handle file loading
            path = Path(data_source)
            log.debug("Loading data from path: %s", path)
            
            if not path.exists():
                raise FileNotFoundError(f"Data Source path does not exist: {path}")
            
            # Check if file or directory
            if path.is_file():
                log.debug("Loading single file: %s", path)
                # Load file
                loaded_df = loader_factory.load_file(path)
                extension = path.suffix
            elif path.is_dir():
                log.debug("Loading directory: %s", path)
                # Load directory
                loaded_df, extension = loader_factory.load_directory(path)
            self.extension_requires_target_column = loader_factory.requires_target_column(extension)
            
            log.debug("Loaded %d records with extension %s", len(loaded_df), extension)
            
            if self.extension_requires_target_column and (not self.target_column or self.target_column == ""):
                raise ValueError(
                    f"Target column is required for {path.suffix} files, file_path: {str(path)}, file_type: {path.suffix}, suggestion: Specify target_column in config"
                )
            if self.target_column not in loaded_df.columns:
                log.error("Target column '%s' not found in columns: %s", self.target_column, loaded_df.columns)
                if self.target_column == SYSTEM_RAW_DATA_COLUMN:
                    raise ValueError(
                        f"Target column {self.target_column} is not allowed for {extension} files, path: {str(path)}, extension: {extension}, suggestion: Remove target_column from config"
                    )
                else:
                    raise ValueError(
                        f"Target column {self.target_column} not found in data columns {loaded_df.columns}, path: {str(path)}, extension: {extension}, suggestion: Specify target_column in config"
                    )
        else:
            # Handle DataFrame input
            log.debug("Loading data from DataFrame with %d records", len(data_source))
            if self.target_column not in data_source.columns:
                raise ValueError(
                    f"Target column {self.target_column} not found in data source, data_source_columns: {data_source.columns}, target_column: {self.target_column}, suggestion: Specify valid target_column in config"
                )
            
            loaded_df = data_source.copy()
        loaded_df[SYSTEM_RECORD_ID_COLUMN] = range(len(loaded_df))
        log.debug("Data loading completed. Total records: %d", len(loaded_df))
        return loaded_df
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply chunking and scoring to DataFrame.
        
        Args:
            df: The DataFrame to process.

        Returns:
            A DataFrame containing the processed data.

        Raises:
            ValueError: If drop_target_column is True and no splitting strategy is specified.
        """
        
        log.debug("Processing DataFrame with %d records", len(df))
        df = df.copy()

        # Check for invalid configuration: dropping target column without splitting
        if self.drop_target_column and self.splitter is None:
            raise ValueError(
                f"Cannot drop target column when no splitting strategy is specified, target_column: {self.target_column}, drop_target_column: {self.drop_target_column}, suggestion: Either specify a splitting strategy or set drop_target_column=False"
            )

        # 1. Chunk the data (or use target column if no splitting)
        if self.splitter is not None:
            log.debug("Applying splitting strategy: %s", type(self.splitter).__name__)
            # Apply splitting strategy - use system chunk column name
            df.loc[:, SYSTEM_CHUNK_COLUMN] = df[self.target_column].apply(self.splitter.split)
            df = df.explode(SYSTEM_CHUNK_COLUMN).reset_index(drop=True)
            log.debug("Splitting completed. Generated %d chunks", len(df))
        else:
            log.debug("No splitting strategy specified, using target column as chunks")
            # No splitting - use target column name as chunk column (no duplication)
            df = df.rename(columns={self.target_column: SYSTEM_CHUNK_COLUMN})
        
        df[SYSTEM_CHUNK_ID_COLUMN] = range(len(df))
        
        # Drop target column if requested (only when splitting was done)
        if self.drop_target_column and self.splitter is not None:
            log.debug("Dropping target column: %s", self.target_column)
            df = df.drop(columns=[self.target_column])
        elif self.drop_target_column and self.splitter is None:
            # This case is handled by the error above, but just in case
            pass

        # 2. Score and filter the chunks (only if scorer is provided)
        if self.scorer is not None:
            log.debug("Applying scoring strategy: %s", type(self.scorer).__name__)
            df[SYSTEM_SCORE_COLUMN] = df[SYSTEM_CHUNK_COLUMN].apply(self.scorer.score)
            if self.pandas_score_filter is not None:
                log.debug("Applying score filter: %s", self.pandas_score_filter)
                original_count = len(df)
                df = df.query(self.pandas_score_filter)
                log.debug("Score filtering completed. Filtered from %d to %d chunks", original_count, len(df))
            else:
                log.warning("Scoring strategy is used but filter is not. This means all chunks will be used for extraction.")

        log.debug("DataFrame processing completed. Final chunks: %d", len(df))
        return df 