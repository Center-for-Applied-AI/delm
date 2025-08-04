"""
Unit tests for DELM data processor.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from delm.core.data_processor import DataProcessor
from delm.config import DataPreprocessingConfig, SplittingConfig, ScoringConfig
from delm.strategies.splitting_strategies import ParagraphSplit, FixedWindowSplit, RegexSplit
from delm.strategies.scoring_strategies import KeywordScorer, FuzzyScorer
from delm.constants import SYSTEM_CHUNK_COLUMN, SYSTEM_SCORE_COLUMN, SYSTEM_CHUNK_ID_COLUMN, SYSTEM_RECORD_ID_COLUMN, SYSTEM_RAW_DATA_COLUMN


class TestDataProcessor:
    """Test the DataProcessor class."""
    
    def test_initialization(self):
        """Test DataProcessor initialization with default config."""
        config = DataPreprocessingConfig()
        processor = DataProcessor(config)
        
        assert processor.config == config
        assert processor.splitter is None
        assert processor.scorer is None
        assert processor.target_column == SYSTEM_RAW_DATA_COLUMN
        assert processor.drop_target_column is False  # Default is False
        assert processor.pandas_score_filter is None
    
    def test_initialization_with_splitting(self):
        """Test DataProcessor initialization with splitting strategy."""
        splitting_config = SplittingConfig(strategy=ParagraphSplit())
        config = DataPreprocessingConfig(splitting=splitting_config)
        processor = DataProcessor(config)
        
        assert isinstance(processor.splitter, ParagraphSplit)
    
    def test_initialization_with_scoring(self):
        """Test DataProcessor initialization with scoring strategy."""
        scoring_config = ScoringConfig(scorer=KeywordScorer(["test", "keyword"]))
        config = DataPreprocessingConfig(scoring=scoring_config)
        processor = DataProcessor(config)
        
        assert isinstance(processor.scorer, KeywordScorer)
    
    def test_initialization_with_custom_target_column(self):
        """Test DataProcessor initialization with custom target column."""
        config = DataPreprocessingConfig(target_column="custom_column")
        processor = DataProcessor(config)
        
        assert processor.target_column == "custom_column"
    
    def test_initialization_with_pandas_score_filter(self):
        """Test DataProcessor initialization with pandas score filter."""
        config = DataPreprocessingConfig(pandas_score_filter="score > 0.5")
        processor = DataProcessor(config)
        
        assert processor.pandas_score_filter == "score > 0.5"
    
    def test_load_data_from_dataframe(self):
        """Test loading data from DataFrame."""
        config = DataPreprocessingConfig(target_column="text")
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            "text": ["Hello world", "Test data"],
            "other": [1, 2]
        })
        
        result = processor.load_data(df)
        
        assert len(result) == 2
        assert SYSTEM_RECORD_ID_COLUMN in result.columns
        assert "text" in result.columns
        assert "other" in result.columns
        assert result[SYSTEM_RECORD_ID_COLUMN].tolist() == [0, 1]
    
    def test_load_data_from_dataframe_missing_target_column(self):
        """Test loading data from DataFrame with missing target column."""
        config = DataPreprocessingConfig(target_column="missing_column")
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            "text": ["Hello world", "Test data"],
            "other": [1, 2]
        })
        
        with pytest.raises(ValueError, match="Target column missing_column not found"):
            processor.load_data(df)
    
    def test_load_data_from_file(self, tmp_path):
        """Test loading data from file."""
        config = DataPreprocessingConfig(target_column="text")
        processor = DataProcessor(config)
        
        # Create a CSV file
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "text": ["Hello world", "Test data"],
            "other": [1, 2]
        })
        df.to_csv(csv_file, index=False)
        
        with patch('delm.core.data_processor.loader_factory') as mock_factory:
            mock_factory.load_file.return_value = df
            mock_factory.requires_target_column.return_value = True
            
            result = processor.load_data(csv_file)
            
            mock_factory.load_file.assert_called_once_with(csv_file)
            assert len(result) == 2
            assert SYSTEM_RECORD_ID_COLUMN in result.columns
    
    def test_load_data_from_directory(self, tmp_path):
        """Test loading data from directory."""
        config = DataPreprocessingConfig(target_column="text")
        processor = DataProcessor(config)
        
        # Create a directory with CSV files
        csv_dir = tmp_path / "data"
        csv_dir.mkdir()
        
        df = pd.DataFrame({
            "text": ["Hello world", "Test data"],
            "other": [1, 2]
        })
        
        with patch('delm.core.data_processor.loader_factory') as mock_factory:
            mock_factory.load_directory.return_value = (df, ".csv")
            mock_factory.requires_target_column.return_value = True
            
            result = processor.load_data(csv_dir)
            
            mock_factory.load_directory.assert_called_once_with(csv_dir)
            assert len(result) == 2
            assert SYSTEM_RECORD_ID_COLUMN in result.columns
    
    def test_load_data_file_not_found(self, tmp_path):
        """Test loading data from non-existent file."""
        config = DataPreprocessingConfig(target_column="text")
        processor = DataProcessor(config)
        
        non_existent_file = tmp_path / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            processor.load_data(non_existent_file)
    
    def test_load_data_csv_requires_target_column_missing(self, tmp_path):
        """Test loading CSV file that requires target column but none specified."""
        config = DataPreprocessingConfig(target_column="")
        processor = DataProcessor(config)
        
        csv_file = tmp_path / "test.csv"
        csv_file.touch()
        
        with patch('delm.core.data_processor.loader_factory') as mock_factory:
            mock_factory.load_file.return_value = pd.DataFrame({"text": ["test"]})
            mock_factory.requires_target_column.return_value = True
            
            with pytest.raises(ValueError, match="Target column is required for .csv files"):
                processor.load_data(csv_file)
    
    def test_load_data_target_column_not_found(self, tmp_path):
        """Test loading file with target column not found in data."""
        config = DataPreprocessingConfig(target_column="missing_column")
        processor = DataProcessor(config)
        
        csv_file = tmp_path / "test.csv"
        csv_file.touch()
        
        with patch('delm.core.data_processor.loader_factory') as mock_factory:
            mock_factory.load_file.return_value = pd.DataFrame({"text": ["test"]})
            mock_factory.requires_target_column.return_value = True
            
            with pytest.raises(ValueError, match="Target column missing_column not found"):
                processor.load_data(csv_file)
    
    def test_load_data_target_column_system_raw_data_not_allowed(self, tmp_path):
        """Test that SYSTEM_RAW_DATA_COLUMN is not allowed for files that require target column."""
        config = DataPreprocessingConfig(target_column=SYSTEM_RAW_DATA_COLUMN)
        processor = DataProcessor(config)
        
        csv_file = tmp_path / "test.csv"
        csv_file.touch()
        
        with patch('delm.core.data_processor.loader_factory') as mock_factory:
            mock_factory.load_file.return_value = pd.DataFrame({"text": ["test"]})
            mock_factory.requires_target_column.return_value = True
            
            with pytest.raises(ValueError, match=f"Target column {SYSTEM_RAW_DATA_COLUMN} is not allowed"):
                processor.load_data(csv_file)
    
    def test_process_dataframe_no_splitting_no_scoring(self):
        """Test processing DataFrame without splitting or scoring."""
        config = DataPreprocessingConfig()
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world", "Test data"],
            "other": [1, 2]
        })
        
        result = processor.process_dataframe(df)
        
        assert len(result) == 2
        assert SYSTEM_CHUNK_COLUMN in result.columns
        assert SYSTEM_CHUNK_ID_COLUMN in result.columns
        # SYSTEM_RECORD_ID_COLUMN is added in load_data, not process_dataframe
        assert result[SYSTEM_CHUNK_COLUMN].tolist() == ["Hello world", "Test data"]
        assert result[SYSTEM_CHUNK_ID_COLUMN].tolist() == [0, 1]
    
    def test_process_dataframe_with_splitting(self):
        """Test processing DataFrame with splitting strategy."""
        splitting_config = SplittingConfig(strategy=ParagraphSplit())
        config = DataPreprocessingConfig(splitting=splitting_config)
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world.\n\nTest data.\n\nAnother paragraph."],
            "other": [1]
        })
        
        result = processor.process_dataframe(df)
        
        assert len(result) == 3  # Split into 3 paragraphs
        assert SYSTEM_CHUNK_COLUMN in result.columns
        assert SYSTEM_CHUNK_ID_COLUMN in result.columns
        # SYSTEM_RECORD_ID_COLUMN is added in load_data, not process_dataframe
        assert "Hello world." in result[SYSTEM_CHUNK_COLUMN].values
        assert "Test data." in result[SYSTEM_CHUNK_COLUMN].values
        assert "Another paragraph." in result[SYSTEM_CHUNK_COLUMN].values
    
    def test_process_dataframe_with_scoring(self):
        """Test processing DataFrame with scoring strategy."""
        scoring_config = ScoringConfig(scorer=KeywordScorer(["test", "data"]))
        config = DataPreprocessingConfig(scoring=scoring_config)
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world", "Test data", "Another text"],
            "other": [1, 2, 3]
        })
        
        result = processor.process_dataframe(df)
        
        assert len(result) == 3
        assert SYSTEM_SCORE_COLUMN in result.columns
        assert SYSTEM_CHUNK_COLUMN in result.columns
        assert SYSTEM_CHUNK_ID_COLUMN in result.columns
        # SYSTEM_RECORD_ID_COLUMN is added in load_data, not process_dataframe
        
        # Check that scores are calculated
        scores = result[SYSTEM_SCORE_COLUMN].tolist()
        assert all(isinstance(score, (int, float)) for score in scores)
        assert scores[1] > scores[0]  # "Test data" should have higher score than "Hello world"
    
    def test_process_dataframe_with_splitting_and_scoring(self):
        """Test processing DataFrame with both splitting and scoring."""
        splitting_config = SplittingConfig(strategy=ParagraphSplit())
        scoring_config = ScoringConfig(scorer=KeywordScorer(["test", "data"]))
        config = DataPreprocessingConfig(splitting=splitting_config, scoring=scoring_config)
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world.\n\nTest data.\n\nAnother text."],
            "other": [1]
        })
        
        result = processor.process_dataframe(df)
        
        assert len(result) == 3  # Split into 3 paragraphs
        assert SYSTEM_SCORE_COLUMN in result.columns
        assert SYSTEM_CHUNK_COLUMN in result.columns
        assert SYSTEM_CHUNK_ID_COLUMN in result.columns
        # SYSTEM_RECORD_ID_COLUMN is added in load_data, not process_dataframe
        
        # Check that scores are calculated for each chunk
        scores = result[SYSTEM_SCORE_COLUMN].tolist()
        assert all(isinstance(score, (int, float)) for score in scores)
    
    def test_process_dataframe_with_pandas_score_filter(self):
        """Test processing DataFrame with pandas score filter."""
        scoring_config = ScoringConfig(scorer=KeywordScorer(["test", "data"]))
        config = DataPreprocessingConfig(
            scoring=scoring_config,
            pandas_score_filter=f"{SYSTEM_SCORE_COLUMN} > 0.5"  # Use the constant
        )
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world", "Test data", "Another text"],
            "other": [1, 2, 3]
        })
        
        result = processor.process_dataframe(df)
        
        # Should filter out chunks with score <= 0.5
        assert len(result) <= 3
        if len(result) > 0:
            assert all(result[SYSTEM_SCORE_COLUMN] > 0.5)
    
    def test_process_dataframe_drop_target_column(self):
        """Test processing DataFrame with drop_target_column=True."""
        # Need splitting strategy when drop_target_column=True
        splitting_config = SplittingConfig(strategy=ParagraphSplit())
        config = DataPreprocessingConfig(drop_target_column=True, splitting=splitting_config)
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world.\n\nTest data."],
            "other": [1]
        })
        
        result = processor.process_dataframe(df)
        
        assert SYSTEM_RAW_DATA_COLUMN not in result.columns
        assert "other" in result.columns
        assert SYSTEM_CHUNK_COLUMN in result.columns
    
    def test_process_dataframe_keep_target_column(self):
        """Test processing DataFrame with drop_target_column=False."""
        config = DataPreprocessingConfig(drop_target_column=False)
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world", "Test data"],
            "other": [1, 2]
        })
        
        result = processor.process_dataframe(df)
        
        # When no splitting, target column is renamed to chunk column
        assert SYSTEM_RAW_DATA_COLUMN not in result.columns  # Renamed to chunk column
        assert "other" in result.columns
        assert SYSTEM_CHUNK_COLUMN in result.columns
    
    def test_process_dataframe_with_fixed_window_split(self):
        """Test processing DataFrame with FixedWindowSplit."""
        splitting_config = SplittingConfig(strategy=FixedWindowSplit(window=2, stride=1))
        config = DataPreprocessingConfig(splitting=splitting_config)
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Sentence one. Sentence two. Sentence three. Sentence four."],
            "other": [1]
        })
        
        result = processor.process_dataframe(df)
        
        # Should create overlapping windows
        assert len(result) > 1
        assert SYSTEM_CHUNK_COLUMN in result.columns
        assert SYSTEM_CHUNK_ID_COLUMN in result.columns
    
    def test_process_dataframe_with_regex_split(self):
        """Test processing DataFrame with RegexSplit."""
        splitting_config = SplittingConfig(strategy=RegexSplit(r"\.\s+"))
        config = DataPreprocessingConfig(splitting=splitting_config)
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world. Test data. Another sentence."],
            "other": [1]
        })
        
        result = processor.process_dataframe(df)
        
        assert len(result) == 3  # Split on periods
        assert SYSTEM_CHUNK_COLUMN in result.columns
        assert "Hello world" in result[SYSTEM_CHUNK_COLUMN].values
        assert "Test data" in result[SYSTEM_CHUNK_COLUMN].values
        assert "Another sentence." in result[SYSTEM_CHUNK_COLUMN].values  # Note the period
    
    def test_process_dataframe_with_fuzzy_scorer(self):
        """Test processing DataFrame with FuzzyScorer."""
        scoring_config = ScoringConfig(scorer=FuzzyScorer(["test", "data"]))
        config = DataPreprocessingConfig(scoring=scoring_config)
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world", "Test data", "Another text"],
            "other": [1, 2, 3]
        })
        
        result = processor.process_dataframe(df)
        
        assert len(result) == 3
        assert SYSTEM_SCORE_COLUMN in result.columns
        scores = result[SYSTEM_SCORE_COLUMN].tolist()
        assert all(isinstance(score, (int, float)) for score in scores)
        assert all(0 <= score <= 1 for score in scores)  # Fuzzy scores are 0-1
    
    def test_process_dataframe_empty_dataframe(self):
        """Test processing empty DataFrame."""
        config = DataPreprocessingConfig()
        processor = DataProcessor(config)
        
        df = pd.DataFrame(columns=[SYSTEM_RAW_DATA_COLUMN, "other"])
        
        result = processor.process_dataframe(df)
        
        assert len(result) == 0
        assert SYSTEM_CHUNK_COLUMN in result.columns
        assert SYSTEM_CHUNK_ID_COLUMN in result.columns
        # SYSTEM_RECORD_ID_COLUMN is added in load_data, not process_dataframe
    
    def test_process_dataframe_single_record(self):
        """Test processing DataFrame with single record."""
        config = DataPreprocessingConfig()
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Single record"],
            "other": [1]
        })
        
        result = processor.process_dataframe(df)
        
        assert len(result) == 1
        assert result[SYSTEM_CHUNK_COLUMN].iloc[0] == "Single record"
        assert result[SYSTEM_CHUNK_ID_COLUMN].iloc[0] == 0
        # SYSTEM_RECORD_ID_COLUMN is added in load_data, not process_dataframe
    
    def test_process_dataframe_preserves_metadata(self):
        """Test that processing preserves metadata columns."""
        config = DataPreprocessingConfig()
        processor = DataProcessor(config)
        
        df = pd.DataFrame({
            SYSTEM_RAW_DATA_COLUMN: ["Hello world", "Test data"],
            "metadata1": ["meta1", "meta2"],
            "metadata2": [100, 200]
        })
        
        result = processor.process_dataframe(df)
        
        assert "metadata1" in result.columns
        assert "metadata2" in result.columns
        assert result["metadata1"].tolist() == ["meta1", "meta2"]
        assert result["metadata2"].tolist() == [100, 200] 