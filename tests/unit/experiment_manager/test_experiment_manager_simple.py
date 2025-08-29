"""
Simplified unit tests for DELM experiment managers.
"""

import pytest
import pandas as pd
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from delm.core.experiment_manager import DiskExperimentManager, InMemoryExperimentManager
from delm.config import DELMConfig, LLMExtractionConfig, DataPreprocessingConfig, SchemaConfig, SemanticCacheConfig
from delm.utils.cost_tracker import CostTracker
from delm.constants import (
    BATCH_FILE_PREFIX, BATCH_FILE_SUFFIX, BATCH_FILE_DIGITS,
    STATE_FILE_NAME, CONSOLIDATED_RESULT_PREFIX, CONSOLIDATED_RESULT_SUFFIX,
    PREPROCESSED_DATA_PREFIX, PREPROCESSED_DATA_SUFFIX
)


class TestInMemoryExperimentManager:
    """Test the InMemoryExperimentManager class."""
    
    def test_initialization(self):
        """Test InMemoryExperimentManager initialization."""
        experiment_name = "test_experiment"
        
        manager = InMemoryExperimentManager(experiment_name)
        
        assert manager.experiment_name == experiment_name
        assert manager._preprocessed_data is None
        assert manager._batches == {}
        assert manager._extracted_data is None
        assert manager._state is None
    
    def test_save_and_load_preprocessed_data(self):
        """Test save_preprocessed_data and load_preprocessed_data methods."""
        experiment_name = "test_experiment"
        manager = InMemoryExperimentManager(experiment_name)
        
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        
        # Save data
        result_path = manager.save_preprocessed_data(test_df)
        assert result_path == "in-memory"
        assert manager._preprocessed_data is not None
        assert len(manager._preprocessed_data) == 3
        
        # Load data
        result = manager.load_preprocessed_data()
        assert len(result) == 3
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3]
    
    def test_save_and_load_batch_checkpoints(self):
        """Test batch checkpoint operations."""
        experiment_name = "test_experiment"
        manager = InMemoryExperimentManager(experiment_name)
        
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        
        # Save batch checkpoint
        result_path = manager.save_batch_checkpoint(test_df, batch_id=1)
        assert result_path == "in-memory-batch-1"
        assert 1 in manager._batches
        assert len(manager._batches[1]) == 3
        
        # List batch checkpoints
        batch_list = manager.list_batch_checkpoints()
        assert 1 in batch_list
        
        # Load batch checkpoint by ID
        result = manager.load_batch_checkpoint_by_id(1)
        assert len(result) == 3
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3]
        
        # Load batch checkpoint by path
        result = manager.load_batch_checkpoint("in-memory-batch-1")
        assert len(result) == 3
        assert "test" in result.columns
    
    def test_consolidate_batches(self):
        """Test consolidate_batches method."""
        experiment_name = "test_experiment"
        manager = InMemoryExperimentManager(experiment_name)
        
        # Add batch checkpoints
        batch1_df = pd.DataFrame({"test": [1, 2]})
        batch2_df = pd.DataFrame({"test": [3, 4]})
        manager._batches[1] = batch1_df
        manager._batches[2] = batch2_df
        
        result = manager.consolidate_batches()
        
        assert len(result) == 4
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3, 4]
    
    def test_get_results(self):
        """Test get_results method."""
        experiment_name = "test_experiment"
        manager = InMemoryExperimentManager(experiment_name)
        
        # Test with no data
        with pytest.raises(ValueError, match="No extracted data available in memory"):
            manager.get_results()
        
        # Test with extracted data
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        manager._extracted_data = test_df
        
        result = manager.get_results()
        assert len(result) == 3
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3]
    
    def test_save_and_load_state(self):
        """Test state management."""
        experiment_name = "test_experiment"
        manager = InMemoryExperimentManager(experiment_name)
        
        # Create a mock cost tracker
        cost_tracker = Mock()
        cost_tracker.provider = "openai"
        cost_tracker.model = "gpt-4"
        
        # Save state
        manager.save_state(cost_tracker)
        assert manager._state is not None
        assert manager._state.provider == "openai"
        assert manager._state.model == "gpt-4"
        
        # Load state
        result = manager.load_state()
        assert result is not None
        assert result.provider == "openai"
        assert result.model == "gpt-4"
    
    def test_save_extracted_data(self):
        """Test save_extracted_data method."""
        experiment_name = "test_experiment"
        manager = InMemoryExperimentManager(experiment_name)
        
        test_df = pd.DataFrame({"extracted": [1, 2, 3]})
        
        result_path = manager.save_extracted_data(test_df)
        
        assert result_path == "in-memory"
        assert manager._extracted_data is not None
        assert len(manager._extracted_data) == 3
        assert "extracted" in manager._extracted_data.columns


class TestDiskExperimentManager:
    """Test the DiskExperimentManager class."""
    
    def test_initialization(self, tmp_path):
        """Test DiskExperimentManager initialization."""
        experiment_name = "test_experiment"
        experiment_dir = tmp_path / "experiments"
        experiment_dir.mkdir()
        
        manager = DiskExperimentManager(
            experiment_name=experiment_name,
            experiment_directory=experiment_dir,
            overwrite_experiment=False,
            auto_checkpoint_and_resume_experiment=True
        )
        
        assert manager.experiment_name == experiment_name
        assert manager.experiment_directory == experiment_dir
        assert manager.overwrite_experiment is False
        assert manager.auto_checkpoint_and_resume_experiment is True
        assert manager.experiment_dir == experiment_dir / experiment_name
    
    def test_save_and_load_preprocessed_data(self, tmp_path):
        """Test save_preprocessed_data and load_preprocessed_data methods."""
        experiment_name = "test_experiment"
        experiment_dir = tmp_path / "experiments"
        experiment_dir.mkdir()
        
        manager = DiskExperimentManager(experiment_name, experiment_dir)
        
        # Create a temporary schema file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            schema_content = """
type: simple
variables:
  - name: test_field
    description: A test field
    data_type: string
"""
            f.write(schema_content)
            schema_path = Path(f.name)
        
        try:
            config = DELMConfig(
                llm_extraction=LLMExtractionConfig(),
                data_preprocessing=DataPreprocessingConfig(),
                schema=SchemaConfig(spec_path=schema_path),
                semantic_cache=SemanticCacheConfig()
            )
            
            # Initialize experiment to set up preprocessed_data_path
            manager.initialize_experiment(config)
            
            test_df = pd.DataFrame({"test": [1, 2, 3]})
            
            # Save data
            result_path = manager.save_preprocessed_data(test_df)
            assert result_path.exists()
            assert result_path.name.endswith(PREPROCESSED_DATA_SUFFIX)
            
            # Load data
            result = manager.load_preprocessed_data()
            assert len(result) == 3
            assert "test" in result.columns
            assert result["test"].tolist() == [1, 2, 3]
        finally:
            # Clean up temporary file
            schema_path.unlink()
    
    def test_save_and_load_batch_checkpoints(self, tmp_path):
        """Test batch checkpoint operations."""
        experiment_name = "test_experiment"
        experiment_dir = tmp_path / "experiments"
        experiment_dir.mkdir()
        
        manager = DiskExperimentManager(experiment_name, experiment_dir)
        manager.experiment_dir.mkdir()
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        
        # Save batch checkpoint
        result_path = manager.save_batch_checkpoint(test_df, batch_id=1)
        assert result_path.exists()
        expected_name = f"{BATCH_FILE_PREFIX}{1:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        assert result_path.name == expected_name
        
        # List batch checkpoints
        batch_list = manager.list_batch_checkpoints()
        assert len(batch_list) == 1
        assert result_path in batch_list
        
        # Load batch checkpoint by ID
        result = manager.load_batch_checkpoint_by_id(1)
        assert len(result) == 3
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3]
    
    def test_consolidate_batches(self, tmp_path):
        """Test consolidate_batches method."""
        experiment_name = "test_experiment"
        experiment_dir = tmp_path / "experiments"
        experiment_dir.mkdir()
        
        manager = DiskExperimentManager(experiment_name, experiment_dir)
        manager.experiment_dir.mkdir()
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create batch files
        batch1_df = pd.DataFrame({"test": [1, 2]})
        batch2_df = pd.DataFrame({"test": [3, 4]})
        
        batch1_path = manager.save_batch_checkpoint(batch1_df, batch_id=1)
        batch2_path = manager.save_batch_checkpoint(batch2_df, batch_id=2)
        
        result = manager.consolidate_batches()
        
        assert len(result) == 4
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3, 4]
    
    def test_save_and_load_state(self, tmp_path):
        """Test state management."""
        experiment_name = "test_experiment"
        experiment_dir = tmp_path / "experiments"
        experiment_dir.mkdir()
        
        manager = DiskExperimentManager(experiment_name, experiment_dir)
        manager.experiment_dir.mkdir()
        manager.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock cost tracker
        cost_tracker = Mock()
        cost_tracker.to_dict.return_value = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "input_tokens": 100,
            "output_tokens": 50,
            "model_input_cost_per_1M_tokens": 0.15,
            "model_output_cost_per_1M_tokens": 0.60,
            "max_budget": None,
            "count_cache_hits_towards_cost": False
        }
        
        # Save state
        result_path = manager.save_state(cost_tracker)
        assert result_path.exists()
        assert result_path.name == STATE_FILE_NAME
        
        # Load state
        result = manager.load_state()
        assert result is not None
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
    
    def test_save_extracted_data(self, tmp_path):
        """Test save_extracted_data method."""
        experiment_name = "test_experiment"
        experiment_dir = tmp_path / "experiments"
        experiment_dir.mkdir()
        
        manager = DiskExperimentManager(experiment_name, experiment_dir)
        manager.experiment_dir.mkdir()
        manager.data_dir.mkdir(parents=True, exist_ok=True)
        
        test_df = pd.DataFrame({"extracted": [1, 2, 3]})
        
        result_path = manager.save_extracted_data(test_df)
        
        assert result_path.exists()
        assert result_path.name.startswith(CONSOLIDATED_RESULT_PREFIX)
        assert result_path.name.endswith(CONSOLIDATED_RESULT_SUFFIX)
        
        # Verify data was saved correctly
        loaded_df = pd.read_feather(result_path)
        assert len(loaded_df) == 3
        assert "extracted" in loaded_df.columns 