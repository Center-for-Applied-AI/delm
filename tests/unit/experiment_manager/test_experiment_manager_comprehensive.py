"""
Comprehensive unit tests for DELM experiment managers.
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
from delm.exceptions import ExperimentManagementError
from delm.constants import (
    BATCH_FILE_PREFIX, BATCH_FILE_SUFFIX, BATCH_FILE_DIGITS,
    STATE_FILE_NAME, CONSOLIDATED_RESULT_PREFIX, CONSOLIDATED_RESULT_SUFFIX,
    PREPROCESSED_DATA_PREFIX, PREPROCESSED_DATA_SUFFIX
)


class TestInMemoryExperimentManagerComprehensive:
    """Comprehensive tests for InMemoryExperimentManager."""
    
    def setup_method(self):
        """Set up test manager."""
        self.manager = InMemoryExperimentManager("test_experiment")
    
    def test_initialization_with_invalid_kwargs(self):
        """Test initialization with unsupported kwargs."""
        with pytest.raises(ValueError, match="overwrite_experiment is not supported"):
            InMemoryExperimentManager("test", overwrite_experiment=True)
        
        with pytest.raises(ValueError, match="auto_checkpoint_and_resume_experiment is not supported"):
            InMemoryExperimentManager("test", auto_checkpoint_and_resume_experiment=True)
    
    def test_save_preprocessed_data(self):
        """Test saving preprocessed data."""
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        
        result_path = self.manager.save_preprocessed_data(test_df)
        
        assert result_path == "in-memory"
        assert self.manager._preprocessed_data is not None
        assert len(self.manager._preprocessed_data) == 3
        assert "test" in self.manager._preprocessed_data.columns
    
    def test_load_preprocessed_data_with_file_path(self):
        """Test loading preprocessed data with file path (should fail)."""
        with pytest.raises(NotImplementedError, match="Loading preprocessed data from a file path is not supported"):
            self.manager.load_preprocessed_data(Path("some/path"))
    
    def test_load_preprocessed_data_without_data(self):
        """Test loading preprocessed data when none exists."""
        with pytest.raises(ValueError, match="No preprocessed data available in memory"):
            self.manager.load_preprocessed_data()
    
    def test_save_batch_checkpoint(self):
        """Test saving batch checkpoint."""
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        
        result_path = self.manager.save_batch_checkpoint(test_df, batch_id=1)
        
        assert result_path == "in-memory-batch-1"
        assert 1 in self.manager._batches
        assert len(self.manager._batches[1]) == 3
    
    def test_load_batch_checkpoint_by_id_existing(self):
        """Test loading existing batch checkpoint by ID."""
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        self.manager._batches[1] = test_df
        
        result = self.manager.load_batch_checkpoint_by_id(1)
        
        assert len(result) == 3
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3]
    
    def test_load_batch_checkpoint_by_id_nonexistent(self):
        """Test loading non-existent batch checkpoint by ID."""
        with pytest.raises(ValueError, match="No batch checkpoint with id 1 in memory"):
            self.manager.load_batch_checkpoint_by_id(1)
    
    def test_load_batch_checkpoint_with_valid_path(self):
        """Test loading batch checkpoint with valid path string."""
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        self.manager._batches[1] = test_df
        
        result = self.manager.load_batch_checkpoint("in-memory-batch-1")
        
        assert len(result) == 3
        assert "test" in result.columns
    
    def test_load_batch_checkpoint_with_invalid_path(self):
        """Test loading batch checkpoint with invalid path string."""
        with pytest.raises(ValueError, match="No batch checkpoint with id 999 in memory"):
            self.manager.load_batch_checkpoint("in-memory-batch-999")
    
    def test_load_batch_checkpoint_with_malformed_path(self):
        """Test loading batch checkpoint with malformed path string."""
        with pytest.raises(ValueError, match="Invalid batch path format"):
            self.manager.load_batch_checkpoint("invalid-path")
    
    def test_list_batch_checkpoints_empty(self):
        """Test listing batch checkpoints when none exist."""
        result = self.manager.list_batch_checkpoints()
        assert result == []
    
    def test_list_batch_checkpoints_with_data(self):
        """Test listing batch checkpoints with data."""
        test_df = pd.DataFrame({"test": [1]})
        self.manager._batches[1] = test_df
        self.manager._batches[3] = test_df
        self.manager._batches[2] = test_df
        
        result = self.manager.list_batch_checkpoints()
        
        assert result == [1, 2, 3]  # Should be sorted
    
    def test_consolidate_batches_empty(self):
        """Test consolidating batches when none exist."""
        with pytest.raises(ValueError, match="No batch checkpoints in memory to consolidate"):
            self.manager.consolidate_batches()
    
    def test_consolidate_batches_with_data(self):
        """Test consolidating batches with data."""
        batch1_df = pd.DataFrame({"test": [1, 2]})
        batch2_df = pd.DataFrame({"test": [3, 4]})
        self.manager._batches[1] = batch1_df
        self.manager._batches[2] = batch2_df
        
        result = self.manager.consolidate_batches()
        
        assert len(result) == 4
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3, 4]
    
    def test_consolidate_batches_preserves_order(self):
        """Test that consolidate_batches preserves batch order."""
        batch1_df = pd.DataFrame({"test": [1, 2]})
        batch3_df = pd.DataFrame({"test": [5, 6]})
        batch2_df = pd.DataFrame({"test": [3, 4]})
        self.manager._batches[1] = batch1_df
        self.manager._batches[3] = batch3_df
        self.manager._batches[2] = batch2_df
        
        result = self.manager.consolidate_batches()
        
        assert len(result) == 6
        assert result["test"].tolist() == [1, 2, 3, 4, 5, 6]  # Should be in batch ID order
    
    def test_cleanup_batch_checkpoints(self):
        """Test cleaning up batch checkpoints."""
        test_df = pd.DataFrame({"test": [1]})
        self.manager._batches[1] = test_df
        self.manager._batches[2] = test_df
        
        assert len(self.manager._batches) == 2
        
        self.manager.cleanup_batch_checkpoints()
        
        assert len(self.manager._batches) == 0
    
    def test_get_all_existing_batch_ids(self):
        """Test getting all existing batch IDs."""
        test_df = pd.DataFrame({"test": [1]})
        self.manager._batches[1] = test_df
        self.manager._batches[3] = test_df
        self.manager._batches[2] = test_df
        
        result = self.manager.get_all_existing_batch_ids()
        
        assert result == {1, 2, 3}
    
    def test_get_batch_checkpoint_path(self):
        """Test getting batch checkpoint path."""
        result = self.manager.get_batch_checkpoint_path(1)
        assert result == "in-memory-batch-1"
    
    def test_delete_batch_checkpoint_existing(self):
        """Test deleting existing batch checkpoint."""
        test_df = pd.DataFrame({"test": [1]})
        self.manager._batches[1] = test_df
        
        assert 1 in self.manager._batches
        
        result = self.manager.delete_batch_checkpoint(1)
        
        assert result is True
        assert 1 not in self.manager._batches
    
    def test_delete_batch_checkpoint_nonexistent(self):
        """Test deleting non-existent batch checkpoint."""
        result = self.manager.delete_batch_checkpoint(1)
        assert result is False
    
    def test_save_state(self):
        """Test saving state."""
        cost_tracker = Mock()
        cost_tracker.provider = "openai"
        cost_tracker.model = "gpt-4"
        
        self.manager.save_state(cost_tracker)
        
        assert self.manager._state is cost_tracker
    
    def test_load_state_with_data(self):
        """Test loading state when data exists."""
        cost_tracker = Mock()
        cost_tracker.provider = "openai"
        cost_tracker.model = "gpt-4"
        self.manager._state = cost_tracker
        
        result = self.manager.load_state()
        
        assert result is cost_tracker
    
    def test_load_state_without_data(self):
        """Test loading state when no data exists."""
        result = self.manager.load_state()
        assert result is None
    
    def test_save_extracted_data(self):
        """Test saving extracted data."""
        test_df = pd.DataFrame({"extracted": [1, 2, 3]})
        
        result_path = self.manager.save_extracted_data(test_df)
        
        assert result_path == "in-memory"
        assert self.manager._extracted_data is not None
        assert len(self.manager._extracted_data) == 3
        assert "extracted" in self.manager._extracted_data.columns
    
    def test_get_results_with_data(self):
        """Test getting results when data exists."""
        test_df = pd.DataFrame({"extracted": [1, 2, 3]})
        self.manager._extracted_data = test_df
        
        result = self.manager.get_results()
        
        assert len(result) == 3
        assert "extracted" in result.columns
        assert result["extracted"].tolist() == [1, 2, 3]
    
    def test_get_results_without_data(self):
        """Test getting results when no data exists."""
        with pytest.raises(ValueError, match="No extracted data available in memory"):
            self.manager.get_results()
    
    def test_initialize_experiment(self):
        """Test initializing experiment."""
        config = Mock()
        config.to_serialized_config_dict.return_value = {"test": "config"}
        config.to_serialized_schema_spec_dict.return_value = {"test": "schema"}
        
        self.manager.initialize_experiment(config)
        
        assert self.manager._config_dict == {"test": "config"}
        assert self.manager._schema_dict == {"test": "schema"}


class TestDiskExperimentManagerComprehensive:
    """Comprehensive tests for DiskExperimentManager."""
    
    def setup_method(self):
        """Set up test manager."""
        self.tmp_path = Path(tempfile.mkdtemp())
        self.experiment_dir = self.tmp_path / "experiments"
        self.experiment_dir.mkdir()
        self.manager = DiskExperimentManager(
            experiment_name="test_experiment",
            experiment_directory=self.experiment_dir,
            overwrite_experiment=False,
            auto_checkpoint_and_resume_experiment=True
        )
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.tmp_path, ignore_errors=True)
    
    def test_initialization(self):
        """Test DiskExperimentManager initialization."""
        assert self.manager.experiment_name == "test_experiment"
        assert self.manager.experiment_directory == self.experiment_dir
        assert self.manager.overwrite_experiment is False
        assert self.manager.auto_checkpoint_and_resume_experiment is True
        assert self.manager.experiment_dir == self.experiment_dir / "test_experiment"
    
    def test_properties_create_directories(self):
        """Test that properties create directories when accessed."""
        # Directories should not exist initially
        assert not self.manager.experiment_dir.exists()
        
        # Accessing properties should create directories
        config_dir = self.manager.config_dir
        data_dir = self.manager.data_dir
        cache_dir = self.manager.cache_dir
        
        assert config_dir.exists()
        assert data_dir.exists()
        assert cache_dir.exists()
    
    def test_is_experiment_completed_false(self):
        """Test is_experiment_completed when experiment is not completed."""
        assert self.manager.is_experiment_completed() is False
    
    def test_is_experiment_completed_true(self):
        """Test is_experiment_completed when experiment is completed."""
        # Create the result file
        result_file = self.manager.data_dir / f"{CONSOLIDATED_RESULT_PREFIX}test_experiment{CONSOLIDATED_RESULT_SUFFIX}"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"test": [1]}).to_feather(result_file)
        
        assert self.manager.is_experiment_completed() is True
    
    def test_get_results_file_not_found(self):
        """Test get_results when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Consolidated result file does not exist"):
            self.manager.get_results()
    
    def test_get_results_file_exists(self):
        """Test get_results when file exists."""
        # Create the result file
        result_file = self.manager.data_dir / f"{CONSOLIDATED_RESULT_PREFIX}test_experiment{CONSOLIDATED_RESULT_SUFFIX}"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        test_df.to_feather(result_file)
        
        result = self.manager.get_results()
        
        assert len(result) == 3
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3]
    
    def test_initialize_experiment_new_directory(self):
        """Test initializing experiment in new directory."""
        config = Mock()
        config.to_serialized_config_dict.return_value = {"test": "config"}
        config.to_serialized_schema_spec_dict.return_value = {"test": "schema"}
        
        self.manager.initialize_experiment(config)
        
        # Check that directories were created
        assert self.manager.config_dir.exists()
        assert self.manager.data_dir.exists()
        assert self.manager.cache_dir.exists()
        
        # Check that config files were created
        config_file = self.manager.config_dir / "config_test_experiment.yaml"
        schema_file = self.manager.config_dir / "schema_spec_test_experiment.yaml"
        
        assert config_file.exists()
        assert schema_file.exists()
    
    def test_initialize_experiment_overwrite_existing(self):
        """Test initializing experiment with overwrite=True."""
        # Create existing experiment directory
        self.manager.experiment_dir.mkdir(parents=True, exist_ok=True)
        existing_file = self.manager.experiment_dir / "existing_file.txt"
        existing_file.write_text("existing content")
        
        assert existing_file.exists()
        
        # Initialize with overwrite
        self.manager.overwrite_experiment = True
        config = Mock()
        config.to_serialized_config_dict.return_value = {"test": "config"}
        config.to_serialized_schema_spec_dict.return_value = {"test": "schema"}
        
        self.manager.initialize_experiment(config)
        
        # Existing file should be gone
        assert not existing_file.exists()
        # New directories should exist
        assert self.manager.config_dir.exists()
    
    def test_initialize_experiment_existing_completed(self):
        """Test initializing experiment when existing experiment is completed."""
        # Create completed experiment
        self.manager.experiment_dir.mkdir(parents=True, exist_ok=True)
        result_file = self.manager.data_dir / f"{CONSOLIDATED_RESULT_PREFIX}test_experiment{CONSOLIDATED_RESULT_SUFFIX}"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"test": [1]}).to_feather(result_file)
        
        config = Mock()
        config.to_serialized_config_dict.return_value = {"test": "config"}
        config.to_serialized_schema_spec_dict.return_value = {"test": "schema"}
        
        with pytest.raises(ExperimentManagementError, match="Experiment exists and is already completed"):
            self.manager.initialize_experiment(config)
    
    def test_save_preprocessed_data(self):
        """Test saving preprocessed data."""
        # Initialize experiment first
        config = Mock()
        config.to_serialized_config_dict.return_value = {"test": "config"}
        config.to_serialized_schema_spec_dict.return_value = {"test": "schema"}
        self.manager.initialize_experiment(config)
        
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        
        result_path = self.manager.save_preprocessed_data(test_df)
        
        assert result_path.exists()
        assert result_path.name.endswith(PREPROCESSED_DATA_SUFFIX)
        
        # Verify data was saved correctly
        loaded_df = pd.read_feather(result_path)
        assert len(loaded_df) == 3
        assert "test" in loaded_df.columns
    
    def test_load_preprocessed_data_file_not_found(self):
        """Test loading preprocessed data when file doesn't exist."""
        with pytest.raises(ValueError, match="Experiment not initialized"):
            self.manager.load_preprocessed_data()
    
    def test_load_preprocessed_data_wrong_extension(self):
        """Test loading preprocessed data with wrong file extension."""
        wrong_file = self.manager.data_dir / "wrong_extension.txt"
        wrong_file.parent.mkdir(parents=True, exist_ok=True)
        wrong_file.write_text("not a feather file")
        
        with pytest.raises(ValueError, match="Preprocessed data file must be a feather file"):
            self.manager.load_preprocessed_data(wrong_file)
    
    def test_save_batch_checkpoint(self):
        """Test saving batch checkpoint."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        
        result_path = self.manager.save_batch_checkpoint(test_df, batch_id=1)
        
        assert result_path.exists()
        expected_name = f"{BATCH_FILE_PREFIX}{1:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        assert result_path.name == expected_name
        
        # Verify data was saved correctly
        loaded_df = pd.read_feather(result_path)
        assert len(loaded_df) == 3
        assert "test" in loaded_df.columns
    
    def test_list_batch_checkpoints_empty(self):
        """Test listing batch checkpoints when none exist."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        result = self.manager.list_batch_checkpoints()
        
        assert result == []
    
    def test_list_batch_checkpoints_with_files(self):
        """Test listing batch checkpoints with files."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some batch files
        batch1 = self.manager.cache_dir / f"{BATCH_FILE_PREFIX}001{BATCH_FILE_SUFFIX}"
        batch2 = self.manager.cache_dir / f"{BATCH_FILE_PREFIX}002{BATCH_FILE_SUFFIX}"
        batch3 = self.manager.cache_dir / f"{BATCH_FILE_PREFIX}003{BATCH_FILE_SUFFIX}"
        
        pd.DataFrame({"test": [1]}).to_feather(batch1)
        pd.DataFrame({"test": [2]}).to_feather(batch2)
        pd.DataFrame({"test": [3]}).to_feather(batch3)
        
        result = self.manager.list_batch_checkpoints()
        
        assert len(result) == 3
        assert all(p.exists() for p in result)
        assert result[0].name == f"{BATCH_FILE_PREFIX}001{BATCH_FILE_SUFFIX}"
        assert result[1].name == f"{BATCH_FILE_PREFIX}002{BATCH_FILE_SUFFIX}"
        assert result[2].name == f"{BATCH_FILE_PREFIX}003{BATCH_FILE_SUFFIX}"
    
    def test_load_batch_checkpoint_file_not_found(self):
        """Test loading batch checkpoint when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Batch checkpoint file does not exist"):
            self.manager.load_batch_checkpoint(Path("nonexistent.feather"))
    
    def test_load_batch_checkpoint_wrong_extension(self):
        """Test loading batch checkpoint with wrong file extension."""
        wrong_file = self.manager.cache_dir / "wrong_extension.txt"
        wrong_file.parent.mkdir(parents=True, exist_ok=True)
        wrong_file.write_text("not a feather file")
        
        with pytest.raises(ValueError, match="Batch checkpoint file must be a feather file"):
            self.manager.load_batch_checkpoint(wrong_file)
    
    def test_load_batch_checkpoint_by_id(self):
        """Test loading batch checkpoint by ID."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        test_df = pd.DataFrame({"test": [1, 2, 3]})
        batch_path = self.manager.save_batch_checkpoint(test_df, batch_id=1)
        
        result = self.manager.load_batch_checkpoint_by_id(1)
        
        assert len(result) == 3
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3]
    
    def test_consolidate_batches_no_files(self):
        """Test consolidating batches when no files exist."""
        with pytest.raises(FileNotFoundError, match="No batch files found for consolidation"):
            self.manager.consolidate_batches()
    
    def test_consolidate_batches_with_files(self):
        """Test consolidating batches with files."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        batch1_df = pd.DataFrame({"test": [1, 2]})
        batch2_df = pd.DataFrame({"test": [3, 4]})
        
        self.manager.save_batch_checkpoint(batch1_df, batch_id=1)
        self.manager.save_batch_checkpoint(batch2_df, batch_id=2)
        
        result = self.manager.consolidate_batches()
        
        assert len(result) == 4
        assert "test" in result.columns
        assert result["test"].tolist() == [1, 2, 3, 4]
    
    def test_cleanup_batch_checkpoints(self):
        """Test cleaning up batch checkpoints."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        batch1_df = pd.DataFrame({"test": [1]})
        batch2_df = pd.DataFrame({"test": [2]})
        
        batch1_path = self.manager.save_batch_checkpoint(batch1_df, batch_id=1)
        batch2_path = self.manager.save_batch_checkpoint(batch2_df, batch_id=2)
        
        assert batch1_path.exists()
        assert batch2_path.exists()
        
        self.manager.cleanup_batch_checkpoints()
        
        assert not batch1_path.exists()
        assert not batch2_path.exists()
    
    def test_get_all_existing_batch_ids(self):
        """Test getting all existing batch IDs."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        batch1_df = pd.DataFrame({"test": [1]})
        batch2_df = pd.DataFrame({"test": [2]})
        batch3_df = pd.DataFrame({"test": [3]})
        
        self.manager.save_batch_checkpoint(batch1_df, batch_id=1)
        self.manager.save_batch_checkpoint(batch2_df, batch_id=2)
        self.manager.save_batch_checkpoint(batch3_df, batch_id=3)
        
        result = self.manager.get_all_existing_batch_ids()
        
        assert result == {1, 2, 3}
    
    def test_get_batch_checkpoint_path(self):
        """Test getting batch checkpoint path."""
        result = self.manager.get_batch_checkpoint_path(1)
        
        expected_name = f"{BATCH_FILE_PREFIX}{1:0{BATCH_FILE_DIGITS}d}{BATCH_FILE_SUFFIX}"
        assert result == self.manager.cache_dir / expected_name
    
    def test_delete_batch_checkpoint_existing(self):
        """Test deleting existing batch checkpoint."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        batch_df = pd.DataFrame({"test": [1]})
        batch_path = self.manager.save_batch_checkpoint(batch_df, batch_id=1)
        
        assert batch_path.exists()
        
        result = self.manager.delete_batch_checkpoint(1)
        
        assert result is True
        assert not batch_path.exists()
    
    def test_delete_batch_checkpoint_nonexistent(self):
        """Test deleting non-existent batch checkpoint."""
        result = self.manager.delete_batch_checkpoint(1)
        assert result is False
    
    def test_save_state(self):
        """Test saving state."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        result_path = self.manager.save_state(cost_tracker)
        
        assert result_path.exists()
        assert result_path.name == STATE_FILE_NAME
        
        # Verify state was saved correctly
        with open(result_path, "r") as f:
            state = json.load(f)
        
        assert "cost_tracker" in state
        assert state["cost_tracker"]["provider"] == "openai"
    
    def test_load_state_file_not_found(self):
        """Test loading state when file doesn't exist."""
        result = self.manager.load_state()
        assert result is None
    
    def test_load_state_file_exists(self):
        """Test loading state when file exists."""
        self.manager.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a real cost tracker for testing
        cost_tracker = CostTracker("openai", "gpt-4o-mini")
        cost_tracker.input_tokens = 100
        cost_tracker.output_tokens = 50
        
        self.manager.save_state(cost_tracker)
        
        result = self.manager.load_state()
        
        assert result is not None
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
    
    def test_save_extracted_data(self):
        """Test saving extracted data."""
        self.manager.data_dir.mkdir(parents=True, exist_ok=True)
        
        test_df = pd.DataFrame({"extracted": [1, 2, 3]})
        
        result_path = self.manager.save_extracted_data(test_df)
        
        assert result_path.exists()
        assert result_path.name.startswith(CONSOLIDATED_RESULT_PREFIX)
        assert result_path.name.endswith(CONSOLIDATED_RESULT_SUFFIX)
        
        # Verify data was saved correctly
        loaded_df = pd.read_feather(result_path)
        assert len(loaded_df) == 3
        assert "extracted" in loaded_df.columns


class TestExperimentManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_inmemory_batch_path_parsing_edge_cases(self):
        """Test edge cases in batch path parsing for InMemoryExperimentManager."""
        manager = InMemoryExperimentManager("test")
        
        # Test with malformed paths
        with pytest.raises(ValueError):
            manager.load_batch_checkpoint("invalid")
        
        with pytest.raises(ValueError):
            manager.load_batch_checkpoint("in-memory-batch-")
        
        with pytest.raises(ValueError):
            manager.load_batch_checkpoint("in-memory-batch-abc")
    
    def test_disk_batch_id_parsing_edge_cases(self):
        """Test edge cases in batch ID parsing for DiskExperimentManager."""
        tmp_path = Path(tempfile.mkdtemp())
        try:
            experiment_dir = tmp_path / "experiments"
            experiment_dir.mkdir()
            manager = DiskExperimentManager("test", experiment_dir)
            manager.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create files with malformed names
            malformed1 = manager.cache_dir / "batch_abc.feather"
            malformed2 = manager.cache_dir / "batch_.feather"
            malformed3 = manager.cache_dir / "not_batch_1.feather"
            
            pd.DataFrame({"test": [1]}).to_feather(malformed1)
            pd.DataFrame({"test": [1]}).to_feather(malformed2)
            pd.DataFrame({"test": [1]}).to_feather(malformed3)
            
            # Should not crash and should return empty set
            result = manager.get_all_existing_batch_ids()
            assert result == set()
            
        finally:
            import shutil
            shutil.rmtree(tmp_path, ignore_errors=True)
    
    def test_large_dataframes(self):
        """Test handling of large DataFrames."""
        manager = InMemoryExperimentManager("test")
        
        # Create a large DataFrame
        large_df = pd.DataFrame({
            "col1": range(10000),
            "col2": [f"string_{i}" for i in range(10000)]
        })
        
        # Should not crash
        manager.save_preprocessed_data(large_df)
        result = manager.load_preprocessed_data()
        
        assert len(result) == 10000
        assert "col1" in result.columns
        assert "col2" in result.columns
    
    def test_empty_dataframes(self):
        """Test handling of empty DataFrames."""
        manager = InMemoryExperimentManager("test")
        
        empty_df = pd.DataFrame()
        
        # Should not crash
        manager.save_preprocessed_data(empty_df)
        result = manager.load_preprocessed_data()
        
        assert len(result) == 0 