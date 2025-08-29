"""
Unit tests for cost_tracker module.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

from delm.utils.cost_tracker import CostTracker


class MockResponse(BaseModel):
    """Mock Pydantic response for testing."""
    field1: str
    field2: int
    field3: list[str]


class TestCostTrackerInitialization:
    """Test CostTracker initialization."""
    
    @patch('delm.utils.cost_tracker.get_model_token_price')
    def test_initialization_with_default_prices(self, mock_get_price):
        """Test initialization with default prices from database."""
        mock_get_price.return_value = (0.15, 0.60)
        
        tracker = CostTracker("openai", "gpt-4")
        
        assert tracker.provider == "openai"
        assert tracker.model == "gpt-4"
        assert tracker.model_input_cost_per_1M_tokens == 0.15
        assert tracker.model_output_cost_per_1M_tokens == 0.60
        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0
        assert tracker.max_budget is None
        assert tracker.count_cache_hits_towards_cost is False
        assert tracker.tokenizer is not None
    
    def test_initialization_with_custom_prices(self):
        """Test initialization with custom prices."""
        tracker = CostTracker(
            "openai", 
            "gpt-4",
            model_input_cost_per_1M_tokens=0.20,
            model_output_cost_per_1M_tokens=0.80
        )
        
        assert tracker.model_input_cost_per_1M_tokens == 0.20
        assert tracker.model_output_cost_per_1M_tokens == 0.80
    
    @patch('delm.utils.cost_tracker.get_model_token_price')
    def test_initialization_with_budget(self, mock_get_price):
        """Test initialization with budget limit."""
        mock_get_price.return_value = (0.15, 0.60)
        tracker = CostTracker("openai", "gpt-4o-mini", max_budget=10.0)
        
        assert tracker.max_budget == 10.0
    
    @patch('delm.utils.cost_tracker.get_model_token_price')
    def test_initialization_with_cache_hits_counting(self, mock_get_price):
        """Test initialization with cache hits counting enabled."""
        mock_get_price.return_value = (0.15, 0.60)
        tracker = CostTracker("openai", "gpt-4o-mini", count_cache_hits_towards_cost=True)
        
        assert tracker.count_cache_hits_towards_cost is True


class TestCostTrackerTokenCounting:
    """Test token counting functionality."""
    
    def setup_method(self):
        """Set up test tracker."""
        self.tracker = CostTracker(
            "openai", 
            "gpt-4",
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
    
    def test_count_tokens_simple_text(self):
        """Test counting tokens in simple text."""
        text = "Hello world"
        tokens = self.tracker.count_tokens(text)
        
        assert isinstance(tokens, int)
        assert tokens > 0
    
    def test_count_tokens_empty_text(self):
        """Test counting tokens in empty text."""
        text = ""
        tokens = self.tracker.count_tokens(text)
        
        assert tokens == 0
    
    def test_count_tokens_long_text(self):
        """Test counting tokens in long text."""
        text = "This is a much longer text that should have more tokens. " * 10
        tokens = self.tracker.count_tokens(text)
        
        assert tokens > 0
        assert tokens > self.tracker.count_tokens("Hello world")
    
    def test_count_tokens_batch(self):
        """Test counting tokens in a batch of texts."""
        texts = ["Hello", "World", "Test"]
        total_tokens = self.tracker.count_tokens_batch(texts)
        
        expected_tokens = sum(self.tracker.count_tokens(text) for text in texts)
        assert total_tokens == expected_tokens
    
    def test_count_tokens_batch_empty(self):
        """Test counting tokens in empty batch."""
        texts = []
        total_tokens = self.tracker.count_tokens_batch(texts)
        
        assert total_tokens == 0


class TestCostTrackerTokenTracking:
    """Test token tracking functionality."""
    
    def setup_method(self):
        """Set up test tracker."""
        self.tracker = CostTracker(
            "openai", 
            "gpt-4",
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
    
    def test_track_input_text(self):
        """Test tracking input text tokens."""
        initial_tokens = self.tracker.input_tokens
        text = "Hello world"
        
        self.tracker.track_input_text(text)
        
        assert self.tracker.input_tokens > initial_tokens
        assert self.tracker.input_tokens == initial_tokens + self.tracker.count_tokens(text)
    
    def test_track_output_text(self):
        """Test tracking output text tokens."""
        initial_tokens = self.tracker.output_tokens
        text = "Response text"
        
        self.tracker.track_output_text(text)
        
        assert self.tracker.output_tokens > initial_tokens
        assert self.tracker.output_tokens == initial_tokens + self.tracker.count_tokens(text)
    
    def test_track_output_pydantic(self):
        """Test tracking Pydantic output tokens."""
        initial_tokens = self.tracker.output_tokens
        response = MockResponse(field1="test", field2=123, field3=["a", "b"])
        
        self.tracker.track_output_pydantic(response)
        
        assert self.tracker.output_tokens > initial_tokens
        # Should count tokens in the JSON representation
        expected_json = json.dumps(response.model_dump(mode="json"))
        expected_tokens = self.tracker.count_tokens(expected_json)
        assert self.tracker.output_tokens == initial_tokens + expected_tokens
    
    def test_track_multiple_inputs(self):
        """Test tracking multiple input texts."""
        texts = ["Hello", "World", "Test"]
        initial_tokens = self.tracker.input_tokens
        
        for text in texts:
            self.tracker.track_input_text(text)
        
        expected_total = initial_tokens + sum(self.tracker.count_tokens(text) for text in texts)
        assert self.tracker.input_tokens == expected_total
    
    def test_track_multiple_outputs(self):
        """Test tracking multiple output texts."""
        texts = ["Response 1", "Response 2", "Response 3"]
        initial_tokens = self.tracker.output_tokens
        
        for text in texts:
            self.tracker.track_output_text(text)
        
        expected_total = initial_tokens + sum(self.tracker.count_tokens(text) for text in texts)
        assert self.tracker.output_tokens == expected_total


class TestCostTrackerCostCalculation:
    """Test cost calculation functionality."""
    
    def setup_method(self):
        """Set up test tracker."""
        self.tracker = CostTracker(
            "openai", 
            "gpt-4",
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
    
    def test_estimate_cost_zero_tokens(self):
        """Test cost estimation with zero tokens."""
        cost = self.tracker.estimate_cost(0, 0)
        
        assert cost == 0.0
    
    def test_estimate_cost_input_only(self):
        """Test cost estimation with input tokens only."""
        cost = self.tracker.estimate_cost(1000, 0)
        
        expected_cost = 1000 * 0.15 / 1_000_000
        assert cost == expected_cost
    
    def test_estimate_cost_output_only(self):
        """Test cost estimation with output tokens only."""
        cost = self.tracker.estimate_cost(0, 1000)
        
        expected_cost = 1000 * 0.60 / 1_000_000
        assert cost == expected_cost
    
    def test_estimate_cost_both_input_output(self):
        """Test cost estimation with both input and output tokens."""
        cost = self.tracker.estimate_cost(1000, 500)
        
        expected_input_cost = 1000 * 0.15 / 1_000_000
        expected_output_cost = 500 * 0.60 / 1_000_000
        expected_total = expected_input_cost + expected_output_cost
        assert cost == expected_total
    
    def test_get_current_cost_no_tokens(self):
        """Test current cost with no tokens tracked."""
        cost = self.tracker.get_current_cost()
        
        assert cost == 0.0
    
    def test_get_current_cost_with_tokens(self):
        """Test current cost with tokens tracked."""
        self.tracker.track_input_text("Hello world")
        self.tracker.track_output_text("Response")
        
        cost = self.tracker.get_current_cost()
        
        assert cost > 0.0
        assert cost == self.tracker.estimate_cost(
            self.tracker.input_tokens, 
            self.tracker.output_tokens
        )


class TestCostTrackerBudgetChecking:
    """Test budget checking functionality."""
    
    @patch('delm.utils.cost_tracker.get_model_token_price')
    def test_is_over_budget_no_budget(self, mock_get_price):
        """Test budget check when no budget is set."""
        mock_get_price.return_value = (0.15, 0.60)
        tracker = CostTracker("openai", "gpt-4o-mini")
        
        assert tracker.is_over_budget() is False
    
    def test_is_over_budget_under_budget(self):
        """Test budget check when under budget."""
        tracker = CostTracker(
            "openai", 
            "gpt-4",
            max_budget=10.0,
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
        
        # Add some tokens but stay under budget
        tracker.track_input_text("Hello")
        tracker.track_output_text("Response")
        
        assert tracker.is_over_budget() is False
    
    def test_is_over_budget_over_budget(self):
        """Test budget check when over budget."""
        tracker = CostTracker(
            "openai", 
            "gpt-4",
            max_budget=0.0001,  # Very small budget
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
        
        # Add enough tokens to exceed budget
        long_text = "This is a very long text that should generate enough tokens to exceed the budget. " * 100
        tracker.track_input_text(long_text)
        tracker.track_output_text(long_text)
        
        assert tracker.is_over_budget() is True


class TestCostTrackerSummary:
    """Test cost summary functionality."""
    
    def setup_method(self):
        """Set up test tracker."""
        self.tracker = CostTracker(
            "openai", 
            "gpt-4",
            max_budget=10.0,
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
        
        # Add some tokens
        self.tracker.track_input_text("Hello world")
        self.tracker.track_output_text("Response")
    
    def test_get_cost_summary_dict(self):
        """Test getting cost summary as dictionary."""
        summary = self.tracker.get_cost_summary_dict()
        
        assert summary["provider"] == "openai"
        assert summary["model"] == "gpt-4"
        assert summary["input_tokens"] == self.tracker.input_tokens
        assert summary["output_tokens"] == self.tracker.output_tokens
        assert summary["model_input_cost_per_1M_tokens"] == 0.15
        assert summary["model_output_cost_per_1M_tokens"] == 0.60
        assert "total_cost" in summary
        assert summary["total_cost"] == self.tracker.get_current_cost()
    
    def test_print_cost_summary(self, capsys):
        """Test printing cost summary."""
        self.tracker.print_cost_summary()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Cost Summary (ESTIMATED)" in output
        assert "openai/gpt-4" in output
        assert "Input tokens:" in output
        assert "Output tokens:" in output
        assert "Total cost of extraction:" in output


class TestCostTrackerSerialization:
    """Test serialization and deserialization functionality."""
    
    def setup_method(self):
        """Set up test tracker."""
        self.tracker = CostTracker(
            "openai", 
            "gpt-4",
            max_budget=10.0,
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
        
        # Add some tokens
        self.tracker.track_input_text("Hello world")
        self.tracker.track_output_text("Response")
    
    def test_to_dict(self):
        """Test converting tracker to dictionary."""
        state_dict = self.tracker.to_dict()
        
        assert state_dict["provider"] == "openai"
        assert state_dict["model"] == "gpt-4"
        assert state_dict["max_budget"] == 10.0
        assert state_dict["input_tokens"] == self.tracker.input_tokens
        assert state_dict["output_tokens"] == self.tracker.output_tokens
        assert state_dict["model_input_cost_per_1M_tokens"] == 0.15
        assert state_dict["model_output_cost_per_1M_tokens"] == 0.60
    
    @patch('delm.utils.cost_tracker.get_model_token_price')
    def test_from_dict(self, mock_get_price):
        """Test creating tracker from dictionary."""
        mock_get_price.return_value = (0.20, 0.80)
        state_dict = {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "max_budget": 5.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "model_input_cost_per_1M_tokens": 0.20,
            "model_output_cost_per_1M_tokens": 0.80
        }
        
        tracker = CostTracker.from_dict(state_dict)
        
        assert tracker.provider == "anthropic"
        assert tracker.model == "claude-3-5-sonnet-20241022"
        assert tracker.max_budget == 5.0
        assert tracker.input_tokens == 100
        assert tracker.output_tokens == 50
        assert tracker.model_input_cost_per_1M_tokens == 0.20
        assert tracker.model_output_cost_per_1M_tokens == 0.80
    
    @patch('delm.utils.cost_tracker.get_model_token_price')
    def test_from_dict_missing_optional_fields(self, mock_get_price):
        """Test creating tracker from dictionary with missing optional fields."""
        mock_get_price.return_value = (0.15, 0.60)
        state_dict = {
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
        
        tracker = CostTracker.from_dict(state_dict)
        
        assert tracker.provider == "openai"
        assert tracker.model == "gpt-4o-mini"
        assert tracker.max_budget is None
        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0
        assert tracker.model_input_cost_per_1M_tokens == 0.0
        assert tracker.model_output_cost_per_1M_tokens == 0.0
    
    @patch('delm.utils.cost_tracker.get_model_token_price')
    def test_round_trip_serialization(self, mock_get_price):
        """Test round-trip serialization and deserialization."""
        mock_get_price.return_value = (0.15, 0.60)
        original_tracker = self.tracker
        
        # Convert to dict and back
        state_dict = original_tracker.to_dict()
        restored_tracker = CostTracker.from_dict(state_dict)
        
        # Check that all important fields are preserved
        assert restored_tracker.provider == original_tracker.provider
        assert restored_tracker.model == original_tracker.model
        assert restored_tracker.max_budget == original_tracker.max_budget
        assert restored_tracker.input_tokens == original_tracker.input_tokens
        assert restored_tracker.output_tokens == original_tracker.output_tokens
        assert restored_tracker.model_input_cost_per_1M_tokens == original_tracker.model_input_cost_per_1M_tokens
        assert restored_tracker.model_output_cost_per_1M_tokens == original_tracker.model_output_cost_per_1M_tokens


class TestCostTrackerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_large_token_counts(self):
        """Test handling of very large token counts."""
        tracker = CostTracker(
            "openai", 
            "gpt-4o-mini",
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
        
        # Simulate very large token counts
        tracker.input_tokens = 1_000_000
        tracker.output_tokens = 500_000
        
        cost = tracker.get_current_cost()
        
        # Should calculate correctly without overflow
        expected_cost = (1_000_000 * 0.15 + 500_000 * 0.60) / 1_000_000
        assert abs(cost - expected_cost) < 1e-10  # Use approximate comparison for floating point
    
    @patch('delm.utils.cost_tracker.get_model_token_price')
    def test_zero_cost_rates(self, mock_get_price):
        """Test handling of zero cost rates."""
        mock_get_price.return_value = (0.15, 0.60)
        tracker = CostTracker(
            "openai",
            "gpt-4o-mini",
            model_input_cost_per_1M_tokens=0.0,
            model_output_cost_per_1M_tokens=0.0
        )
        
        tracker.track_input_text("Hello world")
        tracker.track_output_text("Response")
        
        cost = tracker.get_current_cost()
        assert cost == 0.0
    
    def test_negative_budget(self):
        """Test handling of negative budget."""
        tracker = CostTracker(
            "openai",
            "gpt-4o-mini",
            max_budget=-1.0
        )
        
        # Should not raise an error, but should always be over budget
        assert tracker.is_over_budget() is True
    
    def test_pydantic_response_with_complex_data(self):
        """Test tracking Pydantic response with complex nested data."""
        class ComplexResponse(BaseModel):
            text: str
            metadata: dict
            scores: list[float]
            nested: dict[str, list[str]]
        
        response = ComplexResponse(
            text="Sample response",
            metadata={"key": "value", "count": 42},
            scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            nested={"category": ["a", "b", "c"], "tags": ["x", "y"]}
        )
        
        tracker = CostTracker(
            "openai", 
            "gpt-4o-mini",
            model_input_cost_per_1M_tokens=0.15,
            model_output_cost_per_1M_tokens=0.60
        )
        
        initial_tokens = tracker.output_tokens
        tracker.track_output_pydantic(response)
        
        assert tracker.output_tokens > initial_tokens 