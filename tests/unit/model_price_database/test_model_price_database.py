"""
Unit tests for the tokencost-backed model price database.
"""

import pytest

from delm.utils.model_price_database import get_model_token_price


class TestGetModelTokenPrice:
    """Test get_model_token_price using tokencost backend."""

    def test_openai_gpt4o_mini(self):
        input_price, output_price = get_model_token_price("openai", "gpt-4o-mini")
        assert input_price > 0
        assert output_price > 0
        assert isinstance(input_price, float)
        assert isinstance(output_price, float)

    def test_openai_gpt4o(self):
        input_price, output_price = get_model_token_price("openai", "gpt-4o")
        assert input_price > 0
        assert output_price > 0

    def test_anthropic_claude(self):
        input_price, output_price = get_model_token_price(
            "anthropic", "claude-3-5-haiku-latest"
        )
        assert input_price > 0
        assert output_price > 0

    def test_output_more_expensive_than_input(self):
        input_price, output_price = get_model_token_price("openai", "gpt-4o-mini")
        assert output_price >= input_price

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="not found in tokencost database"):
            get_model_token_price("nonexistent_provider", "nonexistent_model_xyz_123")

    def test_returns_per_1m_tokens(self):
        input_price, output_price = get_model_token_price("openai", "gpt-4o-mini")
        # Prices per 1M tokens should be in a reasonable range (not per-token tiny values)
        assert input_price >= 0.01
        assert output_price >= 0.01

    def test_case_insensitive_lookup(self):
        price_lower = get_model_token_price("openai", "gpt-4o-mini")
        price_upper = get_model_token_price("openai", "GPT-4o-mini")
        assert price_lower == price_upper

    def test_google_gemini(self):
        input_price, output_price = get_model_token_price("google", "gemini-2.0-flash")
        assert input_price >= 0
        assert output_price >= 0


class TestTokencostIntegrationWithCostTracker:
    """Test that CostTracker uses tokencost for price lookups."""

    def test_cost_tracker_uses_tokencost_prices(self):
        from delm.utils.cost_tracker import CostTracker

        tracker = CostTracker("openai", "gpt-4o-mini")
        assert tracker.model_input_cost_per_1M_tokens > 0
        assert tracker.model_output_cost_per_1M_tokens > 0

    def test_cost_tracker_custom_prices_override_tokencost(self):
        from delm.utils.cost_tracker import CostTracker

        tracker = CostTracker(
            "openai",
            "gpt-4o-mini",
            model_input_cost_per_1M_tokens=99.0,
            model_output_cost_per_1M_tokens=199.0,
        )
        assert tracker.model_input_cost_per_1M_tokens == 99.0
        assert tracker.model_output_cost_per_1M_tokens == 199.0

    def test_cost_tracker_unknown_model_with_custom_prices(self):
        from delm.utils.cost_tracker import CostTracker

        tracker = CostTracker(
            "custom_provider",
            "custom_model_xyz",
            model_input_cost_per_1M_tokens=1.0,
            model_output_cost_per_1M_tokens=2.0,
        )
        assert tracker.model_input_cost_per_1M_tokens == 1.0
        assert tracker.model_output_cost_per_1M_tokens == 2.0

    def test_cost_tracker_unknown_model_without_custom_prices_raises(self):
        from delm.utils.cost_tracker import CostTracker

        with pytest.raises(ValueError, match="not found in tokencost database"):
            CostTracker("nonexistent_provider", "nonexistent_model_xyz_123")
