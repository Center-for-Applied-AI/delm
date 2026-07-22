"""
Unit tests for offline cost estimation, including the upper-bound estimate
from issue #44. No API calls are made.
"""

import pandas as pd
import pytest

from delm import DELMConfig, Schema
from delm.models import ExtractionVariable
from delm.utils.cost_estimation import (
    estimate_input_token_cost,
    estimate_max_total_cost,
)


@pytest.fixture(autouse=True)
def fake_openai_api_key(monkeypatch):
    """Instructor client creation requires a key; no API calls are made."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-calls-made")


@pytest.fixture
def simple_schema():
    return Schema.simple(
        variables_list=[
            ExtractionVariable(
                name="company",
                description="Company name",
                data_type="string",
            )
        ]
    )


@pytest.fixture
def report_text_df():
    return pd.DataFrame(
        {
            "text": [
                "Goldman Sachs expects oil prices to rise.",
                "Morgan Stanley cut its gas price forecast.",
                "JP Morgan sees steel demand growing 8%.",
            ]
        }
    )


def _make_config(simple_schema, **overrides):
    defaults = {
        "schema": simple_schema,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "cache_backend": None,
    }
    defaults.update(overrides)
    return DELMConfig(**defaults)


class TestEstimateMaxTotalCost:
    def test_upper_bound_exceeds_input_only_estimate(
        self, simple_schema, report_text_df
    ):
        config = _make_config(simple_schema)
        input_cost = estimate_input_token_cost(
            config, report_text_df, console_log_level="ERROR"
        )
        max_cost = estimate_max_total_cost(
            config, report_text_df, console_log_level="ERROR"
        )
        assert input_cost > 0
        assert max_cost > input_cost

    def test_upper_bound_scales_with_max_completion_tokens(
        self, simple_schema, report_text_df
    ):
        small_cfg = _make_config(simple_schema, max_completion_tokens=100)
        large_cfg = _make_config(simple_schema, max_completion_tokens=10_000)
        small_bound = estimate_max_total_cost(
            small_cfg, report_text_df, console_log_level="ERROR"
        )
        large_bound = estimate_max_total_cost(
            large_cfg, report_text_df, console_log_level="ERROR"
        )
        assert large_bound > small_bound

    def test_upper_bound_formula_with_known_prices(self, simple_schema, report_text_df):
        """Output part of the bound must equal output_price * n_chunks * cap."""
        input_price = 1.0
        output_price = 2.0
        max_completion_tokens = 100
        config = _make_config(
            simple_schema,
            model_input_cost_per_1M_tokens=input_price,
            model_output_cost_per_1M_tokens=output_price,
            max_completion_tokens=max_completion_tokens,
        )
        input_cost = estimate_input_token_cost(
            config, report_text_df, console_log_level="ERROR"
        )
        max_cost = estimate_max_total_cost(
            config, report_text_df, console_log_level="ERROR"
        )
        n_chunks = len(report_text_df)
        # Prompts here are tiny, so the context window never binds and each
        # chunk contributes exactly max_completion_tokens of output.
        expected_output_cost = (
            n_chunks * max_completion_tokens * output_price / 1_000_000
        )
        assert max_cost == pytest.approx(input_cost + expected_output_cost)

    def test_unknown_model_with_manual_prices_falls_back(
        self, simple_schema, report_text_df
    ):
        config = _make_config(
            simple_schema,
            model="totally-unknown-model-xyz-123",
            model_input_cost_per_1M_tokens=1.0,
            model_output_cost_per_1M_tokens=2.0,
            max_completion_tokens=1000,
        )
        max_cost = estimate_max_total_cost(
            config, report_text_df, console_log_level="ERROR"
        )
        assert max_cost > 0
