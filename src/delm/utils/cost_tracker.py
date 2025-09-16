"""Token counting and cost tracking utilities for DELM."""

import logging
import tiktoken
import json
from delm.utils.model_price_database import get_model_token_price
from typing import List, Any, Union, Optional

# Module-level logger
log = logging.getLogger(__name__)


class CostTracker:
    """Track tokens and estimate cost for an extraction run."""
    def __init__(
        self,
        provider: str,
        model: str,
        max_budget: Optional[float] = None,
        count_cache_hits_towards_cost: bool = False,
        model_input_cost_per_1M_tokens: Optional[float] = None,
        model_output_cost_per_1M_tokens: Optional[float] = None,
    ) -> None:
        log.debug("Initializing cost tracker for %s/%s", provider, model)
        self.provider = provider
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model_input_cost_per_1M_tokens = (
            model_input_cost_per_1M_tokens
            if model_input_cost_per_1M_tokens is not None
            else get_model_token_price(provider, model)[0]
        )
        self.model_output_cost_per_1M_tokens = (
            model_output_cost_per_1M_tokens
            if model_output_cost_per_1M_tokens is not None
            else get_model_token_price(provider, model)[1]
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.count_cache_hits_towards_cost = count_cache_hits_towards_cost
        self.max_budget = max_budget

        log.debug(
            "Cost tracker initialized - input: $%.6f/1M tokens, output: $%.6f/1M tokens",
            self.model_input_cost_per_1M_tokens,
            self.model_output_cost_per_1M_tokens,
        )

    def is_over_budget(self) -> bool:
        """Return True if current estimated cost exceeds ``max_budget``."""
        current_cost = self.get_current_cost()
        if self.max_budget is None:
            return False
        is_over = current_cost > self.max_budget
        if is_over:
            log.warning("Budget exceeded: $%.4f > $%.4f", current_cost, self.max_budget)
        return is_over

    def track_input_text(self, text: str):
        """Accumulate input tokens for a single text string."""
        tokens = self.count_tokens(text)
        self.input_tokens += tokens
        log.debug(
            "Tracked input text: %d tokens (total: %d)", tokens, self.input_tokens
        )

    def track_output_text(self, text: str):
        """Accumulate output tokens for a single text string."""
        tokens = self.count_tokens(text)
        self.output_tokens += tokens
        log.debug(
            "Tracked output text: %d tokens (total: %d)", tokens, self.output_tokens
        )

    def track_output_pydantic(self, response: Any) -> None:
        """Accumulate output tokens from a Pydantic model response."""
        response_json = json.dumps(response.model_dump(mode="json"))
        tokens = self.count_tokens(response_json)
        self.output_tokens += tokens
        log.debug(
            "Tracked Pydantic output: %d tokens (total: %d)", tokens, self.output_tokens
        )

    def count_tokens(self, text: str) -> int:
        """Return token count for a given string using the model tokenizer."""
        tokens = len(self.tokenizer.encode(text))
        log.debug("Counted tokens: %d for text length %d", tokens, len(text))
        return tokens

    def count_tokens_batch(self, texts: List[str]) -> int:
        """Return total token count for an iterable of strings."""
        total_tokens = sum(self.count_tokens(t) for t in texts)
        log.debug(
            "Counted batch tokens: %d total for %d texts", total_tokens, len(texts)
        )
        return total_tokens

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate dollar cost for given input and output token counts."""
        input_cost = input_tokens * self.model_input_cost_per_1M_tokens / 1_000_000
        output_cost = output_tokens * self.model_output_cost_per_1M_tokens / 1_000_000
        total_cost = input_cost + output_cost
        log.debug(
            "Estimated cost: input=%d tokens ($%.6f), output=%d tokens ($%.6f), total=$%.6f",
            input_tokens,
            input_cost,
            output_tokens,
            output_cost,
            total_cost,
        )
        return total_cost

    def get_cost_summary_dict(self) -> dict[str, Any]:
        """Return a dictionary summary of the current cost state."""
        summary = {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_input_cost_per_1M_tokens": self.model_input_cost_per_1M_tokens,
            "model_output_cost_per_1M_tokens": self.model_output_cost_per_1M_tokens,
            "total_cost": self.get_current_cost(),
        }
        log.debug("Cost summary: %s", summary)
        return summary

    def print_cost_summary(self) -> None:
        """Print a human‑readable cost summary to stdout."""
        print("=" * 50)
        print("Cost Summary (ESTIMATED)")
        print("=" * 50)
        print(f"Model: {self.provider}/{self.model}")
        print(f"Input tokens: {self.input_tokens}")
        print(f"Output tokens: {self.output_tokens}")
        print(f"Input price per 1M tokens: ${self.model_input_cost_per_1M_tokens:.3f}")
        print(
            f"Output price per 1M tokens: ${self.model_output_cost_per_1M_tokens:.3f}"
        )
        print(f"Total cost of extraction: ${self.get_current_cost():.3f}")

    def get_current_cost(self) -> float:
        """Return the current estimated total cost."""
        current_cost = self.estimate_cost(self.input_tokens, self.output_tokens)
        log.debug("Current cost: $%.6f", current_cost)
        return current_cost

    def to_dict(self) -> dict:
        """Serialize the tracker state to a dictionary."""
        state_dict = {
            "provider": self.provider,
            "model": self.model,
            "max_budget": self.max_budget,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_input_cost_per_1M_tokens": self.model_input_cost_per_1M_tokens,
            "model_output_cost_per_1M_tokens": self.model_output_cost_per_1M_tokens,
        }
        log.debug("CostTracker state: %s", state_dict)
        return state_dict

    @classmethod
    def from_dict(cls, d: dict) -> "CostTracker":
        """Create a tracker from a previously serialized dictionary."""
        log.debug("Creating CostTracker from dict: %s", d)
        # Create object without calling __init__ to avoid database lookup
        obj = cls.__new__(cls)
        obj.provider = d["provider"]
        obj.model = d["model"]
        obj.tokenizer = tiktoken.get_encoding("cl100k_base")
        obj.model_input_cost_per_1M_tokens = d.get(
            "model_input_cost_per_1M_tokens", 0.0
        )
        obj.model_output_cost_per_1M_tokens = d.get(
            "model_output_cost_per_1M_tokens", 0.0
        )
        obj.input_tokens = d.get("input_tokens", 0)
        obj.output_tokens = d.get("output_tokens", 0)
        obj.count_cache_hits_towards_cost = False  # Default value
        obj.max_budget = d.get("max_budget", None)
        log.debug(
            "CostTracker restored from dict: provider=%s, model=%s, input_tokens=%d, output_tokens=%d",
            obj.provider,
            obj.model,
            obj.input_tokens,
            obj.output_tokens,
        )
        return obj
