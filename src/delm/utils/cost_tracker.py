import tiktoken
import json
from delm.utils.model_price_database import get_model_token_price
from typing import List, Any

class CostTracker:
    def __init__(
        self, 
        provider: str,
        model: str,
        # TODO: Let user specify input/output cost per 1M tokens
        # model_input_cost_per_1M_tokens: float,
        # model_output_cost_per_1M_tokens: float,
        # TODO: Let user specify tokenizer, 
        # tokenizer: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base"),
    ) -> None:
        self.provider = provider
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model_input_cost_per_1M_tokens, self.model_output_cost_per_1M_tokens = get_model_token_price(provider, model)
        self.input_tokens = 0
        self.output_tokens = 0
    
    def track_input_text(self, text: str):
        self.input_tokens += self.count_tokens(text)

    def track_output_text(self, text: str):
        self.output_tokens += self.count_tokens(text)
    
    def track_output_pydantic(self, response: Any) -> None:

        self.output_tokens += self.count_tokens(json.dumps(response.model_dump(mode="json")))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> int:
        return sum(self.count_tokens(t) for t in texts)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.model_input_cost_per_1M_tokens / 1_000_000
            + output_tokens * self.model_output_cost_per_1M_tokens / 1_000_000
        )

    def get_cost_summary_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_input_cost_per_1M_tokens": self.model_input_cost_per_1M_tokens,
            "model_output_cost_per_1M_tokens": self.model_output_cost_per_1M_tokens,
            "total_cost": self.get_current_cost(),
        }
    
    def print_cost_summary(self) -> None:
        print("=" * 50)
        print("Cost Summary (ESTIMATED)")
        print("=" * 50)
        print(f"Model: {self.provider}/{self.model}")
        print(f"Input tokens: {self.input_tokens}")
        print(f"Output tokens: {self.output_tokens}")
        print(f"Input price per 1M tokens: ${self.model_input_cost_per_1M_tokens:.3f}")
        print(f"Output price per 1M tokens: ${self.model_output_cost_per_1M_tokens:.3f}")
        print(f"Total cost of extraction: ${self.get_current_cost():.3f}")

    def get_current_cost(self) -> float:
        return self.estimate_cost(self.input_tokens, self.output_tokens)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_input_cost_per_1M_tokens": self.model_input_cost_per_1M_tokens,
            "model_output_cost_per_1M_tokens": self.model_output_cost_per_1M_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CostTracker":
        obj = cls(d["provider"], d["model"])
        obj.input_tokens = d.get("input_tokens", 0)
        obj.output_tokens = d.get("output_tokens", 0)
        obj.model_input_cost_per_1M_tokens = d.get("model_input_cost_per_1M_tokens", 0.0)
        obj.model_output_cost_per_1M_tokens = d.get("model_output_cost_per_1M_tokens", 0.0)
        return obj