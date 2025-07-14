import tiktoken
from .model_price_database import get_model_token_price

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
        self.input_tokens += self._count_tokens(text)

    def track_output_text(self, text: str):
        self.output_tokens += self._count_tokens(text)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_current_cost(self) -> float:
        return (
            self.input_tokens * self.model_input_cost_per_1M_tokens / 1_000_000
            + self.output_tokens * self.model_output_cost_per_1M_tokens / 1_000_000
        )
        
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