"""Model pricing database for input/output tokens per million tokens.

Add new models and providers as needed.
"""

import logging
from typing import Tuple

log = logging.getLogger(__name__)

_MODEL_PRICING_DB: dict[tuple[str, str], dict[str, float]] = {
    # ----------------------------------------------------------------------
    # OpenAI models
    # GPT‑5 family (flagship models) — official pricing as of September 2025.
    ("openai", "gpt-5"): {"input": 1.25, "output": 10.0},
    ("openai", "gpt-5-mini"): {"input": 0.25, "output": 2.0},
    ("openai", "gpt-5-nano"): {"input": 0.05, "output": 0.40},
    # GPT‑4.1 family — pricing announced for 4.1 and its reduced variants.
    ("openai", "gpt-4.1"): {"input": 2.0, "output": 8.0},
    ("openai", "gpt-4.1-mini"): {"input": 0.40, "output": 1.60},
    ("openai", "gpt-4.1-nano"): {"input": 0.10, "output": 0.40},
    # GPT‑4o series — multi‑modal models introduced in 2024/25.
    # Standard GPT‑4o pricing reflects the latest reduction (≈$2.5 input, $10 output).
    ("openai", "gpt-4o"): {"input": 2.50, "output": 10.0},
    ("openai", "gpt-4o-mini"): {"input": 0.15, "output": 0.60},
    # OpenAI family of specialised "o" series models.
    ("openai", "o3"): {"input": 2.0, "output": 8.0},
    ("openai", "o3-pro"): {"input": 20.0, "output": 80.0},
    ("openai", "o4-mini"): {"input": 1.10, "output": 4.40},
    ("openai", "o1-preview"): {"input": 15.0, "output": 60.0},
    ("openai", "o1-mini"): {"input": 0.15, "output": 0.60},
    ("openai", "o1-pro"): {"input": 150.0, "output": 600.0},
    # Other GPT‑4 derivatives and legacy models.
    ("openai", "gpt-4-turbo"): {"input": 10.0, "output": 30.0},
    ("openai", "gpt-4"): {"input": 30.0, "output": 60.0},
    ("openai", "gpt-4-32k"): {"input": 60.0, "output": 120.0},
    ("openai", "gpt-4.5"): {"input": 75.0, "output": 150.0},
    ("openai", "gpt-3.5-turbo"): {"input": 0.50, "output": 1.50},
    ("openai", "gpt-3.5-turbo-instruct"): {"input": 1.50, "output": 2.00},

    # ----------------------------------------------------------------------
    # Anthropic Claude models
    # Claude 4 family
    ("anthropic", "claude-opus-4"): {"input": 15.0, "output": 75.0},
    ("anthropic", "claude-sonnet-4"): {"input": 3.0, "output": 15.0},
    # Claude 3 family (latest versions)
    ("anthropic", "claude-3.7-sonnet"): {"input": 3.0, "output": 15.0},
    ("anthropic", "claude-3.5-haiku"): {"input": 0.80, "output": 4.0},
    # Legacy Claude 3 models retained for backward compatibility.
    ("anthropic", "claude-3-opus"): {"input": 15.0, "output": 75.0},
    ("anthropic", "claude-3-haiku"): {"input": 0.25, "output": 1.25},
    # Aliases for convenience so that callers can reference "latest" variants
    ("anthropic", "claude-opus-4-0"): {"input": 15.0, "output": 75.0},
    ("anthropic", "claude-sonnet-4-0"): {"input": 3.0, "output": 15.0},
    ("anthropic", "claude-3-7-sonnet-latest"): {"input": 3.0, "output": 15.0},
    ("anthropic", "claude-3-5-haiku-latest"): {"input": 0.80, "output": 4.0},

    # ----------------------------------------------------------------------
    # Google (Gemini) models — pricing from Google AI Developer docs and UC Today
    ("google", "gemini-2.5-pro"): {"input": 1.25, "output": 10.0},
    ("google", "gemini-2.5-flash"): {"input": 0.30, "output": 2.50},
    ("google", "gemini-2.5-flash-lite"): {"input": 0.10, "output": 0.40},
    ("google", "gemini-2.0-flash"): {"input": 0.10, "output": 0.40},
    ("google", "gemini-2.0-flash-lite"): {"input": 0.075, "output": 0.30},
    # Gemini 1.5 family (pay‑as‑you‑go pricing; using ≤128 k context tier)
    ("google", "gemini-1.5-pro"): {"input": 1.25, "output": 5.0},
    ("google", "gemini-1.5-flash"): {"input": 0.075, "output": 0.30},
    ("google", "gemini-1.5-flash-8b"): {"input": 0.0375, "output": 0.15},
    # Optional entries for larger context windows (≈1 M tokens)
    ("google", "gemini-1.5-pro-1m"): {"input": 2.50, "output": 10.0},
    ("google", "gemini-1.5-flash-1m"): {"input": 0.15, "output": 0.60},
    ("google", "gemini-1.5-flash-8b-1m"): {"input": 0.075, "output": 0.30},
    # Gemini 1.0 Pro
    ("google", "gemini-1.0-pro"): {"input": 0.50, "output": 1.50},

    # ----------------------------------------------------------------------
    # Groq hosted models — pricing from Groq official pricing page
    ("groq", "gpt-oss-20b"): {"input": 0.10, "output": 0.50},
    ("groq", "gpt-oss-120b"): {"input": 0.15, "output": 0.75},
    ("groq", "kimi-k2-0905-1t"): {"input": 1.00, "output": 3.00},
    ("groq", "llama-4-scout"): {"input": 0.11, "output": 0.34},
    ("groq", "llama-4-maverick"): {"input": 0.20, "output": 0.60},
    ("groq", "llama-guard-4-12b"): {"input": 0.20, "output": 0.20},
    ("groq", "deepseek-r1-distill-llama-70b"): {"input": 0.75, "output": 0.99},
    ("groq", "qwen3-32b"): {"input": 0.29, "output": 0.59},
    ("groq", "mistral-saba-24b"): {"input": 0.79, "output": 0.79},
    ("groq", "llama-3.3-70b-versatile"): {"input": 0.59, "output": 0.79},
    ("groq", "llama-3.1-8b-instant"): {"input": 0.05, "output": 0.08},
    ("groq", "llama-3-70b-8k"): {"input": 0.59, "output": 0.79},
    ("groq", "llama-3-8b-8k"): {"input": 0.05, "output": 0.08},
    ("groq", "gemma-2-9b-8k"): {"input": 0.20, "output": 0.20},
    ("groq", "llama-guard-3-8b-8k"): {"input": 0.20, "output": 0.20},

    # ----------------------------------------------------------------------
    # DeepSeek models — pricing based on DeepSeek API documentation
    # Note: for cacheable models (V3/V3.1/R1) we use the cache‑miss price as
    # a conservative default.
    ("deepseek", "deepseek-v3.1"): {"input": 0.56, "output": 1.68},
    ("deepseek", "deepseek-v3"): {"input": 0.27, "output": 1.10},
    ("deepseek", "deepseek-r1"): {"input": 0.55, "output": 2.19},

    # ----------------------------------------------------------------------
    # Ollama hosted models — local inference; zero cost
    # A handful of commonly used open models are enumerated here.  All cost
    # fields are zero because Ollama runs locally on the user's hardware.
    ("ollama", "llama2"): {"input": 0.0, "output": 0.0},
    ("ollama", "llama3"): {"input": 0.0, "output": 0.0},
    ("ollama", "mixtral-8x7b"): {"input": 0.0, "output": 0.0},
    ("ollama", "mistral"): {"input": 0.0, "output": 0.0},
    ("ollama", "phi3"): {"input": 0.0, "output": 0.0},
    ("ollama", "gemma"): {"input": 0.0, "output": 0.0},
    ("ollama", "qwen2"): {"input": 0.0, "output": 0.0},
    ("ollama", "qwen2-7b-instruct"): {"input": 0.0, "output": 0.0},
    ("ollama", "neural-chat"): {"input": 0.0, "output": 0.0},
    ("ollama", "zephyr"): {"input": 0.0, "output": 0.0},
    ("ollama", "tinyllama"): {"input": 0.0, "output": 0.0},
    ("ollama", "dolphin-mistral"): {"input": 0.0, "output": 0.0},
    ("ollama", "open-hermes-2.5-mistral"): {"input": 0.0, "output": 0.0},
    ("ollama", "redpajama"): {"input": 0.0, "output": 0.0},
    ("ollama", "vicuna"): {"input": 0.0, "output": 0.0},
}

def get_model_token_price(provider: str, model: str) -> Tuple[float, float]:
    """
    Look up the price per 1M input/output tokens for a given provider and model.

    Args:
        provider: The provider of the model.
        model: The name of the model.

    Returns:
        (input_price, output_price): tuple of floats

    Raises:
        ValueError: If the model is not found in the provider's model price database.
    """
    log.debug("Looking up price for provider='%s', model='%s'", provider, model)
    for (prov, mod), prices in _MODEL_PRICING_DB.items():
        if prov.lower() == provider.lower() and mod.lower() == model.lower():
            log.debug("Found price for provider='%s', model='%s': input=$%.2f, output=$%.2f",
                     provider, model, prices["input"], prices["output"])
            return prices["input"], prices["output"]
    log.error("Model '%s' not found in model price database for provider '%s'", model, provider)
    raise ValueError(f"Model {model} not found in model price database for provider {provider}")
