# Model pricing database for input/output tokens per million tokens.
# Add new models and providers as needed.

import logging
from typing import Tuple

log = logging.getLogger(__name__)

_MODEL_PRICING_DB = {
    # ------------------------------------------------------------------------ #
    # OpenAI
    ("openai", "gpt-4.1"): {
        "input": 2.0,    # $2.00 per 1M input tokens
        "output": 8.0,   # $8.00 per 1M output tokens
    },
    ("openai", "gpt-4.1-mini"): {
        "input": 0.4,    # $0.40 per 1M input tokens
        "output": 1.6,   # $1.60 per 1M output tokens
    },
    ("openai", "gpt-4.1-nano"): {
        "input": 0.1,    # $0.10 per 1M input tokens
        "output": 0.4,   # $0.40 per 1M output tokens
    },
    ("openai", "o3"): {
        "input": 2.0,    # $2.00 per 1M input tokens
        "output": 8.0,   # $8.00 per 1M output tokens
    },
    ("openai", "o4-mini"): {
        "input": 1.1,    # $1.10 per 1M input tokens
        "output": 4.4,   # $4.40 per 1M output tokens
    },
    ("openai", "gpt-4o"): {
        "input": 5.0,    # $5.00 per 1M input tokens
        "output": 20.0,  # $20.00 per 1M output tokens
    },
    ("openai", "gpt-4o-mini"): {
        "input": 0.6,    # $0.60 per 1M input tokens
        "output": 2.4,   # $2.40 per 1M output tokens
    },
    
    # ------------------------------------------------------------------------ #
    # ANTROPIC API (UNTESTED)
    # Anthropic - Current Models (Claude 4)
    ("anthropic", "claude-opus-4-20250514"): {
        "input": 15.0,   # $15.00 per 1M input tokens
        "output": 75.0,  # $75.00 per 1M output tokens
    },
    ("anthropic", "claude-sonnet-4-20250514"): {
        "input": 3.0,    # $3.00 per 1M input tokens
        "output": 15.0,  # $15.00 per 1M output tokens
    },
    ("anthropic", "claude-3-7-sonnet-20250219"): {
        "input": 3.0,    # $3.00 per 1M input tokens
        "output": 15.0,  # $15.00 per 1M output tokens
    },
    ("anthropic", "claude-3-5-sonnet-20241022"): {
        "input": 3.0,    # $3.00 per 1M input tokens
        "output": 15.0,  # $15.00 per 1M output tokens
    },
    ("anthropic", "claude-3-5-haiku-20241022"): {
        "input": 0.8,    # $0.80 per 1M input tokens
        "output": 4.0,   # $4.00 per 1M output tokens
    },
    
    # Anthropic - Legacy Models (Claude 3)
    ("anthropic", "claude-3-opus-20240229"): {
        "input": 15.0,   # $15.00 per 1M input tokens
        "output": 75.0,  # $75.00 per 1M output tokens
    },
    ("anthropic", "claude-3-haiku-20240307"): {
        "input": 0.25,   # $0.25 per 1M input tokens
        "output": 1.25,  # $1.25 per 1M output tokens
    },
    
    # Anthropic - Model Aliases (for convenience)
    ("anthropic", "claude-opus-4-0"): {
        "input": 15.0,   # $15.00 per 1M input tokens
        "output": 75.0,  # $75.00 per 1M output tokens
    },
    ("anthropic", "claude-sonnet-4-0"): {
        "input": 3.0,    # $3.00 per 1M input tokens
        "output": 15.0,  # $15.00 per 1M output tokens
    },
    ("anthropic", "claude-3-7-sonnet-latest"): {
        "input": 3.0,    # $3.00 per 1M input tokens
        "output": 15.0,  # $15.00 per 1M output tokens
    },
    ("anthropic", "claude-3-5-sonnet-latest"): {
        "input": 3.0,    # $3.00 per 1M input tokens
        "output": 15.0,  # $15.00 per 1M output tokens
    },
    ("anthropic", "claude-3-5-haiku-latest"): {
        "input": 0.8,    # $0.80 per 1M input tokens
        "output": 4.0,   # $4.00 per 1M output tokens
    },
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
