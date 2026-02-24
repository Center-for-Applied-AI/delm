"""Model pricing lookup via the ``tokencost`` package.

Delegates all price resolution to tokencost's maintained database of 400+ models.
"""

import logging
from typing import Tuple

from tokencost.constants import TOKEN_COSTS
from tokencost.costs import _normalize_model_for_pricing

log = logging.getLogger(__name__)


def get_model_token_price(provider: str, model: str) -> Tuple[float, float]:
    """
    Look up the price per 1M input/output tokens for a given provider and model.

    Tries ``provider/model`` first (litellm convention), then bare ``model``.

    Args:
        provider: The provider of the model (e.g. "openai", "anthropic").
        model: The name of the model (e.g. "gpt-4o-mini").

    Returns:
        (input_price_per_1M, output_price_per_1M): tuple of floats.

    Raises:
        ValueError: If the model is not found in the tokencost database.
    """
    log.debug("Looking up price for provider='%s', model='%s'", provider, model)

    candidates = [
        f"{provider}/{model}",
        model,
    ]

    for candidate in candidates:
        normalized = _normalize_model_for_pricing(candidate)
        if normalized in TOKEN_COSTS:
            entry = TOKEN_COSTS[normalized]
            input_per_token = entry.get("input_cost_per_token", 0.0)
            output_per_token = entry.get("output_cost_per_token", 0.0)
            # tokencost stores cost-per-token; convert to cost-per-1M-tokens
            input_per_1m = float(input_per_token) * 1_000_000
            output_per_1m = float(output_per_token) * 1_000_000
            log.debug(
                "Found price for '%s': input=$%.4f/1M, output=$%.4f/1M",
                candidate,
                input_per_1m,
                output_per_1m,
            )
            return input_per_1m, output_per_1m

    log.error(
        "Model '%s' not found in tokencost database for provider '%s'",
        model,
        provider,
    )
    raise ValueError(
        f"Model {model} not found in tokencost database for provider {provider}. "
        f"Set model_input_cost_per_1M_tokens and model_output_cost_per_1M_tokens manually, "
        f"or register the model via tokencost.configure_model()."
    )
