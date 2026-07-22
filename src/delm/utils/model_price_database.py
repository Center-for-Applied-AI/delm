"""Model pricing lookup via the ``tokencost`` package.

Delegates all price resolution to tokencost's maintained database of 400+ models.
If a model is missing from the bundled database, the live price feed is fetched
once per process before giving up, so newly released models resolve automatically.
"""

import logging
import threading
from typing import Any, Dict, Optional, Tuple

from tokencost import refresh_prices
from tokencost.constants import TOKEN_COSTS
from tokencost.costs import _normalize_model_for_pricing

log = logging.getLogger(__name__)

_REFRESH_LOCK = threading.Lock()
_PRICES_REFRESHED = False


def _find_entry(provider: str, model: str) -> Optional[Dict[str, Any]]:
    """Look up a tokencost entry, trying ``provider/model`` then bare ``model``."""
    candidates = [
        f"{provider}/{model}",
        model,
    ]
    for candidate in candidates:
        normalized = _normalize_model_for_pricing(candidate)
        if normalized in TOKEN_COSTS:
            return TOKEN_COSTS[normalized]
    return None


def _find_entry_with_refresh(provider: str, model: str) -> Optional[Dict[str, Any]]:
    """Look up a tokencost entry, refreshing the live price feed once on a miss."""
    global _PRICES_REFRESHED

    entry = _find_entry(provider, model)
    if entry is not None:
        return entry

    with _REFRESH_LOCK:
        if not _PRICES_REFRESHED:
            log.info(
                "Model '%s/%s' not in bundled tokencost database; "
                "refreshing prices from the live feed",
                provider,
                model,
            )
            refresh_prices(write_file=False)
            _PRICES_REFRESHED = True

    return _find_entry(provider, model)


def get_model_token_price(provider: str, model: str) -> Tuple[float, float]:
    """
    Look up the price per 1M input/output tokens for a given provider and model.

    Tries ``provider/model`` first (litellm convention), then bare ``model``.
    If the model is missing from the bundled database, the live price feed is
    fetched once per process and the lookup is retried.

    Args:
        provider: The provider of the model (e.g. "openai", "anthropic").
        model: The name of the model (e.g. "gpt-4o-mini").

    Returns:
        (input_price_per_1M, output_price_per_1M): tuple of floats.

    Raises:
        ValueError: If the model is not found in the tokencost database.
    """
    log.debug("Looking up price for provider='%s', model='%s'", provider, model)

    entry = _find_entry_with_refresh(provider, model)
    if entry is not None:
        input_per_token = entry.get("input_cost_per_token", 0.0)
        output_per_token = entry.get("output_cost_per_token", 0.0)
        # tokencost stores cost-per-token; convert to cost-per-1M-tokens
        input_per_1m = float(input_per_token) * 1_000_000
        output_per_1m = float(output_per_token) * 1_000_000
        log.debug(
            "Found price for '%s/%s': input=$%.4f/1M, output=$%.4f/1M",
            provider,
            model,
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


def get_model_token_limits(
    provider: str, model: str
) -> Tuple[Optional[int], Optional[int]]:
    """
    Look up the input/output token limits for a given provider and model.

    Args:
        provider: The provider of the model (e.g. "openai", "anthropic").
        model: The name of the model (e.g. "gpt-4o-mini").

    Returns:
        (max_input_tokens, max_output_tokens): tuple of optional ints. Either
        value is None when tokencost does not report the corresponding limit.

    Raises:
        ValueError: If the model is not found in the tokencost database.
    """
    log.debug("Looking up token limits for provider='%s', model='%s'", provider, model)

    entry = _find_entry_with_refresh(provider, model)
    if entry is None:
        raise ValueError(
            f"Model {model} not found in tokencost database for provider {provider}. "
            f"Register the model via tokencost.configure_model()."
        )

    max_input_tokens = entry.get("max_input_tokens")
    max_output_tokens = entry.get("max_output_tokens", entry.get("max_tokens"))
    return (
        int(max_input_tokens) if max_input_tokens is not None else None,
        int(max_output_tokens) if max_output_tokens is not None else None,
    )
