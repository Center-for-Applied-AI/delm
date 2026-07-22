"""Cost estimation helpers for DELM.

Provides utilities to estimate approximate input token costs without API calls,
an upper bound on total cost without API calls, and total extraction costs
using a sampled run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from delm.delm import DELM
from delm.constants import (
    SYSTEM_CHUNK_COLUMN,
    SYSTEM_RANDOM_SEED,
    SYSTEM_LOG_FILE_PREFIX,
    SYSTEM_LOG_FILE_SUFFIX,
)
from delm.config import DELMConfig
from delm.logging import configure as configure_logging
from delm.utils.few_shot import FewShotExampleSelector
from delm.utils.model_price_database import get_model_token_limits

# Module-level logger
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _configure_estimation_logging(
    save_file_log: bool,
    log_dir: Optional[Union[str, Path]],
    console_log_level: str,
    file_log_level: str,
) -> None:
    """Configure logging for a cost estimation run."""
    if save_file_log:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = f"{SYSTEM_LOG_FILE_PREFIX}cost_estimation_{current_time}{SYSTEM_LOG_FILE_SUFFIX}"
    else:
        log_file_name = None

    configure_logging(
        console_level=console_log_level,
        file_dir=log_dir,
        file_name=log_file_name,
        file_level=file_log_level,
    )


def _build_estimation_delm(
    config: Union[str, Dict[str, Any], DELMConfig, DELM],
    data_source: Union[str, Path] | pd.DataFrame,
) -> DELM:
    """Create an in-memory DELM instance with preprocessed data for estimation."""
    if isinstance(config, DELM):
        config = config.config
    config_obj = DELMConfig.from_any(config)
    log.debug(
        "Config loaded: %s",
        config_obj.name if hasattr(config_obj, "name") else "unknown",
    )

    delm = DELM.from_config(
        config=config_obj,
        use_disk_storage=False,
        override_logging=False,
    )
    log.debug("DELM instance created for cost estimation")

    delm.prep_data(data_source)
    log.debug("Data prepared for cost estimation")
    return delm


def _compute_chunk_input_tokens(delm: DELM) -> List[int]:
    """Compute per-chunk input token counts (system + user prompt + schema JSON)."""
    extraction_schema = delm.config.schema.schema
    log.debug("Extraction schema loaded: %s", type(extraction_schema).__name__)

    llm_cfg = delm.config.llm_extraction_cfg
    system_prompt = llm_cfg.system_prompt
    user_prompt_template = llm_cfg.prompt_template
    few_shot_selector = FewShotExampleSelector.from_optional(
        examples=llm_cfg.few_shot_examples,
        num_examples=llm_cfg.few_shot_num_examples,
        truncate_length=llm_cfg.few_shot_truncate_length,
        random_sample=llm_cfg.few_shot_random_sample,
    )
    if few_shot_selector is not None:
        user_prompt_template = few_shot_selector.inject_into_template(
            user_prompt_template
        )
    variables_text = extraction_schema.get_variables_text()
    log.debug(
        "Prompt setup: system_length=%d, template_length=%d, variables_length=%d",
        len(system_prompt),
        len(user_prompt_template),
        len(variables_text),
    )

    # Precompute the schema overhead once (counts toward prompt tokens)
    SchemaType = extraction_schema.create_pydantic_schema()
    schema_text = json.dumps(SchemaType.model_json_schema())
    log.debug("Computed schema overhead for estimation: %d chars", len(schema_text))

    chunks = delm.experiment_manager.load_preprocessed_data()[
        SYSTEM_CHUNK_COLUMN
    ].tolist()
    log.debug("Processing %d chunks for token estimation", len(chunks))

    chunk_input_tokens: List[int] = []
    for i, chunk in enumerate(chunks):
        formatted_prompt = user_prompt_template.format(
            variables=variables_text, text=chunk
        )
        # Include schema JSON for estimation alongside system + user prompt
        complete_prompt = f"{system_prompt}\n\n{formatted_prompt}\n{schema_text}"
        chunk_input_tokens.append(delm.cost_tracker.count_tokens(complete_prompt))
        if i % 100 == 0:  # Log progress every 100 chunks
            log.debug(
                "Processed %d/%d chunks, total tokens so far: %d",
                i + 1,
                len(chunks),
                sum(chunk_input_tokens),
            )
    return chunk_input_tokens


# --------------------------------------------------------------------------- #
# Cost Estimation Methods                                                     #
# --------------------------------------------------------------------------- #


def estimate_input_token_cost(
    config: Union[str, Dict[str, Any], DELMConfig, DELM],
    data_source: Union[str, Path] | pd.DataFrame,
    save_file_log: bool = False,
    log_dir: Optional[Union[str, Path]] = Path(".delm/logs/cost_estimation"),
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG",
) -> float:
    """Estimate input token cost over the entire dataset without API calls.

    Args:
        config: Configuration for the DELM pipeline (config path | dict | ``DELMConfig``).
        data_source: Source data for extraction (path or DataFrame).
        save_file_log: Whether to write a rotating log file.
        log_dir: Directory for log files when ``save_file_log`` is True.
        console_log_level: Log level for console output.
        file_log_level: Log level for file output.

    Returns:
        Estimated dollar cost of input tokens for processing all chunks.
    """
    _configure_estimation_logging(
        save_file_log, log_dir, console_log_level, file_log_level
    )

    log.debug("Estimating input token cost for data source: %s", data_source)
    delm = _build_estimation_delm(config, data_source)
    chunk_input_tokens = _compute_chunk_input_tokens(delm)

    total_input_tokens = sum(chunk_input_tokens)
    input_price_per_1M = delm.cost_tracker.model_input_cost_per_1M_tokens
    total_cost = total_input_tokens * input_price_per_1M / 1_000_000

    log.debug(
        "Input token cost estimation completed: %d total tokens, $%.6f total cost",
        total_input_tokens,
        total_cost,
    )
    return total_cost


def estimate_max_total_cost(
    config: Union[str, Dict[str, Any], DELMConfig, DELM],
    data_source: Union[str, Path] | pd.DataFrame,
    save_file_log: bool = False,
    log_dir: Optional[Union[str, Path]] = Path(".delm/logs/cost_estimation"),
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG",
) -> float:
    """Estimate an upper bound on the total cost without API calls.

    For each chunk the output tokens are bounded by
    ``min(max_completion_tokens, context_window - input_tokens)``, so the
    upper bound of the cost is::

        input_price * input_tokens + output_price * min(
            max_completion_tokens, context_window - input_tokens
        )

    The model's context window and maximum output tokens are looked up from
    the tokencost database. When they are unavailable (e.g. a custom model
    with manual price overrides), the bound falls back to
    ``max_completion_tokens`` alone.

    Args:
        config: Configuration for the DELM pipeline (config path | dict | ``DELMConfig``).
        data_source: Source data for extraction (path or DataFrame).
        save_file_log: Whether to write a rotating log file.
        log_dir: Directory for log files when ``save_file_log`` is True.
        console_log_level: Log level for console output.
        file_log_level: Log level for file output.

    Returns:
        Upper-bound dollar cost for processing all chunks.
    """
    _configure_estimation_logging(
        save_file_log, log_dir, console_log_level, file_log_level
    )

    log.debug("Estimating max total cost for data source: %s", data_source)
    delm = _build_estimation_delm(config, data_source)
    chunk_input_tokens = _compute_chunk_input_tokens(delm)

    llm_cfg = delm.config.llm_extraction_cfg
    max_completion_tokens = llm_cfg.max_completion_tokens

    try:
        context_window, model_max_output_tokens = get_model_token_limits(
            llm_cfg.provider, llm_cfg.model
        )
    except ValueError:
        log.warning(
            "Token limits for %s/%s not found in tokencost database; "
            "upper bound uses max_completion_tokens=%d only",
            llm_cfg.provider,
            llm_cfg.model,
            max_completion_tokens,
        )
        context_window, model_max_output_tokens = None, None

    output_token_cap = max_completion_tokens
    if model_max_output_tokens is not None:
        output_token_cap = min(output_token_cap, model_max_output_tokens)

    total_input_tokens = 0
    total_max_output_tokens = 0
    for input_tokens in chunk_input_tokens:
        chunk_output_bound = output_token_cap
        if context_window is not None:
            chunk_output_bound = min(
                chunk_output_bound, max(context_window - input_tokens, 0)
            )
        total_input_tokens += input_tokens
        total_max_output_tokens += chunk_output_bound

    input_price_per_1M = delm.cost_tracker.model_input_cost_per_1M_tokens
    output_price_per_1M = delm.cost_tracker.model_output_cost_per_1M_tokens
    max_total_cost = (
        total_input_tokens * input_price_per_1M
        + total_max_output_tokens * output_price_per_1M
    ) / 1_000_000

    log.debug(
        "Max total cost estimation completed: %d input tokens, "
        "%d max output tokens, $%.6f upper bound",
        total_input_tokens,
        total_max_output_tokens,
        max_total_cost,
    )
    return max_total_cost


def estimate_total_cost(
    config: Union[str, Dict[str, Any], DELMConfig, DELM],
    data_source: Union[str, Path] | pd.DataFrame,
    sample_size: int = 10,
    save_file_log: bool = False,
    log_dir: Optional[Union[str, Path]] = Path(".delm/logs/cost_estimation"),
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG",
) -> float:
    """Estimate total cost using API calls on a sample of the data.

    Args:
        config: Configuration for the DELM pipeline (config path | dict | ``DELMConfig``).
        data_source: Source data for extraction (path or DataFrame).
        sample_size: Number of records to sample for cost estimation.
        save_file_log: Whether to write a rotating log file.
        log_dir: Directory for log files when ``save_file_log`` is True.
        console_log_level: Log level for console output.
        file_log_level: Log level for file output.

    Returns:
        Estimated dollar cost for processing the entire dataset, scaled from the sample.
    """
    _configure_estimation_logging(
        save_file_log, log_dir, console_log_level, file_log_level
    )

    log.warning(
        "This method will use the API to estimate the cost. This will charge you for the sampled data requests."
    )

    log.debug(
        "Estimating total cost with API calls: data_source=%s, sample_size=%d",
        data_source,
        sample_size,
    )
    if isinstance(config, DELM):
        config = config.config
    config_obj = DELMConfig.from_any(config)
    log.debug(
        "Config loaded: %s",
        config_obj.name if hasattr(config_obj, "name") else "unknown",
    )

    delm = DELM.from_config(
        config=config_obj,
        use_disk_storage=False,
    )
    log.debug("DELM instance created for API cost estimation")

    delm.cost_tracker.count_cache_hits_towards_cost = True
    log.debug("Cache hits will be counted towards cost")

    records_df = delm.data_processor.load_data(data_source)
    total_records = len(records_df)
    log.debug("Loaded %d total records from data source", total_records)

    sample_records_df = records_df.sample(
        n=sample_size, random_state=SYSTEM_RANDOM_SEED
    )
    log.debug("Sampled %d records for cost estimation", len(sample_records_df))

    sample_chunks_df = delm.data_processor.process_dataframe(sample_records_df)
    log.debug("Processed sample records into %d chunks", len(sample_chunks_df))

    delm.experiment_manager.save_preprocessed_data(sample_chunks_df)
    log.debug("Saved preprocessed sample data")

    log.debug("Starting LLM processing for cost estimation")
    delm.process_via_llm()
    log.debug("LLM processing completed")

    sample_cost = delm.cost_tracker.get_current_cost()
    total_estimated_cost = sample_cost * (total_records / sample_size)

    log.debug(
        "Total cost estimation completed: sample_cost=$%.6f, total_estimated_cost=$%.6f",
        sample_cost,
        total_estimated_cost,
    )
    return total_estimated_cost
