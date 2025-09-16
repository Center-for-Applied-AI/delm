"""
Performance estimation utilities for DELM.

Encapsulates schema-aware precision/recall and merging logic for pipeline evaluation.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, Union, Optional
import pandas as pd

from delm.config import DELMConfig
from delm.constants import (
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN,
    SYSTEM_RANDOM_SEED,
    SYSTEM_RECORD_ID_COLUMN,
    SYSTEM_CHUNK_ID_COLUMN,
    DEFAULT_LOG_DIR,
    SYSTEM_LOG_FILE_PREFIX,
    SYSTEM_LOG_FILE_SUFFIX,
    DEFAULT_CONSOLE_LOG_LEVEL,
    DEFAULT_FILE_LOG_LEVEL,
)
from delm.utils.post_processing import merge_jsons_for_record
from delm.schemas.schemas import BaseSchema

# Module-level logger
log = logging.getLogger(__name__)


def estimate_performance(
    config: Union[str, Dict[str, Any], DELMConfig],
    data_source: Union[str, Path, pd.DataFrame],
    expected_extraction_output_df: pd.DataFrame,
    true_json_column: str,
    matching_id_column: str,
    record_sample_size: int = -1,
    save_file_log: bool = True,
    log_dir: Optional[Union[str, Path]] = None,
    console_log_level: str = DEFAULT_CONSOLE_LOG_LEVEL,
    file_log_level: str = DEFAULT_FILE_LOG_LEVEL,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """
    Estimate the performance of the DELM pipeline.
    Returns a dict with both the aggregated_extracted_data and field-level precision/recall metrics.

    Args:
        config: Configuration for the DELM pipeline.
        data_source: Source data for extraction.
        expected_extraction_output_df: DataFrame with expected extraction results.
        true_json_column: Column name containing true JSON data.
        matching_id_column: Column name for matching records.
        record_sample_size: Number of records to sample (-1 for all).
        save_file_log: Whether to save a log file.
        log_dir: Optional path to log directory. If None, creates {DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE_PREFIX}_performance_estimation_run_<timestamp>.log at project root.
        console_log_level: Log level for console output.
        file_log_level: Log level for file output.
    """
    from delm.delm import DELM
    from delm.logging import configure
    from datetime import datetime

    # Configure logging
    if save_file_log:
        if log_dir is None:
            log_dir = Path(DEFAULT_LOG_DIR) / "performance_estimation"
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = f"{SYSTEM_LOG_FILE_PREFIX}performance_estimation_{current_time}{SYSTEM_LOG_FILE_SUFFIX}"
    else:
        log_file_name = None

    configure(
        console_level=console_log_level,
        file_dir=log_dir,
        file_name=log_file_name,
        file_level=file_log_level,
    )

    log.warning(
        "This method will use the API to estimate performance. This will charge you for the sampled data requests."
    )

    log.debug(
        "Estimating performance: data_source=%s, true_json_column=%s, matching_id_column=%s, record_sample_size=%d",
        data_source,
        true_json_column,
        matching_id_column,
        record_sample_size,
    )

    config_obj = DELMConfig.from_any(config)
    log.debug(
        "Config loaded: %s",
        config_obj.name if hasattr(config_obj, "name") else "unknown",
    )

    delm = DELM(
        config=config_obj,
        experiment_name="performance_estimation",
        experiment_directory=Path(),
        overwrite_experiment=False,
        auto_checkpoint_and_resume_experiment=True,
        use_disk_storage=False,
        override_logging=False,
    )
    log.debug("DELM instance created for performance estimation")

    source_df = delm.data_processor.load_data(data_source)
    total_source_records = len(source_df)
    total_expected_records = len(expected_extraction_output_df)
    log.debug(
        "Data loaded: %d source records, %d expected records",
        total_source_records,
        total_expected_records,
    )

    # Sampling
    if record_sample_size < 1:
        record_sample_size = total_expected_records
        log.debug("Using all expected records for sampling: %d", record_sample_size)

    record_sample_size = min(
        record_sample_size, total_expected_records, total_source_records
    )
    log.debug("Final sample size: %d", record_sample_size)

    sampled_expected_df = expected_extraction_output_df.sample(
        n=record_sample_size, random_state=SYSTEM_RANDOM_SEED
    )
    log.debug(
        "Sampled %d expected records for performance estimation",
        len(sampled_expected_df),
    )

    if matching_id_column not in source_df.columns:
        log.error(
            "Matching ID column '%s' not found in source data columns: %s",
            matching_id_column,
            list(source_df.columns),
        )
        raise ValueError(
            f"Matching ID column `{matching_id_column}` not found in source data columns {source_df.columns}"
        )
    if matching_id_column not in sampled_expected_df.columns:
        log.error(
            "Matching ID column '%s' not found in expected data columns: %s",
            matching_id_column,
            list(sampled_expected_df.columns),
        )
        raise ValueError(
            f"Matching ID column `{matching_id_column}` not found in expected data columns {sampled_expected_df.columns}"
        )

    log.debug("Matching ID column validation passed")

    sampled_source_df = source_df[
        source_df[matching_id_column].isin(sampled_expected_df[matching_id_column])
    ]
    log.debug(
        "Filtered source data to %d records matching expected data",
        len(sampled_source_df),
    )

    prepped_data = delm.data_processor.process_dataframe(sampled_source_df)  # type: ignore
    log.debug("Processed source data into %d chunks", len(prepped_data))

    if len(prepped_data) == 0:
        log.error(
            "No data to process. There may be no overlap in '%s' in input data.",
            matching_id_column,
        )
        raise ValueError(
            f"No data to process. There may be no overlap in `{matching_id_column}` in input data."
        )

    delm.experiment_manager.save_preprocessed_data(prepped_data)
    log.debug("Saved preprocessed data for performance estimation")

    log.debug("Starting LLM processing for performance estimation")
    results = delm.process_via_llm()
    log.debug("LLM processing completed, got %d results", len(results))

    if results.empty or SYSTEM_EXTRACTED_DATA_JSON_COLUMN not in results.columns:
        log.error(
            "No results or missing DICT column. Results columns: %s",
            list(results.columns) if not results.empty else "empty",
        )
        raise ValueError("No results or missing DICT column.")

    extraction_schema = delm.schema_manager.get_extraction_schema()
    log.debug("Extraction schema loaded: %s", type(extraction_schema).__name__)

    # Parse expected JSON column if needed (if user provided as string)
    if isinstance(expected_extraction_output_df[true_json_column].iloc[0], str):
        log.debug("Parsing expected JSON column from strings")
        expected_extraction_output_df[true_json_column] = expected_extraction_output_df[
            true_json_column
        ].apply(json.loads)

    # Verify that that expected_extraction_output is valid against the schema
    log.debug("Validating expected extraction output against schema")
    for i, row in expected_extraction_output_df.iterrows():
        extraction_schema.is_valid_json_dict(row[true_json_column], path=f"expected_extraction_output[{i}]")  # type: ignore
    log.debug(
        "Schema validation completed for %d expected records",
        len(expected_extraction_output_df),
    )

    # Group and merge extracted data by record_id using agg to keep dicts as values
    # Drop SYSTEM_CHUNK_ID_COLUMN from results
    log.debug(
        "Preparing results for aggregation: dropping %s column", SYSTEM_CHUNK_ID_COLUMN
    )
    results = results.drop(columns=[SYSTEM_CHUNK_ID_COLUMN])
    other_cols = [
        col
        for col in results.columns
        if col not in [SYSTEM_RECORD_ID_COLUMN, SYSTEM_EXTRACTED_DATA_JSON_COLUMN]
    ]
    log.debug("Other columns for aggregation: %s", other_cols)

    def collapse_or_list(series):
        unique = series.dropna().unique()
        if len(unique) == 1:
            return unique[0]
        else:
            return list(unique)

    agg_dict = {
        SYSTEM_EXTRACTED_DATA_JSON_COLUMN: lambda x: merge_jsons_for_record(
            list(x), extraction_schema
        )
    }
    agg_dict.update({col: collapse_or_list for col in other_cols})
    log.debug("Aggregation dictionary created with %d columns", len(agg_dict))

    log.debug("Aggregating results by record ID")
    extracted_data_df = (
        results.groupby(SYSTEM_RECORD_ID_COLUMN).agg(agg_dict).reset_index()
    )
    log.debug("Aggregation completed: %d unique records", len(extracted_data_df))

    log.debug("Merging expected and extracted data")
    record_id_extracted_expected_dicts_df = pd.merge(
        expected_extraction_output_df[[matching_id_column, true_json_column]],
        extracted_data_df[[matching_id_column, SYSTEM_EXTRACTED_DATA_JSON_COLUMN]],
        on=matching_id_column,
        how="inner",
    )
    record_id_extracted_expected_dicts_df.columns = [
        matching_id_column,
        "expected_dict",
        "extracted_dict",
    ]
    log.debug(
        "Merge completed: %d matched records",
        len(record_id_extracted_expected_dicts_df),
    )

    log.debug("Calculating performance metrics")
    performance_metrics_dict = _aggregate_performance_metrics_across_records(
        record_id_extracted_expected_dicts_df["expected_dict"].tolist(),
        record_id_extracted_expected_dicts_df["extracted_dict"].tolist(),
        extraction_schema,
    )
    log.info(
        "Performance estimation completed: %d fields with metrics",
        len(performance_metrics_dict),
    )
    return performance_metrics_dict, record_id_extracted_expected_dicts_df


def _is_missing(val: Any) -> bool:
    """Return True when `val` is semantically ‘no information’.

    Args:
        val: The value to check.

    Returns:
        True if the value is semantically ‘no information’, False otherwise.
    """
    return val is None or val == "" or (isinstance(val, (list, dict)) and len(val) == 0)


def _make_hashable(val: Any) -> Any:
    """
    Convert lists/dicts to a stable JSON string; return None for missing values
    so they can be filtered out of set calculations.

    Args:
        val: The value to convert.

    Returns:
        A hashable value.
    """
    if _is_missing(val):
        return None
    if isinstance(val, (list, dict)):
        return json.dumps(val, sort_keys=True)
    return val


def _build_required_map(
    schema: BaseSchema, parent: list[str] | None = None
) -> dict[str, bool]:
    """Build a map of required fields.

    Args:
        schema: The schema to build the map from.
        parent: The parent path.

    Returns:
        A map of required fields.
    """
    parent = parent or []
    req_map: dict[str, bool] = {}
    stype = getattr(schema, "schema_type", type(schema).__name__).lower()
    if stype == "simpleschema":
        for var in schema.variables:
            req_map[".".join(parent + [var.name])] = getattr(var, "required", False)
    elif stype == "nestedschema":
        cont = schema.container_name
        for var in schema.variables:
            path = parent + [cont, var.name]
            req_map[".".join(path)] = getattr(var, "required", False)
    elif stype == "multipleschema":
        for name, sub in schema.schemas.items():
            req_map.update(_build_required_map(sub, parent + [name]))
    return req_map


def _calculate_confusion_matrix_counts(t_set: set, p_set: set) -> dict[str, int]:
    """Calculate confusion matrix counts from true and predicted sets.

    Args:
        t_set: Set of true values.
        p_set: Set of predicted values.

    Returns:
        Dictionary with tp, fp, fn counts.
    """
    tp = len(t_set & p_set)
    fp = len(p_set - t_set)
    fn = len(t_set - p_set)
    return {"tp": tp, "fp": fp, "fn": fn}


def _all_levels_precision_recall(
    y_true: Any,
    y_pred: Any,
    required_map: dict[str, bool],
    key: Optional[str] = None,
    path: list[str] | None = None,
) -> dict[str, dict[str, Union[int, float]]]:
    """Calculate precision/recall for a given field.

    Args:
        y_true: The true value.
        y_pred: The predicted value.
        required_map: The map of required fields.
        key: The key to calculate precision/recall for.
        path: The path to the field.

    Returns:
        A dictionary of metrics for the field: ["precision", "recall", "tp", "fp", "fn"].
    """
    path = path or []
    results: dict[str, dict[str, Union[int, float]]] = {}
    if isinstance(y_true, dict) and isinstance(y_pred, dict):
        keys = sorted(set(y_true) | set(y_pred))
        for k in keys:
            sub_path = path + [k]
            t_val, p_val = y_true.get(k), y_pred.get(k)
            pstr = ".".join(sub_path)
            required = required_map.get(pstr, False)
            if not any(isinstance(v, (dict, list)) for v in (t_val, p_val)):
                if required or not _is_missing(t_val):
                    t_set = {_make_hashable(t_val)} - {None}
                    p_set = {_make_hashable(p_val)} - {None}
                    results[pstr] = _calculate_confusion_matrix_counts(t_set, p_set)
            results.update(
                _all_levels_precision_recall(t_val, p_val, required_map, k, sub_path)
            )
        return results
    if isinstance(y_true, list) and isinstance(y_pred, list):
        true_dicts = [d for d in y_true if isinstance(d, dict)]
        pred_dicts = [d for d in y_pred if isinstance(d, dict)]
        path_str = ".".join(path) if path else "root"
        required = required_map.get(path_str, False)
        if true_dicts or pred_dicts:
            if required or true_dicts:
                t_set = {json.dumps(d, sort_keys=True) for d in true_dicts}
                p_set = {json.dumps(d, sort_keys=True) for d in pred_dicts}
                results[path_str] = _calculate_confusion_matrix_counts(t_set, p_set)
            key_union = {k for d in true_dicts + pred_dicts for k in d}
            for k in key_union:
                sub_path = path + [k]
                pstr = ".".join(sub_path)
                required = required_map.get(pstr, True)
                t_vals = {_make_hashable(d.get(k)) for d in true_dicts if k in d} - {
                    None
                }
                p_vals = {_make_hashable(d.get(k)) for d in pred_dicts if k in d} - {
                    None
                }
                if required or t_vals:
                    results[pstr] = _calculate_confusion_matrix_counts(t_vals, p_vals)
                t_nested = [d.get(k) for d in true_dicts if k in d]
                p_nested = [d.get(k) for d in pred_dicts if k in d]
                if any(isinstance(v, (dict, list)) for v in t_nested + p_nested):
                    results.update(
                        _all_levels_precision_recall(
                            t_nested, p_nested, required_map, k, sub_path
                        )
                    )
            return results
        if required or y_true:
            t_set = {_make_hashable(v) for v in y_true} - {None}
            p_set = {_make_hashable(v) for v in y_pred} - {None}
            results[path_str] = _calculate_confusion_matrix_counts(t_set, p_set)
        return results
    return results


def _aggregate_performance_metrics_across_records(
    expected_list: list[Any],
    predicted_list: list[Any],
    schema: BaseSchema,
) -> dict[str, dict[str, float]]:
    log.debug("Aggregating performance metrics across %d records", len(expected_list))
    required_map = _build_required_map(schema)
    log.debug("Built required map with %d fields", len(required_map))

    from collections import defaultdict

    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for i, (y_true, y_pred) in enumerate(zip(expected_list, predicted_list)):
        if i % 100 == 0:  # Log progress every 100 records
            log.debug("Processing record %d/%d for metrics", i + 1, len(expected_list))
        rec_metrics = _all_levels_precision_recall(y_true, y_pred, required_map)
        for field, m in rec_metrics.items():
            agg[field]["tp"] += m["tp"]
            agg[field]["fp"] += m["fp"]
            agg[field]["fn"] += m["fn"]

    log.debug("Calculating final metrics for %d fields", len(agg))
    result = {}
    for field, counts in agg.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        result[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    log.debug("Performance metrics aggregation completed: %d fields", len(result))
    return result
