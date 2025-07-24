"""
Performance estimation utilities for DELM.

Encapsulates schema-aware precision/recall and merging logic for pipeline evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

from delm.config import DELMConfig
from delm.constants import (
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN,
    SYSTEM_RANDOM_SEED,
    SYSTEM_RECORD_ID_COLUMN,
    SYSTEM_CHUNK_ID_COLUMN,
)
from delm.exceptions import ProcessingError
from delm.utils.json_merge import merge_jsons_for_record
from delm.schemas.schemas import BaseSchema

def estimate_performance(
    config: Union[str, Dict[str, Any], DELMConfig],
    data_source: str | Path | pd.DataFrame,
    expected_extraction_output_df: pd.DataFrame,
    true_json_column: str,
    matching_id_column: str,
    record_sample_size: int = -1,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """
    Estimate the performance of the DELM pipeline.
    Returns a dict with both the aggregated_extracted_data and field-level precision/recall metrics.
    """
    from delm.delm import DELM
    print("[WARNING] This method will use the API to estimate performance. This will charge you for the sampled data requests.")
    config_obj = DELMConfig.from_any(config)
    print(DELM)
    delm = DELM(
        config=config_obj,
        experiment_name="cost_estimation",
        experiment_directory=Path(),
        overwrite_experiment=False,
        auto_checkpoint_and_resume_experiment=True,
        use_disk_storage=False,
    )
    source_df = delm.data_processor.load_data(data_source)
    total_source_records = len(source_df)
    total_expected_records = len(expected_extraction_output_df)
    
    # Sampling
    if record_sample_size < 1:
        record_sample_size = total_expected_records

    record_sample_size = min(record_sample_size, total_expected_records, total_source_records)

    sampled_expected_df = expected_extraction_output_df.sample(n=record_sample_size, random_state=SYSTEM_RANDOM_SEED)
    sampled_source_df = source_df[source_df[matching_id_column].isin(sampled_expected_df[matching_id_column])]
    prepped_data = delm.data_processor.process_dataframe(sampled_source_df) # type: ignore
    if len(prepped_data) == 0:
            raise ProcessingError(f"No data to process. There may be no overlap in `{matching_id_column}` in input data.")
    delm.experiment_manager.save_preprocessed_data(prepped_data)

    results = delm.process_via_llm()
    
    if results.empty or SYSTEM_EXTRACTED_DATA_JSON_COLUMN not in results.columns:
        raise ValueError("No results or missing DICT column.")

    extraction_schema = delm.schema_manager.get_extraction_schema()

    # Parse expected JSON column if needed (if user provided as string)
    if isinstance(expected_extraction_output_df[true_json_column].iloc[0], str):
        expected_extraction_output_df[true_json_column] = expected_extraction_output_df[true_json_column].apply(json.loads)
    # Verify that that expected_extraction_output is valid against the schema
    for i, row in expected_extraction_output_df.iterrows():
        extraction_schema.validate_json_dict(row[true_json_column], path=f"expected_extraction_output[{i}]") # type: ignore
    
    # Group and merge extracted data by record_id using agg to keep dicts as values
    # Drop SYSTEM_CHUNK_ID_COLUMN from results
    results = results.drop(columns=[SYSTEM_CHUNK_ID_COLUMN])
    other_cols = [col for col in results.columns if col not in [SYSTEM_RECORD_ID_COLUMN, SYSTEM_EXTRACTED_DATA_JSON_COLUMN]]

    def collapse_or_list(series):
        unique = series.dropna().unique()
        if len(unique) == 1:
            return unique[0]
        else:
            return list(unique)

    agg_dict = {SYSTEM_EXTRACTED_DATA_JSON_COLUMN: lambda x: merge_jsons_for_record(list(x), extraction_schema)}
    agg_dict.update({col: collapse_or_list for col in other_cols})

    extracted_data_df = (
        results.groupby(SYSTEM_RECORD_ID_COLUMN)
        .agg(agg_dict)
        .reset_index()
    )

    record_id_extracted_expected_dicts_df = pd.merge(
        expected_extraction_output_df[[matching_id_column, true_json_column]],
        extracted_data_df[[matching_id_column, SYSTEM_EXTRACTED_DATA_JSON_COLUMN]],
        on=matching_id_column,
        how="inner"
    )
    record_id_extracted_expected_dicts_df.columns = [matching_id_column, "expected_dict", "extracted_dict"]
    performance_metrics_dict = _aggregate_precision_recall_across_records(
        record_id_extracted_expected_dicts_df["expected_dict"].tolist(),
        record_id_extracted_expected_dicts_df["extracted_dict"].tolist(),
        extraction_schema,
    )
    return performance_metrics_dict, record_id_extracted_expected_dicts_df 



def _is_missing(val: Any) -> bool:
    """Return True when `val` is semantically ‘no information’."""
    return (
        val is None
        or val == ""
        or (isinstance(val, (list, dict)) and len(val) == 0)
    )

def _make_hashable(val: Any) -> Any:
    """
    Convert lists/dicts to a stable JSON string; return None for missing values
    so they can be filtered out of set calculations.
    """
    if _is_missing(val):
        return None
    if isinstance(val, (list, dict)):
        return json.dumps(val, sort_keys=True)
    return val

def _build_required_map(schema: BaseSchema, parent: list[str] | None = None) -> dict[str, bool]:
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

def _all_levels_precision_recall(
    y_true: Any,
    y_pred: Any,
    required_map: dict[str, bool],
    key: str | None = None,
    path: list[str] | None = None,
) -> dict[str, dict[str, int | float]]:
    path = path or []
    results: dict[str, dict[str, int | float]] = {}
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
                    tp = len(t_set & p_set)
                    fp = len(p_set - t_set)
                    fn = len(t_set - p_set)
                    prec = tp / (tp + fp) if tp + fp else 0.0
                    rec  = tp / (tp + fn) if tp + fn else 0.0
                    results[pstr] = {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
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
                tp = len(t_set & p_set)
                fp = len(p_set - t_set)
                fn = len(t_set - p_set)
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec  = tp / (tp + fn) if tp + fn else 0.0
                results[path_str] = {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
            key_union = {k for d in true_dicts + pred_dicts for k in d}
            for k in key_union:
                sub_path = path + [k]
                pstr = ".".join(sub_path)
                required = required_map.get(pstr, True)
                t_vals = {_make_hashable(d.get(k)) for d in true_dicts if k in d} - {None}
                p_vals = {_make_hashable(d.get(k)) for d in pred_dicts if k in d} - {None}
                if required or t_vals:
                    tp_f = len(t_vals & p_vals)
                    fp_f = len(p_vals - t_vals)
                    fn_f = len(t_vals - p_vals)
                    prec_f = tp_f / (tp_f + fp_f) if tp_f + fp_f else 0.0
                    rec_f  = tp_f / (tp_f + fn_f) if tp_f + fn_f else 0.0
                    results[pstr] = {"precision": prec_f, "recall": rec_f, "tp": tp_f, "fp": fp_f, "fn": fn_f}
                t_nested = [d.get(k) for d in true_dicts if k in d]
                p_nested = [d.get(k) for d in pred_dicts if k in d]
                if any(isinstance(v, (dict, list)) for v in t_nested + p_nested):
                    results.update(
                        _all_levels_precision_recall(t_nested, p_nested, required_map, k, sub_path)
                    )
            return results
        if required or y_true:
            t_set = {_make_hashable(v) for v in y_true} - {None}
            p_set = {_make_hashable(v) for v in y_pred} - {None}
            tp = len(t_set & p_set)
            fp = len(p_set - t_set)
            fn = len(t_set - p_set)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec  = tp / (tp + fn) if tp + fn else 0.0
            results[path_str] = {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
        return results
    return results

def _aggregate_precision_recall_across_records(
    expected_list: list[Any],
    predicted_list: list[Any],
    schema: BaseSchema,
) -> dict[str, dict[str, float]]:
    required_map = _build_required_map(schema)
    from collections import defaultdict
    agg = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0})
    for y_true, y_pred in zip(expected_list, predicted_list):
        rec_metrics = _all_levels_precision_recall(y_true, y_pred, required_map)
        for field, m in rec_metrics.items():
            agg[field]["tp"] += m["tp"]
            agg[field]["fp"] += m["fp"]
            agg[field]["fn"] += m["fn"]
    for field, c in agg.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        c["precision"] = tp / (tp + fp) if tp + fp else 0.0
        c["recall"]    = tp / (tp + fn) if tp + fn else 0.0
    return dict(agg)

