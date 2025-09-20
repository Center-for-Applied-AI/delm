"""
Run DELM extraction, cost estimation, and performance estimation on
`data/commodity_data.csv` using the configuration in
`examples/cost_vs_coverage/config.yaml` and the schema in
`examples/commodity_schema.yaml`.

Sections: setup -> config -> data import -> data augmentation ->
model fitting and other -> data export.

This script saves artifacts into `examples/cost_vs_coverage/`.
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from delm import DELM, DELMConfig
from delm.utils.performance_estimation import estimate_performance
from delm.utils.cost_estimation import estimate_total_cost


# ----------------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------------

RANDOM_SEED = 42

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

SOURCE_DATA_PATH = PROJECT_ROOT / "data" / "commodity_data.csv"
CONFIG_PATH = CURRENT_DIR / "config.yaml"

SCHEMA_PATH = next(
    (
        p.resolve()
        for p in [
            CURRENT_DIR / "commodity_schema.yaml",
            CURRENT_DIR.parent / "commodity_schema.yaml",
        ]
        if p.is_file()
    ),
    None,
)


EXPERIMENT_NAME = "cost_coverage_example"
EXPERIMENT_DIR = CURRENT_DIR / "experiments"

# Expected JSON container name from the schema
CONTAINER_NAME = "commodity_prices"

EXTRACTED_RESULTS_CSV = CURRENT_DIR / "extracted_results.csv"
EXPECTED_VS_EXTRACTED_CSV = CURRENT_DIR / "expected_vs_extracted.csv"
COST_SUMMARY_JSON = CURRENT_DIR / "cost_summary.json"
ESTIMATED_COSTS_JSON = CURRENT_DIR / "estimated_costs.json"
PERFORMANCE_METRICS_JSON = CURRENT_DIR / "performance_metrics.json"

# Tune these to control API spend during estimation runs
PERF_SAMPLE_SIZE = -1
COST_EST_SAMPLE_SIZE = 30


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def build_expected_df(record_labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Create a nested expected JSON per id, aggregating duplicates.

    The expected JSON aligns with the nested schema container.

    Args:
        record_labeled_df: Labeled DataFrame with columns:
            `id`, `good`, `good_subtype`, `price_expectation`, `price_lower`,
            `price_upper`, `unit`, `currency`, `horizon`.

    Returns:
        DataFrame with columns: `id`, `expected_json` (dict with container).
    """
    output_fields = [
        "good",
        "good_subtype",
        "price_expectation",
        "price_lower",
        "price_upper",
        "unit",
        "currency",
        "horizon",
    ]

    missing = [c for c in ["id", *output_fields] if c not in record_labeled_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    labeled_items_df = record_labeled_df[["id", *output_fields]].copy()

    # Aggregate all labeled items per id into a single list
    items_by_id = (
        labeled_items_df.groupby("id")[output_fields]
        .apply(lambda g: g.to_dict(orient="records"))
        .reset_index(name="items")
    )

    items_by_id["expected_json"] = items_by_id["items"].apply(
        lambda items: {CONTAINER_NAME: items}
    )
    return items_by_id[["id", "expected_json"]]


def dump_json(path: Path, payload: dict | list) -> None:
    """Write a JSON payload with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def stringify_dict_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected dict-like columns to JSON strings for CSV export."""
    result_df = df.copy()
    for col in columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].apply(lambda v: json.dumps(v, ensure_ascii=False))
    return result_df


# ----------------------------------------------------------------------------
# main flow
# ----------------------------------------------------------------------------

def main() -> None:
    """Run extraction, cost estimation, and performance estimation with DELM."""

    # config
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    config_obj = DELMConfig.from_yaml(CONFIG_PATH)
    config_obj.schema.spec_path = SCHEMA_PATH
    pipeline = DELM(
        config=config_obj,
        experiment_name=EXPERIMENT_NAME,
        experiment_directory=EXPERIMENT_DIR,
    )

    # data import
    record_labeled_df = pd.read_csv(SOURCE_DATA_PATH)

    # Ensure single input row per id to avoid duplicate joins later
    record_text_df = (
        record_labeled_df[["id", "text"]]
        .drop_duplicates(subset=["id"], keep="first")
        .copy()
    )

    # data augmentation
    record_expected_df = build_expected_df(record_labeled_df)

    # model fitting and other
    preprocessed_chunks_df = pipeline.prep_data(record_text_df)
    pipeline.process_via_llm()

    extracted_results_df = pipeline.get_extraction_results()
    # Attach original metadata (including 'id') to results via chunk id
    preproc_meta_df = preprocessed_chunks_df.drop(columns=["delm_text_chunk"], errors="ignore")
    extracted_results_df = extracted_results_df.merge(
        preproc_meta_df, on="delm_chunk_id", how="left"
    )
    cost_summary = pipeline.get_cost_summary()

    estimated_total_cost = estimate_total_cost(
        config=config_obj,
        data_source=str(SOURCE_DATA_PATH),
        sample_size=COST_EST_SAMPLE_SIZE,
    )

    performance_metrics, expected_vs_extracted_df = estimate_performance(
        config=config_obj,
        data_source=record_text_df,
        expected_extraction_output_df=record_expected_df,
        true_json_column="expected_json",
        matching_id_column="id",
        record_sample_size=PERF_SAMPLE_SIZE,
    )

    # data export
    extracted_results_df.to_csv(EXTRACTED_RESULTS_CSV, index=False)
    dump_json(COST_SUMMARY_JSON, cost_summary)
    dump_json(
        ESTIMATED_COSTS_JSON,
        {
            "sample_size": COST_EST_SAMPLE_SIZE,
            "estimated_total_cost": estimated_total_cost,
        },
    )
    dump_json(PERFORMANCE_METRICS_JSON, performance_metrics)

    expected_vs_extracted_to_save = stringify_dict_columns(
        expected_vs_extracted_df, columns=["expected_dict", "extracted_dict"]
    )
    expected_vs_extracted_to_save.to_csv(EXPECTED_VS_EXTRACTED_CSV, index=False)


if __name__ == "__main__":
    main()


