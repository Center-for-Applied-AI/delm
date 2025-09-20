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
from typing import Any
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


EXPERIMENT_NAME = "cost_coverage_greedy"
EXPERIMENT_DIR = CURRENT_DIR / "experiments"

# Expected JSON container name from the schema
CONTAINER_NAME = "commodity_prices"

EXTRACTED_RESULTS_CSV = CURRENT_DIR / "extracted_results.csv"
EXPECTED_VS_EXTRACTED_CSV = CURRENT_DIR / "expected_vs_extracted.csv"
COST_SUMMARY_JSON = CURRENT_DIR / "cost_summary.json"
ESTIMATED_COSTS_JSON = CURRENT_DIR / "estimated_costs.json"
PERFORMANCE_METRICS_JSON = CURRENT_DIR / "performance_metrics.json"
SELECTION_RESULTS_CSV = CURRENT_DIR / "keyword_selection_greedy.csv"
PARETO_FIG_PATH = CURRENT_DIR / "pareto_frontier.png"
PARETO_FIG_DEG2_PATH = CURRENT_DIR / "pareto_frontier_deg2.png"
PARETO_FIG_PDF_PATH = CURRENT_DIR / "pareto_frontier.pdf"
PARETO_FIG_DEG2_PDF_PATH = CURRENT_DIR / "pareto_frontier_deg2.pdf"

# Tune these to control API spend during estimation runs
PERF_SAMPLE_SIZE = -1
COST_EST_SAMPLE_SIZE = 30
TEST_SIZE = 0.2


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
# keyword selection utilities
# ----------------------------------------------------------------------------

def _split_train_test_ids(
    record_text_df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[list[object], list[object]]:
    """Split unique ids into train and test subsets.

    Args:
        record_text_df: DataFrame with columns "id" and "text" and unique ids.
        test_size: Fraction for test split.
        random_seed: Seed for reproducibility.

    Returns:
        Tuple of train_ids and test_ids.
    """
    unique_ids = record_text_df["id"].unique().tolist()
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=random_seed, shuffle=True
    )
    return train_ids, test_ids


def _clone_config_with_keywords(
    base_config: DELMConfig,
    keywords: list[str],
) -> DELMConfig:
    """Return a fresh config with a specific KeywordScorer list.

    Args:
        base_config: Base DELM configuration object.
        keywords: Keywords for KeywordScorer.

    Returns:
        New DELMConfig with updated scoring keywords.
    """
    cfg_dict = base_config.to_serialized_config_dict()
    cfg_dict["data_preprocessing"]["scoring"] = {
        "type": "KeywordScorer",
        "keywords": list(keywords),
    }
    cfg_dict["schema"]["spec_path"] = str(SCHEMA_PATH)
    return DELMConfig.from_dict(cfg_dict)


def _evaluate_recall_and_cost(
    config: DELMConfig,
    text_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    perf_sample_size: int,
    cost_est_sample_size: int,
) -> tuple[float, float]:
    """Compute recall for commodity_prices.good and estimated total cost.

    Args:
        config: DELM configuration to evaluate.
        text_df: Source records with columns "id" and target text.
        expected_df: Expected JSON per id from ``build_expected_df``.
        perf_sample_size: Record sample size for performance estimation.
        cost_est_sample_size: Sample size for cost estimation.

    Returns:
        Tuple of (recall, estimated_total_cost).
    """
    metrics, _ = estimate_performance(
        config=config,
        data_source=text_df,
        expected_extraction_output_df=expected_df,
        true_json_column="expected_json",
        matching_id_column="id",
        record_sample_size=perf_sample_size,
    )
    target_key = f"{CONTAINER_NAME}.good"
    recall = float(metrics.get(target_key, {}).get("recall", 0.0))

    total_cost = estimate_total_cost(
        config=config,
        data_source=text_df,
        sample_size=cost_est_sample_size,
    )
    return recall, float(total_cost)


def _greedy_keyword_selection(
    base_config: DELMConfig,
    candidate_keywords: list[str],
    train_text_df: pd.DataFrame,
    train_expected_df: pd.DataFrame,
    test_text_df: pd.DataFrame,
    test_expected_df: pd.DataFrame,
    perf_sample_size: int,
    cost_est_sample_size: int,
) -> pd.DataFrame:
    """Greedy forward selection maximizing train recall of commodity_prices.good.

    Args:
        base_config: Base DELM configuration.
        candidate_keywords: Candidate keyword pool.
        train_text_df: Training records DataFrame.
        train_expected_df: Training expected JSON DataFrame.
        test_text_df: Test records DataFrame.
        test_expected_df: Test expected JSON DataFrame.
        perf_sample_size: Record sample size for performance estimation.
        cost_est_sample_size: Sample size for cost estimation.

    Returns:
        DataFrame with one row per k including selected keywords, recalls, and costs.
    """
    selected: list[str] = []
    remaining: list[str] = list(dict.fromkeys([kw.lower() for kw in candidate_keywords]))
    train_recall_cache: dict[tuple[str, ...], float] = {}

    records: list[dict[str, Any]] = []

    for k in range(1, len(remaining) + 1):
        best_kw = None
        best_recall = -1.0
        for kw in tqdm(remaining, desc=f"k={k} select", leave=False):
            combo = tuple(selected + [kw])
            if combo in train_recall_cache:
                recall_k = train_recall_cache[combo]
            else:
                cfg_k = _clone_config_with_keywords(base_config, list(combo))
                recall_k, _ = _evaluate_recall_and_cost(
                    cfg_k,
                    train_text_df,
                    train_expected_df,
                    perf_sample_size,
                    cost_est_sample_size,
                )
                train_recall_cache[combo] = recall_k
            if recall_k > best_recall:
                best_recall = recall_k
                best_kw = kw

        if best_kw is None:
            break
        selected.append(best_kw)
        remaining.remove(best_kw)

        cfg_selected = _clone_config_with_keywords(base_config, selected)
        train_recall, train_cost = _evaluate_recall_and_cost(
            cfg_selected,
            train_text_df,
            train_expected_df,
            perf_sample_size,
            cost_est_sample_size,
        )
        test_recall, test_cost = _evaluate_recall_and_cost(
            cfg_selected,
            test_text_df,
            test_expected_df,
            perf_sample_size,
            cost_est_sample_size,
        )

        records.append(
            {
                "k": k,
                "keywords": json.dumps(selected, ensure_ascii=False),
                "train_recall": train_recall,
                "train_estimated_total_cost": train_cost,
                "test_recall": test_recall,
                "test_estimated_total_cost": test_cost,
            }
        )

    result_df = pd.DataFrame.from_records(records)
    if not result_df.empty:
        train_max = result_df["train_estimated_total_cost"].max()
        test_max = result_df["test_estimated_total_cost"].max()
        result_df["train_cost_pct"] = result_df["train_estimated_total_cost"] / train_max
        result_df["test_cost_pct"] = result_df["test_estimated_total_cost"] / test_max
    return result_df


# ----------------------------------------------------------------------------
# plotting utilities
# ----------------------------------------------------------------------------

def _fit_concave_quadratic(x: np.ndarray, y: np.ndarray) -> np.poly1d | None:
    """Fit y ≈ a x^2 + b x + c with a ≤ 0 via grid search on a and least squares for b, c.

    Returns None if fewer than 3 points are available.
    """
    if len(x) < 3:
        return None
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return np.poly1d([a, b, c])
    width = max(1.0, abs(a) * 5.0)
    a_grid = np.linspace(-width, 0.0, 201)
    M = np.column_stack([x, np.ones_like(x)])
    best_sse = None
    best_params = None
    for a_candidate in a_grid:
        y_tilde = y - a_candidate * (x ** 2)
        sol, *_ = np.linalg.lstsq(M, y_tilde, rcond=None)
        b_cand, c_cand = float(sol[0]), float(sol[1])
        pred = a_candidate * (x ** 2) + b_cand * x + c_cand
        sse = float(np.sum((y - pred) ** 2))
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_params = (a_candidate, b_cand, c_cand)
    if best_params is None:
        return None
    a_best, b_best, c_best = best_params
    return np.poly1d([a_best, b_best, c_best])
# ----------------------------------------------------------------------------
# main flow
# ----------------------------------------------------------------------------

def main() -> None:
    """Run greedy keyword selection with train/test split, CSV, and Pareto plot."""

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    config_obj = DELMConfig.from_yaml(CONFIG_PATH)
    config_obj.schema.spec_path = SCHEMA_PATH

    record_labeled_df = pd.read_csv(SOURCE_DATA_PATH)

    record_text_df = (
        record_labeled_df[["id", "text"]]
        .drop_duplicates(subset=["id"], keep="first")
        .copy()
    )

    train_ids, test_ids = _split_train_test_ids(
        record_text_df=record_text_df, test_size=TEST_SIZE, random_seed=RANDOM_SEED
    )

    train_text_df = record_text_df[record_text_df["id"].isin(train_ids)].copy()
    test_text_df = record_text_df[record_text_df["id"].isin(test_ids)].copy()

    train_expected_df = build_expected_df(
        record_labeled_df[record_labeled_df["id"].isin(train_ids)].copy()
    )
    test_expected_df = build_expected_df(
        record_labeled_df[record_labeled_df["id"].isin(test_ids)].copy()
    )

    scorer = config_obj.data_preprocessing.scoring.scorer
    if scorer is None or not hasattr(scorer, "keywords"):
        raise ValueError(
            "Config must define a KeywordScorer with a non-empty keywords list."
        )
    candidate_keywords: list[str] = list(scorer.keywords)

    selection_df = _greedy_keyword_selection(
        base_config=config_obj,
        candidate_keywords=candidate_keywords,
        train_text_df=train_text_df,
        train_expected_df=train_expected_df,
        test_text_df=test_text_df,
        test_expected_df=test_expected_df,
        perf_sample_size=PERF_SAMPLE_SIZE,
        cost_est_sample_size=COST_EST_SAMPLE_SIZE,
    )

    selection_df.to_csv(SELECTION_RESULTS_CSV, index=False)

    if not selection_df.empty:
        # ICLR-friendly base style
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.rcParams.update({
            "figure.figsize": (3.0, 2.0),
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        })
        color_palette = sns.color_palette("colorblind")

        selection_df_sorted_train = selection_df.sort_values("train_cost_pct")
        selection_df_sorted_test = selection_df.sort_values("test_cost_pct")

        # Raw frontier (cost % vs recall)
        fig1 = plt.figure()
        sns.lineplot(
            x=selection_df_sorted_train["train_cost_pct"],
            y=selection_df_sorted_train["train_recall"],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=3.5,
            color=color_palette[0],
            label="Train",
        )
        sns.lineplot(
            x=selection_df_sorted_test["test_cost_pct"],
            y=selection_df_sorted_test["test_recall"],
            marker="s",
            linestyle="--",
            linewidth=1.2,
            markersize=3.5,
            color=color_palette[1],
            label="Test",
        )
        plt.xlabel("Cost (% of max)")
        plt.ylabel("Recall (good)")
        plt.title("Pareto frontier")
        plt.legend(loc="lower right", frameon=True)
        fig1.savefig(PARETO_FIG_PATH, dpi=300)
        fig1.savefig(PARETO_FIG_PDF_PATH)
        plt.close(fig1)

        # Degree-2 concave smoothing (with points)
        fig2 = plt.figure()
        sns.scatterplot(
            x=selection_df["train_cost_pct"],
            y=selection_df["train_recall"],
            marker="o",
            s=12,
            color=color_palette[0],
            label="Train pts",
        )
        sns.scatterplot(
            x=selection_df["test_cost_pct"],
            y=selection_df["test_recall"],
            marker="s",
            s=12,
            color=color_palette[1],
            label="Test pts",
        )

        x_train = selection_df_sorted_train["train_cost_pct"].to_numpy()
        y_train = selection_df_sorted_train["train_recall"].to_numpy()
        x_test = selection_df_sorted_test["test_cost_pct"].to_numpy()
        y_test = selection_df_sorted_test["test_recall"].to_numpy()

        p_train = _fit_concave_quadratic(x_train, y_train)
        if p_train is not None:
            x_grid_train = np.linspace(x_train.min(), x_train.max(), 200)
            sns.lineplot(
                x=x_grid_train,
                y=p_train(x_grid_train),
                linestyle="-",
                linewidth=1.2,
                color=color_palette[0],
                label="Train deg2",
            )

        p_test = _fit_concave_quadratic(x_test, y_test)
        if p_test is not None:
            x_grid_test = np.linspace(x_test.min(), x_test.max(), 200)
            sns.lineplot(
                x=x_grid_test,
                y=p_test(x_grid_test),
                linestyle="--",
                linewidth=1.2,
                color=color_palette[1],
                label="Test deg2",
            )

        plt.xlabel("Cost (% of max)")
        plt.ylabel("Recall (good)")
        plt.title("Pareto frontier (deg-2, concave)")
        plt.legend(loc="lower right", frameon=True)
        fig2.savefig(PARETO_FIG_DEG2_PATH, dpi=300)
        fig2.savefig(PARETO_FIG_DEG2_PDF_PATH)
        plt.close(fig2)


if __name__ == "__main__":
    main()


