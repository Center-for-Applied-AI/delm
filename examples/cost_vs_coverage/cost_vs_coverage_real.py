"""
Cost vs Coverage (Real LLM) Analysis
===================================

Runs a real DELM pipeline with actual LLM calls to measure cost versus coverage
across keyword-filter sizes, for both train and test partitions.

Workflow: setup -> config -> data import -> data augmentation -> data filtering
-> model fitting/LLM -> export.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from delm import DELM, DELMConfig
from delm.config import (
    DataPreprocessingConfig,
    ScoringConfig,
    SemanticCacheConfig,
)
from delm.constants import (
    SYSTEM_ERRORS_COLUMN,
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN,
)
from delm.strategies.scoring_strategies import KeywordScorer
from delm.schemas.schema_manager import SchemaManager
from delm.utils.post_processing import merge_jsons_for_record
from delm.utils.performance_estimation import (
    _build_required_map,
    _all_levels_precision_recall,
)


RANDOM_SEED = 42
DEFAULT_KEYWORD_MAX = 100
DEFAULT_SAMPLE_SIZE = -1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run real cost vs coverage analysis with DELM using actual LLM calls and keyword filtering."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/commodity_data_large.csv"),
        help=(
            "Path to CSV with a ‘text’ column and a unique id column. The id column name is set by --id-col."
            " Defaults to data/commodity_data_large.csv."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/cost_vs_coverage/config.yaml"),
        help=(
            "Path to DELM YAML configuration. Ensure llm_extraction.track_cost: true is set."
            " Defaults to examples/cost_vs_coverage/config.yaml."
        ),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("examples/commodity_schema.yaml"),
        help=(
            "Path to schema spec YAML referenced by the pipeline to drive structured extraction."
            " Defaults to examples/commodity_schema.yaml."
        ),
    )
    # Labels are expected in the same --data CSV via --expected-col
    parser.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="Name of the unique record id column in the data file.",
    )
    parser.add_argument(
        "--expected-col",
        type=str,
        default="expected_json",
        help=(
            "Name of the ground-truth JSON column in the data file. If missing, it will be built"
            " from columns matching the schema."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Optional cap per split. Use -1 for all records in the split. Sampling is reproducible."
        ),
    )
    parser.add_argument(
        "--keyword-max",
        type=int,
        default=DEFAULT_KEYWORD_MAX,
        help="Maximum number of keywords to evaluate for keyword filtering.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("examples/cost_vs_coverage"),
        help=(
            "Directory to save artifacts: keywords, CSV, plots, and experiment data."
            " Defaults to examples/cost_vs_coverage."
        ),
    )
    return parser.parse_args()


def ensure_paths(outdir: Path) -> Dict[str, Path]:
    """Create output directories and return standard artifact paths.

    Args:
        outdir: Base path for artifacts.

    Returns:
        Mapping of artifact name to path.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "keywords_txt": outdir / "top_100_keywords.txt",
        "results_csv": outdir / "cost_vs_coverage_results.csv",
        "results_pdf": outdir / "cost_vs_coverage_results.pdf",
        "results_svg": outdir / "cost_vs_coverage_results.svg",
        "experiments_dir": outdir / "experiments",
        "cache_dir": outdir / "cache",
        "split_indices_json": outdir / "split_indices.json",
    }
    artifacts["experiments_dir"].mkdir(parents=True, exist_ok=True)
    artifacts["cache_dir"].mkdir(parents=True, exist_ok=True)
    return artifacts


def load_and_enforce_config(config_path: Path, schema_path: Path, cache_dir: Path) -> DELMConfig:
    """Load DELM config and enforce cost tracking and persistent semantic cache.

    Args:
        config_path: Path to pipeline configuration YAML.
        schema_path: Path to schema specification YAML.
        cache_dir: Directory for the SQLite semantic cache.

    Returns:
        A validated DELMConfig instance.
    """
    cfg = DELMConfig.from_yaml(config_path)
    cfg.llm_extraction.track_cost = True
    cfg.schema.spec_path = schema_path
    cache_cfg = SemanticCacheConfig.from_dict(
        {
            "backend": "sqlite",
            "path": str(cache_dir),
            "max_size_mb": cfg.semantic_cache.max_size_mb,
            "synchronous": cfg.semantic_cache.synchronous,
        }
    )
    cfg.semantic_cache = cache_cfg
    cfg.validate()
    return cfg


def ensure_api_key(provider: str, dotenv_path: Path | None) -> None:
    """Ensure the required API key is present for the given provider.

    Args:
        provider: LLM provider name.
        dotenv_path: Optional path to a .env file to load.

    Raises:
        RuntimeError: If the expected API key is not found.
    """
    if dotenv_path is not None and Path(dotenv_path).exists():
        load_dotenv(dotenv_path)
    provider_lower = provider.lower()
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "google-generativeai": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
    }
    if provider_lower not in mapping:
        raise RuntimeError(f"Unsupported provider for env check: {provider}")
    key_name = mapping[provider_lower]
    if os.environ.get(key_name) in (None, ""):
        raise RuntimeError(
            f"Missing API key: {key_name}. Set it in the environment or reference a .env in the config."
        )


def read_csv_required(path: Path) -> pd.DataFrame:
    """Read a CSV file and fail if it does not exist.

    Args:
        path: CSV path.

    Returns:
        Loaded DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def validate_dataframes(
    data_df: pd.DataFrame,
    id_col: str,
    text_col: str,
) -> None:
    """Validate required columns in data and labels.

    Args:
        data_df: Source data DataFrame.
        labels_df: Ground truth DataFrame.
        id_col: Record id column name.
        text_col: Text column name.
        expected_col: Ground truth JSON column name.
    """
    for col in [id_col, text_col]:
        if col not in data_df.columns:
            raise ValueError(
                f"Missing required column in data: {col}. Available: {list(data_df.columns)}"
            )
    return None


def _coerce_bool(value: Any) -> bool | None:
    """Coerce value to boolean when possible.

    Args:
        value: Input value.

    Returns:
        Bool or None if cannot coerce.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    return None


def build_expected_from_columns(
    data_df: pd.DataFrame,
    id_col: str,
    schema_path: Path,
    expected_col: str,
) -> pd.DataFrame:
    """Build expected JSON column from flat columns using the schema specification.

    Args:
        data_df: Input data DataFrame.
        id_col: Record id column name.
        schema_path: Path to schema spec YAML.
        expected_col: Name of the output expected JSON column.

    Returns:
        DataFrame with columns [id_col, expected_col].
    """
    spec = SchemaManager._load_schema_spec(schema_path)
    schema_type = (spec.get("type") or spec.get("schema_type") or "").lower()
    if schema_type not in {"nested", "simple"}:
        raise ValueError(
            f"Automatic expected JSON building is only supported for simple or nested schemas. Got: {schema_type}"
        )
    if schema_type == "nested":
        container = spec.get("container_name")
        if not container:
            raise ValueError("Nested schema missing container_name in spec.")
        variables = [v.get("name") for v in spec.get("variables", [])]

        def row_to_expected(row: pd.Series) -> dict:
            item: dict[str, Any] = {}
            for name in variables:
                if name not in row.index:
                    continue
                val = row[name]
                if name == "price_expectation":
                    coerced = _coerce_bool(val)
                    if coerced is not None:
                        item[name] = coerced
                        continue
                if pd.isna(val):
                    continue
                item[name] = val
            return {container: [item]} if item else {container: []}

        expected = data_df.apply(row_to_expected, axis=1)
        return pd.DataFrame({id_col: data_df[id_col].values, expected_col: expected.values})

    variables = [v.get("name") for v in spec.get("variables", [])]

    def row_to_expected_simple(row: pd.Series) -> dict:
        d: dict[str, Any] = {}
        for name in variables:
            if name not in row.index:
                continue
            val = row[name]
            if pd.isna(val):
                continue
            d[name] = val
        return d

    expected = data_df.apply(row_to_expected_simple, axis=1)
    return pd.DataFrame({id_col: data_df[id_col].values, expected_col: expected.values})


def drop_na_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Drop rows with missing text.

    Args:
        df: Input DataFrame.
        text_col: Text column name.

    Returns:
        Filtered DataFrame.
    """
    return df[df[text_col].notna()].copy()


def persist_split(out_path: Path, train_ids: Sequence[Any], test_ids: Sequence[Any]) -> None:
    """Persist train/test ids for reproducibility.

    Args:
        out_path: JSON path to write.
        train_ids: Train id values.
        test_ids: Test id values.
    """
    payload = {"train_ids": list(train_ids), "test_ids": list(test_ids)}
    out_path.write_text(json.dumps(payload, indent=2))


def has_non_empty_value(obj: Any) -> bool:
    """Return True if the JSON-like object contains any non-empty value.

    Args:
        obj: JSON-like object.

    Returns:
        Whether the object contains any non-empty value.
    """
    if obj is None:
        return False
    if isinstance(obj, str):
        return obj.strip() != ""
    if isinstance(obj, (int, float, bool)):
        return True
    if isinstance(obj, list):
        return any(has_non_empty_value(v) for v in obj)
    if isinstance(obj, dict):
        return any(has_non_empty_value(v) for v in obj.values())
    return False


def prepare_keyword_training_frame(
    train_df: pd.DataFrame, labels_df: pd.DataFrame, id_col: str, text_col: str, expected_col: str
) -> pd.DataFrame:
    """Join train data with labels for keyword discovery.

    Args:
        train_df: Training split DataFrame.
        labels_df: Ground-truth DataFrame.
        id_col: Record id column name.
        text_col: Text column name.
        expected_col: Ground truth JSON column name.

    Returns:
        DataFrame with text and a binary label column ‘is_positive’.
    """
    merged = pd.merge(
        train_df[[id_col, text_col]],
        labels_df[[id_col, expected_col]],
        on=id_col,
        how="inner",
    )
    parsed = merged.copy()
    if parsed[expected_col].dtype == "object" and isinstance(parsed[expected_col].iloc[0], str):
        parsed[expected_col] = parsed[expected_col].apply(json.loads)
    parsed["is_positive"] = parsed[expected_col].apply(has_non_empty_value).astype(int)
    return parsed[[id_col, text_col, "is_positive"]]


def discover_keywords(
    train_frame: pd.DataFrame,
    text_col: str,
    label_col: str,
    max_features: int = 2000,
    select_top_k: int = 500,
    keyword_max: int = DEFAULT_KEYWORD_MAX,
) -> List[str]:
    """Discover discriminative keywords using TF-IDF and chi2 feature selection.

    Args:
        train_frame: Training DataFrame with text and label columns.
        text_col: Name of the text column.
        label_col: Name of the binary label column.
        max_features: Maximum TF-IDF vocabulary size.
        select_top_k: Maximum number of features to select by chi2.
        keyword_max: Maximum number of keywords to return.

    Returns:
        List of top keywords ranked by chi2 score and constrained to positive association.
    """
    texts = train_frame[text_col].astype(str).tolist()
    y = train_frame[label_col].astype(int).values
    if len(texts) == 0:
        return []
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 1),
    )
    X = vectorizer.fit_transform(texts)
    vocab_size = X.shape[1]
    # Fallback if labels are single-class
    if y.sum() == 0 or y.sum() == len(y):
        if vocab_size == 0:
            return []
        means = np.asarray(X.mean(axis=0)).ravel()
        feature_names = np.array(vectorizer.get_feature_names_out())
        ranked_idx = np.argsort(means)[::-1]
        ranked_features = feature_names[ranked_idx]
        return ranked_features[: min(keyword_max, len(ranked_features))].tolist()
    k = min(select_top_k, vocab_size) if vocab_size > 0 else 0
    if k == 0:
        return []
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    scores = selector.scores_
    feature_names = np.array(vectorizer.get_feature_names_out())
    pos_mask = np.asarray(X[y == 1].sum(axis=0)).ravel() > 0
    ranked_idx = np.argsort(scores)[::-1]
    ranked_features = feature_names[ranked_idx]
    ranked_mask = pos_mask[ranked_idx]
    positives = [kw for kw, ok in zip(ranked_features, ranked_mask) if ok]
    return positives[: min(keyword_max, len(positives))]


def build_preprocessing_config(
    base_cfg: DELMConfig,
    use_keywords: Sequence[str] | None,
) -> DataPreprocessingConfig:
    """Build a DataPreprocessingConfig for baseline or keyword-filtered runs.

    Args:
        base_cfg: The base DELMConfig.
        use_keywords: Keywords for KeywordScorer, or None for baseline.

    Returns:
        DataPreprocessingConfig ready for DELM.
    """
    if use_keywords is None or len(use_keywords) == 0:
        return DataPreprocessingConfig(
            target_column=base_cfg.data_preprocessing.target_column,
            drop_target_column=base_cfg.data_preprocessing.drop_target_column,
            splitting=base_cfg.data_preprocessing.splitting,
            scoring=ScoringConfig(),
            pandas_score_filter=None,
        )
    return DataPreprocessingConfig(
        target_column=base_cfg.data_preprocessing.target_column,
        drop_target_column=base_cfg.data_preprocessing.drop_target_column,
        splitting=base_cfg.data_preprocessing.splitting,
        scoring=ScoringConfig(scorer=KeywordScorer(list(use_keywords))),
        pandas_score_filter="delm_score > 0",
    )


def merge_extractions_by_record(
    results_df: pd.DataFrame,
    id_col: str,
    extraction_schema,
) -> pd.DataFrame:
    """Aggregate chunk-level extractions into a single JSON per record.

    Args:
        results_df: DELM extraction results DataFrame.
        id_col: Original record id column name.
        extraction_schema: Schema object used for merging.

    Returns:
        DataFrame with columns [id_col, extracted_dict].
    """
    grouped = (
        results_df[[id_col, SYSTEM_EXTRACTED_DATA_JSON_COLUMN]]
        .groupby(id_col)[SYSTEM_EXTRACTED_DATA_JSON_COLUMN]
        .apply(lambda js: merge_jsons_for_record(list(js), extraction_schema))
        .reset_index()
    )
    grouped.columns = [id_col, "extracted_dict"]
    return grouped


def compute_micro_metrics(
    extracted_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    id_col: str,
    expected_col: str,
    extraction_schema,
) -> Tuple[float, float, float, int, int, int]:
    """Compute micro-averaged precision, recall, f1, and confusion counts.

    Args:
        extracted_df: DataFrame with [id_col, extracted_dict].
        labels_df: DataFrame with [id_col, expected_col].
        id_col: Record id column name.
        expected_col: Ground truth JSON column name.
        extraction_schema: Schema object.

    Returns:
        Tuple of (precision, recall, f1, tp, fp, fn) using micro-averaging.
    """
    merged = pd.merge(
        labels_df[[id_col, expected_col]], extracted_df[[id_col, "extracted_dict"]], on=id_col, how="inner"
    )
    gt = merged[expected_col].tolist()
    pr = merged["extracted_dict"].tolist()
    req_map = _build_required_map(extraction_schema)
    from collections import defaultdict

    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for y_true, y_pred in zip(gt, pr):
        rec = _all_levels_precision_recall(y_true, y_pred, req_map)
        for field, counts in rec.items():
            agg[field]["tp"] += counts["tp"]
            agg[field]["fp"] += counts["fp"]
            agg[field]["fn"] += counts["fn"]
    tp = sum(v["tp"] for v in agg.values())
    fp = sum(v["fp"] for v in agg.values())
    fn = sum(v["fn"] for v in agg.values())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return precision, recall, f1, int(tp), int(fp), int(fn)


def run_delm_variant(
    dataset_name: str,
    partition_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    base_cfg: DELMConfig,
    experiments_dir: Path,
    id_col: str,
    expected_col: str,
    sample_size: int,
    keywords: Sequence[str] | None,
    keyword_size: int | None,
) -> Tuple[float, Dict[str, Any]]:
    """Run a DELM extraction with a specific preprocessing configuration.

    Args:
        dataset_name: Split name.
        partition_df: DataFrame of the split.
        labels_df: Ground truth labels.
        base_cfg: Loaded and enforced DELM config.
        experiments_dir: Base directory for experiments.
        id_col: Record id column name.
        expected_col: Ground truth JSON column name.
        sample_size: Optional cap per split.
        keywords: Keyword list for filtering, or None for baseline.
        keyword_size: Number of top keywords to use for this run.

    Returns:
        Tuple of (total_cost, metrics_dict).
    """
    if keywords is None or keyword_size is None:
        variant_name = f"{dataset_name}_baseline"
        use_keywords: Sequence[str] | None = None
    else:
        variant_name = f"{dataset_name}_kw_{keyword_size}"
        use_keywords = list(keywords[:keyword_size])

    dp_cfg = build_preprocessing_config(base_cfg, use_keywords)
    cfg_variant = replace(base_cfg, data_preprocessing=dp_cfg)
    exp_dir = experiments_dir
    delm = DELM(
        config=cfg_variant,
        experiment_name=f"cost_coverage_{variant_name}",
        experiment_directory=exp_dir,
        overwrite_experiment=True,
        auto_checkpoint_and_resume_experiment=True,
        use_disk_storage=True,
    )
    prepped = delm.prep_data(partition_df, sample_size=sample_size)
    if len(prepped) == 0:
        raise RuntimeError("No data to process after preprocessing.")
    results = delm.process_via_llm()
    if SYSTEM_EXTRACTED_DATA_JSON_COLUMN not in results.columns:
        raise RuntimeError("Missing extracted JSON column in results.")
    if SYSTEM_ERRORS_COLUMN in results.columns:
        if results[SYSTEM_ERRORS_COLUMN].fillna("").str.contains("Over budget").any():
            raise RuntimeError("Budget exceeded during extraction.")
    cost_summary = delm.get_cost_summary()
    total_cost = float(cost_summary.get("total_cost", 0.0))
    extraction_schema = delm.schema_manager.get_extraction_schema()
    extracted_df = merge_extractions_by_record(results, id_col=id_col, extraction_schema=extraction_schema)
    labels_for_split = labels_df[labels_df[id_col].isin(partition_df[id_col])].copy()
    if labels_for_split[expected_col].dtype == "object" and isinstance(
        labels_for_split[expected_col].iloc[0], str
    ):
        labels_for_split[expected_col] = labels_for_split[expected_col].apply(json.loads)
    precision, recall, f1, tp, fp, fn = compute_micro_metrics(
        extracted_df, labels_for_split, id_col, expected_col, extraction_schema
    )
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    return total_cost, metrics


def plot_coverage_vs_cost(
    results_df: pd.DataFrame,
    out_pdf: Path,
    out_svg: Path,
) -> None:
    """Plot coverage versus cost for train and test datasets.

    Args:
        results_df: Results DataFrame.
        out_pdf: PDF output path.
        out_svg: SVG output path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"train": "tab:blue", "test": "tab:green"}
    for dataset in ["train", "test"]:
        sub = results_df[results_df["dataset"] == dataset].copy()
        sub = sub.sort_values("keyword_size")
        ax.plot(
            sub["cost_pct_baseline"],
            sub["coverage_pct"],
            marker="o",
            label=dataset,
            color=colors.get(dataset, None),
        )
        for n in [0, 1, 5, 10, 20, 50, 100]:
            if n in set(sub["keyword_size"].tolist()):
                row = sub[sub["keyword_size"] == n].iloc[0]
                ax.annotate(
                    f"{dataset}:{n}",
                    (row["cost_pct_baseline"], row["coverage_pct"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
    ax.set_xlabel("Cost (% baseline)")
    ax.set_ylabel("Coverage (% recall)")
    ax.set_title("Cost vs Coverage across keyword filters")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_svg)


def main() -> None:
    """Entry point for the real cost vs coverage analysis script."""
    args = parse_args()

    artifacts = ensure_paths(args.outdir)
    data_df = read_csv_required(args.data)
    validate_dataframes(data_df, args.id_col, "text")
    data_df = drop_na_text(data_df, "text")

    cfg = load_and_enforce_config(args.config, args.schema, artifacts["cache_dir"]) 
    ensure_api_key(cfg.llm_extraction.provider, cfg.llm_extraction.dotenv_path)
    cfg.data_preprocessing.target_column = "text"
    cfg.validate()

    if args.expected_col in data_df.columns:
        labels_df = data_df[[args.id_col, args.expected_col]].copy()
    else:
        labels_df = build_expected_from_columns(
            data_df=data_df,
            id_col=args.id_col,
            schema_path=args.schema,
            expected_col=args.expected_col,
        )

    train_df, test_df = train_test_split(
        data_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=None,
    )
    persist_split(
        artifacts["split_indices_json"], train_df[args.id_col].tolist(), test_df[args.id_col].tolist()
    )

    train_kw_frame = prepare_keyword_training_frame(
        train_df=train_df,
        labels_df=labels_df,
        id_col=args.id_col,
        text_col="text",
        expected_col=args.expected_col,
    )
    keywords = discover_keywords(
        train_kw_frame,
        text_col="text",
        label_col="is_positive",
        max_features=2000,
        select_top_k=500,
        keyword_max=args.keyword_max,
    )
    Path(artifacts["keywords_txt"]).write_text("\n".join(keywords[: min(100, len(keywords))]))

    rows: List[Dict[str, Any]] = []

    for dataset_name, part_df in [("train", train_df), ("test", test_df)]:
        baseline_cost, baseline_metrics = run_delm_variant(
            dataset_name=dataset_name,
            partition_df=part_df,
            labels_df=labels_df,
            base_cfg=cfg,
            experiments_dir=artifacts["experiments_dir"],
            id_col=args.id_col,
            expected_col=args.expected_col,
            sample_size=args.sample_size,
            keywords=None,
            keyword_size=None,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "keyword_size": 0,
                "filtered_cost": baseline_cost,
                "baseline_cost": baseline_cost,
                "cost_savings": 0.0,
                "coverage": baseline_metrics["recall"],
                "precision": baseline_metrics["precision"],
                "f1": baseline_metrics["f1"],
                "tp": baseline_metrics["tp"],
                "fp": baseline_metrics["fp"],
                "fn": baseline_metrics["fn"],
            }
        )

        max_n = min(args.keyword_max, len(keywords))
        if max_n > 0:
            for n in tqdm(
        range(1, max_n + 1), desc=f"{dataset_name} keyword runs", total=max_n
            ):
                filt_cost, filt_metrics = run_delm_variant(
                    dataset_name=dataset_name,
                    partition_df=part_df,
                    labels_df=labels_df,
                    base_cfg=cfg,
                    experiments_dir=artifacts["experiments_dir"],
                    id_col=args.id_col,
                    expected_col=args.expected_col,
                    sample_size=args.sample_size,
                    keywords=keywords,
                    keyword_size=n,
                )
                savings = 0.0 if baseline_cost == 0 else 1.0 - (filt_cost / baseline_cost)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "keyword_size": n,
                        "filtered_cost": filt_cost,
                        "baseline_cost": baseline_cost,
                        "cost_savings": savings,
                        "coverage": filt_metrics["recall"],
                        "precision": filt_metrics["precision"],
                        "f1": filt_metrics["f1"],
                        "tp": filt_metrics["tp"],
                        "fp": filt_metrics["fp"],
                        "fn": filt_metrics["fn"],
                    }
                )

    results_df = pd.DataFrame(rows)
    if len(results_df) == 0:
        raise RuntimeError("No results to save.")
    results_df["cost_pct_baseline"] = (
        results_df["filtered_cost"] / results_df["baseline_cost"].replace(0, np.nan)
    ) * 100.0
    results_df["cost_pct_baseline"] = results_df["cost_pct_baseline"].fillna(0.0)
    results_df["coverage_pct"] = results_df["coverage"] * 100.0
    results_df.to_csv(artifacts["results_csv"], index=False)
    plot_coverage_vs_cost(results_df, artifacts["results_pdf"], artifacts["results_svg"])


if __name__ == "__main__":
    main()


