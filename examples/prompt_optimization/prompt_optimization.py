"""Implements LLM-In-the-Loop PRompt Optimization (LILPRO) using DELM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import dotenv

from delm import DELM, Schema, ExtractionVariable
from delm.utils.performance_estimation import estimate_performance


# ----------------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------------

RANDOM_SEED = 42

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# Load API keys and config from .env at project root (override any existing env vars)
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=True)

SOURCE_DATA_PATH = PROJECT_ROOT / "data" / "commodity_data.csv"

EXPERIMENT_ROOT_DIR = CURRENT_DIR / "experiments" / "prompt_opt"
EXPERIMENT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

CONTAINER_NAME = "commodity_prices"
TARGET_FIELD_KEY = f"{CONTAINER_NAME}.price_expectation"

NUM_BATCHES = 5
SAMPLE_WRONG_EXAMPLES = 10
EVAL_SAMPLE_RATIO = 0.10

# ----------------------------------------------------------------------------
# Schemas & Configs
# ----------------------------------------------------------------------------


def get_base_schema(price_expectation_description: str) -> Schema:
    return Schema.nested(
        CONTAINER_NAME,
        ExtractionVariable(
            name="good",
            data_type="string",
            required=True,
            description="The name of the good or commodity mentioned",
            allowed_values=[
                "silver",
                "gold",
                "soybeans",
                "heating oil",
                "copper",
                "gasoline",
                "natural gas",
                "aluminum",
                "iron ore",
                "corn",
                "cotton",
                "palm",
                "gas",
                "oil",
                "nickel",
                "sugar",
                "cattle",
                "wheat",
                "coal",
                "zinc",
                "coffee",
                "emissions",
                "tin",
                "hogs",
                "cocoa",
                "lead",
                "diesel",
                "uranium",
                "ethanol",
                "platinum",
                "electricity",
                "fuel",
                "energy",
                "other",
            ],
            validate_in_text=True,
        ),
        ExtractionVariable(
            name="good_subtype",
            data_type="string",
            required=False,
            description="Subtype or specific variety of the good if applicable",
        ),
        ExtractionVariable(
            name="price_expectation",
            data_type="boolean",
            required=True,
            description=price_expectation_description,
        ),
        ExtractionVariable(
            name="price_lower",
            data_type="number",
            required=False,
            description="Lower bound of the price range if specified",
        ),
        ExtractionVariable(
            name="price_upper",
            data_type="number",
            required=False,
            description="Upper bound of the price range if specified",
        ),
        ExtractionVariable(
            name="unit",
            data_type="string",
            required=False,
            description="Unit of measurement for the price (e.g., per ton, per barrel, per unit)",
        ),
        ExtractionVariable(
            name="currency",
            data_type="string",
            required=False,
            description="Currency of the price (e.g., USD, EUR, GBP)",
        ),
        ExtractionVariable(
            name="horizon",
            data_type="string",
            required=False,
            description="Time horizon for the price (e.g., Q1 2024, end of year, next quarter)",
        ),
    )


INITIAL_PRICE_EXPECTATION_DESC = (
    "Whether this is a price expectation (future price) or current price"
)

BASE_PROMPT_TEMPLATE = """# Instructions
  Given an excerpt from an investor call transcript, identify and record all instances where a firm representative mentions a definite numeric price for a good. A good is something you can reasonably assume is traded in a market. Ignore instances without a numeric price.

  ## Guidelines

  ### Speaker Verification
  - Ensure the statement comes from a firm representative (e.g., CEO, CFO), not from a third party like an external analyst or an unidentified speaker. The speaker's name and affiliation are often mentioned at the start.
  - Exclude any prices mentioned by external analysts or third parties; only include prices mentioned by firm representatives.

  ### Capture Multiple Instances
  - If a statement contains multiple prices or goods, record each instance separately.
{variables}

{text}"""

OPTIMIZER_SCHEMA = Schema.simple(
    ExtractionVariable(
        name="price_expectation_new_definition",
        description="Corrected definition of what constitutes a price expectation",
        data_type="string",
        required=True,
    )
)

OPTIMIZER_PROMPT_TEMPLATE = """Your task is to refine the definition of the variable "price_expectation" based on 10 wrong examples. The variable is a boolean, so the wrong examples should all have been labeled as opposite. Identify the implicit logic the extractor missed.
{variables}

Current definition of the variable "price_expectation":
{current_definition}

Examples where price_expectation is wrong:
{examples}"""


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def build_expected_df(record_labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Create nested expected JSON per id, aggregating duplicates.

    Args:
        record_labeled_df: DataFrame with columns id and output fields.

    Returns:
        DataFrame with columns id and expected_json.
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

    grouped = (
        labeled_items_df.groupby("id")[output_fields]
        .apply(lambda g: g.to_dict(orient="records"))
        .reset_index(name="items")
    )

    grouped["expected_json"] = grouped["items"].apply(
        lambda items: {CONTAINER_NAME: items}
    )
    return grouped[["id", "expected_json"]]


def _count_price_expectation(items: List[Dict[str, Any]] | None) -> Tuple[int, int]:
    """Return counts of True/False for price_expectation across items."""
    if not items:
        return 0, 0
    true_count = sum(
        1
        for it in items
        if isinstance(it, dict) and it.get("price_expectation") is True
    )
    false_count = sum(
        1
        for it in items
        if isinstance(it, dict) and it.get("price_expectation") is False
    )
    return true_count, false_count


def _extract_items(d: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(d, dict):
        return []
    items = d.get(CONTAINER_NAME)
    return (
        [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []
    )


def _normalize_good(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _good_bool_map(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map normalized good -> boolean or None if conflicting/missing across duplicates."""
    mapping: Dict[str, Any] = {}
    for it in items:
        g_norm = _normalize_good(it.get("good"))
        if not g_norm:
            continue
        val = it.get("price_expectation")
        if not isinstance(val, bool):
            # ignore non-boolean/missing values for determining alignment
            continue
        prev = mapping.get(g_norm, None)
        if prev is None:
            mapping[g_norm] = val
        else:
            if isinstance(prev, bool) and prev != val:
                # conflicting labels for same good in record → mark ambiguous
                mapping[g_norm] = None
    return mapping


def _good_original_map(items: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map normalized good -> first seen original good string (for display)."""
    m: Dict[str, str] = {}
    for it in items:
        g = it.get("good")
        g_norm = _normalize_good(g)
        if g_norm and g_norm not in m:
            m[g_norm] = str(g)
    return m


def find_wrong_price_expectation_records(record_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where price_expectation mismatches for any matched good (id+good).

    A mismatch is when for a given record and good name present in both expected and
    extracted, the boolean values differ. Ambiguous multi-label goods are skipped.
    """
    wrong_records: List[Dict[str, Any]] = []
    for _, row in record_pairs_df.iterrows():
        rid = row.get("id")
        exp_items = _extract_items(row.get("expected_dict"))
        pred_items = _extract_items(row.get("extracted_dict"))
        exp_map = _good_bool_map(exp_items)
        pred_map = _good_bool_map(pred_items)
        exp_orig = _good_original_map(exp_items)
        mismatches: List[Dict[str, Any]] = []

        for g_norm in set(exp_map.keys()) & set(pred_map.keys()):
            e_val = exp_map.get(g_norm)
            p_val = pred_map.get(g_norm)
            if isinstance(e_val, bool) and isinstance(p_val, bool) and e_val != p_val:
                mismatches.append(
                    {
                        "good": exp_orig.get(g_norm, g_norm),
                        "expected_price_expectation": bool(e_val),
                        "predicted_price_expectation": bool(p_val),
                    }
                )

        if mismatches:
            wrong_records.append({"id": rid, "mismatches": mismatches})

    return pd.DataFrame.from_records(wrong_records)


def compute_matched_precision_price_expectation(record_pairs_df: pd.DataFrame) -> float:
    """Compute precision over matched (id+good) pairs for price_expectation.

    Precision here is defined as the fraction of matched goods where the extracted
    boolean equals the expected boolean.
    """
    matched_total = 0
    matched_correct = 0
    for _, row in record_pairs_df.iterrows():
        exp_items = _extract_items(row.get("expected_dict"))
        pred_items = _extract_items(row.get("extracted_dict"))
        exp_map = _good_bool_map(exp_items)
        pred_map = _good_bool_map(pred_items)
        for g_norm in set(exp_map.keys()) & set(pred_map.keys()):
            e_val = exp_map.get(g_norm)
            p_val = pred_map.get(g_norm)
            if isinstance(e_val, bool) and isinstance(p_val, bool):
                matched_total += 1
                if e_val == p_val:
                    matched_correct += 1
    return float(matched_correct) / float(matched_total) if matched_total else 0.0


def annotate_price_expectation_counts(record_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with id and PE true/false counts for expected and predicted.

    Columns: id, exp_true, exp_false, pred_true, pred_false
    """
    df = record_pairs_df.copy()
    counts = df.apply(
        lambda r: (
            _count_price_expectation(_extract_items(r.get("expected_dict"))),
            _count_price_expectation(_extract_items(r.get("extracted_dict"))),
        ),
        axis=1,
    )
    counts_df = pd.DataFrame(
        list(counts), columns=["expected_counts", "predicted_counts"], index=df.index
    )
    out = pd.DataFrame(
        {
            "id": df["id"].tolist(),
            "exp_true": counts_df["expected_counts"].apply(
                lambda x: int(x[0]) if isinstance(x, tuple) else 0
            ),
            "exp_false": counts_df["expected_counts"].apply(
                lambda x: int(x[1]) if isinstance(x, tuple) else 0
            ),
            "pred_true": counts_df["predicted_counts"].apply(
                lambda x: int(x[0]) if isinstance(x, tuple) else 0
            ),
            "pred_false": counts_df["predicted_counts"].apply(
                lambda x: int(x[1]) if isinstance(x, tuple) else 0
            ),
        }
    )
    return out


def compute_batch_stats(
    *,
    record_pairs_df: pd.DataFrame,
    delm_instance: DELM,
    record_text_df: pd.DataFrame,
) -> Dict[str, int]:
    """Compute n_obs, n_chunks, n_extractions, n_extractions_wrong_pe for this batch.

    - n_obs: number of matched records evaluated
    - n_chunks: number of preprocessed chunks for those records using config
    - n_extractions: total number of extracted items across records
    - n_extractions_wrong_pe: count of extracted items whose price_expectation label doesn't
      match expected counts (computed via count differences per record)
    """
    ids = record_pairs_df["id"].tolist()
    n_obs = len(ids)

    # Compute chunks using the same preprocessing config on the matched IDs only
    sample_source_df = record_text_df[record_text_df["id"].isin(ids)].copy()

    delm_tmp = DELM.from_config(
        delm_instance.config,
        experiment_path=EXPERIMENT_ROOT_DIR / "tmp_counts",
        overwrite_experiment=False,
        auto_checkpoint_and_resume_experiment=False,
        use_disk_storage=False,
        save_log_file=False,
        override_logging=False,
    )

    prepped_df = delm_tmp.prep_data(sample_source_df)
    n_chunks = int(len(prepped_df))

    # Total extractions (total predicted items across all records)
    n_extractions = int(
        sum(
            len(_extract_items(d))
            for d in record_pairs_df.get(
                "extracted_dict", pd.Series([{}] * len(record_pairs_df))
            )
        )
    )

    # Wrong price_expectation among matched (id+good) pairs (boolean inequality)
    mismatched_pairs = 0
    for _, row in record_pairs_df.iterrows():
        exp_items = _extract_items(row.get("expected_dict"))
        pred_items = _extract_items(row.get("extracted_dict"))
        exp_map = _good_bool_map(exp_items)
        pred_map = _good_bool_map(pred_items)
        for g_norm in set(exp_map.keys()) & set(pred_map.keys()):
            e_val = exp_map.get(g_norm)
            p_val = pred_map.get(g_norm)
            if isinstance(e_val, bool) and isinstance(p_val, bool) and e_val != p_val:
                mismatched_pairs += 1
    n_extractions_wrong_pe = int(mismatched_pairs)

    return {
        "n_obs": int(n_obs),
        "n_chunks": int(n_chunks),
        "n_extractions": int(n_extractions),
        "n_extractions_wrong_pe": int(n_extractions_wrong_pe),
    }


def append_metrics_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append one row to CSV, creating the file with header if missing."""
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def save_precision_plot(
    csv_path: Path, out_path: Path, series: str = "presence"
) -> None:
    """Render precision-vs-batch plot from CSV with dynamic y-limits.

    series: "presence" to plot estimator precision; "matched" to plot matched_precision
    Only a single line labeled "Precision" is plotted, as requested.
    y-min is (series min - 0.1), y-max is (series max + 0.1).
    """
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "batch" not in df.columns or "precision" not in df.columns:
        return
    # ICLR-friendly style similar to cost_vs_coverage
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
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
        }
    )
    plt.figure()
    if series == "presence":
        y = df["precision"]
    elif series == "matched":
        if "matched_precision" not in df.columns:
            return
        y = df["matched_precision"]
    else:
        return

    y = y.dropna()
    if y.empty:
        return

    sns.lineplot(x=df["batch"], y=y, marker="o", linewidth=2, label="Precision")
    plt.xlabel("Batch number")
    plt.ylabel("Precision")
    ymin = float(y.min()) - 0.1
    ymax = float(y.max()) + 0.1
    plt.ylim(ymin, ymax)
    xticks = sorted(df["batch"].unique().tolist())
    plt.xticks(xticks)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def compose_wrong_examples_text(
    record_text_df: pd.DataFrame, wrong_df: pd.DataFrame, max_examples: int
) -> str:
    """Compact text with goods where price_expectation mismatches for optimizer."""
    if wrong_df.empty:
        return ""
    rng = random.Random(RANDOM_SEED)
    merged = pd.merge(
        wrong_df[["id", "mismatches"]],
        record_text_df,
        on="id",
        how="left",
    )
    ids = merged["id"].tolist()
    rng.shuffle(ids)
    subset = merged.set_index("id").loc[ids[:max_examples]]

    blocks: List[str] = []
    for rid, row in subset.iterrows():
        mm: List[Dict[str, Any]] = row.get("mismatches") or []
        mm_lines = []
        for m in mm:
            mm_lines.append(
                f"- good: {m.get('good')} | expected: {m.get('expected_price_expectation')} | extracted: {m.get('predicted_price_expectation')}"
            )
        mm_text = "\n".join(mm_lines)
        snippet = str(row.get("text", "")).strip().replace("\n\n", "\n").strip()[:800]
        blocks.append(
            f"ID: {rid}\nMismatched goods (price_expectation):\n{mm_text}\nText:\n{snippet}"
        )
    return "\n\n---\n\n".join(blocks)


def run_optimizer_and_get_guidance(
    current_definition: str, examples_text: str
) -> Dict[str, Any]:
    """Run optimizer to produce a refined definition from wrong examples."""

    templ = OPTIMIZER_PROMPT_TEMPLATE
    templ = templ.replace("{current_definition}", current_definition)
    templ = templ.replace("{examples}", examples_text)

    delm = DELM(
        schema=OPTIMIZER_SCHEMA,
        provider="openai",
        model="gpt-4o-mini",  # Changed from gpt-5 to gpt-4o-mini as per delm.py default or available models
        temperature=1.0,
        batch_size=1,
        max_workers=1,
        max_retries=3,
        base_delay=1.0,
        track_cost=True,
        max_budget=10.0,
        target_column="text",
        drop_target_column=False,
        splitting_strategy={"type": "None"},
        relevance_scorer={"type": "None"},
        prompt_template=templ,
        cache_backend="sqlite",
        cache_path=".delm/kirill_cache",
        cache_max_size_mb=100,
        cache_synchronous="normal",
        experiment_path=EXPERIMENT_ROOT_DIR / "optimizer",
        overwrite_experiment=False,
        auto_checkpoint_and_resume_experiment=False,
        use_disk_storage=False,
        override_logging=False,
    )

    df = pd.DataFrame({"id": ["optimizer_input_1"], "text": [""]})
    delm.prep_data(df)
    results_df = delm.process_via_llm()

    json_col = "delm_extracted_data_json"
    if json_col not in results_df.columns:
        raise ValueError("Missing delm_extracted_data_json in optimizer results")

    merged: Dict[str, Any] = {}
    for j in results_df[json_col].tolist():
        obj = json.loads(j) if isinstance(j, str) else j
        if isinstance(obj, dict):
            merged.update(obj)
    return merged


# ----------------------------------------------------------------------------
# main flow
# ----------------------------------------------------------------------------


def main() -> None:
    """Run iterative optimization and plot precision across batches."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    record_labeled_df = pd.read_csv(SOURCE_DATA_PATH)

    record_text_df = (
        record_labeled_df[["id", "text"]]
        .drop_duplicates(subset=["id"], keep="first")
        .copy()
    )

    record_expected_df = build_expected_df(record_labeled_df.copy())

    batch_records: List[Dict[str, Any]] = []

    # Where we incrementally persist batch metrics and plot
    metrics_csv_path = EXPERIMENT_ROOT_DIR / "precision_by_batch.csv"

    # Determine 10% evaluation sample size (at least 1 record)
    eval_record_sample_size = max(
        1, int(np.ceil(EVAL_SAMPLE_RATIO * len(record_expected_df)))
    )

    current_price_expectation_desc = INITIAL_PRICE_EXPECTATION_DESC

    for batch_idx in tqdm(range(NUM_BATCHES + 1), desc="batches", leave=True):

        # Create schema with current description
        current_schema = get_base_schema(current_price_expectation_desc)

        # Initialize DELM with current schema and base config
        delm = DELM(
            schema=current_schema,
            provider="openai",
            model="gpt-4o-mini",  # Changed from gpt-5-mini
            temperature=1.0,
            batch_size=10,
            max_workers=25,
            max_retries=3,
            base_delay=1.0,
            track_cost=True,
            max_budget=50.0,
            target_column="text",
            drop_target_column=False,
            score_filter="delm_score > 0",
            splitting_strategy={"type": "ParagraphSplit"},
            relevance_scorer={
                "type": "KeywordScorer",
                "keywords": [
                    "price",
                    "cost",
                    "market",
                    "commodity",
                    "oil",
                    "gas",
                    "steel",
                    "copper",
                    "aluminum",
                    "gold",
                ],
            },
            prompt_template=BASE_PROMPT_TEMPLATE,
            cache_backend="sqlite",
            cache_path=".delm/kirill_cache",
            cache_max_size_mb=100,
            cache_synchronous="normal",
            # Experiment settings handled by estimate_performance mostly, but we can set defaults here
        )

        exp_dir = EXPERIMENT_ROOT_DIR / f"batch_{batch_idx:02d}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Deterministically resample a new 10% subset per batch
        expected_batch_df = record_expected_df.sample(
            n=eval_record_sample_size, random_state=RANDOM_SEED + batch_idx
        )

        metrics_dict, record_pairs_df = estimate_performance(
            delm_instance=delm,  # Pass instance instead of config
            data_source=record_text_df,
            expected_extraction_output_df=expected_batch_df,
            true_json_column="expected_json",
            matching_id_column="id",
            record_sample_size=-1,  # use all rows of expected_batch_df
            save_file_log=True,
            log_dir=PROJECT_ROOT / "delm_logs" / "prompt_optimization",
        )

        field_metrics = metrics_dict.get(TARGET_FIELD_KEY, {})
        precision = float(field_metrics.get("precision", 0.0))
        matched_precision = compute_matched_precision_price_expectation(record_pairs_df)

        # Collect batch stats
        stats = compute_batch_stats(
            record_pairs_df=record_pairs_df,
            delm_instance=delm,
            record_text_df=record_text_df,
        )

        # Append to in-memory list for reference
        batch_records.append(
            {
                "batch": batch_idx,
                "precision": precision,
                "matched_precision": matched_precision,
                **stats,
            }
        )

        # Persist/update the metrics CSV after each batch
        append_metrics_row(
            metrics_csv_path,
            {
                "batch": batch_idx,
                "precision": precision,
                "matched_precision": matched_precision,
                **stats,
            },
        )

        # Save the per-record trace for price_expectation counts
        pe_trace_df = annotate_price_expectation_counts(record_pairs_df)
        pe_trace_df.insert(0, "batch", batch_idx)
        pe_trace_df.to_csv(exp_dir / "price_expectation_trace.csv", index=False)

        with (exp_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics_dict, fh, ensure_ascii=False, indent=2)

        record_pairs_out_df = record_pairs_df.copy()
        record_pairs_out_df.to_json(
            exp_dir / "record_pairs.json", orient="records", force_ascii=False, indent=2
        )

        # Save or update the precision plots incrementally (PNG + PDF)
        save_precision_plot(
            metrics_csv_path,
            EXPERIMENT_ROOT_DIR / "precision_vs_batch_presence.png",
            series="presence",
        )
        save_precision_plot(
            metrics_csv_path,
            EXPERIMENT_ROOT_DIR / "precision_vs_batch_presence.pdf",
            series="presence",
        )
        save_precision_plot(
            metrics_csv_path,
            EXPERIMENT_ROOT_DIR / "precision_vs_batch_matched.png",
            series="matched",
        )
        save_precision_plot(
            metrics_csv_path,
            EXPERIMENT_ROOT_DIR / "precision_vs_batch_matched.pdf",
            series="matched",
        )

        if batch_idx < NUM_BATCHES:
            wrong_df = find_wrong_price_expectation_records(record_pairs_df)
            if len(wrong_df) == 0:
                continue

            examples_text = compose_wrong_examples_text(
                record_text_df=record_text_df,
                wrong_df=wrong_df,
                max_examples=SAMPLE_WRONG_EXAMPLES,
            )
            (exp_dir / "wrong_examples.txt").write_text(examples_text, encoding="utf-8")

            guidance = run_optimizer_and_get_guidance(
                current_price_expectation_desc, examples_text
            )

            (exp_dir / "optimizer_output.json").write_text(
                json.dumps(guidance, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            new_def = str(guidance.get("price_expectation_new_definition", "")).strip()
            if new_def:
                current_price_expectation_desc = new_def

    # Final plot refresh from accumulated CSV (PNG + PDF)
    save_precision_plot(
        metrics_csv_path,
        EXPERIMENT_ROOT_DIR / "precision_vs_batch_presence.png",
        series="presence",
    )
    save_precision_plot(
        metrics_csv_path,
        EXPERIMENT_ROOT_DIR / "precision_vs_batch_presence.pdf",
        series="presence",
    )
    save_precision_plot(
        metrics_csv_path,
        EXPERIMENT_ROOT_DIR / "precision_vs_batch_matched.png",
        series="matched",
    )
    save_precision_plot(
        metrics_csv_path,
        EXPERIMENT_ROOT_DIR / "precision_vs_batch_matched.pdf",
        series="matched",
    )


if __name__ == "__main__":
    main()
