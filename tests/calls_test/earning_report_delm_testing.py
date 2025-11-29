"""
Test script for DELM - designed for Jupyter REPL usage
Updated to use inline schema and config params (new API)
"""

from pathlib import Path
import pandas as pd
import json
import sys
from dotenv import load_dotenv

load_dotenv(".env")
from delm import DELM, Schema, ExtractionVariable

print(f"=" * 60)
print("Earning Report DELM Testing with REAL DATA")
print("Components Tested:")
print("- Schema definition (nested)")
print("- DELM with inline config")
print("- DELM.prep_data")
print("- DELM.process_via_llm")
print("- Budget Halting")
print("Expected Outputs:")
print("- Extracted data")
print("- Cost of Test")
print(f"=" * 60)
print("\n")

# Test configuration
TEST_FILE_PATH = Path("tests/calls_test/data/input/input2_sample_1000.parquet")


def load_test_data(file_path: Path, num_rows: int = 2) -> pd.DataFrame:
    """
    Load and preprocess test data from parquet file.

    Args:
        file_path: Path to the parquet file
        num_rows: Number of rows to load (default: 2)

    Returns:
        Preprocessed DataFrame ready for DELM processing
    """
    report_text_df = pd.read_parquet(file_path).iloc[:num_rows]
    report_text_df = report_text_df.drop(columns=["Unnamed: 0"])

    # The date is given in an inconsistent format, so it is cropped at 10 characters.
    date_clean = pd.to_datetime(
        report_text_df["date"].astype(str).apply(lambda x: x[:10])
    )
    report_text_df["date"] = date_clean
    report_text_df = report_text_df[
        ["report", "date", "title", "subtitle", "firm_name", "text"]
    ]

    print(f"-" * 40)
    print("Test data loaded successfully!")
    print(f"Shape: {report_text_df.shape}")
    print(f"Columns: {list(report_text_df.columns)}")
    print(f"-" * 40)

    return report_text_df


report_text_df = load_test_data(TEST_FILE_PATH, num_rows=100)

# Define schema inline using new API
schema = Schema.nested(
    container_name="commodities",
    variables_list=[
        ExtractionVariable(
            name="commodity_type",
            description="Type of commodity mentioned",
            data_type="string",
            required=True,
            allowed_values=[
                "oil",
                "gas",
                "copper",
                "gold",
                "silver",
                "steel",
                "aluminum",
            ],
        ),
        ExtractionVariable(
            name="price_mention",
            description="Whether a specific price is mentioned",
            data_type="boolean",
            required=False,
        ),
        ExtractionVariable(
            name="price_value",
            description="Numeric price value if mentioned",
            data_type="number",
            required=False,
        ),
        ExtractionVariable(
            name="price_unit",
            description="Unit of the price (e.g., barrel, ton, MMBtu)",
            data_type="string",
            required=False,
        ),
        ExtractionVariable(
            name="expectation_type",
            description="Type of price expectation mentioned",
            data_type="string",
            required=False,
            allowed_values=[
                "forecast",
                "guidance",
                "estimate",
                "projection",
                "outlook",
            ],
        ),
        ExtractionVariable(
            name="company_mention",
            description="Company names mentioned in relation to commodities",
            data_type="string",
            required=False,
        ),
    ],
)

# Create DELM instance with inline config params
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    batch_size=8,
    max_workers=4,
    tokens_per_minute=2000000,
    requests_per_minute=100,
    track_cost=True,
    max_budget=0.004,
    target_column="text",
    drop_target_column=False,
    score_filter="delm_score > 0.5",
    splitting_strategy={"type": "ParagraphSplit"},
    relevance_scorer={
        "type": "KeywordScorer",
        "keywords": ["price", "forecast", "guidance", "estimate", "expectation"],
    },
    use_disk_storage=True,
    experiment_path=Path("./test_experiments/earning_report_test"),
    overwrite_experiment=True,
    auto_checkpoint_and_resume_experiment=True,
)
result_df = delm.extract(report_text_df)

print(f"-" * 40)
print("Data finished processing")
print(f"-" * 40)

cost_summary = delm.get_cost_summary()
print(json.dumps(cost_summary, indent=2))

# The output is JSON by default - let's show how to work with it
print("=" * 60)
print("VISUALIZE OUTPUT")
print("=" * 60)

import json

for idx, row in result_df.head(3).iterrows():
    # Print all columns except delm_extracted_data
    print(row[["delm_record_id", "delm_chunk_id"]])
    print("delm_extracted_data_json:")
    parsed = json.loads(row["delm_extracted_data_json"])  # type: ignore
    print(json.dumps(parsed, indent=2))
    print("-" * 40)
