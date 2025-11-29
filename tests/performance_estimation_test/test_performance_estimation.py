import os
import pandas as pd
from pandas.io.common import Path
import yaml
from pprint import pprint
from dotenv import load_dotenv

from delm import DELM, Schema
from delm.models import ExtractionVariable
from delm.utils.performance_estimation import estimate_performance

load_dotenv(".env")

DIR = "tests/performance_estimation_test"
INPUT_DATA_FILE = "input_data.csv"

SCHEMA_FILES = [
    "simple_schema.yaml",
    "nested_schema.yaml",
    "multiple_schema.yaml",
    "deeply_nested_multiple_schema.yaml",
]
EXPECTED_FILES = [
    "expected_simple.csv",
    "expected_nested.csv",
    "expected_multiple.csv",
    "expected_deeply_nested_multiple.csv",
]

MATCHING_ID_COLUMN = "record_id"

# Show all columns
pd.set_option("display.max_columns", None)
# Show all rows
pd.set_option("display.max_rows", None)
# Don't truncate wide column content
pd.set_option("display.max_colwidth", None)
# Expand the frame across the full width of the terminal
pd.set_option("display.width", None)


def create_schemas():
    """Create all schema objects in Python."""
    schemas = {}

    # Simple schema
    schemas["simple_schema.yaml"] = Schema.simple(
        variables_list=[
            ExtractionVariable(
                name="author",
                description="Main author of the book",
                data_type="string",
                required=True,
            ),
            ExtractionVariable(
                name="book_title",
                description="Title of the book",
                data_type="string",
                required=True,
            ),
        ],
    )

    # Nested schema
    schemas["nested_schema.yaml"] = Schema.nested(
        container_name="books",
        variables_list=[
            ExtractionVariable(
                name="title",
                description="Title of the book",
                data_type="string",
                required=True,
            ),
            ExtractionVariable(
                name="copies_sold",
                description="Number of copies sold",
                data_type="integer",
            ),
            ExtractionVariable(
                name="price",
                description="Price per copy",
                data_type="number",
            ),
        ],
    )

    # Multiple schema (simple schemas)
    schemas["multiple_schema.yaml"] = Schema.multiple(
        book=Schema.simple(
            variables_list=[
                ExtractionVariable(
                    name="author",
                    description="Main author of the book",
                    data_type="string",
                    required=True,
                ),
                ExtractionVariable(
                    name="title",
                    description="Title of the book",
                    data_type="string",
                    required=True,
                ),
            ],
        ),
        sales_event=Schema.simple(
            variables_list=[
                ExtractionVariable(
                    name="event_name",
                    description="Name of the sales event",
                    data_type="string",
                ),
                ExtractionVariable(
                    name="season",
                    description="Season of the event",
                    data_type="string",
                ),
            ],
        ),
    )

    # Deeply nested multiple schema
    schemas["deeply_nested_multiple_schema.yaml"] = Schema.multiple(
        books=Schema.nested(
            container_name="entries",
            variables_list=[
                ExtractionVariable(
                    name="title",
                    description="Title of the book",
                    data_type="string",
                    required=True,
                ),
                ExtractionVariable(
                    name="author",
                    description="Author of the book",
                    data_type="string",
                    required=True,
                ),
                ExtractionVariable(
                    name="sales",
                    description="Sales info",
                    data_type="[integer]",
                ),
            ],
        ),
        sales_events=Schema.nested(
            container_name="events",
            variables_list=[
                ExtractionVariable(
                    name="event_name",
                    description="Name of the sales event",
                    data_type="string",
                    required=True,
                ),
                ExtractionVariable(
                    name="season",
                    description="Season of the event",
                    data_type="string",
                ),
            ],
        ),
    )

    return schemas


def run_performance_test(schema_file, expected_file):
    print("=" * 60)
    print("Performance Estimation Test: Paragraph Splitting & Keyword Scoring")
    print("Components Tested:")
    print("- DELM with RegexSplit (sentence splitting) and KeywordScorer")
    print("Expected Outputs:")
    print("- Per-sentence extraction results, merged per record")
    print(f"=" * 60)
    print("\n")

    # Get schema for this test
    schemas = create_schemas()
    schema = schemas[schema_file]

    # Create DELM instance with config
    delm = DELM(
        schema=schema,
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        base_delay=0.1,
        track_cost=False,
        target_column="text",
        splitting_strategy={
            "type": "RegexSplit",
            "pattern": r"(?<=[.!?])\s+",
        },
        relevance_scorer={
            "type": "KeywordScorer",
            "keywords": ["sale", "copies", "author", "price", "book", "title"],
        },
    )

    # Load input and expected
    input_df = pd.read_csv(Path(DIR) / INPUT_DATA_FILE)
    expected_df = pd.read_csv(Path(DIR) / expected_file)
    # Convert expected_dict from string to dict
    expected_df["expected_dict"] = expected_df["expected_dict"].apply(eval)
    # Run performance estimation
    metrics, merged_df = estimate_performance(
        delm,
        input_df,
        expected_df,
        true_json_column="expected_dict",
        matching_id_column=MATCHING_ID_COLUMN,
        record_sample_size=5,
    )
    print("-" * 40)
    print("Performance Metrics (Precision and Recall Only)")
    print("-" * 40)
    header = f"{'Field':<30} {'Precision':>10} {'Recall':>10}"
    print(header)
    print("-" * len(header))
    for key, value in metrics.items():
        print(f"{key:<30} {value['precision']:10.3f} {value['recall']:10.3f}")
    print("-" * 40)
    print("Expected:")
    pprint(merged_df["expected_dict"].to_list())
    print("Extracted:")
    pprint(merged_df["extracted_dict"].to_list())
    print("")


def test_all():
    for schema_file, expected_file in zip(SCHEMA_FILES, EXPECTED_FILES):
        run_performance_test(schema_file, expected_file)


if __name__ == "__main__":
    test_all()
