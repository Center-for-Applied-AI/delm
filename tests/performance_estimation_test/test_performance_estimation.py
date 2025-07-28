import os
import pandas as pd
from pandas.io.common import Path
import yaml
from pprint import pprint
from delm import DELMConfig
from delm.utils.performance_estimation import estimate_performance

DIR = "tests/performance_estimation_test"
INPUT_DATA_FILE = "input_data.csv"

SCHEMA_FILES = [
    "simple_schema.yaml",
    "nested_schema.yaml",
    "multiple_schema.yaml",
    "deeply_nested_multiple_schema.yaml"
]
EXPECTED_FILES = [
    "expected_simple.csv",
    "expected_nested.csv",
    "expected_multiple.csv",
    "expected_deeply_nested_multiple.csv"
]

MATCHING_ID_COLUMN = "record_id"

# Show all columns
pd.set_option('display.max_columns', None)
# Show all rows
pd.set_option('display.max_rows', None)
# Don't truncate wide column content
pd.set_option('display.max_colwidth', None)
# Expand the frame across the full width of the terminal
pd.set_option('display.width', None)

def run_performance_test(schema_file, expected_file):
    print("="*60)
    print("Performance Estimation Test: Paragraph Splitting & Keyword Scoring")
    print("Components Tested:")
    print("- DELM with RegexSplit (sentence splitting) and KeywordScorer")
    print("Expected Outputs:")
    print("- Per-sentence extraction results, merged per record")
    print(f"="*60)
    print("\n")
    # Load config and update schema path
    config_obj = DELMConfig.from_yaml(Path(DIR) / "config.yaml")
    config_obj.schema.spec_path = Path(DIR) / schema_file
    # Load input and expected
    input_df = pd.read_csv(Path(DIR) / INPUT_DATA_FILE)
    expected_df = pd.read_csv(Path(DIR) / expected_file)
    # Convert expected_dict from string to dict
    expected_df["expected_dict"] = expected_df["expected_dict"].apply(eval)
    # Run performance estimation
    metrics, merged_df = estimate_performance(
        config_obj,
        input_df,
        expected_df,
        true_json_column="expected_dict",
        matching_id_column=MATCHING_ID_COLUMN,
        record_sample_size=5
    )
    print("-"*40)
    print("Performance Metrics (Precision and Recall Only)")
    print("-"*40)
    header = f"{'Field':<30} {'Precision':>10} {'Recall':>10}"
    print(header)
    print("-" * len(header))
    for key, value in metrics.items():
        print(f"{key:<30} {value['precision']:10.3f} {value['recall']:10.3f}")
    print("-"*40)
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