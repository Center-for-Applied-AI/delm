import os
import pandas as pd
import yaml
from pprint import pprint
from delm import DELM, DELMConfig

DIR = "tests/performance_estimation_test"

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
    print(f"Performance Estimation Test for Schema: {schema_file}")
    print("="*60)
    # Load config and update schema path
    with open(os.path.join(DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)
    config["schema"]["spec_path"] = os.path.join(DIR, schema_file)
    config_obj = DELMConfig.from_any(config)
    # Load input and expected
    input_df = pd.read_csv(os.path.join(DIR, "input_data.csv"))
    expected_df = pd.read_csv(os.path.join(DIR, expected_file))
    # Convert expected_dict from string to dict
    expected_df["expected_dict"] = expected_df["expected_dict"].apply(eval)
    # Run performance estimation
    metrics, merged_df = DELM.estimate_performance(
        config_obj,
        input_df,
        expected_df,
        true_json_column="expected_dict",
        record_sample_size=10
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
    pprint(merged_df.iloc[:2]["expected_dict"].to_list())
    print("Extracted:")
    pprint(merged_df.iloc[:2]["extracted_dict"].to_list())
    print("")

def test_all():
    for schema_file, expected_file in zip(SCHEMA_FILES, EXPECTED_FILES):
        run_performance_test(schema_file, expected_file)

if __name__ == "__main__":
    test_all() 