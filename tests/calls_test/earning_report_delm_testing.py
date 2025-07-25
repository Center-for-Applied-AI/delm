"""
Test script for DELM - designed for Jupyter REPL usage
Updated to use YAML configuration file
"""

from pathlib import Path
import pandas as pd
import json
import sys

from delm import DELM, DELMConfig
from delm.constants import DATA_DIR_NAME, PREPROCESSED_DATA_PREFIX, PREPROCESSED_DATA_SUFFIX, CONSOLIDATED_RESULT_PREFIX, CONSOLIDATED_RESULT_SUFFIX

print(f"="*60)
print("Earning Report DELM Testing with REAL DATA")
print("Components Tested:")
print("- DELMConfig")
print("- DELM")
print("- DELM.prep_data")
print("- DELM.process_via_llm")
print("Expected Outputs:")
print("- Extracted data")
print("- Cost of Test")
print(f"="*60)
print("\n")

# Test configuration
TEST_KEYWORDS = (
    "price",
    "prices", 
    "oil",
    "gas",
    "expect",
    "barrel",
    "ton",
    "used",
    "expectations",
    "using"
)

TEST_FILE_PATH = Path("tests/calls_test/data/input/input2_sample_1000.parquet")
CONFIG_PATH = Path("tests/calls_test/config.yaml")

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
    date_clean = pd.to_datetime(report_text_df["date"].astype(str).apply(lambda x: x[:10]))
    report_text_df["date"] = date_clean
    report_text_df = report_text_df[["report", "date", "title", "subtitle", "firm_name", "text"]]

    print(f"-"*40)
    print("Test data loaded successfully!")
    print(f"Shape: {report_text_df.shape}")
    print(f"Columns: {list(report_text_df.columns)}")
    print(f"-"*40)
    
    return report_text_df

report_text_df = load_test_data(TEST_FILE_PATH, num_rows=100)

config = DELMConfig.from_yaml(CONFIG_PATH)
delm = DELM(
    config=config,
    experiment_name="earning_report_test",
    experiment_directory=Path("./test_experiments"),
    overwrite_experiment=False,
    auto_checkpoint_and_resume_experiment=True,
    use_disk_storage=True,
)
delm.prep_data(report_text_df)
delm.process_via_llm()

print(f"-"*40)
print("Data finished processing")
print(f"-"*40)

result_df = delm.get_extraction_results_df()

cost_summary = delm.get_cost_summary()
print(json.dumps(cost_summary, indent=2))

# The output is JSON by default - let's show how to work with it
print("="*60)
print("VISUALIZE OUTPUT")
print("="*60)

import json

for idx, row in result_df.head(3).iterrows():
    # Print all columns except delm_extracted_data
    for col in result_df.columns:
        if col != "delm_extracted_data_json":
            print(f"{col}: {row[col]}")
    print("delm_extracted_data_json:")
    parsed = json.loads(row["delm_extracted_data_json"]) # type: ignore
    print(json.dumps(parsed, indent=2))
    print("-" * 40)