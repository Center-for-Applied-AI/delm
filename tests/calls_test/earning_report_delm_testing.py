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
    print("Loading test data...")
    report_text_df = pd.read_parquet(file_path).iloc[:num_rows]
    report_text_df = report_text_df.drop(columns=["Unnamed: 0"])

    # The date is given in an inconsistent format, so it is cropped at 10 characters.
    date_clean = pd.to_datetime(report_text_df["date"].astype(str).apply(lambda x: x[:10]))
    report_text_df["date"] = date_clean
    report_text_df = report_text_df[["report", "date", "title", "subtitle", "firm_name", "text"]]

    print("Test data loaded successfully!")
    print(f"Shape: {report_text_df.shape}")
    print(f"Columns: {list(report_text_df.columns)}")
    
    return report_text_df

# Load and prepare test data
report_text_df = load_test_data(TEST_FILE_PATH, num_rows=100)

# Initialize DELM with YAML config
print("\nLoading DELM configuration from YAML...")
config = DELMConfig.from_yaml(CONFIG_PATH)

# Initialize DELM with config, experiment name, and directory
delm = DELM(
    config=config,
    experiment_name="earning_report_test",
    experiment_directory=Path("./test_experiments"),
    overwrite_experiment=False,
    verbose=True,
    auto_checkpoint_and_resume_experiment=True
)

print("DELM initialized successfully!")

# Process data with DELM (run this cell)
delm.prep_data(report_text_df)

print(f"Data preprocessed successfully! It was saved to {delm.experiment_manager.experiment_dir}")
prepped_df = pd.read_feather(delm.experiment_manager.experiment_dir / DATA_DIR_NAME / f"{PREPROCESSED_DATA_PREFIX}{delm.experiment_name}{PREPROCESSED_DATA_SUFFIX}")
print(f"Prepped Data columns: {list(prepped_df.columns)}")

# Process with LLM (no parameters needed - uses constructor config)
delm.process_via_llm()

print(f"LLM processing completed!")
result_df = pd.read_feather(delm.experiment_manager.experiment_dir / DATA_DIR_NAME / f"{CONSOLIDATED_RESULT_PREFIX}{delm.experiment_name}{CONSOLIDATED_RESULT_SUFFIX}")

if not result_df.empty:
    print("\nLLM Output sample:")
    print(result_df.head())

# The output is JSON by default - let's show how to work with it
print("\n" + "="*60)
print("WORKING WITH JSON OUTPUT")
print("="*60)

import json

for idx, row in result_df.head(3).iterrows():
    # Print all columns except delm_extracted_data
    for col in result_df.columns:
        if col != "delm_extracted_data":
            print(f"{col}: {row[col]}")
    print("delm_extracted_data:")
    try:
        parsed = json.loads(str(row["delm_extracted_data"]))
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print(f"(Could not parse as JSON: {e})")
        print(row["delm_extracted_data"])
    print("-" * 40)

print(f"\nThis JSON structure allows you to:")
print("- Access all extracted data in its original structure")
print("- Parse specific fields when needed")
print("- Maintain all relationships between objects")
print("- Handle any schema complexity without data loss") 