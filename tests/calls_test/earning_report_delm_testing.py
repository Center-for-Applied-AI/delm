"""
Test script for DELM - designed for Jupyter REPL usage
Updated to use YAML configuration file
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

from delm import DELM, DELMConfig

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
report_text_df = load_test_data(TEST_FILE_PATH)

# Initialize DELM with YAML config
print("\nLoading DELM configuration from YAML...")
config = DELMConfig.from_yaml(CONFIG_PATH)

# Initialize DELM with config
delm = DELM(config=config)

print("DELM initialized successfully!")

# Process data with DELM
print("\nPreprocessing data...")
output_df = delm.prep_data(report_text_df)

print(f"Data preprocessed successfully!")
print(f"Prepped Data columns: {list(output_df.columns)}")


# Test LLM extraction
llm_output_df = delm.process_via_llm()

print(f"LLM processing completed!")
print(f"LLM output columns: {list(llm_output_df.columns)}")

if not llm_output_df.empty:
    print("\nLLM Output sample:")
    print(llm_output_df.head())

# The structured DataFrame is now returned directly from process_via_llm()
# No need to call parse_to_dataframe() anymore
structured_df = llm_output_df

if not structured_df.empty:
    print("\nStructured output DataFrame sample:")
    print(structured_df.head())
    
# Final summary
print("\n" + "="*50)
print("EXPERIMENT SUMMARY")
print("="*50)
print(f"Input data shape: {report_text_df.shape}")
print(f"Preprocessed chunks: {len(output_df)}")
print(f"LLM processed chunks: {len(llm_output_df)}")
print(f"Structured output rows: {len(structured_df)}")

print("\nStructured DataFrame info:")
print(structured_df.info())

print("\nLLM Output DataFrame info:")
print(llm_output_df.info())

print("\nEarning report testing complete!")