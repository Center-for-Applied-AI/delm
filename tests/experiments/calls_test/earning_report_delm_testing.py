"""
Test script for DELM - designed for Jupyter REPL usage
Updated for unified schema system with external YAML config
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

from delm import DELM, ParagraphSplit, KeywordScorer

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

# Paths
SCHEMA_SPEC_PATH = Path("tests/experiments/calls_test/schema_spec.yaml")
DOTENV_PATH = Path(".env")
TEST_FILE_PATH = Path("tests/experiments/calls_test/data/input/input2_sample_1000.parquet")

# Load and prepare test data
print("Loading test data...")
report_text_df = pd.read_parquet(TEST_FILE_PATH).iloc[:2]
report_text_df = report_text_df.drop(columns=["Unnamed: 0"])

# The date is given in an inconsistent format, so it is cropped at 10 characters.
date_clean = pd.to_datetime(report_text_df["date"].astype(str).apply(lambda x: x[:10]))
report_text_df["date"] = date_clean
report_text_df = report_text_df[["report", "date", "title", "subtitle", "firm_name", "text"]]

print("Test data loaded successfully!")
print(f"Shape: {report_text_df.shape}")
print(f"Columns: {list(report_text_df.columns)}")
print("\nFirst few rows:")
print(report_text_df.head())
print("\nData info:")
print(report_text_df.info())

# Initialize DELM with new structure
print("\nInitializing DELM...")
delm = DELM(
    data_source=report_text_df,
    schema_spec_path=SCHEMA_SPEC_PATH,
    experiment_name="calls_test",
    experiments_dir="test-experiments",
    overwrite_experiment=True,
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_retries=3,
    batch_size=1,
    max_workers=1,
    target_column="text",
    drop_target_column=True,
    split_strategy=ParagraphSplit(),
    relevance_scorer=KeywordScorer(TEST_KEYWORDS),
    verbose=True
)

print("DELM initialized successfully!")

# Process data with DELM
print("\nPreprocessing data...")
output_df = delm.prep_data()

print(f"Data preprocessed successfully!")
print(f"Prepped Data columns: {list(output_df.columns)}")


# Test LLM extraction
llm_output_df = delm.process_via_llm()

print(f"LLM processing completed!")
print(f"LLM output columns: {list(llm_output_df.columns)}")

if not llm_output_df.empty:
    print("\nLLM Output sample:")
    print(llm_output_df.head())

# Parse to structured DataFrame
print("\nParsing to structured output DataFrame...")
structured_df = delm.parse_to_dataframe(llm_output_df)

if not structured_df.empty:
    print("\nStructured output DataFrame sample:")
    print(structured_df.head())
    
# Print formatted JSON output
print("\nRaw LLM JSON output (cleaned):")
for idx, row in llm_output_df.head().iterrows():
    response = row["llm_json"]
    print(f"\nResponse {idx}:")
    clean_dict = response.model_dump(mode="json") # type: ignore
    print(json.dumps(clean_dict, indent=2, default=str))

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