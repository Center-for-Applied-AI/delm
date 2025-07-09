"""
Test script for DELM - designed for Jupyter REPL usage
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from DELM import DELM, ParagraphSplit, KeywordScorer

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

DELM_CONFIG_PATH = Path("example.delm_config.yaml")
DOTENV_PATH = Path(".env")
TEST_FILE_PATH = Path("data/input/input2.csv")

# Load and prepare test data
report_text_df = pd.read_csv(TEST_FILE_PATH).iloc[:10]
report_text_df = report_text_df.drop(columns=["Unnamed: 0"])

# The date is given in an inconsistent format, so it is cropped at 10 characters.
date_clean = pd.to_datetime(report_text_df["date"].astype(str).apply(lambda x: x[:10]))
report_text_df["date"] = date_clean
report_text_df = report_text_df[["report", "date", "title", "subtitle", "firm_name", "text"]]

print("Test data loaded:")
print(report_text_df.head())
print(report_text_df.info())
print(report_text_df.columns)

# Initialize DELM
delm = DELM(
    config_path=DELM_CONFIG_PATH, 
    dotenv_path=DOTENV_PATH, 
    split_strategy=ParagraphSplit(),
    relevance_scorer=KeywordScorer(TEST_KEYWORDS)
)

# Process data
output_df = delm.prep_data_from_df(report_text_df, "text")

# Show score distribution
plt.hist(output_df["score"])
plt.title("Relevance Score Distribution")
plt.show()

# Test LLM extraction on first 2 chunks
llm_output_df = delm.process_via_llm(output_df.iloc[:2], verbose=True)

print("\nOutput DataFrame:")
print(output_df.head())
print(output_df.info())
print(output_df.columns)
print(output_df.iloc[0]["text_chunk"])

# Load expected output for comparison
test_output_df = pd.read_excel("data/output/output.xlsx").iloc[:100]
print("\nExpected output format:")
print(test_output_df.head())
print(test_output_df.info())
print(test_output_df.columns)
print(test_output_df.iloc[0]) 