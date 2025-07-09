"""
Test script for DELM - designed for Jupyter REPL usage
Updated for unified schema system
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from DELM import DELM, ParagraphSplit, KeywordScorer
from schemas import SchemaRegistry

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

# Create config directly in Python (no external file needed)
DELM_CONFIG = {
    "model_name": "gpt-4o-mini",
    "temperature": 0.0,
    "max_retries": 3,
    "batch_size": 10,
    "max_workers": 4,
    "extraction": {
        "schema_type": "nested",
        "container_name": "commodities",
        "variables": [
            {
                "name": "commodity_type",
                "description": "Type of commodity mentioned",
                "data_type": "string",
                "required": True,
                "allowed_values": ["oil", "gas", "copper", "gold", "silver", "steel", "aluminum"]
            },
            {
                "name": "price_mention",
                "description": "Whether a specific price is mentioned",
                "data_type": "boolean",
                "required": False
            },
            {
                "name": "price_value",
                "description": "Numeric price value if mentioned",
                "data_type": "number",
                "required": False
            },
            {
                "name": "price_unit",
                "description": "Unit of the price (e.g., barrel, ton, MMBtu)",
                "data_type": "string",
                "required": False
            },
            {
                "name": "expectation_type",
                "description": "Type of price expectation mentioned",
                "data_type": "string",
                "required": False,
                "allowed_values": ["forecast", "guidance", "estimate", "projection", "outlook"]
            },
            {
                "name": "company_mention",
                "description": "Company names mentioned in relation to commodities",
                "data_type": "string",
                "required": False
            }
        ],
        "prompt_template": """You are assisting a finance professor who expects meticulous and reliable results.

Extract commodity price information from the following text:

{variables}

Text to analyze:
{text}

CRITICAL INSTRUCTIONS:
- ONLY extract commodities that are EXPLICITLY mentioned in the text
- If NO commodities are mentioned, return an empty list []
- Do NOT infer or guess commodity types based on company names or context
- Do NOT extract commodities just because a company might be in the energy sector
- Focus on EXPLICIT mentions of: oil, gas, copper, gold, silver, steel, aluminum
- For each commodity mentioned, create a separate entry with all relevant details
- If a field is not mentioned in the text, leave it as null/None rather than guessing

Examples of what NOT to extract:
- "1-800 CONTACTS" → NOT oil (even though contacts might use oil-based solutions)
- "Apple Inc." → NOT aluminum (even though phones contain aluminum)
- "Bank of America" → NOT gold (even though banks might trade gold)"""
    }
}

DOTENV_PATH = Path(".env")
TEST_FILE_PATH = Path("data/input/input2.parquet")

# Load and prepare test data
report_text_df = pd.read_parquet(TEST_FILE_PATH).iloc[:10]
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
    config_path=None,  # No external config file
    dotenv_path=DOTENV_PATH, 
    split_strategy=ParagraphSplit(),
    relevance_scorer=KeywordScorer(TEST_KEYWORDS)
)

# Set the config manually
delm.config = DELM_CONFIG
delm.extraction_schema = delm.schema_registry.create(DELM_CONFIG['extraction'])

print("DELM initialized successfully!")

# Process data
output_df = delm.prep_data_from_df(report_text_df, "text")

# Show score distribution
plt.figure(figsize=(10, 6))
plt.hist(output_df["score"], bins=20, alpha=0.7, edgecolor='black')
plt.title("Relevance Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.show()

print(f"Score statistics:")
print(f"  Mean: {output_df['score'].mean():.3f}")
print(f"  Median: {output_df['score'].median():.3f}")
print(f"  Std: {output_df['score'].std():.3f}")
print(f"  Min: {output_df['score'].min():.3f}")
print(f"  Max: {output_df['score'].max():.3f}")

# Test LLM extraction on first 2 chunks
print("Testing LLM extraction on first 2 chunks...")
llm_output_df = delm.process_via_llm(output_df.iloc[:2], verbose=True, use_regex_fallback=False)

print(f"LLM processing completed!")
print(f"LLM output shape: {llm_output_df.shape}")

if not llm_output_df.empty:
    print("\nLLM Output sample:")
    print(llm_output_df.head())

# Parse to structured DataFrame
print("\nParsing to structured DataFrame...")
structured_df = delm.parse_to_dataframe(llm_output_df)
print(f"Structured output shape: {structured_df.shape}")

if not structured_df.empty:
    print("\nStructured DataFrame sample:")
    print(structured_df.head())
    
    # Show commodity types found
    if 'commodity_type' in structured_df.columns:
        print(f"\nCommodity types found: {structured_df['commodity_type'].value_counts().to_dict()}")
    
    # Show price mentions
    if 'price_mention' in structured_df.columns:
        price_mentions = structured_df['price_mention'].value_counts()
        print(f"\nPrice mentions: {price_mentions.to_dict()}")
    
    # Show price values
    if 'price_value' in structured_df.columns:
        non_null_prices = structured_df['price_value'].dropna()
        if len(non_null_prices) > 0:
            print(f"\nPrice values found: {non_null_prices.tolist()}")
    
    # Show total instances found
    print(f"\nTotal commodity instances found: {len(structured_df)}")

# Print formatted JSON output
print("\nRaw LLM JSON output (cleaned):")
for idx, row in llm_output_df.head().iterrows():
    response = row["llm_json"]
    print(f"\nResponse {idx}:")
    try:
        if hasattr(response, 'model_dump'):
            # It's a Pydantic model - get clean dict
            clean_dict = response.model_dump(mode="json") # type: ignore
            print(json.dumps(clean_dict, indent=2, default=str))
        elif isinstance(response, dict):
            # It's already a dict
            print(json.dumps(response, indent=2, default=str))
        else:
            print(f"Unknown response type: {type(response)}")
            print(response)
    except Exception as e:
        print(f"Error processing response: {e}")
        print(f"Response type: {type(response)}")
        print(response)

# Test cost tracking
print(f"\nCost summary:")
cost_summary = delm.get_cost_summary()
for key, value in cost_summary.items():
    print(f"  {key}: {value}")

print("\nOutput DataFrame:")
print(output_df.head())
print(output_df.info())
print(output_df.columns)
print(output_df.iloc[0]["text_chunk"])

# Load expected output for comparison
test_output_df = pd.read_parquet("data/output/output.parquet").iloc[:100]
print("\nExpected output format:")
print(test_output_df.head())
print(test_output_df.info())
print(test_output_df.columns)
print(test_output_df.iloc[0]) 


print(structured_df.iloc[0]) 