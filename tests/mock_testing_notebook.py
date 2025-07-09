"""
Mock testing for DELM - Jupyter REPL version
Run this cell by cell in Jupyter for interactive testing
Updated for unified schema system
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

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
        "prompt_template": """You're assisting a finance professor who expects meticulous and reliable results.

Extract commodity price information from the following text:

{variables}

Text to analyze:
{text}

Focus on:
- Specific commodity types (oil, gas, metals, etc.)
- Price mentions and values
- Price expectations and forecasts
- Companies mentioned in commodity context
- Units of measurement (barrel, ton, MMBtu, etc.)

For each commodity mentioned, create a separate entry with all relevant details.
IMPORTANT: If a field is not mentioned in the text, leave it as null/None rather than guessing."""
    }
}

DOTENV_PATH = Path(".env")

# Create mock data (run this cell first)
np.random.seed(42)  # For reproducible results

# Sample data for generating realistic-looking reports
firms = ["Goldman Sachs", "Morgan Stanley", "JP Morgan", "Barclays", "Deutsche Bank"]
report_types = ["Market Analysis", "Economic Outlook", "Sector Review", "Investment Strategy"]

# Generate dates over the last year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(20)]
dates.sort()

# Create mock text content with varying relevance to keywords
mock_texts = [
    # High relevance texts (should score well) - WITH SCHEMA VALUES
    "WTI crude oil prices are expected to remain volatile in the coming quarter. The barrel price of Brent crude has been fluctuating between $70 and $85, with expectations of further increases due to OPEC supply constraints. XOM and CVX are leading producers.",
    
    "Henry Hub natural gas prices have surged by 15% this month, driven by increased LNG demand and limited pipeline supply. We expect this trend to continue through the winter months. TTF prices in Europe are also rising.",
    
    "The price of industrial metals, particularly steel and aluminum, has shown significant increases. Ton prices have risen by 20% year-over-year, with expectations of continued growth. Production volumes reached 1.2 million tons last quarter.",
    
    "Oil and gas companies like BP and SHEL are using advanced technologies to improve extraction efficiency. The barrel cost of production has decreased by 10% due to these innovations. Light Sweet crude production increased by 5%.",
    
    "Market expectations for commodity prices remain bullish. WTI oil prices are expected to reach $90 per barrel by year-end, while Henry Hub gas prices may stabilize around current levels. JKM LNG prices are showing volatility.",
    
    # Medium relevance texts - WITH SOME SCHEMA VALUES
    "The energy sector continues to show strong performance. Companies like AAPL and MSFT are investing heavily in renewable energy sources while maintaining traditional oil and gas operations. GOOGL has announced new energy initiatives.",
    
    "Commodity markets are experiencing increased volatility. Investors should expect continued price fluctuations across various sectors. AMZN's logistics division is adapting to fuel price changes.",
    
    "Supply chain disruptions are affecting multiple industries. Companies are using alternative suppliers to maintain production levels. Heavy Sour crude availability has been impacted.",
    
    "The transportation sector faces challenges due to fuel price increases. Companies are exploring alternative energy sources to reduce costs. Pipeline capacity constraints are affecting gas distribution.",
    
    "Economic indicators suggest moderate growth expectations. The manufacturing sector shows signs of recovery with increased demand for raw materials. Production volumes are expected to grow by 8% in Q4.",
    
    # Low relevance texts (should score poorly) - BUT WITH SOME SCHEMA VALUES
    "Technology stocks like AAPL and MSFT have outperformed the broader market this quarter. Software companies continue to show strong revenue growth. GOOGL's cloud division reported record earnings.",
    
    "The healthcare sector remains resilient despite economic uncertainties. Pharmaceutical companies are developing innovative treatments. AMZN's healthcare initiatives are gaining traction.",
    
    "Consumer spending patterns have shifted significantly. Retail companies are adapting to changing customer preferences. E-commerce platforms are seeing increased adoption.",
    
    "The real estate market shows signs of stabilization. Property prices in major metropolitan areas are beginning to level off. Investment volumes are expected to remain steady.",
    
    "Financial services companies are expanding their digital offerings. Online banking and mobile payment solutions are gaining popularity. Traditional banks are modernizing their platforms.",
    
    # Additional varied texts - WITH SCHEMA VALUES
    "The agricultural sector faces challenges from climate change. Farmers are using new technologies to improve crop yields. Production volumes for key crops have increased by 12%.",
    
    "International trade agreements are reshaping global supply chains. Companies are adapting their strategies to navigate new regulations. Brent crude imports have been affected by trade policies.",
    
    "The automotive industry is undergoing a major transformation. Electric vehicle adoption is accelerating across all markets. Traditional automakers are investing heavily in new technologies.",
    
    "Renewable energy investments are reaching record levels. Solar and wind power projects are becoming increasingly cost-effective. LNG infrastructure development is expanding globally.",
    
    "The telecommunications sector is experiencing rapid technological change. 5G networks are being deployed across major markets. Infrastructure investment volumes are at all-time highs."
]

# Create the DataFrame
data = []
for i in range(20):
    report_type = np.random.choice(report_types)
    quarter = np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'])
    year = np.random.randint(2022, 2024)
    firm = np.random.choice(firms)
    text = np.random.choice(mock_texts)
    
    data.append({
        "report": f"REP_{(i+1):03d}",
        "date": dates[i],
        "title": f"{report_type} - {quarter} {year}",
        "subtitle": f"Market Analysis Report by {firm}",
        "firm_name": firm,
        "text": text
    })

report_text_df = pd.DataFrame(data)

print("Mock dataset created successfully!")
print(f"Shape: {report_text_df.shape}")
print(f"Columns: {list(report_text_df.columns)}")
print("\nFirst few rows:")
print(report_text_df.head())

# Initialize DELM (run this cell)
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

# Process data with DELM (run this cell)
output_df = delm.prep_data_from_df(report_text_df, "text")

print(f"Data processed successfully!")
print(f"Output shape: {output_df.shape}")
print(f"Output columns: {list(output_df.columns)}")

# Show score distribution (run this cell)
plt.figure(figsize=(10, 6))
plt.hist(output_df["score"], bins=20, alpha=0.7, edgecolor='black')
plt.title("Relevance Score Distribution (Mock Data)")
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

# Show top scoring chunks (run this cell)
print("Top 5 scoring chunks:")
top_chunks = output_df.nlargest(5, 'score')
for i, (idx, row) in enumerate(top_chunks.iterrows()):
    print(f"\n{i+1}. Score: {row['score']:.3f}")
    print(f"   Report: {row['report']}")
    print(f"   Text: {row['text_chunk'][:150]}...")

# Test LLM extraction on top chunks (run this cell)
print("Testing LLM extraction on top 3 chunks...")
top_chunks_for_llm = output_df.nlargest(3, 'score')
# Set use_regex_fallback=False to see only LLM results, no fallback to regex
llm_output_df = delm.process_via_llm(top_chunks_for_llm, verbose=True, use_regex_fallback=False)

print(f"LLM processing completed!")
print(f"LLM output shape: {llm_output_df.shape}")

if not llm_output_df.empty:
    print("\nLLM Output sample:")
    print(llm_output_df.head())

# Parse to structured DataFrame using new schema system
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
    
    # Show price units
    if 'price_unit' in structured_df.columns:
        units = structured_df['price_unit'].value_counts()
        print(f"\nPrice units: {units.to_dict()}")
    
    # Show expectation types
    if 'expectation_type' in structured_df.columns:
        expectations = structured_df['expectation_type'].value_counts()
        print(f"\nExpectation types: {expectations.to_dict()}")
    
    # Show companies
    if 'company_mention' in structured_df.columns:
        companies = structured_df['company_mention'].value_counts()
        print(f"\nCompanies mentioned: {companies.to_dict()}")
    
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

print("\nMock testing complete! You can now experiment with the data.") 


print(llm_output_df.loc[4]["text_chunk"])