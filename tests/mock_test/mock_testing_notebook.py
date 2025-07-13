"""
Mock testing for DELM - Jupyter REPL version
Run this cell by cell in Jupyter for interactive testing
Updated to use YAML configuration file
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

from delm import DELM, DELMConfig
from delm.constants import SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN, SYSTEM_EXTRACTED_DATA_COLUMN

# Paths
EXPERIMENT_DIR = Path("test-experiments")
CONFIG_PATH = Path("tests/mock_test/config.yaml")
SCHEMA_SPEC_PATH = Path("tests/mock_test/schema_spec.yaml")
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

# Initialize DELM with YAML config (run this cell)
print("\nLoading DELM configuration from YAML...")
config = DELMConfig.from_yaml(CONFIG_PATH)

# Initialize DELM with config
delm = DELM(config=config)

print("DELM initialized successfully!")

# Process data with DELM (run this cell)
output_df = delm.prep_data(report_text_df.iloc[:3])

print(f"Data preprocessed successfully!")
print(f"Prepped Data columns: {list(output_df.columns)}")

# Process with LLM (no parameters needed - uses constructor config)
llm_output_df = delm.process_via_llm()

print(f"LLM processing completed!")
print(f"LLM output columns: {list(llm_output_df.columns)}")

if not llm_output_df.empty:
    print("\nLLM Output sample:")
    print(llm_output_df.head())

# The output is now JSON by default - let's show how to work with it
print("\n" + "="*60)
print("WORKING WITH JSON OUTPUT")
print("="*60)

if not llm_output_df.empty:
    print(f"\nJSON output format: {list(llm_output_df.columns)}")
    
    # Show how to access JSON data
    print("\nExample: Accessing JSON data from first chunk:")
    first_row = llm_output_df.iloc[0]
    if SYSTEM_EXTRACTED_DATA_COLUMN in first_row and first_row[SYSTEM_EXTRACTED_DATA_COLUMN] is not None:
        import json
        try:
            extracted_data = json.loads(str(first_row[SYSTEM_EXTRACTED_DATA_COLUMN]))
            print(f"Raw JSON: {json.dumps(extracted_data, indent=2)}")
            
            # Extract specific data
            if 'commodities' in extracted_data:
                commodities = extracted_data['commodities']
                print(f"\nFound {len(commodities)} commodities:")
                for i, commodity in enumerate(commodities):
                    print(f"  {i+1}. {commodity.get('commodity_type', 'N/A')} - ${commodity.get('price_value', 'N/A')}")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing JSON: {e}")

print("\nMock testing complete! You can now experiment with the data.")

# DEMONSTRATION: How nested schemas handle multiple objects per text chunk
print("\n" + "="*60)
print("DEMONSTRATION: JSON Output Structure")
print("="*60)

print("\nWhen you have a nested schema (like 'commodities'), the system:")
print("1. Extracts multiple objects from each text chunk")
print(f"2. Stores them as JSON in the '{SYSTEM_EXTRACTED_DATA_COLUMN}' column")
print("3. Maintains the original text_chunk and metadata for each row")

print(f"\nExample: Your current schema extracts 'commodities' objects")
print(f"Input text chunks: {len(report_text_df.iloc[:3])}")
print(f"Output DataFrame rows: {len(llm_output_df)}")
print(f"Each row contains JSON with extracted objects")

if not llm_output_df.empty:
    print(f"\nDataFrame columns: {list(llm_output_df.columns)}")
    print(f"\nSample of how JSON data is structured:")
    
    # Show JSON structure for first few chunks
    for idx, row in llm_output_df.head(3).iterrows():
        print(f"\nChunk {row.get(SYSTEM_CHUNK_ID_COLUMN, idx)}:")
        text_chunk = row.get(SYSTEM_CHUNK_COLUMN, '')
        if text_chunk:
            print(f"Text: {text_chunk[:100]}...")
        else:
            print(f"Text: (no text chunk)")
        
        if SYSTEM_EXTRACTED_DATA_COLUMN in row and row[SYSTEM_EXTRACTED_DATA_COLUMN] is not None:
            import json
            try:
                data = json.loads(str(row[SYSTEM_EXTRACTED_DATA_COLUMN]))
                if 'commodities' in data:
                    commodities = data['commodities']
                    print(f"  Found {len(commodities)} commodities in JSON:")
                    for i, commodity in enumerate(commodities):
                        commodity_info = f"    {i+1}. {commodity.get('commodity_type', 'N/A')}"
                        if commodity.get('price_value'):
                            commodity_info += f" @ ${commodity.get('price_value')} {commodity.get('price_unit', '')}"
                        if commodity.get('company_mention'):
                            commodity_info += f" (companies: {commodity.get('company_mention')})"
                        print(commodity_info)
                else:
                    print(f"  JSON data: {json.dumps(data, indent=4)}")
            except json.JSONDecodeError:
                print(f"  Invalid JSON: {row[SYSTEM_EXTRACTED_DATA_COLUMN]}")

print(f"\nThis JSON structure allows you to:")
print("- Access all extracted data in its original structure")
print("- Parse specific fields when needed")
print("- Maintain all relationships between objects")
print("- Handle any schema complexity without data loss") 