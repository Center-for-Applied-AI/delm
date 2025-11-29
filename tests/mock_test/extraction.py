"""
Mock testing for DELM
"""

from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

from delm import DELM, Schema
from delm.models import ExtractionVariable

load_dotenv(".env")

print(f"=" * 60)
print("Basic Mock Test")
print("Components Tested:")
print("- DELMConfig")
print("- DELM")
print("- DELM.prep_data")
print("- DELM.process_via_llm")
print("Expected Outputs:")
print("- Extracted data")
print("- Cost of Test")
print(f"=" * 60)
print("\n")

# Paths
DOTENV_PATH = Path(".env")

# Create mock data (run this cell first)
np.random.seed(42)  # For reproducible results

# Sample data for generating realistic-looking reports
firms = ["Goldman Sachs", "Morgan Stanley", "JP Morgan", "Barclays", "Deutsche Bank"]
report_types = [
    "Market Analysis",
    "Economic Outlook",
    "Sector Review",
    "Investment Strategy",
]

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
    "The telecommunications sector is experiencing rapid technological change. 5G networks are being deployed across major markets. Infrastructure investment volumes are at all-time highs.",
]

# Create the DataFrame
data = []
for i in range(20):
    report_type = np.random.choice(report_types)
    quarter = np.random.choice(["Q1", "Q2", "Q3", "Q4"])
    year = np.random.randint(2022, 2024)
    firm = np.random.choice(firms)
    text = np.random.choice(mock_texts)

    data.append(
        {
            "report": f"REP_{(i+1):03d}",
            "date": dates[i],
            "title": f"{report_type} - {quarter} {year}",
            "subtitle": f"Market Analysis Report by {firm}",
            "firm_name": firm,
            "text": text,
        }
    )

report_text_df = pd.DataFrame(data)


print(f"-" * 40)
print("Mock dataset created successfully!")
print(f"Shape: {report_text_df.shape}")
print(f"Columns: {list(report_text_df.columns)}")
print(f"-" * 40)

schema = Schema.nested(
    container_name="commodities",
    variables_list=[
        ExtractionVariable(
            name="commodity_type",
            description="Type of commodity mentioned",
            data_type="string",
            required=True,
            allowed_values=[
                "oil",
                "gas",
                "copper",
                "gold",
                "silver",
                "steel",
                "aluminum",
            ],
        ),
        ExtractionVariable(
            name="price_mention",
            description="Whether a specific price is mentioned",
            data_type="boolean",
        ),
        ExtractionVariable(
            name="price_value",
            description="Numeric price value if mentioned",
            data_type="number",
        ),
        ExtractionVariable(
            name="price_unit",
            description="Unit of the price (e.g., barrel, ton, MMBtu)",
            data_type="string",
        ),
        ExtractionVariable(
            name="expectation_type",
            description="Type of price expectation mentioned",
            data_type="string",
            allowed_values=[
                "forecast",
                "guidance",
                "estimate",
                "projection",
                "outlook",
            ],
        ),
        ExtractionVariable(
            name="company_mention",
            description="Company names mentioned in relation to the commodity",
            data_type="[string]",
        ),
    ],
)

delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    batch_size=5,
    max_workers=1,
    max_retries=0,
    base_delay=1.0,
    tokens_per_minute=10000,
    requests_per_minute=5,
    track_cost=False,
    target_column="text",
    drop_target_column=True,
    splitting_strategy={"type": "ParagraphSplit"},
    relevance_scorer={"type": "KeywordScorer", "keywords": ["revenue", "profit"]},
    cache_backend=None,
    use_disk_storage=True,
    experiment_path=Path("test_experiments/mock_test"),
    overwrite_experiment=True,
    save_log_file=True,
)

result_df = delm.extract(report_text_df, sample_size=3)

print(f"-" * 40)
print("Data finished processing")
print(f"-" * 40)

# cost_summary = delm.get_cost_summary()
# print(json.dumps(cost_summary, indent=2))

# The output is JSON by default - let's show how to work with it
print("=" * 60)
print("VISUALIZE OUTPUT")
print("=" * 60)

import json

for idx, row in result_df.head(3).iterrows():
    # Print all columns except delm_extracted_data
    for col in result_df.columns:
        if col != "delm_extracted_data_json":
            print(f"{col}: {row[col]}")
    print("delm_extracted_data_json:")
    try:
        parsed = json.loads(row["delm_extracted_data_json"])  # type: ignore
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(row["delm_extracted_data_json"])
    print("-" * 40)

from delm.utils.post_processing import explode_json_results

exploded_df = explode_json_results(result_df, delm.config.schema.schema)
print(exploded_df)
print(exploded_df.columns)
