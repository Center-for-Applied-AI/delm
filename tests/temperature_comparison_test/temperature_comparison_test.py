"""
Temperature Comparison Test for DELM
Tests different temperature settings and compares outputs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from delm import DELM, Schema
from delm.models import ExtractionVariable

load_dotenv(".env")


def create_mock_data():
    """Create mock dataset for testing."""
    np.random.seed(42)

    firms = ["Goldman Sachs", "Morgan Stanley", "JP Morgan"]
    dates = [
        datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(5)
    ]
    dates.sort()

    mock_texts = [
        "WTI crude oil prices are expected to remain volatile in the coming quarter. The barrel price of Brent crude has been fluctuating between $70 and $85, with expectations of further increases due to OPEC supply constraints.",
        "Henry Hub natural gas prices have surged by 15% this month, driven by increased LNG demand and limited pipeline supply. We expect this trend to continue through the winter months.",
        "The price of industrial metals, particularly steel and aluminum, has shown significant increases. Ton prices have risen by 20% year-over-year, with expectations of continued growth.",
        "Oil and gas companies like BP and SHEL are using advanced technologies to improve extraction efficiency. The barrel cost of production has decreased by 10% due to these innovations.",
        "Market expectations for commodity prices remain bullish. WTI oil prices are expected to reach $90 per barrel by year-end, while Henry Hub gas prices may stabilize around current levels.",
    ]

    data = []
    for i in range(5):
        data.append(
            {
                "report": f"REP_{(i+1):03d}",
                "date": dates[i],
                "title": f"Market Analysis - Q{i+1} 2024",
                "subtitle": f"Report by {firms[i % len(firms)]}",
                "firm_name": firms[i % len(firms)],
                "text": mock_texts[i],
            }
        )

    return pd.DataFrame(data)


def create_schema():
    """Create schema in Python."""
    return Schema.nested(
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
                description="Company names mentioned in relation to commodities",
                data_type="string",
            ),
        ],
    )


def run_temperature_comparison():
    """Run comparison test with different temperatures."""
    print("Creating mock dataset...")
    test_data = create_mock_data().iloc[:3]
    print(f"Dataset created: {len(test_data)} rows")

    # Create schema
    schema = create_schema()

    # Test temperatures
    temperatures = [0.0, 0.5, 1.0]
    results = {}

    for temp in temperatures:
        print(f"\n--- Testing Temperature: {temp} ---")

        exp_name = f"temp_{temp}"

        # Initialize DELM with specific temperature
        delm = DELM(
            schema=schema,
            provider="openai",
            model="gpt-4o-mini",
            temperature=temp,
            batch_size=1,
            max_workers=1,
            max_retries=3,
            target_column="text",
            drop_target_column=True,
            splitting_strategy={"type": "ParagraphSplit"},
            relevance_scorer={
                "type": "KeywordScorer",
                "keywords": [
                    "price",
                    "prices",
                    "oil",
                    "gas",
                    "expect",
                    "barrel",
                    "ton",
                    "used",
                    "expectations",
                    "using",
                ],
            },
        )

        # Process data
        result_df = delm.extract(test_data)
        results[temp] = result_df

    return results


if __name__ == "__main__":
    results = run_temperature_comparison()
    for temp, result in results.items():
        print(f"Temperature: {temp}")
        print(result)
        print("\n")
