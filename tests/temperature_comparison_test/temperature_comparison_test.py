"""
Temperature Comparison Test for DELM
Tests different temperature settings and compares outputs
"""

from copy import deepcopy
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from delm import DELM, DELMConfig
from delm.constants import SYSTEM_EXTRACTED_DATA_JSON_COLUMN

def create_mock_data():
    """Create mock dataset for testing."""
    np.random.seed(42)
    
    firms = ["Goldman Sachs", "Morgan Stanley", "JP Morgan"]
    dates = [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(5)]
    dates.sort()
    
    mock_texts = [
        "WTI crude oil prices are expected to remain volatile in the coming quarter. The barrel price of Brent crude has been fluctuating between $70 and $85, with expectations of further increases due to OPEC supply constraints.",
        "Henry Hub natural gas prices have surged by 15% this month, driven by increased LNG demand and limited pipeline supply. We expect this trend to continue through the winter months.",
        "The price of industrial metals, particularly steel and aluminum, has shown significant increases. Ton prices have risen by 20% year-over-year, with expectations of continued growth.",
        "Oil and gas companies like BP and SHEL are using advanced technologies to improve extraction efficiency. The barrel cost of production has decreased by 10% due to these innovations.",
        "Market expectations for commodity prices remain bullish. WTI oil prices are expected to reach $90 per barrel by year-end, while Henry Hub gas prices may stabilize around current levels."
    ]
    
    data = []
    for i in range(5):
        data.append({
            "report": f"REP_{(i+1):03d}",
            "date": dates[i],
            "title": f"Market Analysis - Q{i+1} 2024",
            "subtitle": f"Report by {firms[i % len(firms)]}",
            "firm_name": firms[i % len(firms)],
            "text": mock_texts[i]
        })
    
    return pd.DataFrame(data)

def create_base_config():
    """Load base configuration from config.yaml."""
    return DELMConfig.from_yaml(Path("tests/temperature_comparison_test/config.yaml"))

def run_temperature_comparison():
    """Run comparison test with different temperatures."""
    print("Creating mock dataset...")
    test_data = create_mock_data().iloc[:3]
    print(f"Dataset created: {len(test_data)} rows")

    # Load base config from YAML
    base_config = create_base_config()

    # Test temperatures
    temperatures = [0.0, 0.5, 1.0]
    results = {}

    for temp in temperatures:
        print(f"\n--- Testing Temperature: {temp} ---")

        exp_name = f"temp_{temp}"
        # Create config variation using dataclasses.replace
        config = deepcopy(base_config)
        config.llm_extraction.temperature = temp

        # Initialize DELM
        delm = DELM(
            config=config,
            experiment_name=exp_name,
            experiment_directory=Path("test_experiments"),
            overwrite_experiment=True,
            auto_checkpoint_and_resume_experiment=False,
        )

        # Process data
        delm.prep_data(test_data)
        delm.process_via_llm()

        cost_summary = delm.get_cost_summary()
        print(json.dumps(cost_summary, indent=2))

        # Get the results from the experiment directory
        results[temp] = delm.get_extraction_results()

    return results

if __name__ == "__main__":
    results = run_temperature_comparison() 
    for temp, result in results.items():
        print(f"Temperature: {temp}")
        print(result[SYSTEM_EXTRACTED_DATA_JSON_COLUMN])
        print("\n")
    