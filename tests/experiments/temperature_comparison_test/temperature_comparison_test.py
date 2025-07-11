"""
Temperature Comparison Test for DELM
Tests different temperature settings and compares outputs
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import replace

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from delm import DELM, DELMConfig
from delm.config import ModelConfig, DataConfig, SchemaConfig, ExperimentConfig, SplittingConfig, ScoringConfig
from delm.scoring_strategies import KeywordScorer
from delm.splitting_strategies import ParagraphSplit

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
    """Create base configuration."""
    model_config = ModelConfig(
        name="gpt-4o-mini",
        temperature=0.0,  # Will be varied
        max_retries=3,
        batch_size=1,
        max_workers=1,
        dotenv_path=None,
        regex_fallback_pattern=None
    )
    
    splitting_config = SplittingConfig(strategy=ParagraphSplit())
    
    scoring_config = ScoringConfig(
        scorer=KeywordScorer([
            "price", "prices", "oil", "gas", "expect", "barrel", 
            "ton", "used", "expectations", "using"
        ])
    )
    
    data_config = DataConfig(
        target_column="text",
        drop_target_column=True,
        splitting=splitting_config,
        scoring=scoring_config
    )
    
    schema_config = SchemaConfig(
        spec_path=Path("tests/experiments/temperature_comparison_test/schema_spec.yaml"),
        container_name="commodities",
        prompt_template=None
    )
    
    experiment_config = ExperimentConfig(
        name="temp_comparison",  # Will be varied
        directory=Path("test-experiments"),
        save_intermediates=False,  # Disable for cleaner output
        overwrite_experiment=True,
        verbose=False
    )
    
    return DELMConfig(
        model=model_config,
        data=data_config,
        schema=schema_config,
        experiment=experiment_config
    )

def run_temperature_comparison():
    """Run comparison test with different temperatures."""
    print("Creating mock dataset...")
    test_data = create_mock_data().iloc[:3]
    print(f"Dataset created: {len(test_data)} rows")
    
    # Create base config
    base_config = create_base_config()
    
    # Test temperatures
    temperatures = [0.0, 0.5, 1.0]
    results = {}
    
    for temp in temperatures:
        print(f"\n--- Testing Temperature: {temp} ---")
        
        # Create config variation using dataclasses.replace
        config = replace(
            base_config,
            model=replace(base_config.model, temperature=temp),
            experiment=replace(base_config.experiment, name=f"temp_{temp}")
        )
        
        # Initialize DELM
        delm = DELM(config=config)
        
        # Process data
        output_df = delm.prep_data(test_data)
        llm_output_df = delm.process_via_llm()
        structured_df = delm.parse_to_dataframe(llm_output_df)
        
        print(f'structured_df: {structured_df}') 

if __name__ == "__main__":
    run_temperature_comparison() 