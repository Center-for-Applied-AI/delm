from copy import deepcopy
import pandas as pd
from pathlib import Path
from delm.config import DELMConfig
from delm.strategies.splitting_strategies import RegexSplit
from delm.utils.cost_estimation import estimate_input_token_cost, estimate_total_cost
import numpy as np
from datetime import datetime, timedelta
import json

def mock_data():
    np.random.seed(42)
    firms = ["Goldman Sachs", "Morgan Stanley", "JP Morgan", "Barclays", "Deutsche Bank"]
    report_types = ["Market Analysis", "Economic Outlook", "Sector Review", "Investment Strategy"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(20)]
    dates.sort()
    mock_texts = [
        "WTI crude oil prices are expected to remain volatile in the coming quarter. The barrel price of Brent crude has been fluctuating between $70 and $85, with expectations of further increases due to OPEC supply constraints. XOM and CVX are leading producers.",
        "Henry Hub natural gas prices have surged by 15% this month, driven by increased LNG demand and limited pipeline supply. We expect this trend to continue through the winter months. TTF prices in Europe are also rising.",
        "The price of industrial metals, particularly steel and aluminum, has shown significant increases. Ton prices have risen by 20% year-over-year, with expectations of continued growth. Production volumes reached 1.2 million tons last quarter.",
        "Oil and gas companies like BP and SHEL are using advanced technologies to improve extraction efficiency. The barrel cost of production has decreased by 10% due to these innovations. Light Sweet crude production increased by 5%.",
        "Market expectations for commodity prices remain bullish. WTI oil prices are expected to reach $90 per barrel by year-end, while Henry Hub gas prices may stabilize around current levels. JKM LNG prices are showing volatility.",
        "The energy sector continues to show strong performance. Companies like AAPL and MSFT are investing heavily in renewable energy sources while maintaining traditional oil and gas operations. GOOGL has announced new energy initiatives.",
        "Commodity markets are experiencing increased volatility. Investors should expect continued price fluctuations across various sectors. AMZN's logistics division is adapting to fuel price changes.",
        "Supply chain disruptions are affecting multiple industries. Companies are using alternative suppliers to maintain production levels. Heavy Sour crude availability has been impacted.",
        "The transportation sector faces challenges due to fuel price increases. Companies are exploring alternative energy sources to reduce costs. Pipeline capacity constraints are affecting gas distribution.",
        "Economic indicators suggest moderate growth expectations. The manufacturing sector shows signs of recovery with increased demand for raw materials. Production volumes are expected to grow by 8% in Q4.",
        "Technology stocks like AAPL and MSFT have outperformed the broader market this quarter. Software companies continue to show strong revenue growth. GOOGL's cloud division reported record earnings.",
        "The healthcare sector remains resilient despite economic uncertainties. Pharmaceutical companies are developing innovative treatments. AMZN's healthcare initiatives are gaining traction.",
        "Consumer spending patterns have shifted significantly. Retail companies are adapting to changing customer preferences. E-commerce platforms are seeing increased adoption.",
        "The real estate market shows signs of stabilization. Property prices in major metropolitan areas are beginning to level off. Investment volumes are expected to remain steady.",
        "Financial services companies are expanding their digital offerings. Online banking and mobile payment solutions are gaining popularity. Traditional banks are modernizing their platforms.",
        "The agricultural sector faces challenges from climate change. Farmers are using new technologies to improve crop yields. Production volumes for key crops have increased by 12%.",
        "International trade agreements are reshaping global supply chains. Companies are adapting their strategies to navigate new regulations. Brent crude imports have been affected by trade policies.",
        "The automotive industry is undergoing a major transformation. Electric vehicle adoption is accelerating across all markets. Traditional automakers are investing heavily in new technologies.",
        "Renewable energy investments are reaching record levels. Solar and wind power projects are becoming increasingly cost-effective. LNG infrastructure development is expanding globally.",
        "The telecommunications sector is experiencing rapid technological change. 5G networks are being deployed across major markets. Infrastructure investment volumes are at all-time highs."
    ]
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
    return report_text_df

def main():
    base_config_path = Path("tests/mock_test/config.yaml")
    config = DELMConfig.from_yaml(base_config_path)
    # Second config: RegexSplit by sentence
    config2 = config.to_dict()
    config2["data_preprocessing"]["splitting"] = {
        "type": "RegexSplit",
        "pattern": r"(?<=[.!?])\s+"
    }
    config2 = DELMConfig.from_dict(config2)

    # Heuristic estimation
    results_heuristic = [
        estimate_input_token_cost(config, mock_data()),
        estimate_input_token_cost(config2, mock_data())
    ]
    print("Heuristic cost estimation results:")
    for res in results_heuristic:
        print(json.dumps(res, indent=2, default=str))

    # API estimation (just for the default config)
    res = estimate_total_cost(config, mock_data(), sample_size=3)
    print("API cost estimation result:")
    print(json.dumps(res, indent=2, default=str))

if __name__ == "__main__":
    main() 