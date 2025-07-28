from pathlib import Path
import pandas as pd
from pprint import pprint
from delm.utils import performance_estimation
from delm import DELMConfig

SOURCE_DATA_PATH = Path("examples/f1_price_expectation/data/commodity_data.csv")
CONFIG_PATH = Path("examples/f1_price_expectation/config.yaml")

# investigate data
df = pd.read_csv(SOURCE_DATA_PATH)
print(df.head())
print(df.info())

input_df = df[['id', 'text']]

print(input_df.iloc[0]['text'])

output_vars = {
    'good': str, 
    'good_subtype': str, 
    'price_expectation': bool, 
    'price_lower': float, 
    'price_upper': float, 
    'unit': str, 
    'currency': str, 
    'horizon': str
}

expected_df = df[['id'] + list(output_vars.keys())]
expected_df = expected_df.astype(output_vars)
expected_df.info()
expected_df['expected_json'] = expected_df[list(output_vars.keys())].to_dict(orient='records')

metrics, processed_df = performance_estimation.estimate_performance(
    config=DELMConfig.from_yaml(Path("examples/f1_price_expectation/config.yaml")),
    data_source=input_df,
    expected_extraction_output_df=expected_df,
    true_json_column="expected_json",
    matching_id_column="id",
    record_sample_size=30
)

print(f'F1 score for price expectation: {metrics["price_expectation"]["f1"]}')