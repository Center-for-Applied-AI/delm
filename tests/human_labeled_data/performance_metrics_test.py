import pandas as pd
import json
from pathlib import Path
from delm import DELM, DELMConfig
from delm.utils.performance_estimation import estimate_performance
from pprint import pprint

print(f"="*60)
print("Human Labeled Data Performance Metrics Test")
print("Components Tested:")
print("- DELM")
print("- DELM.estimate_performance")
print("Expected Outputs:")
print("- Performance Metrics")
print("- Processed Data that was used to calculate performance metrics")
print(f"="*60)
print("\n")

human_labeled_input_df = pd.read_parquet("tests/human_labeled_data/human_labeled_input_records.parquet")
human_labeled_output_df = pd.read_stata("tests/human_labeled_data/KIRILL_priceexp_final_data_sample_raw.dta") #

human_labeled_output_df["report"] = human_labeled_output_df["report"].astype(int) # type: ignore

# Add expected_json as a dict, not a string
human_labeled_output_df["expected_dict"] = human_labeled_output_df.apply(lambda row: { # type: ignore
    "horizon": row["horizon"],
    "good_subtype": row["good_subtype"],
    "price": row["price"],
    "unit": row["unit"],
    "currency": row["currency"],
    "good": row["good"],
    "price_lower": row["price_lower"],
    "price_upper": row["price_upper"],
}, axis=1)

config = DELMConfig.from_any("tests/human_labeled_data/config.yaml")
performance_metrics_dict, processed_df = estimate_performance(
    config=config,
    data_source=human_labeled_input_df,
    expected_extraction_output_df=human_labeled_output_df, # type: ignore
    true_json_column="expected_dict",
    matching_id_column="record_id",
    record_sample_size=5
)

print(f"-"*40)
print("Performance Metrics (Precision and Recall Only)")
print(f"-"*40)
header = f"{'Field':<20} {'Precision':>10} {'Recall':>10}"
print(header)
print("-" * len(header))
for key, value in performance_metrics_dict.items():
    print(f"{key:<20} {value['precision']:10.3f} {value['recall']:10.3f}")

print(f"-"*40)
print("Processed Data")
print(f"-"*40)
print(processed_df.head())