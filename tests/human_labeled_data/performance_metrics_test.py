import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv

from delm import DELMConfig, Schema
from delm.models import ExtractionVariable
from delm.utils.performance_estimation import estimate_performance

load_dotenv(".env")

print(f"=" * 60)
print("Human Labeled Data Performance Metrics Test")
print("Components Tested:")
print("- DELM")
print("- DELM.estimate_performance")
print("Expected Outputs:")
print("- Performance Metrics")
print("- Processed Data that was used to calculate performance metrics")
print(f"=" * 60)
print("\n")

human_labeled_input_df = pd.read_parquet(
    "tests/human_labeled_data/human_labeled_input_records.parquet"
)
human_labeled_output_df = pd.read_stata(
    "tests/human_labeled_data/KIRILL_priceexp_final_data_sample_raw.dta"
)

human_labeled_output_df["report"] = human_labeled_output_df["report"].astype(int)  # type: ignore

# Add expected_json as a dict, not a string
human_labeled_output_df["expected_dict"] = human_labeled_output_df.apply(
    lambda row: {  # type: ignore
        "horizon": row["horizon"],
        "good_subtype": row["good_subtype"],
        "price": row["price"],
        "unit": row["unit"],
        "currency": row["currency"],
        "good": row["good"],
        "price_lower": row["price_lower"],
        "price_upper": row["price_upper"],
    },
    axis=1,
)

# Define schema in Python code
schema = Schema.simple(
    variables_list=[
        ExtractionVariable(
            name="horizon",
            description="Time horizon for the price expectation or forecast, if mentioned",
            data_type="string",
        ),
        ExtractionVariable(
            name="good_subtype",
            description="Subtype or specific variety of the good or commodity mentioned",
            data_type="string",
        ),
        ExtractionVariable(
            name="price",
            description="Price value mentioned in the text",
            data_type="number",
        ),
        ExtractionVariable(
            name="unit",
            description="Unit of measurement for the price (e.g., barrel, ton, MMBtu)",
            data_type="string",
        ),
        ExtractionVariable(
            name="currency",
            description="Currency in which the price is denominated (e.g., USD, EUR)",
            data_type="string",
        ),
        ExtractionVariable(
            name="good",
            description="Name of the good or commodity mentioned",
            data_type="string",
        ),
        ExtractionVariable(
            name="price_lower",
            description="Lower bound of a price range if specified",
            data_type="number",
        ),
        ExtractionVariable(
            name="price_upper",
            description="Upper bound of a price range if specified",
            data_type="number",
        ),
    ],
)

# Define custom prompt template
custom_prompt_template = """You are assisting a finance professor who expects meticulous and reliable results.

Extract the following information from the text:

{variables}

Text to analyze:
{text}

CRITICAL INSTRUCTIONS:
- ONLY extract information that is EXPLICITLY mentioned in the text
- If NO relevant information is mentioned, return empty lists or null values
- Do NOT infer or guess based on context or company names
- Do NOT extract information just because it might be related
- For each item mentioned, create a separate entry with all relevant details
- If a field is not mentioned in the text, leave it as null/None rather than guessing
- Focus on extracting accurate, factual data as stated in the text

Examples of what NOT to extract:
- "1-800 CONTACTS" → NOT oil (even though contacts might use oil-based solutions)
- "Apple Inc." → NOT aluminum (even though phones contain aluminum)
- "Bank of America" → NOT gold (even though banks might trade gold)"""

# Create DELM instance with all config params
delm = DELMConfig(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    batch_size=10,
    max_workers=1,
    max_retries=3,
    base_delay=1.0,
    track_cost=True,
    target_column="text",
    drop_target_column=True,
    splitting_strategy={"type": "ParagraphSplit"},
    relevance_scorer={
        "type": "KeywordScorer",
        "keywords": [
            "price",
            "forecast",
            "guidance",
            "estimate",
            "expectation",
            "revenue",
            "earnings",
        ],
    },
    score_filter="delm_score > 0.5",
    prompt_template=custom_prompt_template,
)

# Run performance estimation
performance_metrics_dict, processed_df = estimate_performance(
    config=delm,
    data_source=human_labeled_input_df,
    expected_extraction_output_df=human_labeled_output_df,  # type: ignore
    true_json_column="expected_dict",
    matching_id_column="report",
    record_sample_size=2,
)

print(f"-" * 40)
print("Performance Metrics (Precision and Recall Only)")
print(f"-" * 40)
header = f"{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}"
print(header)
print("-" * len(header))
for key, value in performance_metrics_dict.items():
    print(
        f"{key:<20} {value['precision']:10.3f} {value['recall']:10.3f} {value['f1']:10.3f}"
    )

print(f"-" * 40)
print("Processed Data")
print(f"-" * 40)
print(processed_df.head())
