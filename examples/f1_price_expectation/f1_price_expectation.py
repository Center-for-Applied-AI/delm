from pathlib import Path
import pandas as pd
from pprint import pprint
from delm.utils import performance_estimation
from delm import DELM, Schema, ExtractionVariable
import dotenv

# Load API keys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=True)

SOURCE_DATA_PATH = Path(__file__).parent / "data" / "commodity_data.csv"

# Define Schema in code
SCHEMA = Schema.simple(
    ExtractionVariable(
        name="good",
        data_type="string",
        description='The type of good. You may infer the good from context if not explicitly stated or if referred to by a general term (e.g., "fuel" as "oil").',
        required=True,
        allowed_values=[
            "silver",
            "gold",
            "soybeans",
            "heating oil",
            "copper",
            "gasoline",
            "natural gas",
            "aluminum",
            "iron ore",
            "corn",
            "cotton",
            "palm",
            "gas",
            "oil",
            "nickel",
            "sugar",
            "cattle",
            "wheat",
            "coal",
            "zinc",
            "coffee",
            "emissions",
            "tin",
            "hogs",
            "cocoa",
            "lead",
            "diesel",
            "uranium",
            "ethanol",
            "platinum",
            "electricity",
            "fuel",
            "energy",
            "other",
        ],
    ),
    ExtractionVariable(
        name="good_subtype",
        data_type="string",
        required=False,
        description="Subtype or specific variety of the good if applicable",
    ),
    ExtractionVariable(
        name="price_expectation",
        data_type="boolean",
        required=True,
        description="Whether this is a price expectation (future price) or current price",
    ),
    ExtractionVariable(
        name="price_lower",
        data_type="number",
        required=False,
        description="Lower bound of the price range if specified",
    ),
    ExtractionVariable(
        name="price_upper",
        data_type="number",
        required=False,
        description="Upper bound of the price range if specified",
    ),
    ExtractionVariable(
        name="unit",
        data_type="string",
        required=False,
        description="Unit of measurement for the price (e.g., per ton, per barrel, per unit)",
    ),
    ExtractionVariable(
        name="currency",
        data_type="string",
        required=False,
        description="Currency of the price (e.g., USD, EUR, GBP)",
    ),
    ExtractionVariable(
        name="horizon",
        data_type="string",
        required=False,
        description="Time horizon for the price (e.g., Q1 2024, end of year, next quarter)",
    ),
)

PROMPT_TEMPLATE = """Extract expected variables for goods mentioned by firm representatives in investor call transcripts.

Extract the following information from the text:

{variables}

Text to analyze:
{text}"""

# investigate data
df = pd.read_csv(SOURCE_DATA_PATH)
print(df.head())
print(df.info())

input_df = df[["id", "text"]]

print(input_df.iloc[0]["text"])

output_vars = {
    "good": str,
    "good_subtype": str,
    "price_expectation": bool,
    "price_lower": float,
    "price_upper": float,
    "unit": str,
    "currency": str,
    "horizon": str,
}

expected_df = df[["id"] + list(output_vars.keys())]
expected_df = expected_df.astype(output_vars)
expected_df.info()
expected_df["expected_json"] = expected_df[list(output_vars.keys())].to_dict(
    orient="records"
)

# Initialize DELM
delm = DELM(
    schema=SCHEMA,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    max_retries=3,
    batch_size=10,
    max_workers=1,
    base_delay=1.0,
    track_cost=True,
    max_budget=0.5,
    target_column="text",
    prompt_template=PROMPT_TEMPLATE,
    cache_path=".delm/kirill_cache",
)

metrics, processed_df = performance_estimation.estimate_performance(
    delm_instance=delm,
    data_source=input_df,
    expected_extraction_output_df=expected_df,
    true_json_column="expected_json",
    matching_id_column="id",
    record_sample_size=30,
)

print(f'F1 score for price expectation: {metrics["price_expectation"]["f1"]}')
