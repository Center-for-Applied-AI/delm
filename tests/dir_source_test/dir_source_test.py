from pathlib import Path
from delm import (
    DELM,
    Schema,
    ExtractionVariable,
)
from dotenv import load_dotenv

load_dotenv(".env")

TXT_DATA_DIR_PATH = Path("tests/dir_source_test/txt_data")
CSV_DATA_DIR_PATH = Path("tests/dir_source_test/csv_data")

print("=" * 100)
print("Directory Source Test\n")
print("Components Tested:")
print("- Data Processor")
print("- Data Loaders")
print("Expected Outputs:")
print("- Prepped Data")
print("- Extracted Data")
print("- Cost Summary")
print("=" * 100 + "\n")

# Define schema in Python
schema = Schema.simple(
    variables_list=[
        ExtractionVariable(
            name="name",
            description="Name of the person",
            data_type="string",
            required=True,
        ),
        ExtractionVariable(
            name="fruit",
            description="Fruit the person likes",
            data_type="[string]",
            required=False,
        ),
    ],
)

# Define config in Python

print("TXT DIR TEST")
delm_txt = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    batch_size=10,
    max_workers=1,
    max_retries=3,
    base_delay=1.0,
    track_cost=True,
    prompt_template="""You are a helpful assistant who extracts information from text.

Extract the following information from the text. Return the information in the specified format.

{variables}

Text to analyze:
{text}""",
)

result_df = delm_txt.extract(TXT_DATA_DIR_PATH)
print(result_df)
cost_summary = delm_txt.get_cost_summary()
print(cost_summary)

print("=" * 100)
print("CSV DIR TEST")
# Create a new config with target_column set for CSV
delm_csv = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    target_column="text",
    temperature=0.0,
    batch_size=10,
    max_workers=1,
    max_retries=3,
    base_delay=1.0,
    track_cost=True,
    prompt_template="""You are a helpful assistant who extracts information from text.

Extract the following information from the text. Return the information in the specified format.

{variables}

Text to analyze:
{text}""",
)

result_df = delm_csv.extract(CSV_DATA_DIR_PATH)
print(result_df)
cost_summary = delm_csv.get_cost_summary()
print(cost_summary)
