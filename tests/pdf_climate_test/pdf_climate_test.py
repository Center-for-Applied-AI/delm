from delm import DELM, Schema
from delm.models import ExtractionVariable
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv(".env")

DATA_DIR = Path("tests/pdf_climate_test/data")
EXPERIMENT_DIR = Path("test_experiments")

print("=" * 100)
print("PDF Climate Test\n")
print("Components Tested:")
print("- PDF Data Loader")
print("- Simple Schema")
print("Expected Outputs:")
print("- Prepped Data")
print("- Extracted Data")
print("- Cost Summary")
print("=" * 100 + "\n")

print("PDF CLIMATE TEST")

# Define schema
schema = Schema.simple(
    variables_list=[
        ExtractionVariable(
            name="climate_action_score",
            description="""1 = Strong opposition to climate action by the regulator. Explicitly resists climate measures. May deny climate change or climate risks.
2 = Skeptical or hesitant. Questions the need for special treatment or warns about costs and unintended consequences.
3 = Neutral. Takes no strong position for or against climate action.
4 = Supportive. Backs climate actions of the regulator. May support other climate measures. May advocate for more incremental steps.
5 = Strong advocate. Fully supports ambitious, binding climate targets and broad reforms. May seek to strengthen proposed initiatives.""",
            data_type="integer",
            required=True,
            allowed_values=[0, 1, 2, 3, 4, 5],
        ),
    ],
)

# Define custom prompt template
custom_prompt_template = """You are a climate change expert who expects meticulous and reliable results.

Extract the following information from the text:

{variables}

Text to analyze:
{text}"""

# Create DELM instance
delm = DELM(
    schema=schema,
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    batch_size=10,
    max_workers=1,
    max_retries=3,
    base_delay=1.0,
    track_cost=True,
    prompt_template=custom_prompt_template,
)

print("=" * 100)
print("Extracting Data")
extracted_df = delm.extract(DATA_DIR, sample_size=5)

print("-" * 100)
print(extracted_df)
print("-" * 100)

print("=" * 100)
print("Cost Summary")
cost_summary = delm.get_cost_summary()
print("-" * 100)
print(json.dumps(cost_summary, indent=2))
