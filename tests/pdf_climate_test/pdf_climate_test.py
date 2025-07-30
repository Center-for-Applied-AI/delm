from delm import DELM, DELMConfig
from pathlib import Path

DATA_DIR = Path("tests/pdf_climate_test/data")
EXPERIMENT_DIR = Path("test_experiments")
CONFIG_PATH = Path("tests/pdf_climate_test/config.yaml")

print("="*100)
print("PDF Climate Test\n")
print("Components Tested:")
print("- PDF Data Loader")
print("- Simple Schema")
print("Expected Outputs:")
print("- Prepped Data")
print("- Extracted Data")
print("- Cost Summary")
print("="*100 + "\n")

print("TXT DIR TEST")

config = DELMConfig.from_yaml(CONFIG_PATH)
delm = DELM(
    config = config,
    experiment_name="pdf_climate_test",
    experiment_directory=EXPERIMENT_DIR,
    overwrite_experiment=True,
    use_disk_storage=True,
)

print("="*100)
print("Prepping PDF Data")
prepped_txt_df = delm.prep_data(DATA_DIR, sample_size=5)

print("-"*100)
print(prepped_txt_df)
print("-"*100)

print("="*100)
print("Processing PDF Data")
result_df = delm.process_via_llm()
print("-"*100)
print(result_df)
print("-"*100)

print("="*100)
print("Getting Cost Summary")
cost_summary = delm.get_cost_summary()
print("-"*100)
print(cost_summary)