from pathlib import Path
from delm import DELM, DELMConfig

CONFIG_PATH = Path("tests/dir_source_test/config.yaml")
TXT_DATA_DIR_PATH = Path("tests/dir_source_test/txt_data")
CSV_DATA_DIR_PATH = Path("tests/dir_source_test/csv_data")

EXPERIMENT_DIR = Path("test_experiments")

print("="*100)
print("Directory Source Test\n")
print("Components Tested:")
print("- Data Processor")
print("- Data Loaders")
print("Expected Outputs:")
print("- Prepped Data")
print("- Extracted Data")
print("- Cost Summary")
print("="*100 + "\n")


print("TXT DIR TEST")
config = DELMConfig.from_yaml(CONFIG_PATH)
delm_txt = DELM(config=config, experiment_name="txt_dir_test", experiment_directory=EXPERIMENT_DIR, overwrite_experiment=True)

print("="*100)
print("Prepping TXTData")
prepped_txt_df = delm_txt.prep_data(TXT_DATA_DIR_PATH)

print("-"*100)
print(prepped_txt_df)
print("-"*100)

print("="*100)
print("Processing TXT Data")
delm_txt.process_via_llm()
print("-"*100)
print(delm_txt.get_extraction_results_json())
print("-"*100)

print("="*100)
print("Getting Cost Summary TXT")
cost_summary = delm_txt.get_cost_summary()
print("-"*100)
print(cost_summary)

print("="*100)
print("CSV DIR TEST")
config = DELMConfig.from_yaml(CONFIG_PATH)
config.data_preprocessing.target_column = "text"
delm_csv = DELM(config=config, experiment_name="csv_dir_test", experiment_directory=EXPERIMENT_DIR, overwrite_experiment=True)

print("="*100)
print("Prepping CSV Data")
prepped_csv_df = delm_csv.prep_data(CSV_DATA_DIR_PATH)

print("-"*100)
print(prepped_csv_df)
print("-"*100)

print("="*100)
print("Processing CSV Data")
delm_csv.process_via_llm()
print("-"*100)
print(delm_csv.get_extraction_results_json())
print("-"*100)

print("="*100)
print("Getting Cost Summary CSV")
cost_summary = delm_csv.get_cost_summary()
print("-"*100)
print(cost_summary)