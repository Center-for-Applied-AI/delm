import shutil
from pathlib import Path
import pandas as pd
import tempfile
import sys

from delm import DELM, DELMConfig
from delm.constants import SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN, SYSTEM_EXTRACTED_DATA_COLUMN
from delm.exceptions import ConfigurationError

import yaml

# Helper to create a minimal DataFrame
def make_df():
    return pd.DataFrame({
        'report': ['R1', 'R2'],
        'date': ['2024-01-01', '2024-01-02'],
        'title': ['T1', 'T2'],
        'subtitle': ['S1', 'S2'],
        'firm_name': ['F1', 'F2'],
        'text': ['This is a test.', 'Another test.']
    })

def make_valid_feather(path):
    df = pd.DataFrame({
        SYSTEM_CHUNK_COLUMN: ['a', 'b'],
        SYSTEM_CHUNK_ID_COLUMN: [0, 1],
        SYSTEM_EXTRACTED_DATA_COLUMN: ['{"foo": "bar"}', '{"foo": "baz"}']
    })
    df.to_feather(path)
    return path

def make_invalid_feather(path):
    df = pd.DataFrame({'not_a_real_column': [1, 2]})
    df.to_feather(path)
    return path

def run_delm_with_config(config_path, experiment_name, experiment_dir):
    config = DELMConfig.from_yaml(config_path)
    delm = DELM(
        config=config,
        experiment_name=experiment_name,
        experiment_directory=experiment_dir,
        overwrite_experiment=True,
        verbose=False,
        auto_checkpoint_and_resume_experiment=True
    )
    return delm

def main():
    tmp_root = Path(tempfile.mkdtemp())
    print(f"[INFO] Using temp root: {tmp_root}")
    try:
        # --- 1. Normal run: preprocess and use data in same experiment ---
        base_schema_path = tmp_root / "schema_a.yaml"
        base_schema_path.write_text('type: object\nproperties:\n  foo:\n    type: string\n')
        base_config = {
            'llm_extraction': {'provider': 'openai', 'name': 'gpt-4o-mini', 'dotenv_path': '.env'},
            'data_preprocessing': {'target_column': 'text', 'drop_target_column': False},
            'schema': {'spec_path': str(base_schema_path), 'container_name': 'root'},
        }
        base_config_path = tmp_root / "config_a.yaml"
        yaml.safe_dump(base_config, open(base_config_path, 'w'))
        exp_a_dir = tmp_root / "exp_a"
        exp_a_dir.mkdir()
        delm_a = run_delm_with_config(base_config_path, "exp_a", exp_a_dir)
        df = make_df()
        out_df = delm_a.prep_data(df)
        preprocessed_path = delm_a.experiment_manager.preprocessed_data_path
        assert out_df.shape[0] == 2, "Normal run: Preprocessing failed."
        print("[PASS] Normal run: Preprocessing and saving works.")

        # --- 2. Use preprocessed_data_path only (should succeed) ---
        config_b = {
            'llm_extraction': {'provider': 'openai', 'name': 'gpt-4o-mini', 'dotenv_path': '.env'},
            'data_preprocessing': {'preprocessed_data_path': str(preprocessed_path)},
            'schema': {'spec_path': str(base_schema_path), 'container_name': 'root'},
        }
        config_b_path = tmp_root / "config_b.yaml"
        yaml.safe_dump(config_b, open(config_b_path, 'w'))
        exp_b_dir = tmp_root / "exp_b"
        exp_b_dir.mkdir()
        delm_b = run_delm_with_config(config_b_path, "exp_b", exp_b_dir)
        try:
            delm_b.prep_data(make_df())  # Pass dummy DataFrame
            print("[PASS] preprocessed_data_path only: accepted valid feather file.")
        except Exception as e:
            assert False, f"preprocessed_data_path only: Unexpected error: {e}"

        # --- 3. Use preprocessed_data_path + other data fields (should error) ---
        config_c = {
            'llm_extraction': {'provider': 'openai', 'name': 'gpt-4o-mini', 'dotenv_path': '.env'},
            'data_preprocessing': {'preprocessed_data_path': str(preprocessed_path), 'target_column': 'text'},
            'schema': {'spec_path': str(base_schema_path), 'container_name': 'root'},
        }
        config_c_path = tmp_root / "config_c.yaml"
        yaml.safe_dump(config_c, open(config_c_path, 'w'))
        exp_c_dir = tmp_root / "exp_c"
        exp_c_dir.mkdir()
        try:
            delm_c = run_delm_with_config(config_c_path, "exp_c", exp_c_dir)
            delm_c.prep_data(make_df())
            assert False, "preprocessed_data_path + other fields: Should have raised ConfigurationError."
        except ConfigurationError as e:
            print("[PASS] preprocessed_data_path + other fields: Correctly raised ConfigurationError.")

        # --- 4. Use invalid feather file (should error) ---
        invalid_feather_path = make_invalid_feather(tmp_root / "invalid.feather")
        config_d = {
            'llm_extraction': {'provider': 'openai', 'name': 'gpt-4o-mini', 'dotenv_path': '.env'},
            'data_preprocessing': {'preprocessed_data_path': str(invalid_feather_path)},
            'schema': {'spec_path': str(base_schema_path), 'container_name': 'root'},
        }
        config_d_path = tmp_root / "config_d.yaml"
        yaml.safe_dump(config_d, open(config_d_path, 'w'))
        exp_d_dir = tmp_root / "exp_d"
        exp_d_dir.mkdir()
        try:
            delm_d = run_delm_with_config(config_d_path, "exp_d", exp_d_dir)
            delm_d.prep_data(make_df())
            assert False, "invalid feather: Should have raised ConfigurationError."
        except ConfigurationError as e:
            print("[PASS] invalid feather: Correctly raised ConfigurationError.")

        # --- 5. Change model/schema config, use preprocessed_data_path (should succeed) ---
        schema2_path = tmp_root / "schema_b.yaml"
        schema2_path.write_text('type: object\nproperties:\n  bar:\n    type: string\n')
        config_e = {
            'llm_extraction': {'provider': 'openai', 'name': 'gpt-4o-mini'},
            'data_preprocessing': {'preprocessed_data_path': str(preprocessed_path)},
            'schema': {'spec_path': str(schema2_path), 'container_name': 'root'},
        }
        config_e_path = tmp_root / "config_e.yaml"
        yaml.safe_dump(config_e, open(config_e_path, 'w'))
        exp_e_dir = tmp_root / "exp_e"
        exp_e_dir.mkdir()
        delm_e = run_delm_with_config(config_e_path, "exp_e", exp_e_dir)
        try:
            delm_e.prep_data(make_df())
            print("[PASS] model/schema change + preprocessed_data_path: accepted valid feather file.")
        except Exception as e:
            assert False, f"model/schema change + preprocessed_data_path: Unexpected error: {e}"

        print("\n[SUMMARY] All cross-experiment preprocessed data tests passed.")

    finally:
        shutil.rmtree(tmp_root)
        print(f"[INFO] Cleaned up temp root: {tmp_root}")

if __name__ == "__main__":
    main() 