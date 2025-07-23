from __future__ import annotations

"""
DELM v0.2 – "Phase‑2" implementation
-----------------------------------
Major upgrades from the Phase‑1 prototype:
• Multi‑format loaders (.txt, .html/.md, .docx, .pdf*)
• Pluggable split strategies (ParagraphSplit, FixedWindowSplit, RegexSplit)
• Relevance scoring abstraction (KeywordScorer, FuzzyScorer)
• Structured extraction via `Instructor` with a Pydantic schema fallback
• Built‑in stub fallback when either OPENAI_API_KEY or Instructor is absent
*PDF uses `marker` OCR if available; else raises NotImplementedError.

The public API and method signatures remain unchanged so downstream code
continues to work.  Future phases should require **only** new strategy
classes or loader helpers – no breaking changes.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Sequence

import dotenv
import pandas as pd
import json
from pprint import pprint

# Required deps -------------------------------------------------------------- #
import openai  # type: ignore

from .config import DELMConfig
from .core import DataProcessor, ExperimentManager, ExtractionManager
from .schemas import SchemaManager
from .constants import (
    SYSTEM_CHUNK_COLUMN,
    DEFAULT_RECORD_ID_COLUMN,
    SYSTEM_EXTRACTED_DATA_JSON_COLUMN
)
from .utils.cost_tracker import CostTracker
from .utils.cost_estimator import CostEstimator
from .exceptions import ProcessingError

# --------------------------------------------------------------------------- #
# Main class                                                                  #
# --------------------------------------------------------------------------- #


class DELM:
    """Extraction pipeline with pluggable strategies."""

    def __init__(
        self,
        *,
        config: DELMConfig,
        experiment_name: str,
        experiment_directory: Path,
        overwrite_experiment: bool = False,
        auto_checkpoint_and_resume_experiment: bool = True,
    ) -> None:
        # Config
        self.config = config
        self.experiment_name = experiment_name
        self.experiment_directory = experiment_directory
        self.overwrite_experiment = overwrite_experiment
        self.auto_checkpoint_and_resume_experiment = auto_checkpoint_and_resume_experiment
        self._initialize_components()

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path], experiment_name: str, experiment_directory: Path, **kwargs) -> "DELM":
        """
        Create a DELM instance from a YAML configuration file.
        
        This is the recommended way to create DELM instances for most use cases.
        
        Args:
            config_path: Path to YAML configuration file
            experiment_name: Name of the experiment (used for directory structure, file naming)
            experiment_directory: Base directory for experiment outputs
        Returns:
            Configured DELM instance
        Example:
            delm = DELM.from_yaml("config.yaml", experiment_name="my_exp", experiment_directory=Path("./experiments"))
        """
        config = DELMConfig.from_yaml(Path(config_path))
        return cls(config=config, experiment_name=experiment_name, experiment_directory=experiment_directory, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], experiment_name: str, experiment_directory: Path, **kwargs) -> "DELM":
        """
        Create a DELM instance from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            experiment_name: Name of the experiment (used for directory structure, file naming)
            experiment_directory: Base directory for experiment outputs
        Returns:
            Configured DELM instance
        Example:
            config = {
                "model": {"name": "gpt-4o-mini"},
                "data": {"target_column": "text"},
                "schema": {"spec_path": "schema.yaml"},
            }
            delm = DELM.from_dict(config, experiment_name="my_exp", experiment_directory=Path("./experiments"))
        """
        config = DELMConfig.from_dict(config_dict)
        return cls(config=config, experiment_name=experiment_name, experiment_directory=experiment_directory, **kwargs)

    ## ------------------------------- Public API ------------------------------- ##


    def process_via_llm(self, preprocessed_file_path: Path | None = None) -> pd.DataFrame:
        """Process data through LLM extraction using configuration from constructor, with batch checkpointing and resuming."""
        # Load preprocessed data from feather file
        data = self.experiment_manager.load_preprocessed_data(preprocessed_file_path)
        text_chunks = data[SYSTEM_CHUNK_COLUMN].tolist()
        record_ids = data[self.record_id_column].tolist()
        
        # Process with persistent batching
        final_df = self.extraction_manager.process_with_persistent_batching(
            text_chunks=text_chunks,
            record_ids=record_ids,
            record_id_column=self.record_id_column,
            batch_size=self.config.llm_extraction.batch_size,
            experiment_manager=self.experiment_manager,
            auto_checkpoint=self.auto_checkpoint_and_resume_experiment,
        )
        
        # Print summary
        if self.config.llm_extraction.extract_to_dataframe:
            print(f"Processed {len(data)} chunks from {len(record_ids)} records. Extracted to DataFrame with {len(final_df)} structured rows.")
        else:
            print(f"Processed {len(data)} chunks from {len(record_ids)} records. JSON output saved to `extracted_data` column.")
        
        if self.config.llm_extraction.track_cost:
            self.cost_tracker.print_cost_summary()
        
        return final_df

    
    def prep_data(self, data: str | Path | pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data using the instance config and always save to the experiment directory.
        Returns a DataFrame of prepped (chunked) data.
        """

        save_path = self.experiment_manager.preprocessed_data_path
        df, _ = DELM._prep_data_stateless(self.config, data, save_path=save_path)
        return df

    @staticmethod
    def estimate_cost(
        config: Union[str, Dict[str, Any], 'DELMConfig'],
        data_source: str | Path | pd.DataFrame,
        use_api_calls: bool = False,
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """
        Estimate the extraction cost for the given data and config.
        """
        config_obj = DELMConfig.from_any(config)
        prepped_data, total_records = DELM._prep_data_stateless(config_obj, data_source)
        # Sample data if needed
        if len(prepped_data) > sample_size:
            sample_df = prepped_data.sample(n=sample_size, random_state=42)
        else:
            sample_df = prepped_data
        # --- Delegate to CostEstimator ---
        cost_tracker = CostTracker(
            provider=config_obj.llm_extraction.provider,
            model=config_obj.llm_extraction.name,
        )
        schema_manager = SchemaManager(config_obj.schema)
        estimator = CostEstimator(config_obj, sample_df, total_chunks=len(prepped_data), total_records=total_records, cost_tracker=cost_tracker, schema_manager=schema_manager)
        return estimator.estimate_cost(use_api_calls=use_api_calls)


    @staticmethod
    def estimate_cost_batch(
        configs: Sequence[Union[str, Dict[str, Any], 'DELMConfig']],
        data_source: str | Path | pd.DataFrame,
        use_api_calls: bool = False,
        sample_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Estimate extraction costs for multiple configs on the same dataset.
        """
        # TODO: split into two functions. 1. CalculateInputCost (should use the whole dataset), 2. EstimateTotalCost (uses api calls)
        results = []
        for i, config in enumerate(configs):
            try:
                result = DELM.estimate_cost(config, data_source, use_api_calls, sample_size)
                result["config_index"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "config_index": i,
                    "error": str(e),
                    "estimated_total_cost": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "price_per_1k_input": None,
                    "price_per_1k_output": None,
                    "method": "error",
                    "sample_size": None,
                    "total_chunks": None
                })
        return results


    @staticmethod
    def estimate_performance(
        config: Union[str, Dict[str, Any], 'DELMConfig'],
        data_source: str | Path | pd.DataFrame,
        expected_extraction_output: pd.DataFrame,
        true_json_column: str, 
        record_sample_size: int = 3,
    ) -> tuple[Dict[str, Any], pd.DataFrame]:
        """
        Estimate the performance of the DELM pipeline.
        Returns a dict with both the aggregated_extracted_data and field-level precision/recall metrics.
        """
        from .utils.json_match_tree import aggregate_precision_recall_across_records, merge_jsons_for_record
        
        config = DELMConfig.from_any(config)
        record_id_col = config.data_preprocessing.record_id_column or DEFAULT_RECORD_ID_COLUMN

        prepped_data, _ = DELM._prep_data_stateless(config, data_source)

        # Filter out input data not in expected_extraction_output
        prepped_data = prepped_data[prepped_data[record_id_col].isin(expected_extraction_output[record_id_col])]

        if len(prepped_data) == 0:
            raise ProcessingError("No data to process. There may be no overlap in record_id in input data.")

        # Sample data
        if len(prepped_data) > record_sample_size:
            prepped_data = prepped_data.sample(n=record_sample_size, random_state=42)

        # Pass through ExtractionManager
        config.llm_extraction.extract_to_dataframe = False
        results = DELM._process_via_llm_stateless(config, prepped_data) # type: ignore
        if results.empty or SYSTEM_EXTRACTED_DATA_JSON_COLUMN not in results.columns:
            raise ValueError("No results or missing DICT column.")

        schema_manager = SchemaManager(config.schema)
        extraction_schema = schema_manager.get_extraction_schema()

        # Parse expected JSON column if needed (if user provided as string)
        if isinstance(expected_extraction_output[true_json_column].iloc[0], str):
            expected_extraction_output[true_json_column] = expected_extraction_output[true_json_column].apply(json.loads)
        # Verify that that expected_extraction_output is valid against the schema
        for i, row in expected_extraction_output.iterrows():
            extraction_schema.validate_json_dict(row[true_json_column], path=f"expected_extraction_output[{i}]") # type: ignore
        
        # Group and merge extracted data by record_id using agg to keep dicts as values
        extracted_data_df = (
            results.groupby(record_id_col)[SYSTEM_EXTRACTED_DATA_JSON_COLUMN]
            .agg(lambda x: merge_jsons_for_record(list(x), extraction_schema))
            .reset_index()
        )

        record_id_extracted_expected_dicts_df = pd.merge(
            expected_extraction_output[[record_id_col, true_json_column]],
            extracted_data_df[[record_id_col, SYSTEM_EXTRACTED_DATA_JSON_COLUMN]],
            on=record_id_col,
            how="inner"
        )
        record_id_extracted_expected_dicts_df.columns = [record_id_col, "expected_dict", "extracted_dict"]
        performance_metrics_dict = aggregate_precision_recall_across_records(record_id_extracted_expected_dicts_df["expected_dict"], record_id_extracted_expected_dicts_df["extracted_dict"], extraction_schema) # type: ignore
        return performance_metrics_dict, record_id_extracted_expected_dicts_df



    def get_extraction_results(self) -> pd.DataFrame:
        """
        Get the results from the experiment directory.
        """
        return self.experiment_manager.get_results()

    ## ------------------------------ Private API ------------------------------- ##


    def _initialize_components(self) -> None:
        """Initialize all components using composition."""
        # 
        self.record_id_column = self.config.data_preprocessing.record_id_column or DEFAULT_RECORD_ID_COLUMN
        # Environment & secrets -------------------------------------------- #
        if self.config.llm_extraction.dotenv_path:
            dotenv.load_dotenv(self.config.llm_extraction.dotenv_path)
        
        # Initialize components
        self.data_processor = DataProcessor(self.config.data_preprocessing)
        self.schema_manager = SchemaManager(self.config.schema)
        self.experiment_manager = ExperimentManager(
            experiment_name=self.experiment_name,
            experiment_directory=self.experiment_directory,
            overwrite_experiment=self.overwrite_experiment,
            auto_checkpoint_and_resume_experiment=self.auto_checkpoint_and_resume_experiment,
        )
        
        # Initialize experiment with config and schema
        # Note that config dict contains everything but the schema, which is in schema_dict
        config_dict = self.config.to_config_dict()
        schema_dict = self.config.to_schema_dict()
        self.experiment_manager.initialize_experiment(config_dict, schema_dict)
        
        # Initialize cost tracker (may be loaded from state if resuming)
        self.cost_tracker = CostTracker(
            provider=self.config.llm_extraction.provider,
            model=self.config.llm_extraction.name,
        )
        
        # Load state if resuming
        if self.auto_checkpoint_and_resume_experiment:
            state = self.experiment_manager.load_state()
            if state and "cost_tracker" in state:
                self.cost_tracker = CostTracker.from_dict(state["cost_tracker"])
        
        self.extraction_manager = ExtractionManager(
            self.config.llm_extraction,
            schema_manager=self.schema_manager,  # Pass the instance
            cost_tracker=self.cost_tracker,
        )



    @staticmethod
    def _prep_data_stateless(config: 'DELMConfig', data: str | Path | pd.DataFrame, save_path: str | Path | None = None) -> tuple[pd.DataFrame, int]:
        """
        Preprocess data using the config. Optionally save to disk if save_path is provided.
        Returns a DataFrame of prepped (chunked) data and the total number of records in the original data.
        """
        processor = DataProcessor(config.data_preprocessing)
        df = processor.load_data(data)

        if config.data_preprocessing.record_id_column is None:
            df[DEFAULT_RECORD_ID_COLUMN] = range(len(df))
            config.data_preprocessing.record_id_column = DEFAULT_RECORD_ID_COLUMN

        df = processor.process_dataframe(df)

        if save_path is not None:
            df.to_feather(str(save_path))
        return df, processor.total_records


    @staticmethod
    def _process_via_llm_stateless(config: DELMConfig, data: pd.DataFrame) -> pd.DataFrame:
        """Process data through LLM extraction using configuration from constructor."""
        text_chunks = data[SYSTEM_CHUNK_COLUMN].tolist()
        record_id_column = config.data_preprocessing.record_id_column or DEFAULT_RECORD_ID_COLUMN
        record_ids = data[record_id_column].tolist()
        
        schema_manager = SchemaManager(config.schema)
        cost_tracker = CostTracker(
            provider=config.llm_extraction.provider,
            model=config.llm_extraction.name,
        )
        extraction_manager = ExtractionManager(
            config.llm_extraction,
            schema_manager=schema_manager,
            cost_tracker=cost_tracker,
        )
        results = extraction_manager.extract_from_text_chunks(text_chunks)
        parsed_df = extraction_manager.parse_results_dataframe(
            results=results,
            text_chunks=text_chunks,
            record_ids=record_ids,
            record_id_column=record_id_column,
            output="exploded" if config.llm_extraction.extract_to_dataframe else "json_string_column" # TODO: rename extract_to_dataframe to extract_to_exploded_dataframe
        )
        return parsed_df
