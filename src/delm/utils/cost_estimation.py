from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

from delm.delm import DELM
from delm.constants import (
    DEFAULT_PROMPT_TEMPLATE, 
    DEFAULT_SYSTEM_PROMPT, 
    SYSTEM_CHUNK_COLUMN, 
    SYSTEM_RANDOM_SEED
)
from delm.config import DELMConfig

# --------------------------------------------------------------------------- #
# Cost Estimation Methods                                                            #
# --------------------------------------------------------------------------- #

def estimate_input_token_cost(
    config: Union[str, Dict[str, Any], DELMConfig],
    data_source: str | Path | pd.DataFrame,
) -> float:
    """
    Should estimate input token cost based on whole dataset.
    """
    config_obj = DELMConfig.from_any(config)
    delm = DELM(
        config=config_obj,
        experiment_name="cost_estimation",
        experiment_directory=Path(),
        overwrite_experiment=False,
        auto_checkpoint_and_resume_experiment=True,
        use_disk_storage=False,
    )
    delm.prep_data(data_source)
    extraction_schema = delm.schema_manager.get_extraction_schema()
    
    system_prompt = delm.config.schema.system_prompt or DEFAULT_SYSTEM_PROMPT
    user_prompt_template = delm.config.schema.prompt_template or DEFAULT_PROMPT_TEMPLATE
    variables_text = extraction_schema.get_variables_text()
    total_input_tokens = 0
    for chunk in delm.experiment_manager.load_preprocessed_data()[SYSTEM_CHUNK_COLUMN].tolist():
        formatted_prompt = user_prompt_template.format(variables=variables_text, text=chunk)
        complete_prompt = f"{system_prompt}\n\n{formatted_prompt}"
        prompt_tokens = delm.cost_tracker.count_tokens(complete_prompt)
        total_input_tokens += prompt_tokens
    input_price_per_1M = delm.cost_tracker.model_input_cost_per_1M_tokens
    return total_input_tokens * input_price_per_1M / 1_000_000
    


def estimate_total_cost(
    config: Union[str, Dict[str, Any], DELMConfig],
    data_source: str | Path | pd.DataFrame,
    sample_size: int = 10
) -> float:
    """"""
    """"""
    print("[WARNING] This method will use the API to estimate the cost. This will charge you for the sampled data requests.")
    config_obj = DELMConfig.from_any(config)
    delm = DELM(
        config=config_obj,
        experiment_name="cost_estimation",
        experiment_directory=Path(),
        overwrite_experiment=False,
        auto_checkpoint_and_resume_experiment=True,
        use_disk_storage=False,
    )
    delm.cost_tracker.count_cache_hits_towards_cost = True

    records_df = delm.data_processor.load_data(data_source)
    total_records = len(records_df)
    sample_records_df = records_df.sample(n=sample_size, random_state=SYSTEM_RANDOM_SEED)
    sample_chunks_df = delm.data_processor.process_dataframe(sample_records_df)
    delm.experiment_manager.save_preprocessed_data(sample_chunks_df)

    delm.process_via_llm()

    sample_cost = delm.cost_tracker.get_current_cost()
    total_estimated_cost = sample_cost * (total_records / sample_size)

    return total_estimated_cost

# class CostEstimator:
#     def __init__(
#         self, 
#         config: Union[str, Dict[str, Any], DELMConfig],
#         data_source: str | Path | pd.DataFrame,
#         sample_size: int = 10
#     ):
#         self.config = DELMConfig.from_any(config)
#         self.data_source = data_source
#         self.use_api_calls = use_api_calls
#         self.sample_size = sample_size

#     def __initialize_components(self):


#     def estimate_total_cost(self, use_api_calls: bool = False, sample_size: int = 5):
#         if use_api_calls:
#             return self._estimate_with_api()
#         else:
#             return self._estimate_heuristic()

#     def _estimate_heuristic(self):
#         sample_text_chunks = self.sample_df[SYSTEM_CHUNK_COLUMN].tolist() 
#         sample_input_tokens = self._get_input_tokens(sample_text_chunks)
#         full_dataset_input_tokens = int(sample_input_tokens * self.total_chunks / len(self.sample_df))
#         # Pricing (input tokens only)
#         input_price_per_1M = self.cost_tracker.model_input_cost_per_1M_tokens
#         total_cost = self.cost_tracker.estimate_cost(full_dataset_input_tokens, 0)
#         return {
#             "estimated_total_cost": total_cost,
#             "cost_per_delm_chunk": total_cost / self.total_chunks,
#             "cost_per_record": total_cost / self.total_records,
#             "total_chunks": self.total_chunks,
#             "total_records": self.total_records,
#             "sample_input_tokens": sample_input_tokens,
#             "estimated_total_input_tokens": full_dataset_input_tokens,
#             "input_price_per_1M_tokens": input_price_per_1M,
#             "method": "heuristic",
#             "sample_size": len(self.sample_df),
#             "warning": "Output token cost is not estimated in heuristic mode. Set use_api_calls=True to estimate output token cost. This will be more accurate but it will charge you for the sampled data requests."
#         }

#     def _get_input_tokens(self, text_chunks: List[str]) -> int:
#         system_prompt = self.config.schema.system_prompt
#         user_prompt_template = self.config.schema.prompt_template
#         variables_text = ""
#         if self.extraction_schema:
#             variables_text = self.extraction_schema.get_variables_text()
#         total_input_tokens = 0
#         for chunk in text_chunks:
#             formatted_prompt = user_prompt_template.format(variables=variables_text, text=chunk)
#             complete_prompt = f"{system_prompt}\n\n{formatted_prompt}"
#             prompt_tokens = self.cost_tracker.count_tokens(complete_prompt)
#             total_input_tokens += prompt_tokens
#         return total_input_tokens

#     def _estimate_output_tokens_heuristic(self, text_chunks: List[str]) -> int:
#         raise NotImplementedError("Output token estimation is not supported in heuristic mode. Set use_api_calls=True to estimate output token cost. This will be more accurate but it will charge you for the sampled data requests.")

#     def _estimate_with_api(self):
#         # Create a DELM instance with InMemoryExperimentManager for estimation
#         delm = DELM(
#             config=self.config,
#             experiment_name="cost_estimation",
#             experiment_directory=None,
#             overwrite_experiment=False,
#             auto_checkpoint_and_resume_experiment=False,
#             use_disk_storage=False,
#         )
#         # Use DELM to process the sample data #TODO FIX THIS
#         delm.experiment_manager.save_preprocessed_data(self.sample_df)
#         results = delm.process_via_llm()

#         sample_cost = self.cost_tracker.get_current_cost()
#         sample_input_tokens = self.cost_tracker.input_tokens
#         sample_output_tokens = self.cost_tracker.output_tokens
#         full_dataset_input_tokens = int(sample_input_tokens * self.total_chunks / len(self.sample_df))
#         full_dataset_output_tokens = int(sample_output_tokens * self.total_chunks / len(self.sample_df))
#         full_dataset_cost = self.cost_tracker.estimate_cost(full_dataset_input_tokens, full_dataset_output_tokens)
#         input_price_per_1M = self.cost_tracker.model_input_cost_per_1M_tokens
#         output_price_per_1M = self.cost_tracker.model_output_cost_per_1M_tokens
#         return {
#             "estimated_total_cost": full_dataset_cost,
#             "cost_per_delm_chunk": full_dataset_cost / self.total_chunks,
#             "cost_per_record": full_dataset_cost / self.total_records,
#             "total_chunks": self.total_chunks,
#             "total_records": self.total_records,
#             "sample_input_tokens": sample_input_tokens,
#             "sample_output_tokens": sample_output_tokens,
#             "estimated_total_input_tokens": full_dataset_input_tokens,
#             "estimated_total_output_tokens": full_dataset_output_tokens,
#             "input_price_per_1M": input_price_per_1M,
#             "output_price_per_1M": output_price_per_1M,
#             "method": "api",
#             "sample_size": len(self.sample_df),
#             "sample_cost": sample_cost
#         }