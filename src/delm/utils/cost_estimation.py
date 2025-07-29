from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

from delm.delm import DELM
from delm.constants import (
    DEFAULT_PROMPT_TEMPLATE, 
    DEFAULT_SYSTEM_PROMPT, 
    SYSTEM_CHUNK_COLUMN, 
    SYSTEM_RANDOM_SEED,
    DEFAULT_LOG_DIR,
    DEFAULT_CONSOLE_LOG_LEVEL,
    DEFAULT_FILE_LOG_LEVEL,
    SYSTEM_LOG_FILE_PREFIX,
    SYSTEM_LOG_FILE_SUFFIX,
)
from delm.config import DELMConfig

# Module-level logger
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Cost Estimation Methods                                                            #
# --------------------------------------------------------------------------- #

def estimate_input_token_cost(
    config: Union[str, Dict[str, Any], DELMConfig],
    data_source: str | Path | pd.DataFrame,
    save_file_log: bool = True,
    log_dir: str | Path | None = Path(DEFAULT_LOG_DIR) / "cost_estimation",
    console_log_level: str = DEFAULT_CONSOLE_LOG_LEVEL,
    file_log_level: str = DEFAULT_FILE_LOG_LEVEL,
) -> float:
    """
    Should estimate input token cost based on whole dataset.
    
    Args:
        config: Configuration for the DELM pipeline.
        data_source: Source data for extraction.
        log_file: Optional path to log file. If None, creates {DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE_PREFIX}_cost_estimation_run_<timestamp>.log at project root.
        console_log_level: Log level for console output.
        file_log_level: Log level for file output.
    """
    from delm.logging import configure
    from datetime import datetime
    
    # Configure logging
    if save_file_log:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = f"{SYSTEM_LOG_FILE_PREFIX}cost_estimation_{current_time}{SYSTEM_LOG_FILE_SUFFIX}"
    else:
        log_file_name = None
    
    configure(
        console_level=console_log_level,
        file_dir=log_dir,
        file_name=log_file_name,
        file_level=file_log_level,
    )
    
    log.debug("Estimating input token cost for data source: %s", data_source)
    config_obj = DELMConfig.from_any(config)
    log.debug("Config loaded: %s", config_obj.name if hasattr(config_obj, 'name') else 'unknown')
    
    delm = DELM(
        config=config_obj,
        experiment_name="cost_estimation",
        experiment_directory=Path(),
        overwrite_experiment=False,
        auto_checkpoint_and_resume_experiment=True,
        use_disk_storage=False,
        override_logging=False,
    )
    log.debug("DELM instance created for cost estimation")
    
    delm.prep_data(data_source)
    log.debug("Data prepared for cost estimation")
    
    extraction_schema = delm.schema_manager.get_extraction_schema()
    log.debug("Extraction schema loaded: %s", type(extraction_schema).__name__)
    
    system_prompt = delm.config.schema.system_prompt or DEFAULT_SYSTEM_PROMPT
    user_prompt_template = delm.config.schema.prompt_template or DEFAULT_PROMPT_TEMPLATE
    variables_text = extraction_schema.get_variables_text()
    log.debug("Prompt setup: system_length=%d, template_length=%d, variables_length=%d", 
             len(system_prompt), len(user_prompt_template), len(variables_text))
    
    total_input_tokens = 0
    chunks = delm.experiment_manager.load_preprocessed_data()[SYSTEM_CHUNK_COLUMN].tolist()
    log.debug("Processing %d chunks for token estimation", len(chunks))
    
    for i, chunk in enumerate(chunks):
        formatted_prompt = user_prompt_template.format(variables=variables_text, text=chunk)
        complete_prompt = f"{system_prompt}\n\n{formatted_prompt}"
        prompt_tokens = delm.cost_tracker.count_tokens(complete_prompt)
        total_input_tokens += prompt_tokens
        if i % 100 == 0:  # Log progress every 100 chunks
            log.debug("Processed %d/%d chunks, total tokens so far: %d", i + 1, len(chunks), total_input_tokens)
    
    input_price_per_1M = delm.cost_tracker.model_input_cost_per_1M_tokens
    total_cost = total_input_tokens * input_price_per_1M / 1_000_000
    
    log.debug("Input token cost estimation completed: %d total tokens, $%.6f total cost", total_input_tokens, total_cost)
    return total_cost
    


def estimate_total_cost(
    config: Union[str, Dict[str, Any], DELMConfig],
    data_source: str | Path | pd.DataFrame,
    sample_size: int = 10,
    save_file_log: bool = True,
    log_dir: str | Path | None = Path(DEFAULT_LOG_DIR) / "cost_estimation",
    console_log_level: str = DEFAULT_CONSOLE_LOG_LEVEL,
    file_log_level: str = DEFAULT_FILE_LOG_LEVEL,
) -> float:
    """
    Estimate total cost using API calls on a sample of the data.
    
    Args:
        config: Configuration for the DELM pipeline.
        data_source: Source data for extraction.
        sample_size: Number of records to sample for cost estimation.
        log_file: Optional path to log file. If None, creates {DEFAULT_LOG_DIR}/{DEFAULT_LOG_FILE_PREFIX}_cost_estimation_run_<timestamp>.log at project root.
        console_log_level: Log level for console output.
        file_log_level: Log level for file output.
    """
    from delm.logging import configure
    from datetime import datetime
    
    # Configure logging
    if save_file_log:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = f"{SYSTEM_LOG_FILE_PREFIX}cost_estimation_{current_time}{SYSTEM_LOG_FILE_SUFFIX}"
    else:
        log_file_name = None
    
    configure(
        console_level=console_log_level,
        file_dir=log_dir,
        file_name=log_file_name,
        file_level=file_log_level,
    )
    
    log.warning("This method will use the API to estimate the cost. This will charge you for the sampled data requests.")
    
    log.debug("Estimating total cost with API calls: data_source=%s, sample_size=%d", data_source, sample_size)
    config_obj = DELMConfig.from_any(config)
    log.debug("Config loaded: %s", config_obj.name if hasattr(config_obj, 'name') else 'unknown')
    
    delm = DELM(
        config=config_obj,
        experiment_name="cost_estimation",
        experiment_directory=Path(),
        overwrite_experiment=False,
        auto_checkpoint_and_resume_experiment=True,
        use_disk_storage=False,
        override_logging=False,
    )
    log.debug("DELM instance created for API cost estimation")
    
    delm.cost_tracker.count_cache_hits_towards_cost = True
    log.debug("Cache hits will be counted towards cost")

    records_df = delm.data_processor.load_data(data_source)
    total_records = len(records_df)
    log.debug("Loaded %d total records from data source", total_records)
    
    sample_records_df = records_df.sample(n=sample_size, random_state=SYSTEM_RANDOM_SEED)
    log.debug("Sampled %d records for cost estimation", len(sample_records_df))
    
    sample_chunks_df = delm.data_processor.process_dataframe(sample_records_df)
    log.debug("Processed sample records into %d chunks", len(sample_chunks_df))
    
    delm.experiment_manager.save_preprocessed_data(sample_chunks_df)
    log.debug("Saved preprocessed sample data")

    log.debug("Starting LLM processing for cost estimation")
    delm.process_via_llm()
    log.debug("LLM processing completed")

    sample_cost = delm.cost_tracker.get_current_cost()
    total_estimated_cost = sample_cost * (total_records / sample_size)
    
    log.debug("Total cost estimation completed: sample_cost=$%.6f, total_estimated_cost=$%.6f", 
             sample_cost, total_estimated_cost)
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