from typing import List
from .cost_tracker import CostTracker
from ..constants import SYSTEM_CHUNK_COLUMN
from ..schemas import SchemaManager
from ..core.extraction_manager import ExtractionManager

class CostEstimator:
    def __init__(self, config, sample_df, total_chunks, total_records, cost_tracker, schema_manager):
        self.config = config
        self.sample_df = sample_df
        self.total_chunks = total_chunks
        self.total_records = total_records 
        self.cost_tracker = cost_tracker
        self.schema_manager = schema_manager
        self.extraction_schema = self.schema_manager.get_extraction_schema()

    def estimate_cost(self, use_api_calls: bool = False):
        if use_api_calls:
            return self._estimate_with_api()
        else:
            return self._estimate_heuristic()

    def _estimate_heuristic(self):
        sample_text_chunks = self.sample_df[SYSTEM_CHUNK_COLUMN].tolist() 
        sample_input_tokens = self._get_input_tokens(sample_text_chunks)
        full_dataset_input_tokens = int(sample_input_tokens * self.total_chunks / len(self.sample_df))
        # Pricing (input tokens only)
        input_price_per_1M = self.cost_tracker.model_input_cost_per_1M_tokens
        total_cost = self.cost_tracker.estimate_cost(full_dataset_input_tokens, 0)
        return {
            "estimated_total_cost": total_cost,
            "cost_per_delm_chunk": total_cost / self.total_chunks,
            "cost_per_record": total_cost / self.total_records,
            "total_chunks": self.total_chunks,
            "total_records": self.total_records,
            "sample_input_tokens": sample_input_tokens,
            "estimated_total_input_tokens": full_dataset_input_tokens,
            "input_price_per_1M_tokens": input_price_per_1M,
            "method": "heuristic",
            "sample_size": len(self.sample_df),
            "warning": "Output token cost is not estimated in heuristic mode. Set use_api_calls=True to estimate output token cost. This will be more accurate but it will charge you for the sampled data requests."
        }

    def _get_input_tokens(self, text_chunks: List[str]) -> int:
        system_prompt = self.config.schema.system_prompt
        user_prompt_template = self.config.schema.prompt_template
        
        # Get variables text if schema is available
        variables_text = ""
        if self.extraction_schema:
            variables_text = self.extraction_schema.get_variables_text()
        
        total_input_tokens = 0
        for chunk in text_chunks:
            # Format the complete prompt with variables included
            formatted_prompt = user_prompt_template.format(variables=variables_text, text=chunk)
            complete_prompt = f"{system_prompt}\n\n{formatted_prompt}"
            prompt_tokens = self.cost_tracker.count_tokens(complete_prompt)
            total_input_tokens += prompt_tokens
        
        return total_input_tokens

    def _estimate_output_tokens_heuristic(self, text_chunks: List[str]) -> int:
        raise NotImplementedError("Output token estimation is not supported in heuristic mode. Set use_api_calls=True to estimate output token cost. This will be more accurate but it will charge you for the sampled data requests.")

    def _estimate_with_api(self):
        # Call extraction manager with the CostTracker and the sample data and extrapolate to the full dataset
        extraction_manager = ExtractionManager(
            model_config=self.config.llm_extraction,
            schema_manager=self.schema_manager,
            cost_tracker=self.cost_tracker
        )

        extraction_manager.extract_from_text_chunks(self.sample_df[SYSTEM_CHUNK_COLUMN].tolist())
        sample_cost = self.cost_tracker.get_current_cost()
        sample_input_tokens = self.cost_tracker.input_tokens
        # Get the total output tokens from the cost tracker
        sample_output_tokens = self.cost_tracker.output_tokens
        
        # Extrapolate to the full dataset
        full_dataset_input_tokens = int(sample_input_tokens * self.total_chunks / len(self.sample_df))
        full_dataset_output_tokens = int(sample_output_tokens * self.total_chunks / len(self.sample_df))
        full_dataset_cost = self.cost_tracker.estimate_cost(full_dataset_input_tokens, full_dataset_output_tokens)

        input_price_per_1M = self.cost_tracker.model_input_cost_per_1M_tokens
        output_price_per_1M = self.cost_tracker.model_output_cost_per_1M_tokens

        return {
            "estimated_total_cost": full_dataset_cost,
            "cost_per_delm_chunk": full_dataset_cost / self.total_chunks,
            "cost_per_record": full_dataset_cost / self.total_records,
            "total_chunks": self.total_chunks,
            "total_records": self.total_records,
            "sample_input_tokens": sample_input_tokens,
            "sample_output_tokens": sample_output_tokens,
            "estimated_total_input_tokens": full_dataset_input_tokens,
            "estimated_total_output_tokens": full_dataset_output_tokens,
            "input_price_per_1M": input_price_per_1M,
            "output_price_per_1M": output_price_per_1M,
            "method": "api",
            "sample_size": len(self.sample_df),
            "sample_cost": sample_cost
        }