"""Result classes for DELM extraction operations."""

from typing import Any, Dict, Optional
import pandas as pd


class ExtractionResult:
    """Result object from DELM extraction.

    Attributes:
        data: DataFrame with extracted data.
        cost: Cost summary dictionary (if tracking enabled).
        num_records: Number of unique records processed.
        num_chunks: Number of chunks processed.
        num_errors: Number of chunks with errors.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cost: Optional[Dict[str, Any]] = None,
        num_records: int = 0,
        num_chunks: int = 0,
        num_errors: int = 0,
    ):
        """Initialize extraction result.

        Args:
            data: DataFrame with extracted data.
            cost: Cost summary dictionary.
            num_records: Number of unique records processed.
            num_chunks: Number of chunks processed.
            num_errors: Number of chunks with errors.
        """
        self.data = data
        self.cost = cost
        self.num_records = num_records
        self.num_chunks = num_chunks
        self.num_errors = num_errors

    def __repr__(self) -> str:
        """String representation of the result."""
        cost_str = ""
        if self.cost:
            total_cost = self.cost.get("total_cost", 0)
            cost_str = f", cost=${total_cost:.4f}"
        
        return (
            f"ExtractionResult(records={self.num_records}, "
            f"chunks={self.num_chunks}, errors={self.num_errors}{cost_str})"
        )

