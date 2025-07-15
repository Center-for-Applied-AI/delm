"""
Direct tests for DataProcessor class (chunking, scoring, filtering, edge cases)
"""

import pandas as pd
from pathlib import Path

from delm.core.data_processor import DataProcessor
from delm.config import DataConfig, SplittingConfig, ScoringConfig
from delm.constants import SYSTEM_CHUNK_COLUMN, SYSTEM_CHUNK_ID_COLUMN, SYSTEM_SCORE_COLUMN
from delm.strategies.splitting_strategies import SplitStrategy
from delm.strategies.scoring_strategies import RelevanceScorer

# --- Mock strategies ---
class MockSplitter(SplitStrategy):
    def split(self, text):
        # Split on ". " for demo
        return text.split(". ")

class MockScorer(RelevanceScorer):
    def score(self, text):
        # Score is length of text (for demo)
        return len(text)

# --- Test Data ---
data = pd.DataFrame({
    "id": [1, 2],
    "text": [
        "This is the first sentence. Here is another. And a third.",
        "Second row only has one sentence."
    ]
})

print("Original DataFrame:")
print(data)

# --- 1. Test with splitting and scoring ---
print("\nTest: With splitting and scoring")
config = DataConfig(
    target_column="text",
    drop_target_column=True,
    splitting=SplittingConfig(strategy=MockSplitter()),
    scoring=ScoringConfig(scorer=MockScorer()),
    pandas_score_filter=None
)
processor = DataProcessor(config)
df1 = processor.load_and_process(data)
print(df1)

# --- 2. Test with splitting, scoring, and filtering ---
print("\nTest: With splitting, scoring, and filtering (score > 20)")
config2 = DataConfig(
    target_column="text",
    drop_target_column=True,
    splitting=SplittingConfig(strategy=MockSplitter()),
    scoring=ScoringConfig(scorer=MockScorer()),
    pandas_score_filter=f"{SYSTEM_SCORE_COLUMN} > 20"
)
processor2 = DataProcessor(config2)
df2 = processor2.load_and_process(data)
print(df2)

# --- 3. Test with no splitting, but scoring ---
print("\nTest: No splitting, but scoring")
config3 = DataConfig(
    target_column="text",
    drop_target_column=False,
    splitting=SplittingConfig(strategy=None),
    scoring=ScoringConfig(scorer=MockScorer()),
    pandas_score_filter=None
)
processor3 = DataProcessor(config3)
df3 = processor3.load_and_process(data)
print(df3)

# --- 4. Test with no splitting, no scoring ---
print("\nTest: No splitting, no scoring")
config4 = DataConfig(
    target_column="text",
    drop_target_column=False,
    splitting=SplittingConfig(strategy=None),
    scoring=ScoringConfig(scorer=None),
    pandas_score_filter=None
)
processor4 = DataProcessor(config4)
df4 = processor4.load_and_process(data)
print(df4)

# --- 5. Test error: drop_target_column True but no splitter ---
print("\nTest: Error case (drop_target_column True, no splitter)")
try:
    config5 = DataConfig(
        target_column="text",
        drop_target_column=True,
        splitting=SplittingConfig(strategy=None),
        scoring=ScoringConfig(scorer=None),
        pandas_score_filter=None
    )
    processor5 = DataProcessor(config5)
    df5 = processor5.load_and_process(data)
    print(df5)
except Exception as e:
    print(f"Expected error: {e}") 