"""
Demonstration of the new DELM API

This script shows the simplified API for using DELM.
"""

import pandas as pd
from delm import DELM, variable, simple_schema, nested_schema

# =============================================================================
# Example 1: Simplest Possible Usage
# =============================================================================

print("=" * 70)
print("Example 1: Simplest Possible Usage")
print("=" * 70)

# Define what to extract
schema = simple_schema(
    variable("company", "Company name mentioned in text", "string"),
    variable("price", "Price value in USD", "number"),
)

# Initialize with just essentials
delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    use_disk_storage=False,  # Use in-memory storage for demo
)

# Create sample data
df = pd.DataFrame({
    "text": [
        "Apple stock is $150.",
        "Microsoft revenue is $50B."
    ]
})

# Extract (NOTE: This would make API calls)
# result = delm.extract(df)
# print(f"Extracted {result.num_records} records")
# print(f"Cost: ${result.cost['total_cost']:.4f}")
# print(result.data[["company", "price"]])

print("✓ Schema defined")
print("✓ DELM initialized")
print("✓ Ready to extract (API key required)")
print()

# =============================================================================
# Example 2: Nested Schema
# =============================================================================

print("=" * 70)
print("Example 2: Nested Schema for Multiple Items")
print("=" * 70)

# Extract multiple companies per text chunk
schema = nested_schema(
    "companies",  # Container name
    variable("name", "Company name", "string", required=True),
    variable("revenue", "Annual revenue in USD", "number"),
    variable("products", "List of products", "[string]"),
    variable("growth_rate", "Annual growth rate percentage", "number",
             validate_in_text=True),
)

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    use_disk_storage=False,  # Use in-memory storage for demo
)

print("✓ Nested schema created for extracting multiple companies")
print("✓ DELM initialized")
print()

# =============================================================================
# Example 3: With Text Processing
# =============================================================================

print("=" * 70)
print("Example 3: With Text Processing and Relevance Scoring")
print("=" * 70)

schema = simple_schema(
    variable("company", "Company name", "string"),
    variable("metric", "Financial metric mentioned", "string"),
    variable("value", "Numeric value", "number"),
)

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    
    # Text splitting (string shortcut)
    splitting="paragraph",
    
    # Relevance scoring (keyword list)
    scoring=["price", "revenue", "forecast", "earnings"],
    
    # Filter by score
    score_filter="delm_score > 0.5",
    
    # Which column to process
    target_column="text",
    
    # Experiment tracking
    experiment="financial_metrics_demo",
    use_disk_storage=False,  # Use in-memory storage for demo
)

print("✓ Schema with text processing configured")
print("✓ Splitting strategy: paragraph")
print("✓ Scoring keywords: price, revenue, forecast, earnings")
print("✓ Score filter: delm_score > 0.5")
print()

# =============================================================================
# Example 4: Loading from Config
# =============================================================================

print("=" * 70)
print("Example 4: Loading from Config File with Overrides")
print("=" * 70)

# Example of loading from config (commented out as config file may not exist)
# delm = DELM.from_config(
#     "config.yaml",
#     temperature=0.5,  # Override config value
#     experiment="experiment_v2",  # Override experiment name
#     max_budget=10.0,  # Override budget
# )

print("✓ from_config() method available")
print("✓ Supports parameter overrides")
print()

# =============================================================================
# Example 5: Allowed Values and Validation
# =============================================================================

print("=" * 70)
print("Example 5: Schema with Allowed Values and Text Validation")
print("=" * 70)

schema = simple_schema(
    variable("company", "Company name", "string", required=True),
    variable("sector", "Business sector", "string",
             allowed_values=["tech", "finance", "healthcare", "retail"]),
    variable("sentiment", "Sentiment of the text", "string",
             allowed_values=["positive", "negative", "neutral"]),
)

delm = DELM(
    provider="openai",
    model="gpt-4o-mini",
    schema=schema,
    temperature=0.0,  # Deterministic responses
    use_disk_storage=False,  # Use in-memory storage for demo
)

print("✓ Schema with constrained values")
print("✓ Required fields specified")
print("✓ Allowed values for categorization")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 70)
print("Summary: New API Benefits")
print("=" * 70)

print("""
1. Simple things are simple:
   - Direct parameters instead of nested config objects
   - Helper functions for schema creation
   - Single extract() method for most use cases

2. Complex things are possible:
   - Advanced splitting and scoring strategies
   - Custom prompt templates
   - Budget constraints and cost tracking

3. Everything is type-safe and validated:
   - IDE autocomplete works
   - Type errors caught early
   - Clear validation messages

4. Backward compatible:
   - Old API continues to work
   - Can mix old and new approaches
   - Gradual migration path
""")

print("=" * 70)
print("Demo completed!")
print("=" * 70)

