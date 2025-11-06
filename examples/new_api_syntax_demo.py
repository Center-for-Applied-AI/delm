"""
Demonstration of the new DELM API Syntax

This script shows the API syntax without requiring API keys.
"""

from delm import variable, simple_schema, nested_schema, multiple_schema

print("=" * 70)
print("NEW DELM API - Syntax Demonstration")
print("=" * 70)
print()

# =============================================================================
# Example 1: Schema Helper Functions
# =============================================================================

print("=" * 70)
print("1. Schema Helper Functions")
print("=" * 70)
print()

# Simple schema
print("Simple schema with helper functions:")
print("```python")
print("schema = simple_schema(")
print("    variable('company', 'Company name', 'string'),")
print("    variable('price', 'Price value', 'number'),")
print(")")
print("```")
print()

schema = simple_schema(
    variable("company", "Company name mentioned in text", "string"),
    variable("price", "Price value in USD", "number"),
)
print(f"✓ Created simple schema with {len(schema['variables'])} variables")
print()

# Nested schema
print("Nested schema for extracting lists:")
print("```python")
print("schema = nested_schema(")
print("    'companies',  # Container name")
print("    variable('name', 'Company name', 'string', required=True),")
print("    variable('revenue', 'Revenue', 'number'),")
print(")")
print("```")
print()

nested = nested_schema(
    "companies",
    variable("name", "Company name", "string", required=True),
    variable("revenue", "Annual revenue", "number"),
)
print(f"✓ Created nested schema with container '{nested['container_name']}'")
print()

# Multiple schemas
print("Multiple schemas for different entity types:")
print("```python")
print("products = nested_schema('products', ...)")
print("companies = nested_schema('companies', ...)")
print("schema = multiple_schema(products=products, companies=companies)")
print("```")
print()

products = nested_schema(
    "products",
    variable("name", "Product name", "string", required=True),
    variable("price", "Price", "number"),
)

companies = nested_schema(
    "companies",
    variable("name", "Company name", "string", required=True),
    variable("revenue", "Revenue", "number"),
)

multi = multiple_schema(products=products, companies=companies)
print(f"✓ Created multiple schema with {len([k for k in multi.keys() if k != 'schema_type'])} sub-schemas")
print()

# =============================================================================
# Example 2: DELM Initialization Patterns
# =============================================================================

print("=" * 70)
print("2. DELM Initialization Patterns")
print("=" * 70)
print()

print("Basic initialization:")
print("```python")
print("from delm import DELM")
print()
print("delm = DELM(")
print("    provider='openai',")
print("    model='gpt-4o-mini',")
print("    schema=schema,")
print(")")
print("```")
print()

print("With common parameters:")
print("```python")
print("delm = DELM(")
print("    provider='openai',")
print("    model='gpt-4o-mini',")
print("    schema=schema,")
print("    temperature=0.0,")
print("    batch_size=10,")
print("    max_workers=4,")
print("    max_budget=5.0,")
print(")")
print("```")
print()

print("With text processing:")
print("```python")
print("delm = DELM(")
print("    provider='openai',")
print("    model='gpt-4o-mini',")
print("    schema=schema,")
print("    splitting='paragraph',")
print("    scoring=['price', 'revenue', 'forecast'],")
print("    score_filter='delm_score > 0.5',")
print("    target_column='text',")
print(")")
print("```")
print()

# =============================================================================
# Example 3: Extraction Methods
# =============================================================================

print("=" * 70)
print("3. Extraction Methods")
print("=" * 70)
print()

print("Single-step extraction (recommended):")
print("```python")
print("import pandas as pd")
print()
print("df = pd.DataFrame({'text': ['Apple stock', 'Microsoft revenue']})")
print("result = delm.extract(df)")
print()
print("# Access results")
print("print(result)  # ExtractionResult(records=2, cost=$0.0023)")
print("print(result.data)  # DataFrame with extracted data")
print("print(result.cost)  # Cost dictionary")
print("```")
print()

print("Two-step processing (advanced):")
print("```python")
print("# Step 1: Prep data")
print("prepped = delm.prep_data(df)")
print("print(prepped[['delm_chunk_id', 'delm_score']].head())")
print()
print("# Step 2: Process through LLM")
print("result_df = delm.process_via_llm()")
print("cost = delm.get_cost_summary()")
print("```")
print()

# =============================================================================
# Example 4: Loading from Config
# =============================================================================

print("=" * 70)
print("4. Loading from Config")
print("=" * 70)
print()

print("Load config with overrides:")
print("```python")
print("delm = DELM.from_config(")
print("    'config.yaml',")
print("    temperature=0.5,")
print("    experiment='test_run_v2',")
print("    max_budget=10.0,")
print(")")
print("```")
print()

# =============================================================================
# Example 5: Variable Options
# =============================================================================

print("=" * 70)
print("5. Variable Options")
print("=" * 70)
print()

print("Variable with constraints:")
print("```python")
print("variable(")
print("    'sector',")
print("    'Business sector',")
print("    'string',")
print("    required=True,")
print("    allowed_values=['tech', 'finance', 'healthcare'],")
print("    validate_in_text=True,")
print(")")
print("```")
print()

var_with_constraints = variable(
    "sector",
    "Business sector",
    "string",
    required=True,
    allowed_values=["tech", "finance", "healthcare"],
    validate_in_text=True,
)
print(f"✓ Created variable with {len(var_with_constraints.allowed_values)} allowed values")
print()

print("List variable:")
print("```python")
print("variable('products', 'List of products', '[string]')")
print("```")
print()

list_var = variable("products", "List of products", "[string]")
print(f"✓ Created list variable (data_type: {list_var.data_type})")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 70)
print("Summary")
print("=" * 70)
print()

print("Key Features:")
print("  ✓ Simple helper functions for schema creation")
print("  ✓ Direct parameter passing (no nested config objects)")
print("  ✓ Single extract() method for most use cases")
print("  ✓ String shortcuts for splitting and scoring")
print("  ✓ from_config() with parameter overrides")
print("  ✓ ExtractionResult with data, cost, and statistics")
print("  ✓ Full backward compatibility with old API")
print()

print("=" * 70)
print("Demo completed!")
print("=" * 70)

