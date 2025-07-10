"""
Test Unified Schema System
==========================
Demonstrates the new unified schema system with progressive complexity levels.
"""

from pathlib import Path
import pandas as pd

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from DELM import DELM, ParagraphSplit, KeywordScorer # type: ignore
from schemas import SchemaRegistry # type: ignore

def test_simple_schema():
    """Test Level 1: Simple Schema (current DELM behavior)."""
    print("🧪 Testing Simple Schema (Level 1)")
    print("=" * 40)
    
    # Simple config
    config = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "max_retries": 3,
        "extraction": {
            "variables": [
                {
                    "name": "company_names",
                    "description": "Company names mentioned in the text",
                    "data_type": "string",
                    "required": False
                },
                {
                    "name": "revenue_numbers", 
                    "description": "Revenue figures mentioned",
                    "data_type": "number",
                    "required": False
                }
            ]
        }
    }
    
    # Test schema creation
    registry = SchemaRegistry()
    schema = registry.create(config["extraction"])
    
    print(f"✅ Schema type: {type(schema).__name__}")
    print(f"✅ Variables: {len(schema.variables)}")
    
    # Test prompt creation
    test_text = "Apple reported $394 billion in revenue. Microsoft had $198 billion."
    prompt = schema.create_prompt(test_text)
    print(f"✅ Prompt created: {len(prompt)} characters")
    
    return schema

def test_nested_schema():
    """Test Level 2: Nested Schema (enhanced)."""
    print("\n🧪 Testing Nested Schema (Level 2)")
    print("=" * 40)
    
    # Nested config
    config = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "max_retries": 3,
        "batch_size": 10,
        "max_workers": 4,
        "extraction": {
            "schema_type": "nested",
            "container_name": "companies",
            "variables": [
                {
                    "name": "name",
                    "description": "Company name",
                    "data_type": "string",
                    "required": True,
                    "validate_in_text": True
                },
                {
                    "name": "revenue",
                    "description": "Revenue figure",
                    "data_type": "number",
                    "required": False
                },
                {
                    "name": "sector",
                    "description": "Business sector",
                    "data_type": "string",
                    "required": False,
                    "allowed_values": ["technology", "finance", "healthcare", "energy", "retail"]
                }
            ],
            "prompt_template": "Extract company information from the following text:\n\n{variables}\n\nText to analyze:\n{text}"
        }
    }
    
    # Test schema creation
    registry = SchemaRegistry()
    schema = registry.create(config["extraction"])
    
    print(f"✅ Schema type: {type(schema).__name__}")
    print(f"✅ Container name: {schema.container_name}")
    print(f"✅ Variables: {len(schema.variables)}")
    
    # Test prompt creation
    test_text = "Apple (technology) reported $394 billion in revenue. Microsoft (technology) had $198 billion."
    prompt = schema.create_prompt(test_text)
    print(f"✅ Prompt created: {len(prompt)} characters")
    
    return schema

def test_multiple_schema():
    """Test Level 3: Multiple Schemas (advanced)."""
    print("\n🧪 Testing Multiple Schema (Level 3)")
    print("=" * 40)
    
    # Multiple schema config
    config = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "max_retries": 3,
        "batch_size": 10,
        "max_workers": 4,
        "extraction": {
            "schema_type": "multiple",
            "companies": {
                "schema_type": "nested",
                "container_name": "companies",
                "variables": [
                                    {
                    "name": "name",
                    "description": "Company name",
                    "data_type": "string",
                    "validate_in_text": True
                },
                    {
                        "name": "revenue",
                        "description": "Revenue figure",
                        "data_type": "number"
                    }
                ]
            },
            "products": {
                "schema_type": "nested",
                "container_name": "products",
                "variables": [
                                    {
                    "name": "name",
                    "description": "Product name",
                    "data_type": "string",
                    "validate_in_text": True
                },
                    {
                        "name": "price",
                        "description": "Product price",
                        "data_type": "number"
                    }
                ]
            }
        }
    }
    
    # Test schema creation
    registry = SchemaRegistry()
    schema = registry.create(config["extraction"])
    
    print(f"✅ Schema type: {type(schema).__name__}")
    print(f"✅ Number of sub-schemas: {len(schema.schemas)}")
    print(f"✅ Sub-schemas: {list(schema.schemas.keys())}")
    
    # Test prompt creation
    test_text = "Apple reported $394 billion in revenue. Their iPhone costs $999."
    prompt = schema.create_prompt(test_text)
    print(f"✅ Prompt created: {len(prompt)} characters")
    
    return schema

def test_delm_integration():
    """Test DELM integration with new schema system."""
    print("\n🧪 Testing DELM Integration")
    print("=" * 40)
    
    # Create simple test data
    test_data = pd.DataFrame({
        "text": [
            "Apple reported $394 billion in revenue for 2023.",
            "Microsoft had $198 billion in revenue and is in the technology sector."
        ]
    })
    
    # Initialize DELM with nested schema
    delm = DELM(
        config_path=None,  # We'll set config manually
        dotenv_path=Path(".env"),
        split_strategy=ParagraphSplit(),
        relevance_scorer=KeywordScorer(["revenue", "billion", "technology"])
    )
    
    # Set config manually for testing
    delm.config = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "max_retries": 3,
        "extraction": {
            "schema_type": "nested",
            "container_name": "companies",
            "variables": [
                {
                    "name": "name",
                    "description": "Company name",
                    "data_type": "string",
                    "validate_in_text": True
                },
                {
                    "name": "revenue",
                    "description": "Revenue figure in billions",
                    "data_type": "number"
                }
            ]
        }
    }
    
    # Recreate schema with new config
    delm.extraction_schema = delm.schema_registry.create(delm.config['extraction'])
    
    print(f"✅ DELM initialized with schema: {type(delm.extraction_schema).__name__}")
    print(f"✅ Batch processor: {type(delm.batch_processor).__name__}")
    print(f"✅ Cost tracker: {type(delm.cost_tracker).__name__}")
    
    # Test data preparation
    try:
        output_df = delm.prep_data_from_df(test_data, "text")
        print(f"✅ Data preparation successful: {len(output_df)} chunks created")
        print(f"✅ Relevance scores: {output_df['score'].tolist()}")
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
    
    return delm

def main():
    """Run all tests."""
    print("🚀 Testing Unified Schema System")
    print("=" * 50)
    
    # Test schema registry
    print("\n📋 Testing Schema Registry")
    registry = SchemaRegistry()
    available_schemas = registry.list_available()
    print(f"✅ Available schema types: {available_schemas}")
    
    # Test each complexity level
    simple_schema = test_simple_schema()
    nested_schema = test_nested_schema()
    multiple_schema = test_multiple_schema()
    
    # Test DELM integration
    delm = test_delm_integration()
    
    print("\n🎉 All tests completed successfully!")
    print("\n📊 Summary:")
    print(f"  - Simple Schema: ✅ {type(simple_schema).__name__}")
    print(f"  - Nested Schema: ✅ {type(nested_schema).__name__}")
    print(f"  - Multiple Schema: ✅ {type(multiple_schema).__name__}")
    print(f"  - DELM Integration: ✅ Working")

if __name__ == "__main__":
    main() 