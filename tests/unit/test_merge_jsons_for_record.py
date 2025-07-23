from delm.utils.json_match_tree import merge_jsons_for_record
from delm.schemas.schemas import NestedSchema, SimpleSchema, MultipleSchema

source_simple_schema_config = {
    "schema_type": "simple",
    "variables": [
        {"name": "source", "data_type": "string", "required": True, "description": ""},
        {"name": "ratings", "data_type": "[integer]", "required": False, "description": ""},
    ]
}

books_nested_schema_config = {
    "schema_type": "nested",
    "container_name": "books",
    "variables": [
        {"name": "title", "data_type": "string", "required": True, "description": ""},
        {"name": "author", "data_type": "string", "required": True, "description": ""},
        {"name": "sales", "data_type": "[integer]", "required": False, "description": ""},
    ]
}

books_multiple_schema_config = {
    "schema_type": "multiple",
    "info": {
        "schema_type": "simple",
        "variables": [
            {"name": "source", "data_type": "string", "required": True, "description": ""},
            {"name": "ratings", "data_type": "[integer]", "required": False, "description": ""},
        ]
    },
    "books": {
        "schema_type": "nested",
        "container_name": "entries",
        "variables": [
            {"name": "title", "data_type": "string", "required": True, "description": ""},
            {"name": "author", "data_type": "string", "required": True, "description": ""},
            {"name": "sales", "data_type": "[integer]", "required": False, "description": ""},
        ]
    }
}

source_simple_schema = SimpleSchema(source_simple_schema_config)
books_nested_schema = NestedSchema(books_nested_schema_config)
books_multiple_schema = MultipleSchema(books_multiple_schema_config)

def test_merge_simple():
    input_jsons = [
        {"source": "A", "ratings": [1]},
        {"source": "B", "ratings": [2, 4]},
        {"source": "B", "ratings": [3]},
    ]
    merged = merge_jsons_for_record(input_jsons, source_simple_schema)
    print("MERGED (simple):", merged)

def test_merge_nested():
    input_jsons = [
        {"books": [
            {"title": "A", "author": "X", "sales": [1]},
            {"title": "B", "author": "Y", "sales": [2]},
        ]},
        {"books": [
            {"title": "C", "author": "Z", "sales": [3]}
        ]},
    ]
    merged = merge_jsons_for_record(input_jsons, books_nested_schema)
    print("MERGED (nested):", merged)


def test_merge_multiple():
    input_jsons = [
        {
            "info": {"source": "A", "ratings": [1]},
            "books": [
                {"title": "A", "author": "X", "sales": [1]},
                {"title": "B", "author": "Y", "sales": [2]},
            ]
        },
        {
            "info": {"source": "A", "ratings": [3]},
            "books": [
                {"title": "C", "author": "Z", "sales": [3]},
            ]
        }
    ]
    merged = merge_jsons_for_record(input_jsons, books_multiple_schema)
    print("MERGED (multiple):", merged)

# def test_merge_empty():
#     input_jsons = []
#     merged = merge_jsons_for_record(input_jsons, books_multiple_schema)
#     print("MERGED (empty):", merged)
#     assert merged == {"books": []}

if __name__ == "__main__":
    test_merge_simple()
    test_merge_nested()
    test_merge_multiple()
    # test_merge_empty() 