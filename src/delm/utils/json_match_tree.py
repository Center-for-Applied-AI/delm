###############################################################################
# json_match_tree.py
###############################################################################
"""
Schema‑aware scoring + merge utilities for DELM.

Key points
----------
* Uses each ExtractionVariable’s `required` flag when computing precision/recall.
  ▸ If **required** and missing → counts as FN/FP.  
  ▸ If **optional** and missing in ground‑truth → field is ignored for that record.
* `_build_required_map(schema)` walks Simple/Nested/Multiple schemas and creates
  a dotted‑path → required? lookup used in every recursion.
* The merge logic (majority‑vote vs list‑concatenation) is unchanged.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Union

from delm.schemas.schemas import BaseSchema, ExtractionVariable, SimpleSchema, NestedSchema

# --------------------------------------------------------------------------- #
# Helpers for missing‑value handling and hashing
# --------------------------------------------------------------------------- #
def _is_missing(val: Any) -> bool:
    """Return True when `val` is semantically ‘no information’."""
    return (
        val is None
        or val == ""
        or (isinstance(val, (list, dict)) and len(val) == 0)
    )

def _make_hashable(val: Any) -> Any:
    """
    Convert lists/dicts to a stable JSON string; return None for missing values
    so they can be filtered out of set calculations.
    """
    if _is_missing(val):
        return None
    if isinstance(val, (list, dict)):
        return json.dumps(val, sort_keys=True)
    return val

# --------------------------------------------------------------------------- #
# Build a dotted‑path → required? map from any schema object
# --------------------------------------------------------------------------- #
def _build_required_map(schema: BaseSchema, parent: List[str] | None = None) -> Dict[str, bool]:
    """
    Walk SimpleSchema / NestedSchema / MultipleSchema to collect `{path: required}`.
    """
    parent = parent or []
    req_map: Dict[str, bool] = {}

    stype = getattr(schema, "schema_type", type(schema).__name__).lower()

    if stype == "simpleschema":
        for var in schema.variables:
            req_map[".".join(parent + [var.name])] = getattr(var, "required", False)

    elif stype == "nestedschema":
        cont = schema.container_name
        for var in schema.variables:
            path = parent + [cont, var.name]
            req_map[".".join(path)] = getattr(var, "required", False)

    elif stype == "multipleschema":
        for name, sub in schema.schemas.items():
            req_map.update(_build_required_map(sub, parent + [name]))

    return req_map

###############################################################################
# Precision / recall computation (schema‑aware)
###############################################################################
def all_levels_precision_recall(
    y_true: Any,
    y_pred: Any,
    required_map: Dict[str, bool],
    key: Optional[str] = None,
    path: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Recursively compute precision / recall at every nested level, obeying
    the `required_map` to skip optional fields that are missing in the label.
    """
    path = path or []
    results: Dict[str, Dict[str, Union[int, float]]] = {}

    # ------------------------------------------------------------------- #
    # Dict branch
    # ------------------------------------------------------------------- #
    if isinstance(y_true, dict) and isinstance(y_pred, dict):
        keys = sorted(set(y_true) | set(y_pred))
        for k in keys:
            sub_path = path + [k]
            t_val, p_val = y_true.get(k), y_pred.get(k)
            pstr = ".".join(sub_path)
            required = required_map.get(pstr, False)

            # Primitive comparison
            if not any(isinstance(v, (dict, list)) for v in (t_val, p_val)):
                if required or not _is_missing(t_val):
                    t_set = {_make_hashable(t_val)} - {None}
                    p_set = {_make_hashable(p_val)} - {None}
                    tp = len(t_set & p_set)
                    fp = len(p_set - t_set)
                    fn = len(t_set - p_set)
                    prec = tp / (tp + fp) if tp + fp else 0.0
                    rec  = tp / (tp + fn) if tp + fn else 0.0
                    results[pstr] = {"precision": prec, "recall": rec,
                                     "tp": tp, "fp": fp, "fn": fn}

            # Recurse into nested structures
            results.update(
                all_levels_precision_recall(t_val, p_val,
                                            required_map, k, sub_path)
            )
        return results

    # ------------------------------------------------------------------- #
    # List branch
    # ------------------------------------------------------------------- #
    if isinstance(y_true, list) and isinstance(y_pred, list):
        true_dicts = [d for d in y_true if isinstance(d, dict)]
        pred_dicts = [d for d in y_pred if isinstance(d, dict)]

        path_str = ".".join(path) if path else "root"
        required = required_map.get(path_str, False)

        if true_dicts or pred_dicts:  # container of dicts
            if required or true_dicts:
                t_set = {json.dumps(d, sort_keys=True) for d in true_dicts}
                p_set = {json.dumps(d, sort_keys=True) for d in pred_dicts}
                tp = len(t_set & p_set)
                fp = len(p_set - t_set)
                fn = len(t_set - p_set)
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec  = tp / (tp + fn) if tp + fn else 0.0
                results[path_str] = {"precision": prec, "recall": rec,
                                     "tp": tp, "fp": fp, "fn": fn}

            # Field‑level within dict list
            key_union = {k for d in true_dicts + pred_dicts for k in d}
            for k in key_union:
                sub_path = path + [k]
                pstr = ".".join(sub_path)
                required = required_map.get(pstr, True)

                t_vals = {_make_hashable(d.get(k)) for d in true_dicts if k in d} - {None}
                p_vals = {_make_hashable(d.get(k)) for d in pred_dicts if k in d} - {None}

                if required or t_vals:
                    tp_f = len(t_vals & p_vals)
                    fp_f = len(p_vals - t_vals)
                    fn_f = len(t_vals - p_vals)
                    prec_f = tp_f / (tp_f + fp_f) if tp_f + fp_f else 0.0
                    rec_f  = tp_f / (tp_f + fn_f) if tp_f + fn_f else 0.0
                    results[pstr] = {"precision": prec_f, "recall": rec_f,
                                     "tp": tp_f, "fp": fp_f, "fn": fn_f}

                # Recurse deeper if nested
                t_nested = [d.get(k) for d in true_dicts if k in d]
                p_nested = [d.get(k) for d in pred_dicts if k in d]
                if any(isinstance(v, (dict, list)) for v in t_nested + p_nested):
                    results.update(
                        all_levels_precision_recall(t_nested, p_nested,
                                                    required_map, k, sub_path)
                    )
            return results

        # list of primitives
        if required or y_true:
            t_set = {_make_hashable(v) for v in y_true} - {None}
            p_set = {_make_hashable(v) for v in y_pred} - {None}
            tp = len(t_set & p_set)
            fp = len(p_set - t_set)
            fn = len(t_set - p_set)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec  = tp / (tp + fn) if tp + fn else 0.0
            results[path_str] = {"precision": prec, "recall": rec,
                                 "tp": tp, "fp": fp, "fn": fn}
        return results

    return results

# --------------------------------------------------------------------------- #
def aggregate_precision_recall_across_records(
    expected_list: List[Any],
    predicted_list: List[Any],
    schema: BaseSchema,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate TP/FP/FN for each field across records, then compute macro
    precision/recall, respecting the schema's required/optional flags.
    """
    required_map = _build_required_map(schema)
    agg = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0})

    for y_true, y_pred in zip(expected_list, predicted_list):
        rec_metrics = all_levels_precision_recall(y_true, y_pred, required_map)
        for field, m in rec_metrics.items():
            agg[field]["tp"] += m["tp"]
            agg[field]["fp"] += m["fp"]
            agg[field]["fn"] += m["fn"]

    for field, c in agg.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        c["precision"] = tp / (tp + fp) if tp + fp else 0.0
        c["recall"]    = tp / (tp + fn) if tp + fn else 0.0

    return dict(agg)

###############################################################################
# Merge‑logic (unchanged, majority vote vs list concat)
###############################################################################
def _is_list_type(var: ExtractionVariable) -> bool:
    """Return True if the ExtractionVariable describes a list."""
    return isinstance(var.data_type, str) and var.data_type.startswith("[")

def _majority_vote(values: List[Any]) -> Any:
    if not values:
        return None
    counts = Counter(values)
    top = max(counts.values())
    for v in values:           # first winner wins
        if counts[v] == top:
            return v
    return values[0]

def merge_jsons_for_record(json_list: List[Dict[str, Any]], schema: BaseSchema):
    """
    Consolidate multiple extraction results for a single record, obeying:
      • Scalars  → majority vote (ties → first encountered)
      • List-types → concatenate, keep duplicates
    """
    if not json_list:
        json_list = []
    if json_list and isinstance(json_list[0], str):
        json_list = [json.loads(j) for j in json_list] # type: ignore

    schema_type = getattr(schema, "schema_type", type(schema).__name__).lower()

    # ------------ SIMPLE ----------------------------------------------------
    if schema_type == "simpleschema":
        merged_simple: Dict[str, Any] = {}
        for schema_var in schema.variables:
            bucket: List[Any] = []
            for json_item in json_list:
                val = json_item.get(schema_var.name)
                if val is None:
                    continue
                elif _is_list_type(schema_var):
                    bucket.extend(val) 
                else:
                    bucket.append(val)
            merged_simple[schema_var.name] = bucket if _is_list_type(schema_var) else _majority_vote(bucket)
        return merged_simple

    # ------------ NESTED ----------------------------------------------------
    if schema_type == "nestedschema":
        nested_container_name = schema.container_name
        merged_nested: List[Dict[str, Any]] = []
        for json_item in json_list:
            items = json_item.get(nested_container_name, [])
            if items:
                merged_nested.extend(items)
        return {nested_container_name: merged_nested}

    # ------------ MULTIPLE --------------------------------------------------
    if schema_type == "multipleschema":
        merged_multiple: Dict[str, Any] = {}
        for sub_schema_spec_name, sub_schema in schema.schemas.items():
            sub_schema_type = getattr(sub_schema, "schema_type", type(sub_schema).__name__).lower()
            nested_container_name = getattr(sub_schema, "container_name", None)
            sub_jsons = []
            for json_item in json_list:
                if sub_schema_type == "simpleschema":
                    sub_jsons.append(json_item[sub_schema_spec_name])
                elif sub_schema_type == "nestedschema":
                    # rename the key to the nested container name
                    nested_json_item = {}
                    if sub_schema_spec_name in json_item:
                        nested_json_item[nested_container_name] = json_item[sub_schema_spec_name]
                    sub_jsons.append(nested_json_item)
            merged_jsons = merge_jsons_for_record(sub_jsons, sub_schema)
            if sub_schema_type == "simpleschema":
                merged_multiple[sub_schema_spec_name] = merged_jsons
            elif sub_schema_type == "nestedschema": 
                # We need this logic so that output doesn't double nest the schema spec name and the container name for nested schemas inside of a multischema
                merged_multiple[sub_schema_spec_name] = merged_jsons.get(nested_container_name, []) # type: ignore
        return merged_multiple

    raise ValueError(f"Unknown schema type: {schema_type}")


# =====================
# Comprehensive Tests
# =====================
# if __name__ == "__main__":
#     from pprint import pprint
#     print("\n==== SIMPLE SCHEMA ====")
#     # Simple key-value
#     y_true = {"company_names": ["Apple", "Microsoft"], "revenue_numbers": [1500, 2000]}
#     y_pred = {"company_names": ["Apple", "Google"], "revenue_numbers": [2000, 1800]}
#     # Expected:
#     # company_names: TP=1 (Apple), FP=1 (Google), FN=1 (Microsoft) => precision=0.5, recall=0.5
#     # revenue_numbers: TP=2000, FP=1800, FN=1500 => precision=0.5, recall=0.5
#     pprint(all_levels_precision_recall(y_true, y_pred))

#     print("\n==== NESTED SCHEMA ====")
#     # Nested schema (list of dicts)
#     y_true = {"companies": [
#         {"name": "Apple", "revenue": 1500},
#         {"name": "Microsoft", "revenue": 2000}
#     ]}
#     y_pred = {"companies": [
#         {"name": "Microsoft", "revenue": 2000},
#         {"name": "Google", "revenue": 1800},
#         {"name": "Apple", "revenue": 1490}
#     ]}
#     # companies: TP=1 (Microsoft), FP=2 (Google, Apple-1490), FN=1 (Apple-1500)
#     # companies.name: TP=2 (Apple, Microsoft), FP=1 (Google), FN=0
#     # companies.revenue: TP=1 (2000), FP=2 (1800, 1490), FN=1 (1500)
#     pprint(all_levels_precision_recall(y_true, y_pred))

#     print("\n==== MULTIPLE SCHEMAS ====")
#     y_true = {
#         "companies": [
#             {"name": "Apple", "revenue": 1500},
#         ],
#         "products": [
#             {"name": "iPhone", "price": 999}
#         ]
#     }
#     y_pred = {
#         "companies": [
#             {"name": "Apple", "revenue": 1500},
#             {"name": "Microsoft", "revenue": 2000}
#         ],
#         "products": [
#             {"name": "iPhone", "price": 999},
#             {"name": "Galaxy", "price": 899}
#         ]
#     }
#     # companies: TP=1, FP=1, FN=0
#     # companies.name: TP=1, FP=1, FN=0
#     # companies.revenue: TP=1, FP=1, FN=0
#     # products: TP=1, FP=1, FN=0
#     # products.name: TP=1, FP=1, FN=0
#     # products.price: TP=1, FP=1, FN=0
#     pprint(all_levels_precision_recall(y_true, y_pred))

#     print("\n==== EMPTY/NULL CASES ====")
#     y_true = {"companies": []}
#     y_pred = {"companies": []}
#     # companies: TP=0, FP=0, FN=0, precision=0, recall=0
#     pprint(all_levels_precision_recall(y_true, y_pred))

#     print("\n==== DEEPLY NESTED ====")
#     y_true = {
#         "market": {
#             "companies": [
#                 {"name": "Apple", "metrics": {"revenue": 1500, "growth": 10}},
#                 {"name": "Microsoft", "metrics": {"revenue": 2000, "growth": 12}}
#             ]
#         }
#     }
#     y_pred = {
#         "market": {
#             "companies": [
#                 {"name": "Apple", "metrics": {"revenue": 1500, "growth": 9}},
#                 {"name": "Microsoft", "metrics": {"revenue": 2100, "growth": 12}},
#                 {"name": "Google", "metrics": {"revenue": 1800, "growth": 8}}
#             ]
#         }
#     }
#     # market.companies: TP=0, FP=3, FN=2 (no exact dict matches)
#     # market.companies.name: TP=2 (Apple, Microsoft), FP=1 (Google), FN=0
#     # market.companies.metrics: TP=0, FP=3, FN=2 (no exact dict matches)
#     # market.companies.metrics.revenue: TP=1 (1500), FP=2 (2100, 1800), FN=1 (2000)
#     # market.companies.metrics.growth: TP=1 (12), FP=2 (9, 8), FN=1 (10)
#     pprint(all_levels_precision_recall(y_true, y_pred))

#     print("\n==== EDGE CASE: MISSING FIELDS ====")
#     y_true = {"companies": [{"name": "Apple"}]}
#     y_pred = {"companies": [{"name": "Apple", "revenue": 1500}]}
#     # companies: TP=0, FP=1, FN=1 (dicts differ)
#     # companies.name: TP=1, FP=0, FN=0
#     # companies.revenue: TP=0, FP=1, FN=0
#     pprint(all_levels_precision_recall(y_true, y_pred))

#     print("\n==== EDGE CASE: TOP-LEVEL LIST ====")
#     y_true = [
#         {"name": "Apple", "revenue": 1500},
#         {"name": "Microsoft", "revenue": 2000}
#     ]
#     y_pred = [
#         {"name": "Apple", "revenue": 1500},
#         {"name": "Google", "revenue": 1800}
#     ]
#     # root: TP=1, FP=1, FN=1
#     # name: TP=1, FP=1, FN=0
#     # revenue: TP=1, FP=1, FN=0
#     pprint(all_levels_precision_recall(y_true, y_pred))

#     print("\n==== AGGREGATE ACROSS RECORDS ====")
#     expected_list = [
#         {"companies": [
#             {"name": "Apple", "revenue": 1500},
#             {"name": "Microsoft", "revenue": 2000}
#         ]},
#         {"companies": [
#             {"name": "Google", "revenue": 1800}
#         ]}
#     ]
#     predicted_list = [
#         {"companies": [
#             {"name": "Microsoft", "revenue": 2000},
#             {"name": "Apple", "revenue": 1490}
#         ]},
#         {"companies": [
#             {"name": "Google", "revenue": 1800},
#             {"name": "Amazon", "revenue": 1700}
#         ]}
#     ]
#     # companies: sum over both records
#     # companies.name: TP=3 (Apple, Microsoft, Google), FP=2 (Amazon, Apple-1490), FN=0
#     # companies.revenue: TP=2 (2000, 1800), FP=2 (1490, 1700), FN=1 (1500)
#     pprint(aggregate_precision_recall_across_records(expected_list, predicted_list))

#     print("\n==== MERGE: SIMPLE SCHEMA, 1 JSON ====")
#     class DummyVar:
#         def __init__(self, name):
#             self.name = name

#     class DummySimpleSchema:
#         def __init__(self, variables):
#             self.variables = [DummyVar(v) for v in variables]
#         @property
#         def schema_type(self):
#             return "SimpleSchema"

#     simple_schema = DummySimpleSchema(["horizon", "price"])
#     input1 = [{"horizon": ["next several weeks"], "price": [100]}]
#     pprint(merge_jsons_for_record(input1, simple_schema))

#     print("\n==== MERGE: SIMPLE SCHEMA, 2 JSONs ====")
#     input2 = [
#         {"horizon": ["next several weeks"], "price": [100]},
#         {"horizon": ["next quarter"], "price": [110]}
#     ]
#     pprint(merge_jsons_for_record(input2, simple_schema))

#     print("\n==== MERGE: SIMPLE SCHEMA, EMPTY ====")
#     input3 = []
#     pprint(merge_jsons_for_record(input3, simple_schema))

#     print("\n==== MERGE: NESTED SCHEMA, 2 JSONs ====")
#     class DummyNestedSchema:
#         def __init__(self, container_name, variables):
#             self.container_name = container_name
#             self.variables = [DummyVar(v) for v in variables]
#         @property
#         def schema_type(self):
#             return "NestedSchema"

#     nested_schema = DummyNestedSchema("companies", ["name", "revenue"])
#     input4 = [
#         {"companies": [{"name": "Microsoft", "revenue": 2000}]},
#         {"companies": [{"name": "Apple", "revenue": 3000}]}
#     ]
#     pprint(merge_jsons_for_record(input4, nested_schema))

#     print("\n==== MERGE: NESTED SCHEMA, EMPTY ====")
#     input5 = []
#     pprint(merge_jsons_for_record(input5, nested_schema))

#     print("\n==== MERGE: MULTIPLE SCHEMA, 2 JSONs ====")
#     class DummyMultipleSchema:
#         def __init__(self, schemas):
#             self.schemas = schemas
#         @property
#         def schema_type(self):
#             return "MultipleSchema"

#     companies_schema = DummyNestedSchema("companies", ["name", "revenue"])
#     products_schema = DummyNestedSchema("products", ["name", "price"])
#     multiple_schema = DummyMultipleSchema({
#         "companies": companies_schema,
#         "products": products_schema
#     })
#     input6 = [
#         {
#             "companies": {"companies": [{"name": "Microsoft", "revenue": 2000}]},
#             "products": {"products": [{"name": "Widget", "price": 10}]}
#         },
#         {
#             "companies": {"companies": [{"name": "Apple", "revenue": 3000}]},
#             "products": {"products": [{"name": "Gadget", "price": 20}]}
#         }
#     ]
#     pprint(merge_jsons_for_record(input6, multiple_schema))

#     print("\n==== MERGE: MULTIPLE SCHEMA, EMPTY ====")
#     input7 = []
#     pprint(merge_jsons_for_record(input7, multiple_schema))

#     print("\n==== MERGE: SIMPLE SCHEMA, MISSING FIELDS ====")
#     input8 = [{"horizon": ["next several weeks"]}]  # price missing
#     pprint(merge_jsons_for_record(input8, simple_schema))

#     print("\n==== MERGE: NESTED SCHEMA, MISSING CONTAINER ====")
#     input9 = [{}]
#     pprint(merge_jsons_for_record(input9, nested_schema))
