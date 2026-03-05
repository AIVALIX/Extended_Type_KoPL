# Extended Type-KoPL Intersection Query Improvement Report

**Date**: 2026-01-27
**Author**: Claude Code
**Version**: v2.0

---

## Executive Summary

Extended Type-KoPL pipeline underwent significant improvements to handle intersection queries correctly. The main issue was the LLM misinterpreting intersection queries (multiple anchors) as multi-hop path queries, causing timeouts and incorrect results.

### Key Results

| Dataset | Before | After | Improvement |
|---------|--------|-------|-------------|
| three_intersection | 32% (65 errors) | **100%** (0 errors) | **+68%** |
| two_intersection | 49% | **83%** | **+34%** |
| one_hop | 85% | **87%** | +2% |
| two_hop | 59% | **61%** | +2% |

**Average Accuracy**: 56.2% -> **82.8%** (+26.6%)

---

## Problem Analysis

### Root Cause

The LLM was incorrectly interpreting intersection queries as path queries due to ambiguous prompt rules:

**Example of Misinterpretation:**
- Question: "Which proteins are targeted by drugs A, B, and C?" (3 anchors)
- Expected: INTERSECTION query with 3 independent 1-hop operations
- Actual: 3-hop PATH query (A -> ? -> ? -> ?)

**Why This Happened:**
1. The prompt's CRITICAL RULES emphasized "N entities = N-hop query"
2. "three drugs" was interpreted as "3-hop" instead of "3 anchors"
3. No clear distinction between PATH vs INTERSECTION patterns

### Symptoms

1. **Timeout Errors**: 65/100 samples on three_intersection timed out (300s limit)
2. **Wrong Query Type**: LLM generated PATH instead of INTERSECTION
3. **Empty Results**: Even when queries completed, they returned wrong entity types

---

## Solution Implementation

### 1. Domain-Independent Generic Examples

Replaced domain-specific examples (drug, gene, disease) with abstract type examples:

```
=== PATH QUERY (single anchor, traverse graph) ===
Pattern: A --[rel1]--> B --[rel2]--> C
- 1 anchor entity, multiple hops
- operations are CONNECTED: op[i].tgt_type == op[i+1].src_type

1-hop: "Find Y related to X"
operations: [{src_type: "TypeX", tgt_type: "TypeY", relation: "rel1", anchor_name: "X"}]
final_operation: "relate"

=== INTERSECTION QUERY (multiple anchors, combine results) ===
Pattern: A --[relA]--> ? <--[relB]-- B  (finding common Y from different anchors)
- Multiple anchor entities with their own relations
- Each anchor has INDEPENDENT 1-hop operation with its own anchor_name
- Key indicator: "Y that is [verb] by A AND also [verb] by/with B" = INTERSECTION

"Find Y that is targeted by A and also associated with B"
-> NOT a path A->Y->B, but intersection of two 1-hop queries
operations: [
  {src_type: "TypeA", tgt_type: "TypeY", relation: "relA", anchor_name: "A"},
  {src_type: "TypeB", tgt_type: "TypeY", relation: "relB", anchor_name: "B"}
]
final_operation: "intersection"
```

### 2. Simplified RULES Section

Removed keyword-based detection (too arbitrary) and simplified to structural rules:

```
RULES:
1. Each operation represents ONE HOP in the path
2. Each operation specifies: src_type, tgt_type, relation, anchor_name (if applicable)
3. For PATH queries: operations are connected (op[i].tgt_type == op[i+1].src_type)
4. For INTERSECTION queries: each operation has its own anchor_name, final_operation="intersection"
```

### 3. Path Selection Fix for Different-Relation Intersections

Fixed bug where intersection queries with different src_types only selected one path:

**Before:**
```python
selected_paths = pruned_paths[:1]  # Only 1 path for all intersection children
```

**After:**
```python
elif kopl_program.op_type == OperationType.INTERSECTION and kopl_program.children:
    selected_paths = []
    needed_src_types = set()
    for child in kopl_program.children:
        for rel in child.relations:
            if rel.src_type:
                needed_src_types.add(rel.src_type)

    for src_type in needed_src_types:
        for p in pruned_paths:
            if p.types and p.types[0] == src_type:
                selected_paths.append(p)
                break
```

---

## Evaluation Results

### PrimeKGQA (n=100, random sampling, workers=16)

| Dataset | Extended Type-KoPL | KGT | SAFE |
|---------|-------------------|-----|------|
| one_hop | **87.0%** | 77.0% | 82.0% |
| two_hop | **61.0%** | 0.0% | 29.0% |
| two_intersection | **83.0%** | 73.0% | - |
| three_intersection | **100.0%** | 100.0% | - |
| **Average** | **82.8%** | 62.5% | 55.5% |

### MetaQA (n=100, random sampling, workers=16)

| Dataset | Extended Type-KoPL | KGT | SAFE |
|---------|-------------------|-----|------|
| 1-hop | 94.0% | 91.0% | **100.0%** |
| 2-hop | 58.0% | 54.0% | **89.0%** |
| 3-hop | **26.0%** | 5.0% | - |
| **Average** | 59.3% | 50.0% | **94.5%** |

### Pipeline Comparison Summary

| Metric | Extended Type-KoPL | KGT | SAFE |
|--------|-------------------|-----|------|
| PrimeKGQA Average | **82.8%** | 62.5% | 55.5% |
| MetaQA Average | 59.3% | 50.0% | **94.5%** |
| Intersection Support | Yes | Yes | No |
| 3-hop Support | Yes | Yes | No |

---

## Technical Details

### Files Modified

1. **`app/pipeline/extended_type_kopl/pipeline.py`**
   - `_get_generic_examples()`: New method for domain-independent examples
   - `_get_prompt_examples()`: Simplified to use generic examples
   - `_generate_type_kopl()`: Improved intersection handling
   - `run()`: Fixed path selection for multi-src_type intersections

### Key Code Changes

**Location**: `pipeline.py:641-679` (Generic Examples)
```python
def _get_generic_examples(self) -> str:
    """Domain-independent examples using abstract types"""
    return """Examples:
    === PATH QUERY ===
    ...
    === INTERSECTION QUERY ===
    ...
    """
```

**Location**: `pipeline.py:603-627` (Path Selection Fix)
```python
elif kopl_program.op_type == OperationType.INTERSECTION and kopl_program.children:
    # Select best path for each unique src_type
    selected_paths = []
    needed_src_types = set()
    for child in kopl_program.children:
        for rel in child.relations:
            if rel.src_type:
                needed_src_types.add(rel.src_type)

    for src_type in needed_src_types:
        for p in pruned_paths:
            if p.types and p.types[0] == src_type:
                selected_paths.append(p)
                break
```

---

## Lessons Learned

1. **Generic > Domain-Specific**: Domain-independent examples generalize better across different KGs
2. **Structural Rules > Keyword Rules**: Structural patterns (PATH vs INTERSECTION) are more robust than keyword detection
3. **Clear Distinction**: Explicitly showing "NOT a path" helps LLM avoid misinterpretation
4. **Path Selection Matters**: For intersection queries with heterogeneous anchors, each src_type needs its own path

---

## Future Improvements

1. **MetaQA 3-hop**: Extended Type-KoPL shows 26% on 3-hop (vs 5% KGT), but still needs improvement
2. **PathAcc for Intersection**: Currently 0% - need to define what "correct path" means for intersection
3. **Relation Hint Matching**: Vector similarity between "treated_by" and "indication" is poor; consider synonym mapping

---

## Conclusion

The intersection query improvements successfully raised Extended Type-KoPL's performance:
- **three_intersection**: 32% -> 100% (+68%)
- **two_intersection**: 49% -> 83% (+34%)
- **Zero errors**: Eliminated all 65 timeout errors

Extended Type-KoPL is now the top-performing pipeline on PrimeKGQA with 82.8% average accuracy, handling all query types including intersection queries that SAFE cannot process.
