"""KQA-Pro-specific pipeline extensions.

Isolated from the core ETK pipeline so that MetaQA, PrimeKGQA, PcQA and
WebQSP paths are not polluted with kb.json / extended-answer-type code.

Every function here takes a reference to the ExtendedTypeKoPLPipeline
instance as its first argument and is invoked via light delegation from
pipeline.py when ``kg_type == "kqapro"``.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pipeline.extended_type_kopl.pipeline import (
        ExtendedTypeKoPLPipeline,
        KoPLOperation,
    )


# Phase 1 prompt fragments per answer_type. build_answer_type_info(...) in
# pipeline.py stitches together the rows that the KG actually supports, so
# the prompt only advertises types we can actually resolve.

# Table row per answer_type (column alignment matches the original layout).
ANSWER_TYPE_ROWS = {
    "entity":             '  "entity"           | operations                                                   | entity names',
    "count":              '  "count"            | operations                                                   | integer',
    "attr":               '  "attr"             | operations, query_key                                        | attribute value',
    "relation":           '  "relation"         | select_entity_a, select_entity_b                             | relation name',
    "verify":             '  "verify"           | operations, query_key, verify_value, verify_op               | "yes" / "no"',
    "select":             '  "select"           | query_key, select_mode, [select_entity_a, select_entity_b]   | entity name',
    "attr_qualifier":     '  "attr_qualifier"   | operations, match_attr_key, match_attr_value, qualifier_key  | qualifier value',
    "relation_qualifier": '  "relation_qualifier"| select_entity_a, select_entity_b, query_key, qualifier_key  | qualifier value',
}

# Qualifier-related guidance; only injected when the KG supports the
# KB-backed types (KQA-Pro's Wikidata qualifiers).
_KB_GUIDANCE = """
THINK STEP BY STEP before choosing answer_type:
1. Identify the entities mentioned in the question
2. Determine what the question is asking for (entity name? count? attribute value? metadata?)
3. Check: does the question mention a KNOWN value and ask WHEN/WHERE/WHO about it? → qualifier
4. Check: does the question ask about metadata of a RELATION between two entities? → relation_qualifier
5. Then set answer_type and fill in the required fields

Key distinctions:
- "attr" asks for a property VALUE → "What is the population of X?"
- "attr_qualifier" asks for METADATA of a property fact → "When did X have population 2060?"
- "relation" asks for the PREDICATE name → "What is the relation between X and Y?"
- "relation_qualifier" asks for METADATA of a relation → "When was X nominated for Y?"
- "select" with two named entities: set select_entity_a/b. With a concept set: use operations + select_mode only."""

# Short note used when only the agnostic types (entity/count/relation) are
# available. Helps the LLM still pick count/relation instead of defaulting
# to entity for "how many" or "what is the relation between".
_AGNOSTIC_GUIDANCE = """
Pick the narrowest answer_type the question asks for:
- "How many X?" → "count"
- "What is the relation between X and Y?" → "relation" (set select_entity_a/b)
- Otherwise → "entity\""""


def build_answer_type_info(supported: set) -> str:
    """Build the Phase 1 prompt fragment listing answer types this KG supports.

    Returns an empty string when the KG only supports the "entity" default
    (so the prompt stays completely agnostic for simple KGs).
    """
    ordered = ["entity", "count", "attr", "relation", "verify", "select", "attr_qualifier", "relation_qualifier"]
    rows = [ANSWER_TYPE_ROWS[t] for t in ordered if t in supported]
    if len(rows) <= 1:
        return ""  # entity-only KG: no need for a section
    header = (
        "\nANSWER TYPE FUNCTIONS — set answer_type and required fields.\n\n"
        "  answer_type        | Required fields                                              | Output\n"
        "  -------------------|--------------------------------------------------------------|------------------\n"
    )
    guidance = _KB_GUIDANCE if supported & {"attr", "verify", "select", "attr_qualifier", "relation_qualifier"} else _AGNOSTIC_GUIDANCE
    return header + "\n".join(rows) + "\n" + guidance

def init_kb_store(kg_type: str):
    """Instantiate KBPropertyStore for KQA-Pro, or return None for other KGs."""
    if kg_type != "kqapro":
        return None
    from pipeline.extended_type_kopl.kb_property_store import KBPropertyStore
    kb_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "kqapro", "kb.json"
    )
    if os.path.exists(kb_path):
        return KBPropertyStore(kb_path)
    return None


def resolve_query_key_dynamic(
    pipeline,
    kopl: KoPLOperation,
    answer_entities: List[str],
    entity_name: Optional[str],
    question: str,
    log: List[str],
) -> Optional[str]:
    """Phase 5.8: Resolve query_key by showing LLM the entity's actual properties.

    For attr/verify/select questions, the LLM's initial query_key may not match
    the exact KB key. This method retrieves the entity's real properties and
    asks the LLM to pick the correct one.

    For qualifier types, also resolves qualifier_key and match_attr_key.
    """
    if not pipeline.kb_store:
        return None

    # --- Qualifier types: resolve qualifier-specific keys ---
    if kopl.answer_type == "attr_qualifier":
        if kopl.match_attr_key:
            resolved = pipeline.kb_store.resolve_key(kopl.match_attr_key)
            if resolved:
                kopl.match_attr_key = resolved
                log.append(f"Phase 5.8: match_attr_key resolved: '{kopl.match_attr_key}' -> '{resolved}'")
        if kopl.qualifier_key:
            resolved = pipeline.kb_store.resolve_key(kopl.qualifier_key)
            if resolved:
                kopl.qualifier_key = resolved
                log.append(f"Phase 5.8: qualifier_key resolved: '{kopl.qualifier_key}' -> '{resolved}'")
        return kopl.query_key  # no change to query_key for attr_qualifier

    if kopl.answer_type == "relation_qualifier":
        if kopl.query_key:
            resolved = pipeline.kb_store.resolve_key(kopl.query_key)
            if resolved:
                log.append(f"Phase 5.8: relation pred resolved: '{kopl.query_key}' -> '{resolved}'")
                kopl.query_key = resolved
        if kopl.qualifier_key:
            resolved = pipeline.kb_store.resolve_key(kopl.qualifier_key)
            if resolved:
                kopl.qualifier_key = resolved
                log.append(f"Phase 5.8: qualifier_key resolved: '{kopl.qualifier_key}' -> '{resolved}'")
        return kopl.query_key

    # --- Standard types: attr/verify/select ---

    # Determine which entity to inspect
    if kopl.answer_type == "select" and kopl.select_entity_a:
        target = kopl.select_entity_a
    else:
        target = answer_entities[0] if answer_entities else entity_name

    if not target:
        return None

    # First try fuzzy match without LLM call
    if kopl.query_key:
        resolved = pipeline.kb_store.resolve_key(kopl.query_key)
        if resolved:
            log.append(f"Phase 5.8: query_key resolved: '{kopl.query_key}' -> '{resolved}'")
            return resolved

    # Get available properties
    props_text = pipeline.kb_store.format_properties_for_prompt(target)
    if not props_text:
        log.append(f"Phase 5.8: no properties found for '{target}'")
        return None

    # Ask LLM to pick the right property
    prompt = (
        f"Question: {question}\n"
        f"Entity: {target}\n"
        f"\n{props_text}\n\n"
        f"Which property key from the list above answers this question? "
        f"Return ONLY the exact property key string, nothing else."
    )
    try:
        response = pipeline.llm.invoke(prompt)
        picked = response.content.strip().strip('"').strip("'")
        # Validate against actual keys
        resolved = pipeline.kb_store.resolve_key(picked)
        if resolved:
            log.append(f"Phase 5.8: LLM picked query_key: '{picked}' -> '{resolved}'")
            return resolved
        # Try the raw pick
        log.append(f"Phase 5.8: LLM picked '{picked}' but not found in KB keys")
        return picked
    except Exception as e:
        log.append(f"Phase 5.8: LLM query_key resolution failed: {e}")
        return None



def resolve_kb_answer(
    pipeline,
    kopl: KoPLOperation,
    answer_entities: List[str],
    entity_name: Optional[str],
    log: List[str],
) -> str:
    """Phase 6 resolver for the KB-backed answer_types.

    Handles attr / verify / select / attr_qualifier / relation_qualifier.
    Requires ``pipeline.kb_store`` to be set (currently only KQA-Pro).
    The KG-agnostic count / relation resolvers live in pipeline.py and
    run before this function, so we can assume ``atype`` is one of the
    KB-backed types here.
    """
    atype = kopl.answer_type
    store = pipeline.kb_store

    log.append(f"Phase 6: Extended answer ({atype})")

    if atype == "attr" and kopl.query_key:
        target = answer_entities[0] if answer_entities else entity_name
        if not target:
            log.append("  No entity to query attribute from")
            return ""
        if store:
            val = store.query_attr(target, kopl.query_key)
            if val is not None:
                log.append(f"  QueryAttr({kopl.query_key}) on '{target}': {val}")
                return val
            log.append(f"  QueryAttr({kopl.query_key}) on '{target}': no value in KB")
        return ""

    if atype == "verify":
        target = answer_entities[0] if answer_entities else entity_name
        if not target or not kopl.query_key:
            log.append("  Verify: missing entity or query_key")
            return "no"
        if store:
            from pipeline.extended_type_kopl.kb_property_store import comp as kb_comp
            actual_vc = store.query_attr_value(target, kopl.query_key)
            if actual_vc is not None:
                expected_vc = store.parse_value_for_key(kopl.query_key, kopl.verify_value or "")
                op = kopl.verify_op or "="
                try:
                    if actual_vc.can_compare(expected_vc):
                        match = kb_comp(actual_vc, expected_vc, op)
                    else:
                        # Fallback: string comparison
                        match = str(actual_vc).lower() == (kopl.verify_value or "").lower()
                    ans = "yes" if match else "no"
                    log.append(f"  Verify: {kopl.query_key}='{actual_vc}' {op} '{kopl.verify_value}' -> {ans}")
                    return ans
                except Exception as e:
                    log.append(f"  Verify comparison error: {e}")
                    return "no"
            log.append(f"  Verify: property '{kopl.query_key}' not found on '{target}'")
            return "no"
        return "no"

    if atype == "select":
        key = kopl.query_key
        mode = (kopl.select_mode or "greater").lower()
        if not key:
            log.append("  Select: missing query_key")
            return ""

        _GREATER_MODES = {"greater", "larger", "more", "later", "latest", "longest", "higher"}
        _LESS_MODES = {"less", "smaller", "fewer", "earlier", "earliest", "shortest", "smallest", "lower"}

        # SelectBetween: compare exactly two named entities
        if kopl.select_entity_a and kopl.select_entity_b and store:
            a_vc = store.query_attr_value(kopl.select_entity_a, key)
            b_vc = store.query_attr_value(kopl.select_entity_b, key)
            if a_vc is not None and b_vc is not None and a_vc.can_compare(b_vc):
                try:
                    if mode in _GREATER_MODES:
                        ans = kopl.select_entity_a if a_vc > b_vc or a_vc == b_vc else kopl.select_entity_b
                    else:
                        ans = kopl.select_entity_a if a_vc < b_vc or a_vc == b_vc else kopl.select_entity_b
                    log.append(f"  SelectBetween: {kopl.select_entity_a}={a_vc} vs {kopl.select_entity_b}={b_vc} ({mode}) -> {ans}")
                    return ans
                except Exception as e:
                    log.append(f"  SelectBetween comparison error: {e}")
                    return ""
            log.append(f"  SelectBetween: could not get values (a={a_vc}, b={b_vc})")
            return ""

        # SelectAmong: pick best from answer_entities
        if answer_entities and store:
            best_ent = None
            best_vc = None
            for ent in answer_entities:
                vc = store.query_attr_value(ent, key)
                if vc is None:
                    continue
                if best_vc is None:
                    best_ent, best_vc = ent, vc
                elif vc.can_compare(best_vc):
                    try:
                        if (mode in _GREATER_MODES and vc > best_vc) or \
                           (mode in _LESS_MODES and vc < best_vc):
                            best_ent, best_vc = ent, vc
                    except Exception:
                        pass
            if best_ent:
                log.append(f"  SelectAmong: {best_ent} ({key}={best_vc}, mode={mode})")
                return best_ent

        log.append("  Select: no result")
        return ""

    if atype == "attr_qualifier":
        target = answer_entities[0] if answer_entities else entity_name
        if not target or not store:
            log.append("  AttrQualifier: missing entity or KB store")
            return ""
        attr_key = kopl.match_attr_key
        attr_val = kopl.match_attr_value
        qual_key = kopl.qualifier_key
        if not attr_key or not attr_val or not qual_key:
            log.append(f"  AttrQualifier: missing fields (attr_key={attr_key}, attr_val={attr_val}, qual_key={qual_key})")
            return ""
        result = store.query_attr_qualifier(target, attr_key, attr_val, qual_key)
        if result is not None:
            log.append(f"  AttrQualifier({attr_key}={attr_val}, {qual_key}) on '{target}': {result}")
            return result
        log.append(f"  AttrQualifier: no qualifier found on '{target}'")
        return ""

    if atype == "relation_qualifier":
        ent_a = kopl.select_entity_a or entity_name
        ent_b = kopl.select_entity_b
        pred = kopl.query_key
        qual_key = kopl.qualifier_key
        if not ent_a or not ent_b or not pred or not qual_key:
            log.append(f"  RelQualifier: missing fields (a={ent_a}, b={ent_b}, pred={pred}, qual={qual_key})")
            return ""
        if store:
            result = store.query_relation_qualifier(ent_a, ent_b, pred, qual_key)
            if result is not None:
                log.append(f"  RelQualifier({pred}, {qual_key}) on '{ent_a}'-'{ent_b}': {result}")
                return result
        log.append(f"  RelQualifier: no qualifier found")
        return ""

    return ""

