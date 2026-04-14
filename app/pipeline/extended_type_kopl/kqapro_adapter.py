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


# Phase 1 prompt section appended to _build_kopl_prompt for kqapro.
ANSWER_TYPE_INFO = """
ANSWER TYPE FUNCTIONS — set answer_type and required fields.

  answer_type        | Required fields                                              | Output
  -------------------|--------------------------------------------------------------|------------------
  "entity"           | operations                                                   | entity names
  "count"            | operations                                                   | integer
  "attr"             | operations, query_key                                        | attribute value
  "relation"         | select_entity_a, select_entity_b                             | relation name
  "verify"           | operations, query_key, verify_value, verify_op               | "yes" / "no"
  "select"           | query_key, select_mode, [select_entity_a, select_entity_b]   | entity name
  "attr_qualifier"   | operations, match_attr_key, match_attr_value, qualifier_key  | qualifier value
  "relation_qualifier"| select_entity_a, select_entity_b, query_key, qualifier_key  | qualifier value

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


def check_kopl_consistency(pipeline, kopl: KoPLOperation, question: str) -> str:
    """Phase 1.5: LLM-based consistency check on generated KoPL.

    Returns empty string if OK, or a description of issues found.
    """
    # 1. Structural checks (required fields)
    at = kopl.answer_type
    if at == "relation_qualifier":
        missing = []
        if not kopl.select_entity_a or not kopl.select_entity_b:
            missing.append("select_entity_a/b")
        if not kopl.query_key:
            missing.append("query_key")
        if not kopl.qualifier_key:
            missing.append("qualifier_key")
        if missing:
            return f"relation_qualifier missing required fields: {', '.join(missing)}"

    if at == "attr_qualifier":
        missing = []
        if not kopl.match_attr_key:
            missing.append("match_attr_key")
        if not kopl.match_attr_value:
            missing.append("match_attr_value")
        if not kopl.qualifier_key:
            missing.append("qualifier_key")
        if missing:
            return f"attr_qualifier missing required fields: {', '.join(missing)}"

    # 2. LLM verification of answer_type
    serialized = pipeline._serialize_kopl_for_prompt(kopl)
    ext_fields = f"answer_type: \"{at}\""
    if kopl.query_key:
        ext_fields += f"\nquery_key: \"{kopl.query_key}\""
    if kopl.qualifier_key:
        ext_fields += f"\nqualifier_key: \"{kopl.qualifier_key}\""
    if kopl.select_entity_a:
        ext_fields += f"\nselect_entity_a: \"{kopl.select_entity_a}\""
    if kopl.select_entity_b:
        ext_fields += f"\nselect_entity_b: \"{kopl.select_entity_b}\""

    prompt = f"""Check if this KoPL program's answer_type is correct for the question.

Question: {question}

Program:
{serialized}
{ext_fields}

Answer types:
- entity: return entity names
- count: return a number
- attr: return an attribute value of the entity itself
- relation: return the relation name between two entities
- verify: return yes/no
- select: compare entities on an attribute
- attr_qualifier: return metadata/qualifier of a specific attribute entry (e.g. "When did X have population Y?" → point in time qualifier of the population attribute)
- relation_qualifier: return metadata/qualifier of a relation between two entities (e.g. "When was X nominated for Y?" → point in time qualifier of the 'nominated for' relation)

Is the answer_type correct? Reply ONLY with either:
- "OK" if correct
- "WRONG: <correct_type>. <brief reason>"
"""
    try:
        response = pipeline.llm.invoke(prompt)
        text = response.content.strip()
        if text.upper().startswith("OK"):
            return ""
        if text.upper().startswith("WRONG"):
            return text
        return ""
    except Exception as e:
        logger.warning("Phase 1.5 LLM check failed: %s", e)
        return ""



def regenerate_kopl_with_feedback(
    pipeline,
    question: str,
    entity_name: Optional[str],
    entity_type: Optional[str],
    target_type: Optional[str],
    failed_kopl: KoPLOperation,
    issues: str,
) -> Optional[KoPLOperation]:
    """Phase 1.5: Regenerate KoPL with consistency feedback.

    Builds the same base prompt but appends diagnostic feedback,
    then parses the result using the same logic as _generate_type_kopl.
    """
    # Lazy-import runtime types from pipeline.py to avoid a circular module load.
    from pipeline.extended_type_kopl.pipeline import (
        AtomicKoPLProgramSchema,
        KoPLOperation,
        OperationType,
        TypeRelation,
    )

    serialized = pipeline._serialize_kopl_for_prompt(failed_kopl)
    extended_fields = []
    if failed_kopl.answer_type:
        extended_fields.append(f"answer_type: \"{failed_kopl.answer_type}\"")
    if failed_kopl.query_key:
        extended_fields.append(f"query_key: \"{failed_kopl.query_key}\"")
    if failed_kopl.qualifier_key:
        extended_fields.append(f"qualifier_key: \"{failed_kopl.qualifier_key}\"")
    if failed_kopl.select_entity_a:
        extended_fields.append(f"select_entity_a: \"{failed_kopl.select_entity_a}\"")
    if failed_kopl.select_entity_b:
        extended_fields.append(f"select_entity_b: \"{failed_kopl.select_entity_b}\"")
    ext_str = "\n".join(extended_fields)

    base_prompt = pipeline._build_kopl_prompt(question, entity_name, entity_type, target_type)
    feedback_prompt = f"""{base_prompt}

CORRECTION — your previous attempt had issues:

Previous program:
{serialized}
{ext_str}

Issues: {issues}

Fix the answer_type and fill ALL required fields. Do NOT repeat the same mistake."""

    try:
        llm_with_output = pipeline.llm.with_structured_output(AtomicKoPLProgramSchema)
        result = llm_with_output.invoke(feedback_prompt)
        if not result or not result.operations:
            return None

        # Reuse the same parsing logic
        valid_types = pipeline.schema.types
        type_normalizer = {t.lower(): t for t in valid_types}

        def normalize_type(t):
            if not t:
                return None
            t_clean = t.strip().rstrip("}],")
            if t_clean in valid_types:
                return t_clean
            t_lower = t_clean.lower()
            if t_lower in type_normalizer:
                return type_normalizer[t_lower]
            return None

        _answer_type = getattr(result, 'answer_type', 'entity') or 'entity'
        _query_key = getattr(result, 'query_key', None)
        _verify_value = getattr(result, 'verify_value', None)
        _verify_op = getattr(result, 'verify_op', None)
        _select_mode = getattr(result, 'select_mode', None)
        _select_entity_a = getattr(result, 'select_entity_a', None)
        _select_entity_b = getattr(result, 'select_entity_b', None)
        _qualifier_key = getattr(result, 'qualifier_key', None)
        _match_attr_key = getattr(result, 'match_attr_key', None)
        _match_attr_value = getattr(result, 'match_attr_value', None)

        ext_kwargs = dict(
            answer_type=_answer_type,
            query_key=_query_key,
            verify_value=_verify_value,
            verify_op=_verify_op,
            select_mode=_select_mode,
            select_entity_a=_select_entity_a,
            select_entity_b=_select_entity_b,
            qualifier_key=_qualifier_key,
            match_attr_key=_match_attr_key,
            match_attr_value=_match_attr_value,
        )

        # Build relations from first operation
        relations = []
        anchor = None
        for op in result.operations:
            src = normalize_type(op.src_type)
            tgt = normalize_type(op.tgt_type)
            if not src and not tgt:
                continue
            relations.append(TypeRelation(
                src_type=src or "Concept",
                tgt_type=tgt or "Concept",
                relation_hint=op.relation,
            ))
            if op.anchor_name and not anchor:
                anchor = op.anchor_name

        return KoPLOperation(
            op_type=OperationType.RELATE,
            relations=relations,
            anchor_name=anchor or entity_name,
            **ext_kwargs,
        )
    except Exception as e:
        logger.warning("Phase 1.5 regeneration failed: %s", e)
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



def resolve_extended_answer(
    pipeline,
    kopl: KoPLOperation,
    answer_entities: List[str],
    entity_name: Optional[str],
    log: List[str],
) -> str:
    """Phase 6: KQA-Pro extended answer type resolution.

    Uses KBPropertyStore (kb.json) for attribute lookups when available,
    falls back to Neo4j for relation queries.
    """
    graph = pipeline.finder.graph
    atype = kopl.answer_type
    store = pipeline.kb_store  # may be None for non-KQA-Pro KGs

    log.append(f"Phase 6: Extended answer ({atype})")

    if atype == "count":
        ans = str(len(answer_entities))
        log.append(f"  Count: {ans}")
        return ans

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

    if atype == "relation":
        ent_a = kopl.select_entity_a or entity_name
        ent_b = kopl.select_entity_b
        if not ent_a or not ent_b:
            log.append("  Need two entities for relation query")
            return ""
        try:
            cypher = (
                "MATCH (a)-[r]-(b) "
                "WHERE toLower(a.name) = toLower($a) AND toLower(b.name) = toLower($b) "
                "RETURN type(r) AS rel LIMIT 5"
            )
            records = graph.run(cypher, a=ent_a, b=ent_b).data()
            if records:
                rels = [r["rel"] for r in records]
                ans = rels[0].replace("_", " ")
                log.append(f"  QueryRelation('{ent_a}', '{ent_b}'): {rels}")
                return ans
            log.append(f"  No relation found between '{ent_a}' and '{ent_b}'")
            return ""
        except Exception as e:
            log.append(f"  QueryRelation error: {e}")
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

