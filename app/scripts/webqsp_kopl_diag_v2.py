"""Diagnose WebQSP KoPL accuracy: type path + relation matching with gold."""
import json, sys, os
from pipeline.extended_type_kopl.pipeline import ExtendedTypeKoPLPipeline
from database.search import GraphPathFinder

with open("data/webqsp/qa/test.jsonl") as f:
    samples = [json.loads(l) for l in f]

import random
random.seed(42)
random.shuffle(samples)
samples = samples[:100]

finder = GraphPathFinder(kg_type="webqsp")
graph = finder.graph
pipe = ExtendedTypeKoPLPipeline(kg_type="webqsp", schema_distill=True)

out = open("result/webqsp_kopl_diag_v2.txt", "w")

# Counters
no_gold_path = 0
total = 0
type_match = 0          # KoPL type path matches gold type path
rel_in_candidates = 0   # gold relation in Phase 2 candidates
rel_in_selected = 0     # gold relation in Phase 3 selected
ans_correct = 0         # final answer overlaps gold
type_wrong_but_ans_ok = 0
type_ok_rel_not_in_cand = 0  # types match but gold rel not in candidates

error_samples = []

for i, s in enumerate(samples):
    q = s["question"]
    entity = s.get("entity", "")
    golds = set()
    for a in s.get("answer_nodes", []):
        if isinstance(a, dict):
            golds.add(a.get("name", ""))
        else:
            golds.add(str(a))
    golds.discard("")
    if not golds or not entity:
        continue
    total += 1

    # Find gold path in KG: get both types and relations
    gold_info = []  # list of (types_tuple, rels_tuple)
    for gold_ans in list(golds)[:3]:
        try:
            cypher = """
            MATCH path = shortestPath((a)-[*1..2]-(b))
            WHERE toLower(a.name) = toLower($anchor) AND toLower(b.name) = toLower($answer)
            RETURN [r IN relationships(path) | type(r)] AS rels,
                   [n IN nodes(path) | labels(n)[0]] AS types
            LIMIT 1
            """
            res = graph.run(cypher, anchor=entity, answer=gold_ans).data()
            if res:
                gold_info.append((
                    tuple(res[0]["types"]),
                    tuple(res[0]["rels"])
                ))
        except:
            pass

    if not gold_info:
        no_gold_path += 1
        continue

    gold_type_paths = set(t for t, r in gold_info)
    gold_rels = set(r for t, r in gold_info)

    # Run ETK
    try:
        result = pipe.run(q, entity_name=entity)
    except Exception as e:
        print(f"  SKIP {i}: {e}", flush=True)
        continue

    # Extract KoPL predicted type path
    etk_types = []
    if result.kopl_program and result.kopl_program.relations:
        etk_types.append(result.kopl_program.relations[0].src_type)
        for rel in result.kopl_program.relations:
            etk_types.append(rel.tgt_type)
    etk_type_tuple = tuple(t.lower() if t else "" for t in etk_types)

    # Normalize gold types for comparison (lowercase)
    gold_types_lower = set(tuple(t.lower() for t in tp) for tp in gold_type_paths)

    # Check type path match
    types_ok = etk_type_tuple in gold_types_lower
    if types_ok:
        type_match += 1

    # Check relation match
    candidate_rels = set(tuple(p.relations) for p in result.candidate_paths)
    selected_rels = set(tuple(p.relations) for p in result.selected_paths)
    pred = set(result.answer_entities) if isinstance(result.answer_entities, (set, list)) else set()

    rel_cand_ok = bool(gold_rels & candidate_rels)
    rel_sel_ok = bool(gold_rels & selected_rels)
    ans_ok = bool(pred & golds)

    if rel_cand_ok: rel_in_candidates += 1
    if rel_sel_ok: rel_in_selected += 1
    if ans_ok: ans_correct += 1

    if not types_ok and ans_ok:
        type_wrong_but_ans_ok += 1
    if types_ok and not rel_cand_ok:
        type_ok_rel_not_in_cand += 1

    # Log mismatches for analysis
    if not types_ok or not rel_cand_ok:
        gold_types_str = " | ".join("->".join(tp) for tp in gold_type_paths)
        gold_rels_str = " | ".join("->".join(r) for r in gold_rels)
        etk_types_str = "->".join(etk_types) if etk_types else "EMPTY"
        error_samples.append(
            f"Q: {q}\n"
            f"  Gold types: {gold_types_str}\n"
            f"  ETK types:  {etk_types_str}\n"
            f"  Gold rels:  {gold_rels_str}\n"
            f"  Candidates: {len(result.candidate_paths)} paths\n"
            f"  Answer OK:  {ans_ok}\n"
        )

    if (i+1) % 25 == 0:
        valid = total - no_gold_path
        msg = (f"{i+1}/100: type_match={type_match}/{valid} "
               f"rel_cand={rel_in_candidates}/{valid} "
               f"rel_sel={rel_in_selected}/{valid} "
               f"ans={ans_correct}/{valid}")
        out.write(msg + "\n")
        out.flush()
        print(msg, flush=True)

valid = total - no_gold_path
out.write(f"\n=== FINAL (n={total}, no_gold_path={no_gold_path}, valid={valid}) ===\n")
out.write(f"KoPL type path correct:             {type_match}/{valid} = {100*type_match//max(valid,1)}%\n")
out.write(f"Gold rel in candidates (Phase 1-2):  {rel_in_candidates}/{valid} = {100*rel_in_candidates//max(valid,1)}%\n")
out.write(f"Gold rel in selected (Phase 3):      {rel_in_selected}/{valid} = {100*rel_in_selected//max(valid,1)}%\n")
out.write(f"Final answer correct:                {ans_correct}/{valid} = {100*ans_correct//max(valid,1)}%\n")
out.write(f"\n--- Error breakdown ---\n")
out.write(f"Type wrong but answer OK:            {type_wrong_but_ans_ok}\n")
out.write(f"Type OK but gold rel not in cand:    {type_ok_rel_not_in_cand}\n")
out.write(f"Phase 3 loss (cand->selected drop):  {rel_in_candidates - rel_in_selected}\n")

# Write first 20 error samples
out.write(f"\n--- Sample errors (first 20) ---\n")
for e in error_samples[:20]:
    out.write(e + "\n")

out.flush()
out.close()
print("DONE", flush=True)
