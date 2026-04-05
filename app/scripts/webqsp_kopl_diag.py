"""Diagnose WebQSP KoPL accuracy by comparing with gold paths from KG."""
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

out = open("result/webqsp_kopl_accuracy.txt", "w")

kopl_correct = path_correct = answer_correct = no_gold_path = total = 0

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

    # Find gold path in KG
    gold_rels = set()
    for gold_ans in list(golds)[:3]:
        try:
            cypher = """
            MATCH path = shortestPath((a)-[*1..2]-(b))
            WHERE toLower(a.name) = toLower($anchor) AND toLower(b.name) = toLower($answer)
            RETURN [r IN relationships(path) | type(r)] AS rels
            LIMIT 1
            """
            res = graph.run(cypher, anchor=entity, answer=gold_ans).data()
            if res:
                gold_rels.add(tuple(res[0]["rels"]))
        except:
            pass

    if not gold_rels:
        no_gold_path += 1
        continue

    # Run ETK
    result = pipe.run(q, entity_name=entity)

    candidate_rels = set(tuple(p.relations) for p in result.candidate_paths)
    selected_rels = set(tuple(p.relations) for p in result.selected_paths)
    pred = set(result.answer_entities) if isinstance(result.answer_entities, (set, list)) else set()

    kopl_ok = bool(gold_rels & candidate_rels)
    path_ok = bool(gold_rels & selected_rels)
    ans_ok = bool(pred & golds)

    if kopl_ok: kopl_correct += 1
    if path_ok: path_correct += 1
    if ans_ok: answer_correct += 1

    if (i+1) % 25 == 0:
        out.write(f"{i+1}/100: kopl={kopl_correct} path={path_correct} ans={answer_correct} no_gold={no_gold_path}\n")
        out.flush()
        print(f"{i+1}/100: kopl={kopl_correct} path={path_correct} ans={answer_correct} no_gold={no_gold_path}", flush=True)

valid = total - no_gold_path
out.write(f"FINAL (n={total}, no_gold_path={no_gold_path}, valid={valid}):\n")
out.write(f"Phase 1-2 (gold rel in candidates): {kopl_correct}/{valid} = {100*kopl_correct//max(valid,1)}%\n")
out.write(f"Phase 3 (gold rel in selected):     {path_correct}/{valid} = {100*path_correct//max(valid,1)}%\n")
out.write(f"Final answer correct:               {answer_correct}/{valid} = {100*answer_correct//max(valid,1)}%\n")
out.write(f"Phase 3 loss: {kopl_correct - path_correct} (in candidates but dropped)\n")
out.write(f"Phase 4 loss: {path_correct - answer_correct} (right path but wrong answer)\n")
out.flush()
out.close()
print("DONE", flush=True)
