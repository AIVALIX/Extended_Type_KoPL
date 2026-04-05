"""Build eval_v3.jsonl by converting missing entries from all.jsonl."""
import json
import re
import sys
import traceback

from database.search import GraphPathFinder
from pipeline.extended_type_kopl.pipeline import ExtendedTypeKoPLPipeline

finder = GraphPathFinder("pcqa")
p = ExtendedTypeKoPLPipeline(kg_type="pcqa")

# Load data
with open("data/pcqa/qa/eval_v2.jsonl") as f:
    eval_v2 = [json.loads(l) for l in f]
with open("data/pcqa/qa/all.jsonl") as f:
    all_data = [json.loads(l) for l in f]

eval_v2_questions = set(s["question"] for s in eval_v2)
missing = [s for s in all_data if s["question"] not in eval_v2_questions]

print(f"Total missing: {len(missing)}")

new_entries = []
failed = []

for i, s in enumerate(missing):
    q = s["question"]
    q_lower = q.lower()

    # Skip describe questions
    if "describe" in q_lower:
        failed.append({"q": q, "reason": "describe_question"})
        continue

    print(f"\n[{i+1}/{len(missing)}] {q[:80]}")

    # Try entity extraction
    try:
        extracted = p._extract_entity_kgt_style(q)
    except Exception as e:
        print(f"  Entity extraction error: {e}")
        failed.append({"q": q, "reason": f"entity_extraction_error: {e}"})
        continue

    if not extracted:
        failed.append({"q": q, "reason": "entity_extraction_failed"})
        print("  -> Entity extraction failed")
        continue

    entity_name, entity_type, target_type = extracted
    print(f"  Entity: {entity_name} ({entity_type}) -> {target_type}")

    # Run pipeline
    try:
        result = p.run(q, entity_name=entity_name)
    except Exception as e:
        print(f"  Pipeline error: {e}")
        traceback.print_exc()
        failed.append({"q": q, "reason": f"pipeline_error: {e}", "entity": entity_name})
        continue

    if result.answer_entities:
        answers = [{"name": e} for e in result.answer_entities]

        # Extract relation from log
        relation = ""
        for log_line in result.processing_log:
            if "Selected path:" in log_line:
                # Try to extract relation type from path notation
                m = re.search(r"\[(\w+)\]", log_line)
                if m:
                    relation = m.group(1)
                    break
            if "Relation hints:" in log_line:
                m = re.search(r"Relation hints: \['([^']+)'\]", log_line)
                if m:
                    relation = m.group(1)

        # Extract path from log
        path = ""
        for log_line in result.processing_log:
            if "Selected path:" in log_line:
                m = re.search(r"Selected path: (.+)", log_line)
                if m:
                    path = m.group(1).strip()
                    break

        entry = {
            "question": q,
            "entity": entity_name,
            "entity_type": entity_type or "",
            "relation": relation,
            "path": path,
            "filters": {},
            "answers": answers,
            "original_index": len(eval_v2) + len(new_entries) + 1,
        }
        new_entries.append(entry)
        print(f"  -> OK: {len(answers)} answers, rel={relation}")
    else:
        failed.append({"q": q, "reason": "no_answers_found", "entity": entity_name})
        print(f"  -> No answers found")

# Summary
print("\n" + "=" * 60)
print(f"Converted: {len(new_entries)}")
print(f"Failed: {len(failed)}")
describe_count = sum(1 for f in failed if f["reason"] == "describe_question")
print(f"  describe: {describe_count}")
print(f"  entity_not_found: {sum(1 for f in failed if 'entity_not_found' in f['reason'])}")
print(f"  extraction_failed: {sum(1 for f in failed if f['reason'] == 'entity_extraction_failed')}")
print(f"  no_answers: {sum(1 for f in failed if f['reason'] == 'no_answers_found')}")
print(f"  errors: {sum(1 for f in failed if 'error' in f['reason'])}")

# Merge and write
eval_v3 = eval_v2 + new_entries
with open("data/pcqa/qa/eval_v3.jsonl", "w") as f:
    for entry in eval_v3:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

non_describe_total = 405 - describe_count
print(f"\neval_v3.jsonl: {len(eval_v3)} entries (was {len(eval_v2)})")
print(f"Coverage: {len(eval_v3)}/{non_describe_total} non-describe = {len(eval_v3)/non_describe_total*100:.1f}%")

# Save failed for debugging
with open("data/pcqa/qa/eval_v3_failed.json", "w") as f:
    json.dump(failed, f, ensure_ascii=False, indent=2)
print(f"Failed entries saved to data/pcqa/qa/eval_v3_failed.json")
