"""
PrimeKGQA Original (bioLLM版) → ETK評価用JSONL変換

入力: data/primekgqa_original/test_call_bioLLM.json
出力: data/primekgqa_original/qa/test_entity.jsonl (entity検索タスクのみ)

Usage:
    python app/dataset_construction/convert_primekgqa_original.py
"""
import json
import re
from pathlib import Path
from urllib.parse import unquote
from collections import Counter


KNOWN_RELATIONS = {
    "associated with", "expression present", "expression absent",
    "phenotype present", "phenotype absent", "target", "indication",
    "contraindication", "off label use", "side effect", "carrier",
    "enzyme", "transporter", "ppi", "interacts with", "linked to",
    "parent child", "synergistic interaction", "linked exposure",
    "absent gene", "expressed gene",
}


def clean_entity_name(name: str) -> str:
    """Remove brackets and URL prefixes from entity names."""
    name = name.strip("[]")
    # Handle URL format: <https://...PrimeKG/node/XXXX>
    if "PrimeKG/node/" in name:
        name = name.split("PrimeKG/node/")[-1].rstrip(">")
    if "PrimeKG/vocab/" in name:
        name = unquote(name.split("PrimeKG/vocab/")[-1].rstrip(">"))
    name = name.strip("<>")
    return name


def clean_relation(rel: str) -> str:
    """Decode relation name."""
    rel = rel.strip("[]<>")
    if "PrimeKG/vocab/" in rel:
        rel = rel.split("PrimeKG/vocab/")[-1]
    return unquote(rel).strip()


def clean_question(q: str) -> str:
    """Clean up LLM-generated question text."""
    # Remove brackets around entity names
    q = re.sub(r'\[([^\]]+)\]', r'\1', q)
    # Remove URL references
    q = re.sub(r'<https?://[^>]+>', '', q)
    return q.strip()


def convert():
    base = Path(__file__).resolve().parents[2]
    input_path = base / "data" / "primekgqa_original" / "test_call_bioLLM.json"
    output_path = base / "data" / "primekgqa_original" / "qa" / "test_entity.jsonl"

    with open(input_path) as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_no_q = 0
    skipped_relation = 0
    skipped_no_answer = 0
    type_counter = Counter()

    with open(output_path, "w") as out:
        for d in data:
            question = d.get("generated_question", "").strip()
            if not question:
                skipped_no_q += 1
                continue

            answers_raw = d.get("answer_sparql", d.get("answer", []))
            if not answers_raw:
                skipped_no_answer += 1
                continue

            # Decode answers
            answers = []
            for a in answers_raw:
                if a is None:
                    continue
                decoded = unquote(str(a)).rstrip(">").strip()
                if decoded:
                    answers.append(decoded)

            if not answers:
                skipped_no_answer += 1
                continue

            # Skip relation prediction tasks
            is_relation = any(a.lower() in KNOWN_RELATIONS for a in answers)
            if is_relation:
                skipped_relation += 1
                continue

            # Extract anchor and relations from triples
            triples = d["value"]
            anchor_name = clean_entity_name(triples[0][0])
            relations = []
            for triple in triples:
                rel = clean_relation(triple[1])
                if rel and rel.lower() not in relations:
                    relations.append(rel)

            # Clean question
            question = clean_question(question)

            # Build ETK format
            sample = {
                "question": question,
                "anchor_name": anchor_name,
                "relation": relations[0] if relations else "",
                "answer_nodes": [{"name": a} for a in answers],
                "gold_relations": relations,
                "original_type": d["type"],
            }

            # For multi-hop, add rel1/rel2
            if len(relations) >= 2:
                sample["rel1"] = relations[0]
                sample["rel2"] = relations[1]

            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            converted += 1
            type_counter[d["type"]] += 1

    print(f"Converted: {converted}")
    print(f"Skipped (no question): {skipped_no_q}")
    print(f"Skipped (relation answer): {skipped_relation}")
    print(f"Skipped (no answer): {skipped_no_answer}")
    print(f"\nBy type:")
    for t, c in type_counter.most_common():
        print(f"  {t}: {c}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    convert()
