"""
KGT Pipeline path selection accuracy analysis on PcQA

Loads PcQA eval_v2.jsonl samples, runs KGT pipeline (without NL generation),
and compares predicted paths (optimal_path.relations) against gold relations.

Usage:
    docker exec python-primekgqa-experiment python -m debug_kgt_paths --num 30
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# Gold relation -> Neo4j relation type mapping
# ============================================================================
# The gold data uses semantic relation names (e.g. "gene_drives_cancer"),
# while KGT pipeline returns Neo4j relation types (e.g. "DRIVING_TO").
# Multi-hop relations are listed in traversal order.
RELATION_TO_NEO4J = {
    "gene_drives_cancer": ["DRIVING_TO"],
    "drug_treats_cancer": ["TREATMENT"],
    "drug_activates_gene": ["ACTIVATION_TO"],
    "gene_inhibited_by_drug": ["INHIBITION_TO"],
    "gene_inhibited_by_drug_fda": ["INHIBITION_TO"],
    "gene_inhibited_by_drug_nmpa": ["INHIBITION_TO"],
    "cancercell_resistance_drug": ["RESISTANCE_TO"],
    "cancercell_sensitivity_drug": ["SENSITIVITY_TO"],
    "treatment_for_cancercell": ["SENSITIVITY_TO"],
    "gene_causes_disease": ["CAUSE_TO"],
    "disease_develops_cancer": ["DEVELOP_TO"],
    "disease_induce_cancer": ["DEVELOP_TO"],
    "cancer_has_mutations": ["ORIGINATED_FROM", "HAS_VAR"],
    "cancer_has_fusions": ["ORIGINATED_FROM", "HAS_VAR"],
    "mutation_drives_cancer": ["HAS_VAR", "ORIGINATED_FROM"],
}


def extract_neo4j_rels_from_path(path_str: str) -> List[str]:
    """Extract Neo4j relation types from the path string in gold data.

    Example: "(Drug)-[INHIBITION_TO]->(Genesymbol)" -> ["INHIBITION_TO"]
    """
    return re.findall(r"\[([A-Z_0-9]+)\]", path_str)


def get_gold_neo4j_relations(sample: Dict[str, Any]) -> List[str]:
    """Get gold Neo4j relation types for a sample.

    Tries the mapping first, falls back to extracting from the path field.
    """
    relation = sample.get("relation", "")
    if relation in RELATION_TO_NEO4J:
        return RELATION_TO_NEO4J[relation]

    # Fallback: extract from path field
    path_str = sample.get("path", "")
    if path_str:
        return extract_neo4j_rels_from_path(path_str)

    return [relation]


def normalize_rel(r: str) -> str:
    """Normalize a relation name for comparison."""
    return r.strip().upper().replace(" ", "_").replace("-", "_")


def relations_match(predicted: List[str], gold: List[str]) -> bool:
    """Check if predicted relations match gold (order-sensitive, case-insensitive)."""
    if len(predicted) != len(gold):
        return False
    return all(
        normalize_rel(p) == normalize_rel(g)
        for p, g in zip(predicted, gold)
    )


def compute_entity_accuracy(
    gold_answers: List[str], predicted: List[str]
) -> Dict[str, Any]:
    """Compute entity-level accuracy (case-insensitive)."""
    gold_set = {a.lower() for a in gold_answers}
    pred_set = {e.lower() for e in predicted}
    overlap = gold_set & pred_set
    recall = len(overlap) / len(gold_set) if gold_set else 0.0
    precision = len(overlap) / len(pred_set) if pred_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = gold_set <= pred_set if gold_set else False
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "overlap": len(overlap),
        "gold_count": len(gold_set),
        "pred_count": len(pred_set),
    }


def truncate(s: str, maxlen: int = 60) -> str:
    """Truncate string for display."""
    if len(s) <= maxlen:
        return s
    return s[: maxlen - 3] + "..."


def main():
    parser = argparse.ArgumentParser(
        description="KGT Pipeline path accuracy analysis on PcQA"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=30,
        help="Number of samples to evaluate (default: 30)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/pcqa/qa/eval_v2.jsonl",
        help="Path to eval data (default: data/pcqa/qa/eval_v2.jsonl)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    # Load samples
    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num:
                break
            samples.append(json.loads(line.strip()))

    print(f"Loaded {len(samples)} samples from {data_path}")
    print()

    # Initialize KGT pipeline for PcQA
    from pipeline.kgt.pipeline import KGTPipeline

    pipeline = KGTPipeline(kg_type="pcqa")
    print("KGT pipeline initialized (kg_type=pcqa)")
    print("=" * 80)

    # Run pipeline on each sample
    results = []
    path_correct = 0
    path_wrong = 0
    entity_correct = 0
    errors = 0

    for i, sample in enumerate(samples):
        question = sample["question"]
        entity_name = sample["entity"]
        gold_relation = sample["relation"]
        gold_neo4j_rels = get_gold_neo4j_relations(sample)
        gold_answers = [a["name"] for a in sample.get("answers", [])]

        print(f"\n[{i:3d}] Q: {truncate(question, 70)}")
        print(f"      Entity: {entity_name} | Gold relation: {gold_relation}")
        print(f"      Gold Neo4j rels: {gold_neo4j_rels} | Gold answers: {len(gold_answers)}")

        t0 = time.time()
        try:
            result = pipeline.run(
                question=question,
                entity_name=entity_name,
                generate_nl=False,
            )
            elapsed_ms = (time.time() - t0) * 1000
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            print(f"      ERROR: {e}")
            errors += 1
            results.append({
                "idx": i,
                "question": question,
                "entity": entity_name,
                "gold_relation": gold_relation,
                "gold_neo4j_rels": gold_neo4j_rels,
                "gold_answers": gold_answers,
                "predicted_rels": None,
                "predicted_answers": [],
                "path_match": False,
                "entity_accuracy": False,
                "error": str(e),
                "elapsed_ms": elapsed_ms,
            })
            continue

        # Extract predicted relations
        predicted_rels = (
            result.optimal_path.relations if result.optimal_path else []
        )
        predicted_answers = result.answer_entities or []

        # Check path match
        is_path_match = relations_match(predicted_rels, gold_neo4j_rels)

        # Check entity accuracy
        entity_metrics = compute_entity_accuracy(gold_answers, predicted_answers)
        is_entity_accurate = entity_metrics["accuracy"]

        if is_path_match:
            path_correct += 1
            status = "PATH OK"
        else:
            path_wrong += 1
            status = "PATH WRONG"

        if is_entity_accurate:
            entity_correct += 1

        print(f"      Predicted rels: {predicted_rels}")
        print(f"      Predicted answers: {len(predicted_answers)} | "
              f"Recall: {entity_metrics['recall']:.1%} | "
              f"F1: {entity_metrics['f1']:.1%}")
        print(f"      -> {status} | Entity acc: {is_entity_accurate} | "
              f"{elapsed_ms:.0f}ms")

        results.append({
            "idx": i,
            "question": question,
            "entity": entity_name,
            "gold_relation": gold_relation,
            "gold_neo4j_rels": gold_neo4j_rels,
            "gold_answers": gold_answers,
            "predicted_rels": predicted_rels,
            "predicted_answers": predicted_answers,
            "path_match": is_path_match,
            "entity_accuracy": is_entity_accurate,
            "entity_recall": entity_metrics["recall"],
            "entity_f1": entity_metrics["f1"],
            "error": None,
            "elapsed_ms": elapsed_ms,
        })

    # ========================================================================
    # Summary
    # ========================================================================
    total = len(results)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total samples:        {total}")
    print(f"  Errors:               {errors}")
    print(f"  Path match:           {path_correct}/{total} ({path_correct / total * 100:.1f}%)")
    print(f"  Path wrong:           {path_wrong}/{total} ({path_wrong / total * 100:.1f}%)")
    print(f"  Entity accuracy:      {entity_correct}/{total} ({entity_correct / total * 100:.1f}%)")

    # Average metrics for non-error samples
    non_error = [r for r in results if r["error"] is None]
    if non_error:
        avg_recall = sum(r["entity_recall"] for r in non_error) / len(non_error)
        avg_f1 = sum(r["entity_f1"] for r in non_error) / len(non_error)
        avg_latency = sum(r["elapsed_ms"] for r in non_error) / len(non_error)
        print(f"  Avg entity recall:    {avg_recall:.1%}")
        print(f"  Avg entity F1:        {avg_f1:.1%}")
        print(f"  Avg latency:          {avg_latency:.0f}ms")

    # ========================================================================
    # Wrong path details
    # ========================================================================
    wrong_cases = [r for r in results if not r["path_match"]]
    if wrong_cases:
        print("\n" + "=" * 80)
        print("WRONG PATH DETAILS")
        print("=" * 80)
        print(
            f"{'Idx':>4s} | {'Question':<55s} | {'Gold rels':<35s} | "
            f"{'Predicted rels':<35s} | {'Gold#':>5s} | {'Pred#':>5s}"
        )
        print("-" * 150)
        for r in wrong_cases:
            gold_str = " -> ".join(r["gold_neo4j_rels"]) if r["gold_neo4j_rels"] else "(none)"
            pred_str = (
                " -> ".join(r["predicted_rels"])
                if r["predicted_rels"]
                else "(none/error)"
            )
            print(
                f"{r['idx']:4d} | {truncate(r['question'], 55):<55s} | "
                f"{truncate(gold_str, 35):<35s} | "
                f"{truncate(pred_str, 35):<35s} | "
                f"{len(r['gold_answers']):5d} | "
                f"{len(r['predicted_answers']):5d}"
            )

    # ========================================================================
    # Breakdown by gold relation type
    # ========================================================================
    print("\n" + "=" * 80)
    print("BREAKDOWN BY RELATION TYPE")
    print("=" * 80)
    from collections import Counter

    rel_counts: Dict[str, Dict[str, int]] = {}
    for r in results:
        rel = r["gold_relation"]
        if rel not in rel_counts:
            rel_counts[rel] = {"total": 0, "path_ok": 0, "entity_ok": 0}
        rel_counts[rel]["total"] += 1
        if r["path_match"]:
            rel_counts[rel]["path_ok"] += 1
        if r.get("entity_accuracy", False):
            rel_counts[rel]["entity_ok"] += 1

    print(
        f"{'Relation':<40s} | {'Total':>5s} | {'Path OK':>7s} | "
        f"{'Path%':>6s} | {'Ent OK':>6s} | {'Ent%':>6s}"
    )
    print("-" * 85)
    for rel in sorted(rel_counts.keys()):
        c = rel_counts[rel]
        path_pct = c["path_ok"] / c["total"] * 100 if c["total"] else 0
        ent_pct = c["entity_ok"] / c["total"] * 100 if c["total"] else 0
        print(
            f"{rel:<40s} | {c['total']:5d} | {c['path_ok']:7d} | "
            f"{path_pct:5.1f}% | {c['entity_ok']:6d} | {ent_pct:5.1f}%"
        )


if __name__ == "__main__":
    main()
