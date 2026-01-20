"""
SAFE Pipeline デバッグスクリプト

使用方法:
  python pipeline/debug_safe_pipeline.py --data result/dataset_v2/two_hop.jsonl --index 0
  python pipeline/debug_safe_pipeline.py --question "What diseases are associated with BRCA1?" --entity "BRCA1"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.safe_pipeline import SAFEPipeline, SAFEResult


def load_samples(data_path: Path, indices: List[int]) -> List[tuple]:
    """複数インデックスのサンプルを読み込む"""
    indices_set = set(indices)
    max_idx = max(indices)
    samples = {}

    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > max_idx:
                break
            if i in indices_set:
                samples[i] = json.loads(line.strip())

    return [(idx, samples[idx]) for idx in indices if idx in samples]


def get_gold_info(sample: Dict[str, Any]) -> Dict[str, Any]:
    """正解情報を抽出"""
    gold_answers = [node["name"] for node in sample.get("answer_nodes", [])]

    # クエリタイプを検出
    if "anchor_c_name" in sample:
        query_type = "three_intersection"
        relations = [sample.get("anchor_a_rel"), sample.get("anchor_b_rel"), sample.get("anchor_c_rel")]
    elif "anchor_b_name" in sample:
        query_type = "two_intersection"
        relations = [sample.get("anchor_a_rel"), sample.get("anchor_b_rel")]
    elif "rel2" in sample:
        query_type = "two_hop"
        relations = [sample.get("rel1"), sample.get("rel2")]
    else:
        query_type = "one_hop"
        relations = [sample.get("relation")]

    return {
        "query_type": query_type,
        "relations": [r for r in relations if r],
        "answers": gold_answers,
    }


def run_debug(
    question: str,
    entity_name: Optional[str],
    pipeline: SAFEPipeline,
    gold_info: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """デバッグ実行"""

    if verbose:
        print("\n" + "=" * 60)
        print("INPUT")
        print("=" * 60)
        print(f"Question: {question}")
        print(f"Entity: {entity_name}")
        if gold_info:
            print(f"Gold: {json.dumps(gold_info, indent=2, ensure_ascii=False)}")

    # パイプライン実行
    result = pipeline.run(question=question, entity_name=entity_name)

    if verbose:
        print("\n" + "=" * 60)
        print("PROCESSING LOG")
        print("=" * 60)
        for line in result.processing_log:
            print(f"  {line}")

        print("\n" + "=" * 60)
        print("PSEUDO EDGES")
        print("=" * 60)
        for e in result.pseudo_edges:
            anchor_mark = " [ANCHOR]" if e.is_anchor else ""
            print(f"  ({e.src_node}:{e.src_type})-[{e.relation}]->({e.tgt_node}:{e.tgt_type}){anchor_mark}")

        print("\n" + "=" * 60)
        print("QUERY GRAPHS")
        print("=" * 60)
        print(f"  Total candidates: {len(result.candidate_query_graphs)}")
        for i, qg in enumerate(result.candidate_query_graphs[:3]):
            print(f"\n  [{i+1}] Distance: {qg.total_distance:.4f}")
            for se, pe in qg.edges:
                print(f"      ({se.src_type})-[{se.relation}]->({se.tgt_type})")

        if result.best_query_graph:
            print("\n  Best Query Graph:")
            for se, pe in result.best_query_graph.edges:
                print(f"    ({se.src_type})-[{se.relation}]->({se.tgt_type})")

        print("\n" + "=" * 60)
        print("MATCHED SUBGRAPHS")
        print("=" * 60)
        print(f"  Total: {len(result.matched_subgraphs)}")
        for i, sg in enumerate(result.matched_subgraphs[:5]):
            print(f"  [{i+1}] Score: {sg.score:.4f}")
            print(f"      Nodes: {sg.nodes}")
            print(f"      Edges: {sg.edges}")

        print("\n" + "=" * 60)
        print("ANSWERS")
        print("=" * 60)
        print(f"  Entities ({len(result.answer_entities)}): {result.answer_entities[:10]}")

    # 評価（集合ベース）
    accuracy = False
    recall = 0.0
    precision = 0.0
    f1 = 0.0

    if gold_info:
        gold_set = set(gold_info["answers"])
        pred_set = set(result.answer_entities)

        # 集合ベースのメトリクス
        overlap = gold_set & pred_set
        accuracy = len(overlap) > 0
        recall = len(overlap) / len(gold_set) if gold_set else 0.0
        precision = len(overlap) / len(pred_set) if pred_set else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        if verbose:
            print("\n" + "=" * 60)
            print("EVALUATION")
            print("=" * 60)
            print(f"  Accuracy:  {accuracy}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  F1:        {f1:.3f}")
            print(f"  Gold Relations: {gold_info['relations']}")

            if overlap:
                print(f"\n  Correct predictions: {list(overlap)[:5]}")
            else:
                print(f"\n  No overlap with gold answers")
                print(f"    Gold (first 5): {gold_info['answers'][:5]}")

    return {
        "question": question,
        "entity_name": entity_name,
        "num_pseudo_edges": len(result.pseudo_edges),
        "num_query_graphs": len(result.candidate_query_graphs),
        "num_subgraphs": len(result.matched_subgraphs),
        "answer_entities": result.answer_entities,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def main():
    p = argparse.ArgumentParser(description="Debug SAFE Pipeline")
    p.add_argument("--data", type=Path, help="JSONL data file")
    p.add_argument("--index", type=int, nargs="+", default=[0], help="Sample index(es)")
    p.add_argument("--question", type=str, help="Direct question input")
    p.add_argument("--entity", type=str, help="Entity name for direct question")
    p.add_argument("--quiet", "-q", action="store_true", help="Show only summary")
    args = p.parse_args()

    print("Initializing SAFE Pipeline...")
    pipeline = SAFEPipeline()

    if args.question:
        # 直接質問入力
        run_debug(args.question, args.entity, pipeline, verbose=not args.quiet)
    elif args.data:
        # データファイルから
        samples = load_samples(args.data, args.index)
        if not samples:
            print("No samples found")
            return

        results = []
        for idx, sample in samples:
            print(f"\n{'#'*60}")
            print(f"# INDEX: {idx}")
            print(f"{'#'*60}")

            question = sample.get("question", "")
            entity_name = sample.get("anchor_name", sample.get("anchor_a_name", ""))
            gold_info = get_gold_info(sample)

            result = run_debug(question, entity_name, pipeline, gold_info, verbose=not args.quiet)
            result["index"] = idx
            results.append(result)

        # サマリー（複数indexの場合）
        if len(results) > 1:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            total = len(results)
            acc_count = sum(1 for r in results if r.get("accuracy"))
            avg_recall = sum(r.get("recall", 0) for r in results) / total
            avg_f1 = sum(r.get("f1", 0) for r in results) / total

            print(f"{'Index':<8} {'Acc':<6} {'Recall':<8} {'F1':<8} {'#Answers':<10}")
            print("-" * 60)
            for r in results:
                acc = "Y" if r.get("accuracy") else "N"
                rec = f"{r.get('recall', 0):.2f}"
                f1_val = f"{r.get('f1', 0):.2f}"
                n_ans = len(r.get("answer_entities", []))
                print(f"{r['index']:<8} {acc:<6} {rec:<8} {f1_val:<8} {n_ans:<10}")

            print("-" * 60)
            print(f"Total: Acc={acc_count}/{total} ({acc_count/total*100:.1f}%), Recall={avg_recall*100:.1f}%, F1={avg_f1*100:.1f}%")
    else:
        print("Please specify --data or --question")


if __name__ == "__main__":
    main()
