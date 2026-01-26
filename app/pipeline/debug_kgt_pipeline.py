"""
KGT Pipeline デバッグスクリプト

使用方法:
  python pipeline/debug_kgt_pipeline.py --data result/dataset_v2/two_hop.jsonl --index 0
  python pipeline/debug_kgt_pipeline.py --question "What diseases are associated with BRCA1?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.kgt_pipeline import KGTPipeline, KGTResult


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
        relations = [
            sample.get("anchor_a_rel"),
            sample.get("anchor_b_rel"),
            sample.get("anchor_c_rel"),
        ]
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
    pipeline: KGTPipeline,
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
        print("ANALYSIS")
        print("=" * 60)
        print(f"  Head Entity: {result.analysis.head_entity_name}")
        print(f"  Head Type: {result.analysis.head_entity_type}")
        print(f"  Tail Type: {result.analysis.tail_entity_type}")

        print("\n" + "=" * 60)
        print("SCHEMA PATHS")
        print("=" * 60)
        print(f"  Total candidates: {len(result.schema_paths)}")
        for i, p in enumerate(result.schema_paths[:5]):
            print(f"    {i+1}. {p.path} (score: {p.score:.3f})")

        if result.optimal_path:
            print(f"\n  Optimal: {result.optimal_path.path}")
            print(f"  Relations: {result.optimal_path.relations}")

        print("\n" + "=" * 60)
        print("CYPHER QUERY")
        print("=" * 60)
        if result.generated_cypher:
            print(f"  {result.generated_cypher}")
        else:
            print("  (none)")

        print("\n" + "=" * 60)
        print("SUBGRAPH")
        print("=" * 60)
        print(f"  Results: {len(result.subgraph)}")
        if result.subgraph:
            for i, r in enumerate(result.subgraph[:5]):
                print(f"    {i+1}. {r}")

        print("\n" + "=" * 60)
        print("ANSWERS")
        print("=" * 60)
        print(
            f"  Entities ({len(result.answer_entities)}): {result.answer_entities[:10]}"
        )
        if result.natural_answer:
            print(f"  Natural: {result.natural_answer}")

    # 評価
    hits_at_1 = False
    hits_at_10 = False
    path_match = False

    if gold_info:
        gold_set = set(gold_info["answers"])
        pred_set = set(result.answer_entities)

        hits_at_1 = bool(gold_set & set(result.answer_entities[:1]))
        hits_at_10 = bool(gold_set & set(result.answer_entities[:10]))

        if result.optimal_path:
            path_match = any(
                r in result.optimal_path.relations for r in gold_info["relations"]
            )

        if verbose:
            print("\n" + "=" * 60)
            print("EVALUATION")
            print("=" * 60)
            print(f"  Hits@1:  {hits_at_1}")
            print(f"  Hits@10: {hits_at_10}")
            print(f"  PathMatch: {path_match}")
            print(f"  Gold Relations: {gold_info['relations']}")
            print(
                f"  Pred Relations: {result.optimal_path.relations if result.optimal_path else []}"
            )

            overlap = gold_set & pred_set
            if overlap:
                print(f"\n  ✓ Correct predictions: {list(overlap)[:5]}")
            else:
                print(f"\n  ✗ No overlap with gold answers")
                print(f"    Gold (first 5): {gold_info['answers'][:5]}")

    return {
        "question": question,
        "entity_name": entity_name,
        "head_type": result.analysis.head_entity_type,
        "tail_type": result.analysis.tail_entity_type,
        "optimal_path": result.optimal_path.relations if result.optimal_path else None,
        "answer_entities": result.answer_entities,
        "hits_at_1": hits_at_1,
        "hits_at_10": hits_at_10,
        "path_match": path_match,
    }


def main():
    p = argparse.ArgumentParser(description="Debug KGT Pipeline")
    p.add_argument("--data", type=Path, help="JSONL data file")
    p.add_argument("--index", type=int, nargs="+", default=[0], help="Sample index(es)")
    p.add_argument("--question", type=str, help="Direct question input")
    p.add_argument("--entity", type=str, help="Entity name for direct question")
    p.add_argument("--quiet", "-q", action="store_true", help="Show only summary")
    args = p.parse_args()

    print("Initializing KGT Pipeline...")
    pipeline = KGTPipeline()

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

            result = run_debug(
                question, entity_name, pipeline, gold_info, verbose=not args.quiet
            )
            result["index"] = idx
            results.append(result)

        # サマリー（複数indexの場合）
        if len(results) > 1:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            hits1 = sum(1 for r in results if r.get("hits_at_1"))
            hits10 = sum(1 for r in results if r.get("hits_at_10"))
            total = len(results)

            print(f"{'Index':<8} {'Hits@1':<8} {'Hits@10':<8} {'Path':<30}")
            print("-" * 60)
            for r in results:
                path_str = "->".join(r.get("optimal_path") or ["None"])[:28]
                h1 = "✓" if r.get("hits_at_1") else "✗"
                h10 = "✓" if r.get("hits_at_10") else "✗"
                print(f"{r['index']:<8} {h1:<8} {h10:<8} {path_str:<30}")

            print("-" * 60)
            print(
                f"Total: Hits@1={hits1}/{total} ({hits1/total*100:.1f}%), Hits@10={hits10}/{total} ({hits10/total*100:.1f}%)"
            )
    else:
        print("Please specify --data or --question")


if __name__ == "__main__":
    main()

"""
docker compose run --rm app python [debug_kgt_pipeline.py](http://_vscodecontentref_/6) --data [two_hop.jsonl](http://_vscodecontentref_/7) --index 0 1 2 3 4
"""
