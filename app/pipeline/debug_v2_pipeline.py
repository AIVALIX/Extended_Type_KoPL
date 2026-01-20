"""
v2データセット用のデバッグスクリプト

使用方法:
  python pipeline/debug_v2_pipeline.py --data result/dataset_v2/two_hop.jsonl --index 0
  python pipeline/debug_v2_pipeline.py --data result/dataset_v2/two_hop.jsonl --index 0 1 2 3 4
  python pipeline/debug_v2_pipeline.py --data result/dataset_v2/one_hop.jsonl --index 5 10 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_construction.add_kopl import (
    build_transition_map_from_kg,
    build_transition_map_multi_from_neo4j,
)
from pipeline.type_kopl_pipeline import (
    TypeKoPLPipeline,
    TypeKoPLPipelineConfig,
)


def load_sample(data_path: Path, index: int) -> Dict[str, Any]:
    """指定インデックスのサンプルを読み込む"""
    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())
    raise ValueError(f"Index {index} out of range")


def load_samples(data_path: Path, indices: List[int]) -> List[Dict[str, Any]]:
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


def detect_query_type(sample: Dict[str, Any]) -> str:
    """サンプルからクエリタイプを検出"""
    if "anchor_c_name" in sample:
        return "three_intersection"
    elif "anchor_b_name" in sample:
        return "two_intersection"
    elif "rel2" in sample:
        return "two_hop"
    else:
        return "one_hop"


def get_gold_info(sample: Dict[str, Any], query_type: str) -> Dict[str, Any]:
    """正解情報を抽出"""
    gold_answers = [node["name"] for node in sample.get("answer_nodes", [])]

    if query_type == "one_hop":
        return {
            "entity_name": sample.get("anchor_name", ""),
            "relations": [sample.get("relation", "")],
            "answers": gold_answers,
        }
    elif query_type == "two_hop":
        return {
            "entity_name": sample.get("anchor_name", ""),
            "mid_name": sample.get("mid_name", ""),
            "relations": [sample.get("rel1", ""), sample.get("rel2", "")],
            "answers": gold_answers,
        }
    elif query_type == "two_intersection":
        return {
            "anchors": {
                "anchorA": sample.get("anchor_a_name", ""),
                "anchorB": sample.get("anchor_b_name", ""),
            },
            "relations": {
                "anchorA": sample.get("anchor_a_rel", ""),
                "anchorB": sample.get("anchor_b_rel", ""),
            },
            "answers": gold_answers,
        }
    elif query_type == "three_intersection":
        return {
            "anchors": {
                "anchorA": sample.get("anchor_a_name", ""),
                "anchorB": sample.get("anchor_b_name", ""),
                "anchorC": sample.get("anchor_c_name", ""),
            },
            "relations": {
                "anchorA": sample.get("anchor_a_rel", ""),
                "anchorB": sample.get("anchor_b_rel", ""),
                "anchorC": sample.get("anchor_c_rel", ""),
            },
            "answers": gold_answers,
        }
    return {"answers": gold_answers}


def run_debug(
    sample: Dict[str, Any],
    query_type: str,
    pipeline: TypeKoPLPipeline,
    verbose: bool = True,
) -> Dict[str, Any]:
    """デバッグ実行"""

    question = sample.get("question", "")
    gold = get_gold_info(sample, query_type)

    if verbose:
        print("\n" + "=" * 60)
        print("INPUT")
        print("=" * 60)
        print(f"Query Type: {query_type}")
        print(f"Question: {question}")
        print(f"Gold Info: {json.dumps(gold, indent=2, ensure_ascii=False)}")

    # パイプライン実行
    if query_type in ["two_intersection", "three_intersection"]:
        anchors = gold["anchors"]
        relations = gold["relations"]
        result = pipeline.run_intersection(
            question=question,
            anchors=anchors,
            relations=relations,
        )
    else:
        entity_name = gold.get("entity_name", "")
        result = pipeline.run(question=question, entity_name=entity_name)

    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE OUTPUT")
        print("=" * 60)
        print(f"\n[Generated Type KoPL]")
        print(result.type_kopl)

        print(f"\n[Parsed KoPL]")
        if result.parsed_kopl:
            print(f"  Steps: {len(result.parsed_kopl.steps)}")
            for i, step in enumerate(result.parsed_kopl.steps):
                print(f"    {i+1}. {step}")
            print(f"  Transitions: {result.parsed_kopl.get_type_path()}")
        else:
            print("  (parse failed)")

        print(f"\n[Schema Candidates]")
        print(f"  Count: {len(result.schema_candidates)}")
        if result.schema_candidates:
            for i, cand in enumerate(result.schema_candidates[:5]):
                print(f"    {i+1}. {cand.relation_path} (types: {cand.type_path})")

        print(f"\n[Valid Paths]")
        print(f"  Count: {len(result.valid_paths)}")
        if result.valid_paths:
            for i, vp in enumerate(result.valid_paths[:5]):
                print(f"    {i+1}. {vp.get('relations', vp)}")

        print(f"\n[Final Path]")
        print(f"  {result.final_path}")

        print(f"\n[Reachable Entities]")
        print(f"  Count: {len(result.reachable_entities)}")
        if result.reachable_entities:
            print(f"  Top 10: {result.reachable_entities[:10]}")

    # 評価（常に計算）
    gold_set = set(gold["answers"])
    pred_set = set(result.reachable_entities)

    hits_at_1 = bool(gold_set & set(result.reachable_entities[:1]))
    hits_at_10 = bool(gold_set & set(result.reachable_entities[:10]))
    path_match = False
    if result.final_path:
        if query_type in ["one_hop", "two_hop"]:
            path_match = any(r in result.final_path for r in gold.get("relations", []))
        else:
            path_match = any(
                r in result.final_path for r in gold.get("relations", {}).values()
            )

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        print(f"  Hits@1:  {hits_at_1}")
        print(f"  Hits@10: {hits_at_10}")
        print(f"  PathMatch: {path_match}")
        print(f"  Gold Relations: {gold.get('relations', {})}")
        print(f"  Pred Path: {result.final_path}")

        # 重複確認
        overlap = gold_set & pred_set
        if overlap:
            print(f"\n  ✓ Correct predictions: {list(overlap)[:5]}")
        else:
            print(f"\n  ✗ No overlap with gold answers")
            print(f"    Gold (first 5): {gold['answers'][:5]}")

    return {
        "type_kopl": result.type_kopl,
        "final_path": result.final_path,
        "entities": result.reachable_entities,
        "hits_at_1": hits_at_1,
        "hits_at_10": hits_at_10,
        "path_match": path_match,
    }


def main():
    p = argparse.ArgumentParser(description="Debug v2 dataset pipeline")
    p.add_argument("--data", type=Path, required=True, help="JSONL data file")
    p.add_argument(
        "--index", type=int, nargs="+", default=[0], help="Sample index(es) to debug"
    )
    p.add_argument("--nodes-csv", type=Path, default=Path("data/kg/import_nodes.csv"))
    p.add_argument("--rels-csv", type=Path, default=Path("data/kg/import_rels.csv"))
    p.add_argument(
        "--use-schema-prompt", action="store_true", help="Add schema to prompt"
    )
    p.add_argument("--no-vector-prune", action="store_true")
    p.add_argument("--no-llm-rerank", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true", help="Show only summary")
    args = p.parse_args()

    print("Loading samples...")
    samples = load_samples(args.data, args.index)
    if not samples:
        print("No samples found for given indices")
        return

    # 最初のサンプルからquery_typeを判定
    first_sample = samples[0][1]
    query_type = detect_query_type(first_sample)

    print(f"Query type: {query_type}")
    print("Building transition maps...")
    transitions = build_transition_map_from_kg(
        nodes_csv=args.nodes_csv, rels_csv=args.rels_csv
    )
    transitions_multi = None
    if query_type in ["two_intersection", "three_intersection"]:
        transitions_multi = build_transition_map_multi_from_neo4j()

    config = TypeKoPLPipelineConfig(
        use_vector_prune=not args.no_vector_prune,
        use_llm_reranker=not args.no_llm_rerank,
        use_schema_in_prompt=args.use_schema_prompt,
    )

    pipeline = TypeKoPLPipeline(
        transitions=transitions,
        transitions_multi=transitions_multi,
        config=config,
    )

    # 各サンプルを評価
    results = []
    for idx, sample in samples:
        print(f"\n{'#'*60}")
        print(f"# INDEX: {idx}")
        print(f"{'#'*60}")

        result = run_debug(sample, query_type, pipeline, verbose=not args.quiet)
        result["index"] = idx
        result["question"] = sample.get("question", "")
        results.append(result)

    # サマリー表示（複数indexの場合）
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        hits1 = sum(1 for r in results if r.get("hits_at_1"))
        hits10 = sum(1 for r in results if r.get("hits_at_10"))
        total = len(results)

        print(f"{'Index':<8} {'Hits@1':<8} {'Hits@10':<8} {'Path':<20} Question")
        print("-" * 80)
        for r in results:
            path_str = "->".join(r.get("final_path", []) or ["None"])[:18]
            h1 = "✓" if r.get("hits_at_1") else "✗"
            h10 = "✓" if r.get("hits_at_10") else "✗"
            q = r.get("question", "")[:40]
            print(f"{r['index']:<8} {h1:<8} {h10:<8} {path_str:<20} {q}")

        print("-" * 80)
        print(
            f"Total: Hits@1={hits1}/{total} ({hits1/total*100:.1f}%), Hits@10={hits10}/{total} ({hits10/total*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
