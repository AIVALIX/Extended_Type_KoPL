"""
統一デバッグスクリプト

使用方法:
  # データセットから実行
  python pipeline/debug.py --pipeline safe --data result/dataset_v2/one_hop.jsonl --index 0
  python pipeline/debug.py --pipeline kgt --data result/dataset_v2/two_hop.jsonl --index 0 1 2

  # 直接質問を指定
  python pipeline/debug.py --pipeline extended_type_kopl --question "What diseases are associated with BRCA1?" --entity "BRCA1"

対応パイプライン:
  - safe: SAFE Pipeline
  - kgt: KGT Pipeline
  - extended_type_kopl: Extended Type-KoPL Pipeline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.common.debug import (
    load_samples,
    get_gold_info,
    compute_metrics,
    print_header,
    print_metrics,
)


def debug_safe(result, gold_info: Optional[Dict[str, Any]] = None):
    """SAFEパイプラインのデバッグ出力"""
    print_header("PROCESSING LOG")
    for line in result.processing_log:
        print(f"  {line}")

    print_header("PSEUDO EDGES")
    for e in result.pseudo_edges:
        anchor_mark = " [ANCHOR]" if e.is_anchor else ""
        print(f"  ({e.src_node}:{e.src_type})-[{e.relation}]->({e.tgt_node}:{e.tgt_type}){anchor_mark}")

    print_header("QUERY GRAPHS")
    print(f"  Total candidates: {len(result.candidate_query_graphs)}")
    for i, qg in enumerate(result.candidate_query_graphs[:3]):
        print(f"\n  [{i+1}] Distance: {qg.total_distance:.4f}")
        for se, pe in qg.edges:
            print(f"      ({se.src_type})-[{se.relation}]->({se.tgt_type})")

    if result.best_query_graph:
        print_header("BEST QUERY GRAPH")
        for se, pe in result.best_query_graph.edges:
            print(f"  ({se.src_type})-[{se.relation}]->({se.tgt_type})")

    print_header("RESULTS")
    print(f"  Matched subgraphs: {len(result.matched_subgraphs)}")
    print(f"  Answer entities: {len(result.answer_entities)}")
    if result.answer_entities:
        print(f"  First 10: {result.answer_entities[:10]}")

    # メトリクス計算
    if gold_info:
        predicted_relations = []
        if result.best_query_graph:
            for se, _ in result.best_query_graph.edges:
                predicted_relations.append(se.relation)

        metrics = compute_metrics(
            gold_info["answers"],
            result.answer_entities,
            gold_info["relations"],
            predicted_relations,
        )
        print_metrics(metrics)


def debug_kgt(result, gold_info: Optional[Dict[str, Any]] = None):
    """KGTパイプラインのデバッグ出力"""
    print_header("PROCESSING LOG")
    for line in result.processing_log:
        print(f"  {line}")

    print_header("ANALYSIS")
    print(f"  Head Entity: {result.analysis.head_entity_name}")
    print(f"  Head Type: {result.analysis.head_entity_type}")
    print(f"  Tail Type: {result.analysis.tail_entity_type}")

    print_header("SCHEMA PATHS")
    print(f"  Total candidates: {len(result.schema_paths)}")
    for i, p in enumerate(result.schema_paths[:5]):
        print(f"    {i+1}. {p.path} (score: {p.score:.3f})")

    if result.optimal_path:
        print_header("OPTIMAL PATH")
        print(f"  Path: {result.optimal_path.path}")
        print(f"  Relations: {result.optimal_path.relations}")
        print(f"  Score: {result.optimal_path.score:.3f}")

    if result.generated_cypher:
        print_header("GENERATED CYPHER")
        print(f"  {result.generated_cypher.strip()}")

    print_header("RESULTS")
    print(f"  Subgraph records: {len(result.subgraph)}")
    print(f"  Answer entities: {len(result.answer_entities)}")
    if result.answer_entities:
        print(f"  First 10: {result.answer_entities[:10]}")

    # メトリクス計算
    if gold_info:
        predicted_relations = result.optimal_path.relations if result.optimal_path else []
        metrics = compute_metrics(
            gold_info["answers"],
            result.answer_entities,
            gold_info["relations"],
            predicted_relations,
        )
        print_metrics(metrics)


def debug_extended_type_kopl(result, gold_info: Optional[Dict[str, Any]] = None):
    """Extended Type-KoPLパイプラインのデバッグ出力"""
    print_header("PROCESSING LOG")
    for line in result.processing_log:
        print(f"  {line}")

    if result.kopl_program:
        print_header("KOPL PROGRAM")
        print(f"  Operation: {result.kopl_program.op_type}")
        print(f"  Relations: {len(result.kopl_program.relations)}")
        for rel in result.kopl_program.relations:
            if rel.intermediate_type:
                print(f"    {rel.src_type} -> {rel.intermediate_type} -> {rel.tgt_type}")
            else:
                print(f"    {rel.src_type} -> {rel.tgt_type}")
        if result.kopl_program.children:
            print(f"  Children: {len(result.kopl_program.children)}")

    print_header("CANDIDATE PATHS")
    print(f"  Total: {len(result.candidate_paths)}")
    for i, p in enumerate(result.candidate_paths[:5]):
        print(f"    {i+1}. [{p.source}] {p.to_text()}")

    print_header("SELECTED PATHS")
    for p in result.selected_paths:
        print(f"  {p.to_text()} (score: {p.score:.4f})")

    print_header("ENTITY SETS")
    for i, es in enumerate(result.entity_sets):
        print(f"  Set {i+1}: {len(es.entities)} entities")
        if es.anchor_name:
            print(f"    Anchor: {es.anchor_name}")

    print_header("RESULTS")
    print(f"  Answer entities: {len(result.answer_entities)}")
    if result.answer_entities:
        print(f"  First 10: {result.answer_entities[:10]}")

    # メトリクス計算
    if gold_info:
        predicted_relations = []
        if result.selected_paths:
            predicted_relations = list(result.selected_paths[0].relations)

        metrics = compute_metrics(
            gold_info["answers"],
            result.answer_entities,
            gold_info["relations"],
            predicted_relations,
        )
        print_metrics(metrics)


def run_debug(
    pipeline_id: str,
    question: str,
    entity_name: Optional[str],
    gold_info: Optional[Dict[str, Any]] = None,
):
    """デバッグ実行"""
    print_header("INPUT")
    print(f"  Pipeline: {pipeline_id}")
    print(f"  Question: {question}")
    print(f"  Entity: {entity_name}")
    if gold_info:
        print(f"  Query Type: {gold_info['query_type']}")
        print(f"  Gold Relations: {gold_info['relations']}")
        print(f"  Gold Answers: {gold_info['answers'][:5]}{'...' if len(gold_info['answers']) > 5 else ''}")

    # パイプライン初期化
    if pipeline_id == "safe":
        from pipeline.safe import SAFEPipeline
        pipeline = SAFEPipeline()
        result = pipeline.run(question=question, entity_name=entity_name)
        debug_safe(result, gold_info)

    elif pipeline_id == "kgt":
        from pipeline.kgt import KGTPipeline
        pipeline = KGTPipeline()
        result = pipeline.run(question=question, entity_name=entity_name)
        debug_kgt(result, gold_info)

    elif pipeline_id == "extended_type_kopl":
        from pipeline.extended_type_kopl import ExtendedTypeKoPLPipeline
        pipeline = ExtendedTypeKoPLPipeline()
        result = pipeline.run(question=question, entity_name=entity_name)
        debug_extended_type_kopl(result, gold_info)

    else:
        raise ValueError(f"Unknown pipeline: {pipeline_id}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Pipeline Debug Tool")
    parser.add_argument("--pipeline", type=str, required=True,
                        choices=["safe", "kgt", "extended_type_kopl"],
                        help="Pipeline to debug")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset file (jsonl)")
    parser.add_argument("--index", type=int, nargs="+", default=[0],
                        help="Sample indices to debug")
    parser.add_argument("--question", type=str, default=None,
                        help="Direct question input")
    parser.add_argument("--entity", type=str, default=None,
                        help="Entity name for direct question")

    args = parser.parse_args()

    if args.question:
        # 直接質問を指定
        run_debug(args.pipeline, args.question, args.entity)
    elif args.data:
        # データセットから読み込み
        data_path = Path(args.data)
        samples = load_samples(data_path, args.index)

        for idx, sample in samples:
            print(f"\n{'#' * 70}")
            print(f"# SAMPLE {idx}")
            print(f"{'#' * 70}")

            gold_info = get_gold_info(sample)
            question = sample.get("question", "")
            entity_name = sample.get(gold_info["entity_key"], "")

            run_debug(args.pipeline, question, entity_name, gold_info)
    else:
        parser.error("Either --data or --question is required")


if __name__ == "__main__":
    main()
