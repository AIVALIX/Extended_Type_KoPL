"""
Type KoPLパイプラインのデバッグスクリプト
各ステップの詳細を出力
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dataset_construction.add_kopl import (
    build_transition_map_from_kg,
    build_transition_map_multi_from_neo4j,
)
from pipeline.type_kopl_pipeline import (
    TypeKoPLPipeline,
    TypeKoPLPipelineConfig,
)


def debug_single_sample(
    sample: Dict[str, Any],
    pipeline: TypeKoPLPipeline,
    dataset_type: str,
) -> None:
    """1サンプルを詳細デバッグ"""

    question = sample.get("question", "")
    is_intersection = dataset_type in [
        "two_anchor_intersection",
        "three_anchor_intersection",
    ]

    # データセットタイプに応じてentityとgold relationsを取得
    if dataset_type == "two_hop_chain":
        entity_name = sample.get("anchor_name", "")
        gold_relations = [sample.get("rel1", ""), sample.get("rel2", "")]
        anchors = {}
    elif dataset_type == "two_anchor_intersection":
        entity_name = sample.get("anchorA_name", "")
        gold_relations = [
            sample.get("anchorA_edge_type", ""),
            sample.get("anchorB_edge_type", ""),
        ]
        anchors = {
            "anchorA": sample.get("anchorA_name", ""),
            "anchorB": sample.get("anchorB_name", ""),
        }
    elif dataset_type == "three_anchor_intersection":
        entity_name = sample.get("anchorA_name", "")
        gold_relations = [
            sample.get("anchorA_edge_type", ""),
            sample.get("anchorB_edge_type", ""),
            sample.get("anchorC_edge_type", ""),
        ]
        anchors = {
            "anchorA": sample.get("anchorA_name", ""),
            "anchorB": sample.get("anchorB_name", ""),
            "anchorC": sample.get("anchorC_name", ""),
        }
    else:
        entity_name = sample.get("anchor_name", "") or sample.get("anchorA_name", "")
        gold_relations = []
        anchors = {}

    gold_answers = [node["name"] for node in sample.get("answer_nodes_sample", [])]

    print("=" * 80)
    print("INPUT")
    print("=" * 80)
    print(f"Question: {question}")
    if is_intersection:
        print(f"Anchors: {anchors}")
    else:
        print(f"Entity: {entity_name}")
    print(f"Gold Relations: {gold_relations}")
    print(f"Gold Answers: {gold_answers}")

    # Step 1: Type KoPL生成
    print("\n" + "=" * 80)
    print("STEP 1: Type KoPL Generation")
    print("=" * 80)
    type_kopl = pipeline.generate_type_kopl(question)
    print(type_kopl)

    # Step 2: パース
    print("\n" + "=" * 80)
    print("STEP 2: Parse Type KoPL")
    print("=" * 80)
    parsed = pipeline.parse_kopl(type_kopl)
    print(f"Anchors ({len(parsed.anchors)}):")
    for a in parsed.anchors:
        print(f"  - {a.var_name}: {a.anchor_type} ({a.entity_name})")
    print(f"Transitions ({len(parsed.transitions)}):")
    for t in parsed.transitions:
        print(
            f"  - {t.var_name}: {t.anchor_type} --[{t.relation_name}]--> {t.target_type}"
        )
    if parsed.and_ops:
        print(f"And ops ({len(parsed.and_ops)}):")
        for a in parsed.and_ops:
            print(f"  - And({a.exp1}, {a.exp2})")

    # Step 3: スキーマグラフ探索
    print("\n" + "=" * 80)
    print("STEP 3: Schema Graph Search")
    print("=" * 80)
    schema_candidates = pipeline.search_schema_candidates(parsed)
    print(f"Found {len(schema_candidates)} candidates:")
    for i, c in enumerate(schema_candidates[:10]):
        print(f"  [{i+1}] {c.relation_path}")
    if len(schema_candidates) > 10:
        print(f"  ... and {len(schema_candidates) - 10} more")

    # Step 5: KoPL Relations抽出
    print("\n" + "=" * 80)
    print("STEP 5: Extract KoPL Relations")
    print("=" * 80)
    kopl_relations = pipeline.extract_kopl_relations(parsed)
    print(f"KoPL Relations: {kopl_relations}")
    print(f"Gold Relations: {gold_relations}")
    print(f"Match: {set(kopl_relations) & set(gold_relations)}")

    if is_intersection:
        # =============================================
        # Intersection処理（1-hop per anchor）
        # =============================================
        print("\n" + "=" * 80)
        print("STEP 4-8: INTERSECTION PROCESSING (Single-Hop)")
        print("=" * 80)

        anchor_results = {}
        anchor_paths = {}
        anchor_keys_list = list(anchors.keys())

        for i, (anchor_key, anchor_entity) in enumerate(anchors.items()):
            print(f"\n--- [{anchor_key}] Entity: {anchor_entity} ---")

            # Type KoPLからこのアンカーの型情報を抽出
            anchor_type, target_types = pipeline._extract_target_types_for_anchor(
                parsed, i
            )
            print(f"  Type KoPL schema: {anchor_type} -> {target_types}")

            # 1-hop候補を取得
            single_hop_paths = pipeline._get_single_hop_candidates(anchor_entity)
            print(f"  Single-hop candidates (raw): {len(single_hop_paths)}")
            for j, sp in enumerate(single_hop_paths[:5]):
                print(
                    f"    [{j+1}] {sp['relation_path']} -> {len(sp['reachable_entities'])} entities"
                )

            if not single_hop_paths:
                anchor_results[anchor_key] = []
                anchor_paths[anchor_key] = []
                continue

            # Type KoPL Schema剪定
            if anchor_type and target_types:
                filtered_paths = pipeline._filter_by_type_kopl_schema(
                    single_hop_paths, anchor_type, target_types
                )
                print(f"  After Type KoPL pruning: {len(filtered_paths)}")
                for fp in filtered_paths[:5]:
                    print(f"    - {fp['relation_path']}")
            else:
                filtered_paths = single_hop_paths
                print(f"  Type KoPL pruning skipped")

            if not filtered_paths:
                print(f"  No paths after pruning, using all candidates")
                filtered_paths = single_hop_paths

            # ベクトル剪定でtop3に絞る
            if kopl_relations:
                scored_paths = pipeline.vector_scoring(kopl_relations, filtered_paths)
                print(f"  After vector scoring (top 3):")
                for idx, score in scored_paths[:3]:
                    print(f"    - {filtered_paths[idx]['relation_path']}: {score:.4f}")
            else:
                scored_paths = [(j, 0.0) for j in range(len(filtered_paths))]

            # LLM Rerankerでtop3から最適なrelationを選択
            final_path = None
            top_k_for_rerank = 3
            if scored_paths:
                top_paths = [
                    filtered_paths[idx]["relation_path"]
                    for idx, _ in scored_paths[:top_k_for_rerank]
                ]
                print(f"  LLM Reranking candidates: {top_paths}")
                final_path = pipeline.llm_rerank(
                    question,
                    kopl_relations,
                    filtered_paths,
                    scored_paths[:top_k_for_rerank],
                )
                if final_path:
                    print(f"  LLM Reranker selected: {final_path}")
                else:
                    print(f"  LLM Reranker returned None")

            if final_path is None and scored_paths:
                best_idx, _ = scored_paths[0]
                final_path = filtered_paths[best_idx]["relation_path"]
                print(f"  Fallback to vector top: {final_path}")

            if final_path:
                anchor_paths[anchor_key] = final_path
                selected_rel = final_path[0] if final_path else None
                if selected_rel:
                    entities = pipeline._get_entities_via_relation(
                        anchor_entity, selected_rel, max_results=200
                    )
                else:
                    entities = []
                anchor_results[anchor_key] = entities
                print(f"  Selected relation: {final_path}")
                print(f"  Reachable entities ({len(entities)}): {entities[:5]}...")
            else:
                anchor_results[anchor_key] = []
                anchor_paths[anchor_key] = []

        # 交差計算
        print("\n" + "=" * 80)
        print("INTERSECTION CALCULATION")
        print("=" * 80)
        all_sets = [set(ents) for ents in anchor_results.values() if ents]
        print(f"Anchor sets sizes: {[len(s) for s in all_sets]}")

        if len(all_sets) >= 2:
            intersection = all_sets[0]
            for s in all_sets[1:]:
                intersection = intersection & s
            entities = list(intersection)[:10]
            print(f"Intersection size: {len(intersection)}")
        elif len(all_sets) == 1:
            entities = list(all_sets[0])[:10]
            print(f"Only one anchor with results")
        else:
            entities = []
            print(f"No valid results for intersection")

        # すべてのアンカーパスを収集
        all_predicted_paths = []
        for path in anchor_paths.values():
            if path:
                all_predicted_paths.extend(path)
        final_path = all_predicted_paths

    else:
        # =============================================
        # 通常処理（単一アンカー）
        # =============================================
        # Step 4: Neo4j検証
        print("\n" + "=" * 80)
        print("STEP 4: Neo4j Validation")
        print("=" * 80)
        valid_paths = pipeline.validate_paths_in_neo4j(entity_name, schema_candidates)
        print(f"Valid paths: {len(valid_paths)}/{len(schema_candidates)}")
        for i, vp in enumerate(valid_paths[:10]):
            print(f"  [{i+1}] {vp['relation_path']}")
            print(f"       -> {vp['reachable_entities'][:3]}...")
        if len(valid_paths) > 10:
            print(f"  ... and {len(valid_paths) - 10} more")

        # Step 6: ベクトル剪定
        print("\n" + "=" * 80)
        print("STEP 6: Vector Scoring")
        print("=" * 80)
        if valid_paths and kopl_relations:
            scored_paths = pipeline.vector_scoring(kopl_relations, valid_paths)
            print(f"Scored {len(scored_paths)} paths:")
            for idx, score in scored_paths[:10]:
                path = valid_paths[idx]["relation_path"]
                print(f"  [{idx}] {path}: {score:.4f}")
        else:
            scored_paths = []
            print("No paths to score")

        # Step 7: LLM Reranker
        print("\n" + "=" * 80)
        print("STEP 7: LLM Reranking")
        print("=" * 80)
        if scored_paths:
            final_path = pipeline.llm_rerank(
                question, kopl_relations, valid_paths, scored_paths
            )
            print(f"Selected path: {final_path}")
        else:
            final_path = None
            print("No paths for reranking")

        # フォールバック
        if final_path is None and scored_paths:
            best_idx, _ = scored_paths[0]
            final_path = valid_paths[best_idx]["relation_path"]
            print(f"Fallback to: {final_path}")

        # Step 8: 最終エンティティ
        print("\n" + "=" * 80)
        print("STEP 8: Final Entities")
        print("=" * 80)
        if final_path:
            entities = pipeline.get_final_entities(entity_name, final_path)
            print(f"Final path: {final_path}")
            print(f"Predicted entities ({len(entities)}): {entities[:10]}")
        else:
            entities = []
            print("No final path")

    # 評価
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print(f"Gold answers: {gold_answers}")
    print(f"Predicted: {entities[:10]}")

    gold_set = set(gold_answers)
    pred_set = set(entities[:10])
    hits = gold_set & pred_set
    print(f"Hits: {hits if hits else 'None'}")

    # Path評価
    if final_path:
        path_match_any = any(rel in final_path for rel in gold_relations)
        path_match_all = all(rel in final_path for rel in gold_relations)
        print(f"Path match (any): {path_match_any}")
        print(f"Path match (all): {path_match_all}")
        print(f"Final path set: {set(final_path)}")
        print(f"Gold relations set: {set(gold_relations)}")


def main():
    p = argparse.ArgumentParser(description="パイプラインデバッグ")
    # Either:
    # - dataset mode: --data + --type (+ --index)
    # - question mode: --question + --entity (+ --dataset-type)
    p.add_argument(
        "--data",
        type=Path,
        required=False,
        help="データセットパス（JSONL）。--type とセットで使用",
    )
    p.add_argument(
        "--type",
        type=str,
        choices=[
            "two_hop_chain",
            "two_anchor_intersection",
            "three_anchor_intersection",
        ],
        required=False,
        help="データセットタイプ。--data とセットで使用",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="サンプルインデックス",
    )
    p.add_argument(
        "--question",
        type=str,
        required=False,
        help="質問文（JSONL無しで1問デバッグする場合に使用）",
    )
    p.add_argument(
        "--entity",
        type=str,
        required=False,
        help="開始エンティティ名（--question モードで必須）",
    )
    p.add_argument(
        "--dataset-type",
        type=str,
        choices=[
            "two_hop_chain",
            "two_anchor_intersection",
            "three_anchor_intersection",
        ],
        default="two_hop_chain",
        help="--question モード時のデータセットタイプ（default: two_hop_chain）",
    )
    p.add_argument("--nodes-csv", type=Path, default=Path("data/kg/nodes.csv"))
    p.add_argument("--rels-csv", type=Path, default=Path("data/kg/relationships.csv"))
    p.add_argument("--no-vector-prune", action="store_true")
    p.add_argument("--no-llm-rerank", action="store_true")

    args = p.parse_args()

    # TransitionMap
    print("Building TransitionMap...")
    transitions = build_transition_map_from_kg(
        nodes_csv=args.nodes_csv, rels_csv=args.rels_csv
    )

    print("Building TransitionMapMulti from Neo4j...")
    transitions_multi = build_transition_map_multi_from_neo4j()

    # Pipeline
    config = TypeKoPLPipelineConfig(
        use_vector_prune=not args.no_vector_prune,
        use_llm_reranker=not args.no_llm_rerank,
    )
    pipeline = TypeKoPLPipeline(
        transitions=transitions,
        config=config,
        transitions_multi=transitions_multi,
    )

    # Mode selection
    if args.question:
        if not args.entity:
            print("Error: --question モードでは --entity が必要です")
            print('例: --question "..." --entity "Glutathione disulfide"')
            return
        sample = {
            "question": args.question,
            "anchor_name": args.entity,
            "answer_nodes_sample": [],
        }
        debug_single_sample(sample, pipeline, args.dataset_type)
        return

    if not args.data or not args.type:
        p.error(
            "--data と --type を指定するか、--question と --entity を指定してください"
        )

    # Dataset mode: サンプル読み込み
    samples = []
    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.index >= len(samples):
        print(f"Error: index {args.index} >= {len(samples)}")
        return

    sample = samples[args.index]
    debug_single_sample(sample, pipeline, args.type)


if __name__ == "__main__":
    main()
"""
docker compose run --rm app python pipeline/debug_pipeline.py --question "Which diseases are linked to genes targeted by Glutathione disulfide?" --entity "Glutathione disulfide" --dataset-type two_hop_chain
"""
