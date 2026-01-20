"""
Type KoPL生成 → サブグラフ抽出のE2Eテストスクリプト

使用方法:
    python llm_process/test_type_kopl_to_subgraph.py --question "糖尿病に関連する遺伝子は？"
    python llm_process/test_type_kopl_to_subgraph.py --question "ACE2に関連する疾患を治療する薬は？"
    python llm_process/test_type_kopl_to_subgraph.py --use-sample  # サンプル質問を使用
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from database.search import GraphPathFinder
from dataset_construction.add_kopl import TransitionMap, build_transition_map_from_kg
from llm_process.schema_subgraph_search import (
    ParsedTypeKoPL,
    SchemaGraph,
    SubgraphCandidate,
    get_subgraph_candidates_from_type_kopl,
    parse_type_kopl,
    search_subgraph_candidates,
)
from llm_process.type_KoPL_process import generate_type_kopl
from prompts.type_KoPL_fewshot import FEW_SHOT_PrimeKGQA, FEW_SHOT_MetaQA


# ────────────────────────────────────────────────────────────────
#  Neo4j連携: サブグラフ候補の検証
# ────────────────────────────────────────────────────────────────
def validate_subgraph_in_neo4j(
    finder: GraphPathFinder,
    entity_name: str,
    relation_path: List[str],
    *,
    max_results: int = 10,
) -> List[str]:
    """
    Neo4jで指定のrelationパスをたどり、到達可能なエンティティを取得する。

    Args:
        finder: GraphPathFinder
        entity_name: 開始エンティティ名
        relation_path: relationのシーケンス
        max_results: 最大結果数

    Returns:
        到達可能なエンティティ名のリスト
    """
    try:
        entities = finder.get_reachable_entities(
            entity_name=entity_name,
            rel_types=relation_path,
            distinct=True,
            no_cycle=True,
        )
        return entities[:max_results]
    except Exception as e:
        print(f"  [Error] Neo4j query failed: {e}")
        return []


def validate_subgraph_in_neo4j_flexible(
    finder: GraphPathFinder,
    entity_name: str,
    relation_path: List[str],
    *,
    max_results: int = 10,
) -> List[str]:
    """
    Neo4jで指定のrelationパスをたどり、到達可能なエンティティを取得する。
    ラベルに依存しない柔軟なクエリを使用。

    Args:
        finder: GraphPathFinder
        entity_name: 開始エンティティ名
        relation_path: relationのシーケンス
        max_results: 最大結果数

    Returns:
        到達可能なエンティティ名のリスト
    """
    if not relation_path:
        return [entity_name]

    try:
        # 動的にCypherクエリを構築（ラベルに依存しない）
        cypher_parts = [f"MATCH (n0 {{name: $entity_name}})"]

        for i, rel in enumerate(relation_path):
            # リレーションタイプをエスケープ
            cypher_parts.append(f"MATCH (n{i})-[r{i}:`{rel}`]-(n{i+1})")

        last_idx = len(relation_path)
        cypher_parts.append(f"RETURN DISTINCT n{last_idx}.name AS name LIMIT {max_results}")

        query = "\n".join(cypher_parts)
        result = finder.graph.run(query, entity_name=entity_name).data()
        return [r["name"] for r in result if r.get("name")]
    except Exception as e:
        print(f"  [Error] Neo4j query failed: {e}")
        return []


def search_subgraphs_with_neo4j_validation(
    schema: SchemaGraph,
    parsed: ParsedTypeKoPL,
    finder: GraphPathFinder,
    entity_name: str,
    *,
    max_candidates: int = 20,
    max_results_per_path: int = 5,
) -> List[dict]:
    """
    スキーマからサブグラフ候補を取得し、Neo4jで検証する。

    Returns:
        検証済みサブグラフのリスト
        [{"relation_path": [...], "type_path": [...], "reachable_entities": [...]}]
    """
    candidates = search_subgraph_candidates(schema, parsed, max_candidates=max_candidates)

    validated_results = []
    for cand in candidates:
        entities = validate_subgraph_in_neo4j_flexible(
            finder,
            entity_name,
            cand.relation_path,
            max_results=max_results_per_path,
        )
        validated_results.append({
            "relation_path": cand.relation_path,
            "type_path": cand.type_path,
            "query_structure": cand.query_structure,
            "reachable_entities": entities,
            "is_valid": len(entities) > 0,
        })

    return validated_results


# ────────────────────────────────────────────────────────────────
#  E2Eパイプライン
# ────────────────────────────────────────────────────────────────
def run_type_kopl_to_subgraph_pipeline(
    question: str,
    entity_name: str,
    *,
    transitions: TransitionMap,
    finder: Optional[GraphPathFinder] = None,
    fewshot: str = FEW_SHOT_PrimeKGQA,
    model: str = "gpt-4.1-mini",
    max_candidates: int = 20,
    skip_llm: bool = False,
    kopl_override: Optional[str] = None,
) -> dict:
    """
    Type KoPL生成からサブグラフ抽出までのE2Eパイプライン。

    Args:
        question: 自然言語の質問
        entity_name: 開始エンティティ名
        transitions: TransitionMap
        finder: GraphPathFinder (Neo4j検証用、Noneの場合はスキップ)
        fewshot: Few-shot例
        model: LLMモデル
        max_candidates: 最大候補数
        skip_llm: LLM生成をスキップ（kopl_overrideを使用）
        kopl_override: LLM生成をスキップする場合のType KoPL

    Returns:
        パイプライン結果
    """
    result = {
        "question": question,
        "entity_name": entity_name,
        "type_kopl": None,
        "parsed": None,
        "subgraph_candidates": [],
        "validated_subgraphs": [],
    }

    # Step 1: Type KoPL生成
    print("\n" + "=" * 60)
    print("Step 1: Type KoPL Generation")
    print("=" * 60)
    print(f"Question: {question}")

    if skip_llm and kopl_override:
        type_kopl = kopl_override
        print("[Skipped LLM, using override]")
    else:
        print("Generating Type KoPL via LLM...")
        type_kopl = generate_type_kopl(
            question=question,
            fewshot=fewshot,
            model=model,
        )

    result["type_kopl"] = type_kopl
    print(f"\nGenerated Type KoPL:\n{type_kopl}")

    # Step 2: Type KoPLパース
    print("\n" + "=" * 60)
    print("Step 2: Parse Type KoPL")
    print("=" * 60)

    parsed = parse_type_kopl(type_kopl)
    result["parsed"] = {
        "anchors": [
            {"var": a.var_name, "type": a.anchor_type, "entity": a.entity_name}
            for a in parsed.anchors
        ],
        "transitions": [
            {
                "var": t.var_name,
                "input": t.input_var,
                "anchor_type": t.anchor_type,
                "target_type": t.target_type,
                "relation": t.relation_name,
            }
            for t in parsed.transitions
        ],
        "and_ops": [
            {"var": a.var_name, "exp1": a.exp1, "exp2": a.exp2}
            for a in parsed.and_ops
        ],
    }

    print(f"Anchors: {len(parsed.anchors)}")
    for a in parsed.anchors:
        print(f"  - {a.var_name}: {a.anchor_type} ({a.entity_name})")

    print(f"Transitions: {len(parsed.transitions)}")
    for t in parsed.transitions:
        print(f"  - {t.var_name}: {t.anchor_type} --[{t.relation_name}]--> {t.target_type}")

    if parsed.and_ops:
        print(f"And operations: {len(parsed.and_ops)}")
        for a in parsed.and_ops:
            print(f"  - {a.var_name}: And({a.exp1}, {a.exp2})")

    # Step 3: スキーマグラフ探索
    print("\n" + "=" * 60)
    print("Step 3: Schema Graph Search")
    print("=" * 60)

    schema = SchemaGraph(transitions=transitions)
    candidates = search_subgraph_candidates(schema, parsed, max_candidates=max_candidates)

    result["subgraph_candidates"] = [
        {
            "relation_path": c.relation_path,
            "type_path": c.type_path,
            "query_structure": c.query_structure,
        }
        for c in candidates
    ]

    print(f"Found {len(candidates)} subgraph candidates:")
    for i, c in enumerate(candidates[:10]):
        print(f"  [{i+1}] {c.relation_path}")
        print(f"      types: {c.type_path}")

    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more")

    # Step 4: Neo4j検証（オプション）
    if finder:
        print("\n" + "=" * 60)
        print("Step 4: Neo4j Validation")
        print("=" * 60)
        print(f"Entity: {entity_name}")

        validated = search_subgraphs_with_neo4j_validation(
            schema,
            parsed,
            finder,
            entity_name,
            max_candidates=max_candidates,
        )

        result["validated_subgraphs"] = validated

        valid_count = sum(1 for v in validated if v["is_valid"])
        print(f"\nValidated {valid_count}/{len(validated)} paths:")

        for i, v in enumerate(validated[:10]):
            status = "OK" if v["is_valid"] else "NO MATCH"
            print(f"  [{i+1}] [{status}] {v['relation_path']}")
            if v["reachable_entities"]:
                print(f"      -> {v['reachable_entities'][:3]}{'...' if len(v['reachable_entities']) > 3 else ''}")
    else:
        print("\n[Skipped Neo4j validation - no finder provided]")

    return result


# ────────────────────────────────────────────────────────────────
#  サンプル質問
# ────────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    {
        "question": "osteogenesis imperfectaに関連する遺伝子は何ですか？",
        "entity_name": "osteogenesis imperfecta",
        "fewshot": FEW_SHOT_PrimeKGQA,
    },
    {
        "question": "IFITM5に関連する疾患は何ですか？",
        "entity_name": "IFITM5",
        "fewshot": FEW_SHOT_PrimeKGQA,
    },
    {
        "question": "osteogenesis imperfectaに関連する遺伝子が関与する生物学的プロセスは何ですか？",
        "entity_name": "osteogenesis imperfecta",
        "fewshot": FEW_SHOT_PrimeKGQA,
    },
]


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Type KoPL生成からサブグラフ抽出までのE2Eテスト"
    )
    p.add_argument(
        "--question",
        type=str,
        help="質問文",
    )
    p.add_argument(
        "--entity",
        type=str,
        help="開始エンティティ名",
    )
    p.add_argument(
        "--use-sample",
        action="store_true",
        help="サンプル質問を使用",
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="サンプル質問のインデックス (default: 0)",
    )
    p.add_argument(
        "--nodes-csv",
        type=Path,
        default=Path("data/kg/nodes.csv"),
        help="KG nodes CSV",
    )
    p.add_argument(
        "--rels-csv",
        type=Path,
        default=Path("data/kg/relationships.csv"),
        help="KG relationships CSV",
    )
    p.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="LLMモデル (default: gpt-4.1-mini)",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="最大候補数 (default: 20)",
    )
    p.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Neo4j検証をスキップ",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="LLM生成をスキップ（テスト用Type KoPLを使用）",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="結果をJSONファイルに出力",
    )

    args = p.parse_args()

    # 質問とエンティティの決定
    if args.use_sample:
        sample = SAMPLE_QUESTIONS[args.sample_index % len(SAMPLE_QUESTIONS)]
        question = sample["question"]
        entity_name = sample["entity_name"]
        fewshot = sample["fewshot"]
    elif args.question:
        question = args.question
        entity_name = args.entity or (question.split("[")[1].split("]")[0] if "[" in question else "")
        fewshot = FEW_SHOT_PrimeKGQA
    else:
        print("Error: --question or --use-sample required")
        return

    if not entity_name:
        print("Error: --entity required (or use [entity] format in question)")
        return

    # TransitionMap構築
    print("Building TransitionMap from KG CSVs...")
    if not args.nodes_csv.exists() or not args.rels_csv.exists():
        print(f"Error: KG CSVs not found")
        print(f"  nodes: {args.nodes_csv}")
        print(f"  rels: {args.rels_csv}")
        return

    transitions = build_transition_map_from_kg(
        nodes_csv=args.nodes_csv,
        rels_csv=args.rels_csv,
    )
    print(f"Transition rules: {len(transitions.mapping)}")

    # GraphPathFinder（Neo4j）
    finder = None
    if not args.skip_neo4j:
        try:
            print("Connecting to Neo4j...")
            finder = GraphPathFinder()
            print("Neo4j connected.")
        except Exception as e:
            print(f"Warning: Neo4j connection failed: {e}")
            print("Continuing without Neo4j validation...")

    # テスト用Type KoPL（--skip-llm用）
    test_kopl = None
    if args.skip_llm:
        test_kopl = f"""
exp1 = Findanchor(anchor_type='Disease', entity_name='{entity_name}')
genes = FindTypeRelate(exp1, anchor_type="Disease", target_type="Gene", relation_name="associated_with", delta=1)
final = Stop(genes)
"""

    # パイプライン実行
    result = run_type_kopl_to_subgraph_pipeline(
        question=question,
        entity_name=entity_name,
        transitions=transitions,
        finder=finder,
        fewshot=fewshot,
        model=args.model,
        max_candidates=args.max_candidates,
        skip_llm=args.skip_llm,
        kopl_override=test_kopl,
    )

    # 結果出力
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResult saved to: {args.output}")

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
