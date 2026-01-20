"""
複数データセットでType KoPLパイプラインを一括評価

使用方法:
  python pipeline/evaluate_all_datasets.py --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from dataset_construction.add_kopl import (
    TransitionMap,
    TransitionMapMulti,
    build_transition_map_from_kg,
    build_transition_map_multi_from_neo4j,
)
from pipeline.type_kopl_pipeline import (
    TypeKoPLPipeline,
    TypeKoPLPipelineConfig,
)


# データセット設定
def get_datasets(lang: str = "ja") -> Dict[str, Dict[str, Any]]:
    suffix = "_en" if lang == "en" else ""
    return {
        "two_hop_chain": {
            "path": f"result/dataset_construction_v2/two_hop_chain_train_with_questions{suffix}.jsonl",
            "entity_key": "anchor_name",
            "gold_relations_keys": ["rel1", "rel2"],
            "is_intersection": False,
        },
        "two_anchor_intersection": {
            "path": f"result/dataset_construction_v2/two_anchor_intersection_train_with_questions{suffix}.jsonl",
            "entity_key": "anchorA_name",
            "gold_relations_keys": ["anchorA_edge_type", "anchorB_edge_type"],
            "is_intersection": True,
            "anchor_keys": ["anchorA_name", "anchorB_name"],
            "relation_keys": ["anchorA_edge_type", "anchorB_edge_type"],
        },
        "three_anchor_intersection": {
            "path": f"result/dataset_construction_v2/three_anchor_intersection_train_with_questions{suffix}.jsonl",
            "entity_key": "anchorA_name",
            "gold_relations_keys": ["anchorA_edge_type", "anchorB_edge_type", "anchorC_edge_type"],
            "is_intersection": True,
            "anchor_keys": ["anchorA_name", "anchorB_name", "anchorC_name"],
            "relation_keys": ["anchorA_edge_type", "anchorB_edge_type", "anchorC_edge_type"],
        },
    }

DATASETS = get_datasets("ja")  # デフォルト


@dataclass
class EvalResult:
    idx: int
    question: str
    entity_name: str
    gold_answers: List[str]
    gold_relations: List[str]
    predicted_path: Optional[List[str]]
    predicted_entities: List[str]
    hits_at_1: bool = False
    hits_at_5: bool = False
    hits_at_10: bool = False
    reciprocal_rank: float = 0.0
    path_match: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0


def evaluate_sample(
    idx: int,
    sample: Dict[str, Any],
    pipeline: TypeKoPLPipeline,
    dataset_config: Dict[str, Any],
) -> EvalResult:
    """単一サンプルを評価"""
    question = sample.get("question", "")
    entity_name = sample.get(dataset_config["entity_key"], "")
    is_intersection = dataset_config.get("is_intersection", False)

    # 正解データ
    gold_answers = [
        node["name"] for node in sample.get("answer_nodes_sample", [])
    ]
    gold_relations = [
        sample.get(k, "") for k in dataset_config["gold_relations_keys"]
        if sample.get(k)
    ]
    gold_set = set(gold_answers)

    result = EvalResult(
        idx=idx,
        question=question,
        entity_name=entity_name,
        gold_answers=gold_answers,
        gold_relations=gold_relations,
        predicted_path=None,
        predicted_entities=[],
    )

    try:
        start = time.time()

        if is_intersection:
            # 交差処理: 複数アンカーを取得
            anchor_keys = dataset_config.get("anchor_keys", [])
            relation_keys = dataset_config.get("relation_keys", [])

            anchors = {}
            relations = {}
            for i, key in enumerate(anchor_keys):
                anchor_id = f"anchor{chr(ord('A') + i)}"
                entity = sample.get(key, "")
                if entity:
                    anchors[anchor_id] = entity
                if i < len(relation_keys):
                    rel = sample.get(relation_keys[i], "")
                    if rel:
                        relations[anchor_id] = rel

            pr = pipeline.run_intersection(
                question=question,
                anchors=anchors,
                relations=relations,
            )

            # 交差処理の場合、全アンカーのパスを結合してpath_matchを評価
            all_paths = []
            if pr.intersection_paths:
                for path in pr.intersection_paths.values():
                    if path:
                        all_paths.extend(path)
            result.predicted_path = all_paths if all_paths else None
        else:
            # 通常処理: 単一アンカー
            pr = pipeline.run(question=question, entity_name=entity_name)
            result.predicted_path = pr.final_path

        result.latency_ms = (time.time() - start) * 1000
        result.predicted_entities = pr.reachable_entities

        if result.predicted_entities:
            result.hits_at_1 = bool(gold_set & set(result.predicted_entities[:1]))
            result.hits_at_5 = bool(gold_set & set(result.predicted_entities[:5]))
            result.hits_at_10 = bool(gold_set & set(result.predicted_entities[:10]))
            for i, pred in enumerate(result.predicted_entities[:100]):
                if pred in gold_set:
                    result.reciprocal_rank = 1.0 / (i + 1)
                    break

        if result.predicted_path:
            pred_set = set(result.predicted_path)
            result.path_match = any(rel in pred_set for rel in gold_relations)

    except Exception as e:
        result.error = str(e)

    return result


def run_dataset_evaluation(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    transitions: TransitionMap,
    config: TypeKoPLPipelineConfig,
    num_workers: int,
    num_samples: int,
    transitions_multi: Optional[TransitionMapMulti] = None,
) -> Dict[str, Any]:
    """単一データセットの評価"""
    path = Path(dataset_config["path"])
    if not path.exists():
        return {"error": f"File not found: {path}"}

    # データ読み込み
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    results: List[EvalResult] = []

    # 並列評価
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, sample in enumerate(samples):
            pipeline = TypeKoPLPipeline(
                transitions=transitions,
                config=config,
                transitions_multi=transitions_multi,
            )
            future = executor.submit(
                evaluate_sample, i, sample, pipeline, dataset_config
            )
            futures.append(future)

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {dataset_name}",
            leave=False,
        ):
            results.append(future.result())

    results.sort(key=lambda x: x.idx)

    # メトリクス集計
    total = len(results)
    success = sum(1 for r in results if not r.error)
    errors = total - success

    if success > 0:
        hits_1 = sum(1 for r in results if r.hits_at_1) / success
        hits_5 = sum(1 for r in results if r.hits_at_5) / success
        hits_10 = sum(1 for r in results if r.hits_at_10) / success
        mrr = sum(r.reciprocal_rank for r in results) / success
        path_acc = sum(1 for r in results if r.path_match) / success
        avg_latency = sum(r.latency_ms for r in results if not r.error) / success
    else:
        hits_1 = hits_5 = hits_10 = mrr = path_acc = avg_latency = 0.0

    return {
        "dataset": dataset_name,
        "total": total,
        "success": success,
        "errors": errors,
        "hits_at_1": hits_1,
        "hits_at_5": hits_5,
        "hits_at_10": hits_10,
        "mrr": mrr,
        "path_accuracy": path_acc,
        "avg_latency_ms": avg_latency,
        "results": [
            {
                "idx": r.idx,
                "question": r.question[:100],
                "gold_relations": r.gold_relations,
                "predicted_path": r.predicted_path,
                "path_match": r.path_match,
                "gold_answers": r.gold_answers[:3],
                "predicted_entities": r.predicted_entities[:5],
                "hits_at_10": r.hits_at_10,
                "error": r.error,
            }
            for r in results
        ],
    }


def main():
    p = argparse.ArgumentParser(description="複数データセット一括評価")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--nodes-csv", type=Path, default=Path("data/kg/nodes.csv"))
    p.add_argument("--rels-csv", type=Path, default=Path("data/kg/relationships.csv"))
    p.add_argument("--no-vector-prune", action="store_true")
    p.add_argument("--no-llm-rerank", action="store_true")
    p.add_argument("--use-schema-prompt", action="store_true", help="Add schema types/relations to prompt")
    p.add_argument("--lang", type=str, choices=["ja", "en"], default="ja", help="Question language")
    p.add_argument("--output", type=Path, default=Path("result/type_kopl_all_eval.json"))
    p.add_argument("--dataset", type=str, nargs="+",
                   choices=["two_hop_chain", "two_anchor_intersection", "three_anchor_intersection"],
                   help="Specific dataset(s) to evaluate (default: all)")
    args = p.parse_args()

    print("Building TransitionMap...")
    transitions = build_transition_map_from_kg(
        nodes_csv=args.nodes_csv, rels_csv=args.rels_csv
    )

    print("Building TransitionMapMulti from Neo4j...")
    transitions_multi = build_transition_map_multi_from_neo4j()

    config = TypeKoPLPipelineConfig(
        use_vector_prune=not args.no_vector_prune,
        use_llm_reranker=not args.no_llm_rerank,
        use_schema_in_prompt=args.use_schema_prompt,
    )

    # 言語に応じたデータセット取得
    datasets = get_datasets(args.lang)

    # データセット指定がある場合はフィルタ
    if args.dataset:
        datasets = {k: v for k, v in datasets.items() if k in args.dataset}

    all_results = {}
    start_time = time.time()

    print(f"\nEvaluating {len(datasets)} datasets (lang={args.lang})...")
    for name, ds_config in datasets.items():
        print(f"\n[{name}]")
        result = run_dataset_evaluation(
            name, ds_config, transitions, config,
            args.num_workers, args.num_samples, transitions_multi
        )
        all_results[name] = result

    total_time = time.time() - start_time

    # 結果表示
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Dataset':<30} {'Hits@1':>8} {'Hits@5':>8} {'Hits@10':>8} {'MRR':>8} {'PathAcc':>8}")
    print("-" * 80)

    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<30} ERROR: {res['error']}")
        else:
            print(
                f"{name:<30} "
                f"{res['hits_at_1']:>8.2%} "
                f"{res['hits_at_5']:>8.2%} "
                f"{res['hits_at_10']:>8.2%} "
                f"{res['mrr']:>8.4f} "
                f"{res['path_accuracy']:>8.2%}"
            )

    print("-" * 80)
    print(f"Total time: {total_time:.1f}s")

    # 保存
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
