"""
SAFE Pipeline 評価スクリプト

使用方法:
  python pipeline/evaluate_safe_pipeline.py --num-samples 20
  python pipeline/evaluate_safe_pipeline.py --dataset one_hop two_hop --num-samples 50
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from pipeline.safe_pipeline import SAFEPipeline, SAFEResult


# データセット設定
DATASETS_V2 = {
    "one_hop": {
        "path": "result/dataset_v2/one_hop.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answer_nodes",
    },
    "two_hop": {
        "path": "result/dataset_v2/two_hop.jsonl",
        "entity_key": "anchor_name",
        "gold_relations_keys": ["rel1", "rel2"],
        "gold_answers_key": "answer_nodes",
    },
    "two_intersection": {
        "path": "result/dataset_v2/two_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel"],
        "gold_answers_key": "answer_nodes",
    },
    "three_intersection": {
        "path": "result/dataset_v2/three_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel", "anchor_c_rel"],
        "gold_answers_key": "answer_nodes",
    },
}


@dataclass
class EvalResult:
    idx: int
    question: str
    entity_name: str
    gold_answers: List[str]
    gold_relations: List[str]
    predicted_entities: List[str]
    hits_at_1: bool = False
    hits_at_5: bool = False
    hits_at_10: bool = False
    reciprocal_rank: float = 0.0
    error: Optional[str] = None
    latency_ms: float = 0.0


def calculate_reciprocal_rank(predicted: List[str], gold: Set[str], max_k: int = 100) -> float:
    for i, pred in enumerate(predicted[:max_k]):
        if pred in gold:
            return 1.0 / (i + 1)
    return 0.0


def check_hits_at_k(predicted: List[str], gold: Set[str], k: int) -> bool:
    return bool(gold & set(predicted[:k]))


def evaluate_sample(
    idx: int,
    sample: Dict[str, Any],
    pipeline: SAFEPipeline,
    dataset_config: Dict[str, Any],
) -> EvalResult:
    """単一サンプルを評価"""
    question = sample.get("question", "")
    entity_name = sample.get(dataset_config["entity_key"], "")

    # 正解データ
    answers_key = dataset_config.get("gold_answers_key", "answer_nodes")
    gold_answers = [node["name"] for node in sample.get(answers_key, [])]
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
        predicted_entities=[],
    )

    try:
        start = time.time()
        safe_result = pipeline.run(question=question, entity_name=entity_name)
        result.latency_ms = (time.time() - start) * 1000

        # 予測エンティティ
        result.predicted_entities = safe_result.answer_entities

        # メトリクス計算
        if result.predicted_entities:
            result.hits_at_1 = check_hits_at_k(result.predicted_entities, gold_set, 1)
            result.hits_at_5 = check_hits_at_k(result.predicted_entities, gold_set, 5)
            result.hits_at_10 = check_hits_at_k(result.predicted_entities, gold_set, 10)
            result.reciprocal_rank = calculate_reciprocal_rank(result.predicted_entities, gold_set)

    except Exception as e:
        result.error = str(e)

    return result


def run_evaluation(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    *,
    num_samples: int = 50,
) -> Tuple[List[EvalResult], Dict[str, float]]:
    """データセットを評価"""

    data_path = Path(dataset_config["path"])
    if not data_path.exists():
        print(f"  [SKIP] File not found: {data_path}")
        return [], {}

    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"  Loaded {len(samples)} samples")

    # パイプライン作成
    pipeline = SAFEPipeline()

    results: List[EvalResult] = []

    # 逐次処理
    for i, sample in enumerate(tqdm(samples, desc=f"  {dataset_name}")):
        result = evaluate_sample(i, sample, pipeline, dataset_config)
        results.append(result)

    # メトリクス集計
    total = len(results)
    if total == 0:
        return results, {}

    errors = sum(1 for r in results if r.error)
    success = total - errors

    metrics = {
        "total": total,
        "success": success,
        "errors": errors,
        "hits_at_1": sum(r.hits_at_1 for r in results) / total * 100,
        "hits_at_5": sum(r.hits_at_5 for r in results) / total * 100,
        "hits_at_10": sum(r.hits_at_10 for r in results) / total * 100,
        "mrr": sum(r.reciprocal_rank for r in results) / total * 100,
        "avg_latency_ms": sum(r.latency_ms for r in results) / total,
    }

    return results, metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate SAFE Pipeline")
    p.add_argument("--dataset", type=str, nargs="+",
                   choices=list(DATASETS_V2.keys()),
                   help="Specific dataset(s) to evaluate (default: all)")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON file for detailed results")
    args = p.parse_args()

    datasets_to_eval = args.dataset if args.dataset else list(DATASETS_V2.keys())

    all_metrics = {}
    all_results = {}

    for ds_name in datasets_to_eval:
        if ds_name not in DATASETS_V2:
            print(f"Unknown dataset: {ds_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {ds_name}")
        print(f"{'='*60}")

        ds_config = DATASETS_V2[ds_name]
        results, metrics = run_evaluation(
            ds_name,
            ds_config,
            num_samples=args.num_samples,
        )

        if metrics:
            all_metrics[ds_name] = metrics
            all_results[ds_name] = [
                {
                    "idx": r.idx,
                    "question": r.question,
                    "entity_name": r.entity_name,
                    "gold_answers": r.gold_answers[:5],
                    "gold_relations": r.gold_relations,
                    "predicted_entities": r.predicted_entities[:10],
                    "hits_at_1": r.hits_at_1,
                    "hits_at_10": r.hits_at_10,
                    "error": r.error,
                }
                for r in results
            ]

            print(f"\n  Results for {ds_name}:")
            print(f"    Hits@1:  {metrics['hits_at_1']:.1f}%")
            print(f"    Hits@5:  {metrics['hits_at_5']:.1f}%")
            print(f"    Hits@10: {metrics['hits_at_10']:.1f}%")
            print(f"    MRR:     {metrics['mrr']:.1f}%")
            print(f"    Errors:  {metrics['errors']}/{metrics['total']}")
            print(f"    Latency: {metrics['avg_latency_ms']:.0f}ms")

    # サマリー表示
    print(f"\n{'='*60}")
    print("SUMMARY (SAFE Pipeline)")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'Hits@1':>8} {'Hits@10':>8} {'MRR':>8}")
    print("-" * 60)
    for ds_name, m in all_metrics.items():
        print(f"{ds_name:<25} {m['hits_at_1']:>7.1f}% {m['hits_at_10']:>7.1f}% {m['mrr']:>7.1f}%")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump({
                "metrics": all_metrics,
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
