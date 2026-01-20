"""
Type KoPLパイプラインの精度評価スクリプト

並列処理で高速化
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from dataset_construction.add_kopl import TransitionMap, build_transition_map_from_kg
from pipeline.type_kopl_pipeline import (
    PipelineResult,
    TypeKoPLPipeline,
    TypeKoPLPipelineConfig,
)


# ────────────────────────────────────────────────────────────────
#  評価メトリクス
# ────────────────────────────────────────────────────────────────
@dataclass
class EvaluationResult:
    """単一サンプルの評価結果"""

    idx: int
    question: str
    entity_name: str
    gold_answers: List[str]
    gold_relations: List[str]  # [anchorA_edge_type, anchorB_edge_type]
    predicted_path: Optional[List[str]]
    predicted_entities: List[str]
    hits_at_1: bool = False
    hits_at_5: bool = False
    hits_at_10: bool = False
    reciprocal_rank: float = 0.0
    path_match: bool = False  # 予測パスが正解relationを含むか
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class AggregatedMetrics:
    """集計されたメトリクス"""

    total: int = 0
    success: int = 0
    errors: int = 0
    hits_at_1: float = 0.0
    hits_at_5: float = 0.0
    hits_at_10: float = 0.0
    mrr: float = 0.0
    path_accuracy: float = 0.0
    avg_latency_ms: float = 0.0


def calculate_reciprocal_rank(
    predicted: List[str], gold: Set[str], max_k: int = 100
) -> float:
    """最初に正解が出現した位置の逆数を計算"""
    for i, pred in enumerate(predicted[:max_k]):
        if pred in gold:
            return 1.0 / (i + 1)
    return 0.0


def check_hits_at_k(predicted: List[str], gold: Set[str], k: int) -> bool:
    """上位k件に正解が含まれるか"""
    return bool(gold & set(predicted[:k]))


def check_path_match(predicted_path: Optional[List[str]], gold_relations: List[str]) -> bool:
    """予測パスが正解relationを含むかチェック"""
    if not predicted_path:
        return False
    predicted_set = set(predicted_path)
    # 正解relationのいずれかが予測パスに含まれていればOK
    for rel in gold_relations:
        if rel in predicted_set:
            return True
    return False


# ────────────────────────────────────────────────────────────────
#  評価ワーカー
# ────────────────────────────────────────────────────────────────
def evaluate_single_sample(
    args: Tuple[int, Dict[str, Any], TypeKoPLPipeline]
) -> EvaluationResult:
    """単一サンプルを評価"""
    idx, sample, pipeline = args

    question = sample.get("question", "")
    # two_anchor_intersectionの場合、最初のアンカーをentityとして使用
    entity_name = sample.get("anchorA_name", "")

    # 正解データ
    gold_answers = [
        node["name"] for node in sample.get("answer_nodes_sample", [])
    ]
    gold_relations = [
        sample.get("anchorA_edge_type", ""),
        sample.get("anchorB_edge_type", ""),
    ]
    gold_set = set(gold_answers)

    result = EvaluationResult(
        idx=idx,
        question=question,
        entity_name=entity_name,
        gold_answers=gold_answers,
        gold_relations=gold_relations,
        predicted_path=None,
        predicted_entities=[],
    )

    try:
        start_time = time.time()
        pipeline_result = pipeline.run(question=question, entity_name=entity_name)
        result.latency_ms = (time.time() - start_time) * 1000

        result.predicted_path = pipeline_result.final_path
        result.predicted_entities = pipeline_result.reachable_entities

        # メトリクス計算
        if result.predicted_entities:
            result.hits_at_1 = check_hits_at_k(result.predicted_entities, gold_set, 1)
            result.hits_at_5 = check_hits_at_k(result.predicted_entities, gold_set, 5)
            result.hits_at_10 = check_hits_at_k(result.predicted_entities, gold_set, 10)
            result.reciprocal_rank = calculate_reciprocal_rank(
                result.predicted_entities, gold_set
            )

        result.path_match = check_path_match(result.predicted_path, gold_relations)

    except Exception as e:
        result.error = str(e)

    return result


# ────────────────────────────────────────────────────────────────
#  メイン評価関数
# ────────────────────────────────────────────────────────────────
def run_evaluation(
    data_path: Path,
    transitions: TransitionMap,
    *,
    num_samples: int = 100,
    num_workers: int = 4,
    config: Optional[TypeKoPLPipelineConfig] = None,
) -> Tuple[List[EvaluationResult], AggregatedMetrics]:
    """並列評価を実行"""

    # データ読み込み
    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    # パイプラインのプール（各ワーカーに1つ）
    # 注意: GraphPathFinderはスレッドセーフではない可能性があるため、
    # 各スレッドで新しいパイプラインを作成
    def create_pipeline() -> TypeKoPLPipeline:
        return TypeKoPLPipeline(transitions=transitions, config=config)

    results: List[EvaluationResult] = []

    # 並列実行
    print(f"Running evaluation with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 各サンプルにパイプラインを割り当て
        futures = []
        for i, sample in enumerate(samples):
            pipeline = create_pipeline()
            future = executor.submit(evaluate_single_sample, (i, sample, pipeline))
            futures.append(future)

        # 結果を収集（プログレスバー付き）
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result = future.result()
            results.append(result)

    # 結果をインデックス順にソート
    results.sort(key=lambda x: x.idx)

    # メトリクス集計
    metrics = aggregate_metrics(results)

    return results, metrics


def aggregate_metrics(results: List[EvaluationResult]) -> AggregatedMetrics:
    """結果を集計"""
    metrics = AggregatedMetrics()
    metrics.total = len(results)

    total_latency = 0.0
    hits_1_count = 0
    hits_5_count = 0
    hits_10_count = 0
    mrr_sum = 0.0
    path_match_count = 0
    valid_count = 0

    for r in results:
        if r.error:
            metrics.errors += 1
            continue

        metrics.success += 1
        valid_count += 1
        total_latency += r.latency_ms

        if r.hits_at_1:
            hits_1_count += 1
        if r.hits_at_5:
            hits_5_count += 1
        if r.hits_at_10:
            hits_10_count += 1
        mrr_sum += r.reciprocal_rank
        if r.path_match:
            path_match_count += 1

    if valid_count > 0:
        metrics.hits_at_1 = hits_1_count / valid_count
        metrics.hits_at_5 = hits_5_count / valid_count
        metrics.hits_at_10 = hits_10_count / valid_count
        metrics.mrr = mrr_sum / valid_count
        metrics.path_accuracy = path_match_count / valid_count
        metrics.avg_latency_ms = total_latency / valid_count

    return metrics


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Type KoPLパイプライン精度評価")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("result/dataset_construction_v2/two_anchor_intersection_train_with_questions.jsonl"),
        help="評価データのパス",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="評価サンプル数",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="並列ワーカー数",
    )
    p.add_argument(
        "--nodes-csv",
        type=Path,
        default=Path("data/kg/nodes.csv"),
    )
    p.add_argument(
        "--rels-csv",
        type=Path,
        default=Path("data/kg/relationships.csv"),
    )
    p.add_argument(
        "--no-vector-prune",
        action="store_true",
        help="ベクトル剪定を無効化",
    )
    p.add_argument(
        "--no-llm-rerank",
        action="store_true",
        help="LLM Rerankerを無効化",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="結果をJSONファイルに出力",
    )

    args = p.parse_args()

    # TransitionMap構築
    print("Building TransitionMap...")
    transitions = build_transition_map_from_kg(
        nodes_csv=args.nodes_csv,
        rels_csv=args.rels_csv,
    )

    # パイプライン設定
    config = TypeKoPLPipelineConfig(
        use_vector_prune=not args.no_vector_prune,
        use_llm_reranker=not args.no_llm_rerank,
    )

    # 評価実行
    start_time = time.time()
    results, metrics = run_evaluation(
        data_path=args.data,
        transitions=transitions,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        config=config,
    )
    total_time = time.time() - start_time

    # 結果表示
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {metrics.total}")
    print(f"Success: {metrics.success}")
    print(f"Errors: {metrics.errors}")
    print()
    print(f"Hits@1:  {metrics.hits_at_1:.4f} ({int(metrics.hits_at_1 * metrics.success)}/{metrics.success})")
    print(f"Hits@5:  {metrics.hits_at_5:.4f} ({int(metrics.hits_at_5 * metrics.success)}/{metrics.success})")
    print(f"Hits@10: {metrics.hits_at_10:.4f} ({int(metrics.hits_at_10 * metrics.success)}/{metrics.success})")
    print(f"MRR:     {metrics.mrr:.4f}")
    print(f"Path Accuracy: {metrics.path_accuracy:.4f}")
    print()
    print(f"Avg latency: {metrics.avg_latency_ms:.1f} ms")
    print(f"Total time: {total_time:.1f} s")

    # エラーサンプルの表示
    error_samples = [r for r in results if r.error]
    if error_samples:
        print(f"\nError samples ({len(error_samples)}):")
        for r in error_samples[:5]:
            print(f"  [{r.idx}] {r.error[:100]}")

    # 結果出力
    if args.output:
        output_data = {
            "metrics": {
                "total": metrics.total,
                "success": metrics.success,
                "errors": metrics.errors,
                "hits_at_1": metrics.hits_at_1,
                "hits_at_5": metrics.hits_at_5,
                "hits_at_10": metrics.hits_at_10,
                "mrr": metrics.mrr,
                "path_accuracy": metrics.path_accuracy,
                "avg_latency_ms": metrics.avg_latency_ms,
            },
            "results": [
                {
                    "idx": r.idx,
                    "question": r.question,
                    "entity_name": r.entity_name,
                    "gold_answers": r.gold_answers,
                    "gold_relations": r.gold_relations,
                    "predicted_path": r.predicted_path,
                    "predicted_entities": r.predicted_entities[:10],
                    "hits_at_1": r.hits_at_1,
                    "hits_at_5": r.hits_at_5,
                    "hits_at_10": r.hits_at_10,
                    "reciprocal_rank": r.reciprocal_rank,
                    "path_match": r.path_match,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
