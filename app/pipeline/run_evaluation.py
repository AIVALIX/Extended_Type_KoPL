"""
統一評価パイプライン

データセット入力から精度評価までを一括で行う

使用方法:
  # 単一パイプライン
  python pipeline/run_evaluation.py --pipeline kgt --num-samples 20
  python pipeline/run_evaluation.py --pipeline safe --dataset one_hop two_hop

  # 複数パイプライン（比較）
  python pipeline/run_evaluation.py --pipeline extended_type_kopl safe kgt --num-samples 20

  # 全パイプライン
  python pipeline/run_evaluation.py --all --num-samples 20

対応パイプライン:
  - extended_type_kopl: Extended Type-KoPL Pipeline
  - safe: SAFE Pipeline（one_hop, two_hopのみ対応）
  - kgt: KGT Pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from pipeline.eval_metrics import (
    PipelineOutput,
    EvalResult,
    evaluate_outputs,
    aggregate_metrics,
    save_pipeline_outputs,
)


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
        "extra_entity_key": "anchor_b_name",
    },
    "three_intersection": {
        "path": "result/dataset_v2/three_intersection.jsonl",
        "entity_key": "anchor_a_name",
        "gold_relations_keys": ["anchor_a_rel", "anchor_b_rel", "anchor_c_rel"],
        "gold_answers_key": "answer_nodes",
        "extra_entity_keys": ["anchor_b_name", "anchor_c_name"],
    },
}

# パイプライン設定
PIPELINE_CONFIGS = {
    "extended_type_kopl": {
        "name": "Extended Type-KoPL",
        "datasets": ["one_hop", "two_hop", "two_intersection", "three_intersection"],
    },
    "safe": {
        "name": "SAFE",
        "datasets": ["one_hop", "two_hop"],  # intersectionは非対応
    },
    "kgt": {
        "name": "KGT",
        "datasets": ["one_hop", "two_hop", "two_intersection", "three_intersection"],
    },
}

# データセット表示名
DATASET_NAMES = {
    "one_hop": "1-hop",
    "two_hop": "2-hop",
    "two_intersection": "2-intersection",
    "three_intersection": "3-intersection",
}


class PipelineRunner:
    """パイプライン実行クラス"""

    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """パイプラインを初期化"""
        if self.pipeline_id == "extended_type_kopl":
            from pipeline.extended_type_kopl_pipeline import ExtendedTypeKoPLPipeline
            self.pipeline = ExtendedTypeKoPLPipeline()
        elif self.pipeline_id == "safe":
            from pipeline.safe_pipeline import SAFEPipeline
            self.pipeline = SAFEPipeline()
        elif self.pipeline_id == "kgt":
            from pipeline.kgt_pipeline import KGTPipeline
            self.pipeline = KGTPipeline()
        else:
            raise ValueError(f"Unknown pipeline: {self.pipeline_id}")

    def _extract_relations(self, result) -> Optional[List[str]]:
        """パイプライン結果からリレーションを抽出"""
        if self.pipeline_id == "extended_type_kopl":
            if not result.selected_paths:
                return None
            best_path = result.selected_paths[0]
            return list(best_path.relations) if best_path.relations else None

        elif self.pipeline_id == "safe":
            if not result.best_query_graph:
                return None
            relations = []
            for schema_edge, _ in result.best_query_graph.edges:
                relations.append(schema_edge.relation)
            return relations if relations else None

        elif self.pipeline_id == "kgt":
            if result.optimal_path:
                return result.optimal_path.relations
            return None

        return None

    def _extract_entities(self, result) -> List[str]:
        """パイプライン結果からエンティティを抽出"""
        return result.answer_entities

    def run_sample(
        self,
        idx: int,
        sample: Dict[str, Any],
        dataset_config: Dict[str, Any],
    ) -> PipelineOutput:
        """単一サンプルを実行"""
        question = sample.get("question", "")
        entity_name = sample.get(dataset_config["entity_key"], "")

        # 正解データ
        answers_key = dataset_config.get("gold_answers_key", "answer_nodes")
        gold_answers = [node["name"] for node in sample.get(answers_key, [])]
        gold_relations = [
            sample.get(k, "") for k in dataset_config["gold_relations_keys"]
            if sample.get(k)
        ]

        output = PipelineOutput(
            idx=idx,
            question=question,
            entity_name=entity_name,
            gold_answers=gold_answers,
            gold_relations=gold_relations,
            predicted_relations=None,
            predicted_entities=[],
        )

        try:
            start = time.time()
            result = self.pipeline.run(question=question, entity_name=entity_name)
            output.latency_ms = (time.time() - start) * 1000

            output.predicted_relations = self._extract_relations(result)
            output.predicted_entities = self._extract_entities(result)

        except Exception as e:
            output.error = str(e)

        return output


def load_dataset(dataset_name: str, num_samples: int, random_sample: bool = False) -> List[Dict[str, Any]]:
    """データセットを読み込み

    Args:
        dataset_name: データセット名
        num_samples: 取得するサンプル数
        random_sample: Trueの場合、ランダムにサンプリング
    """
    import random

    config = DATASETS_V2[dataset_name]
    data_path = Path(config["path"])

    if not data_path.exists():
        return []

    # 全サンプルを読み込み
    all_samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))

    # サンプル数が指定数より少ない場合はそのまま返す
    if len(all_samples) <= num_samples:
        return all_samples

    # ランダムサンプリングまたは先頭から取得
    if random_sample:
        return random.sample(all_samples, num_samples)
    else:
        return all_samples[:num_samples]


def run_pipeline_evaluation(
    pipeline_id: str,
    datasets: List[str],
    num_samples: int,
    output_dir: Optional[Path] = None,
    random_sample: bool = False,
    num_workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    パイプライン評価を実行

    Args:
        pipeline_id: パイプラインID
        datasets: 評価するデータセットのリスト
        num_samples: 各データセットから取得するサンプル数
        output_dir: 結果出力ディレクトリ
        random_sample: Trueの場合、ランダムにサンプリング
        num_workers: 並列ワーカー数（デフォルト: 1）

    Returns:
        データセット名をキーとした結果辞書
    """
    config = PIPELINE_CONFIGS[pipeline_id]
    pipeline_name = config["name"]
    available_datasets = config["datasets"]

    # 対応データセットのみに絞る
    datasets = [d for d in datasets if d in available_datasets]

    if not datasets:
        print(f"  [SKIP] No compatible datasets for {pipeline_name}")
        return {}

    # パイプライン初期化（ワーカー数分）
    if num_workers > 1:
        # 各ワーカー用にパイプラインを初期化
        import threading
        _thread_local = threading.local()

        def get_runner():
            if not hasattr(_thread_local, 'runner'):
                _thread_local.runner = PipelineRunner(pipeline_id)
            return _thread_local.runner
    else:
        runner = PipelineRunner(pipeline_id)
        get_runner = lambda: runner

    results = {}

    for dataset_name in datasets:
        print(f"\n  [{dataset_name}]")

        # データ読み込み（ランダムサンプリング対応）
        samples = load_dataset(dataset_name, num_samples, random_sample)
        if not samples:
            print(f"    [SKIP] No data found")
            continue

        print(f"    Loaded {len(samples)} samples (workers: {num_workers})")
        dataset_config = DATASETS_V2[dataset_name]

        # パイプライン実行
        outputs: List[PipelineOutput] = [None] * len(samples)

        if num_workers > 1:
            # 並列実行
            def process_sample(args):
                idx, sample = args
                local_runner = get_runner()
                return idx, local_runner.run_sample(idx, sample, dataset_config)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_sample, (i, sample)): i
                    for i, sample in enumerate(samples)
                }

                with tqdm(total=len(samples), desc=f"    Running", leave=False) as pbar:
                    for future in as_completed(futures):
                        idx, output = future.result()
                        outputs[idx] = output
                        pbar.update(1)
        else:
            # シーケンシャル実行
            for i, sample in enumerate(tqdm(samples, desc=f"    Running", leave=False)):
                output = get_runner().run_sample(i, sample, dataset_config)
                outputs[i] = output

        # 結果保存
        if output_dir:
            save_pipeline_outputs(outputs, output_dir, pipeline_id, dataset_name)

        # 精度評価
        eval_results = evaluate_outputs(outputs)
        result_dicts = [
            {
                "accuracy": r.accuracy,
                "recall": r.recall,
                "precision": r.precision,
                "f1": r.f1,
                "path_match": r.path_match,
                "error": r.error,
                "latency_ms": r.latency_ms,
            }
            for r in eval_results
        ]
        metrics = aggregate_metrics(result_dicts, len(result_dicts))

        results[dataset_name] = {
            "metrics": metrics,
            "outputs": outputs,
            "eval_results": eval_results,
        }

        # 結果表示
        print(f"    Accuracy:  {metrics['accuracy']:.1f}%")
        print(f"    Recall:    {metrics['recall']:.1f}%")
        print(f"    Precision: {metrics['precision']:.1f}%")
        print(f"    F1:        {metrics['f1']:.1f}%")
        print(f"    PathAcc:   {metrics.get('path_accuracy', 0):.1f}%")
        print(f"    Errors:    {metrics['errors']}/{metrics['total']}")
        print(f"    Latency:   {metrics['avg_latency_ms']:.0f}ms")

    return results


def print_comparison_table(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """比較テーブルを表示"""

    datasets = ["one_hop", "two_hop", "two_intersection", "three_intersection"]

    print(f"\n{'='*100}")
    print("COMPARISON SUMMARY")
    print(f"{'='*100}")

    # データセットごとの比較
    for dataset in datasets:
        pipelines_with_data = [
            (pid, pdata)
            for pid, pdata in all_results.items()
            if dataset in pdata
        ]

        if not pipelines_with_data:
            continue

        dataset_display = DATASET_NAMES.get(dataset, dataset)
        print(f"\n--- {dataset_display} ---")
        print(f"{'Pipeline':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'PathAcc':>10} {'Latency':>10}")
        print("-" * 95)

        for pipeline_id, pipeline_data in pipelines_with_data:
            m = pipeline_data[dataset]["metrics"]
            pipeline_name = PIPELINE_CONFIGS[pipeline_id]["name"]
            print(
                f"{pipeline_name:<25} "
                f"{m['accuracy']:>9.1f}% "
                f"{m['recall']:>9.1f}% "
                f"{m['precision']:>9.1f}% "
                f"{m['f1']:>9.1f}% "
                f"{m.get('path_accuracy', 0):>9.1f}% "
                f"{m['avg_latency_ms']:>8.0f}ms"
            )

    # パイプラインごとの平均
    print(f"\n{'='*100}")
    print("PIPELINE AVERAGES (across all evaluated datasets)")
    print(f"{'='*100}")
    print(f"{'Pipeline':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'PathAcc':>10}")
    print("-" * 85)

    for pipeline_id, pipeline_data in all_results.items():
        if not pipeline_data:
            continue

        n = len(pipeline_data)
        avg_acc = sum(d["metrics"]["accuracy"] for d in pipeline_data.values()) / n
        avg_recall = sum(d["metrics"]["recall"] for d in pipeline_data.values()) / n
        avg_prec = sum(d["metrics"]["precision"] for d in pipeline_data.values()) / n
        avg_f1 = sum(d["metrics"]["f1"] for d in pipeline_data.values()) / n
        avg_path = sum(d["metrics"].get("path_accuracy", 0) for d in pipeline_data.values()) / n

        pipeline_name = PIPELINE_CONFIGS[pipeline_id]["name"]
        print(
            f"{pipeline_name:<25} "
            f"{avg_acc:>9.1f}% "
            f"{avg_recall:>9.1f}% "
            f"{avg_prec:>9.1f}% "
            f"{avg_f1:>9.1f}% "
            f"{avg_path:>9.1f}%"
        )


def main():
    p = argparse.ArgumentParser(
        description="Run evaluation pipeline from dataset to metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pipeline
  python pipeline/run_evaluation.py --pipeline kgt --num-samples 20

  # Multiple pipelines (comparison)
  python pipeline/run_evaluation.py --pipeline extended_type_kopl safe kgt

  # All pipelines
  python pipeline/run_evaluation.py --all --num-samples 20

  # Specific datasets
  python pipeline/run_evaluation.py --pipeline kgt --dataset one_hop two_hop
        """
    )
    p.add_argument("--pipeline", type=str, nargs="+",
                   choices=list(PIPELINE_CONFIGS.keys()),
                   help="Pipeline(s) to evaluate")
    p.add_argument("--all", action="store_true",
                   help="Evaluate all pipelines")
    p.add_argument("--dataset", type=str, nargs="+",
                   choices=list(DATASETS_V2.keys()),
                   help="Specific dataset(s) to evaluate (default: all compatible)")
    p.add_argument("--num-samples", type=int, default=20,
                   help="Number of samples per dataset (default: 20)")
    p.add_argument("--random", action="store_true",
                   help="Randomly sample from dataset instead of taking first N")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers (default: 1)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory for pipeline outputs (optional)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON file for metrics summary")
    args = p.parse_args()

    # パイプライン選択
    if args.all:
        pipelines = list(PIPELINE_CONFIGS.keys())
    elif args.pipeline:
        pipelines = args.pipeline
    else:
        p.print_help()
        print("\nError: Please specify --pipeline or --all")
        return

    # データセット選択
    datasets = args.dataset if args.dataset else list(DATASETS_V2.keys())

    print(f"{'='*60}")
    print("EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"Pipelines: {', '.join(pipelines)}")
    print(f"Datasets:  {', '.join(datasets)}")
    print(f"Samples:   {args.num_samples}" + (" (random)" if args.random else ""))
    print(f"Workers:   {args.workers}")
    if args.output_dir:
        print(f"Output:    {args.output_dir}")

    all_results = {}

    for pipeline_id in pipelines:
        pipeline_name = PIPELINE_CONFIGS[pipeline_id]["name"]
        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline_name}")
        print(f"{'='*60}")

        results = run_pipeline_evaluation(
            pipeline_id,
            datasets,
            args.num_samples,
            args.output_dir,
            args.random,
            args.workers,
        )
        all_results[pipeline_id] = results

    # 複数パイプラインの場合は比較テーブルを表示
    if len(pipelines) > 1:
        print_comparison_table(all_results)

    # メトリクスをJSON出力
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            pipeline_id: {
                dataset_name: data["metrics"]
                for dataset_name, data in pipeline_data.items()
            }
            for pipeline_id, pipeline_data in all_results.items()
        }
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
