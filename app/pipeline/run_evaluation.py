"""
統一評価パイプライン

データセット入力から精度評価までを一括で行う

使用方法:
  # PrimeKGQA（デフォルト）
  python pipeline/run_evaluation.py --pipeline kgt --num-samples 20
  python pipeline/run_evaluation.py --pipeline safe --dataset one_hop two_hop

  # MetaQA
  python pipeline/run_evaluation.py --kg metaqa --pipeline extended_type_kopl --num-samples 20

  # 複数パイプライン（比較）
  python pipeline/run_evaluation.py --pipeline extended_type_kopl safe kgt --num-samples 20

  # 全パイプライン
  python pipeline/run_evaluation.py --all --num-samples 20

対応パイプライン:
  - extended_type_kopl: Extended Type-KoPL Pipeline
  - safe: SAFE Pipeline（one_hop, two_hopのみ対応）
  - kgt: KGT Pipeline

対応KG:
  - primekgqa: PrimeKGQA（バイオメディカル）
  - metaqa: MetaQA（映画ドメイン）
"""

from __future__ import annotations

import argparse
import json
import time
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from pipeline.common.eval_metrics import (
    PipelineOutput,
    EvalResult,
    evaluate_outputs,
    aggregate_metrics,
    save_pipeline_outputs,
)
from pipeline.common.kg_config import KGConfig, KGType


# データセット設定（PrimeKGQA）
DATASETS_PRIMEKGQA = {
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

# データセット設定（MetaQA）
DATASETS_METAQA = {
    "1hop": {
        "path": "result/metaqa/1hop.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation"],
        "gold_answers_key": "answers",
    },
    "2hop": {
        "path": "result/metaqa/2hop.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation1", "relation2"],
        "gold_answers_key": "answers",
    },
    "3hop": {
        "path": "result/metaqa/3hop.jsonl",
        "entity_key": "entity",
        "gold_relations_keys": ["relation1", "relation2", "relation3"],
        "gold_answers_key": "answers",
    },
}

# KGごとのデータセット設定
DATASETS_BY_KG = {
    "primekgqa": DATASETS_PRIMEKGQA,
    "metaqa": DATASETS_METAQA,
}

# 後方互換性のため
DATASETS_V2 = DATASETS_PRIMEKGQA

# パイプライン設定（PrimeKGQA）
PIPELINE_CONFIGS_PRIMEKGQA = {
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

# パイプライン設定（MetaQA）
PIPELINE_CONFIGS_METAQA = {
    "extended_type_kopl": {
        "name": "Extended Type-KoPL",
        "datasets": ["1hop", "2hop", "3hop"],
    },
    "safe": {
        "name": "SAFE",
        "datasets": ["1hop", "2hop"],  # 3hopは非対応（delta=1まで）
    },
    "kgt": {
        "name": "KGT",
        "datasets": ["1hop", "2hop", "3hop"],
    },
}

# KGごとのパイプライン設定
PIPELINE_CONFIGS_BY_KG = {
    "primekgqa": PIPELINE_CONFIGS_PRIMEKGQA,
    "metaqa": PIPELINE_CONFIGS_METAQA,
}

# 後方互換性のため
PIPELINE_CONFIGS = PIPELINE_CONFIGS_PRIMEKGQA

# データセット表示名
DATASET_NAMES = {
    # PrimeKGQA
    "one_hop": "1-hop",
    "two_hop": "2-hop",
    "two_intersection": "2-intersection",
    "three_intersection": "3-intersection",
    # MetaQA
    "1hop": "1-hop",
    "2hop": "2-hop",
    "3hop": "3-hop",
}


class PipelineRunner:
    """パイプライン実行クラス"""

    def __init__(
        self, pipeline_id: str, pipeline_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.pipeline_id = pipeline_id
        self.pipeline_kwargs = pipeline_kwargs or {}
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """パイプラインを初期化"""
        if self.pipeline_id == "extended_type_kopl":
            from pipeline.extended_type_kopl import ExtendedTypeKoPLPipeline

            self.pipeline = ExtendedTypeKoPLPipeline(**self.pipeline_kwargs)
        elif self.pipeline_id == "safe":
            from pipeline.safe import SAFEPipeline

            self.pipeline = SAFEPipeline(**self.pipeline_kwargs)
        elif self.pipeline_id == "kgt":
            from pipeline.kgt import KGTPipeline

            self.pipeline = KGTPipeline(**self.pipeline_kwargs)
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
            sample.get(k, "")
            for k in dataset_config["gold_relations_keys"]
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
            assert self.pipeline is not None
            result = self.pipeline.run(question=question, entity_name=entity_name)
            output.latency_ms = (time.time() - start) * 1000

            output.predicted_relations = self._extract_relations(result)
            output.predicted_entities = self._extract_entities(result)

        except Exception as e:
            output.error = str(e)

        return output


# グローバル変数（ProcessPoolExecutor用）
_worker_runner: Optional[PipelineRunner] = None


def _init_worker(pipeline_id: str, pipeline_kwargs: Optional[Dict[str, Any]] = None):
    """ワーカープロセス初期化"""
    global _worker_runner
    _worker_runner = PipelineRunner(pipeline_id, pipeline_kwargs)


def _process_sample_worker(
    args: Tuple[int, Dict[str, Any], Dict[str, Any]],
) -> PipelineOutput:
    """ワーカープロセスでサンプルを処理"""
    global _worker_runner
    assert _worker_runner is not None
    idx, sample, dataset_config = args
    return _worker_runner.run_sample(idx, sample, dataset_config)


def load_dataset(
    dataset_name: str,
    num_samples: int,
    random_sample: bool = False,
    datasets_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """データセットを読み込み

    Args:
        dataset_name: データセット名
        num_samples: 取得するサンプル数
        random_sample: Trueの場合、ランダムにサンプリング
        datasets_config: データセット設定（省略時はDATASETS_V2を使用）
    """
    import random

    if datasets_config is None:
        datasets_config = DATASETS_V2
    config = datasets_config[dataset_name]
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
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    kg_type: str = "primekgqa",
    per_sample_timeout_s: float = 300.0,
    watch_interval_s: float = 2.0,
    max_inflight: Optional[int] = None,
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
        pipeline_kwargs: パイプライン初期化パラメータ
        kg_type: Knowledge Graphの種類（primekgqa or metaqa）

    Returns:
        データセット名をキーとした結果辞書
    """
    # KGに応じた設定を取得
    pipeline_configs = PIPELINE_CONFIGS_BY_KG.get(kg_type, PIPELINE_CONFIGS_PRIMEKGQA)
    datasets_config = DATASETS_BY_KG.get(kg_type, DATASETS_PRIMEKGQA)

    config = pipeline_configs[pipeline_id]
    pipeline_name = config["name"]
    available_datasets = config["datasets"]
    pipeline_kwargs = pipeline_kwargs or {}

    # 対応データセットのみに絞る
    datasets = [d for d in datasets if d in available_datasets]

    if not datasets:
        print(f"  [SKIP] No compatible datasets for {pipeline_name}")
        return {}

    # シングルワーカー用のランナー
    runner = None if num_workers > 1 else PipelineRunner(pipeline_id, pipeline_kwargs)

    results = {}

    for dataset_name in datasets:
        print(f"\n  [{dataset_name}]")

        # データ読み込み（ランダムサンプリング対応）
        dataset_path = datasets_config[dataset_name]["path"]
        samples = load_dataset(
            dataset_name, num_samples, random_sample, datasets_config
        )
        if not samples:
            print(f"    [SKIP] No data found")
            continue

        print(f"    Loaded {len(samples)} samples (workers: {num_workers})")
        dataset_config = datasets_config[dataset_name]

        # パイプライン実行
        outputs: List[PipelineOutput] = []

        def _timeout_output(
            idx: int, sample: Dict[str, Any], err: str
        ) -> PipelineOutput:
            question = sample.get("question", "")
            entity_name = sample.get(dataset_config["entity_key"], "")
            answers_key = dataset_config.get("gold_answers_key", "answer_nodes")
            gold_answers = [node["name"] for node in sample.get(answers_key, [])]
            gold_relations = [
                sample.get(k, "")
                for k in dataset_config["gold_relations_keys"]
                if sample.get(k)
            ]
            out = PipelineOutput(
                idx=idx,
                question=question,
                entity_name=entity_name,
                gold_answers=gold_answers,
                gold_relations=gold_relations,
                predicted_relations=None,
                predicted_entities=[],
            )
            out.error = err
            return out

        if num_workers > 1:
            # マルチプロセス並列実行（ハング対策：spawn + per-sample timeout + 監視ループ）
            # NOTE: fork は外部API/HTTPクライアント利用時にデッドロックしやすいので spawn を使う
            ctx = mp.get_context("spawn")
            inflight_limit = max_inflight or max(1, num_workers * 2)

            indexed_samples = list(enumerate(samples))
            next_submit = 0

            future_to_meta: Dict[Any, Tuple[int, Dict[str, Any]]] = {}
            submit_time: Dict[Any, float] = {}
            pending = set()

            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(pipeline_id, pipeline_kwargs),
                mp_context=ctx,
            ) as ex:
                with tqdm(total=len(samples), desc=f"    Running", leave=False) as pbar:
                    try:
                        while next_submit < len(indexed_samples) or pending:
                            # 追加投入（過剰キューイングを避ける）
                            while (
                                next_submit < len(indexed_samples)
                                and len(pending) < inflight_limit
                            ):
                                i, sample = indexed_samples[next_submit]
                                fut = ex.submit(
                                    _process_sample_worker, (i, sample, dataset_config)
                                )
                                future_to_meta[fut] = (i, sample)
                                submit_time[fut] = time.time()
                                pending.add(fut)
                                next_submit += 1

                            if not pending:
                                continue

                            done, not_done = wait(
                                pending,
                                timeout=watch_interval_s,
                                return_when=FIRST_COMPLETED,
                            )

                            # 完了回収
                            for fut in done:
                                i, _sample = future_to_meta[fut]
                                try:
                                    output = fut.result()
                                except Exception as e:
                                    output = _timeout_output(
                                        i, _sample, f"future_error: {e}"
                                    )
                                outputs.append(output)
                                pbar.update(1)
                                pending.discard(fut)

                            # タイムアウト検出（待たずに結果化して進める）
                            now = time.time()
                            timed_out = []
                            for fut in not_done:
                                if (
                                    now - submit_time.get(fut, now)
                                    > per_sample_timeout_s
                                ):
                                    i, _sample = future_to_meta[fut]
                                    fut.cancel()
                                    outputs.append(
                                        _timeout_output(
                                            i,
                                            _sample,
                                            f"timeout({per_sample_timeout_s}s)",
                                        )
                                    )
                                    pbar.update(1)
                                    timed_out.append(fut)

                            for fut in timed_out:
                                pending.discard(fut)

                    finally:
                        # 重要：残タスクを待たずに閉じる（1件ハングしても全体が止まらない）
                        ex.shutdown(wait=False, cancel_futures=True)

            outputs.sort(key=lambda x: x.idx)
        else:
            # シーケンシャル実行
            for i, sample in enumerate(tqdm(samples, desc=f"    Running", leave=False)):
                assert runner is not None
                output = runner.run_sample(i, sample, dataset_config)
                outputs.append(output)

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
    kg_type: str = "primekgqa",
) -> None:
    """比較テーブルを表示"""

    # KGに応じたデータセットリスト
    datasets_config = DATASETS_BY_KG.get(kg_type, DATASETS_PRIMEKGQA)
    pipeline_configs = PIPELINE_CONFIGS_BY_KG.get(kg_type, PIPELINE_CONFIGS_PRIMEKGQA)
    datasets = list(datasets_config.keys())

    print(f"\n{'='*100}")
    print(f"COMPARISON SUMMARY ({kg_type.upper()})")
    print(f"{'='*100}")

    # データセットごとの比較
    for dataset in datasets:
        pipelines_with_data = [
            (pid, pdata) for pid, pdata in all_results.items() if dataset in pdata
        ]

        if not pipelines_with_data:
            continue

        dataset_display = DATASET_NAMES.get(dataset, dataset)
        print(f"\n--- {dataset_display} ---")
        print(
            f"{'Pipeline':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'PathAcc':>10} {'Latency':>10}"
        )
        print("-" * 95)

        for pipeline_id, pipeline_data in pipelines_with_data:
            m = pipeline_data[dataset]["metrics"]
            pipeline_name = pipeline_configs[pipeline_id]["name"]
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
    print(
        f"{'Pipeline':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'PathAcc':>10}"
    )
    print("-" * 85)

    for pipeline_id, pipeline_data in all_results.items():
        if not pipeline_data:
            continue

        n = len(pipeline_data)
        avg_acc = sum(d["metrics"]["accuracy"] for d in pipeline_data.values()) / n
        avg_recall = sum(d["metrics"]["recall"] for d in pipeline_data.values()) / n
        avg_prec = sum(d["metrics"]["precision"] for d in pipeline_data.values()) / n
        avg_f1 = sum(d["metrics"]["f1"] for d in pipeline_data.values()) / n
        avg_path = (
            sum(d["metrics"].get("path_accuracy", 0) for d in pipeline_data.values())
            / n
        )

        pipeline_name = pipeline_configs[pipeline_id]["name"]
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
        """,
    )
    p.add_argument(
        "--kg",
        type=str,
        default="primekgqa",
        choices=["primekgqa", "metaqa"],
        help="Knowledge Graph to use (default: primekgqa)",
    )
    p.add_argument("--pipeline", type=str, nargs="+", help="Pipeline(s) to evaluate")
    p.add_argument("--all", action="store_true", help="Evaluate all pipelines")
    p.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help="Specific dataset(s) to evaluate (default: all compatible)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples per dataset (default: 20)",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample from dataset instead of taking first N",
    )
    p.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for pipeline outputs (optional)",
    )
    p.add_argument(
        "--output", type=Path, default=None, help="Output JSON file for metrics summary"
    )
    p.add_argument(
        "--safe-delta",
        type=int,
        default=1,
        help="SAFE pipeline delta parameter (default: 1)",
    )
    p.add_argument(
        "--reranker",
        type=str,
        default="none",
        choices=["none", "llm", "hybrid"],
        help="Extended Type-KoPL reranker type (default: none)",
    )
    p.add_argument(
        "--reranker-input-k",
        type=int,
        default=10,
        help="Number of candidates to pass to reranker (default: 10)",
    )
    p.add_argument(
        "--per-sample-timeout",
        type=float,
        default=300.0,
        help="Per-sample timeout seconds for parallel execution (default: 300)",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="Max inflight tasks for parallel execution (default: workers*2)",
    )
    args = p.parse_args()

    # KGに応じた設定を取得
    kg_type = args.kg
    pipeline_configs = PIPELINE_CONFIGS_BY_KG.get(kg_type, PIPELINE_CONFIGS_PRIMEKGQA)
    datasets_config = DATASETS_BY_KG.get(kg_type, DATASETS_PRIMEKGQA)

    # パイプライン選択
    if args.all:
        pipelines = list(pipeline_configs.keys())
    elif args.pipeline:
        # パイプライン名を検証
        for p_id in args.pipeline:
            if p_id not in pipeline_configs:
                print(f"Error: Unknown pipeline '{p_id}' for KG '{kg_type}'")
                print(f"Available: {list(pipeline_configs.keys())}")
                return
        pipelines = args.pipeline
    else:
        p.print_help()
        print("\nError: Please specify --pipeline or --all")
        return

    # データセット選択
    if args.dataset:
        # データセット名を検証
        for d_id in args.dataset:
            if d_id not in datasets_config:
                print(f"Error: Unknown dataset '{d_id}' for KG '{kg_type}'")
                print(f"Available: {list(datasets_config.keys())}")
                return
        datasets = args.dataset
    else:
        datasets = list(datasets_config.keys())

    print(f"{'='*60}")
    print("EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"KG:        {kg_type.upper()}")
    print(f"Pipelines: {', '.join(pipelines)}")
    print(f"Datasets:  {', '.join(datasets)}")
    print(f"Samples:   {args.num_samples}" + (" (random)" if args.random else ""))
    print(f"Workers:   {args.workers}")
    if args.output_dir:
        print(f"Output:    {args.output_dir}")

    all_results = {}

    for pipeline_id in pipelines:
        pipeline_name = pipeline_configs[pipeline_id]["name"]
        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline_name}")
        print(f"{'='*60}")

        # パイプライン固有のパラメータ
        pipeline_kwargs = {"kg_type": kg_type}
        if pipeline_id == "safe":
            pipeline_kwargs["delta"] = args.safe_delta
            if args.safe_delta != 1:
                print(f"  (delta={args.safe_delta})")
        if pipeline_id == "extended_type_kopl":
            pipeline_kwargs["reranker_type"] = args.reranker
            pipeline_kwargs["reranker_input_k"] = args.reranker_input_k
            if args.reranker != "none":
                print(f"  (reranker={args.reranker}, input_k={args.reranker_input_k})")

        results = run_pipeline_evaluation(
            pipeline_id,
            datasets,
            args.num_samples,
            args.output_dir,
            args.random,
            args.workers,
            pipeline_kwargs,
            kg_type,
            per_sample_timeout_s=args.per_sample_timeout,
            max_inflight=args.max_inflight,
        )
        all_results[pipeline_id] = results

    # 複数パイプラインの場合は比較テーブルを表示
    if len(pipelines) > 1:
        print_comparison_table(all_results, kg_type)

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
