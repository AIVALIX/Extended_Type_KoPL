"""
統一精度算出スクリプト

各パイプラインの出力（JSONL）を読み込み、一括で精度を算出する

使用方法:
  python pipeline/evaluate_all_results.py
  python pipeline/evaluate_all_results.py --input-dir result/pipeline_outputs
  python pipeline/evaluate_all_results.py --pipeline extended_type_kopl safe kgt
  python pipeline/evaluate_all_results.py --dataset one_hop two_hop
  python pipeline/evaluate_all_results.py --output result/evaluation_report.json

評価指標:
  - Accuracy: 正解が予測集合に含まれるか（1つでも含まれればTrue）
  - Recall: 正解のうち何割が予測集合に含まれるか
  - Precision: 予測集合のうち何割が正解か
  - F1: PrecisionとRecallの調和平均
  - PathAcc: 予測パスが正解パスと完全一致するか
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.eval_metrics import (
    PipelineOutput,
    EvalResult,
    load_pipeline_outputs,
    evaluate_outputs,
    aggregate_metrics,
)


# パイプライン定義
PIPELINES = {
    "extended_type_kopl": {
        "name": "Extended Type-KoPL",
        "datasets": ["one_hop", "two_hop", "two_intersection", "three_intersection"],
    },
    "safe": {
        "name": "SAFE",
        "datasets": ["one_hop", "two_hop"],
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


def evaluate_pipeline_dataset(
    input_dir: Path,
    pipeline_id: str,
    dataset_name: str,
) -> Optional[Dict[str, Any]]:
    """
    特定のパイプライン・データセットの結果を評価

    Returns:
        評価結果の辞書、またはファイルが見つからない場合はNone
    """
    file_path = input_dir / f"{pipeline_id}_{dataset_name}.jsonl"
    if not file_path.exists():
        return None

    # 出力を読み込み
    outputs = load_pipeline_outputs(file_path)
    if not outputs:
        return None

    # 評価
    eval_results = evaluate_outputs(outputs)

    # 集計用の辞書リストに変換
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

    # メトリクス集計
    metrics = aggregate_metrics(result_dicts, len(result_dicts))

    return {
        "metrics": metrics,
        "results": [r.to_dict() for r in eval_results],
    }


def print_summary_table(all_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """サマリーテーブルを表示"""

    # ヘッダー
    datasets = ["one_hop", "two_hop", "two_intersection", "three_intersection"]

    print(f"\n{'='*100}")
    print("EVALUATION SUMMARY")
    print(f"{'='*100}")

    # データセットごとの比較表
    for dataset in datasets:
        dataset_display = DATASET_NAMES.get(dataset, dataset)

        # このデータセットに結果があるパイプラインを確認
        pipelines_with_data = [
            (pid, pdata)
            for pid, pdata in all_results.items()
            if dataset in pdata and pdata[dataset] is not None
        ]

        if not pipelines_with_data:
            continue

        print(f"\n--- {dataset_display} ---")
        print(f"{'Pipeline':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'PathAcc':>10} {'Latency':>10}")
        print("-" * 95)

        for pipeline_id, pipeline_data in pipelines_with_data:
            data = pipeline_data[dataset]
            if data is None:
                continue
            m = data["metrics"]
            pipeline_name = PIPELINES[pipeline_id]["name"]
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
    print("PIPELINE AVERAGES")
    print(f"{'='*100}")
    print(f"{'Pipeline':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'PathAcc':>10}")
    print("-" * 85)

    for pipeline_id, pipeline_data in all_results.items():
        valid_datasets = [
            d for d in pipeline_data.values()
            if d is not None and "metrics" in d
        ]
        if not valid_datasets:
            continue

        n = len(valid_datasets)
        avg_acc = sum(d["metrics"]["accuracy"] for d in valid_datasets) / n
        avg_recall = sum(d["metrics"]["recall"] for d in valid_datasets) / n
        avg_prec = sum(d["metrics"]["precision"] for d in valid_datasets) / n
        avg_f1 = sum(d["metrics"]["f1"] for d in valid_datasets) / n
        avg_path = sum(d["metrics"].get("path_accuracy", 0) for d in valid_datasets) / n

        pipeline_name = PIPELINES[pipeline_id]["name"]
        print(
            f"{pipeline_name:<25} "
            f"{avg_acc:>9.1f}% "
            f"{avg_recall:>9.1f}% "
            f"{avg_prec:>9.1f}% "
            f"{avg_f1:>9.1f}% "
            f"{avg_path:>9.1f}%"
        )


def main():
    p = argparse.ArgumentParser(description="Evaluate all pipeline results")
    p.add_argument("--input-dir", type=Path, default=Path("result/pipeline_outputs"),
                   help="Input directory containing pipeline outputs")
    p.add_argument("--pipeline", type=str, nargs="+",
                   choices=list(PIPELINES.keys()),
                   help="Specific pipeline(s) to evaluate (default: all)")
    p.add_argument("--dataset", type=str, nargs="+",
                   choices=["one_hop", "two_hop", "two_intersection", "three_intersection"],
                   help="Specific dataset(s) to evaluate (default: all available)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON file for detailed results")
    args = p.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        print("Please run the pipeline evaluation scripts first:")
        print("  python pipeline/evaluate_extended_type_kopl_pipeline.py")
        print("  python pipeline/evaluate_safe_pipeline.py")
        print("  python pipeline/evaluate_kgt_pipeline.py")
        return

    pipelines_to_eval = args.pipeline if args.pipeline else list(PIPELINES.keys())

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for pipeline_id in pipelines_to_eval:
        if pipeline_id not in PIPELINES:
            print(f"Unknown pipeline: {pipeline_id}")
            continue

        pipeline_config = PIPELINES[pipeline_id]
        pipeline_name = pipeline_config["name"]
        available_datasets = pipeline_config["datasets"]

        # フィルタリング
        datasets_to_eval = args.dataset if args.dataset else available_datasets
        datasets_to_eval = [d for d in datasets_to_eval if d in available_datasets]

        print(f"\n{'='*60}")
        print(f"Evaluating: {pipeline_name}")
        print(f"{'='*60}")

        all_results[pipeline_id] = {}

        for dataset_name in datasets_to_eval:
            result = evaluate_pipeline_dataset(args.input_dir, pipeline_id, dataset_name)
            all_results[pipeline_id][dataset_name] = result

            if result:
                m = result["metrics"]
                print(f"  {dataset_name}:")
                print(f"    Accuracy:  {m['accuracy']:.1f}%")
                print(f"    Recall:    {m['recall']:.1f}%")
                print(f"    Precision: {m['precision']:.1f}%")
                print(f"    F1:        {m['f1']:.1f}%")
                print(f"    PathAcc:   {m.get('path_accuracy', 0):.1f}%")
                print(f"    Errors:    {m['errors']}/{m['total']}")
                print(f"    Latency:   {m['avg_latency_ms']:.0f}ms")
            else:
                print(f"  {dataset_name}: [SKIP] No output file found")

    # サマリー表示
    print_summary_table(all_results)

    # 結果をJSON出力
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # メトリクスのみを出力（詳細結果は大きいため別途オプション）
        output_data = {
            pipeline_id: {
                dataset_name: data["metrics"] if data else None
                for dataset_name, data in pipeline_data.items()
            }
            for pipeline_id, pipeline_data in all_results.items()
        }

        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
