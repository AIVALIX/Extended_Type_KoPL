"""
評価を複数回実行して結果を集計するスクリプト

Usage (ホストから):
  python app/pipeline/run_repeated_eval.py \
    --runs 10 \
    --cmd "docker exec python-primekgqa-experiment python -m pipeline.run_evaluation \
      --kg metaqa --num-samples 400 --random --workers 32 --no-schema \
      --pipeline extended_type_kopl --reranker llm \
      --output-dir /app/result/eval_em_metaqa_reranker"

またはコンテナ内から:
  python -m pipeline.run_repeated_eval \
    --runs 10 \
    --cmd "python -m pipeline.run_evaluation \
      --kg metaqa --num-samples 400 --random --workers 32 --no-schema \
      --pipeline extended_type_kopl --reranker llm \
      --output-dir /app/result/eval_em_metaqa_reranker"
"""

import argparse
import json
import subprocess
import sys
import statistics
from pathlib import Path
from datetime import datetime


def run_single(cmd: str, seed: int, output_json: Path) -> dict:
    """1回の評価を実行"""
    full_cmd = f"{cmd} --seed {seed} --output {output_json}"
    print(f"\n{'='*60}")
    print(f"Run seed={seed}: {output_json.name}")
    print(f"{'='*60}")
    result = subprocess.run(full_cmd, shell=True)
    if result.returncode != 0:
        print(f"  [WARN] Exit code {result.returncode}", file=sys.stderr)
        return {}
    if output_json.exists():
        return json.loads(output_json.read_text())
    return {}


def aggregate(all_results: list[dict]) -> dict:
    """全runの結果を集計（mean, std, min, max）"""
    # pipeline -> dataset -> metric -> [values]
    collected: dict[str, dict[str, dict[str, list]]] = {}

    for result in all_results:
        for pipeline, datasets in result.items():
            if pipeline not in collected:
                collected[pipeline] = {}
            for dataset, metrics in datasets.items():
                if dataset not in collected[pipeline]:
                    collected[pipeline][dataset] = {}
                for metric, value in metrics.items():
                    if not isinstance(value, (int, float)):
                        continue
                    if metric not in collected[pipeline][dataset]:
                        collected[pipeline][dataset][metric] = []
                    collected[pipeline][dataset][metric].append(value)

    # 統計量を計算
    summary = {}
    for pipeline, datasets in collected.items():
        summary[pipeline] = {}
        for dataset, metrics in datasets.items():
            summary[pipeline][dataset] = {}
            for metric, values in metrics.items():
                n = len(values)
                mean = statistics.mean(values)
                std = statistics.stdev(values) if n > 1 else 0.0
                summary[pipeline][dataset][metric] = {
                    "mean": round(mean, 2),
                    "std": round(std, 2),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "n": n,
                }

    return summary


def print_summary(summary: dict, num_runs: int):
    """結果を表形式で表示"""
    print(f"\n{'='*80}")
    print(f"  Aggregated Results ({num_runs} runs)")
    print(f"{'='*80}")

    key_metrics = ["accuracy", "recall", "precision", "f1"]

    for pipeline, datasets in summary.items():
        print(f"\n  Pipeline: {pipeline}")
        print(f"  {'Dataset':<22} {'Metric':<12} {'Mean':>7} {'± Std':>7} {'Min':>7} {'Max':>7}")
        print(f"  {'-'*62}")
        for dataset, metrics in datasets.items():
            first = True
            for metric in key_metrics:
                if metric not in metrics:
                    continue
                s = metrics[metric]
                ds_label = dataset if first else ""
                print(f"  {ds_label:<22} {metric:<12} {s['mean']:>7.2f} {s['std']:>6.2f} {s['min']:>7.2f} {s['max']:>7.2f}")
                first = False


def main():
    parser = argparse.ArgumentParser(description="Run evaluation multiple times and aggregate")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--cmd", type=str, required=True, help="Base evaluation command (without --seed/--output)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for per-run results (default: auto)")
    parser.add_argument("--output", type=Path, default=None, help="Path for aggregated.json (default: output-dir/aggregated.json)")
    parser.add_argument("--start-seed", type=int, default=1, help="Starting seed value")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(f"result/repeated_eval_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i in range(args.runs):
        seed = args.start_seed + i
        output_json = output_dir / f"run_{seed}.json"
        result = run_single(args.cmd, seed, output_json)
        if result:
            all_results.append(result)
            print(f"  -> Run {i+1}/{args.runs} done (seed={seed})")
        else:
            print(f"  -> Run {i+1}/{args.runs} FAILED (seed={seed})")

    if not all_results:
        print("No successful runs.", file=sys.stderr)
        sys.exit(1)

    summary = aggregate(all_results)
    print_summary(summary, len(all_results))

    # 集計結果を保存（統計量 + 各runの生データ）
    summary_path = args.output or (output_dir / "aggregated.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "stats": summary,
        "runs": [
            {"seed": args.start_seed + i, "result": r}
            for i, r in enumerate(all_results)
        ],
    }
    summary_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
