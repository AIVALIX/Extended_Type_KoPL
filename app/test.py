"""
パイプライン評価の実行スクリプト
"""

import argparse
from pathlib import Path
from pipeline.evaluation import PipelineEvaluator, create_sample_test_data


def main():
    parser = argparse.ArgumentParser(description="Pipeline Evaluation Tool")
    parser.add_argument("--data", type=str, help="Path to JSONL test data file")
    parser.add_argument(
        "--pattern",
        type=int,
        choices=[1, 2, 3, 4],
        help="Specific pattern to evaluate (1-4)",
    )
    parser.add_argument(
        "--hop",
        type=int,
        help="Number of hops for the pipeline (default: pattern-specific default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for detailed results",
    )
    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample test data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of test cases to process (default: all)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for processing test cases (default: 0)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (equivalent to --limit 10)",
    )

    args = parser.parse_args()

    # クイックモードの場合はlimitを10に設定
    if args.quick and args.limit is None:
        args.limit = 10

    # サンプルデータ作成
    if args.create_sample:
        sample_file = create_sample_test_data("sample_test_data.jsonl")
        print(f"Sample data created: {sample_file}")
        return

    # データファイルの確認
    if not args.data:
        print("Error: Please specify --data or use --create-sample")
        return

    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        return

    # 評価実行
    evaluator = PipelineEvaluator()

    # データ範囲の情報表示
    range_info = ""
    if args.start > 0 or args.limit is not None:
        end_idx = (args.start + args.limit) if args.limit else "end"
        range_info = f" (range: {args.start}-{end_idx})"
    elif args.limit is not None:
        range_info = f" (limit: {args.limit} cases)"

    # hop情報の表示
    hop_info = f" with hop={args.hop}" if args.hop is not None else ""

    if args.pattern:
        # 単一パターン評価
        print(
            f"Evaluating Pipeline {args.pattern} with {args.data}{range_info}{hop_info}"
        )
        summary = evaluator.evaluate_dataset(
            args.data, args.pattern, limit=args.limit, start=args.start, hop=args.hop
        )

        print(f"\nResults for Pipeline {args.pattern}:")
        print(f"  Total Cases: {summary['total_cases']}")
        print(f"  Processed Cases: {summary['processed_cases']}")
        print(f"  Hop Count: {summary['hop_count']}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"  Full Hit Rate: {summary['full_hit_rate']*100:.1f}%")
        print(f"  Average F1: {summary['avg_f1']:.3f}")
        print(f"  Average Time: {summary['avg_execution_time']:.2f}s")

        # 詳細結果保存
        evaluator.save_detailed_results({"pattern_summary": summary}, args.output)

    else:
        # 全パターン比較（hopは各パターンのデフォルトを使用）
        print(f"Comparing all patterns with {args.data}{range_info}")
        if args.hop is not None:
            print(
                f"Note: --hop={args.hop} specified but ignored for pattern comparison"
            )

        results = evaluator.compare_all_patterns(
            args.data, limit=args.limit, start=args.start
        )

        # 詳細結果保存
        evaluator.save_detailed_results(results, args.output)


if __name__ == "__main__":
    main()

# Usage examples:
# python test.py --create-sample
# python test.py --data sample_test_data.jsonl --pattern 2
# python test.py --data /app/data/metaqa/qa/2hop.jsonl --output data/metaqa/result/results.json
# python test.py --data /app/data/metaqa/qa/2hop.jsonl --pattern 2 --limit 5
# python test.py --data /app/data/metaqa/qa/2hop.jsonl --pattern 2 --hop 3 --limit 5
# python test.py --data /app/data/metaqa/qa/3hop.jsonl --pattern 4 --hop 3 --limit 5
