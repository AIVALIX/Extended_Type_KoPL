"""
パイプライン評価テスト（シンプル版）
dataset, pattern, limit, result-dirのみ指定可能
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from pipeline.evaluation import PipelineEvaluator
from core.config import get_settings


@dataclass
class TestConfiguration:
    """テスト設定の定義"""

    pattern: int
    hop: Optional[int]
    dataset_name: str
    dataset_path: str
    output_dir: str
    description: str


# 環境変数設定
settings = get_settings()
os.environ.update(
    {
        "LANGCHAIN_TRACING_V2": "false",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGSMITH_API_KEY": settings.LANGSMITH_API_KEY,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
    }
)


class SimpleTestRunner:
    """シンプルなテスト実行システム"""

    def __init__(self, base_result_dir: str = "result"):
        self.evaluator = PipelineEvaluator()
        self.base_result_dir = Path(base_result_dir)
        self.base_result_dir.mkdir(exist_ok=True)

        # データセット定義
        self.datasets = {
            "1hop": "/app/data/metaqa/qa/1hop.jsonl",
            "2hop": "/app/data/metaqa/qa/2hop.jsonl",
            "3hop": "/app/data/metaqa/qa/3hop.jsonl",
            "mixed": "/app/data/metaqa/qa/mixed_generated_path_refined.jsonl",
        }

    def _generate_test_configurations(
        self, patterns: List[int], datasets: List[str]
    ) -> List[TestConfiguration]:
        """指定されたパターンとデータセットの組み合わせを生成"""
        configurations = []

        for dataset_name in datasets:
            dataset_path = self.datasets[dataset_name]

            for pattern in patterns:
                # hop設定：mixedはNone、2hop/3hopは該当hop数
                if dataset_name == "mixed":
                    hop = None
                    hop_suffix = "_default"
                elif dataset_name == "1hop":
                    hop = None
                    hop_suffix = "_1hop"
                elif dataset_name == "2hop":
                    hop = None
                    hop_suffix = "_2hop"
                elif dataset_name == "3hop":
                    hop = None
                    hop_suffix = "_3hop"
                else:
                    hop = None
                    hop_suffix = "_default"

                # 出力ディレクトリ名
                pattern_names = {
                    1: "basic",
                    2: "vector_llm",
                    3: "type_vector",
                    4: "full",
                    5: "type_path",
                    6: "type_path_sub_vector",
                    7: "hop_no_typepath_vector_llm",
                }
                output_dir = (
                    f"pipeline_{pattern_names[pattern]}{hop_suffix}_{dataset_name}"
                )

                # 説明文
                descriptions = {
                    1: "Basic: SubQuery + Vector",
                    2: "Vector+LLM: SubQuery + Vector + LLM",
                    3: "Type+Vector: Type + SubQuery + Vector",
                    4: "Full: All modules enabled",
                    5: "TypePath: Type + SubQuery + Vector",
                    6: "TypePath+Vector: Type + SubQuery + Vector + LLM",
                    7: "Hop+Vector+LLM: Hop + SubQuery + Vector + LLM",
                }
                description = f"{descriptions[pattern]} on {dataset_name}"
                if hop is not None:
                    description += f" (hop={hop})"

                configurations.append(
                    TestConfiguration(
                        pattern=pattern,
                        hop=hop,
                        dataset_name=dataset_name,
                        dataset_path=dataset_path,
                        output_dir=output_dir,
                        description=description,
                    )
                )

        return configurations

    def run_single_test(
        self, config: TestConfiguration, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """単一テスト設定の実行"""
        print(f"\n{'='*80}")
        print(f"Running: {config.description}")
        print(
            f"Pattern: {config.pattern}, Dataset: {config.dataset_name}, Hop: {config.hop}"
        )
        print(f"Output: {config.output_dir}")
        print(f"{'='*80}")

        # 出力ディレクトリの作成
        output_path = self.base_result_dir / config.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # データセットの存在確認
        if not Path(config.dataset_path).exists():
            error_msg = f"Dataset not found: {config.dataset_path}"
            print(f"❌ {error_msg}")
            return {
                "config": config.__dict__,
                "success": False,
                "error": error_msg,
                "results": None,
            }

        try:
            # テスト実行
            start_time = time.time()
            print(f"🔄 Starting evaluation...")

            summary = self.evaluator.evaluate_dataset_parallel(
                file_path=config.dataset_path,
                pattern=config.pattern,
                hop=config.hop,
                limit=limit,
                max_workers=32,
            )
            execution_time = time.time() - start_time

            # 結果の拡張
            summary["total_execution_time"] = execution_time
            summary["config"] = config.__dict__

            # 詳細結果をJSONで保存
            result_file = output_path / "detailed_results.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            # サマリーをテキストで保存
            summary_file = output_path / "summary.txt"
            self._save_text_summary(summary, summary_file, config)

            print(f"✅ Test completed successfully")
            print(f"   Success Rate: {summary['success_rate']*100:.1f}%")
            print(f"   Full Hit Rate: {summary['full_hit_rate']*100:.1f}%")
            print(f"   Average F1: {summary['avg_f1']:.3f}")
            print(f"   Execution Time: {execution_time:.2f}s")
            print(f"   Results saved to: {output_path}")

            return {
                "config": config.__dict__,
                "success": True,
                "error": None,
                "results": summary,
            }

        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}"
            print(f"❌ {error_msg}")

            # エラー情報を保存
            error_file = output_path / "error.txt"
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"Test Configuration: {config.__dict__}\n")
                f.write(f"Error: {error_msg}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                import traceback

                f.write(f"\nTraceback:\n{traceback.format_exc()}")

            return {
                "config": config.__dict__,
                "success": False,
                "error": error_msg,
                "results": None,
            }

    def _save_text_summary(
        self, summary: Dict[str, Any], file_path: Path, config: TestConfiguration
    ):
        """テキスト形式でサマリーを保存"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Pipeline Evaluation Summary\n")
            f.write(f"{'='*50}\n\n")

            f.write(f"Configuration:\n")
            f.write(f"  Pattern: {config.pattern}\n")
            f.write(f"  Hop: {config.hop}\n")
            f.write(f"  Dataset: {config.dataset_name} ({config.dataset_path})\n")
            f.write(f"  Description: {config.description}\n")
            f.write(f"  Output Directory: {config.output_dir}\n\n")

            f.write(f"Results:\n")
            f.write(f"  Total Cases: {summary['total_cases']}\n")
            f.write(f"  Processed Cases: {summary['processed_cases']}\n")
            f.write(f"  Successful Cases: {summary['successful_cases']}\n")
            f.write(f"  Success Rate: {summary['success_rate']*100:.1f}%\n")
            f.write(f"  Full Hit Rate: {summary['full_hit_rate']*100:.1f}%\n")
            f.write(f"  Average Precision: {summary['avg_precision']:.3f}\n")
            f.write(f"  Average Recall: {summary['avg_recall']:.3f}\n")
            f.write(f"  Average F1: {summary['avg_f1']:.3f}\n")
            f.write(f"  Average Execution Time: {summary['avg_execution_time']:.2f}s\n")
            f.write(f"  Total Execution Time: {summary['total_execution_time']:.2f}s\n")

            f.write(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def run_tests(
        self,
        patterns: List[int],
        datasets: List[str],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """指定されたパターンとデータセットでテスト実行"""
        print(f"🚀 Starting Pipeline Tests")
        print(f"Patterns: {patterns}")
        print(f"Datasets: {datasets}")
        if limit:
            print(f"Test case limit per configuration: {limit}")
        print(f"Result directory: {self.base_result_dir.absolute()}")

        # テスト設定の生成
        configurations = self._generate_test_configurations(patterns, datasets)

        if not configurations:
            print("❌ No valid configurations generated")
            return {"success": False, "error": "No valid configurations"}

        print(f"\n📋 Test configurations to run ({len(configurations)}):")
        for i, config in enumerate(configurations, 1):
            print(f"  {i}. {config.output_dir}")

        # テスト実行
        all_results = []
        successful_tests = 0
        failed_tests = 0

        start_time = time.time()

        for i, config in enumerate(configurations, 1):
            print(f"\n🔄 Test {i}/{len(configurations)}: {config.output_dir}")

            result = self.run_single_test(config, limit=limit)
            all_results.append(result)

            if result["success"]:
                successful_tests += 1
                print(f"✅ Test {i} completed successfully")
            else:
                failed_tests += 1
                print(f"❌ Test {i} failed: {result['error']}")

        total_time = time.time() - start_time

        # 全体サマリーの生成
        overall_summary = {
            "total_configurations": len(configurations),
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (
                successful_tests / len(configurations) if configurations else 0
            ),
            "total_execution_time": total_time,
            "test_results": all_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_result_dir": str(self.base_result_dir.absolute()),
            "filters_applied": {
                "patterns": patterns,
                "datasets": datasets,
                "limit": limit,
            },
        }

        # 全体結果の保存
        # overall_result_file = self.base_result_dir / "overall_summary.json"
        # with open(overall_result_file, "w", encoding="utf-8") as f:
        #     json.dump(overall_summary, f, ensure_ascii=False, indent=2)

        # 全体サマリーテキストの生成
        self._generate_overall_summary_text(overall_summary)

        print(f"\n🎉 Tests Completed!")
        print(f"   Total Tests: {len(configurations)}")
        print(f"   Successful: {successful_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {overall_summary['success_rate']*100:.1f}%")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Results saved to: {self.base_result_dir.absolute()}")

        return overall_summary

    def _generate_overall_summary_text(self, overall_summary: Dict[str, Any]):
        """全体サマリーテキストの生成"""
        summary_file = self.base_result_dir / "overall_summary.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Overall Pipeline Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Execution Summary:\n")
            f.write(
                f"  Total Configurations: {overall_summary['total_configurations']}\n"
            )
            f.write(f"  Successful Tests: {overall_summary['successful_tests']}\n")
            f.write(f"  Failed Tests: {overall_summary['failed_tests']}\n")
            f.write(
                f"  Overall Success Rate: {overall_summary['success_rate']*100:.1f}%\n"
            )
            f.write(
                f"  Total Execution Time: {overall_summary['total_execution_time']:.2f}s\n"
            )
            f.write(f"  Generated: {overall_summary['timestamp']}\n\n")

            # フィルタ情報
            filters = overall_summary["filters_applied"]
            f.write(f"Applied Filters:\n")
            f.write(f"  Patterns: {filters['patterns']}\n")
            f.write(f"  Datasets: {filters['datasets']}\n")
            f.write(f"  Limit: {filters['limit']}\n\n")

            # 成功したテストの詳細
            f.write("Test Results:\n")
            f.write("-" * 40 + "\n")

            successful_results = [
                r for r in overall_summary["test_results"] if r["success"]
            ]

            if successful_results:
                f.write(
                    f"{'Config':<30} {'Dataset':<8} {'Hit%':<8} {'F1':<8} {'Time':<8}\n"
                )
                f.write("-" * 75 + "\n")

                for result in successful_results:
                    config = result["config"]
                    summary = result["results"]
                    config_name = f"Pattern{config['pattern']}"

                    f.write(
                        f"{config_name:<30} {config['dataset_name']:<8} "
                        f"{summary['full_hit_rate']*100:<8.1f} {summary['avg_f1']:<8.3f} "
                        f"{summary['avg_execution_time']:<8.2f}\n"
                    )
            else:
                f.write("No successful tests.\n")

            # 失敗したテスト
            failed_results = [
                r for r in overall_summary["test_results"] if not r["success"]
            ]
            if failed_results:
                f.write(f"\nFailed Tests:\n")
                for result in failed_results:
                    config = result["config"]
                    f.write(
                        f"Pattern {config['pattern']}, {config['dataset_name']}: {result['error']}\n"
                    )


def main():
    """メイン実行関数"""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Pipeline Test Runner")
    parser.add_argument(
        "--pattern",
        type=int,
        nargs="+",  # 複数指定可能
        choices=[1, 2, 3, 4, 5, 6, 7],
        default=[1, 2, 3, 4],  # デフォルトは1-4
        help="Pattern(s) to test (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",  # 複数指定可能
        choices=["1hop", "2hop", "3hop", "mixed"],
        default=["2hop", "3hop"],  # デフォルトは2hop, 3hop
        help="Dataset(s) to test (default: 2hop 3hop)",
    )
    parser.add_argument("--limit", type=int, help="Limit test cases per configuration")
    parser.add_argument(
        "--result-dir", default="result", help="Base result directory (default: result)"
    )

    args = parser.parse_args()

    print(f"🚀 Simple Pipeline Test Runner")
    print(f"Patterns: {args.pattern}")
    print(f"Datasets: {args.dataset}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print(f"Result Directory: {args.result_dir}")
    # テスト実行
    try:
        runner = SimpleTestRunner(base_result_dir=args.result_dir)
        runner.run_tests(
            patterns=args.pattern,
            datasets=args.dataset,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

# Usage examples:
# python overall_test.py --pattern 1 2 --dataset 2hop 3hop --limit 100
# python overall_test.py --pattern 4 --dataset mixed --limit 50
# python overall_test.py --pattern 1 2 3 4 --dataset 2hop 3hop mixed --limit 200 --result-dir metaqa_results
# python overall_test.py  # デフォルト: pattern 1-4, dataset 2hop+3hop
# python overall_test.py --pattern 7 --dataset mixed --result-dir metaqa_results
# python overall_test.py --pattern 2 4 --dataset 1hop --result-dir metaqa_results
# python overall_test.py --pattern 1 2 3 4 --dataset mixed --limit 1000 --result-dir metaqa_results_1000
# python overall_test.py --pattern 1 --dataset 1hop --limit 500 --result-dir metaqa_results_1hop
# python overall_test.py --pattern 2 4 --dataset 1hop 2hop 3hop mixed --limit 1000 --result-dir metaqa_results_1000
# python overall_test.py --pattern 2 4 --dataset 1hop 2hop 3hop mixed --limit 1000 --result-dir metaqa_results_1000
# python overall_test.py --pattern 1 2 3 4 --dataset mixed --limit 1000 --result-dir metaqa_results_1000
# python overall_test.py --pattern 2 4 --dataset 2hop 3hop  --limit 100 --result-dir metaqa_results_100
# python overall_test.py --pattern 4 --dataset 1hop 2hop 3hop mixed --limit 1000 --result-dir metaqa_results_1000/shex_ex
