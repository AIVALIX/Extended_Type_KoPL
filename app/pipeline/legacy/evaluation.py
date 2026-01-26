import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import threading

from pipeline.graph_pipeline import GraphQAPipeline, get_pipeline_config, PipelineState
from pipeline.type_path_prefix_pipeline import (
    TypePathPrefixPipeline,
    get_type_path_pipeline_config,
    TypePathPipelineState,
)
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
    TimeoutError,
)
import time
import multiprocessing
import threading


class PipelineEvaluator:
    """パイプライン評価システム（並列化対応）"""

    def __init__(self):
        # 元のパイプライン
        self.graph_pipeline = GraphQAPipeline()
        self.graph = self.graph_pipeline.build_graph()

        # TypePathPrefixPipeline（すぐに初期化）
        self.type_path_pipeline = TypePathPrefixPipeline()
        self.type_path_graph = self.type_path_pipeline.build_graph()

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """JSONLファイルを読み込み"""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def extract_entity_from_question(self, question: str, ner: str) -> str:
        """質問からエンティティ名を抽出（[entity]形式から）"""
        # [John Krasinski] -> John Krasinski
        if "[" in question and "]" in question:
            start = question.find("[")
            end = question.find("]")
            if start != -1 and end != -1:
                return question[start + 1 : end]

        # フォールバック: nerフィールドを使用
        return ner

    def evaluate_single_case(
        self, test_case: Dict[str, Any], pattern: int, hop: Optional[int] = None
    ) -> Dict[str, Any]:
        """単一テストケースの評価（並列化対応）"""
        question = test_case["question"]
        ner = test_case["ner"]
        expected_answers = set(test_case["answers"])

        # エンティティ名を抽出
        entity_name = self.extract_entity_from_question(question, ner)

        if pattern in [1, 2, 3, 4]:
            # 元のGraphQAPipeline - スレッド毎に新しいインスタンスを作成
            graph_pipeline = GraphQAPipeline()
            graph = graph_pipeline.build_graph()

            config = get_pipeline_config(pattern, hop=hop)
            state = PipelineState(
                question=question,
                entity_name=entity_name,
                config=config,
                subqueries=None,
                meta_subqueries=None,
                candidate_paths=[],
                path_embeddings=[],
                query_embeddings=[],
                scored_paths=[],
                reranked_paths=None,
                final_paths=[],
                reachable_entities=[],
                processing_log=[],
            )
            graph_to_use = graph

        elif pattern in [5, 6, 7]:
            # type_pathの取得 - test_caseに含まれていることを前提
            if "predicted_type_path" not in test_case:
                raise ValueError(
                    f"Pattern {pattern} requires 'predicted_type_path' in test_case"
                )

            # TypePathPrefixPipeline - スレッド毎に新しいインスタンスを作成
            type_path_pipeline = TypePathPrefixPipeline()
            type_path_graph = type_path_pipeline.build_graph()

            # TypePathPrefixPipeline用の設定 (pattern 5 -> config 1, pattern 6 -> config 2)
            config = get_type_path_pipeline_config(pattern - 4)
            state = TypePathPipelineState(
                question=question,
                entity_name=entity_name,
                type_path=test_case["predicted_type_path"],
                config=config,
                subqueries=None,
                candidate_paths=[],
                path_embeddings=[],
                query_embeddings=[],
                scored_paths=[],
                reranked_paths=None,
                final_paths=[],
                reachable_entities=[],
                processing_log=[],
            )
            graph_to_use = type_path_graph
        else:
            raise ValueError(f"Unsupported pattern: {pattern}")

        try:
            # パイプライン実行
            start_time = time.time()
            result = graph_to_use.invoke(state)
            execution_time = time.time() - start_time

            # 結果の取得
            predicted_entities = set(result["reachable_entities"])
            selected_path = result["final_paths"][0] if result["final_paths"] else None

            meta_queries = result.get("meta_subqueries")

            subqueries = result.get("subqueries")
            if subqueries is None:
                if isinstance(meta_queries, list):
                    subqueries = [
                        item.get("query") for item in meta_queries if "query" in item
                    ]
                else:
                    subqueries = []

            type_path_from_meta = None
            if isinstance(meta_queries, list):
                extracted = [
                    item.get("answer_type")
                    for item in meta_queries
                    if item.get("answer_type")
                ]
                if extracted:
                    type_path_from_meta = extracted

            returned_type_path = result.get("type_path")
            if not returned_type_path:
                if type_path_from_meta:
                    returned_type_path = type_path_from_meta
                elif pattern in [5, 6, 7]:
                    returned_type_path = test_case.get("predicted_type_path")

            # 評価指標の計算
            is_full_hit = predicted_entities == expected_answers
            precision = (
                len(predicted_entities & expected_answers) / len(predicted_entities)
                if predicted_entities
                else 0
            )
            recall = (
                len(predicted_entities & expected_answers) / len(expected_answers)
                if expected_answers
                else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            return {
                "question": question,
                "entity_name": entity_name,
                "expected_answers": list(expected_answers),
                "predicted_entities": list(predicted_entities),
                "subqueries": subqueries,
                "selected_path": selected_path,
                "type_path": returned_type_path,
                "hop_count": getattr(config, "hop_count", None),
                "execution_time": execution_time,
                "full_hit": is_full_hit,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "success": True,
                "error": None,
            }

        except Exception as e:
            import traceback

            error_details = f"{str(e)}\n{traceback.format_exc()}"

            return {
                "question": question,
                "entity_name": entity_name,
                "expected_answers": list(expected_answers),
                "predicted_entities": [],
                "selected_path": None,
                "subqueries": [],
                "type_path": (
                    test_case.get("predicted_type_path") if pattern in [5, 6] else None
                ),
                "hop_count": (
                    getattr(config, "hop_count", None) if "config" in locals() else None
                ),
                "execution_time": 0,
                "full_hit": False,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "success": False,
                "error": error_details,
            }

    def _timeout_result(self, test_case, err_msg):
        return {
            "question": test_case.get("question"),
            "entity_name": self.extract_entity_from_question(
                test_case.get("question", ""), test_case.get("ner", "")
            ),
            "expected_answers": test_case.get("answers", []),
            "predicted_entities": [],
            "selected_path": None,
            "subqueries": [],
            "type_path": test_case.get("predicted_type_path"),
            "hop_count": None,
            "execution_time": 0.0,
            "full_hit": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "success": False,
            "error": err_msg,
        }

    def evaluate_dataset_parallel(
        self,
        file_path: str,
        pattern: int,
        limit: int = None,
        start: int = 0,
        hop: Optional[int] = None,
        max_workers: Optional[int] = None,
        use_threading: bool = True,
        *,
        per_task_timeout: float = 300.0,  # ★ 追加：各タスクの上限秒
        watch_interval: float = 5.0,  # ★ 追加：監視ループの周期
        max_inflight: int = 32,  # ★ 追加：API等の同時実行上限
    ) -> Dict[str, Any]:
        """データセット全体の評価（タイムアウト安全な並列版）"""
        all_test_cases = self.load_jsonl(file_path)

        if start > 0:
            all_test_cases = all_test_cases[start:]
        test_cases = all_test_cases[:limit] if limit is not None else all_test_cases

        if max_workers is None:
            max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)

        total_available = len(all_test_cases) + start
        processing_count = len(test_cases)
        range_str = (
            f"[{start}:{start + processing_count}]" if start > 0 or limit else ""
        )
        hop_str = f" hop={hop}" if hop is not None else ""
        executor_type = "ThreadPool" if use_threading else "ProcessPool"

        print(
            f"Evaluating {processing_count} test cases {range_str} out of {total_available} total "
            f"with Pipeline {pattern}{hop_str} using {executor_type} (workers={max_workers})..."
        )

        start_time = time.time()
        results_by_idx: Dict[int, Dict[str, Any]] = {}
        completed_count = 0

        # 追加：同時APIコールを絞るためのセマフォ
        sem = threading.Semaphore(max_inflight)

        def evaluate_task(idx, test_case):
            # セマフォで同時実行を制御（OpenAI等の外部呼び出し対策）
            with sem:
                return idx, self.evaluate_single_case(test_case, pattern, hop=hop)

        # —— submit & 監視ループ（未完了をタイムアウト切り）——
        executor_cls = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
        executor = executor_cls(max_workers=max_workers)

        try:
            indexed = list(enumerate(test_cases))
            # submit
            future_to_idx = {}
            submit_time = {}
            for idx, tc in indexed:
                fut = executor.submit(evaluate_task, idx, tc)
                future_to_idx[fut] = idx
                submit_time[fut] = time.time()

            pending = set(future_to_idx.keys())

            last_progress_log = time.time()

            while pending:
                # 完了分を回収
                done, not_done = wait(
                    pending, timeout=watch_interval, return_when=FIRST_COMPLETED
                )

                # 完了分の回収
                for fut in done:
                    idx = future_to_idx[fut]
                    try:
                        _idx, result = fut.result()  # ここでは即時取得できる
                    except Exception as e:
                        result = self._timeout_result(
                            test_cases[idx], f"future_error: {e}"
                        )
                    results_by_idx[idx] = result
                    completed_count += 1

                # タイムアウト検出（期限超のものは“結果化して”待列から外す）
                now = time.time()
                timed_out = []
                for fut in not_done:
                    if now - submit_time[fut] > per_task_timeout:
                        idx = future_to_idx[fut]
                        # ThreadPool の cancel は実行中だと効かないが、
                        # 我々は“待たない”ため、ここで結果を立てて pending から外す
                        timed_out.append(fut)
                        results_by_idx[idx] = self._timeout_result(
                            test_cases[idx], f"timeout({per_task_timeout}s)"
                        )
                        completed_count += 1

                # pending 集合を更新（完了＋タイムアウトを除去）
                for fut in done:
                    pending.discard(fut)
                for fut in timed_out:
                    pending.discard(fut)

                # 進捗ログ
                if (
                    now - last_progress_log
                ) >= 5 or completed_count == processing_count:
                    elapsed = now - start_time
                    avg = (elapsed / completed_count) if completed_count else 0.0
                    eta = max(0.0, avg * (processing_count - completed_count))
                    print(
                        f"Progress: {completed_count}/{processing_count} "
                        f"({completed_count/processing_count*100:.1f}%) "
                        f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
                    )
                    last_progress_log = now

                # すべて結果が埋まったら終了
                if completed_count >= processing_count:
                    break

        finally:
            # ★ 重要：残タスクを待たずに閉じる（pending があってもハングしない）
            executor.shutdown(wait=False, cancel_futures=True)

        # 順序を保ってリスト化
        results = [results_by_idx[i] for i in range(processing_count)]

        # 先頭10件だけ軽く表示（元のロジック踏襲）
        print("\n--- Individual Results ---")
        for i, result in enumerate(results[:10], 1):
            actual_index = start + i
            if result["success"]:
                hit_mark = "✓" if result["full_hit"] else "✗"
                pred_count = len(result["predicted_entities"])
                exp_count = len(result["expected_answers"])
                print(
                    f"{actual_index:3d}. {hit_mark} F1:{result['f1']:.3f} "
                    f"P/E:{pred_count}/{exp_count} T:{result['execution_time']:.2f}s"
                )
            else:
                print(f"{actual_index:3d}. ✗ ERROR: {str(result['error'])[:50]}...")

        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")

        # 集計
        successful = [r for r in results if r["success"]]
        if successful:
            full_hit_rate = sum(r["full_hit"] for r in successful) / len(successful)
            avg_precision = sum(r["precision"] for r in successful) / len(successful)
            avg_recall = sum(r["recall"] for r in successful) / len(successful)
            avg_f1 = sum(r["f1"] for r in successful) / len(successful)
            avg_time = sum(r["execution_time"] for r in successful) / len(successful)
        else:
            full_hit_rate = avg_precision = avg_recall = avg_f1 = avg_time = 0.0

        total_time = time.time() - start_time

        summary = {
            "pattern": pattern,
            "hop_count": hop,
            "total_cases": total_available,
            "processed_cases": processing_count,
            "start_index": start,
            "successful_cases": len(successful),
            "success_rate": (
                (len(successful) / processing_count) if processing_count else 0.0
            ),
            "full_hit_rate": full_hit_rate,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_execution_time": avg_time,
            "total_execution_time": total_time,
            "parallel_efficiency": (
                ((avg_time * processing_count) / total_time) if total_time > 0 else 0.0
            ),
            "max_workers": max_workers,
            "executor_type": executor_type,
            "detailed_results": results,
        }

        print(f"\n--- Performance Summary ---")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Average per case: {avg_time:.2f}s")
        print(f"Parallel efficiency: {summary['parallel_efficiency']:.2f}x")
        print(f"Workers used: {max_workers}")
        return summary

    def evaluate_dataset(
        self,
        file_path: str,
        pattern: int,
        limit: int = None,
        start: int = 0,
        hop: Optional[int] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_threading: bool = True,
    ) -> Dict[str, Any]:
        """データセット全体の評価（並列化オプション付き）"""
        if parallel:
            return self.evaluate_dataset_parallel(
                file_path, pattern, limit, start, hop, max_workers, use_threading
            )
        else:
            return self.evaluate_dataset_sequential(
                file_path, pattern, limit, start, hop
            )

    def evaluate_dataset_sequential(
        self,
        file_path: str,
        pattern: int,
        limit: int = None,
        start: int = 0,
        hop: Optional[int] = None,
    ) -> Dict[str, Any]:
        """データセット全体の評価（シーケンシャル版）"""
        all_test_cases = self.load_jsonl(file_path)

        # データ範囲の適用
        if start > 0:
            all_test_cases = all_test_cases[start:]

        if limit is not None:
            test_cases = all_test_cases[:limit]
        else:
            test_cases = all_test_cases

        results = []

        # 処理範囲の情報表示
        total_available = len(all_test_cases) + start
        processing_count = len(test_cases)
        range_str = (
            f"[{start}:{start + processing_count}]" if start > 0 or limit else ""
        )
        hop_str = f" hop={hop}" if hop is not None else ""

        print(
            f"Evaluating {processing_count} test cases {range_str} out of {total_available} total "
            f"with Pipeline {pattern}{hop_str} (Sequential)..."
        )

        start_time = time.time()

        for i, test_case in enumerate(test_cases, 1):
            actual_index = start + i
            print(
                f"Processing {i}/{processing_count} (#{actual_index}): {test_case['question'][:50]}..."
            )

            result = self.evaluate_single_case(test_case, pattern, hop=hop)
            results.append(result)

            # 進捗表示
            if result["success"]:
                hit_mark = "✓" if result["full_hit"] else "✗"
                pred_count = len(result["predicted_entities"])
                exp_count = len(result["expected_answers"])
                hop_info = (
                    f"hop={result['hop_count']}"
                    if result["hop_count"]
                    else "hop=default"
                )
                print(
                    f"  {hit_mark} Full Hit: {result['full_hit']}, "
                    f"Predicted: {pred_count}, Expected: {exp_count}, "
                    f"F1: {result['f1']:.3f}, Time: {result['execution_time']:.2f}s, {hop_info}"
                )

                # 詳細な不一致情報
                if not result["full_hit"] and result["predicted_entities"]:
                    pred_set = set(result["predicted_entities"])
                    exp_set = set(result["expected_answers"])
                    missing = exp_set - pred_set
                    extra = pred_set - exp_set
                    if missing:
                        print(f"    Missing: {list(missing)[:3]}...")
                    if extra:
                        print(f"    Extra: {list(extra)[:3]}...")
            else:
                print(f"  ✗ Error: {result['error'][:100]}...")

            # スロットリング対策は並列化では不要
            # time.sleep(0.2)

        # 全体統計の計算
        successful_results = [r for r in results if r["success"]]

        if successful_results:
            full_hit_rate = sum(r["full_hit"] for r in successful_results) / len(
                successful_results
            )
            avg_precision = sum(r["precision"] for r in successful_results) / len(
                successful_results
            )
            avg_recall = sum(r["recall"] for r in successful_results) / len(
                successful_results
            )
            avg_f1 = sum(r["f1"] for r in successful_results) / len(successful_results)
            avg_time = sum(r["execution_time"] for r in successful_results) / len(
                successful_results
            )
        else:
            full_hit_rate = avg_precision = avg_recall = avg_f1 = avg_time = 0

        total_time = time.time() - start_time

        summary = {
            "pattern": pattern,
            "hop_count": hop,
            "total_cases": total_available,
            "processed_cases": processing_count,
            "start_index": start,
            "successful_cases": len(successful_results),
            "success_rate": (
                len(successful_results) / processing_count
                if processing_count > 0
                else 0
            ),
            "full_hit_rate": full_hit_rate,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_execution_time": avg_time,
            "total_execution_time": total_time,
            "detailed_results": results,
        }

        return summary

    def compare_all_patterns(
        self,
        file_path: str,
        limit: int = None,
        start: int = 0,
        include_type_path: bool = False,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """全パターンの比較評価（並列化オプション付き）"""
        print("=" * 80)
        print("PIPELINE PATTERN COMPARISON EVALUATION")
        if parallel:
            workers_info = f" (Parallel: {max_workers or 'auto'} workers)"
            print(f"Execution Mode: Parallel{workers_info}")
        else:
            print("Execution Mode: Sequential")

        if limit or start > 0:
            range_info = f" (Range: {start}:{start + (limit or 'end')})"
            print(f"Data Range: {range_info}")
        print("=" * 80)

        pattern_summaries = {}
        patterns_to_test = [1, 2, 3, 4]

        if include_type_path:
            # TypePathPrefixPipelineを含める場合
            patterns_to_test.extend([5, 6, 7])
            print("Including TypePathPrefixPipeline (patterns 5-6)")

        for pattern in patterns_to_test:
            pipeline_name = (
                f"GraphQA Pipeline {pattern}"
                if pattern <= 4
                else f"TypePath Pipeline {pattern-4}"
            )
            print(f"\n--- Evaluating {pipeline_name} ---")

            try:
                summary = self.evaluate_dataset(
                    file_path,
                    pattern,
                    limit=limit,
                    start=start,
                    parallel=parallel,
                    max_workers=max_workers,
                )
                pattern_summaries[pattern] = summary
            except Exception as e:
                print(f"Error evaluating pattern {pattern}: {e}")
                # エラー時のダミーサマリー
                pattern_summaries[pattern] = {
                    "pattern": pattern,
                    "processed_cases": 0,
                    "success_rate": 0,
                    "full_hit_rate": 0,
                    "avg_precision": 0,
                    "avg_recall": 0,
                    "avg_f1": 0,
                    "avg_execution_time": 0,
                    "total_execution_time": 0,
                    "error": str(e),
                }

        # 比較表の作成
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(
            f"{'Pattern':<8} {'Type':<12} {'Processed':<10} {'Success%':<10} {'Full Hit%':<12} "
            f"{'Precision':<12} {'Recall':<10} {'F1':<10} {'Time(s)':<10} {'Total(s)':<10}"
        )
        print("-" * 120)

        for pattern in patterns_to_test:
            s = pattern_summaries[pattern]
            pipeline_type = "GraphQA" if pattern <= 4 else "TypePath"

            if "error" not in s:
                print(
                    f"{pattern:<8} {pipeline_type:<12} {s['processed_cases']:<10} {s['success_rate']*100:<10.1f} {s['full_hit_rate']*100:<12.1f} "
                    f"{s['avg_precision']:<12.3f} {s['avg_recall']:<10.3f} {s['avg_f1']:<10.3f} {s['avg_execution_time']:<10.2f} {s.get('total_execution_time', 0):<10.1f}"
                )
            else:
                print(
                    f"{pattern:<8} {pipeline_type:<12} {'ERROR':<10} {'0.0':<10} {'0.0':<12} {'0.000':<12} {'0.000':<10} {'0.000':<10} {'0.00':<10} {'0.0':<10}"
                )

        return {
            "pattern_summaries": pattern_summaries,
            "comparison_complete": True,
            "data_range": {"start": start, "limit": limit},
            "included_type_path": include_type_path,
            "execution_mode": "parallel" if parallel else "sequential",
            "max_workers": max_workers,
        }

    def save_detailed_results(self, results: Dict[str, Any], output_path: str):
        """詳細結果をJSONファイルに保存"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Detailed results saved to: {output_path}")


# 既存の関数は変更なし
def create_sample_test_data(output_path: str = "sample_test.jsonl"):
    """サンプルテストデータの作成"""
    sample_data = [
        {
            "question": "which person directed the movies starred by [John Krasinski]",
            "ner": "John Krasinski",
            "answers": [
                "Nancy Meyers",
                "Sam Mendes",
                "George Clooney",
                "Ken Kwapis",
                "Luke Greenfield",
            ],
        },
        {
            "question": "what movies did [Jack Nicholson] appear in",
            "ner": "Jack Nicholson",
            "answers": [
                "Chinatown",
                "One Flew Over the Cuckoo's Nest",
                "The Shining",
                "Batman",
                "A Few Good Men",
            ],
        },
        {
            "question": "who wrote the screenplay for movies starring [Tom Hanks]",
            "ner": "Tom Hanks",
            "answers": [
                "Robert Zemeckis",
                "Frank Darabont",
                "Cameron Crowe",
                "Nora Ephron",
            ],
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Sample test data created: {output_path}")
    return output_path


def create_sample_test_data_with_type_path(
    output_path: str = "sample_test_with_type_path.jsonl",
):
    """TypePath対応のサンプルテストデータの作成"""
    sample_data = [
        {
            "question": "which person directed the movies starred by [John Krasinski]",
            "ner": "John Krasinski",
            "answers": [
                "Nancy Meyers",
                "Sam Mendes",
                "George Clooney",
                "Ken Kwapis",
                "Luke Greenfield",
            ],
            "predicted_type_path": ["CreativeWork", "Person"],  # movies -> directors
        },
        {
            "question": "what movies did [Jack Nicholson] appear in",
            "ner": "Jack Nicholson",
            "answers": [
                "Chinatown",
                "One Flew Over the Cuckoo's Nest",
                "The Shining",
                "Batman",
                "A Few Good Men",
            ],
            "predicted_type_path": ["CreativeWork"],  # direct movies
        },
        {
            "question": "when were the films directed by actors who worked with [Tom Hanks] released",
            "ner": "Tom Hanks",
            "answers": ["1994", "1999", "2000", "2002"],
            "predicted_type_path": [
                "Person",
                "CreativeWork",
                "Date",
            ],  # actors -> films -> dates
        },
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Sample test data with type paths created: {output_path}")
    return output_path


def run_parallel_evaluation_test():
    """並列化評価システムのテスト実行"""
    # サンプルデータの作成
    test_file = create_sample_test_data()

    # 評価実行
    evaluator = PipelineEvaluator()

    # 並列評価のテスト
    print("=" * 80)
    print("PARALLEL EVALUATION TEST")
    print("=" * 80)

    # ThreadPoolExecutor使用
    print("\n--- ThreadPoolExecutor Test ---")
    summary_thread = evaluator.evaluate_dataset(
        test_file, pattern=2, parallel=True, use_threading=True, max_workers=4
    )

    print(f"\nThreadPool Results:")
    print(f"  Full Hit Rate: {summary_thread['full_hit_rate']*100:.1f}%")
    print(f"  Average F1: {summary_thread['avg_f1']:.3f}")
    print(f"  Total Time: {summary_thread['total_execution_time']:.2f}s")
    print(f"  Parallel Efficiency: {summary_thread['parallel_efficiency']:.2f}x")

    # ProcessPoolExecutor使用
    print("\n--- ProcessPoolExecutor Test ---")
    summary_process = evaluator.evaluate_dataset(
        test_file, pattern=2, parallel=True, use_threading=False, max_workers=2
    )

    print(f"\nProcessPool Results:")
    print(f"  Full Hit Rate: {summary_process['full_hit_rate']*100:.1f}%")
    print(f"  Average F1: {summary_process['avg_f1']:.3f}")
    print(f"  Total Time: {summary_process['total_execution_time']:.2f}s")
    print(f"  Parallel Efficiency: {summary_process['parallel_efficiency']:.2f}x")

    # シーケンシャル実行との比較
    print("\n--- Sequential Comparison ---")
    summary_seq = evaluator.evaluate_dataset(test_file, pattern=2, parallel=False)

    print(f"\nSequential Results:")
    print(f"  Full Hit Rate: {summary_seq['full_hit_rate']*100:.1f}%")
    print(f"  Average F1: {summary_seq['avg_f1']:.3f}")
    print(f"  Total Time: {summary_seq['total_execution_time']:.2f}s")

    print(f"\n--- Performance Comparison ---")
    print(
        f"Thread speedup: {summary_seq['total_execution_time'] / summary_thread['total_execution_time']:.2f}x"
    )
    print(
        f"Process speedup: {summary_seq['total_execution_time'] / summary_process['total_execution_time']:.2f}x"
    )


def run_full_parallel_evaluation_test():
    """TypePathPipelineを含む完全な並列評価テスト"""
    # TypePath対応テストデータの作成
    test_file = create_sample_test_data_with_type_path()

    # 評価実行
    evaluator = PipelineEvaluator()

    # 全パターン並列比較
    print("=" * 80)
    print("FULL PIPELINE PARALLEL COMPARISON")
    print("=" * 80)

    comparison_results = evaluator.compare_all_patterns(
        test_file, include_type_path=True, parallel=True, max_workers=8
    )

    # 結果保存
    evaluator.save_detailed_results(
        comparison_results, "parallel_evaluation_results.json"
    )

    print("\nParallel evaluation completed!")


if __name__ == "__main__":
    # 並列化テスト
    run_parallel_evaluation_test()

    print("\n" + "=" * 80)

    # TypePathPipelineを含む完全並列テスト
    run_full_parallel_evaluation_test()

# python pipeline/evaluation.py
