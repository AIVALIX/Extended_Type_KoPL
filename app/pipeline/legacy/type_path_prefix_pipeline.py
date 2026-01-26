import os
from typing import Dict, Any, List, Optional, Literal, TypedDict
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

from core.config import get_settings
from llm_process.embedder import OpenAIEmbedder
from llm_process.gen_subquery import TypePathMetaSubQuery, SubQueryResponse
from llm_process.path_reranker import PathReranker
from database.search import GraphPathFinder, score_paths
from models.model import MetaQAType


# ────────────────────────────────────────────────────────────────
# Configuration and State
# ────────────────────────────────────────────────────────────────
@dataclass
class TypePathPipelineConfig:
    """Type Path Pipeline設定"""

    use_vector_prune: bool = True  # ベクトル類似度による剪定
    use_llm_reranker: bool = False  # LLM Rerankerの使用
    use_type_path_constraints: bool = True  # Type path制約の使用
    max_paths: int = 50  # 最大パス数
    top_k: int = 1  # 最終結果数


class TypePathPipelineState(TypedDict):
    """Type Path Pipeline状態管理"""

    # 入力
    question: str
    entity_name: str
    type_path: List[str]  # 指定されたtype path (例: ["CreativeWork", "Person"])
    config: TypePathPipelineConfig

    # 中間処理
    subqueries: Optional[List[str]]  # type pathに基づいて生成されたサブクエリ
    candidate_paths: List[List[str]]
    path_embeddings: List[List[Any]]
    query_embeddings: List[Any]
    scored_paths: List[tuple]
    reranked_paths: Optional[List[List[str]]]

    # 出力
    final_paths: List[List[str]]
    reachable_entities: List[str]
    processing_log: List[str]


# ────────────────────────────────────────────────────────────────
# Pipeline Components
# ────────────────────────────────────────────────────────────────
class TypePathPrefixPipeline:
    """Type Pathを前提とした専用パイプライン"""

    def __init__(self):
        self.settings = get_settings()
        self.embedder = OpenAIEmbedder()
        self.graph_finder = GraphPathFinder()

        # LLMコンポーネント（必要に応じて初期化）
        self.llm = None
        self.type_path_subquery_generator = None
        self.path_reranker = None

    def _init_llm_components(self):
        """LLMコンポーネントを遅延初期化"""
        if self.llm is None:
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = self.settings.OPENAI_API_KEY

            self.llm = init_chat_model(
                "gpt-4o-mini", model_provider="openai", temperature=0.0
            )
            self.type_path_subquery_generator = TypePathMetaSubQuery(self.llm)
            self.path_reranker = PathReranker(self.llm)

            # 通常のサブクエリ生成器も初期化
            from llm_process.gen_subquery import SubQueryRunnable

            self.subquery_generator = SubQueryRunnable(self.llm)

    # ノード関数群
    def input_processing(self, state: TypePathPipelineState) -> TypePathPipelineState:
        """入力処理とログ初期化"""
        state["processing_log"] = [
            f"Starting Type Path Pipeline with config: {state['config']}"
        ]
        state["processing_log"].append(f"Question: {state['question']}")
        state["processing_log"].append(f"Entity: {state['entity_name']}")
        state["processing_log"].append(f"Type Path: {state['type_path']}")
        return state

    def type_path_subquery_generation(
        self, state: TypePathPipelineState
    ) -> TypePathPipelineState:
        """Type Pathに基づくサブクエリ生成"""
        self._init_llm_components()

        if state["config"].use_type_path_constraints:
            # Type path制約ありの場合：TypePathMetaSubQueryを使用
            subquery_result = self.type_path_subquery_generator.invoke(
                type_path=state["type_path"], sentence=state["question"]
            )

            state["subqueries"] = subquery_result.sub_queries

            state["processing_log"].append(
                f"Type Path Subquery Generation: Generated {len(state['subqueries'])} subqueries following type path {state['type_path']}"
            )
            for idx, sq in enumerate(state["subqueries"], 1):
                expected_type = (
                    state["type_path"][idx - 1]
                    if idx - 1 < len(state["type_path"])
                    else "Unknown"
                )
                state["processing_log"].append(
                    f"  {idx}. {sq} (expected type: {expected_type})"
                )
        else:
            # Type path制約なしの場合：通常のSubQueryRunnableを使用（hop数のみ指定）
            hop_count = len(state["type_path"])  # type pathからhop数を取得

            subquery_result = self.subquery_generator.invoke(
                state["question"], hop=hop_count
            )

            state["subqueries"] = subquery_result.sub_queries

            state["processing_log"].append(
                f"Type Path Subquery Generation: Generated {len(state['subqueries'])} using normal SubQuery (hop={hop_count})"
            )
            for idx, sq in enumerate(state["subqueries"], 1):
                state["processing_log"].append(f"  {idx}. {sq}")

        return state

    def path_discovery(self, state: TypePathPipelineState) -> TypePathPipelineState:
        """Type Pathを使用したパス探索"""
        hop_count = len(state["type_path"])

        if state["config"].use_type_path_constraints:
            state["processing_log"].append(
                f"Path Discovery: Searching with type_path={state['type_path']} (hop={hop_count})"
            )

            paths, path_vecs = self.graph_finder.get_sequences_and_embeddings(
                entity_name=state["entity_name"],
                hop=hop_count,
                type_path=state["type_path"],
            )

            state["processing_log"].append(
                f"Path Discovery: Found {len(paths)} paths with type constraints"
            )
        else:
            state["processing_log"].append(
                f"Path Discovery: Searching with hop={hop_count} (no type constraints)"
            )

            paths, path_vecs = self.graph_finder.get_sequences_and_embeddings(
                entity_name=state["entity_name"],
                hop=hop_count,
                type_path=None,
            )

            state["processing_log"].append(
                f"Path Discovery: Found {len(paths)} paths without type constraints"
            )

        state["candidate_paths"] = paths
        state["path_embeddings"] = path_vecs

        if paths:
            state["processing_log"].append("Path Discovery: Sample paths:")
            for idx, path in enumerate(paths[:3], 1):
                state["processing_log"].append(f"  {idx}. {' -> '.join(path)}")

        return state

    def query_embedding(self, state: TypePathPipelineState) -> TypePathPipelineState:
        """クエリ埋め込み生成"""
        try:
            if state["subqueries"]:
                # サブクエリの埋め込み
                embeddings = self.embedder.embed_many(state["subqueries"])
                state["query_embeddings"] = embeddings
                state["processing_log"].append(
                    f"Query Embedding: Generated embeddings for {len(state['subqueries'])} subqueries"
                )
            else:
                # メインクエリの埋め込み
                embedding = self.embedder.embed_query(state["question"])
                # type_pathの長さに合わせて複製
                hop_count = len(state["type_path"])
                state["query_embeddings"] = [embedding] * hop_count
                state["processing_log"].append(
                    f"Query Embedding: Generated embedding for main query replicated {hop_count} times"
                )

        except Exception as e:
            state["processing_log"].append(f"Query Embedding: ❌ Error - {e}")
            state["query_embeddings"] = []

        return state

    def vector_scoring(self, state: TypePathPipelineState) -> TypePathPipelineState:
        """ベクトルスコアリング"""
        if not state["config"].use_vector_prune:
            state["scored_paths"] = [
                (i, 0.0) for i in range(len(state["candidate_paths"]))
            ]
            state["processing_log"].append("Vector Scoring: Skipping vector pruning")
            return state

        scored = score_paths(
            query_vecs=state["query_embeddings"],
            path_vectors=state["path_embeddings"],
            top_k=state["config"].max_paths,
        )
        state["scored_paths"] = scored
        state["processing_log"].append(
            f"Vector Scoring: Selected top {len(scored)} paths"
        )

        if scored:
            state["processing_log"].append("Vector Scoring: Top scored paths:")
            for idx, (path_idx, score) in enumerate(scored[:3], 1):
                path = state["candidate_paths"][path_idx]
                state["processing_log"].append(
                    f"  {idx}. {' -> '.join(path)} (score: {score:.4f})"
                )

        return state

    def llm_reranking(self, state: TypePathPipelineState) -> TypePathPipelineState:
        """LLMによる再ランキング"""
        if not state["config"].use_llm_reranker:
            state["reranked_paths"] = None
            state["processing_log"].append("LLM Reranking: Skipping LLM reranking")
            return state

        self._init_llm_components()

        # スコア順の上位パスを取得
        top_paths = [
            state["candidate_paths"][idx] for idx, _ in state["scored_paths"][:20]
        ]

        state["processing_log"].append(
            f"LLM Reranking: Reranking {len(top_paths)} paths"
        )

        # LLMで再ランキング
        rerank_result = self.path_reranker.invoke(
            queries=state["subqueries"], paths=top_paths
        )

        state["reranked_paths"] = [rerank_result.reranked_combos]
        state["processing_log"].append("LLM Reranking: Applied LLM reranking")
        state["processing_log"].append(
            f"LLM Reranking: Selected path: {rerank_result.reranked_combos}"
        )

        return state

    def final_processing(self, state: TypePathPipelineState) -> TypePathPipelineState:
        """最終処理と結果生成"""
        if state["config"].use_llm_reranker and state["reranked_paths"]:
            final_paths = [state["reranked_paths"][0]]
            state["processing_log"].append("Final Processing: Using LLM reranked path")
        elif state["scored_paths"]:
            best_idx, best_score = state["scored_paths"][0]
            final_paths = [state["candidate_paths"][best_idx]]
            state["processing_log"].append(
                f"Final Processing: Using top vector-scored path (score: {best_score:.4f})"
            )
        else:
            final_paths = [state["candidate_paths"][0]]
            state["processing_log"].append(
                "Final Processing: Using first candidate path"
            )

        state["final_paths"] = final_paths

        # 到達可能エンティティの取得
        selected_path = final_paths[0]
        state["processing_log"].append(
            f"Final Processing: Getting reachable entities for path: {selected_path}"
        )

        reachable = self.graph_finder.get_reachable_entities(
            entity_name=state["entity_name"],
            rel_types=selected_path,
            distinct=True,
            no_cycle=True,
        )

        state["reachable_entities"] = list(dict.fromkeys(reachable))

        state["processing_log"].append(
            f"Final Processing: Found {len(state['reachable_entities'])} reachable entities"
        )

        if state["reachable_entities"]:
            state["processing_log"].append(
                f"Final Processing: Sample entities: {state['reachable_entities'][:5]}"
            )

        return state

    def build_graph(self) -> StateGraph:
        """LangGraphの構築"""
        workflow = StateGraph(TypePathPipelineState)

        # ノードの追加
        workflow.add_node("input_processing", self.input_processing)
        workflow.add_node(
            "type_path_subquery_generation", self.type_path_subquery_generation
        )
        workflow.add_node("path_discovery", self.path_discovery)
        workflow.add_node("query_embedding", self.query_embedding)
        workflow.add_node("vector_scoring", self.vector_scoring)
        workflow.add_node("llm_reranking", self.llm_reranking)
        workflow.add_node("final_processing", self.final_processing)

        # エッジの定義（線形フロー）
        workflow.add_edge(START, "input_processing")
        workflow.add_edge("input_processing", "type_path_subquery_generation")
        workflow.add_edge("type_path_subquery_generation", "path_discovery")
        workflow.add_edge("path_discovery", "query_embedding")
        workflow.add_edge("query_embedding", "vector_scoring")
        workflow.add_edge("vector_scoring", "llm_reranking")
        workflow.add_edge("llm_reranking", "final_processing")
        workflow.add_edge("final_processing", END)

        return workflow.compile()


# ────────────────────────────────────────────────────────────────
# Configuration Factory
# ────────────────────────────────────────────────────────────────
def get_type_path_pipeline_config(
    pattern: Literal[1, 2, 3] = 1,
) -> TypePathPipelineConfig:
    """Type Path Pipeline設定を返す

    Parameters
    ----------
    pattern : Literal[1, 2, 3]
        パイプラインパターン
        1: Type Path制約 + Vector Pruning only
        2: Type Path制約 + Vector Pruning + LLM Reranking
        3: Hop数のみ + Vector Pruning only (通常のサブクエリ分割)
    """
    configs = {
        1: TypePathPipelineConfig(
            use_vector_prune=True,
            use_llm_reranker=False,
            use_type_path_constraints=True,  # Type path制約あり
            max_paths=50,
            top_k=1,
        ),
        2: TypePathPipelineConfig(
            use_vector_prune=True,
            use_llm_reranker=True,
            use_type_path_constraints=True,  # Type path制約あり
            max_paths=30,
            top_k=1,
        ),
        3: TypePathPipelineConfig(
            use_vector_prune=True,
            use_llm_reranker=True,
            use_type_path_constraints=False,  # Type path制約なし（hop数のみ）
            max_paths=50,
            top_k=1,
        ),
    }
    return configs[pattern]


# ────────────────────────────────────────────────────────────────
# Test Functions
# ────────────────────────────────────────────────────────────────
def test_type_path_pipeline():
    """Type Path Pipelineのテスト"""
    pipeline = TypePathPrefixPipeline()
    graph = pipeline.build_graph()

    test_cases = [
        {
            "question": "which person directed the movies starred by John Krasinski",
            "entity_name": "John Krasinski",
            "type_path": ["CreativeWork", "Person"],
            "description": "2-hop: movies -> directors",
        },
        {
            "question": "when were the films directed by actors who worked with Tom Hanks released",
            "entity_name": "Tom Hanks",
            "type_path": ["Person", "CreativeWork", "Date"],
            "description": "3-hop: actors -> films -> release dates",
        },
        {
            "question": "what movies did Jack Nicholson appear in",
            "entity_name": "Jack Nicholson",
            "type_path": ["CreativeWork"],
            "description": "1-hop: actor -> movies",
        },
    ]

    print("=" * 80)
    print("TYPE PATH PREFIX PIPELINE TEST")
    print("=" * 80)

    for test_case in test_cases:
        print(f"\nTest Case: {test_case['description']}")
        print(f"Question: {test_case['question']}")
        print(f"Entity: {test_case['entity_name']}")
        print(f"Type Path: {test_case['type_path']}")
        print("-" * 60)

        for pattern in [1, 2]:
            print(f"\n--- Type Path Pipeline Pattern {pattern} ---")

            config = get_type_path_pipeline_config(pattern)
            state = TypePathPipelineState(
                question=test_case["question"],
                entity_name=test_case["entity_name"],
                type_path=test_case["type_path"],
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

            try:
                import time

                start_time = time.time()
                result = graph.invoke(state)
                execution_time = time.time() - start_time

                print(f"Execution Time: {execution_time:.2f}s")

                # サブクエリの確認
                if result["subqueries"]:
                    print(f"Generated Subqueries ({len(result['subqueries'])}):")
                    for i, sq in enumerate(result["subqueries"], 1):
                        expected_type = (
                            test_case["type_path"][i - 1]
                            if i - 1 < len(test_case["type_path"])
                            else "Unknown"
                        )
                        print(f"  {i}. {sq} (expected: {expected_type})")

                # 結果の確認
                if result["final_paths"]:
                    selected_path = result["final_paths"][0]
                    entity_count = len(result["reachable_entities"])

                    print(f"✓ Selected Path: {selected_path}")
                    print(
                        f"✓ Path Length: {len(selected_path)} (expected: {len(test_case['type_path'])})"
                    )
                    print(f"✓ Reachable Entities: {entity_count}")

                    if result["reachable_entities"]:
                        print(f"✓ Sample Entities: {result['reachable_entities'][:5]}")

                    # Type pathとの一致確認
                    if len(selected_path) == len(test_case["type_path"]):
                        print(f"✓ Path length matches type_path length")
                    else:
                        print(
                            f"⚠️ Path length ({len(selected_path)}) != type_path length ({len(test_case['type_path'])})"
                        )
                else:
                    print("✗ No path selected")

                # 重要なログを表示
                important_logs = [
                    log
                    for log in result["processing_log"]
                    if any(
                        keyword in log
                        for keyword in [
                            "Generated",
                            "Found",
                            "Selected",
                            "FINAL RESULT",
                        ]
                    )
                ]
                if important_logs:
                    print("Key Process Logs:")
                    for log in important_logs[-5:]:  # 最後の5件
                        print(f"  {log}")

            except Exception as e:
                print(f"✗ Error in Pattern {pattern}: {e}")
                import traceback

                traceback.print_exc()

        print("\n" + "=" * 60)


def compare_with_original_pipeline():
    """元のパイプラインとの比較テスト"""
    from pipeline.graph_pipeline import (
        GraphQAPipeline,
        get_pipeline_config,
        PipelineState,
    )

    print("=" * 80)
    print("TYPE PATH PIPELINE VS ORIGINAL PIPELINE COMPARISON")
    print("=" * 80)

    # テストケース
    test_question = "which person directed the movies starred by John Krasinski"
    test_entity = "John Krasinski"
    test_type_path = ["CreativeWork", "Person"]

    print(f"Question: {test_question}")
    print(f"Entity: {test_entity}")
    print(f"Type Path: {test_type_path}")
    print("-" * 80)

    results = {}

    # Type Path Pipeline
    print("\n=== Type Path Pipeline ===")
    type_path_pipeline = TypePathPrefixPipeline()
    type_path_graph = type_path_pipeline.build_graph()

    for pattern in [1, 2]:
        print(f"\nType Path Pipeline Pattern {pattern}:")
        try:
            config = get_type_path_pipeline_config(pattern)
            state = TypePathPipelineState(
                question=test_question,
                entity_name=test_entity,
                type_path=test_type_path,
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

            import time

            start_time = time.time()
            result = type_path_graph.invoke(state)
            execution_time = time.time() - start_time

            selected_path = result["final_paths"][0] if result["final_paths"] else None
            entity_count = len(result["reachable_entities"])

            results[f"TypePath_{pattern}"] = {
                "time": execution_time,
                "path": selected_path,
                "entities": entity_count,
                "success": selected_path is not None,
            }

            print(f"  Time: {execution_time:.2f}s")
            print(f"  Path: {selected_path}")
            print(f"  Entities: {entity_count}")

        except Exception as e:
            print(f"  Error: {e}")
            results[f"TypePath_{pattern}"] = {"error": str(e)}

    # Original Pipeline
    print("\n=== Original Pipeline ===")
    original_pipeline = GraphQAPipeline()
    original_graph = original_pipeline.build_graph()

    for pattern in [1, 2, 3, 4]:
        print(f"\nOriginal Pipeline Pattern {pattern}:")
        try:
            config = get_pipeline_config(pattern, hop=None)
            state = PipelineState(
                question=test_question,
                entity_name=test_entity,
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

            import time

            start_time = time.time()
            result = original_graph.invoke(state)
            execution_time = time.time() - start_time

            selected_path = result["final_paths"][0] if result["final_paths"] else None
            entity_count = len(result["reachable_entities"])

            results[f"Original_{pattern}"] = {
                "time": execution_time,
                "path": selected_path,
                "entities": entity_count,
                "success": selected_path is not None,
            }

            print(f"  Time: {execution_time:.2f}s")
            print(f"  Path: {selected_path}")
            print(f"  Entities: {entity_count}")

        except Exception as e:
            print(f"  Error: {e}")
            results[f"Original_{pattern}"] = {"error": str(e)}

    # 比較表
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"{'Pipeline':<15} {'Time(s)':<8} {'Entities':<10} {'Path':<30} {'Success':<8}"
    )
    print("-" * 80)

    for name, result in results.items():
        if "error" not in result:
            success_mark = "✓" if result["success"] else "✗"
            path_str = str(result["path"])[:30] if result["path"] else "None"
            print(
                f"{name:<15} {result['time']:<8.2f} {result['entities']:<10} {path_str:<30} {success_mark:<8}"
            )
        else:
            print(f"{name:<15} {'ERROR':<8} {'0':<10} {'None':<30} {'✗':<8}")


if __name__ == "__main__":
    # Type Path Pipelineテスト
    test_type_path_pipeline()

    print("\n" + "=" * 80)

    # 元のパイプラインとの比較
    compare_with_original_pipeline()

# python pipeline/type_path_prefix_pipeline.py
