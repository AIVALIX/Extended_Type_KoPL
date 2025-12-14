import os
from typing import Dict, Any, List, Optional, Literal, TypedDict
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

from core.config import get_settings
from llm_process.embedder import OpenAIEmbedder
from llm_process.gen_subquery import SubQueryRunnable, MetaSubQueryRunnable
from llm_process.path_reranker import PathReranker
from database.search import GraphPathFinder, score_paths
from models.model import MetaQAType, MetaSubQueryResponse, SubQueryResponse


# ────────────────────────────────────────────────────────────────
# Configuration and State
# ────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """パイプライン実行設定"""

    use_type_pruning: bool = False  # Module1: type剪定の使用
    use_subquery_split: bool = False  # Module2: サブクエリ分割とvector pruneの使用
    use_vector_prune: bool = False  # Module3: ベクトル類似度による剪定
    use_llm_reranker: bool = False  # Module4: LLM Rerankerの使用
    hop_count: int = 2  # 探索hop数
    max_paths: int = 50  # 最大パス数
    top_k: int = 10  # 最終結果数


class PipelineState(TypedDict):
    """パイプライン状態管理"""

    # 入力
    question: str
    entity_name: str
    config: PipelineConfig

    # 中間処理
    subqueries: Optional[List[str]]
    meta_subqueries: Optional[List[Dict[str, str]]]  # MetaSubQueryItemのリスト
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
class GraphQAPipeline:
    """4パターンの処理フローを持つGraph QAパイプライン"""

    def __init__(self):
        self.settings = get_settings()
        self.embedder = OpenAIEmbedder()
        self.graph_finder = GraphPathFinder()

        # LLMコンポーネント（必要に応じて初期化）
        self.llm = None
        self.subquery_generator = None
        self.meta_subquery_generator = None
        self.path_reranker = None

    def _init_llm_components(self):
        """LLMコンポーネントを遅延初期化"""
        if self.llm is None:
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = self.settings.OPENAI_API_KEY

            self.llm = init_chat_model(
                "gpt-4.1-mini", model_provider="openai", temperature=0.0
            )
            self.subquery_generator = SubQueryRunnable(self.llm)
            self.meta_subquery_generator = MetaSubQueryRunnable(self.llm)
            self.path_reranker = PathReranker(self.llm)

    # ノード関数群
    def input_processing(self, state: PipelineState) -> PipelineState:
        """入力処理とログ初期化"""
        state["processing_log"] = [f"Starting pipeline with config: {state['config']}"]
        state["processing_log"].append(f"Question: {state['question']}")
        state["processing_log"].append(f"Entity: {state['entity_name']}")
        return state

    def path_discovery(self, state: PipelineState) -> PipelineState:
        """Module1: パス探索（MetaSubQueryRunnableを使用したtype剪定あり/なし）"""
        config = state["config"]

        # Module1: type剪定の設定
        type_path = None
        if config.use_type_pruning:
            # MetaSubQueryRunnableを使用してtype pathを生成
            self._init_llm_components()

            try:
                # hop_countが設定されている場合のみhopを指定
                if hasattr(config, "hop_count") and config.hop_count is not None:
                    # MetaSubQueryRunnableでtype pathを推論（hop数を指定）
                    meta_result: MetaSubQueryResponse = (
                        self.meta_subquery_generator.invoke(
                            state["question"], hop=config.hop_count
                        )
                    )
                    state["processing_log"].append(
                        f"Module1: MetaSubQueryRunnable called with hop={config.hop_count}"
                    )
                else:
                    # hop数を指定せずにtype pathを推論
                    meta_result: MetaSubQueryResponse = (
                        self.meta_subquery_generator.invoke(state["question"])
                    )
                    state["processing_log"].append(
                        "Module1: MetaSubQueryRunnable called without hop specification"
                    )
                # MetaSubQueryResponseからtype pathを抽出
                if (
                    meta_result
                    and "sub_queries" in meta_result
                    and meta_result["sub_queries"]
                ):
                    # 各サブクエリのanswer_typeからtype_pathを構築
                    type_path = []
                    meta_subqueries_info = []

                    for sub_query_item in meta_result["sub_queries"]:
                        query_text = sub_query_item["query"]
                        answer_type = sub_query_item["answer_type"]

                        type_path.append(answer_type)
                        meta_subqueries_info.append(
                            {"query": query_text, "answer_type": answer_type}
                        )

                    # hop_countがNoneの場合、生成されたtype_pathの長さをhop_countに設定
                    if not hasattr(config, "hop_count") or config.hop_count is None:
                        config.hop_count = len(type_path)
                        state["config"] = config  # 更新されたconfigを状態に保存
                        state["processing_log"].append(
                            f"Module1: Inferred hop_count={config.hop_count} from MetaSubQuery type_path length"
                        )

                    # メタ情報を状態に保存
                    state["meta_subqueries"] = meta_subqueries_info

                    state["processing_log"].append(
                        f"Module1: MetaSubQueryRunnable generated type_path: {type_path}"
                    )

                    # サブクエリ情報もログに記録
                    state["processing_log"].append(
                        "Module1: Meta sub-queries with types:"
                    )
                    for i, sq_info in enumerate(meta_subqueries_info, 1):
                        state["processing_log"].append(
                            f"  {i}. {sq_info['query']} -> {sq_info['answer_type']}"
                        )

            except Exception as e:
                state["processing_log"].append(
                    f"Module1: Error in MetaSubQueryRunnable: {e}"
                )

        try:
            # hop_countの取得（デフォルト値の設定）
            hop_count = (
                getattr(config, "hop_count", 2) if hasattr(config, "hop_count") else 2
            )

            # パスと埋め込みを取得
            paths, path_vecs = self.graph_finder.get_sequences_and_embeddings(
                entity_name=state["entity_name"],
                hop=hop_count,
                type_path=type_path,
            )

            state["candidate_paths"] = paths
            state["path_embeddings"] = path_vecs

            log_msg = f"Module1: Found {len(paths)} candidate paths (hop={hop_count})"
            if type_path:
                log_msg += f" with MetaSubQuery type pruning {type_path}"
            else:
                log_msg += " without type pruning"
            state["processing_log"].append(log_msg)

            # パスが見つからない場合の段階的代替処理
            if not paths and config.use_type_pruning:
                state["processing_log"].append(
                    "Module1: No paths found with MetaSubQuery type pruning, trying relaxed constraints..."
                )

                # より緩い制約で再試行（CreativeWorkを中心に）
                relaxed_type_path = ["CreativeWork"] * hop_count
                paths, path_vecs = self.graph_finder.get_sequences_and_embeddings(
                    entity_name=state["entity_name"],
                    hop=hop_count,
                    type_path=relaxed_type_path,
                )

                if paths:
                    state["candidate_paths"] = paths
                    state["path_embeddings"] = path_vecs
                    state["processing_log"].append(
                        f"Module1: Relaxed CreativeWork constraints found {len(paths)} candidate paths"
                    )
                else:
                    # 最終的にtype制約なしで再試行
                    state["processing_log"].append(
                        "Module1: Relaxed constraints failed, trying without any type constraints..."
                    )
                    paths, path_vecs = self.graph_finder.get_sequences_and_embeddings(
                        entity_name=state["entity_name"],
                        hop=hop_count,
                        type_path=state["type_path"],  # Noneに戻す
                    )
                    state["candidate_paths"] = paths
                    state["path_embeddings"] = path_vecs
                    state["processing_log"].append(
                        f"Module1: No constraints found {len(paths)} candidate paths"
                    )

        except Exception as e:
            state["processing_log"].append(f"Module1: Error in path discovery: {e}")
            state["candidate_paths"] = []
            state["path_embeddings"] = []

        return state

    def subquery_generation(self, state: PipelineState) -> PipelineState:
        """Module2: サブクエリ生成（質問を単一ホップ質問に分解）"""
        if state["config"].use_type_pruning == False:
            if not state["config"].use_subquery_split:
                state["subqueries"] = None
                state["processing_log"].append("Module2: Skipping subquery generation")
                return state

            self._init_llm_components()

            try:
                config = state["config"]

                # SubQueryRunnableでサブクエリを生成
                if hasattr(config, "hop_count") and config.hop_count is not None:
                    # hop数を指定してサブクエリ生成
                    subquery_result: SubQueryResponse = self.subquery_generator.invoke(
                        state["question"], hop=config.hop_count
                    )
                    state["processing_log"].append(
                        f"Module2: SubQueryRunnable called with explicit hop={config.hop_count}"
                    )
                else:
                    # hop数を指定せずにサブクエリ生成（LLMが自動判断）
                    subquery_result: SubQueryResponse = self.subquery_generator.invoke(
                        state["question"]
                    )
                    state["processing_log"].append(
                        "Module2: SubQueryRunnable called without hop specification (LLM auto-determines)"
                    )

                # SubQueryResponseからサブクエリリストを抽出
                if subquery_result and hasattr(subquery_result, "sub_queries"):
                    state["subqueries"] = subquery_result.sub_queries

                    # hop_count=Noneの場合、生成されたサブクエリ数をhop_countに設定
                    if not hasattr(config, "hop_count") or config.hop_count is None:
                        inferred_hop_count = len(state["subqueries"])
                        # configオブジェクトを更新（dataclassなので直接変更）
                        config.hop_count = inferred_hop_count
                        state["config"] = config
                        state["processing_log"].append(
                            f"Module2: Inferred hop_count={inferred_hop_count} from {len(state['subqueries'])} generated subqueries"
                        )

                    state["processing_log"].append(
                        f"Module2: SubQueryRunnable generated {len(state['subqueries'])} subqueries"
                    )
                    for i, sq in enumerate(state["subqueries"], 1):
                        state["processing_log"].append(f"  {i}. {sq}")
                else:
                    # フォールバック: 元の質問を使用
                    state["subqueries"] = [state["question"]]

                    # hop_count=Noneの場合、デフォルトの2hopに設定
                    if not hasattr(config, "hop_count") or config.hop_count is None:
                        config.hop_count = 2
                        state["config"] = config
                        state["processing_log"].append(
                            "Module2: SubQueryRunnable failed, using original question with default hop_count=2"
                        )
                    else:
                        state["processing_log"].append(
                            "Module2: SubQueryRunnable failed, using original question"
                        )

            except Exception as e:
                state["processing_log"].append(
                    f"Module2: Error in subquery generation: {e}"
                )
                state["subqueries"] = [
                    state["question"]
                ]  # フォールバック: 元の質問を使用

                # hop_count=Noneの場合、エラー時もデフォルトの2hopに設定
                if (
                    not hasattr(state["config"], "hop_count")
                    or state["config"].hop_count is None
                ):
                    state["config"].hop_count = 2
                    state["processing_log"].append(
                        "Module2: Error fallback - setting hop_count=2"
                    )
        return state

    def query_embedding(self, state: PipelineState) -> PipelineState:
        """クエリ埋め込み生成"""
        config = state["config"]

        # hop_countは既にサブクエリ生成で設定されている
        hop_count = getattr(config, "hop_count", 2)

        if config.use_subquery_split and state["subqueries"]:
            # サブクエリの埋め込み
            embeddings = self.embedder.embed_many(state["subqueries"])
            state["query_embeddings"] = embeddings
            state["processing_log"].append(
                f"Generated embeddings for {len(state['subqueries'])} subqueries (hop_count={hop_count})"
            )
        elif config.use_type_pruning and state["meta_subqueries"]:
            # メタサブクエリの埋め込み
            embeddings = self.embedder.embed_many(
                [item["query"] for item in state["meta_subqueries"]]
            )
            state["query_embeddings"] = embeddings
            state["processing_log"].append(
                f"Generated embeddings for {len(state['meta_subqueries'])} meta subqueries (hop_count={hop_count})"
            )
        else:
            # メインクエリの埋め込み
            embedding = self.embedder.embed_query(state["question"])
            # hop数に合わせて複製（簡易的）
            state["query_embeddings"] = [embedding] * hop_count
            state["processing_log"].append(
                f"Generated embedding for main query replicated {hop_count} times"
            )

        return state

    def vector_scoring(self, state: PipelineState) -> PipelineState:
        """Module3: ベクトルスコアリング（質問群とリレーションとのコサイン類似度による剪定）"""
        if not state["candidate_paths"] or not state["path_embeddings"]:
            state["scored_paths"] = []
            state["processing_log"].append("Module3: No paths to score")
            return state

        # Module3: vector prune の適用
        if not state["config"].use_vector_prune:
            # vector pruneを使わない場合は、全てのパスをそのまま通す
            state["scored_paths"] = [
                (i, 0.0) for i in range(len(state["candidate_paths"]))
            ]
            state["processing_log"].append("Module3: Skipping vector pruning")
            return state

        # vector pruneを適用
        max_paths = state["config"].max_paths
        if state["config"].use_subquery_split:
            # サブクエリ使用時はより厳しい制限
            max_paths = min(max_paths, 30)

        try:
            scored = score_paths(
                query_vecs=state["query_embeddings"],
                path_vectors=state["path_embeddings"],
                top_k=max_paths,
            )
            state["scored_paths"] = scored
            state["processing_log"].append(
                f"Module3: Vector pruning applied - selected top {len(scored)} paths"
            )

        except Exception as e:
            state["processing_log"].append(f"Module3: Error in vector scoring: {e}")
            state["scored_paths"] = []

        return state

    def llm_reranking(self, state: PipelineState) -> PipelineState:
        """Module4: LLMによる再ランキング（パスをリストワイズでLLMに入力しtop1のパスを出力）"""
        if not state["config"].use_llm_reranker:
            state["reranked_paths"] = None
            state["processing_log"].append("Module4: Skipping LLM reranking")
            return state

        if not state["scored_paths"]:
            state["reranked_paths"] = []
            state["processing_log"].append("Module4: No scored paths for reranking")
            return state

        self._init_llm_components()

        try:
            # スコア順の上位パスを取得
            top_paths = [
                state["candidate_paths"][idx] for idx, _ in state["scored_paths"][:20]
            ]

            # クエリリストの準備
            if state["config"].use_subquery_split and state["subqueries"]:
                # サブクエリ分割が使用されている場合
                query_list = state["subqueries"]
                state["processing_log"].append(
                    f"Module4: Using {len(query_list)} subqueries for reranking"
                )
                for i, subquery in enumerate(query_list, 1):
                    state["processing_log"].append(f"  Subquery {i}: {subquery}")
            else:
                # 元の質問を使用
                query_list = [state["question"]]
                state["processing_log"].append(
                    "Module4: Using original question for reranking"
                )

            # Module4: LLMで再ランキング（クエリリストと上位パスを入力）
            rerank_result = self.path_reranker.invoke(
                queries=query_list, paths=top_paths  # サブクエリリストまたは元の質問
            )

            # RerankerResponseからreranked_combosを取得
            if hasattr(rerank_result, "reranked_combos"):
                state["reranked_paths"] = [rerank_result.reranked_combos]  # 最良パス1つ
                state["processing_log"].append("Module4: LLM reranking applied")
                state["processing_log"].append(
                    f"Module4: Selected top1 path: {rerank_result.reranked_combos}"
                )
                state["processing_log"].append(
                    f"Module4: Reranking based on {len(query_list)} queries and {len(top_paths)} paths"
                )
            else:
                state["reranked_paths"] = []
                state["processing_log"].append(
                    "Module4: LLM reranking failed - no reranked_combos"
                )

        except Exception as e:
            state["processing_log"].append(f"Module4: Error in LLM reranking: {e}")
            state["reranked_paths"] = []

        return state

    def final_processing(self, state: PipelineState) -> PipelineState:
        """最終処理と結果生成"""
        config = state["config"]

        # 最終パスの決定（一つのパスのみ選定）
        final_paths = []

        if config.use_llm_reranker and state["reranked_paths"]:
            # LLM rerankerが有効な場合、最良の1パスを選定
            final_paths = [state["reranked_paths"][0]]  # 最良パス1つのみ
            state["processing_log"].append(
                "Final path selection: Using LLM reranked top path"
            )

        elif state["scored_paths"]:
            # vector pruningのスコア上位1パスを使用
            best_idx, best_score = state["scored_paths"][0]  # 最高スコアの1パス
            final_paths = [state["candidate_paths"][best_idx]]
            state["processing_log"].append(
                f"Final path selection: Using top vector-scored path (score: {best_score:.4f})"
            )

        elif state["candidate_paths"]:
            # 候補パスがある場合は最初の1つを使用
            final_paths = [state["candidate_paths"][0]]
            state["processing_log"].append(
                "Final path selection: Using first candidate path"
            )

        else:
            # パスが見つからない場合
            final_paths = []
            state["processing_log"].append("Final path selection: No paths found")

        state["final_paths"] = final_paths

        # 到達可能エンティティの取得（1つのパスのみ処理）
        all_reachable = []

        if final_paths:
            selected_path = final_paths[0]
            state["processing_log"].append(
                f"Getting reachable entities for selected path: {selected_path}"
            )

            try:
                reachable = self.graph_finder.get_reachable_entities(
                    entity_name=state["entity_name"],
                    rel_types=selected_path,
                    distinct=True,
                    no_cycle=True,
                )
                all_reachable = reachable

                state["processing_log"].append(
                    f"Selected path found {len(reachable)} reachable entities"
                )
                if reachable:
                    state["processing_log"].append(
                        f"Sample entities from selected path: {reachable[:5]}"
                    )

            except Exception as e:
                state["processing_log"].append(
                    f"Error getting reachable entities for selected path {selected_path}: {e}"
                )
                all_reachable = []

        # エンティティリストを設定（重複除去は1つのパスなので不要だが念のため）
        state["reachable_entities"] = list(dict.fromkeys(all_reachable))

        # 結果サマリー
        selected_path_str = str(final_paths[0]) if final_paths else "None"
        state["processing_log"].append(
            f"Final result: 1 selected path ({selected_path_str}), {len(state['reachable_entities'])} reachable entities"
        )

        return state

    def build_graph(self) -> StateGraph:
        """LangGraphの構築"""
        workflow = StateGraph(PipelineState)

        # ノードの追加
        workflow.add_node("input_processing", self.input_processing)
        workflow.add_node("subquery_generation", self.subquery_generation)
        workflow.add_node("path_discovery", self.path_discovery)
        workflow.add_node("query_embedding", self.query_embedding)
        workflow.add_node("vector_scoring", self.vector_scoring)
        workflow.add_node("llm_reranking", self.llm_reranking)
        workflow.add_node("final_processing", self.final_processing)

        # エッジの定義（線形フロー）
        workflow.add_edge(START, "input_processing")
        workflow.add_edge("input_processing", "subquery_generation")
        workflow.add_edge("subquery_generation", "path_discovery")
        workflow.add_edge("path_discovery", "query_embedding")
        workflow.add_edge("query_embedding", "vector_scoring")
        workflow.add_edge("vector_scoring", "llm_reranking")
        workflow.add_edge("llm_reranking", "final_processing")
        workflow.add_edge("final_processing", END)

        return workflow.compile()


# ────────────────────────────────────────────────────────────────
# 設定更新（single pathに最適化）
# ────────────────────────────────────────────────────────────────
def get_pipeline_config(
    pattern: Literal[1, 2, 3, 4], hop: Optional[int] = None
) -> PipelineConfig:
    """4パターンの設定を返す（1つのパス選定に最適化）

    Parameters
    ----------
    pattern : Literal[1, 2, 3, 4]
        パイプラインパターン
    hop : Optional[int]
        hop数の指定。Noneの場合はデフォルト値を使用

    Module説明:
    1. type path: 分割された質問が期待するエンティティのtype
    2. サブクエリ分割: 質問を単一ホップ質問に分解する
    3. vector prune: 質問群とリレーションとのコサイン類似度による剪定
    4. LLM reranker: パスをリストワイズでLLMに入力しtop1のパスを出力

    Pipeline 1: サブクエリ分割とvector pruneを利用した初歩的な分割 (×○○×)
    Pipeline 2: type剪定の有無による検証 (×○○○)
    Pipeline 3: LLM Rerankerの有無による検証 (○○○×)
    Pipeline 4: 全部入り (○○○○)
    """
    configs = {
        1: PipelineConfig(
            use_type_pruning=False,  # × Module1: type path
            use_subquery_split=True,  # ○ Module2: サブクエリ分割
            use_vector_prune=True,  # ○ Module3: vector prune
            use_llm_reranker=False,  # × Module4: LLM reranker
            hop_count=hop if hop is not None else None,
            max_paths=50,
            top_k=1,  # 1つのパスのみ
        ),
        2: PipelineConfig(
            use_type_pruning=False,  # × Module1: type path
            use_subquery_split=True,  # ○ Module2: サブクエリ分割
            use_vector_prune=True,  # ○ Module3: vector prune
            use_llm_reranker=True,  # ○ Module4: LLM reranker
            hop_count=hop if hop is not None else None,
            max_paths=30,
            top_k=1,  # 1つのパスのみ
        ),
        3: PipelineConfig(
            use_type_pruning=True,  # ○ Module1: type path
            use_subquery_split=True,  # ○ Module2: サブクエリ分割
            use_vector_prune=True,  # ○ Module3: vector prune
            use_llm_reranker=False,  # × Module4: LLM reranker
            hop_count=hop if hop is not None else None,
            max_paths=50,
            top_k=1,  # 1つのパスのみ
        ),
        4: PipelineConfig(
            use_type_pruning=True,  # ○ Module1: type path
            use_subquery_split=True,  # ○ Module2: サブクエリ分割
            use_vector_prune=True,  # ○ Module3: vector prune
            use_llm_reranker=True,  # ○ Module4: LLM reranker
            hop_count=hop if hop is not None else None,
            max_paths=50,
            top_k=1,  # 1つのパスのみ
        ),
    }
    return configs[pattern]


# ────────────────────────────────────────────────────────────────
# 単一パス選定テスト
# ────────────────────────────────────────────────────────────────
def test_single_path_selection():
    """単一パス選定の動作テスト"""
    pipeline = GraphQAPipeline()
    graph = pipeline.build_graph()

    test_cases = [
        {
            "question": "which person directed the movies starred by John Krasinski",
            "entity_name": "John Krasinski",
        },
        {
            "question": "what movies did Jack Nicholson appear in",
            "entity_name": "Jack Nicholson",
        },
    ]

    print("=" * 80)
    print("SINGLE PATH SELECTION TEST")
    print("=" * 80)

    for test_case in test_cases:
        print(f"\nTest Case: {test_case['question']}")
        print(f"Entity: {test_case['entity_name']}")
        print("-" * 60)

        for pattern in [1, 2, 3, 4]:
            print(f"\n--- Pipeline {pattern} ---")

            config = get_pipeline_config(pattern)
            state = PipelineState(
                question=test_case["question"],
                entity_name=test_case["entity_name"],
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

            try:
                import time

                start_time = time.time()
                result = graph.invoke(state)
                execution_time = time.time() - start_time

                print(
                    f"Configuration: {pattern} ({config.use_type_pruning}, {config.use_subquery_split}, {config.use_vector_prune}, {config.use_llm_reranker})"
                )
                print(f"Execution Time: {execution_time:.2f}s")

                # 単一パスの確認
                final_paths_count = len(result["final_paths"])
                print(f"Final Paths Count: {final_paths_count} (should be 1)")

                if final_paths_count == 1:
                    selected_path = result["final_paths"][0]
                    entity_count = len(result["reachable_entities"])

                    print(f"✓ Selected Path: {selected_path}")
                    print(f"✓ Reachable Entities: {entity_count}")

                    if result["reachable_entities"]:
                        print(f"✓ Sample Entities: {result['reachable_entities'][:5]}")

                elif final_paths_count == 0:
                    print("✗ No path selected")
                else:
                    print(f"✗ Multiple paths selected ({final_paths_count})")

                # パス選定関連のログを表示
                selection_logs = [
                    log
                    for log in result["processing_log"]
                    if "Final path selection" in log or "Selected path" in log
                ]
                if selection_logs:
                    print("Path Selection Log:")
                    for log in selection_logs:
                        print(f"  {log}")

            except Exception as e:
                print(f"✗ Error in Pipeline {pattern}: {e}")

        print("\n" + "=" * 60)


def run_single_path_comparison():
    """単一パス選定での4パターン比較"""
    print("=" * 80)
    print("SINGLE PATH PIPELINE COMPARISON")
    print("=" * 80)

    pipeline = GraphQAPipeline()
    graph = pipeline.build_graph()

    test_question = "which person directed the movies starred by John Krasinski"
    test_entity = "John Krasinski"

    print(f"Question: {test_question}")
    print(f"Entity: {test_entity}")
    print("-" * 80)

    results_summary = []

    for pattern in [1, 2, 3, 4]:
        config = get_pipeline_config(pattern)
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

        try:
            import time

            start_time = time.time()
            result = graph.invoke(state)
            execution_time = time.time() - start_time

            selected_path = result["final_paths"][0] if result["final_paths"] else None
            entity_count = len(result["reachable_entities"])

            results_summary.append(
                {
                    "pattern": pattern,
                    "time": execution_time,
                    "path": selected_path,
                    "entities": entity_count,
                    "success": selected_path is not None,
                }
            )

        except Exception as e:
            results_summary.append(
                {
                    "pattern": pattern,
                    "time": 0,
                    "path": None,
                    "entities": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # 結果比較表
    print(
        f"{'Pipeline':<10} {'Time(s)':<8} {'Entities':<10} {'Selected Path':<30} {'Success':<8}"
    )
    print("-" * 80)

    for r in results_summary:
        success_mark = "✓" if r["success"] else "✗"
        path_str = str(r["path"])[:30] if r["path"] else "None"
        print(
            f"{r['pattern']:<10} {r['time']:<8.2f} {r['entities']:<10} {path_str:<30} {success_mark:<8}"
        )

    # 成功したパイプラインの分析
    successful = [r for r in results_summary if r["success"]]
    if successful:
        best_entities = max(successful, key=lambda x: x["entities"])
        fastest = min(successful, key=lambda x: x["time"])

        print(f"\nAnalysis:")
        print(
            f"  Most Entities: Pipeline {best_entities['pattern']} ({best_entities['entities']} entities)"
        )
        print(f"  Fastest: Pipeline {fastest['pattern']} ({fastest['time']:.2f}s)")
        print(f"  Success Rate: {len(successful)}/4 pipelines")


if __name__ == "__main__":
    # 単一パス選定テスト
    test_single_path_selection()

    print("\n" + "=" * 80)

    # 単一パス比較
    run_single_path_comparison()

# python pipeline/pipeline.py
