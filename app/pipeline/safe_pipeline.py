"""
SAFE (Semantic-Aware Subgraph Retrieval Framework) Pipeline

論文に基づく実装:
1. ADJ (Approximate Distance Join) - スキーマレベル探索 (Algorithm 1)
2. Ranked Semantic Subgraph Matching - インスタンスレベル探索 (Algorithm 2)
"""

from __future__ import annotations

import heapq
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from core.config import BASEMODEL, get_settings
from database.search import GraphPathFinder


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PseudoEdge:
    """擬似クエリグラフのエッジ"""
    src_node: str  # ソースノード名
    src_type: str  # ソースノードタイプ
    relation: str  # リレーション
    tgt_node: str  # ターゲットノード名
    tgt_type: str  # ターゲットノードタイプ
    is_anchor: bool = False  # アンカー（既知エンティティ）かどうか


@dataclass
class SchemaEdge:
    """スキーマエッジ"""
    src_type: str
    relation: str
    tgt_type: str
    embedding: Optional[np.ndarray] = None

    def to_text(self) -> str:
        return f"{self.src_type} {self.relation} {self.tgt_type}"


@dataclass
class CandidateSchemaEdge:
    """候補スキーマエッジ（距離付き）"""
    schema_edge: SchemaEdge
    distance: float  # L2距離


@dataclass
class QueryGraph:
    """補正されたクエリグラフ"""
    edges: List[Tuple[SchemaEdge, PseudoEdge]]  # (スキーマエッジ, 元の擬似エッジ)
    total_distance: float  # 合計意味的距離


@dataclass
class MatchedSubgraph:
    """マッチしたサブグラフ"""
    nodes: Dict[str, str]  # query_node -> kg_entity
    edges: List[Tuple[str, str, str]]  # (src, rel, tgt)
    score: float  # 累積スコア（低いほど良い）


@dataclass
class SAFEResult:
    """SAFE結果"""
    question: str
    pseudo_edges: List[PseudoEdge]
    candidate_query_graphs: List[QueryGraph]
    best_query_graph: Optional[QueryGraph]
    matched_subgraphs: List[MatchedSubgraph]
    answer_entities: List[str]
    processing_log: List[str] = field(default_factory=list)


# =============================================================================
# Pydantic Models for LLM
# =============================================================================

class PseudoEdgeSchema(BaseModel):
    """擬似エッジのスキーマ"""
    src_node: str = Field(description="Source node name (use '?' for unknown)")
    src_type: str = Field(description="Source node type")
    relation: str = Field(description="Relationship name in natural language")
    tgt_node: str = Field(description="Target node name (use '?' for unknown/answer)")
    tgt_type: str = Field(description="Target node type")
    is_anchor: bool = Field(default=False, description="True if node is a known entity")


class PseudoQueryGraphResponse(BaseModel):
    """LLMによる擬似クエリグラフ生成"""
    edges: List[PseudoEdgeSchema] = Field(
        default=[],
        description="List of edges representing the query structure"
    )


# =============================================================================
# Schema Graph with APSP
# =============================================================================

class SchemaGraphWithAPSP:
    """全点対最短経路(APSP)を持つスキーマグラフ"""

    def __init__(self):
        self.edges: List[SchemaEdge] = []
        self.types: Set[str] = set()
        self.type_to_idx: Dict[str, int] = {}
        self.idx_to_type: Dict[int, str] = {}
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.apsp: Optional[np.ndarray] = None  # All-Pairs Shortest Path距離行列
        self.embeddings: Optional[Any] = None  # OpenAI Embeddings

    def add_edge(self, src_type: str, relation: str, tgt_type: str):
        """エッジ追加"""
        edge = SchemaEdge(src_type, relation, tgt_type)
        self.edges.append(edge)
        self.types.add(src_type)
        self.types.add(tgt_type)
        # 無向グラフとして追加
        self.adjacency[src_type].append((tgt_type, relation))
        if src_type != tgt_type:
            self.adjacency[tgt_type].append((src_type, relation))

    def build_apsp(self):
        """Floyd-Warshallで全点対最短経路を計算"""
        n = len(self.types)
        type_list = sorted(self.types)
        self.type_to_idx = {t: i for i, t in enumerate(type_list)}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}

        # 距離行列を初期化（無限大）
        INF = float('inf')
        dist = np.full((n, n), INF)

        # 対角成分は0
        for i in range(n):
            dist[i, i] = 0

        # 隣接エッジの距離は1（自己参照は除く、対角は0を維持）
        for src_type, neighbors in self.adjacency.items():
            src_idx = self.type_to_idx[src_type]
            for tgt_type, _ in neighbors:
                tgt_idx = self.type_to_idx[tgt_type]
                if src_idx != tgt_idx:  # 自己参照エッジは距離0を維持
                    dist[src_idx, tgt_idx] = 1

        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        self.apsp = dist

    def get_type_distance(self, type1: str, type2: str) -> float:
        """2つのタイプ間の最短距離を取得（O(1)）"""
        if type1 not in self.type_to_idx or type2 not in self.type_to_idx:
            return float('inf')
        i = self.type_to_idx[type1]
        j = self.type_to_idx[type2]
        return self.apsp[i, j]

    def get_edge_distance(self, edge1: SchemaEdge, edge2: SchemaEdge) -> float:
        """2つのエッジ間のチェーン接続距離（edge1.tgt → edge2.src）

        チェーン接続のみをチェック:
        - d1 = 0: 直接接続 (edge1.tgt_type == edge2.src_type)
        - d1 = 1: 1ノードを介した接続
        """
        return self.get_type_distance(edge1.tgt_type, edge2.src_type)

    def compute_edge_embeddings(self, embeddings):
        """全エッジの埋め込みを計算"""
        self.embeddings = embeddings
        texts = [e.to_text() for e in self.edges]
        if texts:
            vectors = embeddings.embed_documents(texts)
            for i, edge in enumerate(self.edges):
                edge.embedding = np.array(vectors[i])


def build_schema_with_apsp() -> SchemaGraphWithAPSP:
    """スキーマグラフをAPSP付きで構築"""
    from dataset_construction.schema_v2 import SCHEMA_GRAPH

    schema = SchemaGraphWithAPSP()
    for src, rel, tgt, _ in SCHEMA_GRAPH:
        schema.add_edge(src, rel, tgt)

    schema.build_apsp()
    return schema


# =============================================================================
# SAFE Pipeline
# =============================================================================

class SAFEPipeline:
    """SAFEパイプライン"""

    def __init__(
        self,
        model: str = BASEMODEL,
        embedding_model: str = "text-embedding-3-small",
        k_schema: int = 3,  # 候補スキーマエッジ数
        k_qg: int = 5,  # 候補クエリグラフ数
        k_retrieval: int = 10,  # 検索結果数
        delta: int = 1,  # エッジ距離閾値
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings

        self.llm = init_chat_model(model, model_provider="openai", temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.schema = build_schema_with_apsp()
        self.schema.compute_edge_embeddings(self.embeddings)

        self.finder = GraphPathFinder()

        self.k_schema = k_schema
        self.k_qg = k_qg
        self.k_retrieval = k_retrieval
        self.delta = delta

    def run(self, question: str, entity_name: Optional[str] = None) -> SAFEResult:
        """パイプライン実行"""
        log = []
        log.append(f"Question: {question}")

        # Step 1: LLMで擬似クエリグラフを生成
        log.append("Phase 1: Generate Pseudo Query Graph")
        pseudo_edges = self._generate_pseudo_query_graph(question, entity_name)
        log.append(f"  Generated {len(pseudo_edges)} pseudo edges")

        if not pseudo_edges:
            return SAFEResult(
                question=question,
                pseudo_edges=[],
                candidate_query_graphs=[],
                best_query_graph=None,
                matched_subgraphs=[],
                answer_entities=[],
                processing_log=log,
            )

        # Step 2: ADJ - スキーマレベル補正
        log.append("Phase 2: ADJ (Approximate Distance Join)")
        candidate_qgs = self._approximate_distance_join(pseudo_edges)
        log.append(f"  Found {len(candidate_qgs)} candidate query graphs")

        if not candidate_qgs:
            return SAFEResult(
                question=question,
                pseudo_edges=pseudo_edges,
                candidate_query_graphs=[],
                best_query_graph=None,
                matched_subgraphs=[],
                answer_entities=[],
                processing_log=log,
            )

        best_qg = candidate_qgs[0]
        log.append(f"  Best QG distance: {best_qg.total_distance:.4f}")

        # Step 3: Ranked Semantic Subgraph Matching
        log.append("Phase 3: Subgraph Matching")
        matched_subgraphs = self._subgraph_matching(best_qg, pseudo_edges)
        log.append(f"  Found {len(matched_subgraphs)} matched subgraphs")

        # 回答エンティティを抽出
        answer_entities = self._extract_answers(matched_subgraphs, pseudo_edges)
        log.append(f"  Extracted {len(answer_entities)} answer entities")

        return SAFEResult(
            question=question,
            pseudo_edges=pseudo_edges,
            candidate_query_graphs=candidate_qgs,
            best_query_graph=best_qg,
            matched_subgraphs=matched_subgraphs,
            answer_entities=answer_entities,
            processing_log=log,
        )

    def _generate_pseudo_query_graph(
        self,
        question: str,
        entity_name: Optional[str] = None
    ) -> List[PseudoEdge]:
        """LLMで擬似クエリグラフを生成"""

        type_list = ", ".join(sorted(self.schema.types))

        prompt = f"""Convert this question into a query graph structure.

Question: {question}
{"Known entity: " + entity_name if entity_name else ""}

Available node types: {type_list}

Generate edges that represent the query structure. Each edge should have:
- src_node: source node name (use "?" for unknown)
- src_type: source node type
- relation: relationship name (use natural language)
- tgt_node: target node name (use "?" for unknown/answer)
- tgt_type: target node type
- is_anchor: true if the node is a known entity, false otherwise

Example for "What diseases are associated with BRCA1?":
[
  {{"src_node": "BRCA1", "src_type": "gene/protein", "relation": "associated with", "tgt_node": "?", "tgt_type": "disease", "is_anchor": true}}
]

Return JSON with "edges" key containing the list."""

        llm_with_output = self.llm.with_structured_output(PseudoQueryGraphResponse)

        try:
            result = llm_with_output.invoke(prompt)
            edges = []
            for e in result.edges:
                edges.append(PseudoEdge(
                    src_node=e.src_node or "?",
                    src_type=e.src_type or "",
                    relation=e.relation or "",
                    tgt_node=e.tgt_node or "?",
                    tgt_type=e.tgt_type or "",
                    is_anchor=e.is_anchor or (entity_name and e.src_node == entity_name),
                ))
            return edges
        except Exception as e:
            print(f"Error generating pseudo query graph: {e}")
            # フォールバック: 単純なエッジを生成
            if entity_name:
                return [PseudoEdge(
                    src_node=entity_name,
                    src_type="",
                    relation="related to",
                    tgt_node="?",
                    tgt_type="",
                    is_anchor=True,
                )]
            return []

    def _approximate_distance_join(
        self,
        pseudo_edges: List[PseudoEdge]
    ) -> List[QueryGraph]:
        """ADJ: スキーマレベルでの補正 (Algorithm 1)"""

        # Step 1: 各擬似エッジに対して候補スキーマエッジを選定
        candidates_per_edge: List[List[CandidateSchemaEdge]] = []

        for p_edge in pseudo_edges:
            candidates = self._find_candidate_schema_edges(p_edge)
            candidates_per_edge.append(candidates)

        if not candidates_per_edge or not all(candidates_per_edge):
            return []

        # Step 2: 距離によるエッジ結合
        # 全組み合わせを探索し、エッジ間距離が閾値以下のものを選択
        query_graphs = self._match_sg(pseudo_edges, candidates_per_edge, 0, [], 0.0)

        # スコア順にソート
        query_graphs.sort(key=lambda qg: qg.total_distance)

        return query_graphs[:self.k_qg]

    def _find_candidate_schema_edges(
        self,
        pseudo_edge: PseudoEdge
    ) -> List[CandidateSchemaEdge]:
        """候補スキーマエッジを検索"""

        # 擬似エッジをテキスト化してベクトル化
        pseudo_text = f"{pseudo_edge.src_type} {pseudo_edge.relation} {pseudo_edge.tgt_type}"
        pseudo_vec = np.array(self.embeddings.embed_query(pseudo_text))

        # 全スキーマエッジとの距離を計算
        distances = []
        for schema_edge in self.schema.edges:
            if schema_edge.embedding is not None:
                dist = np.linalg.norm(pseudo_vec - schema_edge.embedding)
                distances.append((schema_edge, dist))

        # 距離順にソート
        distances.sort(key=lambda x: x[1])

        # 上位k_schema個を返す
        return [
            CandidateSchemaEdge(schema_edge=se, distance=d)
            for se, d in distances[:self.k_schema]
        ]

    def _match_sg(
        self,
        pseudo_edges: List[PseudoEdge],
        candidates_per_edge: List[List[CandidateSchemaEdge]],
        idx: int,
        current_matches: List[Tuple[SchemaEdge, PseudoEdge]],
        current_distance: float,
    ) -> List[QueryGraph]:
        """再帰的にスキーマエッジをマッチング"""

        # 全エッジをマッチし終わったら結果を返す
        if idx >= len(pseudo_edges):
            return [QueryGraph(
                edges=current_matches.copy(),
                total_distance=current_distance,
            )]

        results = []

        for candidate in candidates_per_edge[idx]:
            # エッジ間距離をチェック
            if current_matches:
                last_schema_edge = current_matches[-1][0]
                edge_dist = self.schema.get_edge_distance(last_schema_edge, candidate.schema_edge)
                if edge_dist > self.delta:
                    continue  # 閾値を超えたらスキップ

            # 再帰的に次のエッジをマッチ
            new_matches = current_matches + [(candidate.schema_edge, pseudo_edges[idx])]
            new_distance = current_distance + candidate.distance

            sub_results = self._match_sg(
                pseudo_edges,
                candidates_per_edge,
                idx + 1,
                new_matches,
                new_distance,
            )
            results.extend(sub_results)

        return results

    def _subgraph_matching(
        self,
        query_graph: QueryGraph,
        pseudo_edges: List[PseudoEdge]
    ) -> List[MatchedSubgraph]:
        """Ranked Semantic Subgraph Matching (Algorithm 2)"""

        # アンカーノードを全て特定
        anchors = []
        for schema_edge, pseudo_edge in query_graph.edges:
            if pseudo_edge.src_node != "?" and pseudo_edge.src_node not in [a[0] for a in anchors]:
                anchors.append((pseudo_edge.src_node, schema_edge.src_type, schema_edge))

        # 複数アンカーがある場合は交差クエリ（独自拡張、元のSAFEにはない）
        # if len(anchors) >= 2:
        #     return self._intersection_search(query_graph, anchors)

        # 単一アンカーの場合
        if not anchors:
            return []

        anchor_name, anchor_type, _ = anchors[0]

        # マルチホップ検索
        matched_subgraphs = self._multi_hop_search(
            query_graph,
            anchor_name,
            anchor_type,
        )

        # スコア順にソート
        matched_subgraphs.sort(key=lambda sg: sg.score)

        # 全てのサブグラフを返す（k_retrievalはパス選択のみに使用）
        return matched_subgraphs

    def _intersection_search(
        self,
        query_graph: QueryGraph,
        anchors: List[tuple],
    ) -> List[MatchedSubgraph]:
        """交差クエリを検索（複数アンカーからの共通ターゲット）"""

        graph = self.finder.graph

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        # 各アンカーからのターゲットセットを取得
        anchor_targets = []

        for anchor_name, anchor_type, schema_edge in anchors:
            cypher = f"""
            MATCH (a:{get_label(schema_edge.src_type)})-[r:{schema_edge.relation}]->(tgt:{get_label(schema_edge.tgt_type)})
            WHERE a.name = $anchor_name
            RETURN DISTINCT tgt.name AS target
            """

            try:
                records = graph.run(cypher, anchor_name=anchor_name).data()
                targets = {r["target"] for r in records}
                anchor_targets.append((anchor_name, targets))
            except Exception as e:
                print(f"Intersection search error for {anchor_name}: {e}")
                anchor_targets.append((anchor_name, set()))

        # 共通ターゲットを計算
        if not anchor_targets:
            return []

        common_targets = anchor_targets[0][1]
        for _, targets in anchor_targets[1:]:
            common_targets = common_targets & targets

        # 結果を構築（全ての共通ターゲットを返す）
        results = []
        for target in common_targets:
            nodes = {"?": target}
            for anchor_name, _ in anchor_targets:
                nodes[anchor_name] = anchor_name

            subgraph = MatchedSubgraph(
                nodes=nodes,
                edges=[(anchor_name, "related_to", target) for anchor_name, _ in anchor_targets],
                score=0.0,
            )
            results.append(subgraph)

        return results

    def _multi_hop_search(
        self,
        query_graph: QueryGraph,
        anchor_name: str,
        anchor_type: str,
    ) -> List[MatchedSubgraph]:
        """マルチホップパスを検索"""

        graph = self.finder.graph

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        edges = query_graph.edges

        if len(edges) == 1:
            # 1-hop: 単純な検索
            return self._dfs_search(query_graph, anchor_name, anchor_type)

        elif len(edges) == 2:
            # 2-hop: チェーンパターン
            se1, pe1 = edges[0]
            se2, pe2 = edges[1]

            results = []

            # チェーン接続距離をチェック
            chain_dist = self.schema.get_edge_distance(se1, se2)

            if chain_dist == 0:
                # 直接接続: se1.tgt_type == se2.src_type
                cypher = f"""
                MATCH (a:{get_label(se1.src_type)})-[r1:{se1.relation}]->(mid:{get_label(se1.tgt_type)})-[r2:{se2.relation}]->(ans:{get_label(se2.tgt_type)})
                WHERE a.name = $anchor_name AND a <> mid AND mid <> ans AND a <> ans
                RETURN DISTINCT
                    a.name AS anchor,
                    type(r1) AS rel1,
                    mid.name AS mid_node,
                    type(r2) AS rel2,
                    ans.name AS answer
                LIMIT 100
                """

                try:
                    records = graph.run(cypher, anchor_name=anchor_name).data()

                    for record in records:
                        subgraph = MatchedSubgraph(
                            nodes={
                                pe1.src_node: record["anchor"],
                                pe1.tgt_node: record["mid_node"],
                                pe2.tgt_node: record["answer"],
                            },
                            edges=[
                                (record["anchor"], record["rel1"], record["mid_node"]),
                                (record["mid_node"], record["rel2"], record["answer"]),
                            ],
                            score=0.0,
                        )
                        results.append(subgraph)

                except Exception as e:
                    print(f"Multi-hop search error (direct): {e}")

            elif chain_dist == 1:
                # 1ノード介在: se1.tgt_type と se2.src_type の間に1ノード
                # 中間ノードを含む3ホップCypherを生成
                cypher = f"""
                MATCH (a:{get_label(se1.src_type)})-[r1:{se1.relation}]->(mid1:{get_label(se1.tgt_type)})-[r_bridge]-(mid2:{get_label(se2.src_type)})-[r2:{se2.relation}]->(ans:{get_label(se2.tgt_type)})
                WHERE a.name = $anchor_name
                    AND a <> mid1 AND mid1 <> mid2 AND mid2 <> ans AND a <> ans
                RETURN DISTINCT
                    a.name AS anchor,
                    type(r1) AS rel1,
                    mid1.name AS mid1_node,
                    type(r_bridge) AS rel_bridge,
                    mid2.name AS mid2_node,
                    type(r2) AS rel2,
                    ans.name AS answer
                LIMIT 100
                """

                try:
                    records = graph.run(cypher, anchor_name=anchor_name).data()

                    for record in records:
                        subgraph = MatchedSubgraph(
                            nodes={
                                pe1.src_node: record["anchor"],
                                pe1.tgt_node: record["mid1_node"],
                                "bridge": record["mid2_node"],
                                pe2.tgt_node: record["answer"],
                            },
                            edges=[
                                (record["anchor"], record["rel1"], record["mid1_node"]),
                                (record["mid1_node"], record["rel_bridge"], record["mid2_node"]),
                                (record["mid2_node"], record["rel2"], record["answer"]),
                            ],
                            score=0.0,
                        )
                        results.append(subgraph)

                except Exception as e:
                    print(f"Multi-hop search error (bridged): {e}")

            return results

        else:
            # 3ホップ以上: 単純化のため各エッジを個別に検索
            return self._dfs_search(query_graph, anchor_name, anchor_type)

    def _dfs_search(
        self,
        query_graph: QueryGraph,
        anchor_name: str,
        anchor_type: str,
    ) -> List[MatchedSubgraph]:
        """DFSでサブグラフを検索"""

        graph = self.finder.graph

        results = []

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        # クエリグラフの各エッジについて検索
        for schema_edge, pseudo_edge in query_graph.edges:
            # アンカーがsrc側かtgt側かを判定
            # 方向ありで検索（src -> tgt の方向）
            queries = []

            # Case 1: anchor is source（順方向）
            cypher1 = f"""
            MATCH (src:{get_label(schema_edge.src_type)})-[r:{schema_edge.relation}]->(tgt:{get_label(schema_edge.tgt_type)})
            WHERE src.name = $anchor_name
            RETURN DISTINCT
                src.name AS src_name,
                type(r) AS rel_type,
                tgt.name AS tgt_name
            LIMIT 50
            """
            queries.append(("src", cypher1))

            # Case 2: anchor is target（逆方向検索）
            cypher2 = f"""
            MATCH (src:{get_label(schema_edge.src_type)})-[r:{schema_edge.relation}]->(tgt:{get_label(schema_edge.tgt_type)})
            WHERE tgt.name = $anchor_name
            RETURN DISTINCT
                src.name AS src_name,
                type(r) AS rel_type,
                tgt.name AS tgt_name
            LIMIT 50
            """
            queries.append(("tgt", cypher2))

            for anchor_pos, cypher in queries:
                try:
                    records = graph.run(cypher, anchor_name=anchor_name).data()

                    for record in records:
                        # スコア計算（簡略化: 0で固定）
                        if anchor_pos == "src":
                            nodes = {
                                pseudo_edge.src_node: record["src_name"],
                                pseudo_edge.tgt_node: record["tgt_name"],
                            }
                        else:
                            nodes = {
                                pseudo_edge.src_node: record["tgt_name"],
                                pseudo_edge.tgt_node: record["src_name"],
                            }

                        subgraph = MatchedSubgraph(
                            nodes=nodes,
                            edges=[(record["src_name"], record["rel_type"], record["tgt_name"])],
                            score=0.0,
                        )
                        results.append(subgraph)

                except Exception as e:
                    print(f"DFS search error: {e}")

        return results

    def _extract_answers(
        self,
        subgraphs: List[MatchedSubgraph],
        pseudo_edges: List[PseudoEdge]
    ) -> List[str]:
        """回答エンティティを抽出"""

        # 回答ノード（"?"で表される）を特定
        answer_node_names = set()
        for p_edge in pseudo_edges:
            if p_edge.tgt_node == "?":
                answer_node_names.add(p_edge.tgt_node)
            if p_edge.src_node == "?":
                answer_node_names.add(p_edge.src_node)

        # サブグラフから回答を抽出
        answers = []
        for sg in subgraphs:
            for query_node, kg_entity in sg.nodes.items():
                if query_node in answer_node_names or query_node == "?":
                    if kg_entity and kg_entity not in answers:
                        answers.append(kg_entity)

        return answers


# =============================================================================
# Main
# =============================================================================

def main():
    """テスト実行"""
    import argparse

    parser = argparse.ArgumentParser(description="SAFE Pipeline Test")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    pipeline = SAFEPipeline()
    result = pipeline.run(args.question, args.entity)

    print("\n" + "=" * 60)
    print("SAFE Pipeline Result")
    print("=" * 60)

    for log_line in result.processing_log:
        print(log_line)

    print("\n--- Pseudo Edges ---")
    for e in result.pseudo_edges:
        print(f"  ({e.src_node}:{e.src_type})-[{e.relation}]->({e.tgt_node}:{e.tgt_type})")

    if result.best_query_graph:
        print("\n--- Best Query Graph ---")
        for se, pe in result.best_query_graph.edges:
            print(f"  ({se.src_type})-[{se.relation}]->({se.tgt_type})")

    print("\n--- Answers ---")
    print(f"  {result.answer_entities[:10]}")


if __name__ == "__main__":
    main()
