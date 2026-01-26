"""
SAFE (Semantic-Aware Subgraph Retrieval Framework) Pipeline

論文に基づく実装:
1. ADJ (Approximate Distance Join) - スキーマレベル探索 (Algorithm 1)
2. Ranked Semantic Subgraph Matching - インスタンスレベル探索 (Algorithm 2)
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np

from core.config import BASEMODEL, get_settings
from database.search import GraphPathFinder
from pipeline.safe.models import (
    PseudoEdge,
    SchemaEdge,
    CandidateSchemaEdge,
    QueryGraph,
    MatchedSubgraph,
    SAFEResult,
    PseudoQueryGraphResponse,
)
from pipeline.safe.schema import SchemaGraphWithAPSP, build_schema_with_apsp


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
        kg_type: str = "primekgqa",
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings

        self.llm = init_chat_model(model, model_provider="openai", temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.kg_type = kg_type
        self.schema = build_schema_with_apsp(kg_type)
        self.schema.compute_edge_embeddings(self.embeddings)

        self.finder = GraphPathFinder(kg_type=kg_type)

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

    def _get_prompt_examples(self) -> str:
        """KGタイプに応じたプロンプト例を返す"""
        if self.kg_type == "metaqa":
            return """Examples:

1. One-hop: "What movies did Tom Hanks star in?" (Tom Hanks is Person)
[
  {{"src_node": "Tom Hanks", "src_type": "Person", "relation": "starred in", "tgt_node": "?", "tgt_type": "Movie", "is_anchor": true}}
]

2. Two-hop: "Who directed the movies that Tom Hanks starred in?" (Tom Hanks is Person)
[
  {{"src_node": "Tom Hanks", "src_type": "Person", "relation": "starred in", "tgt_node": "?", "tgt_type": "Movie", "is_anchor": true}},
  {{"src_node": "?", "src_type": "Movie", "relation": "directed by", "tgt_node": "?", "tgt_type": "Person", "is_anchor": false}}
]

3. One-hop: "What year was Titanic released?" (Titanic is Movie)
[
  {{"src_node": "Titanic", "src_type": "Movie", "relation": "released in", "tgt_node": "?", "tgt_type": "Date", "is_anchor": true}}
]"""
        else:
            return """Examples:

1. One-hop: "What diseases are associated with BRCA1?" (BRCA1 is gene/protein)
[
  {{"src_node": "BRCA1", "src_type": "gene/protein", "relation": "associated with", "tgt_node": "?", "tgt_type": "disease", "is_anchor": true}}
]

2. Two-hop: "Which diseases are linked to genes targeted by Tacrolimus?" (Tacrolimus is drug)
[
  {{"src_node": "Tacrolimus", "src_type": "drug", "relation": "targets", "tgt_node": "?", "tgt_type": "gene/protein", "is_anchor": true}},
  {{"src_node": "?", "src_type": "gene/protein", "relation": "associated with", "tgt_node": "?", "tgt_type": "disease", "is_anchor": false}}
]

3. Two-hop reverse: "Which drugs target genes linked to heart failure?" (heart failure is disease)
[
  {{"src_node": "heart failure", "src_type": "disease", "relation": "associated with", "tgt_node": "?", "tgt_type": "gene/protein", "is_anchor": true}},
  {{"src_node": "?", "src_type": "gene/protein", "relation": "targeted by", "tgt_node": "?", "tgt_type": "drug", "is_anchor": false}}
]"""

    def _generate_pseudo_query_graph(
        self,
        question: str,
        entity_name: Optional[str] = None
    ) -> List[PseudoEdge]:
        """LLMで擬似クエリグラフを生成"""

        type_list = ", ".join(sorted(self.schema.types))
        examples = self._get_prompt_examples()

        prompt = f"""Convert this question into a query graph structure.

Question: {question}
{"Known entity: " + entity_name if entity_name else ""}

Available node types: {type_list}

Generate edges that represent the query structure. Each edge should have:
- src_node: source node name (use "?" for unknown)
- src_type: source node type
- relation: relationship name (use natural language, e.g., "starred in", "directed by", "released in")
- tgt_node: target node name (use "?" for unknown/answer)
- tgt_type: target node type
- is_anchor: true if the node is a known entity, false otherwise

CRITICAL: The direction matters! The anchor entity should be the src_node of the first edge.

{examples}

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

        return matched_subgraphs

    def _multi_hop_search(
        self,
        query_graph: QueryGraph,
        anchor_name: str,
        anchor_type: str,
    ) -> List[MatchedSubgraph]:
        """マルチホップパスを検索（両方向対応）"""

        graph = self.finder.graph

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        edges = query_graph.edges

        if len(edges) == 1:
            # 1-hop: 単純な検索
            return self._dfs_search(query_graph, anchor_name, anchor_type)

        elif len(edges) == 2:
            # 2-hop: チェーンパターン（全方向組み合わせを試行）
            se1, pe1 = edges[0]
            se2, pe2 = edges[1]

            results = []

            # 4つの方向パターンを試行:
            # Pattern 1: (a)->(mid)->(ans) - 両方順方向
            # Pattern 2: (mid)->(a), (mid)->(ans) - 1番目逆、2番目順
            # Pattern 3: (a)->(mid), (ans)->(mid) - 1番目順、2番目逆
            # Pattern 4: (mid)->(a), (ans)->(mid) - 両方逆

            patterns = [
                # (direction1, direction2, cypher_template)
                ("fwd", "fwd", f"""
                    MATCH (a:{get_label(se1.src_type)})-[r1:{se1.relation}]->(mid:{get_label(se1.tgt_type)})-[r2:{se2.relation}]->(ans:{get_label(se2.tgt_type)})
                    WHERE a.name = $anchor_name AND a <> mid AND mid <> ans
                    RETURN DISTINCT a.name AS anchor, mid.name AS mid_node, ans.name AS answer
                    LIMIT 100
                """),
                ("rev", "fwd", f"""
                    MATCH (mid:{get_label(se1.tgt_type)})-[r1:{se1.relation}]->(a:{get_label(se1.src_type)})
                    MATCH (mid)-[r2:{se2.relation}]->(ans:{get_label(se2.tgt_type)})
                    WHERE a.name = $anchor_name AND a <> mid AND mid <> ans
                    RETURN DISTINCT a.name AS anchor, mid.name AS mid_node, ans.name AS answer
                    LIMIT 100
                """),
                ("fwd", "rev", f"""
                    MATCH (a:{get_label(se1.src_type)})-[r1:{se1.relation}]->(mid:{get_label(se1.tgt_type)})
                    MATCH (ans:{get_label(se2.tgt_type)})-[r2:{se2.relation}]->(mid)
                    WHERE a.name = $anchor_name AND a <> mid AND mid <> ans
                    RETURN DISTINCT a.name AS anchor, mid.name AS mid_node, ans.name AS answer
                    LIMIT 100
                """),
                ("rev", "rev", f"""
                    MATCH (mid:{get_label(se1.tgt_type)})-[r1:{se1.relation}]->(a:{get_label(se1.src_type)})
                    MATCH (ans:{get_label(se2.tgt_type)})-[r2:{se2.relation}]->(mid)
                    WHERE a.name = $anchor_name AND a <> mid AND mid <> ans
                    RETURN DISTINCT a.name AS anchor, mid.name AS mid_node, ans.name AS answer
                    LIMIT 100
                """),
            ]

            for dir1, dir2, cypher in patterns:
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
                                (record["anchor"], se1.relation, record["mid_node"]),
                                (record["mid_node"], se2.relation, record["answer"]),
                            ],
                            score=0.0,
                        )
                        results.append(subgraph)

                except Exception as e:
                    pass  # このパターンはマッチしなかった

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
        """DFSでサブグラフを検索（両方向を試行）"""

        graph = self.finder.graph

        results = []

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        # クエリグラフの各エッジについて検索
        for schema_edge, pseudo_edge in query_graph.edges:
            queries = []

            # Case 1: 順方向、anchor is source
            cypher1 = f"""
            MATCH (src:{get_label(schema_edge.src_type)})-[r:{schema_edge.relation}]->(tgt:{get_label(schema_edge.tgt_type)})
            WHERE src.name = $anchor_name
            RETURN DISTINCT
                src.name AS src_name,
                type(r) AS rel_type,
                tgt.name AS tgt_name
            LIMIT 50
            """
            queries.append(("fwd_src", cypher1))

            # Case 2: 順方向、anchor is target
            cypher2 = f"""
            MATCH (src:{get_label(schema_edge.src_type)})-[r:{schema_edge.relation}]->(tgt:{get_label(schema_edge.tgt_type)})
            WHERE tgt.name = $anchor_name
            RETURN DISTINCT
                src.name AS src_name,
                type(r) AS rel_type,
                tgt.name AS tgt_name
            LIMIT 50
            """
            queries.append(("fwd_tgt", cypher2))

            # Case 3: 逆方向（実際のエッジがtgt->srcの場合）、anchor is target (of reversed edge = src in schema)
            cypher3 = f"""
            MATCH (tgt:{get_label(schema_edge.tgt_type)})-[r:{schema_edge.relation}]->(src:{get_label(schema_edge.src_type)})
            WHERE src.name = $anchor_name
            RETURN DISTINCT
                src.name AS src_name,
                type(r) AS rel_type,
                tgt.name AS tgt_name
            LIMIT 50
            """
            queries.append(("rev_src", cypher3))

            # Case 4: 逆方向、anchor is source (of reversed edge = tgt in schema)
            cypher4 = f"""
            MATCH (tgt:{get_label(schema_edge.tgt_type)})-[r:{schema_edge.relation}]->(src:{get_label(schema_edge.src_type)})
            WHERE tgt.name = $anchor_name
            RETURN DISTINCT
                src.name AS src_name,
                type(r) AS rel_type,
                tgt.name AS tgt_name
            LIMIT 50
            """
            queries.append(("rev_tgt", cypher4))

            for anchor_pos, cypher in queries:
                try:
                    records = graph.run(cypher, anchor_name=anchor_name).data()

                    for record in records:
                        # anchor位置に応じてノードマッピングを決定
                        # pseudo_edge.src_nodeがアンカー、tgt_nodeが回答
                        if anchor_pos in ("fwd_src", "rev_src"):
                            # アンカーがsrc_name側
                            nodes = {
                                pseudo_edge.src_node: record["src_name"],
                                pseudo_edge.tgt_node: record["tgt_name"],
                            }
                            edge_tuple = (record["src_name"], record["rel_type"], record["tgt_name"])
                        else:
                            # アンカーがtgt_name側
                            nodes = {
                                pseudo_edge.src_node: record["tgt_name"],
                                pseudo_edge.tgt_node: record["src_name"],
                            }
                            edge_tuple = (record["tgt_name"], record["rel_type"], record["src_name"])

                        subgraph = MatchedSubgraph(
                            nodes=nodes,
                            edges=[edge_tuple],
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
        """回答エンティティを抽出 - パスの終点エンティティをそのまま返す"""

        if not subgraphs:
            return []

        # アンカー（既知エンティティ）を特定
        anchors = set()
        for p_edge in pseudo_edges:
            if p_edge.src_node and p_edge.src_node != "?":
                anchors.add(p_edge.src_node)
                anchors.add(p_edge.src_node.lower())

        answers = []
        for sg in subgraphs:
            if not sg.edges:
                continue

            # パスの終点エンティティを取得（最後のエッジから）
            last_edge = sg.edges[-1]
            src, rel, tgt = last_edge

            # アンカーでない方を回答とする
            src_is_anchor = src in anchors or src.lower() in anchors
            tgt_is_anchor = tgt in anchors or tgt.lower() in anchors

            if not tgt_is_anchor:
                endpoint = tgt
            elif not src_is_anchor:
                endpoint = src
            else:
                # 両方ともアンカーでない/両方ともアンカーの場合はtgtを使用
                endpoint = tgt

            if endpoint and endpoint not in answers:
                answers.append(endpoint)

        return answers
