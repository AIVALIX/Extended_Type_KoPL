"""
SAFE (Semantic-Aware Subgraph Retrieval Framework) Pipeline

論文に基づく実装:
1. ADJ (Approximate Distance Join) - スキーマレベル探索 (Algorithm 1)
2. Ranked Semantic Subgraph Matching - インスタンスレベル探索 (Algorithm 2)
"""

from __future__ import annotations

import heapq
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.config import BASEMODEL, get_settings
from database.search import GraphPathFinder
from pipeline.safe.models import (
    PseudoEdge,
    SchemaEdge,
    CandidateSchemaEdge,
    QueryGraph,
    MatchedSubgraph,
    PartialMatch,
    SAFEResult,
    PseudoQueryGraphResponse,
)
from pipeline.safe.schema import (
    SchemaGraphWithAPSP,
    CandidateIndex,
    build_schema_with_apsp,
    build_schema_from_kg,
)


class SAFEPipeline:
    """SAFEパイプライン"""

    def __init__(
        self,
        model: str = BASEMODEL,
        embedding_model: str = "text-embedding-3-small",
        k_schema: int = 3,  # 候補スキーマエッジ数
        k_qg: int = 5,  # 候補クエリグラフ数
        k_retrieval: int = 10,  # 検索結果数
        k_sim_ent: int = 3,  # SimEnt候補数（論文デフォルト）
        k_sim_rel: int = 10,  # SimRel候補数
        k_sim_typ: int = 8,  # SimTyp候補数
        delta: int = 1,  # エッジ距離閾値
        kg_type: str = "primekgqa",
        use_schema_relations: bool = True,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings

        self.llm = init_chat_model(model, model_provider="openai", temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.kg_type = kg_type
        self.use_schema_relations = use_schema_relations
        self.finder = GraphPathFinder(kg_type=kg_type)

        if use_schema_relations:
            self.schema = build_schema_with_apsp(kg_type)
        else:
            self.schema = build_schema_from_kg(self.finder)

        self.schema.compute_edge_embeddings(self.embeddings)

        self.k_schema = k_schema
        self.k_qg = k_qg
        self.k_retrieval = k_retrieval
        self.k_sim_ent = k_sim_ent
        self.k_sim_rel = k_sim_rel
        self.k_sim_typ = k_sim_typ
        self.delta = delta

        # CandidateIndex の構築 (SimEnt/SimRel/SimTyp用)
        self.candidate_index = CandidateIndex(self.embeddings)
        all_relations = sorted({e.relation for e in self.schema.edges})
        all_types = sorted(self.schema.types)
        self.candidate_index.build_relation_index(all_relations)
        self.candidate_index.build_type_index(all_types)

        # SimEnt: エンティティインデックス構築（.npzファイル → フォールバックでAPI呼び出し）
        from pipeline.safe.precompute_entity_embeddings import get_cache_path
        npz_path = get_cache_path(kg_type)
        if npz_path.exists():
            data = np.load(npz_path)
            ent_names = data["names"].tolist()
            ent_vecs = data["vecs"]  # 2D array (N, dim) をそのまま渡す（コピー回避）
            self.candidate_index.load_entity_index(ent_names, ent_vecs)
        else:
            all_entity_names = self.finder.get_all_entity_names()
            self.candidate_index.build_entity_index(all_entity_names)

        # キャッシュ（run()呼び出しごとにクリア）
        self._adj_rel_cache: Dict[str, List[str]] = {}
        self._cand_node_cache: Dict[Tuple[str, str], List[Tuple[str, List[str]]]] = {}
        self._entity_labels_cache: Dict[str, List[str]] = {}

    def run(self, question: str, entity_name: Optional[str] = None, n_gold: int = 0) -> SAFEResult:
        """パイプライン実行"""
        # キャッシュクリア
        self._adj_rel_cache.clear()
        self._cand_node_cache.clear()
        self._entity_labels_cache.clear()

        log = []
        log.append(f"Question: {question}")

        # PcQA: CancerCell複合名マッチング
        if self.kg_type == "pcqa" and entity_name:
            # Case-insensitive entity type lookup
            entity_type = None
            labels = self.finder.get_entity_labels(entity_name)
            if not labels:
                # Try case-insensitive lookup
                query = """
                MATCH (n) WHERE toLower(n.name) = toLower($name)
                RETURN labels(n) AS labels LIMIT 1
                """
                rows = self.finder.graph.run(query, name=entity_name).data()
                if rows:
                    labels = rows[0]["labels"]
            if labels:
                filtered = [l for l in labels if l not in ("_Entity", "Entity")]
                entity_type = filtered[0] if filtered else None
            if entity_type in ("Genesymbol", "Fusion"):
                from pipeline.common.pcqa import resolve_pcqa_compound_entity
                compound = resolve_pcqa_compound_entity(
                    self.llm, self.finder, question, entity_name, entity_type
                )
                if compound:
                    compound_name, compound_type, compound_names = compound
                    log.append(f"  PcQA CancerCell resolved: {entity_name} -> {compound_name} ({compound_type})")
                    entity_name = compound_name

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

        # Step 3: Cypher Execution (primary) -> Algorithm 2 (fallback)
        log.append("Phase 3: Cypher Execution")
        answer_entities = self._cypher_execution(best_qg, pseudo_edges)
        log.append(f"  Cypher returned {len(answer_entities)} answers")

        matched_subgraphs = []
        if not answer_entities:
            # Fallback: Algorithm 2
            log.append("  Cypher returned 0 results, falling back to Algorithm 2")
            k = max(self.k_retrieval, n_gold) if n_gold > 0 else self.k_retrieval
            matched_subgraphs = self._subgraph_matching(best_qg, pseudo_edges, k=k)
            log.append(f"  Algorithm 2 found {len(matched_subgraphs)} matched subgraphs")
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
        """KGタイプに応じたプロンプト例を返す（3例のみ）"""
        if self.kg_type == "metaqa":
            return """Examples:

1. 1-hop: "Who directed Titanic?"
[
  {{"src_node": "Titanic", "src_type": "Movie", "relation": "directed by", "tgt_node": "?", "tgt_type": "Person", "is_anchor": true}}
]

2. 2-hop: "Who directed the movies that Tom Hanks starred in?"
[
  {{"src_node": "Tom Hanks", "src_type": "Person", "relation": "starred in", "tgt_node": "?", "tgt_type": "Movie", "is_anchor": true}},
  {{"src_node": "?", "src_type": "Movie", "relation": "directed by", "tgt_node": "?", "tgt_type": "Person", "is_anchor": false}}
]

3. 3-hop: "Who starred in the movies written by the writers of The Matrix?"
[
  {{"src_node": "The Matrix", "src_type": "Movie", "relation": "written by", "tgt_node": "?", "tgt_type": "Person", "is_anchor": true}},
  {{"src_node": "?", "src_type": "Person", "relation": "wrote", "tgt_node": "?", "tgt_type": "Movie", "is_anchor": false}},
  {{"src_node": "?", "src_type": "Movie", "relation": "starred by", "tgt_node": "?", "tgt_type": "Person", "is_anchor": false}}
]

IMPORTANT: Build the path from anchor entity to final answer. Each edge = one hop."""
        else:
            return """Examples:

1. 1-hop: "What diseases is Metformin indicated for?"
[
  {{"src_node": "Metformin", "src_type": "drug", "relation": "indicated for", "tgt_node": "?", "tgt_type": "disease", "is_anchor": true}}
]

2. 2-hop: "What phenotypes are present in diseases treated by Aspirin?"
[
  {{"src_node": "Aspirin", "src_type": "drug", "relation": "indicated for", "tgt_node": "?", "tgt_type": "disease", "is_anchor": true}},
  {{"src_node": "?", "src_type": "disease", "relation": "has phenotype", "tgt_node": "?", "tgt_type": "effect/phenotype", "is_anchor": false}}
]

3. Intersection: "What genes are targeted by both Aspirin and Ibuprofen?"
[
  {{"src_node": "Aspirin", "src_type": "drug", "relation": "targets", "tgt_node": "?", "tgt_type": "gene/protein", "is_anchor": true}},
  {{"src_node": "Ibuprofen", "src_type": "drug", "relation": "targets", "tgt_node": "?", "tgt_type": "gene/protein", "is_anchor": true}}
]\""""

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

        candidates_per_edge: List[List[CandidateSchemaEdge]] = []

        for p_edge in pseudo_edges:
            candidates = self._find_candidate_schema_edges(p_edge)
            candidates_per_edge.append(candidates)

        if not candidates_per_edge or not all(candidates_per_edge):
            return []

        query_graphs = self._match_sg(pseudo_edges, candidates_per_edge, 0, [], 0.0)

        query_graphs.sort(key=lambda qg: qg.total_distance)

        return query_graphs[:self.k_qg]

    def _find_candidate_schema_edges(
        self,
        pseudo_edge: PseudoEdge
    ) -> List[CandidateSchemaEdge]:
        """候補スキーマエッジを検索"""

        pseudo_text = f"{pseudo_edge.src_type} {pseudo_edge.relation} {pseudo_edge.tgt_type}"
        pseudo_vec = np.array(self.embeddings.embed_query(pseudo_text))

        distances = []
        for schema_edge in self.schema.edges:
            if schema_edge.embedding is not None:
                dist = np.linalg.norm(pseudo_vec - schema_edge.embedding)
                distances.append((schema_edge, dist))

        distances.sort(key=lambda x: x[1])

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

        if idx >= len(pseudo_edges):
            return [QueryGraph(
                edges=current_matches.copy(),
                total_distance=current_distance,
            )]

        results = []

        for candidate in candidates_per_edge[idx]:
            if current_matches:
                last_schema_edge = current_matches[-1][0]
                edge_dist = self.schema.get_edge_distance(last_schema_edge, candidate.schema_edge)
                if edge_dist > self.delta:
                    continue

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

    # ──────────────────────────────────────────────
    #  Algorithm 2: Ranked Semantic Subgraph Matching
    # ──────────────────────────────────────────────

    def _cached_adj_rels(self, entity_name: str) -> List[str]:
        """get_adjacent_relations結果をキャッシュ"""
        if entity_name not in self._adj_rel_cache:
            self._adj_rel_cache[entity_name] = self.finder.get_adjacent_relations(entity_name)
        return self._adj_rel_cache[entity_name]

    def _cached_cand_nodes(self, entity_name: str, relation: str) -> List[Tuple[str, List[str]]]:
        """get_candidate_nodes結果をキャッシュ（ラベルキャッシュも同時に構築）"""
        key = (entity_name, relation)
        if key not in self._cand_node_cache:
            results = self.finder.get_candidate_nodes(entity_name, relation)
            self._cand_node_cache[key] = results
            # ラベルキャッシュにも格納
            for name, labels in results:
                if name and name not in self._entity_labels_cache:
                    self._entity_labels_cache[name] = labels
        return self._cand_node_cache[key]

    def _cached_entity_labels(self, entity_name: str) -> List[str]:
        """エンティティラベルをキャッシュ付きで取得"""
        if entity_name not in self._entity_labels_cache:
            self._entity_labels_cache[entity_name] = self.finder.get_entity_labels(entity_name)
        return self._entity_labels_cache[entity_name]

    def _build_dfs_edge_order(
        self,
        query_graph: QueryGraph,
        pseudo_edges: List[PseudoEdge],
    ) -> List[Tuple[str, str, str, SchemaEdge, PseudoEdge]]:
        """QueryGraphのエッジをDFS順に並べ替え（anchor起点）

        "?"変数を一意にリネーム: チェーン接続を保ちつつ衝突を防ぐ。
        例: 2-hop → Edge0: src="Aspirin", tgt="?_0"
                    Edge1: src="?_0", tgt="?_1"

        Returns:
            List of (src_var, query_relation, tgt_var, schema_edge, pseudo_edge)
        """
        edge_order = []
        var_counter = 0
        prev_tgt_var: Optional[str] = None

        for schema_edge, pseudo_edge in query_graph.edges:
            src_var = pseudo_edge.src_node
            tgt_var = pseudo_edge.tgt_node
            query_rel = schema_edge.relation  # 論文: r_Q は Q_G のスキーマ関係

            # src が "?" の場合、前エッジの tgt と同じ変数（チェーン接続）
            if src_var == "?" and prev_tgt_var is not None:
                src_var = prev_tgt_var

            # tgt が "?" の場合、一意の変数名を割り当て
            if tgt_var == "?":
                tgt_var = f"?_{var_counter}"
                var_counter += 1

            prev_tgt_var = tgt_var
            edge_order.append((src_var, query_rel, tgt_var, schema_edge, pseudo_edge))

        return edge_order

    def _subgraph_matching(
        self,
        query_graph: QueryGraph,
        pseudo_edges: List[PseudoEdge],
        k: Optional[int] = None,
    ) -> List[MatchedSubgraph]:
        """Ranked Semantic Subgraph Matching (Algorithm 2)

        Priority queueベースの実装:
        1. アンカー特定、DFSエッジ順序構築
        2. Priority queue初期化（アンカーマッピングのみ）
        3. While queue not empty:
           a. Pop最小距離のPartialMatch
           b. 完了チェック → final_resultsに追加
           c. 次エッジ取得
           d. AdjRel ∩ SimRel で関係候補
           e. 各関係候補について GetCandNode でノード候補取得
           f. SimTyp check + 枝刈り → 新PartialMatchをpush
        4. Return top-k results
        """
        if k is None:
            k = self.k_retrieval

        # アンカーノードを特定
        anchors = []
        for schema_edge, pseudo_edge in query_graph.edges:
            if pseudo_edge.src_node != "?" and pseudo_edge.src_node not in [a[0] for a in anchors]:
                anchors.append((pseudo_edge.src_node, schema_edge.src_type, schema_edge))

        if not anchors:
            return []

        anchor_name = anchors[0][0]

        # DFSエッジ順序構築
        edge_order = self._build_dfs_edge_order(query_graph, pseudo_edges)
        total_edges = len(edge_order)

        if total_edges == 0:
            return []

        # SimRelの候補関係（各クエリ関係に対して事前計算、距離付き）
        sim_rel_cache: Dict[str, Dict[str, float]] = {}
        for _, query_rel, _, _, _ in edge_order:
            if query_rel not in sim_rel_cache:
                sim_rels = self.candidate_index.sim_rel(query_rel, k=self.k_sim_rel)
                sim_rel_cache[query_rel] = {r: d for r, d in sim_rels}

        # SimTypの候補タイプセット（各スキーマタイプに対して事前計算）
        sim_typ_cache: Dict[str, Set[str]] = {}
        for _, _, _, schema_edge, _ in edge_order:
            tgt_type = schema_edge.tgt_type
            if tgt_type and tgt_type not in sim_typ_cache:
                sim_typs = self.candidate_index.sim_typ(tgt_type, k=self.k_sim_typ)
                sim_typ_cache[tgt_type] = {t for t, _ in sim_typs}

        # Priority queue初期化（SimEnt: アンカーエンティティのtop-k候補）
        pq: List[Tuple[float, int, PartialMatch]] = []
        counter = 0
        sim_ents = self.candidate_index.sim_ent(anchor_name, k=self.k_sim_ent)
        for ent_name, ent_dist in sim_ents:
            initial_match = PartialMatch(
                node_mapping={anchor_name: ent_name},
                edge_mapping=[],
                current_distance=ent_dist,
                edges_matched=0,
            )
            heapq.heappush(pq, (ent_dist, counter, initial_match))
            counter += 1

        final_results: List[MatchedSubgraph] = []
        max_iterations = 50000
        iterations = 0

        # top-kの最悪スコアを追跡（枝刈り用）
        worst_score = float('inf')

        while pq and iterations < max_iterations:
            iterations += 1
            dist, _, pm = heapq.heappop(pq)

            # 枝刈り: 既にk個の結果があり、このdistがworst以上なら打ち切り
            if len(final_results) >= k and dist >= worst_score:
                continue

            # 完了チェック
            if pm.edges_matched >= total_edges:
                subgraph = MatchedSubgraph(
                    nodes=dict(pm.node_mapping),
                    edges=list(pm.edge_mapping),
                    score=pm.current_distance,
                )
                final_results.append(subgraph)
                # worst_score更新
                if len(final_results) >= k:
                    worst_score = max(sg.score for sg in final_results)
                continue

            # 次エッジ取得
            edge_idx = pm.edges_matched
            src_var, query_rel, tgt_var, schema_edge, pseudo_edge = edge_order[edge_idx]

            # src_varからマッピング済みエンティティを取得
            # src_varが既にマッピングされている場合はそのエンティティを使う
            # マッピングされていない場合は、前のエッジのtgt_varのマッピングを使う
            src_entity = pm.node_mapping.get(src_var)
            if src_entity is None:
                # src_varが未マッピング → 前エッジのチェーン接続
                # edge_orderは順番なので、前のtgt_varがこのsrc_varに対応する可能性がある
                # 全マッピングから探す
                for var, entity in pm.node_mapping.items():
                    if var == src_var:
                        src_entity = entity
                        break
                if src_entity is None:
                    continue  # マッピング不可

            # AdjRel: このエンティティに隣接する全関係
            adj_rels = set(self._cached_adj_rels(src_entity))

            # SimRel: クエリ関係に類似するKG関係（距離付き）
            sim_rels = sim_rel_cache.get(query_rel, {})

            # 候補関係 C_r = AdjRel ∩ SimRel（論文通り、フォールバックなし）
            candidate_rels = adj_rels & set(sim_rels.keys())

            if not candidate_rels:
                continue  # 候補なし → この部分マッチは行き止まり

            # tgt_varの期待タイプ（スキーマから）
            tgt_type = schema_edge.tgt_type
            sim_typs = sim_typ_cache.get(tgt_type, set()) if tgt_type else set()

            for r_g in candidate_rels:
                # d_r: リレーション距離（SimRelキャッシュから取得、再計算回避）
                d_r = sim_rels.get(r_g, self.candidate_index.sem_dist_rel(query_rel, r_g))

                # このエンティティからr_gで到達可能なノード候補
                cand_nodes = self._cached_cand_nodes(src_entity, r_g)

                for node_name, labels in cand_nodes:
                    if node_name is None:
                        continue

                    # 一貫性チェック: 既にマッピングされた変数との矛盾防止
                    existing = pm.node_mapping.get(tgt_var)
                    if existing is not None and existing != node_name:
                        continue

                    # サイクル防止: 同じエンティティが複数変数にマッピングされるのを防ぐ
                    if node_name in pm.node_mapping.values() and existing is None:
                        continue

                    # SimTyp check: tgt_typeが指定されていればラベルフィルタ
                    if tgt_type and sim_typs:
                        filtered_labels = [l for l in labels if l in sim_typs]
                        if not filtered_labels and labels:
                            continue

                    # SemDist: 分解距離（論文準拠）
                    # d_e = d_r + d_node
                    # d_r = ||emb(r_Q) - emb(r_G)||₂
                    # d_node = min_{t ∈ type(u'_G)} ||emb(type(u'_Q)) - emb(t)||₂
                    tgt_schema_type = schema_edge.tgt_type
                    d_node = self.candidate_index.sem_dist_node(tgt_schema_type, labels)
                    d_e = d_r + d_node

                    new_dist = pm.current_distance + d_e

                    # 枝刈り
                    if len(final_results) >= k and new_dist >= worst_score:
                        continue

                    # 新しいPartialMatchを作成
                    new_mapping = dict(pm.node_mapping)
                    new_mapping[tgt_var] = node_name
                    new_edges = list(pm.edge_mapping)
                    new_edges.append((src_entity, r_g, node_name))

                    new_pm = PartialMatch(
                        node_mapping=new_mapping,
                        edge_mapping=new_edges,
                        current_distance=new_dist,
                        edges_matched=pm.edges_matched + 1,
                    )
                    heapq.heappush(pq, (new_dist, counter, new_pm))
                    counter += 1

        # スコア順にソート
        final_results.sort(key=lambda sg: sg.score)

        # フォールバック: Algorithm 2で結果が0件の場合、旧Cypherベースで検索
        if not final_results:
            final_results = self._fallback_cypher_search(query_graph, pseudo_edges)

        return final_results[:k]

    def _resolve_anchor_type(
        self,
        schema_edge: "SchemaEdge",
        pseudo_edge: PseudoEdge,
    ) -> str:
        """アンカーエンティティに対応するスキーマエッジ側のタイプを決定"""
        pe_type = pseudo_edge.src_type.lower()
        if pe_type == schema_edge.src_type.lower():
            return schema_edge.src_type
        elif pe_type == schema_edge.tgt_type.lower():
            return schema_edge.tgt_type
        # フォールバック: Neo4jから実際のラベルを取得
        labels = self.finder.get_entity_labels(pseudo_edge.src_node)
        if labels:
            labels_lower = [l.lower() for l in labels]
            if schema_edge.src_type.lower() in labels_lower:
                return schema_edge.src_type
            if schema_edge.tgt_type.lower() in labels_lower:
                return schema_edge.tgt_type
        return schema_edge.src_type

    def _cypher_execution(
        self,
        query_graph: QueryGraph,
        pseudo_edges: List[PseudoEdge],
    ) -> List[str]:
        """ADJ選定のスキーマパスからCypherクエリを構築・実行し、回答エンティティを返す"""
        anchors = []
        for schema_edge, pseudo_edge in query_graph.edges:
            if pseudo_edge.src_node != "?" and pseudo_edge.src_node not in [a[0] for a in anchors]:
                anchor_type = self._resolve_anchor_type(schema_edge, pseudo_edge)
                anchors.append((pseudo_edge.src_node, anchor_type))

        if not anchors:
            return []

        graph = self.finder.graph

        def esc(name: str) -> str:
            if " " in name or "-" in name or "/" in name:
                return f"`{name}`"
            return name

        edges = query_graph.edges

        # Intersection: 複数anchorがある場合、各anchorで1-hop実行して積集合
        if len(anchors) >= 2:
            sets = []
            for anchor_name, anchor_type in anchors:
                se, pe = next(
                    ((se, pe) for se, pe in edges if pe.src_node == anchor_name),
                    (edges[0][0], edges[0][1]),
                )
                resolved_type = self._resolve_anchor_type(se, pe)
                # 回答側のタイプを決定
                ans_type = se.tgt_type if resolved_type == se.src_type else se.src_type
                cypher = f"""
                    MATCH (a:{esc(resolved_type)})-[r:{esc(se.relation)}]-(ans:{esc(ans_type)})
                    WHERE a.name = $anchor_name
                    RETURN DISTINCT ans.name AS answer
                """
                try:
                    records = graph.run(cypher, anchor_name=anchor_name).data()
                    sets.append({r["answer"] for r in records if r["answer"]})
                except Exception:
                    sets.append(set())
            if sets and all(sets):
                return list(sets[0].intersection(*sets[1:]))
            return []

        anchor_name = anchors[0][0]
        anchor_type = anchors[0][1]

        # タイプチェーンを構築（アンカーから辿る方向に合わせる）
        rels = [se.relation for se, pe in edges]
        types = [anchor_type]
        current_type = anchor_type
        for se, pe in edges:
            if current_type.lower() == se.src_type.lower():
                next_type = se.tgt_type
            else:
                next_type = se.src_type
            types.append(next_type)
            current_type = next_type

        try:
            if len(edges) == 1:
                cypher = f"""
                    MATCH (a:{esc(types[0])})-[r:{esc(rels[0])}]-(ans:{esc(types[1])})
                    WHERE a.name = $anchor_name
                    RETURN DISTINCT ans.name AS answer
                """
                records = graph.run(cypher, anchor_name=anchor_name).data()
                return [r["answer"] for r in records if r["answer"]]

            elif len(edges) == 2:
                cypher = f"""
                    MATCH (a:{esc(types[0])})-[r1:{esc(rels[0])}]-(mid:{esc(types[1])})-[r2:{esc(rels[1])}]-(ans:{esc(types[2])})
                    WHERE a.name = $anchor_name AND a <> mid AND mid <> ans AND a <> ans
                    RETURN DISTINCT ans.name AS answer
                """
                records = graph.run(cypher, anchor_name=anchor_name).data()
                return [r["answer"] for r in records if r["answer"]]

            elif len(edges) == 3:
                cypher = f"""
                    MATCH (a:{esc(types[0])})-[r1:{esc(rels[0])}]-(n1:{esc(types[1])})-[r2:{esc(rels[1])}]-(n2:{esc(types[2])})-[r3:{esc(rels[2])}]-(ans:{esc(types[3])})
                    WHERE a.name = $anchor_name AND a <> n1 AND n1 <> n2 AND n2 <> ans AND a <> ans AND a <> n2
                    RETURN DISTINCT ans.name AS answer
                """
                records = graph.run(cypher, anchor_name=anchor_name).data()
                return [r["answer"] for r in records if r["answer"]]

        except Exception:
            pass

        return []

    def _fallback_cypher_search(
        self,
        query_graph: QueryGraph,
        pseudo_edges: List[PseudoEdge],
    ) -> List[MatchedSubgraph]:
        """フォールバック: 旧Cypherベースのサブグラフ検索（回帰防止）"""
        anchors = []
        for schema_edge, pseudo_edge in query_graph.edges:
            if pseudo_edge.src_node != "?" and pseudo_edge.src_node not in [a[0] for a in anchors]:
                anchors.append((pseudo_edge.src_node, schema_edge.src_type))

        if not anchors:
            return []

        anchor_name, anchor_type = anchors[0]
        graph = self.finder.graph

        def get_rel(r: str) -> str:
            if " " in r or "-" in r or "/" in r:
                return f"`{r}`"
            return r

        edges = query_graph.edges
        results = []

        if len(edges) == 1:
            se, pe = edges[0]
            cypher = f"""
                MATCH (a)-[r:{get_rel(se.relation)}]-(ans)
                WHERE a.name = $anchor_name
                RETURN DISTINCT a.name AS anchor, ans.name AS answer
                LIMIT 200
            """
            try:
                records = graph.run(cypher, anchor_name=anchor_name).data()
                for record in records:
                    results.append(MatchedSubgraph(
                        nodes={pe.src_node: record["anchor"], pe.tgt_node: record["answer"]},
                        edges=[(record["anchor"], se.relation, record["answer"])],
                        score=0.0,
                    ))
            except Exception:
                pass

        elif len(edges) == 2:
            se1, pe1 = edges[0]
            se2, pe2 = edges[1]
            cypher = f"""
                MATCH (a)-[r1:{get_rel(se1.relation)}]-(mid)-[r2:{get_rel(se2.relation)}]-(ans)
                WHERE a.name = $anchor_name
                  AND a <> mid AND mid <> ans AND a <> ans
                RETURN DISTINCT a.name AS anchor, mid.name AS mid_node, ans.name AS answer
                LIMIT 200
            """
            try:
                records = graph.run(cypher, anchor_name=anchor_name).data()
                for record in records:
                    results.append(MatchedSubgraph(
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
                    ))
            except Exception:
                pass

        elif len(edges) == 3:
            se1, pe1 = edges[0]
            se2, pe2 = edges[1]
            se3, pe3 = edges[2]
            cypher = f"""
                MATCH (a)-[r1:{get_rel(se1.relation)}]-(n1)-[r2:{get_rel(se2.relation)}]-(n2)-[r3:{get_rel(se3.relation)}]-(ans)
                WHERE a.name = $anchor_name
                  AND a <> n1 AND n1 <> n2 AND n2 <> ans AND a <> ans AND a <> n2
                RETURN DISTINCT a.name AS anchor, n1.name AS node1, n2.name AS node2, ans.name AS answer
                LIMIT 200
            """
            try:
                records = graph.run(cypher, anchor_name=anchor_name).data()
                for record in records:
                    results.append(MatchedSubgraph(
                        nodes={
                            pe1.src_node: record["anchor"],
                            pe1.tgt_node: record["node1"],
                            pe2.tgt_node: record["node2"],
                            pe3.tgt_node: record["answer"],
                        },
                        edges=[
                            (record["anchor"], se1.relation, record["node1"]),
                            (record["node1"], se2.relation, record["node2"]),
                            (record["node2"], se3.relation, record["answer"]),
                        ],
                        score=0.0,
                    ))
            except Exception:
                pass

        return results

    def _extract_answers(
        self,
        subgraphs: List[MatchedSubgraph],
        pseudo_edges: List[PseudoEdge]
    ) -> List[str]:
        """回答エンティティを抽出 - スコア順に重複排除して返す"""

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
                endpoint = tgt

            if endpoint and endpoint not in answers:
                answers.append(endpoint)

        return answers
