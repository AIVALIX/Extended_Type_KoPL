"""
Type KoPLベースのGraph QAパイプライン

処理フロー:
1. Type KoPL生成（LLM）
2. Type KoPLパース → relation_name列を抽出
3. スキーマグラフ探索 → サブグラフ候補（relation列）
4. Neo4jでパス検証 → 有効なパスのフィルタリング
5. ベクトル剪定（Type KoPLのrelation_name列 vs 有効パスのrelation列）
6. LLM Reranker（上位パスから最良を選定）
7. 最終エンティティ取得
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain.chat_models import init_chat_model

from core.config import get_settings
from database.search import GraphPathFinder, score_paths
from dataset_construction.add_kopl import (
    TransitionMap,
    TransitionMapMulti,
    build_transition_map_from_kg,
    build_transition_map_multi_from_neo4j,
)
from llm_process.embedder import OpenAIEmbedder
from llm_process.path_reranker import PathReranker
from llm_process.schema_subgraph_search import (
    ParsedTypeKoPL,
    SchemaGraph,
    SubgraphCandidate,
    parse_type_kopl,
    search_subgraph_candidates,
)
from llm_process.type_KoPL_process import generate_type_kopl
from prompts.type_KoPL_fewshot import FEW_SHOT_PrimeKGQA


# ────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────
@dataclass
class TypeKoPLPipelineConfig:
    """パイプライン設定"""

    use_vector_prune: bool = True  # ベクトル剪定の使用
    use_llm_reranker: bool = True  # LLM Rerankerの使用
    use_schema_in_prompt: bool = False  # スキーマ情報をプロンプトに追加
    max_schema_candidates: int = 50  # スキーマ探索の最大候補数
    max_vector_candidates: int = 20  # ベクトル剪定後の最大候補数
    top_k_rerank: int = 10  # LLM Rerankerに入力する上位パス数
    max_results: int = 10  # 最終結果の最大エンティティ数
    model: str = "gpt-4.1-mini"  # LLMモデル
    fewshot: str = FEW_SHOT_PrimeKGQA  # Few-shot例


@dataclass
class PipelineResult:
    """パイプライン結果"""

    question: str
    entity_name: str
    type_kopl: str
    parsed_kopl: Optional[ParsedTypeKoPL]
    schema_candidates: List[SubgraphCandidate]
    valid_paths: List[Dict[str, Any]]
    vector_scored_paths: List[Tuple[int, float]]
    final_path: Optional[List[str]]
    reachable_entities: List[str]
    processing_log: List[str] = field(default_factory=list)
    # Intersection関連
    anchor_results: Optional[Dict[str, List[str]]] = None  # 各アンカーからの到達エンティティ
    intersection_paths: Optional[Dict[str, List[str]]] = None  # 各アンカーのパス


# ────────────────────────────────────────────────────────────────
#  Pipeline
# ────────────────────────────────────────────────────────────────
class TypeKoPLPipeline:
    """Type KoPLベースのGraph QAパイプライン"""

    def __init__(
        self,
        transitions: TransitionMap,
        config: Optional[TypeKoPLPipelineConfig] = None,
        transitions_multi: Optional[TransitionMapMulti] = None,
    ):
        self.transitions = transitions
        self.transitions_multi = transitions_multi  # Type KoPL剪定用（複数target対応）
        self.schema = SchemaGraph(transitions=transitions)
        self.config = config or TypeKoPLPipelineConfig()
        self.settings = get_settings()

        # コンポーネント（遅延初期化）
        self._finder: Optional[GraphPathFinder] = None
        self._embedder: Optional[OpenAIEmbedder] = None
        self._reranker: Optional[PathReranker] = None
        self._llm = None

    @property
    def finder(self) -> GraphPathFinder:
        if self._finder is None:
            self._finder = GraphPathFinder()
        return self._finder

    @property
    def embedder(self) -> OpenAIEmbedder:
        if self._embedder is None:
            self._embedder = OpenAIEmbedder()
        return self._embedder

    @property
    def reranker(self) -> PathReranker:
        if self._reranker is None:
            self._ensure_openai_key()
            self._llm = init_chat_model(
                self.config.model, model_provider="openai", temperature=0.0
            )
            self._reranker = PathReranker(self._llm)
        return self._reranker

    def _ensure_openai_key(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.settings.OPENAI_API_KEY

    def _get_schema_context(self) -> str:
        """スキーマ情報（型とrelation）をコンテキスト文字列として取得"""
        types = set()
        relations = set()
        for (rel, src), tgt in self.transitions.mapping.items():
            types.add(src)
            types.add(tgt)
            relations.add(rel)

        return f"""【Available Schema】
Types: {', '.join(sorted(types))}
Relations: {', '.join(sorted(relations))}
"""

    # ────────────────────────────────────────────────────────────
    #  Step 1: Type KoPL生成
    # ────────────────────────────────────────────────────────────
    def generate_type_kopl(self, question: str) -> str:
        """LLMでType KoPLを生成"""
        fewshot = self.config.fewshot

        # スキーマ情報をプロンプトに追加
        if self.config.use_schema_in_prompt:
            schema_context = self._get_schema_context()
            fewshot = schema_context + "\n" + fewshot

        return generate_type_kopl(
            question=question,
            fewshot=fewshot,
            model=self.config.model,
        )

    # ────────────────────────────────────────────────────────────
    #  Step 2: Type KoPLパース
    # ────────────────────────────────────────────────────────────
    def parse_kopl(self, kopl_text: str) -> ParsedTypeKoPL:
        """Type KoPLをパース"""
        return parse_type_kopl(kopl_text)

    # ────────────────────────────────────────────────────────────
    #  Step 3: スキーマグラフ探索
    # ────────────────────────────────────────────────────────────
    def search_schema_candidates(
        self, parsed: ParsedTypeKoPL
    ) -> List[SubgraphCandidate]:
        """スキーマグラフからサブグラフ候補を探索"""
        return search_subgraph_candidates(
            self.schema,
            parsed,
            max_candidates=self.config.max_schema_candidates,
        )

    # ────────────────────────────────────────────────────────────
    #  Step 4: Neo4j検証
    # ────────────────────────────────────────────────────────────
    def validate_paths_in_neo4j(
        self,
        entity_name: str,
        candidates: List[SubgraphCandidate],
    ) -> List[Dict[str, Any]]:
        """Neo4jで各候補パスを検証し、有効なパスのみを返す"""
        valid_paths = []

        for cand in candidates:
            entities = self._query_neo4j_path(entity_name, cand.relation_path)
            if entities:
                valid_paths.append({
                    "relation_path": cand.relation_path,
                    "type_path": cand.type_path,
                    "reachable_entities": entities,
                })

        return valid_paths

    def _query_neo4j_path(
        self,
        entity_name: str,
        relation_path: List[str],
        max_results: int = 10,
    ) -> List[str]:
        """Neo4jでrelationパスをたどり到達エンティティを取得"""
        if not relation_path:
            return [entity_name]

        try:
            cypher_parts = [f"MATCH (n0 {{name: $entity_name}})"]
            for i, rel in enumerate(relation_path):
                cypher_parts.append(f"MATCH (n{i})-[r{i}:`{rel}`]->(n{i+1})")

            last_idx = len(relation_path)
            cypher_parts.append(
                f"RETURN DISTINCT n{last_idx}.name AS name LIMIT {max_results}"
            )

            query = "\n".join(cypher_parts)
            result = self.finder.graph.run(query, entity_name=entity_name).data()
            return [r["name"] for r in result if r.get("name")]
        except Exception:
            return []

    # ────────────────────────────────────────────────────────────
    #  Step 5: Type KoPLからrelation_name列を抽出
    # ────────────────────────────────────────────────────────────
    def extract_kopl_relations(self, parsed: ParsedTypeKoPL) -> List[str]:
        """Type KoPLの遷移からrelation_name列を抽出"""
        return [trans.relation_name for trans in parsed.transitions]

    # ────────────────────────────────────────────────────────────
    #  Step 6: ベクトル剪定
    # ────────────────────────────────────────────────────────────
    def vector_scoring(
        self,
        kopl_relations: List[str],
        valid_paths: List[Dict[str, Any]],
    ) -> List[Tuple[int, float]]:
        """
        Type KoPLのrelation_name列と有効パスのrelation列のベクトル類似度でスコアリング

        Args:
            kopl_relations: Type KoPLで生成されたrelation_name列 (例: ["related_to", "targets"])
            valid_paths: Neo4jで検証済みの有効パス

        Returns:
            スコア付きパスのリスト [(path_index, score), ...]
        """
        if not valid_paths or not kopl_relations:
            return []

        # Type KoPLのrelation_name列の埋め込み
        kopl_embeddings = self._get_relation_embeddings(kopl_relations)

        # 各有効パスのrelation列の埋め込みを取得
        path_embeddings = []
        for path_info in valid_paths:
            rel_path = path_info["relation_path"]
            rel_vecs = self._get_relation_embeddings(rel_path)
            path_embeddings.append(rel_vecs)

        # スコアリング（Type KoPLのrelation vs 有効パスのrelation）
        scored = score_paths(
            query_vecs=kopl_embeddings,
            path_vectors=path_embeddings,
            top_k=self.config.max_vector_candidates,
        )

        return scored

    def _get_relation_embeddings(
        self, relation_path: List[str]
    ) -> List[np.ndarray]:
        """relationの埋め込みを取得（Neo4jまたはembedder）"""
        embeddings = []

        for rel in relation_path:
            # まずNeo4jのRelationEmbeddingを試す
            try:
                result = self.finder.graph.run(
                    """
                    MATCH (e:RelationEmbedding {relation_name: $rel})
                    RETURN e.embedding_vector AS vec
                    """,
                    rel=rel,
                ).data()

                if result and result[0].get("vec"):
                    embeddings.append(np.asarray(result[0]["vec"], dtype=np.float32))
                    continue
            except Exception:
                pass

            # フォールバック: relationテキストを埋め込み
            vec = self.embedder.embed_one(rel)
            embeddings.append(np.asarray(vec, dtype=np.float32))

        return embeddings

    # ────────────────────────────────────────────────────────────
    #  Step 7: LLM Reranker
    # ────────────────────────────────────────────────────────────
    def llm_rerank(
        self,
        question: str,
        kopl_relations: List[str],
        valid_paths: List[Dict[str, Any]],
        scored_paths: List[Tuple[int, float]],
    ) -> Optional[List[str]]:
        """
        LLMで上位パスを再ランキングし、最良パスを返す

        Args:
            question: 元の質問文
            kopl_relations: Type KoPLのrelation_name列
            valid_paths: 有効なパス
            scored_paths: ベクトルスコア付きパス

        Returns:
            最良のrelationパス
        """
        if not scored_paths:
            return None

        # 上位パスを取得
        top_paths = [
            valid_paths[idx]["relation_path"]
            for idx, _ in scored_paths[: self.config.top_k_rerank]
        ]

        # クエリとして元の質問とType KoPLのrelation列を組み合わせる
        queries = [question] + [f"relation: {rel}" for rel in kopl_relations]

        try:
            result = self.reranker.invoke(queries=queries, paths=top_paths)
            if hasattr(result, "reranked_combos") and result.reranked_combos:
                return result.reranked_combos
        except Exception:
            pass

        return None

    # ────────────────────────────────────────────────────────────
    #  Step 8: 最終エンティティ取得
    # ────────────────────────────────────────────────────────────
    def get_final_entities(
        self,
        entity_name: str,
        final_path: List[str],
    ) -> List[str]:
        """最終パスで到達可能なエンティティを取得"""
        return self._query_neo4j_path(
            entity_name, final_path, max_results=self.config.max_results
        )

    # ────────────────────────────────────────────────────────────
    #  メインパイプライン
    # ────────────────────────────────────────────────────────────
    def run(
        self,
        question: str,
        entity_name: str,
        *,
        type_kopl_override: Optional[str] = None,
    ) -> PipelineResult:
        """パイプラインを実行"""
        log = []
        log.append(f"Question: {question}")
        log.append(f"Entity: {entity_name}")

        # Step 1: Type KoPL生成
        log.append("\n[Step 1] Type KoPL Generation")
        if type_kopl_override:
            type_kopl = type_kopl_override
            log.append("  Using override Type KoPL")
        else:
            type_kopl = self.generate_type_kopl(question)
        log.append(f"  Generated:\n{type_kopl}")

        # Step 2: Type KoPLパース
        log.append("\n[Step 2] Parse Type KoPL")
        parsed = self.parse_kopl(type_kopl)
        log.append(f"  Anchors: {len(parsed.anchors)}")
        log.append(f"  Transitions: {len(parsed.transitions)}")
        for t in parsed.transitions:
            log.append(f"    - {t.anchor_type} --[{t.relation_name}]--> {t.target_type}")

        # Step 3: スキーマグラフ探索
        log.append("\n[Step 3] Schema Graph Search")
        schema_candidates = self.search_schema_candidates(parsed)
        log.append(f"  Found {len(schema_candidates)} candidates")

        # Step 4: Neo4j検証
        log.append("\n[Step 4] Neo4j Validation")
        valid_paths = self.validate_paths_in_neo4j(entity_name, schema_candidates)
        log.append(f"  Valid paths: {len(valid_paths)}/{len(schema_candidates)}")
        for vp in valid_paths[:5]:
            log.append(f"    - {vp['relation_path']}")

        # Step 5: Type KoPLからrelation_name列を抽出
        log.append("\n[Step 5] Extract KoPL Relations")
        kopl_relations = self.extract_kopl_relations(parsed)
        log.append(f"  KoPL relation_names: {kopl_relations}")

        # Step 6: ベクトル剪定（KoPL relations vs 有効パスのrelations）
        scored_paths = []
        if self.config.use_vector_prune and valid_paths and kopl_relations:
            log.append("\n[Step 6] Vector Scoring (KoPL relations vs Valid paths)")
            scored_paths = self.vector_scoring(kopl_relations, valid_paths)
            log.append(f"  Scored {len(scored_paths)} paths")
            for idx, score in scored_paths[:5]:
                log.append(f"    - {valid_paths[idx]['relation_path']}: {score:.4f}")
        else:
            log.append("\n[Step 6] Vector Scoring (skipped)")
            scored_paths = [(i, 0.0) for i in range(len(valid_paths))]

        # Step 7: LLM Reranker
        final_path = None
        if self.config.use_llm_reranker and scored_paths:
            log.append("\n[Step 7] LLM Reranking")
            final_path = self.llm_rerank(question, kopl_relations, valid_paths, scored_paths)
            if final_path:
                log.append(f"  Selected path: {final_path}")
            else:
                log.append("  Reranking failed, using top vector-scored path")
        else:
            log.append("\n[Step 7] LLM Reranking (skipped)")

        # フォールバック: ベクトルスコア最高のパスを使用
        if final_path is None and scored_paths:
            best_idx, _ = scored_paths[0]
            final_path = valid_paths[best_idx]["relation_path"]
            log.append(f"  Fallback to top vector-scored path: {final_path}")

        # Step 7: 最終エンティティ取得
        log.append("\n[Step 7] Final Entity Retrieval")
        reachable_entities = []
        if final_path:
            reachable_entities = self.get_final_entities(entity_name, final_path)
            log.append(f"  Final path: {final_path}")
            log.append(f"  Reachable entities ({len(reachable_entities)}): {reachable_entities[:5]}")
        else:
            log.append("  No valid path found")

        return PipelineResult(
            question=question,
            entity_name=entity_name,
            type_kopl=type_kopl,
            parsed_kopl=parsed,
            schema_candidates=schema_candidates,
            valid_paths=valid_paths,
            vector_scored_paths=scored_paths,
            final_path=final_path,
            reachable_entities=reachable_entities,
            processing_log=log,
        )

    # ────────────────────────────────────────────────────────────
    #  Type KoPL Schema剪定（Intersection用）
    # ────────────────────────────────────────────────────────────
    def _filter_by_type_kopl_schema(
        self,
        candidates: List[Dict[str, Any]],
        anchor_type: str,
        target_types: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Type KoPLのスキーマ情報を使って候補を剪定

        Args:
            candidates: 1-hop候補 [{"relation_path": [rel], "reachable_entities": [...]}]
            anchor_type: アンカーの型（例: "Disease"）
            target_types: 許容されるターゲット型のリスト（例: ["Gene", "Phenotype"]）

        Returns:
            スキーマに合致する候補のみ
        """
        if not target_types:
            return candidates

        # 型名を正規化（大文字小文字両方を試す）
        target_types_normalized = set()
        for t in target_types:
            target_types_normalized.add(t)
            target_types_normalized.add(t.lower())
            target_types_normalized.add(t.capitalize())

        filtered = []
        for cand in candidates:
            rel = cand["relation_path"][0] if cand["relation_path"] else None
            if not rel:
                continue

            # TransitionMapMultiがあれば使用（複数のtarget_typesをサポート）
            if self.transitions_multi:
                actual_targets = self.transitions_multi.get_target_types(
                    current_type=anchor_type, relation=rel
                )
                # actual_targetsのいずれかがtarget_types_normalizedに含まれるか
                if actual_targets:
                    for at in actual_targets:
                        if at in target_types_normalized or at.lower() in target_types_normalized:
                            filtered.append(cand)
                            break
            else:
                # フォールバック: 従来のTransitionMap（単一target）
                actual_target = None
                for at in [anchor_type, anchor_type.lower(), anchor_type.capitalize()]:
                    actual_target = self.transitions.mapping.get((rel, at))
                    if actual_target:
                        break

                if actual_target:
                    if actual_target in target_types_normalized or actual_target.lower() in target_types_normalized:
                        filtered.append(cand)

        return filtered

    def _extract_target_types_for_anchor(
        self,
        parsed: ParsedTypeKoPL,
        anchor_index: int,
    ) -> Tuple[str, List[str]]:
        """
        Type KoPLから特定アンカーに対応する(anchor_type, target_types)を抽出

        Args:
            parsed: パース済みType KoPL
            anchor_index: アンカーのインデックス (0=anchorA, 1=anchorB, ...)

        Returns:
            (anchor_type, [target_type1, target_type2, ...])
        """
        if anchor_index >= len(parsed.anchors):
            return ("", [])

        anchor = parsed.anchors[anchor_index]
        anchor_type = anchor.anchor_type

        # このアンカーに関連する遷移のtarget_typesを収集
        target_types = []
        for trans in parsed.transitions:
            # 遷移のanchor_typeがこのアンカーのtypeと一致する場合
            if trans.anchor_type.lower() == anchor_type.lower():
                target_types.append(trans.target_type)

        return (anchor_type, target_types)

    # ────────────────────────────────────────────────────────────
    #  Single-hop候補取得（Intersection用）
    # ────────────────────────────────────────────────────────────
    def _get_single_hop_candidates(
        self,
        entity_name: str,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        エンティティから1-hopで到達可能な(relation, entities)ペアを取得

        Returns:
            [{"relation_path": [rel], "reachable_entities": [...]}, ...]
        """
        try:
            # エンティティから出ている全relationsを取得（有向）
            query = """
            MATCH (n {name: $entity_name})-[r]->(m)
            RETURN type(r) AS relation, collect(DISTINCT m.name)[..10] AS entities
            """
            result = self.finder.graph.run(query, entity_name=entity_name).data()

            candidates = []
            for row in result:
                rel = row.get("relation")
                entities = row.get("entities", [])
                if rel and entities:
                    candidates.append({
                        "relation_path": [rel],
                        "reachable_entities": [e for e in entities if e],
                    })
            return candidates
        except Exception:
            return []

    def _get_entities_via_relation(
        self,
        entity_name: str,
        relation: str,
        max_results: int = 100,
    ) -> List[str]:
        """
        特定のrelationで到達可能なエンティティを取得
        """
        try:
            query = f"""
            MATCH (n {{name: $entity_name}})-[r:`{relation}`]->(m)
            RETURN DISTINCT m.name AS name
            LIMIT {max_results}
            """
            result = self.finder.graph.run(query, entity_name=entity_name).data()
            return [r["name"] for r in result if r.get("name")]
        except Exception:
            return []

    # ────────────────────────────────────────────────────────────
    #  Intersection パイプライン（複数アンカー対応）
    # ────────────────────────────────────────────────────────────
    def run_intersection(
        self,
        question: str,
        anchors: Dict[str, str],  # {"anchorA": "entity1", "anchorB": "entity2", ...}
        relations: Optional[Dict[str, str]] = None,  # {"anchorA": "rel1", "anchorB": "rel2", ...}
        *,
        type_kopl_override: Optional[str] = None,
    ) -> PipelineResult:
        """
        複数アンカーの交差処理を行うパイプライン

        Args:
            question: 質問文
            anchors: アンカー名とエンティティ名のマッピング
            relations: アンカー名とrelation名のマッピング（オプション、デバッグ用）
            type_kopl_override: Type KoPLの上書き

        Returns:
            交差結果を含むPipelineResult
        """
        log = []
        log.append(f"Question: {question}")
        log.append(f"Anchors: {anchors}")
        if relations:
            log.append(f"Relations (gold): {relations}")

        # Step 1: Type KoPL生成
        log.append("\n[Step 1] Type KoPL Generation")
        if type_kopl_override:
            type_kopl = type_kopl_override
            log.append("  Using override Type KoPL")
        else:
            type_kopl = self.generate_type_kopl(question)
        log.append(f"  Generated:\n{type_kopl}")

        # Step 2: Type KoPLパース
        log.append("\n[Step 2] Parse Type KoPL")
        parsed = self.parse_kopl(type_kopl)
        log.append(f"  Anchors: {len(parsed.anchors)}")
        log.append(f"  Transitions: {len(parsed.transitions)}")
        log.append(f"  And ops: {len(parsed.and_ops)}")
        for t in parsed.transitions:
            log.append(f"    - {t.anchor_type} --[{t.relation_name}]--> {t.target_type}")

        # Step 3: スキーマグラフ探索
        log.append("\n[Step 3] Schema Graph Search")
        schema_candidates = self.search_schema_candidates(parsed)
        log.append(f"  Found {len(schema_candidates)} candidates")

        # Step 4: 各アンカーからの1-hop検索と到達エンティティ取得
        log.append("\n[Step 4] Multi-Anchor Single-Hop Search")
        anchor_results: Dict[str, List[str]] = {}
        intersection_paths: Dict[str, List[str]] = {}

        # Type KoPLから抽出したrelationsを使用
        kopl_relations = self.extract_kopl_relations(parsed)
        log.append(f"  KoPL relations: {kopl_relations}")

        # 各アンカーに対して1-hopパスを検索
        anchor_list = list(anchors.items())
        for i, (anchor_key, entity_name) in enumerate(anchor_list):
            log.append(f"\n  [{anchor_key}] Entity: {entity_name}")

            # Type KoPLからこのアンカーの型情報を抽出
            anchor_type, target_types = self._extract_target_types_for_anchor(parsed, i)
            log.append(f"    Type KoPL: {anchor_type} -> {target_types}")

            # この実体から1-hopで到達可能な候補を取得
            single_hop_paths = self._get_single_hop_candidates(entity_name)
            log.append(f"    Single-hop candidates (raw): {len(single_hop_paths)}")

            if not single_hop_paths:
                log.append(f"    No single-hop paths found")
                anchor_results[anchor_key] = []
                intersection_paths[anchor_key] = []
                continue

            # Step 4a: Type KoPL Schema剪定
            if anchor_type and target_types:
                filtered_paths = self._filter_by_type_kopl_schema(
                    single_hop_paths, anchor_type, target_types
                )
                log.append(f"    After Type KoPL pruning: {len(filtered_paths)}")
                for fp in filtered_paths[:5]:
                    log.append(f"      - {fp['relation_path']}")
            else:
                filtered_paths = single_hop_paths
                log.append(f"    Type KoPL pruning skipped (no type info)")

            if not filtered_paths:
                log.append(f"    No paths after Type KoPL pruning, using all candidates")
                filtered_paths = single_hop_paths

            # Step 4b: ベクトル剪定でtop3に絞る
            scored_paths = []
            if self.config.use_vector_prune and kopl_relations:
                scored_paths = self.vector_scoring(kopl_relations, filtered_paths)
                log.append(f"    After vector scoring (top 3):")
                for idx, score in scored_paths[:3]:
                    log.append(f"      - {filtered_paths[idx]['relation_path']}: {score:.4f}")
            else:
                scored_paths = [(j, 0.0) for j in range(len(filtered_paths))]

            # Step 4c: LLM Rerankerでtop3から最適なrelationを選択
            final_path = None
            top_k_for_rerank = 3  # intersection用はtop3
            if self.config.use_llm_reranker and len(scored_paths) > 0:
                top_paths = [
                    filtered_paths[idx]["relation_path"]
                    for idx, _ in scored_paths[:top_k_for_rerank]
                ]
                log.append(f"    LLM Reranking candidates: {top_paths}")

                queries = [question] + [f"relation: {rel}" for rel in kopl_relations]
                try:
                    result = self.reranker.invoke(queries=queries, paths=top_paths)
                    if hasattr(result, "reranked_combos") and result.reranked_combos:
                        final_path = result.reranked_combos
                        log.append(f"    LLM Reranker selected: {final_path}")
                except Exception as e:
                    log.append(f"    LLM Reranker failed: {e}")

            # フォールバック: ベクトルスコア最高のrelationを使用
            if final_path is None and scored_paths:
                best_idx, _ = scored_paths[0]
                final_path = filtered_paths[best_idx]["relation_path"]
                log.append(f"    Fallback to vector top: {final_path}")

            if final_path:
                intersection_paths[anchor_key] = final_path
                # 選択されたrelationで到達可能なエンティティを取得（多めに）
                selected_rel = final_path[0] if final_path else None
                if selected_rel:
                    reachable = self._get_entities_via_relation(
                        entity_name, selected_rel, max_results=200
                    )
                else:
                    reachable = []
                anchor_results[anchor_key] = reachable
                log.append(f"    Selected relation: {final_path}")
                log.append(f"    Reachable entities: {len(reachable)}")
            else:
                anchor_results[anchor_key] = []
                intersection_paths[anchor_key] = []
                log.append(f"    No path selected")

        # Step 5: 交差計算
        log.append("\n[Step 5] Intersection Calculation")
        all_entity_sets = [set(entities) for entities in anchor_results.values() if entities]

        if len(all_entity_sets) >= 2:
            # 全アンカーの交差を計算
            intersection_set = all_entity_sets[0]
            for entity_set in all_entity_sets[1:]:
                intersection_set = intersection_set & entity_set

            reachable_entities = list(intersection_set)[:self.config.max_results]
            log.append(f"  Anchor sets: {[len(s) for s in all_entity_sets]}")
            log.append(f"  Intersection size: {len(intersection_set)}")
            log.append(f"  Result entities: {reachable_entities[:5]}")
        elif len(all_entity_sets) == 1:
            # アンカーが1つしか有効でない場合はそのまま使用
            reachable_entities = list(all_entity_sets[0])[:self.config.max_results]
            log.append(f"  Only one anchor with results, using directly")
            log.append(f"  Result entities: {reachable_entities[:5]}")
        else:
            reachable_entities = []
            log.append(f"  No valid anchor results for intersection")

        # 最初のアンカーの情報を代表として使用
        first_anchor_key = anchor_list[0][0] if anchor_list else ""
        first_entity = anchors.get(first_anchor_key, "")
        final_path = intersection_paths.get(first_anchor_key)

        return PipelineResult(
            question=question,
            entity_name=first_entity,
            type_kopl=type_kopl,
            parsed_kopl=parsed,
            schema_candidates=schema_candidates,
            valid_paths=[],  # 各アンカーのパスはintersection_pathsに格納
            vector_scored_paths=[],
            final_path=final_path,
            reachable_entities=reachable_entities,
            processing_log=log,
            anchor_results=anchor_results,
            intersection_paths=intersection_paths,
        )


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Type KoPLベースのGraph QAパイプライン")
    p.add_argument("--question", type=str, required=True, help="質問文")
    p.add_argument("--entity", type=str, required=True, help="開始エンティティ名")
    p.add_argument(
        "--nodes-csv",
        type=Path,
        default=Path("data/kg/nodes.csv"),
    )
    p.add_argument(
        "--rels-csv",
        type=Path,
        default=Path("data/kg/relationships.csv"),
    )
    p.add_argument("--no-vector-prune", action="store_true", help="ベクトル剪定を無効化")
    p.add_argument("--no-llm-rerank", action="store_true", help="LLM Rerankerを無効化")
    p.add_argument("--verbose", "-v", action="store_true", help="詳細ログを表示")

    args = p.parse_args()

    # TransitionMap構築
    print("Building TransitionMap...")
    transitions = build_transition_map_from_kg(
        nodes_csv=args.nodes_csv,
        rels_csv=args.rels_csv,
    )

    # パイプライン設定
    config = TypeKoPLPipelineConfig(
        use_vector_prune=not args.no_vector_prune,
        use_llm_reranker=not args.no_llm_rerank,
    )

    # パイプライン実行
    pipeline = TypeKoPLPipeline(transitions=transitions, config=config)
    result = pipeline.run(question=args.question, entity_name=args.entity)

    # 結果表示
    if args.verbose:
        print("\n" + "=" * 60)
        print("Processing Log")
        print("=" * 60)
        for line in result.processing_log:
            print(line)

    print("\n" + "=" * 60)
    print("Final Result")
    print("=" * 60)
    print(f"Question: {result.question}")
    print(f"Entity: {result.entity_name}")
    print(f"Final Path: {result.final_path}")
    print(f"Reachable Entities: {result.reachable_entities}")


if __name__ == "__main__":
    main()
