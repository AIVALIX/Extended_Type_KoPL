"""
KGT (Knowledge Graph Transformer) Pipeline

KGT論文に基づく検索パイプライン:
1. Question Analysis - 質問からエンティティ・タイプ情報を抽出
2. Schema-Based Path Finding - BFS + ベクトル類似度でパス選択
3. Cypher Query Generation - 最適パスからCypherクエリを生成
4. Subgraph Retrieval & Answer Generation - サブグラフ取得と回答生成
"""

from __future__ import annotations

import json
import os
from collections import deque
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
class QuestionAnalysis:
    """質問分析結果"""
    head_entity_name: str  # H_n: ヘッドエンティティ名
    head_entity_type: Optional[str] = None  # H_t: ヘッドエンティティタイプ（DBから取得）
    tail_entity_type: Optional[str] = None  # T_t: テールエンティティタイプ
    tail_attributes: List[str] = field(default_factory=list)  # T_a: テールエンティティ属性
    head_entity_id: Optional[str] = None  # DBで見つかったエンティティID


@dataclass
class SchemaPath:
    """スキーマパス"""
    path: List[str]  # [type1, rel1, type2, rel2, type3, ...]
    types: List[str]  # [type1, type2, type3, ...]
    relations: List[str]  # [rel1, rel2, ...]
    score: float = 0.0  # ベクトル類似度スコア


@dataclass
class KGTResult:
    """KGTパイプライン結果"""
    question: str
    analysis: QuestionAnalysis
    schema_paths: List[SchemaPath]
    optimal_path: Optional[SchemaPath]
    generated_cypher: Optional[str]
    subgraph: List[Dict[str, Any]]
    answer_entities: List[str]
    natural_answer: Optional[str] = None
    processing_log: List[str] = field(default_factory=list)


# =============================================================================
# Pydantic Models for LLM Structured Output
# =============================================================================

class QuestionAnalysisResponse(BaseModel):
    """LLMによる質問分析の出力"""
    head_entity_name: str = Field(..., description="The main entity mentioned in the question (anchor)")
    tail_entity_type: str = Field(..., description="The type of entity being asked about (e.g., disease, drug, gene/protein)")
    tail_attributes: List[str] = Field(default_factory=list, description="Any specific attributes or constraints for the answer")


class CypherQueryResponse(BaseModel):
    """LLMによるCypherクエリ生成の出力"""
    cypher_query: str = Field(..., description="The Cypher query to retrieve the subgraph")
    explanation: str = Field(default="", description="Brief explanation of the query")


class PrunedAnswerResponse(BaseModel):
    """LLMによる回答生成の出力"""
    relevant_entities: List[str] = Field(..., description="List of relevant entity names that answer the question")
    natural_answer: str = Field(..., description="Natural language answer to the question")


# =============================================================================
# Schema Graph
# =============================================================================

class SchemaGraph:
    """スキーマグラフ（エンティティタイプとリレーションの無向グラフ）"""

    def __init__(self):
        self.adjacency: Dict[str, List[Tuple[str, str]]] = {}  # type -> [(neighbor_type, relation), ...]
        self.all_types: Set[str] = set()
        self.all_relations: Set[str] = set()

    def add_edge(self, src_type: str, relation: str, tgt_type: str):
        """エッジを追加（有向）"""
        self.all_types.add(src_type)
        self.all_types.add(tgt_type)
        self.all_relations.add(relation)

        if src_type not in self.adjacency:
            self.adjacency[src_type] = []
        if tgt_type not in self.adjacency:
            self.adjacency[tgt_type] = []

        # 有向グラフとして順方向のみ追加
        self.adjacency[src_type].append((tgt_type, relation))

    def bfs_all_shortest_paths(
        self,
        start_type: str,
        end_type: str,
        max_depth: int = 4
    ) -> List[SchemaPath]:
        """BFSで最短パスを探索（shortest_onlyモードでは最短のみ）"""
        return self.find_all_paths(start_type, end_type, max_depth, shortest_only=True)

    def find_all_paths(
        self,
        start_type: str,
        end_type: str,
        max_depth: int = 3,
        shortest_only: bool = False
    ) -> List[SchemaPath]:
        """全てのパスを探索（DFS）"""
        if start_type not in self.adjacency or end_type not in self.adjacency:
            return []

        all_paths: List[SchemaPath] = []
        visited = {start_type}

        def dfs(current: str, type_path: List[str], rel_path: List[str], depth: int):
            if depth > max_depth:
                return

            # 目的地に到達
            if current == end_type and depth > 0:
                full_path = []
                for i, t in enumerate(type_path):
                    full_path.append(t)
                    if i < len(rel_path):
                        full_path.append(rel_path[i])
                all_paths.append(SchemaPath(
                    path=full_path,
                    types=type_path.copy(),
                    relations=rel_path.copy()
                ))
                return

            # 隣接ノードを探索
            for neighbor, relation in self.adjacency.get(current, []):
                if neighbor not in visited or neighbor == end_type:
                    visited.add(neighbor)
                    dfs(
                        neighbor,
                        type_path + [neighbor],
                        rel_path + [relation],
                        depth + 1
                    )
                    if neighbor != end_type:
                        visited.discard(neighbor)

        dfs(start_type, [start_type], [], 0)

        # 深さでソート
        all_paths.sort(key=lambda p: len(p.relations))

        if shortest_only and all_paths:
            min_depth = len(all_paths[0].relations)
            all_paths = [p for p in all_paths if len(p.relations) == min_depth]

        return all_paths


def build_schema_graph_from_neo4j() -> SchemaGraph:
    """スキーマグラフを構築（schema_v2.pyから）"""
    from dataset_construction.schema_v2 import SCHEMA_GRAPH

    schema = SchemaGraph()

    for src, rel, tgt, _ in SCHEMA_GRAPH:
        schema.add_edge(src, rel, tgt)

    return schema


# =============================================================================
# KGT Pipeline
# =============================================================================

class KGTPipeline:
    """KGTパイプライン"""

    def __init__(
        self,
        schema: Optional[SchemaGraph] = None,
        model: str = BASEMODEL,
        embedding_model: str = "text-embedding-3-small",
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings

        self.llm = init_chat_model(model, model_provider="openai", temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.schema = schema or build_schema_graph_from_neo4j()
        self.finder = GraphPathFinder()

    def run(self, question: str, entity_name: Optional[str] = None) -> KGTResult:
        """パイプライン実行"""
        log = []
        log.append(f"Input question: {question}")

        # 1. Question Analysis
        log.append("Phase 1: Question Analysis")
        analysis = self._analyze_question(question, entity_name)
        log.append(f"  Head entity: {analysis.head_entity_name} (type: {analysis.head_entity_type})")
        log.append(f"  Tail type: {analysis.tail_entity_type}")

        # ヘッドエンティティタイプが取得できない場合
        if not analysis.head_entity_type:
            log.append("  ERROR: Could not determine head entity type")
            return KGTResult(
                question=question,
                analysis=analysis,
                schema_paths=[],
                optimal_path=None,
                generated_cypher=None,
                subgraph=[],
                answer_entities=[],
                processing_log=log,
            )

        # 2. Schema-Based Path Finding
        log.append("Phase 2: Schema-Based Path Finding")
        schema_paths = self._find_schema_paths(
            analysis.head_entity_type,
            analysis.tail_entity_type,
            question
        )
        log.append(f"  Found {len(schema_paths)} candidate paths")

        if not schema_paths:
            log.append("  ERROR: No paths found in schema")
            return KGTResult(
                question=question,
                analysis=analysis,
                schema_paths=[],
                optimal_path=None,
                generated_cypher=None,
                subgraph=[],
                answer_entities=[],
                processing_log=log,
            )

        # 最適パス選択
        optimal_path = schema_paths[0]  # スコア順でソート済み
        log.append(f"  Optimal path: {optimal_path.path} (score: {optimal_path.score:.3f})")

        # 3. Cypher Query Generation
        log.append("Phase 3: Cypher Query Generation")
        cypher_query = self._generate_cypher(analysis, optimal_path)
        log.append(f"  Generated Cypher: {cypher_query[:100]}...")

        # 4. Subgraph Retrieval
        log.append("Phase 4: Subgraph Retrieval")
        subgraph = self._retrieve_subgraph(cypher_query)
        log.append(f"  Retrieved {len(subgraph)} results")

        # 5. Answer Generation
        log.append("Phase 5: Answer Generation")
        answer_entities, natural_answer = self._generate_answer(question, subgraph)
        log.append(f"  Found {len(answer_entities)} answer entities")

        return KGTResult(
            question=question,
            analysis=analysis,
            schema_paths=schema_paths,
            optimal_path=optimal_path,
            generated_cypher=cypher_query,
            subgraph=subgraph,
            answer_entities=answer_entities,
            natural_answer=natural_answer,
            processing_log=log,
        )

    def _analyze_question(
        self,
        question: str,
        entity_name: Optional[str] = None
    ) -> QuestionAnalysis:
        """Phase 1: 質問分析"""

        # エンティティ名が指定されていない場合はLLMで抽出
        if entity_name:
            head_name = entity_name
            # LLMでテールタイプのみ抽出
            llm_with_output = self.llm.with_structured_output(QuestionAnalysisResponse)
            prompt = f"""Analyze this question about a knowledge graph.

Question: {question}
Known head entity: {entity_name}

Identify:
1. tail_entity_type: What type of entity is being asked about?
   Choose from: anatomy, biological_process, cellular_component, disease, drug, effect/phenotype, exposure, gene/protein, molecular_function, pathway
2. tail_attributes: Any specific attributes or constraints mentioned

Return JSON."""

            try:
                result = llm_with_output.invoke(prompt)
                tail_type = result.tail_entity_type
                tail_attrs = result.tail_attributes
            except Exception:
                tail_type = None
                tail_attrs = []
        else:
            # LLMで全て抽出
            llm_with_output = self.llm.with_structured_output(QuestionAnalysisResponse)
            prompt = f"""Analyze this question about a biomedical knowledge graph.

Question: {question}

Identify:
1. head_entity_name: The main entity mentioned in the question (the starting point)
2. tail_entity_type: What type of entity is being asked about?
   Choose from: anatomy, biological_process, cellular_component, disease, drug, effect/phenotype, exposure, gene/protein, molecular_function, pathway
3. tail_attributes: Any specific attributes or constraints mentioned

Return JSON."""

            try:
                result = llm_with_output.invoke(prompt)
                head_name = result.head_entity_name
                tail_type = result.tail_entity_type
                tail_attrs = result.tail_attributes
            except Exception as e:
                return QuestionAnalysis(head_entity_name="", tail_entity_type=None)

        # DBからヘッドエンティティのタイプを取得
        head_type, head_id = self._lookup_entity_type(head_name)

        return QuestionAnalysis(
            head_entity_name=head_name,
            head_entity_type=head_type,
            tail_entity_type=tail_type,
            tail_attributes=tail_attrs,
            head_entity_id=head_id,
        )

    def _lookup_entity_type(self, entity_name: str) -> Tuple[Optional[str], Optional[str]]:
        """DBからエンティティタイプを検索"""
        graph = self.finder.graph

        # 完全一致検索
        cypher = """
        MATCH (n)
        WHERE n.name = $name
        RETURN elementId(n) AS id, [l IN labels(n) WHERE l <> '_Entity'][0] AS type
        LIMIT 1
        """

        try:
            result = graph.run(cypher, name=entity_name).data()
            if result:
                return result[0]["type"], result[0]["id"]
        except Exception:
            pass

        # 部分一致検索
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($name)
        RETURN elementId(n) AS id, [l IN labels(n) WHERE l <> '_Entity'][0] AS type
        LIMIT 1
        """

        try:
            result = graph.run(cypher, name=entity_name).data()
            if result:
                return result[0]["type"], result[0]["id"]
        except Exception:
            pass

        return None, None

    def _find_schema_paths(
        self,
        head_type: str,
        tail_type: Optional[str],
        question: str
    ) -> List[SchemaPath]:
        """Phase 2: スキーマベースのパス探索（全深度）"""

        # テールタイプが指定されていない場合は全タイプを候補に
        if not tail_type:
            target_types = list(self.schema.all_types)
        else:
            target_types = [tail_type]

        all_paths = []
        for t_type in target_types:
            # 最短パスのみを探索
            paths = self.schema.find_all_paths(head_type, t_type, max_depth=3, shortest_only=True)
            all_paths.extend(paths)

        if not all_paths:
            return []

        # ベクトル類似度でスコアリング
        question_embedding = self.embeddings.embed_query(question)

        path_texts = []
        for p in all_paths:
            # パスをテキスト化
            text = " -> ".join(p.path)
            path_texts.append(text)

        path_embeddings = self.embeddings.embed_documents(path_texts)

        # コサイン類似度計算
        q_vec = np.array(question_embedding)
        for i, p in enumerate(all_paths):
            p_vec = np.array(path_embeddings[i])
            similarity = np.dot(q_vec, p_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(p_vec))
            p.score = float(similarity)

        # スコア順にソート
        all_paths.sort(key=lambda x: x.score, reverse=True)

        return all_paths

    def _generate_cypher(self, analysis: QuestionAnalysis, path: SchemaPath) -> str:
        """Phase 3: Cypherクエリ生成（テンプレートベース）"""
        # LLM生成は信頼性が低いため、テンプレートベースで構築
        return self._build_fallback_cypher(analysis, path)

    def _build_fallback_cypher(self, analysis: QuestionAnalysis, path: SchemaPath) -> str:
        """テンプレートベースのCypher構築（無向マッチング）"""

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        if len(path.types) == 2:
            # 1-hop（有向マッチング）
            return f"""
            MATCH (h:{get_label(path.types[0])})-[r:{path.relations[0]}]->(t:{get_label(path.types[1])})
            WHERE h.name = "{analysis.head_entity_name}"
            RETURN DISTINCT t.name AS answer
            LIMIT 50
            """
        elif len(path.types) == 3:
            # 2-hop（有向マッチング）
            return f"""
            MATCH (h:{get_label(path.types[0])})-[r1:{path.relations[0]}]->(m:{get_label(path.types[1])})-[r2:{path.relations[1]}]->(t:{get_label(path.types[2])})
            WHERE h.name = "{analysis.head_entity_name}"
            RETURN DISTINCT t.name AS answer
            LIMIT 50
            """
        else:
            # 3-hop以上（有向マッチング）
            pattern_parts = [f"(n0:{get_label(path.types[0])})"]
            for i, rel in enumerate(path.relations):
                pattern_parts.append(f"-[r{i}:{rel}]->(n{i+1}:{get_label(path.types[i+1])})")
            pattern = "".join(pattern_parts)

            return f"""
            MATCH {pattern}
            WHERE n0.name = "{analysis.head_entity_name}"
            RETURN DISTINCT n{len(path.relations)}.name AS answer
            LIMIT 50
            """

    def _retrieve_subgraph(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Phase 4: サブグラフ取得"""
        graph = self.finder.graph

        try:
            results = graph.run(cypher_query).data()
            return results
        except Exception as e:
            print(f"Cypher execution error: {e}")
            return []

    def _generate_answer(
        self,
        question: str,
        subgraph: List[Dict[str, Any]]
    ) -> Tuple[List[str], Optional[str]]:
        """Phase 5: 回答生成（全エンティティを返す）"""

        if not subgraph:
            return [], None

        # サブグラフからエンティティを抽出
        entities = []
        for record in subgraph:
            for key, value in record.items():
                if isinstance(value, str) and key.lower() == "answer":
                    entities.append(value)
                elif isinstance(value, dict) and "name" in value:
                    entities.append(value["name"])

        # 重複除去
        entities = list(dict.fromkeys(entities))

        # LLMによるフィルタリングは行わず、全エンティティを返す
        # （パス選択の段階で絞り込みは完了している）
        return entities, None


# =============================================================================
# Main
# =============================================================================

def main():
    """テスト実行"""
    import argparse

    parser = argparse.ArgumentParser(description="KGT Pipeline Test")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    pipeline = KGTPipeline()
    result = pipeline.run(args.question, args.entity)

    print("\n" + "=" * 60)
    print("KGT Pipeline Result")
    print("=" * 60)

    for log_line in result.processing_log:
        print(log_line)

    print("\n--- Answer ---")
    print(f"Entities: {result.answer_entities[:10]}")
    if result.natural_answer:
        print(f"Natural: {result.natural_answer}")


if __name__ == "__main__":
    main()
