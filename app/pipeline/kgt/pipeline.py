"""
KGT (Knowledge Graph Transformer) Pipeline

KGT論文に基づく検索パイプライン:
1. Question Analysis - 質問からエンティティ・タイプ情報を抽出
2. Schema-Based Path Finding - BFS + ベクトル類似度でパス選択
3. Cypher Query Generation - 最適パスからCypherクエリを生成
4. Subgraph Retrieval & Answer Generation - サブグラフ取得と回答生成
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import BASEMODEL, get_settings
from database.search import GraphPathFinder
from pipeline.kgt.models import (
    QuestionAnalysis,
    SchemaPath,
    KGTResult,
    QuestionAnalysisResponse,
)
from pipeline.kgt.schema import KGSchema, build_schema_graph


class KGTPipeline:
    """KGTパイプライン"""

    # KGタイプごとのエンティティタイプリスト
    ENTITY_TYPES = {
        "primekgqa": [
            "anatomy", "biological_process", "cellular_component", "disease",
            "drug", "effect/phenotype", "exposure", "gene/protein",
            "molecular_function", "pathway"
        ],
        "metaqa": [
            "Movie", "Person", "Organization", "Text", "Date", "Language", "Number"
        ],
    }

    def __init__(
        self,
        schema: Optional[KGSchema] = None,
        model: str = BASEMODEL,
        embedding_model: str = "text-embedding-3-small",
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
        self.schema = schema or build_schema_graph(kg_type)
        self.finder = GraphPathFinder(kg_type=kg_type)

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

    def _get_analysis_examples(self) -> str:
        """KGタイプに応じた分析例を返す"""
        if self.kg_type == "metaqa":
            return """Examples:
- "What movies did Tom Hanks star in?" → tail_entity_type: "Movie"
- "Who directed Titanic?" → tail_entity_type: "Person" or "Organization"
- "Who directed the movies that Tom Hanks starred in?" → tail_entity_type: "Person" (NOT Movie)
- "What year was Titanic released?" → tail_entity_type: "Date"
- "What genre is The Matrix?" → tail_entity_type: "Text\""""
        else:
            return """Examples:
- "What diseases are associated with BRCA1?" → tail_entity_type: "disease"
- "Which genes does Tacrolimus target?" → tail_entity_type: "gene/protein"
- "Which diseases are linked to genes targeted by Tacrolimus?" → tail_entity_type: "disease" (NOT gene/protein)
- "Which drugs target genes linked to heart failure?" → tail_entity_type: "drug"
- "What conditions is Aspirin indicated for?" → tail_entity_type: "disease\""""

    def _analyze_question(
        self,
        question: str,
        entity_name: Optional[str] = None
    ) -> QuestionAnalysis:
        """Phase 1: 質問分析"""

        type_list = ", ".join(self.ENTITY_TYPES.get(self.kg_type, []))
        examples = self._get_analysis_examples()

        # エンティティ名が指定されていない場合はLLMで抽出
        if entity_name:
            head_name = entity_name
            # LLMでテールタイプのみ抽出
            llm_with_output = self.llm.with_structured_output(QuestionAnalysisResponse)
            prompt = f"""Analyze this question about a knowledge graph.

Question: {question}
Known head entity: {entity_name}

Identify:
1. tail_entity_type: What type of entity is being asked about (the FINAL answer type)?
   Choose from: {type_list}
2. tail_attributes: Any specific attributes or constraints mentioned

{examples}

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
            prompt = f"""Analyze this question about a knowledge graph.

Question: {question}

Identify:
1. head_entity_name: The main entity mentioned in the question (the starting point/anchor)
2. tail_entity_type: What type of entity is being asked about (the FINAL answer type)?
   Choose from: {type_list}
3. tail_attributes: Any specific attributes or constraints mentioned

{examples}

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
        return self._build_cypher(analysis, path)

    def _build_cypher(self, analysis: QuestionAnalysis, path: SchemaPath) -> str:
        """テンプレートベースのCypher構築（方向を考慮）"""

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        def get_direction(idx: int) -> str:
            """パスのidx番目のエッジの方向を取得"""
            if path.directions and idx < len(path.directions):
                return path.directions[idx]
            return "->"  # デフォルトは順方向

        head_name = analysis.head_entity_name

        if len(path.types) == 2:
            # 1-hop
            direction = get_direction(0)
            if direction == "<-":
                # Reverse: 実際のエッジは (tgt)-[r]->(src) なので、アンカーを右側に
                return f"""
                MATCH (t:{get_label(path.types[1])})-[r:{path.relations[0]}]->(h:{get_label(path.types[0])})
                WHERE h.name = "{head_name}"
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
            else:
                # Forward: (src)-[r]->(tgt)
                return f"""
                MATCH (h:{get_label(path.types[0])})-[r:{path.relations[0]}]->(t:{get_label(path.types[1])})
                WHERE h.name = "{head_name}"
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
        elif len(path.types) == 3:
            # 2-hop: 各エッジの方向を個別にチェック
            dir1 = get_direction(0)
            dir2 = get_direction(1)

            if dir1 == "<-" and dir2 == "<-":
                # Both reverse: (t)->(m)->(h)
                return f"""
                MATCH (t:{get_label(path.types[2])})-[r2:{path.relations[1]}]->(m:{get_label(path.types[1])})-[r1:{path.relations[0]}]->(h:{get_label(path.types[0])})
                WHERE h.name = "{head_name}"
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
            elif dir1 == "<-" and dir2 == "->":
                # First reverse, second forward: (m)->(h), (m)->(t)
                return f"""
                MATCH (m:{get_label(path.types[1])})-[r1:{path.relations[0]}]->(h:{get_label(path.types[0])})
                MATCH (m)-[r2:{path.relations[1]}]->(t:{get_label(path.types[2])})
                WHERE h.name = "{head_name}"
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
            elif dir1 == "->" and dir2 == "<-":
                # First forward, second reverse: (h)->(m), (t)->(m)
                return f"""
                MATCH (h:{get_label(path.types[0])})-[r1:{path.relations[0]}]->(m:{get_label(path.types[1])})
                MATCH (t:{get_label(path.types[2])})-[r2:{path.relations[1]}]->(m)
                WHERE h.name = "{head_name}"
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
            else:
                # Both forward: (h)->(m)->(t)
                return f"""
                MATCH (h:{get_label(path.types[0])})-[r1:{path.relations[0]}]->(m:{get_label(path.types[1])})-[r2:{path.relations[1]}]->(t:{get_label(path.types[2])})
                WHERE h.name = "{head_name}"
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
        else:
            # 3-hop以上: 方向を考慮した動的パターン構築
            # 簡略化のため順方向のみ対応（逆方向は3-hop以上では稀）
            pattern_parts = [f"(n0:{get_label(path.types[0])})"]
            for i, rel in enumerate(path.relations):
                pattern_parts.append(f"-[r{i}:{rel}]->(n{i+1}:{get_label(path.types[i+1])})")
            pattern = "".join(pattern_parts)

            return f"""
            MATCH {pattern}
            WHERE n0.name = "{head_name}"
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

        return entities, None
