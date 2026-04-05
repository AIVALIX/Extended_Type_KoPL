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
        "pcqa": [
            "Cancer", "CancerCell", "CancerAlias", "Drug", "DrugAlias",
            "Genesymbol", "SnvFull", "Fusion", "GeneticDisease", "ClinicalTrial"
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

    def run(self, question: str, entity_name: Optional[str] = None, generate_nl: bool = False) -> KGTResult:
        """パイプライン実行

        Args:
            question: 入力質問
            entity_name: エンティティ名（省略時はLLMで抽出）
            generate_nl: Trueの場合、サブグラフからLLMで自然言語回答を生成
        """
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

        # 6. Natural Language Answer Generation (optional)
        if generate_nl:
            log.append("Phase 6: NL Answer Generation")
            chains = self._retrieve_subgraph_chains(analysis, optimal_path)
            log.append(f"  Retrieved {len(chains)} relationship chains")
            natural_answer = self._generate_natural_answer(question, chains)
            log.append(f"  Generated NL answer: {natural_answer[:80] if natural_answer else 'None'}...")

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

1. 1-hop: "Who directed Titanic?" → head_entity_name: "Titanic", tail_entity_type: "Person"

2. 2-hop: "Who directed the movies that Tom Hanks starred in?" → head_entity_name: "Tom Hanks", tail_entity_type: "Person" (the director, NOT Movie)

3. 3-hop: "Who starred in the movies written by the writers of The Matrix?" → head_entity_name: "The Matrix", tail_entity_type: "Person" (the actors)

IMPORTANT: For multi-hop questions, tail_entity_type should be the FINAL answer type, not intermediate types."""
        elif self.kg_type == "pcqa":
            return """Examples:
- "What drugs can treat lung cancer?" → head_entity_name: "lung cancer", tail_entity_type: "Drug"
- "What types of cancer can EGFR drive?" → head_entity_name: "EGFR", tail_entity_type: "Cancer"
- "What genetic mutations are in breast cancer?" → head_entity_name: "breast cancer", tail_entity_type: "SnvFull"
- "Which genes does gefitinib inhibit?" → head_entity_name: "gefitinib", tail_entity_type: "Genesymbol"
- "What drugs can treat cancers with TERT mutations?" → head_entity_name: "TERT" (NOT "TERT mutations"), tail_entity_type: "Drug"
- "What NMPA-approved drugs for cancers with DDR2 mutations?" → head_entity_name: "DDR2" (extract gene name only), tail_entity_type: "Drug"
- "What cancers can be treated with drugs that target BRAF?" → head_entity_name: "BRAF", tail_entity_type: "Cancer"

IMPORTANT for head_entity_name extraction:
- For "cancers with X mutations", extract just "X" (the gene name)
- For "X mutations", extract just "X"
- Remove words like "mutations", "cancers with", "gene" from the entity name
- The entity should be a single word that exists in the knowledge graph (gene, drug, or cancer name)"""
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
        head_type, head_id, matched_name = self._lookup_entity_type(head_name)

        # マッチした名前があればそれを使用（部分一致の場合に重要）
        if matched_name:
            head_name = matched_name

        # PcQA: CancerCell複合名マッチング
        compound_names: list = []
        compound_search_term: Optional[str] = None
        if self.kg_type == "pcqa":
            if head_type in ("Genesymbol", "Fusion"):
                gene_name = head_name  # 元の遺伝子名を保持
                from pipeline.common.pcqa import resolve_pcqa_compound_entity, CANCERCELL_KEYWORDS
                compound = resolve_pcqa_compound_entity(
                    self.llm, self.finder, question, head_name, head_type
                )
                if compound:
                    compound_name, compound_type, compound_names = compound
                    head_name = compound_name
                    head_type = compound_type
                    head_id = None
                    compound_search_term = gene_name
                elif any(kw in question.lower() for kw in CANCERCELL_KEYWORDS):
                    # Compound resolution失敗でもキーワードからCancerCellと推定
                    head_type = "CancerCell"
                    compound_search_term = gene_name
            elif head_type == "CancerCell" and entity_name and entity_name.lower() != head_name.lower():
                # 部分一致でCancerCellが返された場合（例: "alk" → "ALK-...-CancerCell"）
                compound_search_term = entity_name  # 元の遺伝子名でCONTAINS検索

        return QuestionAnalysis(
            head_entity_name=head_name,
            head_entity_type=head_type,
            tail_entity_type=tail_type,
            tail_attributes=tail_attrs,
            head_entity_id=head_id,
            compound_names=compound_names,
            compound_search_term=compound_search_term,
        )

    def _lookup_entity_type(self, entity_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """DBからエンティティタイプを検索

        Returns:
            (type, id, matched_name) - タイプ、ID、実際にマッチした名前
        """
        graph = self.finder.graph

        # 完全一致検索
        cypher = """
        MATCH (n)
        WHERE n.name = $name
        RETURN elementId(n) AS id, [l IN labels(n) WHERE l <> '_Entity'][0] AS type, n.name AS name
        LIMIT 1
        """

        try:
            result = graph.run(cypher, name=entity_name).data()
            if result:
                return result[0]["type"], result[0]["id"], result[0]["name"]
        except Exception:
            pass

        # 大文字小文字を無視した完全一致検索
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($name)
        RETURN elementId(n) AS id, [l IN labels(n) WHERE l <> '_Entity'][0] AS type, n.name AS name
        LIMIT 1
        """

        try:
            result = graph.run(cypher, name=entity_name).data()
            if result:
                return result[0]["type"], result[0]["id"], result[0]["name"]
        except Exception:
            pass

        # 部分一致検索（短い名前を優先）
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($name)
        RETURN elementId(n) AS id, [l IN labels(n) WHERE l <> '_Entity'][0] AS type, n.name AS name
        ORDER BY length(n.name)
        LIMIT 1
        """

        try:
            result = graph.run(cypher, name=entity_name).data()
            if result:
                return result[0]["type"], result[0]["id"], result[0]["name"]
        except Exception:
            pass

        return None, None, None

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
            # 論文準拠: Cypher記法でリレーションチェーンを表現
            # 例: "(Drug)-[:inhibition_to {}]->(Genesymbol)"
            edges = []
            for i, rel in enumerate(p.relations):
                src = p.types[i]
                tgt = p.types[i + 1]
                direction = p.directions[i] if i < len(p.directions) else "->"
                rel_lower = rel.lower()
                if direction == "->":
                    edges.append(f"({src})-[:{rel_lower} {{}}]->({tgt})")
                else:
                    edges.append(f"({tgt})-[:{rel_lower} {{}}]->({src})")
            text = ",".join(edges)
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

        def get_rel(r: str) -> str:
            """リレーション名をエスケープ（スペースやハイフンを含む場合）"""
            if " " in r or "-" in r or "/" in r:
                return f"`{r}`"
            return r

        def get_direction(idx: int) -> str:
            """パスのidx番目のエッジの方向を取得"""
            if path.directions and idx < len(path.directions):
                return path.directions[idx]
            return "->"  # デフォルトは順方向

        def where_head(var: str) -> str:
            """ヘッドエンティティのWHERE句（compound対応）"""
            if analysis.compound_search_term:
                # PcQA CancerCell: CONTAINS検索で全マッチを対象にする
                return f'toLower({var}.name) CONTAINS toLower("{analysis.compound_search_term}")'
            if analysis.compound_names:
                names_str = ", ".join(f'"{n}"' for n in analysis.compound_names)
                return f'{var}.name IN [{names_str}]'
            return f'{var}.name = "{analysis.head_entity_name}"'

        head_name = analysis.head_entity_name

        if len(path.types) == 2:
            # 1-hop
            direction = get_direction(0)
            if direction == "<-":
                # Reverse: 実際のエッジは (tgt)-[r]->(src) なので、アンカーを右側に
                return f"""
                MATCH (t:{get_label(path.types[1])})-[r:{get_rel(path.relations[0])}]->(h:{get_label(path.types[0])})
                WHERE {where_head("h")}
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
            else:
                # Forward: (src)-[r]->(tgt)
                return f"""
                MATCH (h:{get_label(path.types[0])})-[r:{get_rel(path.relations[0])}]->(t:{get_label(path.types[1])})
                WHERE {where_head("h")}
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
                MATCH (t:{get_label(path.types[2])})-[r2:{get_rel(path.relations[1])}]->(m:{get_label(path.types[1])})-[r1:{get_rel(path.relations[0])}]->(h:{get_label(path.types[0])})
                WHERE {where_head("h")}
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
            elif dir1 == "<-" and dir2 == "->":
                # First reverse, second forward: (m)->(h), (m)->(t)
                return f"""
                MATCH (m:{get_label(path.types[1])})-[r1:{get_rel(path.relations[0])}]->(h:{get_label(path.types[0])})
                MATCH (m)-[r2:{get_rel(path.relations[1])}]->(t:{get_label(path.types[2])})
                WHERE {where_head("h")}
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
            elif dir1 == "->" and dir2 == "<-":
                # First forward, second reverse: (h)->(m), (t)->(m)
                return f"""
                MATCH (h:{get_label(path.types[0])})-[r1:{get_rel(path.relations[0])}]->(m:{get_label(path.types[1])})
                MATCH (t:{get_label(path.types[2])})-[r2:{get_rel(path.relations[1])}]->(m)
                WHERE {where_head("h")}
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
            else:
                # Both forward: (h)->(m)->(t)
                return f"""
                MATCH (h:{get_label(path.types[0])})-[r1:{get_rel(path.relations[0])}]->(m:{get_label(path.types[1])})-[r2:{get_rel(path.relations[1])}]->(t:{get_label(path.types[2])})
                WHERE {where_head("h")}
                RETURN DISTINCT t.name AS answer
                LIMIT 50
                """
        else:
            # 3-hop以上: 方向を考慮した動的パターン構築
            # 簡略化のため順方向のみ対応（逆方向は3-hop以上では稀）
            pattern_parts = [f"(n0:{get_label(path.types[0])})"]
            for i, rel in enumerate(path.relations):
                pattern_parts.append(f"-[r{i}:{get_rel(rel)}]->(n{i+1}:{get_label(path.types[i+1])})")
            pattern = "".join(pattern_parts)

            return f"""
            MATCH {pattern}
            WHERE {where_head("n0")}
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

    # =========================================================================
    # Phase 6: Natural Language Answer Generation (KGT論文準拠)
    # =========================================================================

    # Shared NLG prompt (aligned with ETK for fair comparison)
    # Based on KGT paper's 2-step CoT approach with English few-shot examples
    _NL_INFERENCE_PROMPT = """You are a reasoning robot, and you need to perform the following two steps step by step:
1. Output a corresponding natural language sentence for each relationship chain.
2. Answer my question using natural language from step 1.
Note: The output format is: Output: One sentence in natural language.

IMPORTANT RULES:
- Property values YES/true/True = approved, NO/false/False = not approved
- Filter by fda_approved/nmpa_approved when the question asks about approval status
- If no entities match the criteria, state that none exist

=== EXAMPLES ===

Example 1 - Cancer association:
Subgraph:
(MET)-[:DRIVING_TO]->(low-grade glioma)
(MET)-[:DRIVING_TO]->(renal clear cell carcinoma)
Question: Which types of cancer are associated with MET?
Step 1: MET drives low-grade glioma. MET drives renal clear cell carcinoma.
Output: MET is associated with low-grade glioma and renal clear cell carcinoma.

Example 2 - Drug treatment:
Subgraph:
(diethylstilbestrol)-[:TREATMENT]->(breast cancer)
(diethylstilbestrol)-[:TREATMENT]->(prostate cancer)
Question: What types of cancer can be treated with diethylstilbestrol?
Step 1: Diethylstilbestrol is a treatment for breast cancer. Diethylstilbestrol is a treatment for prostate cancer.
Output: Diethylstilbestrol can treat breast cancer and prostate cancer.

Example 3 - Drug inhibition:
Subgraph:
(TERT)-[:INHIBITION_TO]->(doxorubicin {fda_approved: YES})
Question: What drugs can treat cancers with TERT mutations?
Step 1: TERT is inhibited by doxorubicin (FDA approved).
Output: Cancers with TERT mutations can be inhibited by doxorubicin.

Example 4 - Gene activation:
Subgraph:
(codeine)-[:ACTIVATION_TO]->(OPRD1)
(codeine)-[:ACTIVATION_TO]->(OPRK1)
(codeine)-[:ACTIVATION_TO]->(OPRM1)
Question: Which genes can be activated by codeine?
Step 1: Codeine activates OPRD1. Codeine activates OPRK1. Codeine activates OPRM1.
Output: The following genes can be activated by codeine: OPRD1, OPRK1, and OPRM1.

Example 5 - NMPA-approved drugs (with filtering):
Subgraph:
(DDR2)-[:INHIBITION_TO]->(nilotinib {fda_approved: YES, nmpa_approved: YES})
(DDR2)-[:INHIBITION_TO]->(dasatinib {fda_approved: YES, nmpa_approved: YES})
(DDR2)-[:INHIBITION_TO]->(sitravatinib {fda_approved: NO, nmpa_approved: NO})
Question: What are the NMPA-approved drugs for cancers with DDR2 mutations?
Step 1: DDR2 is inhibited by nilotinib (NMPA approved). DDR2 is inhibited by dasatinib (NMPA approved). DDR2 is inhibited by sitravatinib (not NMPA approved).
Output: The NMPA-approved drugs for DDR2 are nilotinib and dasatinib.

Example 6 - No matching results:
Subgraph:
(TNKS)-[:INHIBITION_TO]->(drug1 {fda_approved: NO, nmpa_approved: NO})
Question: What are the NMPA-approved drugs for cancers with TNKS mutations?
Step 1: TNKS is inhibited by drug1 (not NMPA approved).
Output: There are no NMPA-approved drugs that can treat cancers with TNKS mutations.

Example 7 - Genetic mutations:
Subgraph:
(astrocytoma)-[:HAS_VAR]->(EGFR-p.L861R)
(astrocytoma)-[:HAS_VAR]->(EGFR-p.G719A)
Question: What genetic mutations are present in astrocytoma?
Step 1: Astrocytoma has the variant EGFR-p.L861R. Astrocytoma has the variant EGFR-p.G719A.
Output: Astrocytoma can be caused by the following genetic mutations: EGFR-p.L861R and EGFR-p.G719A.

=== YOUR TASK ===

Subgraph:
"""

    # フォールバック用プロンプト（サブグラフ取得失敗時）
    _NL_FALLBACK_PROMPT = """You are a reasoning robot, and you need to output natural language to answer my questions.
    For example:
    What drugs are ALK mutations in giant cell lung cancer resistant to?
    Output: ALK mutations in giant cell lung cancer are resistant to clotozantinib and luminaspib.
    """

    def _retrieve_subgraph_chains(
        self, analysis: QuestionAnalysis, path: SchemaPath
    ) -> List[str]:
        """サブグラフをリレーションチェーン文字列として取得"""
        graph = self.finder.graph
        head_name = analysis.head_entity_name

        def get_label(t: str) -> str:
            return f"`{t}`" if "/" in t else t

        def get_rel(r: str) -> str:
            if " " in r or "-" in r or "/" in r:
                return f"`{r}`"
            return r

        def where_head(var: str) -> str:
            if analysis.compound_search_term:
                return f'toLower({var}.name) CONTAINS toLower("{analysis.compound_search_term}")'
            if analysis.compound_names:
                names_str = ", ".join(f'"{n}"' for n in analysis.compound_names)
                return f'{var}.name IN [{names_str}]'
            return f'{var}.name = "{head_name}"'

        try:
            if len(path.types) == 2:
                # 1-hop
                d = path.directions[0] if path.directions else "->"
                rel = get_rel(path.relations[0])
                if d == "<-":
                    cypher = f"""
                    MATCH (t:{get_label(path.types[1])})-[r:{rel}]->(h:{get_label(path.types[0])})
                    WHERE {where_head("h")}
                    RETURN h.name AS head, type(r) AS rel, t.name AS tail
                    LIMIT 10
                    """
                else:
                    cypher = f"""
                    MATCH (h:{get_label(path.types[0])})-[r:{rel}]->(t:{get_label(path.types[1])})
                    WHERE {where_head("h")}
                    RETURN h.name AS head, type(r) AS rel, t.name AS tail
                    LIMIT 10
                    """
                results = graph.run(cypher).data()
                return [
                    f"({row['head']})-[:{row['rel']}]->({row['tail']}) {row['tail']}"
                    for row in results
                ]

            elif len(path.types) == 3:
                # 2-hop
                dir1 = path.directions[0] if path.directions else "->"
                dir2 = path.directions[1] if len(path.directions) > 1 else "->"
                rel1 = get_rel(path.relations[0])
                rel2 = get_rel(path.relations[1])

                if dir1 == "<-" and dir2 == "<-":
                    cypher = f"""
                    MATCH (t:{get_label(path.types[2])})-[r2:{rel2}]->(m:{get_label(path.types[1])})-[r1:{rel1}]->(h:{get_label(path.types[0])})
                    WHERE {where_head("h")}
                    RETURN h.name AS head, type(r1) AS rel1, m.name AS mid, type(r2) AS rel2, t.name AS tail
                    LIMIT 10
                    """
                elif dir1 == "<-" and dir2 == "->":
                    cypher = f"""
                    MATCH (m:{get_label(path.types[1])})-[r1:{rel1}]->(h:{get_label(path.types[0])})
                    MATCH (m)-[r2:{rel2}]->(t:{get_label(path.types[2])})
                    WHERE {where_head("h")}
                    RETURN h.name AS head, type(r1) AS rel1, m.name AS mid, type(r2) AS rel2, t.name AS tail
                    LIMIT 10
                    """
                elif dir1 == "->" and dir2 == "<-":
                    cypher = f"""
                    MATCH (h:{get_label(path.types[0])})-[r1:{rel1}]->(m:{get_label(path.types[1])})
                    MATCH (t:{get_label(path.types[2])})-[r2:{rel2}]->(m)
                    WHERE {where_head("h")}
                    RETURN h.name AS head, type(r1) AS rel1, m.name AS mid, type(r2) AS rel2, t.name AS tail
                    LIMIT 10
                    """
                else:
                    cypher = f"""
                    MATCH (h:{get_label(path.types[0])})-[r1:{rel1}]->(m:{get_label(path.types[1])})-[r2:{rel2}]->(t:{get_label(path.types[2])})
                    WHERE {where_head("h")}
                    RETURN h.name AS head, type(r1) AS rel1, m.name AS mid, type(r2) AS rel2, t.name AS tail
                    LIMIT 10
                    """
                results = graph.run(cypher).data()
                return [
                    f"({row['head']})-[:{row['rel1']}]->({row['mid']})-[:{row['rel2']}]->({row['tail']}) {row['tail']}"
                    for row in results
                ]

            else:
                # 3-hop+: 動的構築
                pattern_parts = [f"(n0:{get_label(path.types[0])})"]
                for i, rel in enumerate(path.relations):
                    pattern_parts.append(f"-[r{i}:{get_rel(rel)}]->(n{i+1}:{get_label(path.types[i+1])})")
                pattern = "".join(pattern_parts)

                return_parts = ["n0.name AS n0"]
                for i in range(len(path.relations)):
                    return_parts.append(f"type(r{i}) AS r{i}")
                    return_parts.append(f"n{i+1}.name AS n{i+1}")

                cypher = f"""
                MATCH {pattern}
                WHERE {where_head("n0")}
                RETURN {', '.join(return_parts)}
                LIMIT 10
                """
                results = graph.run(cypher).data()
                chains = []
                for row in results:
                    parts = [f"({row['n0']})"]
                    for i in range(len(path.relations)):
                        parts.append(f"-[:{row[f'r{i}']}]->({row[f'n{i+1}']})")
                    last = row[f"n{len(path.relations)}"]
                    chains.append("".join(parts) + f" {last}")
                return chains

        except Exception as e:
            print(f"Chain retrieval error: {e}")
            return []

    def _generate_natural_answer(self, question: str, chains: List[str]) -> Optional[str]:
        """サブグラフのチェーンからLLMで自然言語回答を生成（KGT論文準拠）"""
        if chains:
            chain_text = "\n".join(chains)
            prompt = self._NL_INFERENCE_PROMPT + chain_text + "\n\nQuestion: " + question + "\n\nStep 1:"
        else:
            # フォールバック: サブグラフなしでLLMに直接回答させる
            prompt = self._NL_FALLBACK_PROMPT + question

        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()

            # Extract "Output: ..." from CoT response (same logic as ETK)
            if "Output:" in answer:
                output_idx = answer.rfind("Output:")
                return answer[output_idx:]
            else:
                return f"Output: {answer}" if answer else None
        except Exception as e:
            print(f"NL generation error: {e}")
            return None
