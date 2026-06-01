"""
KG-GPT Pipeline (Kim et al., 2023)

3段階パイプライン（原論文準拠）:
1. Sentence Segmentation - 質問を部分クエリに分割
2. Graph Retrieval - 関連リレーションをLLMで選択しトリプルを取得
3. Inference - 取得トリプルをエビデンスとしてLLMで回答

プロンプトは https://github.com/jiho283/KG-GPT に準拠。
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set, Tuple

from core.config import BASEMODEL, LLM_API_BASE, get_settings
from database.search import GraphPathFinder
from pipeline.kg_gpt.models import KGGPTResult


# --- 原論文準拠プロンプト ---

SENTENCE_SEGMENTATION_PROMPT = """Please divide the given question into sub-questions, each of which can be represented by one triplet. Each sub-question should ask about one relationship. If the question is already atomic (single-hop), return it as a single sub-question.

Examples)
Question A: What diseases are associated with genes targeted by Aspirin?
--> Divided:
1. genes targeted by Aspirin, Entity set: ['Aspirin']
2. What diseases are associated with these genes, Entity set: []

Question B: Which proteins does Acetylcysteine target?
--> Divided:
1. Which proteins does Acetylcysteine target?, Entity set: ['Acetylcysteine']

Question C: Who directed the movies that Tom Hanks starred in?
--> Divided:
1. the movies that Tom Hanks starred in, Entity set: ['Tom Hanks']
2. Who directed the movies, Entity set: []

Your Task)
Question: {question}
--> Divided:
"""

RELATION_RETRIEVAL_PROMPT = """I will give you a set of relations.
Find the top {top_k} elements from Relations set which are most semantically related to the given sentence. You may select up to {top_k} relations. If there is nothing that looks semantically related, pick out any {top_k} elements and give them to me.

Sentence: {sentence}
Relations set: {relations}
Top {top_k} Answer: """

INFERENCE_PROMPT = """Answer the question based on evidence.
Each evidence is in the form of [head, relation, tail] and it means "head's relation is tail.".

Please answer with entity names from the evidence. If you think a question can have multiple answers, list all of them.

Now let's answer the question based on the Evidence set.
Question: {question}
Evidence set: {evidence}
Answer: """


def _parse_entity_list(text: str) -> List[str]:
    """LLM出力からエンティティ名リストをパース"""
    entities = []
    # {entity} 形式を検出
    braced = re.findall(r"\{([^}]+)\}", text)
    if braced:
        for item in braced:
            for ent in item.split(","):
                ent = ent.strip().strip("'").strip('"')
                if ent:
                    entities.append(ent)
        return entities

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # 説明文をスキップ
        if line.lower().startswith(("based on", "the answer", "therefore", "from the",
                                     "the evidence", "there is no", "note:")):
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", line)
            entities.extend(quoted)
            continue
        if "entities mentioned" in line.lower():
            continue
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        # カンマ区切りの場合（短いトークンが多い → エンティティリスト）
        if ", " in line and len(line.split(", ")) > 2:
            for ent in line.split(", "):
                ent = ent.strip().strip("'").strip('"')
                if ent:
                    entities.append(ent)
        else:
            entities.append(line)
    return entities


def _parse_relations(text: str, valid_relations: Set[str]) -> List[str]:
    """LLM出力からリレーション名をパース"""
    # ["rel1", "rel2"] 形式
    bracket_match = re.search(r"\[([^\]]+)\]", text)
    if bracket_match:
        items = [r.strip().strip("'").strip('"') for r in bracket_match.group(1).split(",")]
        return [r for r in items if r in valid_relations]

    # カンマ区切り or 改行
    results = []
    for part in re.split(r"[,\n]", text):
        part = part.strip().strip("'").strip('"')
        part = re.sub(r"^\d+[\.\)]\s*", "", part).strip()
        if part in valid_relations:
            results.append(part)
    return results


class KGGPTPipeline:
    """KG-GPT パイプライン"""

    def __init__(
        self,
        model: str = BASEMODEL,
        kg_type: str = "primekgqa",
        max_triples: int = 50,
        max_hops: int = 2,
        top_k_relations: int = 3,
        **kwargs,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model

        llm_kwargs = {"model_provider": "openai", "temperature": 0, "max_tokens": 4096}
        api_base = os.getenv("LLM_API_BASE", "") or LLM_API_BASE or None
        if api_base:
            llm_kwargs["base_url"] = api_base
            llm_kwargs["api_key"] = "sk-local"
        self.llm = init_chat_model(model, **llm_kwargs)
        self.kg_type = kg_type
        self.max_triples = max_triples
        self.max_hops = max_hops
        self.top_k_relations = top_k_relations
        self.finder = GraphPathFinder(kg_type=kg_type)

    def run(
        self, question: str, entity_name: Optional[str] = None, **kwargs
    ) -> KGGPTResult:
        log = []
        log.append(f"Question: {question}")

        # Phase 1: Sentence Segmentation
        log.append("Phase 1: Sentence Segmentation")
        sub_queries = self._segment_question(question)
        log.append(f"  Sub-queries: {sub_queries}")

        # Phase 2: Graph Retrieval
        log.append("Phase 2: Graph Retrieval")
        if not entity_name:
            log.append("  No entity_name provided")
            return KGGPTResult(
                question=question, entity_name=entity_name,
                answer_entities=[], processing_log=log,
            )

        triples, explored_rels = self._retrieve_triples(entity_name, sub_queries)
        log.append(f"  Retrieved {len(triples)} triples from entity '{entity_name}'")

        if not triples:
            log.append("  No triples found")
            return KGGPTResult(
                question=question, entity_name=entity_name,
                answer_entities=[], explored_relations=explored_rels,
                processing_log=log,
            )

        # Phase 3: Inference
        log.append("Phase 3: LLM Inference")
        answer_entities = self._infer_answer(question, triples)
        log.append(f"  Inferred {len(answer_entities)} answer entities")

        return KGGPTResult(
            question=question, entity_name=entity_name,
            retrieved_triples=[str(t) for t in triples],
            answer_entities=answer_entities, explored_relations=explored_rels,
            processing_log=log,
        )

    def _segment_question(self, question: str) -> List[str]:
        """Phase 1: 質問を部分クエリに分割（原論文準拠）"""
        prompt = SENTENCE_SEGMENTATION_PROMPT.format(question=question)
        try:
            response = self.llm.invoke(prompt)
            lines = []
            for line in response.content.strip().split("\n"):
                line = line.strip()
                # "1. ..." 形式の行を抽出
                m = re.match(r"^\d+\.\s*(.+?)(?:,\s*Entity set:.*)?$", line)
                if m:
                    lines.append(m.group(1).strip())
            return lines if lines else [question]
        except Exception:
            return [question]

    def _retrieve_triples(
        self, entity_name: str, sub_queries: List[str]
    ) -> Tuple[List[List[str]], List[str]]:
        """Phase 2: LLMでリレーション選択 → トリプル取得（原論文準拠）"""
        all_triples: List[List[str]] = []
        explored_rels: List[str] = []
        seen: Set[str] = set()

        # 1-hop: エンティティの隣接リレーションを取得
        relations = self.finder.get_adjacent_relations(entity_name)
        if not relations:
            return [], []

        # LLMでリレーション選択
        selected_rels = self._select_relations(
            sub_queries[0] if sub_queries else entity_name, relations
        )
        explored_rels.extend(selected_rels)

        # 選択リレーションのトリプルを取得
        hop1_entities: List[str] = []
        for rel in selected_rels:
            neighbors = self.finder.get_candidate_nodes(entity_name, rel)
            for node_name, _labels in neighbors:
                triple = [entity_name, rel, node_name]
                key = str(triple)
                if key not in seen:
                    seen.add(key)
                    all_triples.append(triple)
                    hop1_entities.append(node_name)
                if len(all_triples) >= self.max_triples:
                    break
            if len(all_triples) >= self.max_triples:
                break

        # 2-hop: sub_queryが2つ以上ある場合
        if len(sub_queries) > 1 and self.max_hops >= 2:
            # 上位10エンティティの全リレーションを統合して1回でLLM選択
            hop2_targets = hop1_entities[:10]
            entity_to_rels: Dict[str, List[str]] = {}
            all_rels2: Set[str] = set()
            for ent in hop2_targets:
                rels2 = self.finder.get_adjacent_relations(ent)
                entity_to_rels[ent] = rels2
                all_rels2.update(rels2)

            if all_rels2:
                # 1回のLLM呼び出しでhop2のリレーションを選択
                selected_rels2 = self._select_relations(
                    sub_queries[1] if len(sub_queries) > 1 else sub_queries[0],
                    list(all_rels2),
                )
                explored_rels.extend(selected_rels2)

                for ent in hop2_targets:
                    if len(all_triples) >= self.max_triples:
                        break
                    ent_rels = entity_to_rels.get(ent, [])
                    for rel2 in selected_rels2:
                        if rel2 not in ent_rels:
                            continue
                        neighbors2 = self.finder.get_candidate_nodes(ent, rel2)
                        for node_name2, _ in neighbors2:
                            triple = [ent, rel2, node_name2]
                            key = str(triple)
                            if key not in seen:
                                seen.add(key)
                                all_triples.append(triple)
                            if len(all_triples) >= self.max_triples:
                                break
                        if len(all_triples) >= self.max_triples:
                            break

        return all_triples, explored_rels

    def _select_relations(self, sentence: str, relations: List[str]) -> List[str]:
        """LLMで関連リレーションを選択（原論文準拠）"""
        relations_str = str(relations)
        prompt = RELATION_RETRIEVAL_PROMPT.format(
            top_k=self.top_k_relations, sentence=sentence, relations=relations_str,
        )
        try:
            response = self.llm.invoke(prompt)
            selected = _parse_relations(response.content, set(relations))
            return selected[:self.top_k_relations] if selected else relations[:self.top_k_relations]
        except Exception:
            return relations[:self.top_k_relations]

    def _infer_answer(self, question: str, triples: List[List[str]]) -> List[str]:
        """Phase 3: エビデンスベースの推論（原論文準拠）"""
        evidence_str = str(triples[:self.max_triples])
        prompt = INFERENCE_PROMPT.format(question=question, evidence=evidence_str)
        try:
            response = self.llm.invoke(prompt)
            return _parse_entity_list(response.content)
        except Exception:
            return []
