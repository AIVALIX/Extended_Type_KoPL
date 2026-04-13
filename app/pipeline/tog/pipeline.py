"""
ToG (Think-on-Graph) Pipeline (Sun et al., ICLR 2024)

LLMとKGの対話型探索（原論文準拠）:
1. Entity Initialization - トピックエンティティを特定
2. Exploration (反復) - リレーション選択 → エンティティ選択 → 終了判定
3. Reasoning - 探索パスから最終回答を生成

プロンプトは https://github.com/GasolSun36/ToG に準拠。
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set, Tuple

from core.config import BASEMODEL, LLM_API_BASE, get_settings
from database.search import GraphPathFinder
from pipeline.tog.models import ExplorationStep, ToGResult


# --- 原論文準拠プロンプト ---

RELATION_PRUNE_PROMPT = """Please retrieve {width} relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of {width} relations is 1).

Q: {question}
Topic Entity: {entity}
Relations: {relations}
A: """

ENTITY_SCORE_PROMPT = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).

Q: {question}
Relation: {relation}
Entities: {entities}
Score: """

EVALUATE_PROMPT = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).

Q: {question}
Knowledge Triplets: {triplets}
A: """

ANSWER_PROMPT = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.

Q: {question}
Knowledge Triplets: {triplets}
A: """


def _parse_relation_scores(text: str, relations: List[str]) -> List[Tuple[str, float]]:
    """リレーションスコアをパース。例: '{relation (Score: 0.4)}' """
    scored = []
    rel_set = set(relations)

    # {relation (Score: 0.X)} 形式
    for m in re.finditer(r"\{([^(}]+?)\s*\(Score:\s*([\d.]+)\)\}", text):
        rel_name = m.group(1).strip()
        score = float(m.group(2))
        if rel_name in rel_set:
            scored.append((rel_name, score))

    if scored:
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # フォールバック: セミコロン区切り or 改行でリレーション名を探す
    for part in re.split(r"[;\n]", text):
        part = part.strip()
        for rel in relations:
            if rel in part and rel not in [s[0] for s in scored]:
                scored.append((rel, 0.0))
    return scored


def _parse_entity_scores(text: str, entities: List[str]) -> List[Tuple[str, float]]:
    """エンティティスコアをパース。例: '0.0, 1.0, 0.0' """
    # スコア行を検出
    scores_match = re.search(r"([\d.]+(?:\s*,\s*[\d.]+)+)", text)
    if scores_match and entities:
        scores = [float(s.strip()) for s in scores_match.group(1).split(",")]
        if len(scores) == len(entities):
            paired = list(zip(entities, scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            return paired

    # フォールバック: エンティティ名が出現する行を探す
    scored = []
    ent_set = set(entities)
    for line in text.split("\n"):
        for ent in entities:
            if ent in line and ent not in [s[0] for s in scored]:
                scored.append((ent, 0.5))
    return scored if scored else [(e, 0.0) for e in entities]


def _parse_answer_entities(
    text: str,
    known_entities: Optional[Set[str]] = None,
    anchor: Optional[str] = None,
) -> List[str]:
    """最終回答からエンティティ名を抽出。{answer} 形式を優先。

    anchor が指定された場合、answer set から anchor 自身を除外する。
    ToG の reasoning プロンプトは anchor を含む triplets を LLM に渡すため、
    LLM の応答文が anchor を言及すると known_entities 経由で answer に
    混入してしまう (self-hit)。
    """
    entities = []
    # {entity} 形式
    braced = re.findall(r"\{([^}]+)\}", text)
    if braced:
        for item in braced:
            for ent in item.split(","):
                ent = ent.strip().strip("'").strip('"')
                if ent and not ent.lower().startswith(("yes", "no")):
                    entities.append(ent)
        if anchor:
            entities = [e for e in entities if e != anchor]
        return entities

    # 探索チェーンに登場するエンティティ名とマッチ
    if known_entities:
        match_set = known_entities - {anchor} if anchor else known_entities
        for ent in match_set:
            if ent in text:
                entities.append(ent)
        if entities:
            return entities

    # カンマ区切り or 改行区切り
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if line.lower().startswith(("based on", "the answer", "therefore")):
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", line)
            entities.extend(quoted)
            continue
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        if ", " in line and len(line.split(", ")) > 2:
            for ent in line.split(", "):
                ent = ent.strip().strip("'").strip('"')
                if ent:
                    entities.append(ent)
        else:
            entities.append(line)
    return entities


class ToGPipeline:
    """ToG (Think-on-Graph) パイプライン"""

    def __init__(
        self,
        model: str = BASEMODEL,
        kg_type: str = "primekgqa",
        max_depth: int = 3,
        beam_width: int = 3,
        prune_top_k: int = 3,
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
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.prune_top_k = prune_top_k
        self.finder = GraphPathFinder(kg_type=kg_type)

    def run(
        self, question: str, entity_name: Optional[str] = None, **kwargs
    ) -> ToGResult:
        log = []
        log.append(f"Question: {question}")

        if not entity_name:
            log.append("No entity_name provided")
            return ToGResult(
                question=question, entity_name=entity_name,
                answer_entities=[], processing_log=log,
            )

        # Phase 1: Entity Initialization
        log.append(f"Phase 1: Entity Initialization → {entity_name}")
        current_entities = [entity_name]
        exploration_steps: List[ExplorationStep] = []
        all_explored_relations: List[str] = []
        # 探索チェーン: (entity, relation, entity) のリスト
        chain_of_entities: List[List[str]] = []

        # Phase 2: Iterative Exploration
        for depth in range(self.max_depth):
            log.append(f"Phase 2: Exploration depth {depth + 1}/{self.max_depth}")
            log.append(f"  Current entities: {current_entities[:5]}{'...' if len(current_entities) > 5 else ''}")

            # a) Relation Exploration
            entity_relations: Dict[str, List[str]] = {}
            for ent in current_entities[: self.beam_width]:
                rels = self.finder.get_adjacent_relations(ent)
                entity_relations[ent] = rels

            # 各エンティティごとにリレーションをスコアリング
            all_candidates: List[Tuple[str, str, str, float]] = []  # (src, rel, tgt, score)

            for ent in current_entities[: self.beam_width]:
                rels = entity_relations.get(ent, [])
                if not rels:
                    continue

                selected_rels = self._score_relations(question, ent, rels)
                log.append(f"  [{ent}] Selected relations: {[r for r, _ in selected_rels[:3]]}")
                all_explored_relations.extend([r for r, _ in selected_rels])

                # b) Entity Exploration per selected relation
                for rel, rel_score in selected_rels[: self.prune_top_k]:
                    neighbors = self.finder.get_candidate_nodes(ent, rel)
                    if not neighbors:
                        continue
                    neighbor_names = [n for n, _ in neighbors]

                    entity_scores = self._score_entities(
                        question, rel, neighbor_names
                    )
                    for tgt, ent_score in entity_scores[: self.beam_width]:
                        all_candidates.append((ent, rel, tgt, rel_score * 0.5 + ent_score * 0.5))

            if not all_candidates:
                log.append("  No candidates found, stopping exploration")
                break

            # スコアでソートし top beam_width を選択
            all_candidates.sort(key=lambda x: x[3], reverse=True)
            selected = all_candidates[: self.beam_width]

            selected_entities = []
            for src, rel, tgt, score in selected:
                chain_of_entities.append([src, rel, tgt])
                if tgt not in selected_entities:
                    selected_entities.append(tgt)

            log.append(f"  Selected next entities: {selected_entities}")

            step = ExplorationStep(
                depth=depth + 1,
                entities=current_entities[: self.beam_width],
                selected_relations=list(set(r for _, r, _, _ in selected)),
                next_entities=selected_entities,
            )
            exploration_steps.append(step)

            # c) Termination Check
            sufficient = self._check_termination(question, chain_of_entities)
            if sufficient:
                log.append(f"  Termination: sufficient at depth {depth + 1}")
                break

            current_entities = selected_entities

        # Phase 3: Reasoning
        log.append("Phase 3: Reasoning")
        # 探索チェーンに登場する全エンティティを収集 (パースフォールバック用)
        known_ents: Set[str] = set()
        for chain in chain_of_entities:
            for item in chain:
                if isinstance(item, str):
                    known_ents.add(item)
        answer_entities = self._reason(
            question, chain_of_entities, known_ents, anchor=entity_name
        )
        log.append(f"  Answer entities: {answer_entities}")

        return ToGResult(
            question=question,
            entity_name=entity_name,
            exploration_steps=exploration_steps,
            answer_entities=answer_entities,
            explored_relations=all_explored_relations,
            processing_log=log,
        )

    def _score_relations(
        self, question: str, entity: str, relations: List[str]
    ) -> List[Tuple[str, float]]:
        """原論文準拠: リレーションスコアリング"""
        relations_str = "; ".join(relations)
        prompt = RELATION_PRUNE_PROMPT.format(
            width=self.prune_top_k, question=question,
            entity=entity, relations=relations_str,
        )
        try:
            response = self.llm.invoke(prompt)
            scored = _parse_relation_scores(response.content, relations)
            if scored:
                return scored[: self.prune_top_k]
        except Exception:
            pass
        return [(r, 1.0 / len(relations)) for r in relations[: self.prune_top_k]]

    def _score_entities(
        self, question: str, relation: str, entities: List[str]
    ) -> List[Tuple[str, float]]:
        """原論文準拠: エンティティスコアリング"""
        if len(entities) <= self.beam_width:
            return [(e, 1.0 / max(len(entities), 1)) for e in entities]

        entities_str = "; ".join(entities[:30])
        prompt = ENTITY_SCORE_PROMPT.format(
            question=question, relation=relation, entities=entities_str,
        )
        try:
            response = self.llm.invoke(prompt)
            scored = _parse_entity_scores(response.content, entities[:30])
            if scored:
                return scored[: self.beam_width]
        except Exception:
            pass
        return [(e, 0.0) for e in entities[: self.beam_width]]

    def _check_termination(
        self, question: str, chains: List[List[str]]
    ) -> bool:
        """原論文準拠: {Yes}/{No} で判定"""
        if not chains:
            return False

        triplets_str = "\n".join(
            ", ".join(str(x) for x in chain) for chain in chains
        )
        prompt = EVALUATE_PROMPT.format(question=question, triplets=triplets_str)
        try:
            response = self.llm.invoke(prompt)
            return "{yes}" in response.content.lower() or response.content.strip().lower().startswith("yes")
        except Exception:
            return False

    def _reason(
        self, question: str, chains: List[List[str]],
        known_entities: Optional[Set[str]] = None,
        anchor: Optional[str] = None,
    ) -> List[str]:
        """原論文準拠: 探索チェーンから最終回答を生成"""
        if not chains:
            return []

        triplets_str = "\n".join(
            ", ".join(str(x) for x in chain) for chain in chains
        )
        prompt = ANSWER_PROMPT.format(question=question, triplets=triplets_str)
        try:
            response = self.llm.invoke(prompt)
            return _parse_answer_entities(response.content, known_entities, anchor=anchor)
        except Exception:
            return []
