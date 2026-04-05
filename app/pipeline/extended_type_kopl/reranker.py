"""
LLM Reranker Module for Extended Type-KoPL Pipeline

ベクトル剪定後の候補パスからLLMで最適なパスを選択する
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field

from core.config import BASEMODEL, get_settings


class PathSelectionResponse(BaseModel):
    """LLM出力用スキーマ"""
    selected_index: int = Field(description="The index of the best matching path (0-indexed)")
    reasoning: Optional[str] = Field(default=None, description="Brief reasoning for the selection")


class BaseReranker(ABC):
    """Rerankerの基底クラス"""

    @abstractmethod
    def rerank(
        self,
        question: str,
        candidate_paths: List,
        top_k: int = 1,
        relation_hints: Optional[List[str]] = None,
        kopl_program=None,
        trial_samples=None,
    ) -> List:
        """候補パスを再ランキングしてtop_k個を返す"""
        pass


class NoOpReranker(BaseReranker):
    """何もしないReranker（パススルー）"""

    def rerank(
        self,
        question: str,
        candidate_paths: List,
        top_k: int = 1,
        relation_hints: Optional[List[str]] = None,
        kopl_program=None,
        trial_samples=None,
    ) -> List:
        return candidate_paths[:top_k]


class LLMReranker(BaseReranker):
    """LLMベースのReranker"""

    def __init__(
        self,
        model: str = BASEMODEL,
        kg_type: str = "primekgqa",
    ):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model

        self.llm = init_chat_model(model, model_provider="openai", temperature=0)
        self.kg_type = kg_type

    def _get_path_description(self, path) -> str:
        """パスを人間が読みやすい形式に変換"""
        parts = []
        for i, t in enumerate(path.types):
            parts.append(t)
            if i < len(path.relations):
                rel = path.relations[i]
                direction = path.directions[i] if i < len(path.directions) else "->"
                if direction == "<-":
                    parts.append(f" <-[{rel}]- ")
                else:
                    parts.append(f" -[{rel}]-> ")
        return "".join(parts)

    def _build_prompt(self, question: str, candidate_paths: List, relation_hints: Optional[List[str]] = None, kopl_program=None, trial_samples=None) -> str:
        """プロンプトを構築"""
        paths_text = ""
        for i, path in enumerate(candidate_paths):
            desc = self._get_path_description(path)
            paths_text += f"{i}. {desc}"
            # Cypher trial結果を追加
            if trial_samples and id(path) in trial_samples:
                samples = trial_samples[id(path)]
                paths_text += f"  → Example results: {', '.join(samples[:5])}"
            paths_text += "\n"

        # KoPLプログラムの情報をプロンプトに追加
        kopl_text = ""
        if kopl_program:
            kopl_lines = []
            kopl_lines.append(f"Operation: {kopl_program.op_type.value if hasattr(kopl_program.op_type, 'value') else kopl_program.op_type}")
            if kopl_program.anchor_name:
                kopl_lines.append(f"Anchor entity: {kopl_program.anchor_name}")
            for i, rel in enumerate(kopl_program.relations):
                hint = f", hint: {rel.relation_hint}" if rel.relation_hint else ""
                kopl_lines.append(f"  Hop {i+1}: {rel.src_type} -> {rel.tgt_type}{hint}")
            kopl_text = "LLM-generated query plan:\n" + "\n".join(kopl_lines) + "\n"

        # trial結果がある場合は追加指示
        trial_instruction = ""
        if trial_samples:
            trial_instruction = """
- IMPORTANT: Each candidate path shows "Example results" from the actual knowledge graph.
- Compare the example results against what the question is asking for.
- Choose the path whose example results best match the expected answer type."""

        prompt = f"""You are a knowledge graph expert. Select the single best path that answers the question.

Question: {question}

{kopl_text}
Candidate paths:
{paths_text}
IMPORTANT:
- Do NOT simply match relation names to words in the question. Relation names in the KG may use abbreviations or different terminology than the question.
- Consider what each relation semantically means in the context of the knowledge graph.
- The query plan hints are approximate and may not exactly match any relation name in the KG.
- Think step-by-step about which path correctly captures the reasoning chain needed to answer the question.{trial_instruction}

Return the index (0-indexed) of the best path."""

        return prompt

    def rerank(
        self,
        question: str,
        candidate_paths: List,
        top_k: int = 1,
        relation_hints: Optional[List[str]] = None,
        kopl_program=None,
        trial_samples=None,
    ) -> List:
        """LLMで最適なパスを選択"""

        if not candidate_paths:
            return []

        if len(candidate_paths) == 1:
            return candidate_paths

        prompt = self._build_prompt(question, candidate_paths, relation_hints, kopl_program, trial_samples)

        try:
            llm_with_output = self.llm.with_structured_output(PathSelectionResponse)
            result = llm_with_output.invoke(prompt)

            selected_idx = result.selected_index

            # インデックスの範囲チェック
            if 0 <= selected_idx < len(candidate_paths):
                # 選択されたパスを先頭に、残りを後ろに
                selected = [candidate_paths[selected_idx]]
                others = [p for i, p in enumerate(candidate_paths) if i != selected_idx]
                reranked = selected + others
                return reranked[:top_k]
            else:
                # 無効なインデックスの場合は元の順序を維持
                return candidate_paths[:top_k]

        except Exception as e:
            print(f"LLM Reranker error: {e}")
            # エラー時は元の順序を維持
            return candidate_paths[:top_k]


class HybridReranker(BaseReranker):
    """ベクトルスコアとLLMを組み合わせたReranker"""

    def __init__(
        self,
        model: str = BASEMODEL,
        kg_type: str = "primekgqa",
        vector_weight: float = 0.3,
    ):
        self.llm_reranker = LLMReranker(model=model, kg_type=kg_type)
        self.vector_weight = vector_weight

    def rerank(
        self,
        question: str,
        candidate_paths: List,
        top_k: int = 1,
        relation_hints: Optional[List[str]] = None,
        kopl_program=None,
        trial_samples=None,
    ) -> List:
        """ベクトルスコアを考慮しつつLLMで再ランキング"""
        # まずLLMで再ランキング
        return self.llm_reranker.rerank(question, candidate_paths, top_k, relation_hints, kopl_program, trial_samples)


def create_reranker(
    reranker_type: str = "none",
    model: str = BASEMODEL,
    kg_type: str = "primekgqa",
) -> BaseReranker:
    """Rerankerファクトリ関数

    Args:
        reranker_type: "none", "llm", or "hybrid"
        model: LLMモデル名
        kg_type: KGタイプ

    Returns:
        BaseReranker instance
    """
    if reranker_type == "llm":
        return LLMReranker(model=model, kg_type=kg_type)
    elif reranker_type == "hybrid":
        return HybridReranker(model=model, kg_type=kg_type)
    else:
        return NoOpReranker()
