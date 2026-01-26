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

    def _build_prompt(self, question: str, candidate_paths: List) -> str:
        """プロンプトを構築"""
        paths_text = ""
        for i, path in enumerate(candidate_paths):
            desc = self._get_path_description(path)
            paths_text += f"{i}. {desc}\n"

        if self.kg_type == "metaqa":
            domain_hint = """
Domain: Movie database
- DIRECTED_BY: who directed the movie
- STARRED_ACTORS: who acted/starred in the movie
- WRITTEN_BY: who wrote the screenplay
- IN_LANGUAGE: what language the movie is in
- HAS_GENRE: what genre the movie belongs to
- RELEASE_YEAR: when the movie was released
"""
        else:
            domain_hint = """
Domain: Biomedical knowledge graph
- target: drug targets a gene/protein
- indication: drug is used to treat a disease
- associated_with/associated_disease: gene is associated with a disease
- ppi: protein-protein interaction
"""

        prompt = f"""Select the path that best matches the question's intent.

Question: {question}
{domain_hint}
Candidate paths:
{paths_text}
IMPORTANT: Pay attention to the relation names. Match them to the keywords in the question:
- "director/directed" → DIRECTED_BY
- "star/starred/actor/acted" → STARRED_ACTORS
- "writer/wrote/written" → WRITTEN_BY
- "language" → IN_LANGUAGE
- "genre/type" → HAS_GENRE
- "release/year" → RELEASE_YEAR

Return the index (0-indexed) of the path that best answers the question."""

        return prompt

    def rerank(
        self,
        question: str,
        candidate_paths: List,
        top_k: int = 1,
    ) -> List:
        """LLMで最適なパスを選択"""

        if not candidate_paths:
            return []

        if len(candidate_paths) == 1:
            return candidate_paths

        prompt = self._build_prompt(question, candidate_paths)

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
    ) -> List:
        """ベクトルスコアを考慮しつつLLMで再ランキング"""
        # まずLLMで再ランキング
        return self.llm_reranker.rerank(question, candidate_paths, top_k)


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
