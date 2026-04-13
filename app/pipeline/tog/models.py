"""ToG (Think-on-Graph) パイプライン用データモデル"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field


@dataclass
class ExplorationStep:
    """探索の1ステップ"""
    depth: int
    entities: List[str]
    selected_relations: List[str]
    next_entities: List[str]


@dataclass
class ToGResult:
    """ToG パイプライン結果"""
    question: str
    entity_name: Optional[str]
    exploration_steps: List[ExplorationStep] = field(default_factory=list)
    answer_entities: List[str] = field(default_factory=list)
    explored_relations: Optional[List[str]] = None
    processing_log: List[str] = field(default_factory=list)


class RelationScoreResponse(BaseModel):
    """リレーションスコアリング結果"""
    selected_relations: List[str] = Field(
        ...,
        description="Relations most relevant to answering the question, ranked by relevance.",
    )


class EntityScoreResponse(BaseModel):
    """エンティティスコアリング結果"""
    selected_entities: List[str] = Field(
        ...,
        description="Entities most relevant to answering the question, ranked by relevance.",
    )


class TerminationResponse(BaseModel):
    """探索終了判定"""
    sufficient: bool = Field(
        ...,
        description="True if the explored information is sufficient to answer the question.",
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the decision.",
    )


class ReasoningResponse(BaseModel):
    """最終推論結果"""
    entities: List[str] = Field(
        ...,
        description="List of entity names that answer the question.",
    )
