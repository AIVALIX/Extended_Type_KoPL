"""KG-GPT パイプライン用データモデル"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field


@dataclass
class KGGPTResult:
    """KG-GPT パイプライン結果"""
    question: str
    entity_name: Optional[str]
    retrieved_triples: List[str] = field(default_factory=list)
    answer_entities: List[str] = field(default_factory=list)
    explored_relations: Optional[List[str]] = None
    processing_log: List[str] = field(default_factory=list)


class SentenceSegmentationResponse(BaseModel):
    """質問の部分クエリ分割"""
    sub_queries: List[str] = Field(
        ...,
        description="The question broken into atomic sub-queries, each asking about one relationship.",
    )


class InferenceResponse(BaseModel):
    """トリプルベースの推論結果"""
    entities: List[str] = Field(
        ...,
        description="List of entity names that answer the question based on the provided triples.",
    )
