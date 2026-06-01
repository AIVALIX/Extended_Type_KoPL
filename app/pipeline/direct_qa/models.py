"""Direct QA パイプライン用データモデル"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field


@dataclass
class DirectQAResult:
    """Direct QA パイプライン結果"""
    question: str
    entity_name: Optional[str]
    answer_entities: List[str] = field(default_factory=list)
    processing_log: List[str] = field(default_factory=list)


class AnswerResponse(BaseModel):
    """LLMによる回答の出力"""
    entities: List[str] = Field(
        ...,
        description="List of entity names that answer the question. "
        "Return exact entity names, not descriptions.",
    )
