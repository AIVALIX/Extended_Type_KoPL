"""
KGT パイプライン用データモデル
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    compound_names: List[str] = field(default_factory=list)  # PcQA: CancerCell複合名リスト
    compound_search_term: Optional[str] = None  # PcQA: CONTAINS検索用の元エンティティ名（遺伝子名等）


@dataclass
class SchemaPath:
    """スキーマパス"""
    path: List[str]  # [type1, rel1, type2, rel2, type3, ...]
    types: List[str]  # [type1, type2, type3, ...]
    relations: List[str]  # [rel1, rel2, ...]
    score: float = 0.0  # ベクトル類似度スコア
    directions: List[str] = field(default_factory=list)  # ["->", "<-", ...] for each relation


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
