"""
SAFE パイプライン用データモデル
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PseudoEdge:
    """擬似クエリグラフのエッジ"""
    src_node: str  # ソースノード名
    src_type: str  # ソースノードタイプ
    relation: str  # リレーション
    tgt_node: str  # ターゲットノード名
    tgt_type: str  # ターゲットノードタイプ
    is_anchor: bool = False  # アンカー（既知エンティティ）かどうか


@dataclass
class SchemaEdge:
    """スキーマエッジ"""
    src_type: str
    relation: str
    tgt_type: str
    embedding: Optional[np.ndarray] = None

    def to_text(self) -> str:
        return f"{self.src_type} {self.relation} {self.tgt_type}"


@dataclass
class CandidateSchemaEdge:
    """候補スキーマエッジ（距離付き）"""
    schema_edge: SchemaEdge
    distance: float  # L2距離


@dataclass
class QueryGraph:
    """補正されたクエリグラフ"""
    edges: List[Tuple[SchemaEdge, PseudoEdge]]  # (スキーマエッジ, 元の擬似エッジ)
    total_distance: float  # 合計意味的距離


@dataclass
class MatchedSubgraph:
    """マッチしたサブグラフ"""
    nodes: Dict[str, str]  # query_node -> kg_entity
    edges: List[Tuple[str, str, str]]  # (src, rel, tgt)
    score: float  # 累積スコア（低いほど良い）


@dataclass
class SAFEResult:
    """SAFE結果"""
    question: str
    pseudo_edges: List[PseudoEdge]
    candidate_query_graphs: List[QueryGraph]
    best_query_graph: Optional[QueryGraph]
    matched_subgraphs: List[MatchedSubgraph]
    answer_entities: List[str]
    processing_log: List[str] = field(default_factory=list)


# =============================================================================
# Pydantic Models for LLM
# =============================================================================

class PseudoEdgeSchema(BaseModel):
    """擬似エッジのスキーマ"""
    src_node: str = Field(description="Source node name (use '?' for unknown)")
    src_type: str = Field(description="Source node type")
    relation: str = Field(description="Relationship name in natural language")
    tgt_node: str = Field(description="Target node name (use '?' for unknown/answer)")
    tgt_type: str = Field(description="Target node type")
    is_anchor: bool = Field(default=False, description="True if node is a known entity")


class PseudoQueryGraphResponse(BaseModel):
    """LLMによる擬似クエリグラフ生成"""
    edges: List[PseudoEdgeSchema] = Field(
        default=[],
        description="List of edges representing the query structure"
    )
