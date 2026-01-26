"""
SAFE (Semantic-Aware Subgraph Retrieval Framework) パイプライン

論文に基づく実装:
1. ADJ (Approximate Distance Join) - スキーマレベル探索
2. Ranked Semantic Subgraph Matching - インスタンスレベル探索
"""

from pipeline.safe.pipeline import SAFEPipeline
from pipeline.safe.models import (
    PseudoEdge,
    SchemaEdge,
    CandidateSchemaEdge,
    QueryGraph,
    MatchedSubgraph,
    SAFEResult,
)
from pipeline.safe.schema import SchemaGraphWithAPSP, build_schema_with_apsp

__all__ = [
    "SAFEPipeline",
    "PseudoEdge",
    "SchemaEdge",
    "CandidateSchemaEdge",
    "QueryGraph",
    "MatchedSubgraph",
    "SAFEResult",
    "SchemaGraphWithAPSP",
    "build_schema_with_apsp",
]
