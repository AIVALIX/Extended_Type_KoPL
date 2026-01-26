"""
Pipeline モジュール

各パイプラインへのアクセスを提供:
- SAFE: Semantic-Aware Subgraph Retrieval Framework
- KGT: Knowledge Graph Transformer
- Extended Type-KoPL: Extended Type-aware KoPL

共通モジュール:
- common: 評価メトリクス、基底クラス等
"""

# 後方互換性のためのエイリアス
from pipeline.safe import SAFEPipeline
from pipeline.kgt import KGTPipeline
from pipeline.extended_type_kopl import ExtendedTypeKoPLPipeline

__all__ = [
    "SAFEPipeline",
    "KGTPipeline",
    "ExtendedTypeKoPLPipeline",
]
