"""
SAFE スキーマグラフ（APSP付き）
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from pipeline.safe.models import SchemaEdge


class SchemaGraphWithAPSP:
    """全点対最短経路(APSP)を持つスキーマグラフ"""

    def __init__(self):
        self.edges: List[SchemaEdge] = []
        self.types: Set[str] = set()
        self.type_to_idx: Dict[str, int] = {}
        self.idx_to_type: Dict[int, str] = {}
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.apsp: Optional[np.ndarray] = None  # All-Pairs Shortest Path距離行列
        self.embeddings: Optional[Any] = None  # OpenAI Embeddings

    def add_edge(self, src_type: str, relation: str, tgt_type: str):
        """エッジ追加"""
        edge = SchemaEdge(src_type, relation, tgt_type)
        self.edges.append(edge)
        self.types.add(src_type)
        self.types.add(tgt_type)
        # 無向グラフとして追加
        self.adjacency[src_type].append((tgt_type, relation))
        if src_type != tgt_type:
            self.adjacency[tgt_type].append((src_type, relation))

    def build_apsp(self):
        """Floyd-Warshallで全点対最短経路を計算"""
        n = len(self.types)
        type_list = sorted(self.types)
        self.type_to_idx = {t: i for i, t in enumerate(type_list)}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}

        # 距離行列を初期化（無限大）
        INF = float('inf')
        dist = np.full((n, n), INF)

        # 対角成分は0
        for i in range(n):
            dist[i, i] = 0

        # 隣接エッジの距離は1（自己参照は除く、対角は0を維持）
        for src_type, neighbors in self.adjacency.items():
            src_idx = self.type_to_idx[src_type]
            for tgt_type, _ in neighbors:
                tgt_idx = self.type_to_idx[tgt_type]
                if src_idx != tgt_idx:  # 自己参照エッジは距離0を維持
                    dist[src_idx, tgt_idx] = 1

        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        self.apsp = dist

    def get_type_distance(self, type1: str, type2: str) -> float:
        """2つのタイプ間の最短距離を取得（O(1)）"""
        if type1 not in self.type_to_idx or type2 not in self.type_to_idx:
            return float('inf')
        i = self.type_to_idx[type1]
        j = self.type_to_idx[type2]
        return self.apsp[i, j]

    def get_edge_distance(self, edge1: SchemaEdge, edge2: SchemaEdge) -> float:
        """2つのエッジ間のチェーン接続距離（edge1.tgt → edge2.src）

        チェーン接続のみをチェック:
        - d1 = 0: 直接接続 (edge1.tgt_type == edge2.src_type)
        - d1 = 1: 1ノードを介した接続
        """
        return self.get_type_distance(edge1.tgt_type, edge2.src_type)

    def compute_edge_embeddings(self, embeddings):
        """全エッジの埋め込みを計算"""
        self.embeddings = embeddings
        texts = [e.to_text() for e in self.edges]
        if texts:
            vectors = embeddings.embed_documents(texts)
            for i, edge in enumerate(self.edges):
                edge.embedding = np.array(vectors[i])


def build_schema_with_apsp(kg_type: str = "primekgqa") -> SchemaGraphWithAPSP:
    """スキーマグラフをAPSP付きで構築

    Args:
        kg_type: "primekgqa" or "metaqa"
    """
    if kg_type == "metaqa":
        from dataset_construction.schema_metaqa import SCHEMA_GRAPH
    else:
        from dataset_construction.schema_v2 import SCHEMA_GRAPH

    schema = SchemaGraphWithAPSP()
    for src, rel, tgt, _ in SCHEMA_GRAPH:
        schema.add_edge(src, rel, tgt)

    schema.build_apsp()
    return schema
