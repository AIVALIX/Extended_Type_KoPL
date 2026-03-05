"""
KGT スキーマグラフ
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from pipeline.kgt.models import SchemaPath


class KGSchema:
    """スキーマグラフ（エンティティタイプとリレーションの有向グラフ）"""

    def __init__(self):
        self.adjacency: Dict[str, List[Tuple[str, str, str]]] = {}  # type -> [(neighbor_type, relation, direction), ...]
        self.all_types: Set[str] = set()
        self.all_relations: Set[str] = set()

    def add_edge(self, src_type: str, relation: str, tgt_type: str, direction: str = "->"):
        """エッジを追加（有向、方向情報も保持）"""
        self.all_types.add(src_type)
        self.all_types.add(tgt_type)
        self.all_relations.add(relation)

        if src_type not in self.adjacency:
            self.adjacency[src_type] = []
        if tgt_type not in self.adjacency:
            self.adjacency[tgt_type] = []

        # 有向グラフとして順方向のみ追加（方向情報も保持）
        self.adjacency[src_type].append((tgt_type, relation, direction))

    def bfs_all_shortest_paths(
        self,
        start_type: str,
        end_type: str,
        max_depth: int = 4
    ) -> List[SchemaPath]:
        """BFSで最短パスを探索（shortest_onlyモードでは最短のみ）"""
        return self.find_all_paths(start_type, end_type, max_depth, shortest_only=True)

    def find_all_paths(
        self,
        start_type: str,
        end_type: str,
        max_depth: int = 3,
        shortest_only: bool = False
    ) -> List[SchemaPath]:
        """全てのパスを探索（DFS）"""
        if start_type not in self.adjacency or end_type not in self.adjacency:
            return []

        all_paths: List[SchemaPath] = []
        visited = {start_type}

        def dfs(current: str, type_path: List[str], rel_path: List[str], dir_path: List[str], depth: int):
            if depth > max_depth:
                return

            # 目的地に到達
            if current == end_type and depth > 0:
                full_path = []
                for i, t in enumerate(type_path):
                    full_path.append(t)
                    if i < len(rel_path):
                        full_path.append(rel_path[i])
                all_paths.append(SchemaPath(
                    path=full_path,
                    types=type_path.copy(),
                    relations=rel_path.copy(),
                    directions=dir_path.copy()
                ))
                return

            # 隣接ノードを探索
            for neighbor, relation, direction in self.adjacency.get(current, []):
                if neighbor not in visited or neighbor == end_type:
                    visited.add(neighbor)
                    dfs(
                        neighbor,
                        type_path + [neighbor],
                        rel_path + [relation],
                        dir_path + [direction],
                        depth + 1
                    )
                    if neighbor != end_type:
                        visited.discard(neighbor)

        dfs(start_type, [start_type], [], [], 0)

        # 深さでソート
        all_paths.sort(key=lambda p: len(p.relations))

        if shortest_only and all_paths:
            min_depth = len(all_paths[0].relations)
            all_paths = [p for p in all_paths if len(p.relations) == min_depth]

        return all_paths


def build_schema_graph(kg_type: str = "primekgqa") -> KGSchema:
    """スキーマグラフを構築

    Args:
        kg_type: "primekgqa", "metaqa", or "pcqa"
    """
    schema = KGSchema()

    if kg_type == "metaqa":
        from dataset_construction.schema_metaqa import SCHEMA_GRAPH
        for src, rel, tgt, direction in SCHEMA_GRAPH:
            schema.add_edge(src, rel, tgt, direction)
    elif kg_type == "pcqa":
        from dataset_construction.schema_pcqa import SCHEMA_GRAPH
        for src, rel, tgt, direction in SCHEMA_GRAPH:
            schema.add_edge(src, rel, tgt, direction)
    else:
        # PrimeKGQA: v3スキーマを使用（3要素タプル、方向はデフォルトで"->"）
        from dataset_construction.schema_v3 import SCHEMA_GRAPH
        for src, rel, tgt in SCHEMA_GRAPH:
            schema.add_edge(src, rel, tgt, "->")

    return schema
