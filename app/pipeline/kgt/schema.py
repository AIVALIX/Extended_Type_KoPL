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
        """BFSで全パスを探索（原論文準拠: nx.all_shortest_paths相当）

        shortest_only=True の場合、最短深さのパスのみ返す。
        shortest_only=False の場合、max_depth以内の全パスを返す。
        """
        if start_type not in self.adjacency or end_type not in self.adjacency:
            return []

        from collections import deque

        all_paths: List[SchemaPath] = []
        shortest_depth: Optional[int] = None

        # BFS: (current_type, type_path, rel_path, dir_path, visited_set)
        queue: deque = deque()
        queue.append((start_type, [start_type], [], [], {start_type}))

        while queue:
            current, type_path, rel_path, dir_path, visited = queue.popleft()
            depth = len(rel_path)

            # shortest_only で最短より深いパスはスキップ
            if shortest_only and shortest_depth is not None and depth >= shortest_depth:
                continue
            if depth >= max_depth:
                continue

            for neighbor, relation, direction in self.adjacency.get(current, []):
                if neighbor == end_type:
                    new_type_path = type_path + [neighbor]
                    new_rel_path = rel_path + [relation]
                    new_dir_path = dir_path + [direction]
                    full_path = []
                    for i, t in enumerate(new_type_path):
                        full_path.append(t)
                        if i < len(new_rel_path):
                            full_path.append(new_rel_path[i])
                    all_paths.append(SchemaPath(
                        path=full_path,
                        types=new_type_path,
                        relations=new_rel_path,
                        directions=new_dir_path,
                    ))
                    if shortest_only and shortest_depth is None:
                        shortest_depth = depth + 1
                elif neighbor not in visited:
                    new_visited = visited | {neighbor}
                    queue.append((
                        neighbor,
                        type_path + [neighbor],
                        rel_path + [relation],
                        dir_path + [direction],
                        new_visited,
                    ))

        all_paths.sort(key=lambda p: len(p.relations))
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
