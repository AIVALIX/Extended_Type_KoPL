"""
SAFE スキーマグラフ（APSP付き）
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from pipeline.safe.models import SchemaEdge


class CandidateIndex:
    """SimRel/SimTyp/SimEnt用のインメモリ最近傍インデックス（numpy L2距離）

    KGが小規模（~12関係, ~5タイプ）なのでFAISSなしで十分。
    エンティティ数が大きい場合（~29K）はバッチ分割で対応。
    """

    EMBED_BATCH_SIZE = 2000  # OpenAI Embeddings バッチ制限

    def __init__(self, embeddings_model):
        self._embeddings = embeddings_model
        self._relation_names: List[str] = []
        self._relation_vecs: Optional[np.ndarray] = None  # (N, dim)
        self._type_names: List[str] = []
        self._type_vecs: Optional[np.ndarray] = None  # (M, dim)
        self._entity_names: List[str] = []
        self._entity_vecs: Optional[np.ndarray] = None  # (E, dim)
        self._query_cache: Dict[str, np.ndarray] = {}

    def _embed_query(self, text: str) -> np.ndarray:
        """クエリをベクトル化（キャッシュ付き）"""
        if text not in self._query_cache:
            self._query_cache[text] = np.array(
                self._embeddings.embed_query(text), dtype=np.float32
            )
        return self._query_cache[text]

    def _embed_documents(self, texts: List[str]) -> np.ndarray:
        """複数テキストをベクトル化"""
        vecs = self._embeddings.embed_documents(texts)
        return np.array(vecs, dtype=np.float32)

    def build_relation_index(self, relations: List[str]):
        """KG全関係をベクトル化してインデックス構築"""
        self._relation_names = list(relations)
        if relations:
            self._relation_vecs = self._embed_documents(relations)

    def build_type_index(self, types: List[str]):
        """KG全タイプをベクトル化してインデックス構築"""
        self._type_names = list(types)
        if types:
            self._type_vecs = self._embed_documents(types)

    def sim_rel(self, query_rel: str, k: int = 10) -> List[Tuple[str, float]]:
        """類似関係 top-k を返す (relation_name, L2_distance)"""
        if self._relation_vecs is None or len(self._relation_names) == 0:
            return []
        q_vec = self._embed_query(query_rel)
        dists = np.linalg.norm(self._relation_vecs - q_vec, axis=1)
        top_k = min(k, len(self._relation_names))
        indices = np.argsort(dists)[:top_k]
        return [(self._relation_names[i], float(dists[i])) for i in indices]

    def sim_typ(self, query_type: str, k: int = 8) -> List[Tuple[str, float]]:
        """類似タイプ top-k を返す (type_name, L2_distance)"""
        if self._type_vecs is None or len(self._type_names) == 0:
            return []
        q_vec = self._embed_query(query_type)
        dists = np.linalg.norm(self._type_vecs - q_vec, axis=1)
        top_k = min(k, len(self._type_names))
        indices = np.argsort(dists)[:top_k]
        return [(self._type_names[i], float(dists[i])) for i in indices]

    def build_entity_index(self, entity_names: List[str]):
        """KG全エンティティをベクトル化してインデックス構築（バッチ分割対応）"""
        self._entity_names = list(entity_names)
        if not entity_names:
            return
        # バッチ分割で埋め込み
        all_vecs = []
        for i in range(0, len(entity_names), self.EMBED_BATCH_SIZE):
            batch = entity_names[i : i + self.EMBED_BATCH_SIZE]
            all_vecs.append(self._embed_documents(batch))
        self._entity_vecs = np.concatenate(all_vecs, axis=0)

    def load_entity_index(self, entity_names: List[str], entity_vecs):
        """事前計算済みベクトルからエンティティインデックスを構築（API呼び出しなし）

        entity_vecs: 2D numpy array (N, dim) or List[np.ndarray]
        """
        self._entity_names = entity_names
        if isinstance(entity_vecs, np.ndarray) and entity_vecs.ndim == 2:
            self._entity_vecs = entity_vecs
        elif entity_vecs:
            self._entity_vecs = np.stack(entity_vecs)
        else:
            self._entity_vecs = None

    def sim_ent(self, query_entity: str, k: int = 3) -> List[Tuple[str, float]]:
        """類似エンティティ top-k を返す (entity_name, L2_distance)

        完全一致が結果に含まれない場合は距離0で先頭に挿入する。
        """
        if self._entity_vecs is None or len(self._entity_names) == 0:
            return [(query_entity, 0.0)]
        q_vec = self._embed_query(query_entity)
        dists = np.linalg.norm(self._entity_vecs - q_vec, axis=1)
        top_k = min(k, len(self._entity_names))
        indices = np.argsort(dists)[:top_k]
        results = [(self._entity_names[i], float(dists[i])) for i in indices]

        # 完全一致を保証
        result_names = {r[0] for r in results}
        if query_entity not in result_names:
            # 完全一致をdist=0で先頭に追加し、末尾を削除
            results = [(query_entity, 0.0)] + results[: top_k - 1]

        return results

    def sem_dist_rel(self, query_rel: str, kg_rel: str) -> float:
        """リレーション単体のSemDist（L2距離）

        論文: d_r = ||emb(r) - emb(r_G)||₂
        """
        q_vec = self._embed_query(query_rel)
        kg_vec = self._embed_query(kg_rel)
        return float(np.linalg.norm(q_vec - kg_vec))

    def sem_dist_node(self, query_type: str, kg_labels: List[str]) -> float:
        """ノードタイプのSemDist（L2距離、最小値）

        論文: d_u' = min_{t ∈ type(u'_G)} ||emb(type(u')) - emb(t)||₂
        タイプが不明な場合は0を返す。
        """
        if not query_type or not kg_labels:
            return 0.0
        q_vec = self._embed_query(query_type)
        min_dist = float('inf')
        for label in kg_labels:
            kg_vec = self._embed_query(label)
            d = float(np.linalg.norm(q_vec - kg_vec))
            min_dist = min(min_dist, d)
        return min_dist

    def sem_dist_edge(self, query_edge_text: str, kg_edge_text: str) -> float:
        """エッジ全体のSemDist（フルトリプル埋め込みのL2距離）

        論文: SemDist(e_Q, e_G) = ||emb(e_Q) - emb(e_G)||₂
        e_Q, e_G は "src_type relation tgt_type" 形式のテキスト
        """
        q_vec = self._embed_query(query_edge_text)
        kg_vec = self._embed_query(kg_edge_text)
        return float(np.linalg.norm(q_vec - kg_vec))


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


def build_schema_with_apsp(kg_type: str = "primekgqa", types_only: bool = False) -> SchemaGraphWithAPSP:
    """スキーマグラフをAPSP付きで構築

    Args:
        kg_type: "primekgqa" or "metaqa"
        types_only: Trueの場合、タイプ情報のみを含むスキーマを構築（エッジはKGから動的取得）
    """
    schema = SchemaGraphWithAPSP()

    if types_only:
        # タイプ情報のみを追加
        if kg_type == "metaqa":
            from dataset_construction.schema_metaqa import ENTITY_TYPES
            entity_types = ENTITY_TYPES
        elif kg_type == "pcqa":
            from dataset_construction.schema_pcqa import ENTITY_TYPES
            entity_types = ENTITY_TYPES
        else:
            # PrimeKGQA: v3スキーマを使用
            from dataset_construction.schema_v3 import NODE_TYPES
            entity_types = NODE_TYPES

        for t in entity_types:
            schema.types.add(t)
    else:
        # 通常通りスキーマからエッジを追加
        if kg_type == "metaqa":
            from dataset_construction.schema_metaqa import SCHEMA_GRAPH
            for src, rel, tgt, _ in SCHEMA_GRAPH:
                schema.add_edge(src, rel, tgt)
        elif kg_type == "pcqa":
            from dataset_construction.schema_pcqa import SCHEMA_GRAPH
            for src, rel, tgt, _ in SCHEMA_GRAPH:
                schema.add_edge(src, rel, tgt)
        else:
            # PrimeKGQA: v3スキーマを使用（3要素タプル）
            from dataset_construction.schema_v3 import SCHEMA_GRAPH
            for src, rel, tgt in SCHEMA_GRAPH:
                schema.add_edge(src, rel, tgt)

    schema.build_apsp()
    return schema


def build_schema_from_kg(finder) -> SchemaGraphWithAPSP:
    """KGから動的にスキーマグラフを構築

    Args:
        finder: GraphPathFinder instance

    Returns:
        SchemaGraphWithAPSP: KGから構築されたスキーマグラフ
    """
    schema = SchemaGraphWithAPSP()

    # KGから全リレーション情報を取得
    relations = finder.get_all_relations()
    for src, rel, tgt in relations:
        schema.add_edge(src, rel, tgt)

    schema.build_apsp()
    return schema
