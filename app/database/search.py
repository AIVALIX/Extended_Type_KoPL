from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from py2neo import Graph
from core.config import get_settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ────────────────────────────────────────────────────────────────
#  Graph utilities
# ────────────────────────────────────────────────────────────────
class GraphPathFinder:
    r"""
    指定エンティティから **ちょうど hop ホップ** で到達できるパスを取得し，
    リレーションタイプ列と RelationEmbedding ベクトル列を返す。

    * ノードは `type` プロパティに **リスト** で実体タイプを保持
      例: {type:["Person"]} もしくは {type:["Date"]}

    * `type_path` が与えられた場合
      （例: ["Person","CreativeWork"]、hop=2 なら 1→2 ホップのターゲット順）
      各ホップ先ノードの `type` リストに期待要素が含まれるものだけを採用
    """

    def __init__(self, kg_type: str = "primekgqa") -> None:
        s = get_settings()
        # KGタイプに応じたNeo4j接続を使用
        if kg_type == "metaqa":
            import os
            uri = os.getenv("NEO4J_METAQA_URI", "bolt://neo4j_metaqa:7687")
            user = os.getenv("NEO4J_METAQA_USER", "neo4j")
            password = os.getenv("NEO4J_METAQA_PASSWORD", "password")
        else:
            uri = s.NEO4J_URI
            user = s.NEO4J_USERNAME
            password = s.NEO4J_PASSWORD
        self.graph = Graph(uri, auth=(user, password))

    # ──────────────────────────────────────────────
    #  Public
    # ──────────────────────────────────────────────
    def get_sequences_and_embeddings(
        self,
        entity_name: str,
        hop: int,
        type_path: Optional[List[str]] = None,
    ) -> Tuple[List[List[str]], List[List[np.ndarray]]]:
        """パスのリレーションタイプ列と対応ベクトル列を返す"""
        if type_path and len(type_path) != hop:
            raise ValueError("len(type_path) must equal hop")

        paths = self._get_relation_sequences(entity_name, hop, type_path)
        if not paths:
            return [], []

        uniq_rels = sorted({rel for seq in paths for rel in seq})
        rel2vec = self._fetch_relation_embeddings(uniq_rels)

        vecs: List[List[np.ndarray]] = [
            [rel2vec[rel] for rel in seq if rel in rel2vec] for seq in paths
        ]
        return paths, vecs

    def get_reachable_entities(
        self,
        entity_name: str,
        rel_types: List[str],
        distinct: bool = True,
        no_cycle: bool = True,
    ) -> List[str]:
        """
        連続したリレーションタイプ列をたどって到達できる終点エンティティ名を返す（hop展開・固定長）。
        """
        if not rel_types:
            return [entity_name]

        # 1) 先頭の start マッチ
        cypher_lines = ["MATCH (start:Entity {name:$entity_name})"]

        # 2) hop を展開して固定長にする
        prev = "start"
        for i, _ in enumerate(rel_types):
            curr = f"n{i}"
            ri = f"r{i}"

            # 無向マッチ + リレーション種別の一致（r.text があれば優先）
            cypher_lines.append(f"MATCH ({prev})-[{ri}]-({curr}:Entity)")
            where_parts = [f"coalesce({ri}.text, type({ri})) = $rel{i}"]

            # サイクル禁止（start 再訪 & 既出ノード再訪禁止）
            if no_cycle:
                # start 再訪禁止
                where_parts.append(f"{curr}.name <> start.name")

            cypher_lines.append("WHERE " + " AND ".join(where_parts))
            prev = curr

        # 3) 終端ノード（最後の n{L-1}）を返す
        last = f"n{len(rel_types) - 1}"
        ret_distinct = "DISTINCT " if distinct else ""
        cypher_lines.append(f"RETURN {ret_distinct}{last}.name AS name")

        query = "\n".join(cypher_lines)

        # パラメータ
        params = {"entity_name": entity_name}
        for i, rel in enumerate(rel_types):
            params[f"rel{i}"] = rel

        rows = self.graph.run(query, **params).data()
        return [row["name"] for row in rows]

    # ──────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────
    def _get_relation_sequences(
        self,
        entity_name: str,
        hop: int,
        type_path: Optional[List[str]],
    ) -> List[List[str]]:
        """Cypher でパス列取得（ノード再訪禁止 & type_path フィルタ）"""
        hop_int = int(hop)  # ← hop を文字列埋め込み（可変長パターンにパラメータ不可）

        # ---- type_path フィルタ句を動的生成 -------------------- #
        type_checks = ""
        params: Dict[str, str] = {"entity_name": entity_name}
        if type_path:
            clauses = []
            for i, etype in enumerate(type_path):
                key = f"tp{i}"
                params[key] = etype
                # nodes(p)[1] が 1hop 先, nodes(p)[i+1] が i+1 hop 先
                #   例: 'Person' IN nds[1].type
                clauses.append(f"${key} IN nds[{i + 1}].type")
            type_checks = "AND " + " AND ".join(clauses)

        query = f"""
        MATCH p = (start:Entity {{name:$entity_name}})-[*{hop_int}]-(dest)
        WHERE size(nodes(p)) = size(apoc.coll.toSet(nodes(p)))   /* ノード再訪禁止 */
        WITH nodes(p) AS nds, relationships(p) AS rels
        WHERE 1=1 {type_checks}                                  /* type_path フィルタ */
        WITH [r IN rels | type(r)] AS rel_types
        RETURN DISTINCT rel_types AS rel_types
        """
        rows = self.graph.run(query, **params).data()
        return [row["rel_types"] for row in rows]

    def _fetch_relation_embeddings(self, rel_names: List[str]) -> Dict[str, np.ndarray]:
        """RelationEmbedding ノードからベクトルを辞書化"""
        if not rel_names:
            return {}

        query = """
        UNWIND $rels AS r
        MATCH (e:RelationEmbedding {relation_name:r})
        RETURN e.relation_name AS rel, e.embedding_vector AS vec
        """
        rows = self.graph.run(query, rels=rel_names).data()

        rel2vec: Dict[str, np.ndarray] = {
            row["rel"]: np.asarray(row["vec"], dtype=np.float32) for row in rows
        }

        missing = set(rel_names) - rel2vec.keys()
        if missing:
            logger.warning("Missing embeddings: %s", missing)

        return rel2vec


# ────────────────────────────────────────────────────────────────
#  Scoring utilities
# ────────────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom) if denom else 0.0


def score_paths(
    query_vecs: List[np.ndarray],
    path_vectors: List[List[np.ndarray]],
    top_k: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    同インデックス要素同士のコサイン類似度を総和してパスを評価。
    スコア降順に並べ、top_k が指定されれば上位 k 件を返す。
    """
    scores: List[Tuple[int, float]] = []
    Q = len(query_vecs)

    for idx, p_vecs in enumerate(path_vectors):
        L = min(Q, len(p_vecs))
        if L == 0:
            scores.append((idx, -np.inf))
            continue
        sim_sum = sum(cosine_sim(query_vecs[i], p_vecs[i]) for i in range(L))
        scores.append((idx, sim_sum))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k] if top_k is not None else scores


# ────────────────────────────────────────────────────────────────
#  Example usage
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    finder = GraphPathFinder()

    start_entity = "Chinatown"
    hop = 3
    type_path = ["Person", "CreativeWork", "Date"]

    # ❶ パスと埋め込みの取得
    paths, path_vecs = finder.get_sequences_and_embeddings(
        entity_name=start_entity,
        hop=hop,
        type_path=type_path,
    )

    if not paths:
        print("No path matched the given constraints.")
        exit(0)

    # ❷ クエリ側ベクトル列（ここではダミーを使用）
    query_vecs = [np.random.rand(1536).astype(np.float32) for _ in range(hop)]

    # ❸ スコアリング
    ranked = score_paths(query_vecs, path_vecs, top_k=1)  # ← 最良 1 本だけ

    best_idx, best_score = ranked[0]
    best_rel_path = paths[best_idx]

    print("\nBest path:")
    print(f"  score  : {best_score:.3f}")
    print(f"  rels   : {best_rel_path}")

    # ❹ 最高スコアのリレーション列で到達エンティティを取得
    reachable = finder.get_reachable_entities(
        entity_name=start_entity,
        rel_types=best_rel_path,
        distinct=True,
        no_cycle=True,
    )

    print("\nReachable entities via best path:")
    for name in reachable:
        print("  •", name)
    reachable = finder.get_reachable_entities(
        entity_name="A Home at the End of the World",
        rel_types=["directed_by", "directed_by", "written_by"],
        distinct=True,
        no_cycle=True,
    )
    print("\nReachable entities via best path:", reachable)
# python database/search.py
