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
        import os
        self.kg_type = kg_type  # KGタイプを保存
        # KGタイプに応じたNeo4j接続を使用
        if kg_type == "metaqa":
            uri = os.getenv("NEO4J_METAQA_URI", "bolt://neo4j_metaqa:7687")
            user = os.getenv("NEO4J_METAQA_USER", "neo4j")
            password = os.getenv("NEO4J_METAQA_PASSWORD", "password")
        elif kg_type == "pcqa":
            uri = os.getenv("NEO4J_PCQA_URI", "bolt://neo4j_pcqa:7687")
            user = os.getenv("NEO4J_PCQA_USER", "neo4j")
            password = os.getenv("NEO4J_PCQA_PASSWORD", "password")
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

    def get_relations_between_types(
        self,
        src_type: str,
        tgt_type: str,
    ) -> List[Tuple[str, str]]:
        """KGから2つのタイプ間で利用可能なリレーションを動的取得

        Args:
            src_type: ソースノードのタイプ
            tgt_type: ターゲットノードのタイプ

        Returns:
            List of (relation_name, direction) tuples
            direction: "->" for forward, "<-" for reverse
        """
        relations = []

        # ラベルにスラッシュが含まれる場合はバッククォートでエスケープ
        def escape_label(label: str) -> str:
            if "/" in label:
                return f"`{label}`"
            return label

        src_label = escape_label(src_type)
        tgt_label = escape_label(tgt_type)

        # 全KGタイプで統一: ノードのラベルがタイプ
        query_fwd = f"""
        MATCH (s:{src_label})-[r]->(t:{tgt_label})
        RETURN DISTINCT type(r) AS rel
        """
        rows_fwd = self.graph.run(query_fwd).data()
        for row in rows_fwd:
            relations.append((row["rel"], "->"))

        # 逆方向
        query_rev = f"""
        MATCH (s:{src_label})<-[r]-(t:{tgt_label})
        RETURN DISTINCT type(r) AS rel
        """
        rows_rev = self.graph.run(query_rev).data()
        for row in rows_rev:
            if (row["rel"], "->") not in relations:
                relations.append((row["rel"], "<-"))

        return relations

    def get_all_types(self) -> List[str]:
        """KGから全てのエンティティタイプを取得"""
        # 全KGタイプで統一: ラベルがタイプ
        query = """
        CALL db.labels() YIELD label
        WHERE label <> 'RelationEmbedding'
        RETURN label AS type
        ORDER BY label
        """
        rows = self.graph.run(query).data()
        return [row["type"] for row in rows]

    def get_adjacent_relations(self, entity_name: str) -> List[str]:
        """エンティティに隣接する全関係タイプを取得 (AdjRel)

        Args:
            entity_name: エンティティ名

        Returns:
            関係タイプ名のリスト
        """
        query = """
        MATCH (e {name: $name})-[r]-()
        RETURN DISTINCT type(r) AS rel
        """
        rows = self.graph.run(query, name=entity_name).data()
        return [row["rel"] for row in rows]

    def get_candidate_nodes(
        self, entity_name: str, relation: str
    ) -> List[Tuple[str, List[str]]]:
        """特定関係で到達可能なノード名とラベルを取得 (GetCandNode)

        Args:
            entity_name: 起点エンティティ名
            relation: 関係タイプ

        Returns:
            List of (node_name, labels) タプル
        """
        def escape_rel(r: str) -> str:
            if " " in r or "-" in r or "/" in r:
                return f"`{r}`"
            return r

        rel_escaped = escape_rel(relation)

        # 両方向を検索
        query = f"""
        MATCH (e {{name: $name}})-[r:{rel_escaped}]-(n)
        RETURN DISTINCT n.name AS name, labels(n) AS labels
        """
        rows = self.graph.run(query, name=entity_name).data()
        return [(row["name"], row["labels"]) for row in rows]

    def get_entity_labels(self, entity_name: str) -> List[str]:
        """エンティティのNeo4jラベル一覧を取得

        Args:
            entity_name: エンティティ名

        Returns:
            ラベル名のリスト
        """
        query = """
        MATCH (e {name: $name})
        RETURN labels(e) AS labels
        LIMIT 1
        """
        rows = self.graph.run(query, name=entity_name).data()
        if rows:
            return rows[0]["labels"]
        return []

    def find_compound_entity(
        self, terms: List[str], label: Optional[str] = None
    ) -> List[Tuple[str, List[str]]]:
        """CONTAINS検索で複合名エンティティを検索（PcQA CancerCell用）

        Args:
            terms: 検索キーワードのリスト（例: ["EGFR", "lung cancer"]）
            label: ラベルでフィルタ（例: "CancerCell"）

        Returns:
            List of (entity_name, labels) tuples
        """
        if not terms:
            return []

        where_parts = []
        for i in range(len(terms)):
            where_parts.append(
                f"(toLower(n.name) CONTAINS toLower($term{i}) "
                f"OR toLower(coalesce(n.name_en, '')) CONTAINS toLower($term{i}))"
            )

        label_clause = f":{label}" if label else ""
        query = f"""
        MATCH (n{label_clause})
        WHERE {' AND '.join(where_parts)}
        RETURN n.name AS name, labels(n) AS labels
        LIMIT 10
        """
        params = {f"term{i}": term for i, term in enumerate(terms)}
        try:
            rows = self.graph.run(query, **params).data()
            return [(row["name"], row["labels"]) for row in rows]
        except Exception:
            return []

    def find_compound_entity_via_cancer(
        self, gene_term: str, cancer_term: str
    ) -> List[Tuple[str, List[str]]]:
        """Cancer→CancerCell関係を使って複合エンティティを特定

        Gene名でCancerCellをCONTAINS検索し、さらにCancer名（英語）で
        ORIGINATED_FROM関係を使ってフィルタする。

        Args:
            gene_term: 遺伝子名（例: "ALK"）
            cancer_term: 癌種名（英語、例: "giant cell carcinoma of the lung"）

        Returns:
            List of (entity_name, labels) tuples
        """
        query = """
        MATCH (cc:CancerCell)-[:ORIGINATED_FROM]->(c:Cancer)
        WHERE toLower(cc.name) CONTAINS toLower($gene)
          AND toLower(c.name) CONTAINS toLower($cancer)
        RETURN cc.name AS name, labels(cc) AS labels
        LIMIT 10
        """
        try:
            rows = self.graph.run(query, gene=gene_term, cancer=cancer_term).data()
            return [(row["name"], row["labels"]) for row in rows]
        except Exception:
            return []

    def get_all_entity_names(self) -> List[str]:
        """KGから全エンティティ名を取得（SimEntインデックス構築用）

        Returns:
            エンティティ名のリスト
        """
        query = """
        MATCH (e)
        WHERE NOT 'RelationEmbedding' IN labels(e)
          AND NOT 'EntityEmbedding' IN labels(e)
          AND e.name IS NOT NULL
        RETURN DISTINCT e.name AS name
        """
        rows = self.graph.run(query).data()
        return [row["name"] for row in rows]

    def get_entity_embeddings(self) -> Tuple[List[str], List[np.ndarray]]:
        """事前計算済みEntityEmbeddingをDBから一括読み込み

        Returns:
            (entity_names, embedding_vectors) のタプル
        """
        query = """
        MATCH (e:EntityEmbedding)
        RETURN e.entity_name AS name, e.embedding_vector AS vec
        """
        rows = self.graph.run(query).data()
        if not rows:
            return [], []
        names = [row["name"] for row in rows]
        vecs = [np.asarray(row["vec"], dtype=np.float32) for row in rows]
        return names, vecs

    def get_all_relations(self) -> List[Tuple[str, str, str]]:
        """KGから全てのリレーション情報を取得

        Returns:
            List of (src_type, relation, tgt_type) tuples
        """
        # 全KGタイプで統一: ラベルがタイプ
        query = """
        MATCH (s)-[r]->(t)
        WHERE NOT 'RelationEmbedding' IN labels(s) AND NOT 'RelationEmbedding' IN labels(t)
        WITH labels(s) AS src_labels, type(r) AS rel, labels(t) AS tgt_labels
        UNWIND src_labels AS src_t
        UNWIND tgt_labels AS tgt_t
        RETURN DISTINCT src_t AS src_type, rel, tgt_t AS tgt_type
        """
        rows = self.graph.run(query).data()
        return [(row["src_type"], row["rel"], row["tgt_type"]) for row in rows]


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
