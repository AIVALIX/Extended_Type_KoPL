"""
neo4j_jsonl_cli.processor
=========================
Import JSON Lines (JSONL) into Neo4j.

• Entity nodes           : (:Entity {name, type:[…]})
• RelationEmbedding nodes: (:RelationEmbedding {relation_name, embedding_vector})
• Dynamic relationships  : (source)-[:<relation> {id, source_type, target_type}]->(target)

Assumptions
-----------
1. APOC Core is installed on the Neo4j server (for apoc.coll.toSet and apoc.merge.relationship).
2. `core.config.get_settings()` exposes NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD.
3. `llm_process.embedder.OpenAIEmbedder` implements `.embed_many(iterable[str]) -> list[list[float]]`.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable, List, Dict

from py2neo import Graph
from llm_process.embedder import OpenAIEmbedder
from core.config import get_settings


# --------------------------------------------------------------------------- #
#                                Core processor                               #
# --------------------------------------------------------------------------- #
class GraphBatchProcessor:
    """Stream-imports a JSONL file into Neo4j."""

    def __init__(self, batch_size: int = 5_000, embed_batch_size: int = 100):
        cfg = get_settings()
        self.graph = Graph(cfg.NEO4J_URI, auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD))
        self.embedder = OpenAIEmbedder()
        self.batch_size = batch_size
        self.embed_batch_size = embed_batch_size

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def import_jsonl(self, jsonl_path: Path) -> None:
        """Stream a JSONL file and import it in batches."""
        with jsonl_path.open(encoding="utf-8") as f:
            buf: List[str] = []
            for line in f:
                if line.strip():
                    buf.append(line)
                if len(buf) >= self.batch_size:
                    self._process_batch(buf)
                    buf.clear()
            if buf:
                self._process_batch(buf)

    def process_jsonl_file(self, jsonl_path: str) -> None:
        """Process JSONL file - wrapper for import_jsonl"""
        self.import_jsonl(Path(jsonl_path))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _process_batch(self, lines: List[str]) -> None:
        records = [json.loads(l) for l in lines]

        # 1) Entity nodes ---------------------------------------------- #
        self._upsert_entities(self._collect_entities(records))

        # 2) RelationEmbedding nodes ----------------------------------- #
        self._upsert_relation_embeddings(records)

        # 3) Dynamic relationships ------------------------------------- #
        self._merge_relationships(records)

    # -- Entity -------------------------------------------------------- #
    @staticmethod
    def _collect_entities(rels: List[dict]) -> Dict[str, dict]:
        ent: Dict[str, dict] = {}
        for r in rels:
            ent.setdefault(r["source"], {"name": r["source"], "type": r["source_type"]})
            ent.setdefault(r["target"], {"name": r["target"], "type": r["target_type"]})
        return ent

    def _upsert_entities(self, entities: Dict[str, dict]) -> None:
        """
        MERGE Entity nodes.
        `type` is kept as a list[str] and new types are appended without duplication.
        Requires apoc.coll.toSet.
        """
        q = """
        UNWIND $rows AS row
        MERGE (e:Entity {name: row.name})
        WITH e, row.type AS newList
        SET  e.type = apoc.coll.toSet(coalesce(e.type, []) + newList)
        """
        self.graph.run(q, rows=list(entities.values()))

    # -- RelationEmbedding -------------------------------------------- #
    def _upsert_relation_embeddings(self, records: List[dict]) -> None:
        uniq = sorted({r["relation"] for r in records})
        self._upsert_relation_embeddings_from_names(uniq)

    def _upsert_relation_embeddings_from_names(self, rel_names: List[str]) -> None:
        if not rel_names:
            return

        for chunk in _chunked(rel_names, self.embed_batch_size):
            vectors = self.embedder.embed_many(chunk)
            rows = [{"rel": r, "vec": v} for r, v in zip(chunk, vectors)]
            q = """
            UNWIND $rows AS row
            MERGE (re:RelationEmbedding {relation_name: row.rel})
            SET   re.embedding_vector = row.vec
            """
            self.graph.run(q, rows=rows)

    def refresh_relation_embeddings_from_graph(self) -> None:
        """Fetch DISTINCT relationship types already stored in Neo4j and refresh embeddings."""
        q = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel"
        rels = sorted(row["rel"] for row in self.graph.run(q) if row["rel"])
        self._upsert_relation_embeddings_from_names(rels)

    # -- Dynamic relationships ---------------------------------------- #
    def _merge_relationships(self, records: List[dict]) -> None:
        """
        Create or merge dynamic relationships whose type matches the JSONL `relation`
        string exactly (e.g., :directed_by, :written_by).
        """
        rows = [
            {
                "src": r["source"],
                "dst": r["target"],
                "rel": r["relation"],  # keep as-is
                "props": {
                    "id": str(uuid.uuid4()),
                    "source_type": r["source_type"],
                    "target_type": r["target_type"],
                },
            }
            for r in records
        ]

        q = """
        UNWIND $rows AS row
        MATCH (s:Entity {name: row.src})
        MATCH (t:Entity {name: row.dst})
        CALL apoc.merge.relationship(s, row.rel, {}, row.props, t) YIELD rel
        RETURN count(rel)
        """

        for chunk in _chunked(rows, self.batch_size):
            self.graph.run(q, rows=chunk)


# --------------------------------------------------------------------------- #
#                                util helpers                                #
# --------------------------------------------------------------------------- #
def _chunked(seq: Iterable, size: int):
    """Yield lists of length `size` (or smaller for the last chunk)."""
    buf: List = []
    for item in seq:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


# CLI / サンプル実行
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    sample_path = "/app/data/metaqa/seed/seed.jsonl"  # 例: JSONL ファイルパス
    processor = GraphBatchProcessor(batch_size=2_000, embed_batch_size=50)
    processor.refresh_relation_embeddings_from_graph()
    print("インポート完了")

# python database/seed.py
