from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Set

from neo4j import Driver, GraphDatabase

from core.config import get_settings
from dataset_construction.Cypher_query import (
    build_one_hop_chain_cypher,
    build_two_hop_chain_cypher,
    build_two_anchor_intersection_cypher,
    build_three_anchor_intersection_cypher,
    simple_one_hop_chain_cypher,
    simple_two_hop_chain_cypher,
    simple_two_anchor_intersection_cypher,
    simple_three_anchor_intersection_cypher,
)


@dataclass(frozen=True)
class QuerySpec:
    name: str
    cypher: str


def _iter_jsonl_rows(rows: Iterable[Dict[str, Any]]) -> Iterable[str]:
    for row in rows:
        yield json.dumps(row, ensure_ascii=False)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in _iter_jsonl_rows(rows):
            f.write(line)
            f.write("\n")


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in _iter_jsonl_rows(rows):
            f.write(line)
            f.write("\n")


def _connect_graph() -> Driver:
    s = get_settings()
    return GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USERNAME, s.NEO4J_PASSWORD))


def run_queries_and_save(
    out_dir: Path,
    *,
    node_label: str,
    limit_per_query: int,
    onehop_anchor_pool: int,
    twohop_anchor_pool: int,
    two_anchor_pool: int,
    two_anchor_tries: int,
    three_anchor_pool: int,
    three_anchor_tries: int,
    seed: int,
    simple: bool,
) -> Tuple[Path, List[Tuple[str, int]]]:
    """Run the 4 dataset-construction queries and write results as JSONL.

    Returns
    -------
    (combined_jsonl_path, counts)
      counts: list of (query_name, num_rows)
    """

    if simple:
        specs = [
            QuerySpec("one_hop_chain", simple_one_hop_chain_cypher),
            QuerySpec("two_hop_chain", simple_two_hop_chain_cypher),
            QuerySpec("two_anchor_intersection", simple_two_anchor_intersection_cypher),
            QuerySpec(
                "three_anchor_intersection", simple_three_anchor_intersection_cypher
            ),
        ]
    else:
        specs = [
            QuerySpec(
                "one_hop_chain",
                build_one_hop_chain_cypher(
                    node_label=node_label,
                    anchor_pool=onehop_anchor_pool,
                    limit=limit_per_query,
                ),
            ),
            QuerySpec(
                "two_hop_chain",
                build_two_hop_chain_cypher(
                    node_label=node_label,
                    anchor_pool=twohop_anchor_pool,
                    limit=limit_per_query,
                ),
            ),
            QuerySpec(
                "two_anchor_intersection",
                build_two_anchor_intersection_cypher(
                    node_label=node_label,
                    anchor_pool=two_anchor_pool,
                    num_tries=two_anchor_tries,
                    limit=limit_per_query,
                ),
            ),
            QuerySpec(
                "three_anchor_intersection",
                build_three_anchor_intersection_cypher(
                    node_label=node_label,
                    anchor_pool=three_anchor_pool,
                    num_tries=three_anchor_tries,
                    limit=limit_per_query,
                ),
            ),
        ]

    driver = _connect_graph()

    counts: List[Tuple[str, int]] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "all.jsonl"
    # reset outputs
    if combined_path.exists():
        combined_path.unlink()

    def key_for(name: str, row: Dict[str, Any]) -> Tuple[Any, ...]:
        def first_answer_id() -> Any:
            ans = row.get("answer_ids_sample")
            if isinstance(ans, list) and len(ans) >= 1:
                return ans[0]
            return None

        if name == "one_hop_chain":
            base = (row.get("anchor_id"), row.get("rel_type"))
            if simple:
                return (*base, first_answer_id())
            return base
        if name == "two_hop_chain":
            base = (row.get("anchor_id"), row.get("rel1"), row.get("rel2"))
            if simple:
                return (*base, first_answer_id())
            return base
        if name == "two_anchor_intersection":
            base = (
                row.get("anchorA_id"),
                row.get("anchorB_id"),
                row.get("anchorA_edge_type"),
                row.get("anchorB_edge_type"),
            )
            if simple:
                return (*base, first_answer_id())
            return base
        if name == "three_anchor_intersection":
            base = (
                row.get("anchorA_id"),
                row.get("anchorB_id"),
                row.get("anchorC_id"),
                row.get("anchorA_edge_type"),
                row.get("anchorB_edge_type"),
                row.get("anchorC_edge_type"),
            )
            if simple:
                return (*base, first_answer_id())
            return base
        return (json.dumps(row, sort_keys=True),)

    def resolve_node_label(session) -> str | None:
        raw = (node_label or "").strip()
        if raw == "":
            return None

        lowered = raw.lower()
        if lowered in {"none", "null", "all", "*"}:
            return None

        if lowered != "auto":
            return raw

        # Prefer the historical label when it exists, otherwise do not constrain.
        rec = session.run(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels"
        ).single()
        labels = set((rec and rec.get("labels")) or [])
        if "Entity" in labels:
            return "Entity"
        return None

    with driver.session() as session:
        resolved_label = resolve_node_label(session)

        for spec in specs:
            per_query_path = out_dir / f"{spec.name}.jsonl"
            if per_query_path.exists():
                per_query_path.unlink()

            seen: Set[Tuple[Any, ...]] = set()
            written = 0

            simple_keep_prob = 0.01
            simple_attempts = 0
            simple_no_progress = 0

            # Run multiple smaller transactions to avoid Neo4j transaction memory blowups.
            # For intersection queries, num_tries dominates memory/time; keep it moderate per batch.
            while written < limit_per_query:
                remaining = limit_per_query - written
                # Fetch more than needed (no DB-side ORDER BY rand), then randomize in Python.
                fetch_limit = min(max(500, remaining * 5), 5_000)

                if simple:
                    cypher = spec.cypher
                elif spec.name == "two_anchor_intersection":
                    cypher = build_two_anchor_intersection_cypher(
                        node_label=resolved_label,
                        anchor_pool=two_anchor_pool,
                        num_tries=min(two_anchor_tries, 50_000),
                        limit=min(fetch_limit, remaining),
                    )
                elif spec.name == "three_anchor_intersection":
                    cypher = build_three_anchor_intersection_cypher(
                        node_label=resolved_label,
                        anchor_pool=three_anchor_pool,
                        num_tries=min(three_anchor_tries, 100_000),
                        limit=min(fetch_limit, remaining),
                    )
                elif spec.name == "one_hop_chain":
                    cypher = build_one_hop_chain_cypher(
                        node_label=resolved_label,
                        anchor_pool=min(onehop_anchor_pool, 10_000),
                        limit=min(fetch_limit, remaining),
                    )
                elif spec.name == "two_hop_chain":
                    cypher = build_two_hop_chain_cypher(
                        node_label=resolved_label,
                        anchor_pool=min(twohop_anchor_pool, 10_000),
                        limit=min(fetch_limit, remaining),
                    )
                else:
                    cypher = spec.cypher

                if simple:
                    simple_attempts += 1
                    result = session.run(
                        cypher,
                        limit=fetch_limit,
                        keep_prob=float(simple_keep_prob),
                    )
                else:
                    result = session.run(cypher)
                batch = [record.data() for record in result]
                if not batch:
                    if not simple:
                        break
                    simple_keep_prob = min(1.0, simple_keep_prob * 2)
                    if simple_attempts >= 30:
                        break
                    continue

                random.shuffle(batch)

                new_rows: List[Dict[str, Any]] = []
                for row in batch:
                    k = key_for(spec.name, row)
                    if k in seen:
                        continue
                    seen.add(k)
                    new_rows.append(row)

                    # Don't accumulate huge lists in memory; write up to remaining.
                    if len(new_rows) >= remaining:
                        break

                if not new_rows:
                    # If we keep getting duplicates, stop to avoid infinite loops.
                    if not simple:
                        break
                    simple_no_progress += 1
                    simple_keep_prob = min(1.0, simple_keep_prob * 1.5)
                    if simple_no_progress >= 10 or simple_attempts >= 30:
                        break
                    continue

                _append_jsonl(per_query_path, new_rows)
                _append_jsonl(
                    combined_path, ({"query_type": spec.name, **r} for r in new_rows)
                )

                written += len(new_rows)

                if simple:
                    simple_no_progress = 0

            counts.append((spec.name, written))

    driver.close()

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "counts": [{"query_type": name, "rows": n} for name, n in counts],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return combined_path, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dataset construction Cypher queries against Neo4j and save results as JSONL."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/app/result") / "dataset_construction",
        help="Output directory (default: /app/result/dataset_construction)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for Python-side shuffling (default: 42)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use very simple fast queries (no heavy sampling). Good for quickly getting ~100 examples.",
    )
    parser.add_argument(
        "--node-label",
        type=str,
        default="auto",
        help=(
            "Node label constraint for heavy queries. "
            "Use 'auto' to use :Entity if it exists, else no label constraint. "
            "Use 'none' to force no label constraint, or pass a label name. (default: auto)"
        ),
    )
    parser.add_argument(
        "--limit-per-query",
        type=int,
        default=10_000,
        help="Target number of rows per query (default: 10000)",
    )
    parser.add_argument(
        "--onehop-anchor-pool",
        type=int,
        default=50_000,
        help="Anchor pool size for one-hop (default: 50000)",
    )
    parser.add_argument(
        "--twohop-anchor-pool",
        type=int,
        default=50_000,
        help="Anchor pool size for two-hop (default: 50000)",
    )
    parser.add_argument(
        "--two-anchor-pool",
        type=int,
        default=10_000,
        help="Candidate x pool size for 2-anchor intersection (default: 10000)",
    )
    parser.add_argument(
        "--two-anchor-tries",
        type=int,
        default=200_000,
        help="Random tries for 2-anchor intersection (default: 200000)",
    )
    parser.add_argument(
        "--three-anchor-pool",
        type=int,
        default=10_000,
        help="Candidate x pool size for 3-anchor intersection (default: 10000)",
    )
    parser.add_argument(
        "--three-anchor-tries",
        type=int,
        default=800_000,
        help="Random tries for 3-anchor intersection (default: 800000)",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=1000,
        help="(reserved) batch limit; generator uses internal safe batch sizes",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    combined_path, counts = run_queries_and_save(
        args.out_dir,
        node_label=args.node_label,
        limit_per_query=args.limit_per_query,
        onehop_anchor_pool=args.onehop_anchor_pool,
        twohop_anchor_pool=args.twohop_anchor_pool,
        two_anchor_pool=args.two_anchor_pool,
        two_anchor_tries=args.two_anchor_tries,
        three_anchor_pool=args.three_anchor_pool,
        three_anchor_tries=args.three_anchor_tries,
        seed=args.seed,
        simple=args.simple,
    )

    print(f"Wrote: {combined_path}")
    for name, n in counts:
        print(f"  {name}: {n} rows")


if __name__ == "__main__":
    main()
