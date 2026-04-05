"""
Build WebQSP Neo4j import CSVs from RoG-WebQSP parquet files.

Reads the RoG-WebQSP dataset (HuggingFace: rmanluo/RoG-webqsp) and extracts
all graph triples across all splits to produce Neo4j-compatible CSV files.

Key decisions:
  - CVT nodes (Freebase MIDs like "m.0xxxx") are KEPT as nodes.
    They are needed because many answer paths traverse through CVTs.
    The pipeline can handle them via multi-hop traversal.
  - Entity types are derived from `common.topic.notable_types` relations
    in the subgraph. Entities without a notable_type get label "Entity".
  - Relations like rdf-schema#domain, rdf-schema#range, type.*, freebase.*
    are filtered out as they are ontology metadata, not KG edges.
  - Relation names use Freebase dot notation (e.g., "people.person.nationality").
    These are uppercased and dots replaced with underscores for Neo4j rel types.

Output files:
  - data/webqsp/import_nodes.csv  (id:ID, name, :LABEL)
  - data/webqsp/import_rels.csv   (:START_ID, :END_ID, :TYPE)

Usage:
  python -m app.dataset_construction.build_webqsp_neo4j

  # Or with custom paths:
  python -m app.dataset_construction.build_webqsp_neo4j \
    --parquet-dir data/webqsp \
    --output-dir data/webqsp
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────────

# Relations that are ontology/schema metadata, not KG edges
EXCLUDE_RELATION_PREFIXES = {
    "rdf-schema#",
    "type.property.",
    "type.type.",
    "type.object.",
    "freebase.type_hints.",
    "freebase.type_profile.",
    "freebase.valuenotation.",
    "freebase.property_hints.",
    "freebase.documented_object.",
    "freebase.domain_profile.",
    "freebase.domain_category.",
    "freebase.equivalent_topic.",
    "freebase.user_profile.",
    "freebase.account.",
    "freebase.review.",
}

# The relation used to assign entity types
NOTABLE_TYPE_RELATION = "common.topic.notable_types"
# Fallback: also check these relations for type hints
TYPE_HINT_RELATIONS = {
    "common.topic.notable_types",
    "common.topic.notable_for",
}

# Sanitize label: remove spaces, slashes, special chars for Neo4j labels
def sanitize_label(label: str) -> str:
    """Convert a Freebase type name to a valid Neo4j label."""
    # Replace / and spaces with underscores
    label = label.replace("/", "_").replace(" ", "_")
    # Remove special characters except underscores
    label = re.sub(r"[^a-zA-Z0-9_]", "", label)
    # Ensure it doesn't start with a number
    if label and label[0].isdigit():
        label = "_" + label
    return label or "Entity"


def sanitize_rel_type(rel: str) -> str:
    """Convert a Freebase relation to a valid Neo4j relationship type.

    Keeps the original dot notation as the :TYPE value (like MetaQA uses
    DIRECTED_BY, STARRED_ACTORS, etc.). We uppercase and replace dots
    with double underscores to preserve the hierarchy.
    """
    # Keep the original Freebase relation name - the pipeline uses these
    # directly for matching, so we preserve them as-is.
    return rel


def is_cvt_node(name: str) -> bool:
    """Check if an entity name looks like a Freebase CVT/MID node."""
    return bool(re.match(r"^m\.[0-9a-z_]+$", name))


# ────────────────────────────────────────────────────────────────
#  Data extraction
# ────────────────────────────────────────────────────────────────

def should_exclude_relation(rel: str) -> bool:
    """Check if a relation should be excluded (schema/ontology metadata)."""
    for prefix in EXCLUDE_RELATION_PREFIXES:
        if rel.startswith(prefix):
            return True
    return False


def extract_all_triples(
    parquet_files: List[Path],
) -> Tuple[Set[Tuple[str, str, str]], Dict[str, str]]:
    """Extract all unique triples and entity types from parquet files.

    Returns:
        Tuple of:
          - set of (head, relation, tail) triples
          - dict of entity_name -> type_label
    """
    all_triples: Set[Tuple[str, str, str]] = set()
    entity_types: Dict[str, str] = {}
    total_raw = 0
    excluded_count = 0

    for parquet_file in parquet_files:
        logger.info("Reading %s ...", parquet_file.name)
        df = pd.read_parquet(parquet_file)

        for idx in range(len(df)):
            graph = df.iloc[idx]["graph"]
            if graph is None:
                continue

            for triple in graph:
                if len(triple) != 3:
                    continue
                h, r, t = str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip()
                total_raw += 1

                # Extract type information before filtering
                if r == NOTABLE_TYPE_RELATION:
                    entity_types[h] = t
                    # Don't add this as a graph edge - it's metadata
                    continue

                if r in TYPE_HINT_RELATIONS:
                    if h not in entity_types:
                        entity_types[h] = t
                    continue

                # Filter out schema/ontology relations
                if should_exclude_relation(r):
                    excluded_count += 1
                    continue

                all_triples.add((h, r, t))

    logger.info(
        "Raw triples: %d, excluded: %d, kept: %d",
        total_raw, excluded_count, len(all_triples),
    )
    logger.info("Entity types discovered: %d", len(entity_types))

    return all_triples, entity_types


def build_node_and_edge_lists(
    triples: Set[Tuple[str, str, str]],
    entity_types: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Build deduplicated node and edge lists for Neo4j import.

    Returns:
        Tuple of (nodes, edges, stats)
    """
    # Collect all unique entities
    entities: Set[str] = set()
    for h, r, t in triples:
        entities.add(h)
        entities.add(t)

    # Assign labels
    label_counter = Counter()
    nodes = []
    for entity in sorted(entities):
        if is_cvt_node(entity):
            label = "CVT"
        elif entity in entity_types:
            raw_label = entity_types[entity]
            label = sanitize_label(raw_label)
        else:
            label = "Entity"

        label_counter[label] += 1
        # Node ID format: Label:Name (following MetaQA pattern)
        node_id = f"{label}:{entity}"
        nodes.append({
            "id": node_id,
            "name": entity,
            "label": label,
        })

    # Build edge list
    rel_counter = Counter()
    edges = []

    # Build entity -> node_id lookup
    entity_to_id: Dict[str, str] = {}
    for node in nodes:
        entity_to_id[node["name"]] = node["id"]

    for h, r, t in sorted(triples):
        start_id = entity_to_id[h]
        end_id = entity_to_id[t]
        rel_type = sanitize_rel_type(r)
        rel_counter[rel_type] += 1
        edges.append({
            "start_id": start_id,
            "end_id": end_id,
            "rel_type": rel_type,
        })

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_relations": len(rel_counter),
        "total_entity_types": len(label_counter),
        "label_distribution": label_counter,
        "top_relations": rel_counter.most_common(30),
        "cvt_nodes": label_counter.get("CVT", 0),
        "named_nodes": len(nodes) - label_counter.get("CVT", 0),
    }

    return nodes, edges, stats


# ────────────────────────────────────────────────────────────────
#  CSV export
# ────────────────────────────────────────────────────────────────

def write_nodes_csv(nodes: List[Dict], output_path: Path) -> None:
    """Write nodes to Neo4j import CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id:ID", "name", ":LABEL"])
        for node in nodes:
            writer.writerow([node["id"], node["name"], node["label"]])
    logger.info("Wrote %d nodes to %s", len(nodes), output_path)


def write_rels_csv(edges: List[Dict], output_path: Path) -> None:
    """Write relationships to Neo4j import CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID", ":END_ID", ":TYPE"])
        for edge in edges:
            writer.writerow([edge["start_id"], edge["end_id"], edge["rel_type"]])
    logger.info("Wrote %d edges to %s", len(edges), output_path)


def print_stats(stats: Dict) -> None:
    """Print import statistics."""
    print(f"\n{'=' * 60}")
    print(f"  WebQSP Neo4j Import Statistics")
    print(f"{'=' * 60}")
    print(f"\nTotal nodes:          {stats['total_nodes']:>10,}")
    print(f"  Named entities:     {stats['named_nodes']:>10,}")
    print(f"  CVT nodes:          {stats['cvt_nodes']:>10,}")
    print(f"Total edges:          {stats['total_edges']:>10,}")
    print(f"Unique relation types:{stats['total_relations']:>10,}")
    print(f"Unique entity labels: {stats['total_entity_types']:>10,}")

    print(f"\nEntity label distribution (top 30):")
    for label, count in stats["label_distribution"].most_common(30):
        print(f"  {label:<30} {count:>8,}")

    print(f"\nTop 30 relation types:")
    for rel, count in stats["top_relations"]:
        print(f"  {rel:<55} {count:>8,}")
    print()


# ────────────────────────────────────────────────────────────────
#  Evaluation JSONL conversion
# ────────────────────────────────────────────────────────────────

def convert_to_eval_jsonl(
    parquet_files: List[Path],
    output_path: Path,
    entity_types: Dict[str, str],
    split: str = "test",
) -> None:
    """Convert a split to evaluation JSONL format.

    Uses the existing convert_webqsp.py logic adapted for parquet input.
    """
    import json
    from collections import deque

    results = []
    for parquet_file in parquet_files:
        if split not in parquet_file.name:
            continue
        logger.info("Converting %s to eval JSONL ...", parquet_file.name)
        df = pd.read_parquet(parquet_file)

        for idx in range(len(df)):
            row = df.iloc[idx]
            question = str(row["question"]).strip()
            if not question:
                continue

            answer_names = list(row["answer"]) if row["answer"] is not None else []
            q_entities = list(row["q_entity"]) if row["q_entity"] is not None else []
            graph = row["graph"]

            # Resolve q_entity: use name (RoG-WebQSP already uses names)
            entity_name = q_entities[0] if q_entities else ""

            # Infer relation chain via BFS through graph
            chain = _infer_chain(graph, q_entities, answer_names)

            hop_count = len(chain) if chain else 0

            entry = {
                "question": question,
                "entity": entity_name,
                "answers": [{"name": name} for name in answer_names],
                "hop_count": hop_count,
                "has_constraint": False,
                "constraint_type": None,
                "inferential_chain": chain,
                "topic_mid": None,
                "question_id": str(row.get("id", "")),
            }

            if hop_count == 1:
                entry["relation"] = chain[0]
            elif hop_count == 2:
                entry["relation1"] = chain[0]
                entry["relation2"] = chain[1]
            elif hop_count > 2:
                for i, rel in enumerate(chain):
                    entry[f"relation{i + 1}"] = rel

            results.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(results), output_path)


def _infer_chain(graph, q_entities, answers):
    """BFS to find shortest relation path from q_entity to answer."""
    from collections import deque

    if graph is None or len(graph) == 0 or not q_entities or not answers:
        return []

    adj = defaultdict(list)
    for triple in graph:
        if len(triple) != 3:
            continue
        h, r, t = str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip()
        if should_exclude_relation(r):
            continue
        if r == NOTABLE_TYPE_RELATION or r in TYPE_HINT_RELATIONS:
            continue
        adj[h].append((r, t))

    answer_set = set(str(a).strip() for a in answers)

    for start in q_entities:
        start = str(start)
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            node, path = queue.popleft()
            if node in answer_set and path:
                return [r for r, _ in path]
            if len(path) >= 3:
                continue
            for rel, neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(rel, neighbor)]))

    return []


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build WebQSP Neo4j import CSVs from RoG-WebQSP parquet files"
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("data/webqsp"),
        help="Directory containing downloaded parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/webqsp"),
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=Path("result/webqsp"),
        help="Directory for evaluation JSONL files",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation JSONL generation",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, do not write files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Find parquet files
    parquet_files = sorted(args.parquet_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found in %s", args.parquet_dir)
        sys.exit(1)

    logger.info("Found %d parquet files in %s", len(parquet_files), args.parquet_dir)

    # Step 1: Extract all triples and entity types
    logger.info("Extracting triples from all splits ...")
    all_triples, entity_types = extract_all_triples(parquet_files)

    # Step 2: Build node and edge lists
    logger.info("Building node and edge lists ...")
    nodes, edges, stats = build_node_and_edge_lists(all_triples, entity_types)

    # Print statistics
    print_stats(stats)

    if args.stats_only:
        return

    # Step 3: Write CSVs
    nodes_path = args.output_dir / "import_nodes.csv"
    rels_path = args.output_dir / "import_rels.csv"
    write_nodes_csv(nodes, nodes_path)
    write_rels_csv(edges, rels_path)

    # Step 4: Generate evaluation JSONL files
    if not args.skip_eval:
        for split in ["train", "test", "validation"]:
            split_files = [f for f in parquet_files if split in f.name]
            if split_files:
                output_path = args.eval_output_dir / f"{split}.jsonl"
                convert_to_eval_jsonl(split_files, output_path, entity_types, split)

    logger.info("Done!")
    logger.info("Neo4j import files:")
    logger.info("  Nodes: %s", nodes_path)
    logger.info("  Rels:  %s", rels_path)


if __name__ == "__main__":
    main()
