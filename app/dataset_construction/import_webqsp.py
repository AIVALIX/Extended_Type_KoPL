"""
WebQSP Freebase Subgraph Import Script

Imports the Freebase subgraph used by WebQSP into Neo4j.

Steps:
  1. Download WebQSP dataset and the Freebase dump
  2. Extract the subgraph relevant to WebQSP questions (2-hop neighborhood)
  3. Collapse CVT (Compound Value Type) nodes into direct edges
  4. Import nodes and edges into Neo4j (neo4j_webqsp service)

Prerequisites:
  - Freebase dump: Follow https://github.com/dki-lab/Freebase-Setup to set up
    a Virtuoso instance with the full Freebase dump, or use the pre-extracted
    subgraph from the GrailQA / WebQSP resources.
  - WebQSP dataset: Download from
    https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling/

References:
  - dki-lab/Freebase-Setup: https://github.com/dki-lab/Freebase-Setup
  - WebQSP paper: Yih et al., "The Value of Semantic Parse Labeling for KBQA" (ACL 2016)
  - Freebase CVT docs: https://developers.google.com/freebase/guide/basic_concepts

Usage:
  docker exec -it python-primekgqa-experiment python -m dataset_construction.import_webqsp
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────────

# TODO: Update these paths after downloading the data
WEBQSP_DATA_DIR = Path("data/webqsp")
WEBQSP_TRAIN_PATH = WEBQSP_DATA_DIR / "WebQSP.train.json"
WEBQSP_TEST_PATH = WEBQSP_DATA_DIR / "WebQSP.test.json"
FREEBASE_DUMP_DIR = WEBQSP_DATA_DIR / "freebase"

# Known CVT types in Freebase that should be collapsed
# TODO: Extend this list based on actual data analysis
CVT_TYPES: Set[str] = {
    "film.performance",
    "sports.sports_team_roster",
    "education.education",
    "people.marriage",
    "people.sibling_relationship",
    "government.government_position_held",
    "business.employment_tenure",
    "music.group_membership",
    "award.award_honor",
    "organization.organization_board_membership",
    "tv.regular_tv_appearance",
    "fictional_universe.fictional_character_portrayed_in",
}


# ────────────────────────────────────────────────────────────────
#  Step 1: Download WebQSP dataset
# ────────────────────────────────────────────────────────────────
def download_webqsp(output_dir: Path = WEBQSP_DATA_DIR) -> None:
    """Download the WebQSP dataset files.

    TODO: Implement download logic.
    - Download WebQSP.train.json and WebQSP.test.json
    - Extract entity MIDs and topic entity MIDs from SPARQL parses
    - Save to output_dir

    Args:
        output_dir: Directory to save downloaded files
    """
    # TODO: Implement
    raise NotImplementedError(
        "Download WebQSP data manually from "
        "https://www.microsoft.com/en-us/research/publication/"
        "the-value-of-semantic-parse-labeling/ "
        f"and place files in {output_dir}"
    )


# ────────────────────────────────────────────────────────────────
#  Step 2: Extract Freebase subgraph
# ────────────────────────────────────────────────────────────────
def extract_subgraph(
    webqsp_path: Path = WEBQSP_TEST_PATH,
    freebase_dir: Path = FREEBASE_DUMP_DIR,
    max_hops: int = 2,
) -> Tuple[List[Dict], List[Dict]]:
    """Extract the Freebase subgraph relevant to WebQSP questions.

    For each topic entity in WebQSP, extract all triples within max_hops
    of the entity from the Freebase dump or Virtuoso SPARQL endpoint.

    TODO: Implement subgraph extraction.
    - Parse WebQSP JSON to collect all topic entity MIDs
    - Query Freebase (via dump or SPARQL) for 2-hop neighborhoods
    - Deduplicate nodes and edges
    - Return as lists of node/edge dicts

    Args:
        webqsp_path: Path to WebQSP JSON file
        freebase_dir: Path to Freebase dump directory
        max_hops: Maximum hops from topic entities to include

    Returns:
        Tuple of (nodes, edges) where:
          nodes: List of {"mid": str, "name": str, "types": List[str]}
          edges: List of {"src": str, "rel": str, "tgt": str}
    """
    # TODO: Implement
    raise NotImplementedError(
        "Subgraph extraction requires a Freebase dump or Virtuoso endpoint. "
        "See https://github.com/dki-lab/Freebase-Setup for setup instructions."
    )


# ────────────────────────────────────────────────────────────────
#  Step 3: Collapse CVT nodes
# ────────────────────────────────────────────────────────────────
def collapse_cvt_nodes(
    nodes: List[Dict],
    edges: List[Dict],
    cvt_types: Set[str] = CVT_TYPES,
) -> Tuple[List[Dict], List[Dict]]:
    """Collapse CVT (Compound Value Type) nodes into direct edges.

    Freebase uses CVT nodes as n-ary relation mediators. For example:
      (Film)-[film.film.starring]->(film.performance)-[film.performance.actor]->(Person)
    becomes:
      (Film)-[film.film.starring..film.performance.actor]->(Person)

    The collapsed relation name is formed by concatenating the two relation
    names with ".." as separator.

    TODO: Implement CVT collapse logic.
    - Identify CVT nodes by checking if any of their types is in cvt_types
    - For each CVT node, find incoming and outgoing edges
    - Create direct edges between non-CVT neighbors
    - Remove CVT nodes and their original edges
    - Handle chains of CVT nodes (rare but possible)

    Args:
        nodes: List of node dicts with "mid", "name", "types" keys
        edges: List of edge dicts with "src", "rel", "tgt" keys
        cvt_types: Set of Freebase type IDs that are CVT types

    Returns:
        Tuple of (filtered_nodes, collapsed_edges)
    """
    # TODO: Implement
    raise NotImplementedError(
        "CVT collapse logic not yet implemented. "
        "See schema_webqsp.py for CVT handling strategy documentation."
    )


# ────────────────────────────────────────────────────────────────
#  Step 4: Import to Neo4j
# ────────────────────────────────────────────────────────────────
def import_to_neo4j(
    nodes: List[Dict],
    edges: List[Dict],
    neo4j_uri: Optional[str] = None,
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    batch_size: int = 1000,
) -> None:
    """Import nodes and edges into Neo4j (neo4j_webqsp service).

    TODO: Implement Neo4j import.
    - Connect to neo4j_webqsp using py2neo
    - Create uniqueness constraints on Entity(name) and Entity(mid)
    - Batch-create nodes with labels based on Freebase types
    - Batch-create relationships
    - Create indexes for efficient querying

    Args:
        nodes: List of node dicts with "mid", "name", "types" keys
        edges: List of edge dicts with "src", "rel", "tgt" keys
        neo4j_uri: Neo4j bolt URI (default: from NEO4J_WEBQSP_URI env var)
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        batch_size: Number of items per batch transaction
    """
    # TODO: Implement
    import os

    if neo4j_uri is None:
        neo4j_uri = os.getenv("NEO4J_WEBQSP_URI", "bolt://neo4j_webqsp:7687")

    raise NotImplementedError(
        "Neo4j import not yet implemented. "
        "Follow the pattern in dataset_construction/ for other KG imports."
    )


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    """Run the full WebQSP import pipeline.

    TODO: Implement the full pipeline:
    1. download_webqsp()
    2. nodes, edges = extract_subgraph()
    3. nodes, edges = collapse_cvt_nodes(nodes, edges)
    4. import_to_neo4j(nodes, edges)
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("WebQSP import pipeline - not yet implemented")
    logger.info("Steps to complete:")
    logger.info("  1. Download WebQSP dataset to %s", WEBQSP_DATA_DIR)
    logger.info("  2. Set up Freebase dump (see dki-lab/Freebase-Setup)")
    logger.info("  3. Extract subgraph for WebQSP topic entities")
    logger.info("  4. Collapse CVT nodes into direct edges")
    logger.info("  5. Import into Neo4j (neo4j_webqsp service)")


if __name__ == "__main__":
    main()
