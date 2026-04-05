"""
WebQSP Dataset Conversion Script

Converts the WebQSP (Web Questions Semantic Parses) dataset into our
evaluation-compatible JSONL format.

Supported input formats:
  1. Original Microsoft WebQSP (WebQSP.train.json, WebQSP.test.json)
     - Download from: https://www.microsoft.com/en-us/research/publication/
       the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/
  2. RoG-WebQSP (HuggingFace: rmanluo/RoG-webqsp)
     - Already has parsed relation chains

Output format (one JSON per line):
  {
    "question": "What country is Obama from?",
    "entity": "Barack Obama",
    "relation": "people.person.nationality",        # 1-hop only
    "relation1": "rel_a", "relation2": "rel_b",     # 2-hop only
    "answers": [{"name": "United States of America"}],
    "hop_count": 1,
    "has_constraint": false,
    "constraint_type": null,
    "inferential_chain": ["people.person.nationality"],
    "topic_mid": "m.02mjmr"
  }

Usage:
  python -m app.dataset_construction.convert_webqsp \\
    --input data/webqsp/WebQSP.test.json \\
    --output result/webqsp/test.jsonl

  # Or for RoG format:
  python -m app.dataset_construction.convert_webqsp \\
    --input data/webqsp/rog_test.jsonl --format rog \\
    --output result/webqsp/test.jsonl

  # Process both train and test:
  python -m app.dataset_construction.convert_webqsp \\
    --input data/webqsp/WebQSP.train.json data/webqsp/WebQSP.test.json \\
    --output result/webqsp/train.jsonl result/webqsp/test.jsonl

  # Show statistics only:
  python -m app.dataset_construction.convert_webqsp \\
    --input data/webqsp/WebQSP.test.json --stats-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
#  Known CVT types for chain collapsing
# ────────────────────────────────────────────────────────────────
CVT_RELATION_PREFIXES = {
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
#  Constraint detection
# ────────────────────────────────────────────────────────────────
TEMPORAL_KEYWORDS = [
    "first", "last", "latest", "earliest", "newest", "oldest",
    "before", "after", "during", "since", "until",
    "current", "now", "present", "recent",
    "year", "date", "when",
]

ORDINAL_KEYWORDS = [
    "first", "last", "second", "third", "most", "least",
    "largest", "smallest", "biggest", "highest", "lowest",
    "tallest", "shortest", "youngest", "oldest",
]


def detect_constraint_type(question: str) -> Optional[str]:
    """Detect if a question has temporal or ordinal constraints."""
    q_lower = question.lower()

    # Check ordinal first (more specific)
    ordinal_count = sum(1 for kw in ORDINAL_KEYWORDS if kw in q_lower)
    temporal_count = sum(1 for kw in TEMPORAL_KEYWORDS if kw in q_lower)

    if ordinal_count >= 2 or any(
        kw in q_lower for kw in ["most", "least", "largest", "smallest", "tallest"]
    ):
        return "ordinal"
    if temporal_count >= 2 or any(
        kw in q_lower for kw in ["before", "after", "during", "since", "until"]
    ):
        return "temporal"
    if ordinal_count >= 1:
        return "ordinal"
    if temporal_count >= 1:
        return "temporal"
    return None


# ────────────────────────────────────────────────────────────────
#  Parse Microsoft WebQSP format
# ────────────────────────────────────────────────────────────────
def _extract_chain_from_parse(parse: Dict[str, Any]) -> Tuple[List[str], bool]:
    """Extract the inferential chain from a WebQSP semantic parse.

    The parse contains an InferentialChain field with a list of Freebase
    relation paths. We also check for Constraints to flag constrained questions.

    Returns:
        Tuple of (chain, has_constraint)
    """
    chain = parse.get("InferentialChain") or []
    constraints = parse.get("Constraints") or []
    has_constraint = len(constraints) > 0
    return chain, has_constraint


def _get_topic_entity(parse: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract topic entity name and MID from a parse.

    Returns:
        Tuple of (entity_name, topic_mid)
    """
    topic_mid = parse.get("TopicEntityMid")
    topic_name = parse.get("TopicEntityName")

    # Sometimes the name is missing but PotentialTopicEntityMention has it
    if not topic_name:
        topic_name = parse.get("PotentialTopicEntityMention")

    return topic_name, topic_mid


def _get_answer_entities(parse: Dict[str, Any]) -> List[str]:
    """Extract answer entity names from a parse's Answers field."""
    answers = parse.get("Answers") or []
    result = []
    for ans in answers:
        # Each answer has AnswerType, AnswerArgument, EntityName
        name = ans.get("EntityName")
        if name and name != "(null)":
            result.append(name)
        elif ans.get("AnswerArgument"):
            # For literal answers (dates, numbers), use AnswerArgument
            result.append(str(ans["AnswerArgument"]))
    return result


def parse_microsoft_format(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse the official Microsoft WebQSP JSON format.

    The file structure is:
    {
      "Questions": [
        {
          "QuestionId": "WebQTest-0",
          "RawQuestion": "what is the name of ...",
          "Parses": [
            {
              "ParseId": "...",
              "InferentialChain": ["relation1", "relation2"],
              "Constraints": [...],
              "TopicEntityMid": "m.0xxxx",
              "TopicEntityName": "Entity Name",
              "Answers": [{"AnswerType": "Entity", "EntityName": "...", ...}]
            }
          ]
        }
      ]
    }
    """
    questions = data.get("Questions", [])
    results = []

    for q_data in questions:
        question = q_data.get("RawQuestion", "").strip()
        if not question:
            logger.warning("Skipping entry with empty question: %s", q_data.get("QuestionId"))
            continue

        parses = q_data.get("Parses", [])
        if not parses:
            logger.warning("Skipping question with no parses: %s", question)
            continue

        # Use the first parse (the gold parse)
        parse = parses[0]
        chain, has_constraint = _extract_chain_from_parse(parse)
        entity_name, topic_mid = _get_topic_entity(parse)
        answer_names = _get_answer_entities(parse)

        if not entity_name:
            logger.warning("Skipping question with no topic entity: %s", question)
            continue

        if not chain:
            logger.debug("Question has empty chain (no SPARQL parse): %s", question)
            # Still include it - some questions have answers but no chain
            # (e.g., when the parse couldn't be obtained)

        if not answer_names:
            logger.debug("Question has no answer entities: %s", question)

        hop_count = len(chain) if chain else 0
        constraint_type = detect_constraint_type(question)

        entry = {
            "question": question,
            "entity": entity_name,
            "answers": [{"name": name} for name in answer_names],
            "hop_count": hop_count,
            "has_constraint": has_constraint,
            "constraint_type": constraint_type,
            "inferential_chain": chain,
            "topic_mid": topic_mid,
            "question_id": q_data.get("QuestionId"),
        }

        # Add relation fields for compatibility with evaluation framework
        if hop_count == 1:
            entry["relation"] = chain[0]
        elif hop_count == 2:
            entry["relation1"] = chain[0]
            entry["relation2"] = chain[1]
        elif hop_count > 2:
            for i, rel in enumerate(chain):
                entry[f"relation{i + 1}"] = rel

        results.append(entry)

    return results


# ────────────────────────────────────────────────────────────────
#  Parse RoG-WebQSP format (HuggingFace: rmanluo/RoG-webqsp)
# ────────────────────────────────────────────────────────────────
def parse_rog_format(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse the RoG-WebQSP JSONL format.

    Each line is a JSON object with fields like:
    {
      "id": "WebQTest-0",
      "question": "...",
      "answer": ["entity1", "entity2"],
      "q_entity": ["m.0xxxx"],
      "a_entity": ["m.0yyyy"],
      "graph": [["head", "relation", "tail"], ...],
      "choices": [...]
    }
    """
    results = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON line")
            continue

        question = data.get("question", "").strip()
        if not question:
            continue

        answer_names = data.get("answer", [])
        if isinstance(answer_names, str):
            answer_names = [answer_names]

        # Extract entity name from graph triples or q_entity
        q_entities = data.get("q_entity", [])
        entity_name = q_entities[0] if q_entities else None

        # Try to extract relation chain from graph
        graph = data.get("graph", [])
        chain = _infer_chain_from_graph(graph, q_entities, answer_names)

        hop_count = len(chain) if chain else 0
        constraint_type = detect_constraint_type(question)

        entry = {
            "question": question,
            "entity": entity_name or "",
            "answers": [{"name": name} for name in answer_names],
            "hop_count": hop_count,
            "has_constraint": False,
            "constraint_type": constraint_type,
            "inferential_chain": chain,
            "topic_mid": q_entities[0] if q_entities else None,
            "question_id": data.get("id"),
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

    return results


def _infer_chain_from_graph(
    graph: List[List[str]],
    q_entities: List[str],
    answers: List[str],
) -> List[str]:
    """Try to infer the relation chain from the subgraph triples.

    This is a best-effort approach: find a path from a q_entity to an answer
    through the graph triples.
    """
    if not graph or not q_entities or not answers:
        return []

    # Build adjacency: entity -> [(relation, neighbor)]
    adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for triple in graph:
        if len(triple) != 3:
            continue
        h, r, t = triple
        adj[h].append((r, t))
        # Also add reverse
        adj[t].append((r + "_reverse", h))

    answer_set = set(answers)
    q_entity_set = set(q_entities)

    # BFS from q_entities to find shortest path to any answer
    from collections import deque

    for start in q_entities:
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            node, path = queue.popleft()

            if node in answer_set and path:
                return [r for r, _ in path]

            if len(path) >= 3:  # Max 3 hops
                continue

            for rel, neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(rel, neighbor)]))

    return []


# ────────────────────────────────────────────────────────────────
#  Detect input format
# ────────────────────────────────────────────────────────────────
def detect_format(filepath: Path) -> str:
    """Auto-detect input format based on file content."""
    with filepath.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    if not first_line:
        raise ValueError(f"Empty file: {filepath}")

    # If it starts with "{" and contains "Questions", it's Microsoft format
    try:
        data = json.loads(first_line)
        if "Questions" in data:
            return "microsoft"
        # Single JSON object on first line -> JSONL (likely RoG)
        if "question" in data:
            return "rog"
    except json.JSONDecodeError:
        pass

    # Try reading as full JSON
    with filepath.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if "Questions" in data:
                return "microsoft"
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot detect format of {filepath}")


# ────────────────────────────────────────────────────────────────
#  Main conversion
# ────────────────────────────────────────────────────────────────
def convert_file(filepath: Path, fmt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert a single WebQSP file to our format.

    Args:
        filepath: Path to input file
        fmt: Format override ("microsoft" or "rog"). Auto-detected if None.

    Returns:
        List of converted entries
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    if fmt is None:
        fmt = detect_format(filepath)

    logger.info("Parsing %s (format: %s)", filepath, fmt)

    if fmt == "microsoft":
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return parse_microsoft_format(data)
    elif fmt == "rog":
        with filepath.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        return parse_rog_format(lines)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def write_jsonl(entries: List[Dict[str, Any]], output_path: Path) -> None:
    """Write entries to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), output_path)


# ────────────────────────────────────────────────────────────────
#  Statistics
# ────────────────────────────────────────────────────────────────
def print_statistics(entries: List[Dict[str, Any]], label: str = "") -> None:
    """Print dataset statistics."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  Statistics: {label}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print(f"  WebQSP Dataset Statistics")
        print(f"{'=' * 60}")

    total = len(entries)
    print(f"\nTotal questions: {total}")

    # Hop count distribution
    hop_counter = Counter(e.get("hop_count", 0) for e in entries)
    print("\nHop count distribution:")
    for hop in sorted(hop_counter.keys()):
        count = hop_counter[hop]
        pct = 100.0 * count / total if total else 0
        label_str = f"  {hop}-hop" if hop > 0 else "  0-hop (no chain)"
        print(f"{label_str}: {count:>5} ({pct:5.1f}%)")

    # Unique relations
    all_relations = set()
    for e in entries:
        chain = e.get("inferential_chain", [])
        for rel in chain:
            all_relations.add(rel)
    print(f"\nUnique relations: {len(all_relations)}")
    if len(all_relations) <= 30:
        for rel in sorted(all_relations):
            print(f"  - {rel}")
    else:
        print("  (showing top 30 by frequency)")
        rel_counter = Counter()
        for e in entries:
            for rel in e.get("inferential_chain", []):
                rel_counter[rel] += 1
        for rel, count in rel_counter.most_common(30):
            print(f"  - {rel}: {count}")

    # Unique entity types (from Freebase relation domains)
    type_counter = Counter()
    for e in entries:
        chain = e.get("inferential_chain", [])
        for rel in chain:
            parts = rel.split(".")
            if len(parts) >= 2:
                domain = parts[0]
                entity_type = parts[1]
                type_counter[f"{domain}.{entity_type}"] += 1
    print(f"\nUnique entity types (from relation domains): {len(type_counter)}")
    if type_counter:
        print("  Top 20:")
        for t, count in type_counter.most_common(20):
            print(f"    - {t}: {count}")

    # Constraint analysis
    constrained = sum(1 for e in entries if e.get("has_constraint"))
    constraint_types = Counter(
        e.get("constraint_type") for e in entries if e.get("constraint_type")
    )
    print(f"\nQuestions with constraints (from parse): {constrained} ({100.0 * constrained / total:.1f}%)")
    print(f"Questions with detected constraint keywords:")
    for ct, count in constraint_types.most_common():
        print(f"  - {ct}: {count} ({100.0 * count / total:.1f}%)")

    # Answer count distribution
    ans_counts = [len(e.get("answers", [])) for e in entries]
    print(f"\nAnswer count statistics:")
    print(f"  Min answers: {min(ans_counts) if ans_counts else 0}")
    print(f"  Max answers: {max(ans_counts) if ans_counts else 0}")
    print(f"  Mean answers: {sum(ans_counts) / len(ans_counts):.1f}" if ans_counts else "  N/A")
    no_answer = sum(1 for c in ans_counts if c == 0)
    print(f"  Questions with no answer: {no_answer}")

    print()


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WebQSP dataset to evaluation JSONL format"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input file(s): WebQSP.train.json, WebQSP.test.json, or RoG JSONL",
    )
    parser.add_argument(
        "--output", "-o",
        nargs="*",
        default=None,
        help="Output JSONL file(s). If not specified, defaults to result/webqsp/<split>.jsonl",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["microsoft", "rog", "auto"],
        default="auto",
        help="Input format (default: auto-detect)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, do not write output",
    )
    parser.add_argument(
        "--filter-hops",
        type=int,
        nargs="+",
        default=None,
        help="Only include questions with specified hop counts (e.g., --filter-hops 1 2)",
    )
    parser.add_argument(
        "--exclude-no-chain",
        action="store_true",
        help="Exclude questions with no inferential chain",
    )
    parser.add_argument(
        "--exclude-no-answer",
        action="store_true",
        help="Exclude questions with no answer entities",
    )
    parser.add_argument(
        "--split-by-hops",
        action="store_true",
        help="Split output into separate files by hop count (1hop.jsonl, 2hop.jsonl, ...)",
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

    fmt = None if args.format == "auto" else args.format
    all_entries: List[Dict[str, Any]] = []

    input_paths = [Path(p) for p in args.input]

    # Determine output paths
    if args.output:
        output_paths = [Path(p) for p in args.output]
        if len(output_paths) != len(input_paths):
            parser.error(
                f"Number of outputs ({len(output_paths)}) must match "
                f"number of inputs ({len(input_paths)})"
            )
    else:
        # Default output paths
        output_paths = []
        for inp in input_paths:
            stem = inp.stem.lower().replace("webqsp.", "").replace("webqsp_", "")
            if not stem or stem == inp.stem.lower():
                stem = inp.stem.lower()
            output_paths.append(Path(f"result/webqsp/{stem}.jsonl"))

    # Process each input file
    for i, input_path in enumerate(input_paths):
        try:
            entries = convert_file(input_path, fmt)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue
        except Exception as e:
            logger.error("Error processing %s: %s", input_path, e)
            continue

        # Apply filters
        if args.exclude_no_chain:
            before = len(entries)
            entries = [e for e in entries if e.get("inferential_chain")]
            logger.info("Filtered no-chain: %d -> %d", before, len(entries))

        if args.exclude_no_answer:
            before = len(entries)
            entries = [e for e in entries if e.get("answers")]
            logger.info("Filtered no-answer: %d -> %d", before, len(entries))

        if args.filter_hops:
            before = len(entries)
            hop_set = set(args.filter_hops)
            entries = [e for e in entries if e.get("hop_count") in hop_set]
            logger.info("Filtered by hops %s: %d -> %d", args.filter_hops, before, len(entries))

        # Print statistics
        split_label = input_path.stem
        print_statistics(entries, label=split_label)

        all_entries.extend(entries)

        # Write output
        if not args.stats_only:
            if args.split_by_hops:
                # Split by hop count
                by_hop: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
                for entry in entries:
                    by_hop[entry.get("hop_count", 0)].append(entry)

                base_dir = output_paths[i].parent
                for hop, hop_entries in sorted(by_hop.items()):
                    hop_path = base_dir / f"{hop}hop.jsonl"
                    write_jsonl(hop_entries, hop_path)
            else:
                write_jsonl(entries, output_paths[i])

    # Print combined statistics if multiple files
    if len(input_paths) > 1 and all_entries:
        print_statistics(all_entries, label="COMBINED")


if __name__ == "__main__":
    main()
