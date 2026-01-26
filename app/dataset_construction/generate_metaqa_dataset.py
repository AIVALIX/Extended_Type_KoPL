"""
MetaQA用データセット生成スクリプト

既存のMetaQA QAデータから評価用データセットを生成する
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from py2neo import Graph


def get_graph() -> Graph:
    """MetaQA Neo4j接続"""
    uri = os.getenv("NEO4J_URI", "bolt://neo4j_metaqa:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return Graph(uri, auth=(user, password))


def extract_entity_from_question(question: str) -> Optional[str]:
    """質問文から[entity]形式のエンティティを抽出"""
    match = re.search(r'\[([^\]]+)\]', question)
    if match:
        return match.group(1)
    return None


def find_relation_for_1hop(graph: Graph, entity: str, answers: List[str]) -> Optional[str]:
    """1-hopの質問に対するリレーションを特定"""
    query = """
    MATCH (e)-[r]->(a)
    WHERE e.name = $entity AND a.name IN $answers
    RETURN DISTINCT type(r) AS relation
    LIMIT 1
    """
    result = graph.run(query, entity=entity, answers=answers).data()
    if result:
        return result[0]["relation"]

    # 逆方向も試す
    query_rev = """
    MATCH (e)<-[r]-(a)
    WHERE e.name = $entity AND a.name IN $answers
    RETURN DISTINCT type(r) AS relation
    LIMIT 1
    """
    result = graph.run(query_rev, entity=entity, answers=answers).data()
    if result:
        return result[0]["relation"]

    return None


def find_relations_for_2hop(graph: Graph, entity: str, answers: List[str]) -> Optional[Tuple[str, str]]:
    """2-hopの質問に対するリレーションペアを特定"""
    query = """
    MATCH (e)-[r1]->(mid)-[r2]->(a)
    WHERE e.name = $entity AND a.name IN $answers
    RETURN DISTINCT type(r1) AS rel1, type(r2) AS rel2
    LIMIT 1
    """
    result = graph.run(query, entity=entity, answers=answers).data()
    if result:
        return result[0]["rel1"], result[0]["rel2"]

    # 他の方向も試す
    query_patterns = [
        "MATCH (e)-[r1]->(mid)<-[r2]-(a) WHERE e.name = $entity AND a.name IN $answers RETURN DISTINCT type(r1) AS rel1, type(r2) AS rel2 LIMIT 1",
        "MATCH (e)<-[r1]-(mid)-[r2]->(a) WHERE e.name = $entity AND a.name IN $answers RETURN DISTINCT type(r1) AS rel1, type(r2) AS rel2 LIMIT 1",
        "MATCH (e)<-[r1]-(mid)<-[r2]-(a) WHERE e.name = $entity AND a.name IN $answers RETURN DISTINCT type(r1) AS rel1, type(r2) AS rel2 LIMIT 1",
    ]
    for q in query_patterns:
        result = graph.run(q, entity=entity, answers=answers).data()
        if result:
            return result[0]["rel1"], result[0]["rel2"]

    return None


def infer_relations_from_question(question: str, num_hops: int) -> Optional[List[str]]:
    """質問文からリレーションを推論する"""
    q = question.lower()

    # 3-hop patterns: (share_rel, share_rel, final_rel)
    if num_hops == 3:
        # Pattern: "share actors/directors/writers" -> same relation twice, then final
        share_rel = None
        if "share actor" in q or "actors also appear" in q or "starred by" in q or "acted by the actors" in q:
            share_rel = "STARRED_ACTORS"
        elif "share director" in q or "directors also directed" in q or "directed by the director" in q:
            share_rel = "DIRECTED_BY"
        elif "share writer" in q or "writers also wrote" in q or "written by the writer" in q or "screenwriters also wrote" in q:
            share_rel = "WRITTEN_BY"

        # Final relation
        final_rel = None
        if "language" in q:
            final_rel = "IN_LANGUAGE"
        elif "genre" in q or "types are" in q:
            final_rel = "HAS_GENRE"
        elif "release" in q or "when did" in q:
            final_rel = "RELEASE_YEAR"
        elif "who starred" in q or "who acted" in q or "starred who" in q or "actors in" in q:
            final_rel = "STARRED_ACTORS"
        elif "who directed" in q or "directed by who" in q or "director of" in q or "directors of" in q:
            final_rel = "DIRECTED_BY"
        elif "who wrote" in q or "writer" in q or "screenwriter" in q:
            final_rel = "WRITTEN_BY"

        if share_rel and final_rel:
            return [share_rel, share_rel, final_rel]

        # Alternative patterns without "share"
        # "films written by the writer of X starred who" -> WRITTEN_BY, WRITTEN_BY, STARRED_ACTORS
        # "who directed films for the writer of X" -> WRITTEN_BY, WRITTEN_BY, DIRECTED_BY
        if "writer of" in q or "written by" in q:
            first_rel = "WRITTEN_BY"
        elif "director of" in q or "directed by" in q:
            first_rel = "DIRECTED_BY"
        elif "actor" in q or "starred" in q:
            first_rel = "STARRED_ACTORS"
        else:
            first_rel = None

        if first_rel and final_rel:
            return [first_rel, first_rel, final_rel]

    return None


def find_relations_for_3hop(graph: Graph, entity: str, answers: List[str], question: str = "") -> Optional[Tuple[str, str, str]]:
    """3-hopの質問に対するリレーショントリプルを特定"""

    # まず質問文からリレーションを推論
    inferred = infer_relations_from_question(question, 3)
    if inferred:
        # 推論したリレーションでパスが存在するか確認
        rel1, rel2, rel3 = inferred
        verify_query = """
        MATCH (e)-[r1:%s]-(m1)-[r2:%s]-(m2)-[r3:%s]-(a)
        WHERE e.name = $entity AND a.name IN $answers AND e <> a AND e <> m1 AND e <> m2
        RETURN count(*) AS cnt
        LIMIT 1
        """ % (rel1, rel2, rel3)
        result = graph.run(verify_query, entity=entity, answers=answers).data()
        if result and result[0]["cnt"] > 0:
            return tuple(inferred)

    # フォールバック: 任意のパスを検索
    query = """
    MATCH (e)-[r1]-(m1)-[r2]-(m2)-[r3]-(a)
    WHERE e.name = $entity AND a.name IN $answers AND e <> a AND e <> m1 AND e <> m2
    RETURN DISTINCT type(r1) AS rel1, type(r2) AS rel2, type(r3) AS rel3
    LIMIT 1
    """
    result = graph.run(query, entity=entity, answers=answers).data()
    if result:
        return result[0]["rel1"], result[0]["rel2"], result[0]["rel3"]

    return None


def generate_1hop_dataset(graph: Graph, source_path: Path, output_path: Path, limit: int = 500):
    """1-hopデータセットを生成"""
    samples = []
    skipped = 0

    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= limit:
                break

            data = json.loads(line.strip())
            question = data["question"]
            answers = data["answer"]

            entity = extract_entity_from_question(question)
            if not entity:
                skipped += 1
                continue

            relation = find_relation_for_1hop(graph, entity, answers)
            if not relation:
                skipped += 1
                continue

            sample = {
                "question": question.replace(f"[{entity}]", entity),
                "entity": entity,
                "relation": relation,
                "answers": [{"name": a} for a in answers],
            }
            samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"1-hop: {len(samples)} samples generated, {skipped} skipped")
    return samples


def generate_2hop_dataset(graph: Graph, source_path: Path, output_path: Path, limit: int = 500):
    """2-hopデータセットを生成"""
    samples = []
    skipped = 0

    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= limit:
                break

            data = json.loads(line.strip())
            question = data["question"]
            answers = data["answer"]

            entity = extract_entity_from_question(question)
            if not entity:
                skipped += 1
                continue

            relations = find_relations_for_2hop(graph, entity, answers)
            if not relations:
                skipped += 1
                continue

            sample = {
                "question": question.replace(f"[{entity}]", entity),
                "entity": entity,
                "relation1": relations[0],
                "relation2": relations[1],
                "answers": [{"name": a} for a in answers],
            }
            samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"2-hop: {len(samples)} samples generated, {skipped} skipped")
    return samples


def generate_3hop_dataset(graph: Graph, source_path: Path, output_path: Path, limit: int = 500):
    """3-hopデータセットを生成"""
    samples = []
    skipped = 0

    with source_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= limit:
                break

            data = json.loads(line.strip())
            question = data["question"]
            answers = data["answer"]

            entity = extract_entity_from_question(question)
            if not entity:
                skipped += 1
                continue

            relations = find_relations_for_3hop(graph, entity, answers, question)
            if not relations:
                skipped += 1
                continue

            sample = {
                "question": question.replace(f"[{entity}]", entity),
                "entity": entity,
                "relation1": relations[0],
                "relation2": relations[1],
                "relation3": relations[2],
                "answers": [{"name": a} for a in answers],
            }
            samples.append(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"3-hop: {len(samples)} samples generated, {skipped} skipped")
    return samples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate MetaQA evaluation datasets")
    parser.add_argument("--limit", type=int, default=500, help="Max samples per dataset")
    parser.add_argument("--source-dir", type=Path, default=Path("data/metaqa/qa"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/metaqa"))
    args = parser.parse_args()

    graph = get_graph()

    # Test connection
    result = graph.run("MATCH (n) RETURN count(n) AS count").data()
    print(f"Connected to MetaQA Neo4j: {result[0]['count']} nodes")

    # Generate datasets
    generate_1hop_dataset(
        graph,
        args.source_dir / "hop1.jsonl",
        args.output_dir / "1hop.jsonl",
        args.limit,
    )

    generate_2hop_dataset(
        graph,
        args.source_dir / "hop2.jsonl",
        args.output_dir / "2hop.jsonl",
        args.limit,
    )

    generate_3hop_dataset(
        graph,
        args.source_dir / "hop3.jsonl",
        args.output_dir / "3hop.jsonl",
        args.limit,
    )

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
