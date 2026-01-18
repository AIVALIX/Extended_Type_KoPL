"""
PrimeKG v2 Dataset Generator
スキーマベースのデータ生成（Answer-first approach）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from pydantic import BaseModel, Field
from tqdm import tqdm

from dataset_construction.schema_v2 import (
    ONE_HOP_TEMPLATES,
    TWO_HOP_TEMPLATES,
    TWO_ANCHOR_INTERSECTION_TEMPLATES,
    THREE_ANCHOR_INTERSECTION_TEMPLATES,
    get_cypher_label,
)
from core.config import BASEMODEL, get_settings


class QuestionResponse(BaseModel):
    question: str = Field(..., description="A single natural-language question")


@dataclass
class Sample:
    """生成されたサンプル"""
    query_type: str
    template_name: str
    question_template: str
    question: str  # パラフレーズ後の質問
    anchors: Dict[str, Dict[str, Any]]  # anchor_name -> {id, name, type}
    relations: List[str]
    answer_nodes: List[Dict[str, Any]]
    cypher_query: str


def get_neo4j_connection():
    """Neo4j接続を取得"""
    from database.search import GraphPathFinder
    finder = GraphPathFinder()
    return finder.graph


def generate_one_hop_samples(
    graph,
    template: Dict,
    num_samples: int = 100,
    min_answers: int = 1,
    max_answers: int = 50,
) -> List[Dict]:
    """1-hop サンプルを生成（Answer-first approach）"""

    src_type, rel, tgt_type = template["path"]
    src_label = get_cypher_label(src_type)
    tgt_label = get_cypher_label(tgt_type)

    # Answer-first: まずAnswerをサンプリングし、逆引きでAnchorを取得
    cypher = f"""
    MATCH (answer:{tgt_label})<-[r:{rel}]-(anchor:{src_label})
    WITH anchor, collect(DISTINCT answer)[0..{max_answers}] AS answers
    WHERE size(answers) >= {min_answers} AND size(answers) <= {max_answers}
    RETURN
        elementId(anchor) AS anchor_id,
        anchor.name AS anchor_name,
        labels(anchor)[0] AS anchor_type,
        [a IN answers | {{id: elementId(a), name: a.name, type: labels(a)[0]}}] AS answer_nodes
    ORDER BY rand()
    LIMIT {num_samples}
    """

    results = []
    try:
        for record in graph.run(cypher):
            sample = {
                "query_type": "one_hop",
                "template_name": template["name"],
                "anchor_id": record["anchor_id"],
                "anchor_name": record["anchor_name"],
                "anchor_type": record["anchor_type"],
                "relation": rel,
                "answer_nodes": record["answer_nodes"],
                "answer_count": len(record["answer_nodes"]),
                "question_templates": template["question_templates"],
            }
            results.append(sample)
    except Exception as e:
        print(f"Error in {template['name']}: {e}")

    return results


def generate_two_hop_samples(
    graph,
    template: Dict,
    num_samples: int = 100,
    min_answers: int = 1,
    max_answers: int = 50,
) -> List[Dict]:
    """2-hop サンプルを生成（Answer-first approach）"""

    a_type, rel1, z_type, rel2, x_type = template["path"]
    a_label = get_cypher_label(a_type)
    z_label = get_cypher_label(z_type)
    x_label = get_cypher_label(x_type)

    # Answer-first: Answer(x) -> Mid(z) -> Anchor(a) の順で逆引き
    cypher = f"""
    MATCH (x:{x_label})<-[r2:{rel2}]-(z:{z_label})<-[r1:{rel1}]-(a:{a_label})
    WHERE a <> z AND z <> x AND a <> x
    WITH a, z, collect(DISTINCT x)[0..{max_answers}] AS answers
    WHERE size(answers) >= {min_answers} AND size(answers) <= {max_answers}
    RETURN
        elementId(a) AS anchor_id,
        a.name AS anchor_name,
        labels(a)[0] AS anchor_type,
        elementId(z) AS mid_id,
        z.name AS mid_name,
        labels(z)[0] AS mid_type,
        [x IN answers | {{id: elementId(x), name: x.name, type: labels(x)[0]}}] AS answer_nodes
    ORDER BY rand()
    LIMIT {num_samples}
    """

    results = []
    try:
        for record in graph.run(cypher):
            sample = {
                "query_type": "two_hop",
                "template_name": template["name"],
                "anchor_id": record["anchor_id"],
                "anchor_name": record["anchor_name"],
                "anchor_type": record["anchor_type"],
                "mid_id": record["mid_id"],
                "mid_name": record["mid_name"],
                "mid_type": record["mid_type"],
                "rel1": rel1,
                "rel2": rel2,
                "answer_nodes": record["answer_nodes"],
                "answer_count": len(record["answer_nodes"]),
                "question_templates": template["question_templates"],
            }
            results.append(sample)
    except Exception as e:
        print(f"Error in {template['name']}: {e}")

    return results


def generate_two_intersection_samples(
    graph,
    template: Dict,
    num_samples: int = 100,
    min_answers: int = 1,
    max_answers: int = 50,
) -> List[Dict]:
    """2-anchor intersection サンプルを生成"""

    (a_type, a_rel), (b_type, b_rel) = template["anchors"]
    int_type = template["intersection_type"]

    a_label = get_cypher_label(a_type)
    b_label = get_cypher_label(b_type)
    int_label = get_cypher_label(int_type)

    # 共通のAnswerを持つ2つのAnchorを探す
    cypher = f"""
    MATCH (a:{a_label})-[ra:{a_rel}]->(x:{int_label})<-[rb:{b_rel}]-(b:{b_label})
    WHERE a <> b
    WITH a, b, collect(DISTINCT x)[0..{max_answers}] AS answers
    WHERE size(answers) >= {min_answers} AND size(answers) <= {max_answers}
    RETURN
        elementId(a) AS anchor_a_id,
        a.name AS anchor_a_name,
        labels(a)[0] AS anchor_a_type,
        elementId(b) AS anchor_b_id,
        b.name AS anchor_b_name,
        labels(b)[0] AS anchor_b_type,
        [x IN answers | {{id: elementId(x), name: x.name, type: labels(x)[0]}}] AS answer_nodes
    ORDER BY rand()
    LIMIT {num_samples}
    """

    results = []
    try:
        for record in graph.run(cypher):
            sample = {
                "query_type": "two_intersection",
                "template_name": template["name"],
                "anchor_a_id": record["anchor_a_id"],
                "anchor_a_name": record["anchor_a_name"],
                "anchor_a_type": record["anchor_a_type"],
                "anchor_a_rel": a_rel,
                "anchor_b_id": record["anchor_b_id"],
                "anchor_b_name": record["anchor_b_name"],
                "anchor_b_type": record["anchor_b_type"],
                "anchor_b_rel": b_rel,
                "intersection_type": int_type,
                "answer_nodes": record["answer_nodes"],
                "answer_count": len(record["answer_nodes"]),
                "question_templates": template["question_templates"],
            }
            results.append(sample)
    except Exception as e:
        print(f"Error in {template['name']}: {e}")

    return results


def generate_three_intersection_samples(
    graph,
    template: Dict,
    num_samples: int = 100,
    min_answers: int = 1,
    max_answers: int = 50,
) -> List[Dict]:
    """3-anchor intersection サンプルを生成"""

    (a_type, a_rel), (b_type, b_rel), (c_type, c_rel) = template["anchors"]
    int_type = template["intersection_type"]

    a_label = get_cypher_label(a_type)
    b_label = get_cypher_label(b_type)
    c_label = get_cypher_label(c_type)
    int_label = get_cypher_label(int_type)

    # 共通のAnswerを持つ3つのAnchorを探す
    cypher = f"""
    MATCH (a:{a_label})-[ra:{a_rel}]->(x:{int_label})<-[rb:{b_rel}]-(b:{b_label}),
          (x)<-[rc:{c_rel}]-(c:{c_label})
    WHERE a <> b AND b <> c AND a <> c
    WITH a, b, c, collect(DISTINCT x)[0..{max_answers}] AS answers
    WHERE size(answers) >= {min_answers} AND size(answers) <= {max_answers}
    RETURN
        elementId(a) AS anchor_a_id, a.name AS anchor_a_name, labels(a)[0] AS anchor_a_type,
        elementId(b) AS anchor_b_id, b.name AS anchor_b_name, labels(b)[0] AS anchor_b_type,
        elementId(c) AS anchor_c_id, c.name AS anchor_c_name, labels(c)[0] AS anchor_c_type,
        [x IN answers | {{id: elementId(x), name: x.name, type: labels(x)[0]}}] AS answer_nodes
    ORDER BY rand()
    LIMIT {num_samples}
    """

    results = []
    try:
        for record in graph.run(cypher):
            sample = {
                "query_type": "three_intersection",
                "template_name": template["name"],
                "anchor_a_id": record["anchor_a_id"],
                "anchor_a_name": record["anchor_a_name"],
                "anchor_a_type": record["anchor_a_type"],
                "anchor_a_rel": a_rel,
                "anchor_b_id": record["anchor_b_id"],
                "anchor_b_name": record["anchor_b_name"],
                "anchor_b_type": record["anchor_b_type"],
                "anchor_b_rel": b_rel,
                "anchor_c_id": record["anchor_c_id"],
                "anchor_c_name": record["anchor_c_name"],
                "anchor_c_type": record["anchor_c_type"],
                "anchor_c_rel": c_rel,
                "intersection_type": int_type,
                "answer_nodes": record["answer_nodes"],
                "answer_count": len(record["answer_nodes"]),
                "question_templates": template["question_templates"],
            }
            results.append(sample)
    except Exception as e:
        print(f"Error in {template['name']}: {e}")

    return results


def fill_template(template: str, anchors: Dict[str, str]) -> str:
    """質問テンプレートにAnchorを埋め込む"""
    question = template
    for key, value in anchors.items():
        question = question.replace(f"{{{key}}}", value)
    return question


class QuestionParaphraser:
    """LLMを使って質問をパラフレーズするクラス"""

    def __init__(self, model: str = BASEMODEL, temperature: float = 0.5, lang: str = "en"):
        if not os.getenv("OPENAI_API_KEY"):
            settings = get_settings()
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

        from langchain.chat_models import init_chat_model
        llm = init_chat_model(model, model_provider="openai", temperature=temperature)
        self._model = llm.with_structured_output(QuestionResponse)
        self._lang = lang

    def paraphrase(self, filled_question: str, context: Dict[str, Any]) -> str:
        """テンプレートから埋め込み済みの質問をパラフレーズする"""

        if self._lang.lower().startswith("ja"):
            lang_instruction = "Output Language: Japanese (Keep technical terms/entity names in English)."
        else:
            lang_instruction = "Output Language: English"

        prompt = f"""
You are an expert in biomedical domain knowledge graphs.
Your task is to paraphrase a question to make it sound more natural while preserving its exact meaning.

Original question: "{filled_question}"

Context:
- Query type: {context.get('query_type', 'unknown')}
- Relations: {context.get('relations', [])}

Guidelines:
1. Keep ALL entity names exactly as they appear (do not modify drug names, gene names, disease names, etc.)
2. Make the question sound natural and fluent
3. Use domain-appropriate verbs:
   - For "target" relation: use "targets", "is targeted by"
   - For "indication" relation: use "treats", "is used for", "is indicated for"
   - For "side_effect" relation: use "causes", "is associated with"
   - For "associated_disease" relation: use "is implicated in", "is linked to"
   - For "ppi" relation: use "interacts with", "binds to"
   - For "expression_present" relation: use "is expressed in"
   - For "parent_child" relation: use "is a type of", "belongs to"
4. Do NOT add new information
5. Do NOT use graph jargon like "node", "edge", "hop", "anchor"

{lang_instruction}

Return JSON with a single key "question".
""".strip()

        try:
            res = self._model.invoke(prompt)
            return res.question.strip()
        except Exception as e:
            print(f"Paraphrase error: {e}")
            return filled_question  # fallback to original


def paraphrase_questions_parallel(
    paraphraser: QuestionParaphraser,
    samples: List[Dict[str, Any]],
    max_workers: int = 8,
    max_inflight: int = 16,
) -> List[Dict[str, Any]]:
    """並列でパラフレーズを実行"""

    if not samples:
        return []

    sem = threading.Semaphore(max_inflight)

    def task(idx: int, sample: Dict[str, Any]):
        with sem:
            # テンプレートからランダムに1つ選択
            templates = sample.get("question_templates", [])
            if not templates:
                return idx, sample

            template = random.choice(templates)

            # Anchorsを抽出して埋め込み
            anchors = {}
            if "anchor_name" in sample:
                anchors["anchor"] = sample["anchor_name"]
            if "anchor_a_name" in sample:
                anchors["anchor_a"] = sample["anchor_a_name"]
            if "anchor_b_name" in sample:
                anchors["anchor_b"] = sample["anchor_b_name"]
            if "anchor_c_name" in sample:
                anchors["anchor_c"] = sample["anchor_c_name"]

            filled = fill_template(template, anchors)

            context = {
                "query_type": sample.get("query_type", ""),
                "relations": [sample.get("relation", sample.get("rel1", ""))],
            }
            if "rel2" in sample:
                context["relations"].append(sample["rel2"])

            paraphrased = paraphraser.paraphrase(filled, context)

            result = {**sample}
            result["question_template"] = template
            result["question"] = paraphrased
            del result["question_templates"]  # 元のテンプレートリストは削除

            return idx, result

    results_by_idx: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, i, s): i for i, s in enumerate(samples)}

        for future in tqdm(futures, desc="Paraphrasing", total=len(samples)):
            try:
                idx, result = future.result(timeout=60)
                results_by_idx[idx] = result
            except Exception as e:
                idx = futures[future]
                results_by_idx[idx] = {**samples[idx], "paraphrase_error": str(e)}

    return [results_by_idx[i] for i in range(len(samples))]


def main():
    parser = argparse.ArgumentParser(description="PrimeKG v2 Dataset Generator")
    parser.add_argument("--output-dir", type=Path, default=Path("result/dataset_v2"))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--min-answers", type=int, default=1)
    parser.add_argument("--max-answers", type=int, default=50)
    parser.add_argument(
        "--query-type",
        type=str,
        choices=["one_hop", "two_hop", "two_intersection", "three_intersection", "all"],
        default="all",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="Enable LLM paraphrasing for questions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=BASEMODEL,
        help="LLM model for paraphrasing",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "ja"],
        help="Question language",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max worker threads for paraphrasing",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to Neo4j...")
    graph = get_neo4j_connection()

    all_samples = {
        "one_hop": [],
        "two_hop": [],
        "two_intersection": [],
        "three_intersection": [],
    }

    # 1-hop
    if args.query_type in ["one_hop", "all"]:
        print("\n=== Generating 1-hop samples ===")
        for template in tqdm(ONE_HOP_TEMPLATES, desc="1-hop templates"):
            samples = generate_one_hop_samples(
                graph, template, args.num_samples, args.min_answers, args.max_answers
            )
            all_samples["one_hop"].extend(samples)
            print(f"  {template['name']}: {len(samples)} samples")

    # 2-hop
    if args.query_type in ["two_hop", "all"]:
        print("\n=== Generating 2-hop samples ===")
        for template in tqdm(TWO_HOP_TEMPLATES, desc="2-hop templates"):
            samples = generate_two_hop_samples(
                graph, template, args.num_samples, args.min_answers, args.max_answers
            )
            all_samples["two_hop"].extend(samples)
            print(f"  {template['name']}: {len(samples)} samples")

    # 2-intersection
    if args.query_type in ["two_intersection", "all"]:
        print("\n=== Generating 2-intersection samples ===")
        for template in tqdm(TWO_ANCHOR_INTERSECTION_TEMPLATES, desc="2-intersection"):
            samples = generate_two_intersection_samples(
                graph, template, args.num_samples, args.min_answers, args.max_answers
            )
            all_samples["two_intersection"].extend(samples)
            print(f"  {template['name']}: {len(samples)} samples")

    # 3-intersection
    if args.query_type in ["three_intersection", "all"]:
        print("\n=== Generating 3-intersection samples ===")
        for template in tqdm(THREE_ANCHOR_INTERSECTION_TEMPLATES, desc="3-intersection"):
            samples = generate_three_intersection_samples(
                graph, template, args.num_samples, args.min_answers, args.max_answers
            )
            all_samples["three_intersection"].extend(samples)
            print(f"  {template['name']}: {len(samples)} samples")

    # パラフレーズ（オプション）
    if args.paraphrase:
        print("\n=== Paraphrasing questions ===")
        paraphraser = QuestionParaphraser(model=args.model, lang=args.lang)

        for query_type, samples in all_samples.items():
            if samples:
                print(f"  Paraphrasing {query_type}...")
                all_samples[query_type] = paraphrase_questions_parallel(
                    paraphraser, samples, max_workers=args.max_workers
                )
    else:
        # パラフレーズしない場合はテンプレートをそのまま使用
        for query_type, samples in all_samples.items():
            for sample in samples:
                templates = sample.get("question_templates", [])
                if templates:
                    template = random.choice(templates)
                    anchors = {}
                    if "anchor_name" in sample:
                        anchors["anchor"] = sample["anchor_name"]
                    if "anchor_a_name" in sample:
                        anchors["anchor_a"] = sample["anchor_a_name"]
                    if "anchor_b_name" in sample:
                        anchors["anchor_b"] = sample["anchor_b_name"]
                    if "anchor_c_name" in sample:
                        anchors["anchor_c"] = sample["anchor_c_name"]

                    sample["question_template"] = template
                    sample["question"] = fill_template(template, anchors)
                    del sample["question_templates"]

    # 保存
    print("\n=== Saving results ===")
    for query_type, samples in all_samples.items():
        if samples:
            output_path = args.output_dir / f"{query_type}.jsonl"
            with output_path.open("w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"  {query_type}: {len(samples)} samples -> {output_path}")

    # サマリー
    print("\n=== Summary ===")
    total = 0
    for query_type, samples in all_samples.items():
        print(f"  {query_type}: {len(samples)}")
        total += len(samples)
    print(f"  Total: {total}")


if __name__ == "__main__":
    main()
