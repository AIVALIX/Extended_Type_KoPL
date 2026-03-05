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

from core.config import BASEMODEL, get_settings


def load_schema(schema_version: str = "v2"):
    """スキーマバージョンに応じてテンプレートを読み込む"""
    if schema_version == "v4":
        from dataset_construction.schema_v4 import (
            ONE_HOP_TEMPLATES,
            TWO_HOP_TEMPLATES,
            TWO_ANCHOR_INTERSECTION_TEMPLATES,
            THREE_ANCHOR_INTERSECTION_TEMPLATES,
            get_cypher_label,
        )
    elif schema_version == "v3":
        from dataset_construction.schema_v3 import (
            ONE_HOP_TEMPLATES,
            TWO_HOP_TEMPLATES,
            TWO_ANCHOR_INTERSECTION_TEMPLATES,
            THREE_ANCHOR_INTERSECTION_TEMPLATES,
            get_cypher_label,
        )
    else:
        from dataset_construction.schema_v2 import (
            ONE_HOP_TEMPLATES,
            TWO_HOP_TEMPLATES,
            TWO_ANCHOR_INTERSECTION_TEMPLATES,
            THREE_ANCHOR_INTERSECTION_TEMPLATES,
            get_cypher_label,
        )
    return (
        ONE_HOP_TEMPLATES,
        TWO_HOP_TEMPLATES,
        TWO_ANCHOR_INTERSECTION_TEMPLATES,
        THREE_ANCHOR_INTERSECTION_TEMPLATES,
        get_cypher_label,
    )


import re

def is_readable_entity_name(name: str) -> bool:
    """
    人間が読める・理解しやすいエンティティ名かどうかを判定

    除外対象:
    - 長すぎる化学名 (30文字超)
    - コードネーム (CRA_8696, DB00001など) ※遺伝子名は許可
    - 括弧だらけの化学名
    - 数字のみや特殊文字のみ
    """
    if not name or len(name.strip()) == 0:
        return False

    # 長すぎる名前を除外
    if len(name) > 30:
        return False

    # 遺伝子名パターンを許可 (BRCA1, TP53, EGFR, ALK, MET, etc.)
    # 2-6文字の大文字(数字を含んでもよい)は遺伝子名として許可
    if re.match(r'^[A-Z][A-Z0-9]{1,5}$', name):
        return True

    # コードネームパターンを除外 (CRA_8696, CHEMBL123456, etc.)
    # アンダースコアを含む大文字+数字のパターン
    if re.match(r'^[A-Z]{2,}_\d+$', name):
        return False

    # DrugBank IDを除外
    if re.match(r'^DB\d+', name):
        return False

    # CHEMBL IDを除外
    if re.match(r'^CHEMBL\d+', name):
        return False

    # 括弧が多すぎる化学名を除外
    if name.count('(') > 2 or name.count('[') > 1:
        return False

    # 数字や特殊文字のみの名前を除外
    if re.match(r'^[\d\s\-\.,]+$', name):
        return False

    # 化学式っぽいパターンを除外 (大文字+数字の繰り返しで7文字以上)
    if len(name) > 6 and re.match(r'^[A-Z][a-z]?\d+([A-Z][a-z]?\d*)+$', name):
        return False

    return True


def filter_samples_by_entity_name(samples: List[Dict], entity_key: str = "anchor_name") -> List[Dict]:
    """
    読みやすいエンティティ名のサンプルのみをフィルタリング
    """
    filtered = []
    for sample in samples:
        name = sample.get(entity_key, "")
        if is_readable_entity_name(name):
            filtered.append(sample)
    return filtered


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


def get_cypher_rel(rel: str) -> str:
    """リレーション名をCypher用にエスケープ（スペース、ハイフン等）"""
    if " " in rel or "-" in rel:
        return f"`{rel}`"
    return rel


def get_cypher_label(node_type: str) -> str:
    """ノードタイプをCypher用のラベルに変換"""
    if "/" in node_type:
        return f"`{node_type}`"
    return node_type


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
    rel_escaped = get_cypher_rel(rel)

    # Answer-first: まずAnswerをサンプリングし、逆引きでAnchorを取得
    cypher = f"""
    MATCH (answer:{tgt_label})<-[r:{rel_escaped}]-(anchor:{src_label})
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
    max_total_answers: int = 100,
) -> List[Dict]:
    """2-hop サンプルを生成（Answer-first approach）

    Args:
        max_total_answers: 全パス経由での総回答数の上限。
            これを超えるアンカーは除外する（パイプライン評価時の爆発を防ぐ）
    """

    a_type, rel1, z_type, rel2, x_type = template["path"]
    a_label = get_cypher_label(a_type)
    z_label = get_cypher_label(z_type)
    x_label = get_cypher_label(x_type)

    def get_rel(r: str) -> str:
        """リレーション名をCypher用にエスケープ"""
        if " " in r or "-" in r:
            return f"`{r}`"
        return r

    rel1_escaped = get_rel(rel1)
    rel2_escaped = get_rel(rel2)

    # Step 1: 全パス経由での総回答数が上限以下のアンカーを特定
    # パイプラインは全ての中間ノードを経由するため、総回答数でフィルタ
    cypher_filter = f"""
    MATCH (a:{a_label})-[r1:{rel1_escaped}]-(z:{z_label})-[r2:{rel2_escaped}]-(x:{x_label})
    WHERE a <> z AND z <> x AND a <> x
    WITH a, collect(DISTINCT x) AS all_answers
    WHERE size(all_answers) >= {min_answers} AND size(all_answers) <= {max_total_answers}
    RETURN
        elementId(a) AS anchor_id,
        a.name AS anchor_name,
        labels(a)[0] AS anchor_type,
        [x IN all_answers[0..{max_answers}] | {{id: elementId(x), name: x.name, type: labels(x)[0]}}] AS answer_nodes,
        size(all_answers) AS total_answer_count
    ORDER BY rand()
    LIMIT {num_samples}
    """

    results = []
    try:
        for record in graph.run(cypher_filter):
            sample = {
                "query_type": "two_hop",
                "template_name": template["name"],
                "anchor_id": record["anchor_id"],
                "anchor_name": record["anchor_name"],
                "anchor_type": record["anchor_type"],
                "rel1": rel1,
                "rel2": rel2,
                "answer_nodes": record["answer_nodes"],
                "answer_count": len(record["answer_nodes"]),
                "total_answer_count": record["total_answer_count"],
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
    a_rel_escaped = get_cypher_rel(a_rel)
    b_rel_escaped = get_cypher_rel(b_rel)

    # 共通のAnswerを持つ2つのAnchorを探す
    cypher = f"""
    MATCH (a:{a_label})-[ra:{a_rel_escaped}]->(x:{int_label})<-[rb:{b_rel_escaped}]-(b:{b_label})
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
    a_rel_escaped = get_cypher_rel(a_rel)
    b_rel_escaped = get_cypher_rel(b_rel)
    c_rel_escaped = get_cypher_rel(c_rel)

    # 共通のAnswerを持つ3つのAnchorを探す
    cypher = f"""
    MATCH (a:{a_label})-[ra:{a_rel_escaped}]->(x:{int_label})<-[rb:{b_rel_escaped}]-(b:{b_label}),
          (x)<-[rc:{c_rel_escaped}]-(c:{c_label})
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
    parser = argparse.ArgumentParser(description="PrimeKG Dataset Generator")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: result/dataset_{schema})")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--min-answers", type=int, default=1)
    parser.add_argument("--max-answers", type=int, default=50)
    parser.add_argument(
        "--max-total-answers",
        type=int,
        default=100,
        help="Max total answers for 2-hop queries (prevents explosion)",
    )
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
    parser.add_argument(
        "--schema",
        type=str,
        default="v2",
        choices=["v2", "v3", "v4"],
        help="Schema version to use (v3 uses clear relations, v4 adds improved templates and filtering)",
    )
    parser.add_argument(
        "--filter-entities",
        action="store_true",
        help="Filter out samples with unreadable entity names (long chemical names, codes, etc.)",
    )
    args = parser.parse_args()

    # デフォルトの出力ディレクトリを設定
    if args.output_dir is None:
        args.output_dir = Path(f"result/dataset_{args.schema}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # スキーマを読み込み
    print(f"Loading schema {args.schema}...")
    (
        ONE_HOP_TEMPLATES,
        TWO_HOP_TEMPLATES,
        TWO_ANCHOR_INTERSECTION_TEMPLATES,
        THREE_ANCHOR_INTERSECTION_TEMPLATES,
        _,
    ) = load_schema(args.schema)

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
                graph, template, args.num_samples, args.min_answers, args.max_answers,
                max_total_answers=args.max_total_answers,
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

    # エンティティ名フィルタリング（オプション）
    if args.filter_entities:
        print("\n=== Filtering samples by entity name ===")
        for query_type, samples in all_samples.items():
            if samples:
                original_count = len(samples)
                # フィルタリング対象のエンティティキーを決定
                if query_type in ["two_intersection", "three_intersection"]:
                    # インターセクションは複数アンカーがあるので、すべてをチェック
                    filtered = []
                    for sample in samples:
                        names_to_check = []
                        for key in ["anchor_a_name", "anchor_b_name", "anchor_c_name"]:
                            if key in sample:
                                names_to_check.append(sample[key])
                        if all(is_readable_entity_name(name) for name in names_to_check):
                            filtered.append(sample)
                    all_samples[query_type] = filtered
                else:
                    all_samples[query_type] = filter_samples_by_entity_name(samples, "anchor_name")
                filtered_count = len(all_samples[query_type])
                print(f"  {query_type}: {original_count} -> {filtered_count} ({original_count - filtered_count} removed)")

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
