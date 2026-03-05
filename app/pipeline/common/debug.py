"""
共通デバッグユーティリティ
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_samples(data_path: Path, indices: List[int]) -> List[tuple]:
    """複数インデックスのサンプルを読み込む"""
    indices_set = set(indices)
    max_idx = max(indices) if indices else 0
    samples = {}

    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > max_idx:
                break
            if i in indices_set:
                samples[i] = json.loads(line.strip())

    return [(idx, samples[idx]) for idx in indices if idx in samples]


def get_gold_info(sample: Dict[str, Any]) -> Dict[str, Any]:
    """正解情報を抽出"""
    # PcQA形式: "answers" フィールド（[{"name": "..."}]）
    if "answers" in sample and "answer_nodes" not in sample:
        gold_answers = [a["name"] for a in sample.get("answers", []) if "name" in a]
        relations = [sample.get("relation")] if sample.get("relation") else []
        return {
            "query_type": "one_hop",
            "relations": relations,
            "answers": gold_answers,
            "entity_key": "entity",
        }

    # PrimeKGQA/MetaQA形式
    gold_answers = [node["name"] for node in sample.get("answer_nodes", [])]

    # クエリタイプを検出
    if "anchor_c_name" in sample:
        query_type = "three_intersection"
        relations = [sample.get("anchor_a_rel"), sample.get("anchor_b_rel"), sample.get("anchor_c_rel")]
        entity_key = "anchor_a_name"
    elif "anchor_b_name" in sample:
        query_type = "two_intersection"
        relations = [sample.get("anchor_a_rel"), sample.get("anchor_b_rel")]
        entity_key = "anchor_a_name"
    elif "rel2" in sample:
        query_type = "two_hop"
        relations = [sample.get("rel1"), sample.get("rel2")]
        entity_key = "anchor_name"
    else:
        query_type = "one_hop"
        relations = [sample.get("relation")]
        entity_key = "anchor_name"

    return {
        "query_type": query_type,
        "relations": [r for r in relations if r],
        "answers": gold_answers,
        "entity_key": entity_key,
    }


def compute_metrics(
    gold_answers: List[str],
    predicted: List[str],
    gold_relations: Optional[List[str]] = None,
    predicted_relations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """メトリクスを計算"""
    gold_set = set(gold_answers)
    pred_set = set(predicted)
    overlap = gold_set & pred_set

    accuracy = gold_set <= pred_set if gold_set else False
    recall = len(overlap) / len(gold_set) if gold_set else 0.0
    precision = len(overlap) / len(pred_set) if pred_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    path_match = False
    if gold_relations and predicted_relations:
        path_match = gold_relations == predicted_relations

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "path_match": path_match,
        "overlap_count": len(overlap),
        "gold_count": len(gold_set),
        "pred_count": len(pred_set),
    }


def print_header(title: str, width: int = 60):
    """ヘッダーを出力"""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_metrics(metrics: Dict[str, Any]):
    """メトリクスを出力"""
    print_header("METRICS")
    print(f"  Accuracy:  {'✓' if metrics['accuracy'] else '✗'}")
    print(f"  Recall:    {metrics['recall']:.1%} ({metrics['overlap_count']}/{metrics['gold_count']})")
    print(f"  Precision: {metrics['precision']:.1%} ({metrics['overlap_count']}/{metrics['pred_count']})")
    print(f"  F1:        {metrics['f1']:.1%}")
    if 'path_match' in metrics:
        print(f"  PathMatch: {'✓' if metrics['path_match'] else '✗'}")
