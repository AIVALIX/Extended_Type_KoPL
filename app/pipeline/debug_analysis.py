"""
Extended Type-KoPL パイプラインのデバッグ分析

各段階でのミスマッチを調査する
"""

import json
import random
from pathlib import Path
from collections import Counter

from pipeline.extended_type_kopl import ExtendedTypeKoPLPipeline


def load_dataset(path: str, num_samples: int = 10, random_sample: bool = True):
    """データセットを読み込み"""
    data_path = Path(path)
    all_samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))

    if random_sample and len(all_samples) > num_samples:
        return random.sample(all_samples, num_samples)
    return all_samples[:num_samples]


def analyze_sample(pipeline: ExtendedTypeKoPLPipeline, sample: dict, idx: int):
    """サンプルを分析"""
    question = sample.get("question", "")
    entity_name = sample.get("entity", "")
    gold_relations = [
        sample.get("relation1", ""),
        sample.get("relation2", ""),
        sample.get("relation3", ""),
    ]
    gold_relations = [r for r in gold_relations if r]  # 空を除去
    gold_answers = sample.get("answers", [])

    print(f"\n{'='*80}")
    print(f"Sample {idx}")
    print(f"{'='*80}")
    print(f"Question: {question}")
    print(f"Entity: {entity_name}")
    print(f"Gold Relations: {gold_relations}")
    print(f"Gold Answers: {gold_answers[:5]}..." if len(gold_answers) > 5 else f"Gold Answers: {gold_answers}")

    # パイプライン実行
    result = pipeline.run(question=question, entity_name=entity_name)

    # Phase 1: relation_hints の分析
    print(f"\n--- Phase 1: Relation Hints ---")
    relation_hints = pipeline._extract_relation_hints(result.kopl_program) if result.kopl_program else None
    print(f"Generated hints: {relation_hints}")
    print(f"Gold relations:  {gold_relations}")

    if relation_hints and gold_relations:
        hint_match = relation_hints == gold_relations
        partial_match = sum(1 for h, g in zip(relation_hints, gold_relations) if h == g)
        print(f"Exact match: {hint_match}")
        print(f"Partial match: {partial_match}/{len(gold_relations)}")

    # Phase 2: 候補パスの分析
    print(f"\n--- Phase 2: Candidate Paths ---")
    print(f"Total candidates: {len(result.candidate_paths)}")

    # 正解パスが候補に含まれているか
    gold_path_in_candidates = False
    for p in result.candidate_paths:
        if p.relations == gold_relations:
            gold_path_in_candidates = True
            print(f"  [FOUND] Gold path in candidates: {p.to_text()}")
            break

    if not gold_path_in_candidates:
        print(f"  [MISS] Gold path NOT in candidates!")
        print(f"  Candidate relations:")
        rel_counter = Counter([tuple(p.relations) for p in result.candidate_paths])
        for rels, count in rel_counter.most_common(5):
            print(f"    {list(rels)}: {count}")

    # Phase 3: ベクトル剪定後の分析
    print(f"\n--- Phase 3: After Vector Pruning ---")
    print(f"Selected paths: {len(result.selected_paths)}")

    gold_path_in_selected = False
    for i, p in enumerate(result.selected_paths):
        is_gold = p.relations == gold_relations
        marker = "[GOLD]" if is_gold else ""
        print(f"  {i}: {p.relations} (score: {p.score:.4f}) {marker}")
        if is_gold:
            gold_path_in_selected = True

    if not gold_path_in_selected:
        print(f"  [MISS] Gold path NOT in selected!")

    # Phase 4-5: 最終結果
    print(f"\n--- Final Result ---")
    print(f"Predicted: {result.answer_entities[:5]}..." if len(result.answer_entities) > 5 else f"Predicted: {result.answer_entities}")

    # 精度計算
    pred_set = set(result.answer_entities)
    # gold_answers が辞書のリストの場合は name を取り出す
    if gold_answers and isinstance(gold_answers[0], dict):
        gold_set = set(a.get("name", "") for a in gold_answers)
    else:
        gold_set = set(gold_answers)

    if gold_set:
        recall = len(pred_set & gold_set) / len(gold_set) * 100
        precision = len(pred_set & gold_set) / len(pred_set) * 100 if pred_set else 0
        print(f"Recall: {recall:.1f}%, Precision: {precision:.1f}%")

    return {
        "idx": idx,
        "hint_exact_match": relation_hints == gold_relations if relation_hints else False,
        "hint_partial_match": sum(1 for h, g in zip(relation_hints or [], gold_relations) if h == g),
        "gold_in_candidates": gold_path_in_candidates,
        "gold_in_selected": gold_path_in_selected,
        "recall": len(pred_set & gold_set) / len(gold_set) * 100 if gold_set else 0,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="3hop")
    parser.add_argument("--reranker", type=str, default="none")
    parser.add_argument("--reranker-input-k", type=int, default=10)
    args = parser.parse_args()

    # データセットパス
    dataset_paths = {
        "1hop": "result/metaqa/1hop.jsonl",
        "2hop": "result/metaqa/2hop.jsonl",
        "3hop": "result/metaqa/3hop.jsonl",
    }

    print("Loading pipeline...")
    pipeline = ExtendedTypeKoPLPipeline(
        kg_type="metaqa",
        reranker_type=args.reranker,
        reranker_input_k=args.reranker_input_k,
    )

    print(f"Loading dataset: {args.dataset}")
    samples = load_dataset(dataset_paths[args.dataset], args.num_samples)

    results = []
    for i, sample in enumerate(samples):
        result = analyze_sample(pipeline, sample, i)
        results.append(result)

    # サマリー
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    hint_exact = sum(1 for r in results if r["hint_exact_match"])
    hint_partial_avg = sum(r["hint_partial_match"] for r in results) / len(results)
    gold_in_cand = sum(1 for r in results if r["gold_in_candidates"])
    gold_in_sel = sum(1 for r in results if r["gold_in_selected"])
    avg_recall = sum(r["recall"] for r in results) / len(results)

    print(f"Phase 1 - Hint Exact Match: {hint_exact}/{len(results)} ({hint_exact/len(results)*100:.1f}%)")
    print(f"Phase 1 - Hint Partial Match Avg: {hint_partial_avg:.2f}/3")
    print(f"Phase 2 - Gold in Candidates: {gold_in_cand}/{len(results)} ({gold_in_cand/len(results)*100:.1f}%)")
    print(f"Phase 3 - Gold in Selected: {gold_in_sel}/{len(results)} ({gold_in_sel/len(results)*100:.1f}%)")
    print(f"Final - Average Recall: {avg_recall:.1f}%")


if __name__ == "__main__":
    main()
