"""
Reranker有無のETK評価結果を比較し、
rerankerが結果を悪化させたサンプルを特定する
"""
import json
import sys
from pathlib import Path
from collections import Counter


def load_outputs(path: Path):
    outputs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line.strip()))
    return outputs


def compute_accuracy(output):
    """predicted_entitiesとgold_answersの交差判定"""
    gold = set(str(a).lower() for a in output.get("gold_answers", []))
    pred = set(str(a).lower() for a in output.get("predicted_entities", []))
    if not gold:
        return False
    return len(gold & pred) > 0


def main():
    no_reranker_path = Path("volume_pcqa/outputs_no_reranker/extended_type_kopl_all.jsonl")
    reranker_path = Path("volume_pcqa/outputs_reranker/extended_type_kopl_all.jsonl")

    if not no_reranker_path.exists() or not reranker_path.exists():
        print("Output files not found yet. Waiting for evaluations to complete.")
        sys.exit(1)

    no_rr = load_outputs(no_reranker_path)
    rr = load_outputs(reranker_path)

    print(f"No-Reranker: {len(no_rr)} samples")
    print(f"Reranker:    {len(rr)} samples")
    print()

    # 全体比較
    no_rr_correct = sum(1 for o in no_rr if compute_accuracy(o))
    rr_correct = sum(1 for o in rr if compute_accuracy(o))
    print(f"No-Reranker accuracy: {no_rr_correct}/{len(no_rr)} ({100*no_rr_correct/len(no_rr):.1f}%)")
    print(f"Reranker accuracy:    {rr_correct}/{len(rr)} ({100*rr_correct/len(rr):.1f}%)")
    print()

    # per-sample比較
    improved = []   # rerankerで改善
    degraded = []   # rerankerで悪化
    both_correct = 0
    both_wrong = 0

    for i, (nr, r) in enumerate(zip(no_rr, rr)):
        nr_ok = compute_accuracy(nr)
        r_ok = compute_accuracy(r)

        if nr_ok and r_ok:
            both_correct += 1
        elif not nr_ok and not r_ok:
            both_wrong += 1
        elif not nr_ok and r_ok:
            improved.append((i, nr, r))
        elif nr_ok and not r_ok:
            degraded.append((i, nr, r))

    print(f"Both correct:        {both_correct}")
    print(f"Both wrong:          {both_wrong}")
    print(f"Improved by reranker: {len(improved)}")
    print(f"Degraded by reranker: {len(degraded)}")
    print()

    # rerankerで悪化したサンプルの詳細
    if degraded:
        print("=" * 80)
        print(f"DEGRADED BY RERANKER ({len(degraded)} samples):")
        print("=" * 80)
        for idx, nr, r in degraded:
            print(f"\n--- Sample {idx} ---")
            print(f"  Question: {nr['question']}")
            print(f"  Entity:   {nr['entity_name']}")
            print(f"  Gold relations: {nr['gold_relations']}")
            print(f"  Gold answers:   {nr['gold_answers'][:5]}{'...' if len(nr['gold_answers']) > 5 else ''}")
            print(f"  [No-Reranker] predicted_relations: {nr['predicted_relations']}")
            print(f"  [No-Reranker] predicted_entities:  {nr['predicted_entities'][:5]}{'...' if len(nr['predicted_entities']) > 5 else ''}")
            print(f"  [Reranker]    predicted_relations: {r['predicted_relations']}")
            print(f"  [Reranker]    predicted_entities:  {r['predicted_entities'][:5]}{'...' if len(r['predicted_entities']) > 5 else ''}")

    # rerankerで改善したサンプルの詳細
    if improved:
        print()
        print("=" * 80)
        print(f"IMPROVED BY RERANKER ({len(improved)} samples):")
        print("=" * 80)
        for idx, nr, r in improved:
            print(f"\n--- Sample {idx} ---")
            print(f"  Question: {nr['question']}")
            print(f"  Entity:   {nr['entity_name']}")
            print(f"  Gold relations: {nr['gold_relations']}")
            print(f"  Gold answers:   {nr['gold_answers'][:5]}{'...' if len(nr['gold_answers']) > 5 else ''}")
            print(f"  [No-Reranker] predicted_relations: {nr['predicted_relations']}")
            print(f"  [No-Reranker] predicted_entities:  {nr['predicted_entities'][:5]}{'...' if len(nr['predicted_entities']) > 5 else ''}")
            print(f"  [Reranker]    predicted_relations: {r['predicted_relations']}")
            print(f"  [Reranker]    predicted_entities:  {r['predicted_entities'][:5]}{'...' if len(r['predicted_entities']) > 5 else ''}")

    # relation変更パターン分析
    print()
    print("=" * 80)
    print("RELATION CHANGE PATTERNS (degraded samples):")
    print("=" * 80)
    changes = Counter()
    for idx, nr, r in degraded:
        nr_rels = tuple(nr['predicted_relations']) if nr['predicted_relations'] else ('None',)
        r_rels = tuple(r['predicted_relations']) if r['predicted_relations'] else ('None',)
        changes[(nr_rels, r_rels)] += 1

    for (from_rel, to_rel), cnt in changes.most_common():
        print(f"  {cnt}x: {' -> '.join(from_rel)} ==> {' -> '.join(to_rel)}")


if __name__ == "__main__":
    main()
