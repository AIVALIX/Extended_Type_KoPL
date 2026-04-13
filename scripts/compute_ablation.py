"""Aggregate ablation study results across multiple runs.

Usage:
    python scripts/compute_ablation.py
"""
import json
import os
from typing import Dict, List, Optional, Tuple


CONFIGS = [
    ("A: vanilla ETK",
     "result/multimodel_20260411/gemma3_27b/metaqa",
     "extended_type_kopl_{ds}.jsonl"),
    ("B: +Fix A",
     "result/fix_anchor_reorient/gemma3_27b/metaqa",
     "extended_type_kopl_{ds}.jsonl"),
    ("C: +correction=1 (best baseline)",
     "result/fix_anchor_reorient_final/gemma3_27b/metaqa",
     "extended_type_kopl_{ds}.jsonl"),
    ("D: +n_kopl=3 plain",
     "result/ablation/gemma3_27b/metaqa_nkopl3",
     "extended_type_kopl_{ds}.jsonl"),
    ("E: -reranker",
     "result/ablation/gemma3_27b/metaqa_noreranker",
     "extended_type_kopl_{ds}.jsonl"),
    ("F: +CIR",
     "result/ablation/gemma3_27b/metaqa_cir",
     "extended_type_kopl_{ds}.jsonl"),
    ("G: +n_kopl=3 +enhanced scoring",
     "result/ablation/gemma3_27b/metaqa_nkopl3_enhanced",
     "extended_type_kopl_{ds}.jsonl"),
]

DATASETS = ["1hop", "2hop", "3hop"]


def score(jsonl_path: str) -> Optional[Tuple[int, float, float, int]]:
    """Return (n, accuracy, f1, errors) for a result file."""
    if not os.path.exists(jsonl_path):
        return None
    n = 0
    exact = 0
    f1_total = 0.0
    errors = 0
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                n += 1
                if r.get("error"):
                    errors += 1
                    continue
                gold = set(r.get("gold_answers", []))
                pred = set(r.get("predicted_entities", []))
                if gold == pred and gold:
                    exact += 1
                tp = len(gold & pred)
                rec = tp / len(gold) if gold else 0
                prec = tp / len(pred) if pred else 0
                f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
                f1_total += f1
            except Exception:
                pass
    if n == 0:
        return None
    return (n, 100 * exact / n, 100 * f1_total / n, errors)


def main() -> None:
    print("=" * 80)
    print("ETK Ablation Study — MetaQA (gemma3:27b, n=100, seed=42)")
    print("=" * 80)
    header = f"{'Config':<38s}" + "".join(f"{ds:>15s}" for ds in DATASETS)
    print(header)
    print("-" * len(header))

    results: Dict[str, Dict[str, Optional[Tuple]]] = {}
    for name, path, tmpl in CONFIGS:
        row: Dict[str, Optional[Tuple]] = {}
        display = [f"{name:<38s}"]
        for ds in DATASETS:
            p = os.path.join(path, tmpl.format(ds=ds))
            s = score(p)
            row[ds] = s
            if s:
                n, acc, f1, err = s
                errstr = f" ({err}e)" if err else ""
                display.append(f"{acc:5.1f}/{f1:5.1f}{errstr:>5s}")
            else:
                display.append(f"{'--':>15s}")
        results[name] = row
        print("  " + "  ".join(display))

    print()
    print("=" * 80)
    print("Contribution decomposition (2-hop Accuracy Δ)")
    print("=" * 80)

    def acc(name: str, ds: str) -> Optional[float]:
        r = results.get(name, {}).get(ds)
        return r[1] if r else None

    base_a = acc("A: vanilla ETK", "2hop")
    base_b = acc("B: +Fix A", "2hop")
    base_c = acc("C: +correction=1 (best baseline)", "2hop")
    d = acc("D: +n_kopl=3 plain", "2hop")
    e = acc("E: -reranker", "2hop")
    f_ = acc("F: +CIR", "2hop")
    g = acc("G: +n_kopl=3 +enhanced scoring", "2hop")

    def fmt(x: Optional[float]) -> str:
        return f"{x:.1f}" if x is not None else "N/A"

    if base_a is not None and base_b is not None:
        print(f"Fix A               : {fmt(base_b)} - {fmt(base_a)} = {(base_b - base_a):+.1f}")
    if base_b is not None and base_c is not None:
        print(f"correction=1        : {fmt(base_c)} - {fmt(base_b)} = {(base_c - base_b):+.1f}")
    if base_c is not None and d is not None:
        print(f"n_kopl=3 (plain)    : {fmt(d)} - {fmt(base_c)} = {(d - base_c):+.1f}")
    if base_c is not None and e is not None:
        print(f"Reranker (inv from E): {fmt(base_c)} - {fmt(e)} = {(base_c - e):+.1f}")
    if base_c is not None and f_ is not None:
        print(f"CIR                 : {fmt(f_)} - {fmt(base_c)} = {(f_ - base_c):+.1f}")
    if d is not None and g is not None:
        print(f"Enhanced scoring    : {fmt(g)} - {fmt(d)} = {(g - d):+.1f}")


def main_pcqa_primekgqa() -> None:
    """PrimeKGQA ablation (placeholder, will use once results exist)."""
    configs = [
        ("C: +correction=1 (baseline)",
         "result/fix_anchor_reorient_final/gemma3_27b/primekgqa"),
        ("D: +n_kopl=3 plain",
         "result/ablation/gemma3_27b/primekgqa_nkopl3"),
        ("G: +n_kopl=3 +enhanced scoring",
         "result/ablation/gemma3_27b/primekgqa_nkopl3_enhanced"),
    ]
    datasets = ["one_hop", "two_hop", "two_intersection"]
    print("=" * 80)
    print("ETK Ablation — PrimeKGQA (gemma3:27b, n=100, seed=42)")
    print("=" * 80)
    header = f"{'Config':<38s}" + "".join(f"{ds:>18s}" for ds in datasets)
    print(header)
    print("-" * len(header))
    for name, path in configs:
        display = [f"{name:<38s}"]
        for ds in datasets:
            p = os.path.join(path, f"extended_type_kopl_{ds}.jsonl")
            s = score(p)
            if s:
                n, acc, f1, err = s
                errstr = f" ({err}e)" if err else ""
                display.append(f"{acc:5.1f}/{f1:5.1f}{errstr:>6s}")
            else:
                display.append(f"{'--':>18s}")
        print("  " + "  ".join(display))


if __name__ == "__main__":
    import sys
    main()
    if "--primekgqa" in sys.argv or "--all" in sys.argv:
        print()
        main_pcqa_primekgqa()
