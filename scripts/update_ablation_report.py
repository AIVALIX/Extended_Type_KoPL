"""Fill in the ablation report tables from measured jsonl files.

Reads docs/ablation_report_20260414.md, computes metrics for each config,
and rewrites the result table with the latest numbers.

Usage: python scripts/update_ablation_report.py
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple


REPORT = "docs/ablation_report_20260414.md"

# Each row: (label, path_to_result_dir, datasets_to_include)
ROWS = [
    ("A: vanilla ETK",
     "result/multimodel_20260411/gemma3_27b/metaqa",
     ["1hop", "2hop", "3hop"]),
    ("B: +Fix A",
     "result/fix_anchor_reorient/gemma3_27b/metaqa",
     ["1hop", "2hop", "3hop"]),
    ("C: +correction (= best baseline)",
     "result/fix_anchor_reorient_final/gemma3_27b/metaqa",
     ["1hop", "2hop", "3hop"]),
    ("D: +n_kopl=3 (plain)",
     "result/ablation/gemma3_27b/metaqa_nkopl3_2hop_plain",
     ["2hop"]),
    ("E: -reranker",
     "result/ablation/gemma3_27b/metaqa_noreranker",
     ["1hop", "2hop", "3hop"]),
    ("F: +CIR",
     "result/ablation/gemma3_27b/metaqa_cir",
     ["1hop", "2hop", "3hop"]),
    ("G: +n_kopl=3 +enhanced scoring",
     "result/ablation/gemma3_27b/metaqa_nkopl3_2hop_enhanced",
     ["2hop"]),
]

DATASETS = ["1hop", "2hop", "3hop"]


def score(jsonl_path: str) -> Optional[Tuple[int, float, float, int]]:
    if not os.path.exists(jsonl_path):
        return None
    n = exact = errors = 0
    f1_total = 0.0
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


def fmt_cell(s: Optional[Tuple]) -> str:
    if s is None:
        return "_pending_"
    n, acc, f1, err = s
    if err > 0:
        return f"{acc:.0f} / {f1:.0f} ({err}e)"
    return f"{acc:.0f} / {f1:.0f}"


def build_table() -> str:
    lines = [
        "| 構成 | 1-hop Acc/F1 | 2-hop Acc/F1 | 3-hop Acc/F1 | Δ 2-hop vs C |",
        "|---|---|---|---|---|",
    ]
    c_2hop = None
    for label, path, ds_list in ROWS:
        cells = {}
        for ds in DATASETS:
            if ds not in ds_list:
                cells[ds] = "—"
                continue
            p = os.path.join(path, f"extended_type_kopl_{ds}.jsonl")
            s = score(p)
            cells[ds] = fmt_cell(s)
            if label.startswith("C:") and ds == "2hop" and s is not None:
                c_2hop = s[1]
        delta = ""
        if "2hop" in ds_list and c_2hop is not None:
            d_s = score(os.path.join(path, "extended_type_kopl_2hop.jsonl"))
            if d_s is not None:
                d = d_s[1] - c_2hop
                delta = f"{d:+.0f}"
            else:
                delta = "_pending_"
        else:
            delta = "_pending_" if c_2hop is None else "—"
        lines.append(
            f"| {label} | {cells['1hop']} | {cells['2hop']} | {cells['3hop']} | {delta} |"
        )
    return "\n".join(lines)


def update_report() -> None:
    if not os.path.exists(REPORT):
        print(f"Report not found: {REPORT}")
        return
    with open(REPORT) as f:
        content = f.read()
    new_table = build_table()
    # Replace the result table between the header and the next ## section
    pattern = re.compile(
        r"### MetaQA \(gemma3:27b, n=100, seed=42\)\n\n(\|.*?\n(?:\|.*?\n)+)",
        re.DOTALL,
    )
    if not pattern.search(content):
        print("Result table pattern not found; writing table to stdout instead")
        print(new_table)
        return
    new_content = pattern.sub(
        f"### MetaQA (gemma3:27b, n=100, seed=42)\n\n{new_table}\n",
        content,
    )
    with open(REPORT, "w") as f:
        f.write(new_content)
    print(f"Updated {REPORT}")
    print(new_table)


if __name__ == "__main__":
    update_report()
