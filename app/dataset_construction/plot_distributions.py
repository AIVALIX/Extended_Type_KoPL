from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import matplotlib

# Headless backend for Docker
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_paths(in_dir: Path, dataset: str, split: str) -> List[Path]:
    base = in_dir / dataset
    if not base.exists():
        raise SystemExit(f"dataset folder not found: {base}")

    if split == "all":
        paths = [base / "train.jsonl", base / "val.jsonl", base / "test.jsonl"]
    else:
        paths = [base / f"{split}.jsonl"]

    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing split files: {', '.join(str(p) for p in missing)}")

    return paths


def _extract_anchor_types(row: Dict[str, Any]) -> Iterable[str]:
    # one/two hop
    if "anchor_types" in row and isinstance(row["anchor_types"], list):
        yield from row["anchor_types"]

    # intersections
    for k in ("anchorA_types", "anchorB_types", "anchorC_types"):
        v = row.get(k)
        if isinstance(v, list):
            yield from v


def _extract_rel_types(row: Dict[str, Any]) -> Iterable[str]:
    # hop
    if isinstance(row.get("rel_type"), str):
        yield row["rel_type"]
    if isinstance(row.get("rel1"), str):
        yield row["rel1"]
    if isinstance(row.get("rel2"), str):
        yield row["rel2"]

    # intersections
    for k in ("anchorA_edge_type", "anchorB_edge_type", "anchorC_edge_type"):
        v = row.get(k)
        if isinstance(v, str):
            yield v


def _plot_top_bar(
    counts: Counter[str],
    *,
    title: str,
    out_path: Path,
    top_n: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = counts.most_common(top_n)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(max(10, int(top_n * 0.45)), 6))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot anchor type distribution and relation type distribution from dataset_splits JSONL."
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path("/app/result") / "dataset_splits",
        help="Input directory containing split datasets (default: /app/result/dataset_splits)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset subfolder name (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="train",
        help="Which split to plot (default: train). Use 'all' to combine train/val/test.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/app/result") / "dataset_plots",
        help="Output directory for plots (default: /app/result/dataset_plots)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Top N categories to display (default: 30)",
    )
    args = parser.parse_args()

    paths = _iter_paths(args.in_dir, args.dataset, args.split)

    anchor_counts: Counter[str] = Counter()
    rel_counts: Counter[str] = Counter()

    for p in paths:
        for row in _iter_jsonl(p):
            anchor_counts.update(_extract_anchor_types(row))
            rel_counts.update(_extract_rel_types(row))

    prefix = f"{args.dataset}_{args.split}"
    anchor_png = args.out_dir / f"{prefix}_anchor_types.png"
    rel_png = args.out_dir / f"{prefix}_relation_types.png"

    _plot_top_bar(
        anchor_counts,
        title=f"Anchor types (top {args.top_n}) - {args.dataset}/{args.split}",
        out_path=anchor_png,
        top_n=args.top_n,
    )
    _plot_top_bar(
        rel_counts,
        title=f"Relation types (top {args.top_n}) - {args.dataset}/{args.split}",
        out_path=rel_png,
        top_n=args.top_n,
    )

    print(f"Wrote: {anchor_png}")
    print(f"Wrote: {rel_png}")


if __name__ == "__main__":
    main()
