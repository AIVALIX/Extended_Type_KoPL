from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import random


DEFAULT_QUERY_FILES: Tuple[Tuple[str, str], ...] = (
    ("one_hop_chain", "one_hop_chain.jsonl"),
    ("two_hop_chain", "two_hop_chain.jsonl"),
    ("two_anchor_intersection", "two_anchor_intersection.jsonl"),
    ("three_anchor_intersection", "three_anchor_intersection.jsonl"),
)


@dataclass(frozen=True)
class SplitSpec:
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def reservoir_sample(
    rows: Iterable[Dict[str, Any]],
    *,
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Uniformly sample k items from an iterable without loading all items."""

    if k <= 0:
        return []

    sample: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        if i <= k:
            sample.append(row)
            continue
        j = rng.randrange(1, i + 1)
        if j <= k:
            sample[j - 1] = row
    return sample


def split_rows(
    rows: List[Dict[str, Any]], *, split: SplitSpec, rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(rows) < split.total:
        raise ValueError(
            f"not enough rows to split: have={len(rows)} need={split.total}"
        )

    rng.shuffle(rows)
    train_rows = rows[: split.train]
    val_rows = rows[split.train : split.train + split.val]
    test_rows = rows[split.train + split.val : split.total]
    return train_rows, val_rows, test_rows


def _add_query_type(row: Dict[str, Any], query_type: str) -> Dict[str, Any]:
    if "query_type" in row:
        return row
    return {"query_type": query_type, **row}


def process_one_file(
    *,
    query_type: str,
    in_path: Path,
    out_dir: Path,
    sample_size: int,
    split: SplitSpec,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    rows = reservoir_sample(_iter_jsonl(in_path), k=sample_size, rng=rng)
    if len(rows) < split.total:
        raise ValueError(
            f"{in_path.name}: sampled {len(rows)} rows, but split requires {split.total}. "
            "Increase generation, reduce sample/split sizes, or check the input file."
        )

    train_rows, val_rows, test_rows = split_rows(rows, split=split, rng=rng)

    q_out = out_dir / query_type
    train_path = q_out / "train.jsonl"
    val_path = q_out / "val.jsonl"
    test_path = q_out / "test.jsonl"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_jsonl(test_path, test_rows)

    meta = {
        "query_type": query_type,
        "input": str(in_path),
        "seed": seed,
        "sample_size": sample_size,
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "written": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
    }
    (q_out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def process_all(
    *,
    sources: List[Tuple[str, Path]],
    out_dir: Path,
    sample_size: int,
    split: SplitSpec,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    def iter_union() -> Iterator[Dict[str, Any]]:
        for query_type, path in sources:
            for row in _iter_jsonl(path):
                yield _add_query_type(row, query_type)

    rows = reservoir_sample(iter_union(), k=sample_size, rng=rng)
    if len(rows) < split.total:
        raise ValueError(
            f"all: sampled {len(rows)} rows, but split requires {split.total}. "
            "Increase generation, reduce sample/split sizes, or check the input files."
        )

    train_rows, val_rows, test_rows = split_rows(rows, split=split, rng=rng)

    a_out = out_dir / "all"
    train_path = a_out / "train.jsonl"
    val_path = a_out / "val.jsonl"
    test_path = a_out / "test.jsonl"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    _write_jsonl(test_path, test_rows)

    meta = {
        "query_type": "all",
        "inputs": [str(p) for _, p in sources],
        "seed": seed,
        "sample_size": sample_size,
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "written": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "note": "all is sampled from the union of per-query files; query_type is injected if missing",
    }
    (a_out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample N rows from generated dataset_construction JSONL files and split into train/val/test."
        )
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path("/app/result") / "dataset_construction",
        help="Input directory containing per-query JSONL files (default: /app/result/dataset_construction)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/app/result") / "dataset_splits",
        help="Output directory (default: /app/result/dataset_splits)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=15_000,
        help="Rows to sample per dataset (default: 15000)",
    )
    parser.add_argument(
        "--train", type=int, default=10_000, help="Train size (default: 10000)"
    )
    parser.add_argument(
        "--val", type=int, default=2_500, help="Val size (default: 2500)"
    )
    parser.add_argument(
        "--test", type=int, default=2_500, help="Test size (default: 2500)"
    )
    args = parser.parse_args()

    split = SplitSpec(train=args.train, val=args.val, test=args.test)
    if split.total != args.sample_size:
        raise SystemExit(
            f"train+val+test must equal sample-size (got {split.total} vs {args.sample_size})"
        )

    sources: List[Tuple[str, Path]] = []
    for query_type, filename in DEFAULT_QUERY_FILES:
        p = args.in_dir / filename
        if not p.exists():
            raise SystemExit(f"missing input file: {p}")
        sources.append((query_type, p))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metas: List[Dict[str, Any]] = []
    for query_type, path in sources:
        metas.append(
            process_one_file(
                query_type=query_type,
                in_path=path,
                out_dir=args.out_dir,
                sample_size=args.sample_size,
                split=split,
                seed=args.seed,
            )
        )

    metas.append(
        process_all(
            sources=sources,
            out_dir=args.out_dir,
            sample_size=args.sample_size,
            split=split,
            seed=args.seed,
        )
    )

    summary = {
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "sample_size": args.sample_size,
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "outputs": metas,
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote splits to: {args.out_dir}")
    for m in metas:
        qt = m["query_type"]
        w = m["written"]
        print(f"  {qt}: train={w['train']} val={w['val']} test={w['test']}")


if __name__ == "__main__":
    main()
