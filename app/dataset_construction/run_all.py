from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end runner: generate dataset_construction JSONL from Neo4j, "
            "sample/split into train/val/test, then plot anchor/rel distributions."
        )
    )

    # Shared
    parser.add_argument("--seed", type=int, default=42)

    # Generate
    parser.add_argument(
        "--generate-out-dir",
        type=Path,
        default=Path("/app/result") / "dataset_construction",
    )
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--node-label", type=str, default="auto")
    parser.add_argument("--limit-per-query", type=int, default=100_000)
    parser.add_argument("--onehop-anchor-pool", type=int, default=50_000)
    parser.add_argument("--twohop-anchor-pool", type=int, default=50_000)
    parser.add_argument("--two-anchor-pool", type=int, default=10_000)
    parser.add_argument("--two-anchor-tries", type=int, default=200_000)
    parser.add_argument("--three-anchor-pool", type=int, default=10_000)
    parser.add_argument("--three-anchor-tries", type=int, default=800_000)

    # Split
    parser.add_argument(
        "--splits-out-dir",
        type=Path,
        default=Path("/app/result") / "dataset_splits",
    )
    parser.add_argument("--sample-size", type=int, default=15_000)
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--val", type=int, default=2_500)
    parser.add_argument("--test", type=int, default=2_500)

    # Plot
    parser.add_argument(
        "--plots-out-dir",
        type=Path,
        default=Path("/app/result") / "dataset_plots",
    )
    parser.add_argument("--top-n", type=int, default=30)

    # Optional skips (useful for re-running only split/plot)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")

    args = parser.parse_args()

    if args.train + args.val + args.test != args.sample_size:
        raise SystemExit(
            f"train+val+test must equal sample-size (got {args.train + args.val + args.test} vs {args.sample_size})"
        )

    generate_py = Path(__file__).with_name("generate_dataset.py")
    split_py = Path(__file__).with_name("split_dataset.py")
    plot_py = Path(__file__).with_name("plot_distributions.py")

    if not args.skip_generate:
        cmd = [
            sys.executable,
            str(generate_py),
            "--out-dir",
            str(args.generate_out_dir),
            "--seed",
            str(args.seed),
            "--node-label",
            str(args.node_label),
            "--limit-per-query",
            str(args.limit_per_query),
            "--onehop-anchor-pool",
            str(args.onehop_anchor_pool),
            "--twohop-anchor-pool",
            str(args.twohop_anchor_pool),
            "--two-anchor-pool",
            str(args.two_anchor_pool),
            "--two-anchor-tries",
            str(args.two_anchor_tries),
            "--three-anchor-pool",
            str(args.three_anchor_pool),
            "--three-anchor-tries",
            str(args.three_anchor_tries),
        ]
        if args.simple:
            cmd.append("--simple")
        _run(cmd)

    if not args.skip_split:
        _run(
            [
                sys.executable,
                str(split_py),
                "--in-dir",
                str(args.generate_out_dir),
                "--out-dir",
                str(args.splits_out_dir),
                "--seed",
                str(args.seed),
                "--sample-size",
                str(args.sample_size),
                "--train",
                str(args.train),
                "--val",
                str(args.val),
                "--test",
                str(args.test),
            ]
        )

    if not args.skip_plot:
        datasets = [
            "one_hop_chain",
            "two_hop_chain",
            "two_anchor_intersection",
            "three_anchor_intersection",
            "all",
        ]
        for ds in datasets:
            _run(
                [
                    sys.executable,
                    str(plot_py),
                    "--in-dir",
                    str(args.splits_out_dir),
                    "--dataset",
                    ds,
                    "--split",
                    "all",
                    "--out-dir",
                    str(args.plots_out_dir),
                    "--top-n",
                    str(args.top_n),
                ]
            )


if __name__ == "__main__":
    main()
