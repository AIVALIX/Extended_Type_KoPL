"""
Ablation Study Script for Extended Type-KoPL (ETK) Pipeline

Runs the ETK pipeline under multiple configurations to quantify
the contribution of each component (fallbacks, path verification).

Usage:
  # Inside Docker container
  python -m app.pipeline.run_ablation --kg primekgqa --dataset one_hop two_hop --num-samples 100 --random

  # From host
  docker exec -it python-primekgqa-experiment python -m app.pipeline.run_ablation \
    --kg primekgqa --dataset one_hop two_hop --num-samples 100 --random --seed 42

Configurations evaluated:
  1. full          - All fallbacks enabled (default production config)
  2. no_fallback2  - Disable _try_direct_paths (Fallback 2)
  3. no_fallbacks  - Disable all fallbacks (primary path selection only)
  4. beam3_verify  - beam_width=3 with _verify_path enabled
  5. beam3_noverify- beam_width=3 without _verify_path

Note: Configurations 2 and 3 require the --no-fallback and --no-fallback2 flags
to be implemented in pipeline.py. See TODO comments below.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AblationConfig:
    """A single ablation configuration."""
    name: str
    description: str
    extra_flags: List[str]
    requires_pipeline_changes: bool = False
    todo_note: Optional[str] = None


# Define ablation configurations
ABLATION_CONFIGS = [
    AblationConfig(
        name="full",
        description="All fallbacks enabled (production default)",
        extra_flags=[],
    ),
    AblationConfig(
        name="no_fallback2",
        description="Disable Fallback 2 (_try_direct_paths brute-force 1-hop search)",
        extra_flags=["--no-fallback2"],
        requires_pipeline_changes=True,
        todo_note=(
            "TODO: Add --no-fallback2 flag to run_evaluation.py argparse, "
            "and pass it to ExtendedTypeKoPLPipeline.__init__() as "
            "disable_fallback2=True. In pipeline.py, guard the "
            "_try_direct_paths calls at lines 1700-1714 and 1804-1816 "
            "with `if not self.disable_fallback2:`."
        ),
    ),
    AblationConfig(
        name="no_fallbacks",
        description="Disable all fallbacks (primary path selection only)",
        extra_flags=["--no-fallbacks"],
        requires_pipeline_changes=True,
        todo_note=(
            "TODO: Add --no-fallbacks flag to run_evaluation.py argparse, "
            "and pass it to ExtendedTypeKoPLPipeline.__init__() as "
            "disable_all_fallbacks=True. In pipeline.py, guard:\n"
            "  - Fallback 1 (lines 1675-1698): alternative candidate paths\n"
            "  - Fallback 2 (lines 1700-1714): _try_direct_paths_with_properties\n"
            "  - Intersection child fallback (lines 1804-1816): _try_direct_paths\n"
            "with `if not self.disable_all_fallbacks:`."
        ),
    ),
    AblationConfig(
        name="beam3_verify",
        description="beam_width=3 with _verify_path enabled (default when beam_width>1)",
        extra_flags=["--beam-width", "3"],
    ),
    AblationConfig(
        name="beam3_noverify",
        description="beam_width=3 without _verify_path",
        extra_flags=["--beam-width", "3", "--no-verify-path"],
        requires_pipeline_changes=True,
        todo_note=(
            "TODO: Add --no-verify-path flag to run_evaluation.py argparse, "
            "and pass it to ExtendedTypeKoPLPipeline.__init__() as "
            "disable_verify_path=True. In pipeline.py, change the condition "
            "at lines 1633-1634 from:\n"
            "  if self.beam_width > 1 and len(selected_paths) > 1:\n"
            "to:\n"
            "  if self.beam_width > 1 and len(selected_paths) > 1 "
            "and not self.disable_verify_path:\n"
            "Same change at line 1789."
        ),
    ),
]


def build_eval_command(
    kg: str,
    datasets: List[str],
    num_samples: int,
    random: bool,
    seed: Optional[int],
    workers: int,
    extra_flags: List[str],
    output_dir: Optional[Path],
    output_file: Optional[Path],
    reranker: str,
    no_schema: bool,
) -> List[str]:
    """Build the run_evaluation.py command for a given configuration."""
    cmd = [
        sys.executable, "-m", "app.pipeline.run_evaluation",
        "--kg", kg,
        "--pipeline", "extended_type_kopl",
        "--num-samples", str(num_samples),
        "--workers", str(workers),
        "--reranker", reranker,
    ]
    if datasets:
        cmd.extend(["--dataset"] + datasets)
    if random:
        cmd.append("--random")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if no_schema:
        cmd.append("--no-schema")
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    if output_file:
        cmd.extend(["--output", str(output_file)])

    cmd.extend(extra_flags)
    return cmd


def run_single_ablation(
    config: AblationConfig,
    kg: str,
    datasets: List[str],
    num_samples: int,
    random: bool,
    seed: Optional[int],
    workers: int,
    base_output_dir: Path,
    reranker: str,
    no_schema: bool,
    dry_run: bool,
) -> Optional[Path]:
    """Run a single ablation configuration and return the output file path."""
    config_dir = base_output_dir / config.name
    config_dir.mkdir(parents=True, exist_ok=True)
    output_file = config_dir / "summary.json"

    if config.requires_pipeline_changes:
        print(f"\n{'='*70}")
        print(f"SKIPPING: {config.name} - {config.description}")
        print(f"  Reason: Requires pipeline.py changes not yet implemented.")
        if config.todo_note:
            print(f"\n  {config.todo_note}")
        print(f"{'='*70}")
        return None

    cmd = build_eval_command(
        kg=kg,
        datasets=datasets,
        num_samples=num_samples,
        random=random,
        seed=seed,
        workers=workers,
        extra_flags=config.extra_flags,
        output_dir=config_dir,
        output_file=output_file,
        reranker=reranker,
        no_schema=no_schema,
    )

    print(f"\n{'='*70}")
    print(f"Running: {config.name} - {config.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output:  {output_file}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping actual execution.")
        return None

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"  WARNING: {config.name} exited with code {result.returncode}")
            return None
        return output_file if output_file.exists() else None
    except Exception as e:
        print(f"  ERROR: {config.name} failed: {e}")
        return None


def load_results(output_file: Path) -> Optional[Dict]:
    """Load evaluation results from a summary JSON file."""
    try:
        with open(output_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Could not load {output_file}: {e}")
        return None


def print_comparison_table(results: Dict[str, Optional[Dict]], datasets: List[str]):
    """Print a formatted comparison table of ablation results."""
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}")

    # Determine which datasets have results
    all_datasets = set()
    for config_name, data in results.items():
        if data and "results" in data:
            all_datasets.update(data["results"].keys())

    if datasets:
        display_datasets = [d for d in datasets if d in all_datasets]
    else:
        display_datasets = sorted(all_datasets)

    if not display_datasets:
        print("No results to display.")
        return

    # Header
    config_col_width = 20
    dataset_col_width = 15
    header = f"{'Configuration':<{config_col_width}}"
    for ds in display_datasets:
        header += f" | {ds:>{dataset_col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for config in ABLATION_CONFIGS:
        data = results.get(config.name)
        if data is None:
            row = f"{config.name:<{config_col_width}}"
            for ds in display_datasets:
                row += f" | {'(skipped)':>{dataset_col_width}}"
            print(row)
            continue

        row = f"{config.name:<{config_col_width}}"
        for ds in display_datasets:
            if "results" in data and ds in data["results"]:
                ds_results = data["results"][ds]
                # Try to extract accuracy/F1
                if "extended_type_kopl" in ds_results:
                    etk = ds_results["extended_type_kopl"]
                    acc = etk.get("accuracy", etk.get("f1", None))
                    if acc is not None:
                        row += f" | {acc:>{dataset_col_width}.1f}%"
                    else:
                        row += f" | {'N/A':>{dataset_col_width}}"
                else:
                    row += f" | {'N/A':>{dataset_col_width}}"
            else:
                row += f" | {'--':>{dataset_col_width}}"
        print(row)

    print(f"\n{'='*80}")

    # Print configuration descriptions
    print("\nConfiguration descriptions:")
    for config in ABLATION_CONFIGS:
        status = "(requires pipeline changes)" if config.requires_pipeline_changes else ""
        print(f"  {config.name:<20} {config.description} {status}")


def main():
    p = argparse.ArgumentParser(
        description="Run ablation study for ETK pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the ETK pipeline under multiple configurations
and produces a comparison table. Some configurations require
additional flags to be implemented in pipeline.py (see TODO notes).

Example:
  python -m app.pipeline.run_ablation \\
    --kg primekgqa --dataset one_hop two_hop \\
    --num-samples 100 --random --seed 42 --workers 4
        """,
    )
    p.add_argument(
        "--kg", type=str, default="primekgqa",
        choices=["primekgqa", "primekgqa_raw", "metaqa", "pcqa"],
        help="Knowledge graph to evaluate (default: primekgqa)",
    )
    p.add_argument(
        "--dataset", type=str, nargs="+", default=None,
        help="Specific dataset(s) to evaluate (default: all for the KG)",
    )
    p.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of samples per dataset (default: 100)",
    )
    p.add_argument(
        "--random", action="store_true",
        help="Randomly sample from dataset",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("/app/result/ablation"),
        help="Base output directory (default: /app/result/ablation)",
    )
    p.add_argument(
        "--reranker", type=str, default="none",
        choices=["none", "llm", "hybrid"],
        help="Reranker type (default: none)",
    )
    p.add_argument(
        "--no-schema", action="store_true",
        help="Run without schema relations",
    )
    p.add_argument(
        "--configs", type=str, nargs="+", default=None,
        help="Specific ablation configs to run (default: all). "
             f"Choices: {[c.name for c in ABLATION_CONFIGS]}",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    p.add_argument(
        "--results-only", action="store_true",
        help="Skip execution; only load and display existing results",
    )
    args = p.parse_args()

    # Select configs to run
    if args.configs:
        configs = [c for c in ABLATION_CONFIGS if c.name in args.configs]
        if not configs:
            print(f"Error: No matching configs. Available: {[c.name for c in ABLATION_CONFIGS]}")
            sys.exit(1)
    else:
        configs = ABLATION_CONFIGS

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Print TODO notes for unimplemented configurations
    unimplemented = [c for c in configs if c.requires_pipeline_changes]
    if unimplemented:
        print("\n" + "=" * 70)
        print("IMPLEMENTATION NOTES")
        print("The following configurations require changes to pipeline.py:")
        print("=" * 70)
        for c in unimplemented:
            print(f"\n  [{c.name}] {c.description}")
            if c.todo_note:
                for line in c.todo_note.split("\n"):
                    print(f"    {line}")
        print()

    # Run evaluations
    results: Dict[str, Optional[Dict]] = {}

    if not args.results_only:
        for config in configs:
            output_file = run_single_ablation(
                config=config,
                kg=args.kg,
                datasets=args.dataset or [],
                num_samples=args.num_samples,
                random=args.random,
                seed=args.seed,
                workers=args.workers,
                base_output_dir=args.output_dir,
                reranker=args.reranker,
                no_schema=args.no_schema,
                dry_run=args.dry_run,
            )
            if output_file:
                results[config.name] = load_results(output_file)
            else:
                results[config.name] = None
    else:
        # Load existing results
        for config in configs:
            output_file = args.output_dir / config.name / "summary.json"
            if output_file.exists():
                results[config.name] = load_results(output_file)
            else:
                results[config.name] = None

    # Print comparison table
    print_comparison_table(results, args.dataset or [])

    # Save combined results
    combined_output = args.output_dir / "ablation_summary.json"
    serializable = {}
    for name, data in results.items():
        if data is not None:
            serializable[name] = data
    if serializable:
        with open(combined_output, "w") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\nCombined results saved to: {combined_output}")


if __name__ == "__main__":
    main()
