#!/usr/bin/env python3
"""
Merge multi-GPU results and run analytical experiments.
Run after both GPU 2 and GPU 3 finish.

Usage:
    python scripts/merge_and_analyze.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DRIFT_DIR = PROJECT_ROOT / "results" / "drift_curves"


def merge_results():
    print("=" * 60)
    print("MERGING MULTI-GPU RESULTS")
    print("=" * 60)

    # Merge turn-level results
    turn_files = sorted(DRIFT_DIR.glob("drift_results_gpu*.csv"))
    if turn_files:
        dfs = [pd.read_csv(f) for f in turn_files]
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(DRIFT_DIR / "drift_results.csv", index=False)
        print(f"  Merged {len(turn_files)} turn files -> {len(merged)} rows")

    # Merge summary results
    summary_files = sorted(DRIFT_DIR.glob("drift_summary_gpu*.csv"))
    if summary_files:
        dfs = [pd.read_csv(f) for f in summary_files]
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(DRIFT_DIR / "drift_summary.csv", index=False)
        print(f"  Merged {len(summary_files)} summary files -> {len(merged)} rows")

    # Merge context lengths
    ctx_files = sorted(DRIFT_DIR.glob("context_lengths_gpu*.json"))
    merged_ctx = {}
    for f in ctx_files:
        with open(f) as fh:
            merged_ctx.update(json.load(fh))
    if merged_ctx:
        with open(DRIFT_DIR / "context_lengths.json", "w") as f:
            json.dump(merged_ctx, f)
        print(f"  Merged {len(ctx_files)} context files -> {len(merged_ctx)} entries")

    # Merge per-level decay
    level_files = sorted(DRIFT_DIR.glob("per_level_decay_gpu*.json"))
    merged_levels = {}
    for f in level_files:
        with open(f) as fh:
            merged_levels.update(json.load(fh))
    if merged_levels:
        with open(DRIFT_DIR / "per_level_decay.json", "w") as f:
            json.dump(merged_levels, f, default=str)
        print(f"  Merged {len(level_files)} level files -> {len(merged_levels)} entries")

    print("  ✅ Merge complete\n")


def run_analytical():
    """Run all analytical experiments on merged data."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.experiments.exp9_drift_velocity import run_experiment_9
    from src.experiments.exp10_cross_model import run_experiment_10
    from src.experiments.exp11_tier_effect import run_experiment_11
    from src.experiments.exp12_level_ordering import run_experiment_12
    from src.experiments.exp7_context_budget import run_experiment_7

    results_root = PROJECT_ROOT / "results"

    print("=" * 60)
    print("RUNNING ANALYTICAL EXPERIMENTS")
    print("=" * 60)

    # Exp 4: Regression
    try:
        from src.experiments.runner import ExperimentRunner
        runner = ExperimentRunner()
        runner.run_experiment_4()
    except Exception as e:
        print(f"  Exp 4 (Regression): {e}")

    # Exp 7: Context Budget
    try:
        import yaml
        with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
            config = yaml.safe_load(f)
        run_experiment_7(
            str(DRIFT_DIR / "drift_results.csv"),
            config["models"],
            str(results_root / "context_budget"),
            str(DRIFT_DIR / "context_lengths.json"),
        )
    except Exception as e:
        print(f"  Exp 7 (Context Budget): {e}")

    # Exp 9: Drift Velocity
    try:
        run_experiment_9(
            str(DRIFT_DIR / "drift_results.csv"),
            str(results_root / "drift_velocity"),
        )
    except Exception as e:
        print(f"  Exp 9 (Velocity): {e}")

    # Exp 10: Cross-Model
    try:
        run_experiment_10(
            str(DRIFT_DIR / "drift_summary.csv"),
            str(results_root / "cross_model"),
        )
    except Exception as e:
        print(f"  Exp 10 (Cross-Model): {e}")

    # Exp 11: Tier Effect
    try:
        run_experiment_11(
            str(DRIFT_DIR / "drift_summary.csv"),
            str(results_root / "tier_effect"),
        )
    except Exception as e:
        print(f"  Exp 11 (Tier Effect): {e}")

    # Exp 12: Level Ordering
    try:
        run_experiment_12(
            str(DRIFT_DIR / "drift_summary.csv"),
            str(DRIFT_DIR / "per_level_decay.json"),
            str(results_root / "level_ordering"),
        )
    except Exception as e:
        print(f"  Exp 12 (Level Ordering): {e}")

    print("\n✅ ALL ANALYTICAL EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    merge_results()
    run_analytical()
