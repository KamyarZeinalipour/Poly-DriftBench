#!/usr/bin/env python3
"""
Full Production Experiment — All 13 Experiments
=================================================
Runs the complete Token Squeeze Hypothesis pipeline:
    Phase 1: Token Fertility Profiling (CPU)
    Phase 2: GPU Inference (Exp 2, 3, 5, 6, 7, 8, 13)
    Phase 3: Analytical (Exp 4, 9, 10, 11, 12)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_full_experiment.py
    
    # Dry run (2 conversations per tier):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_full_experiment.py --dry-run
    
    # Skip GPU experiments (analytical only, uses existing data):
    python scripts/run_full_experiment.py --skip-gpu
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ── Environment Setup ──
os.environ.setdefault("HF_HUB_CACHE", "/home4/kamyar/italian_detox/hf_cache/hub")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ──
log_file = PROJECT_ROOT / f"experiment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger("experiment")


def main():
    parser = argparse.ArgumentParser(description="Run Poly-DriftBench experiments")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run with 2 conversations per tier",
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip GPU experiments (run analytical only on existing data)",
    )
    parser.add_argument(
        "--data-dir", default=str(PROJECT_ROOT / "data" / "production"),
        help="Data directory",
    )
    parser.add_argument(
        "--max-convs", type=int, default=None,
        help="Max conversations per tier (overrides --dry-run)",
    )
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=None,
        help="Run specific experiments only (e.g., --experiments 2 9 10)",
    )
    args = parser.parse_args()

    max_convs = args.max_convs
    if args.dry_run and max_convs is None:
        max_convs = 2

    logger.info(f"{'=' * 70}")
    logger.info(f"POLY-DRIFTBENCH — Full Experiment Pipeline")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Data directory:  {args.data_dir}")
    logger.info(f"  Output:          {PROJECT_ROOT / 'results'}")
    logger.info(f"  Mode:            {'DRY RUN' if args.dry_run else 'FULL'}")
    logger.info(f"  GPU experiments: {'SKIPPED' if args.skip_gpu else 'ENABLED'}")
    logger.info(f"  Log file:        {log_file}")
    logger.info(f"  Max convs/tier:  {max_convs or 'all'}")
    logger.info(f"{'=' * 70}\n")

    from src.experiments.runner import ExperimentRunner

    runner = ExperimentRunner()

    if args.experiments:
        # Run specific experiments
        logger.info(f"Running selected experiments: {args.experiments}")
        
        exp_map = {
            1: lambda: runner.run_experiment_1(args.data_dir),
            2: lambda: runner.run_experiment_2(
                args.data_dir, max_conversations_per_tier=max_convs,
            ),
            3: lambda: runner.run_experiment_3(args.data_dir),
            4: lambda: runner.run_experiment_4(),
            5: lambda: runner.run_experiment_5(args.data_dir, max_conversations=max_convs or 10),
            6: lambda: runner.run_experiment_6(args.data_dir, max_conversations=max_convs or 10),
            7: lambda: runner.run_experiment_7(),
            9: lambda: runner.run_experiment_9(),
            10: lambda: runner.run_experiment_10(),
            11: lambda: runner.run_experiment_11(),
            12: lambda: runner.run_experiment_12(),
        }
        
        for exp_num in args.experiments:
            if exp_num in exp_map:
                try:
                    exp_map[exp_num]()
                except Exception as e:
                    logger.error(f"Experiment {exp_num} failed: {e}", exc_info=True)
            else:
                logger.warning(f"Experiment {exp_num} not found in quick-run map")
    else:
        # Run full pipeline
        runner.run_all(
            data_dir=args.data_dir,
            skip_gpu=args.skip_gpu,
            max_conversations=max_convs,
        )

    logger.info(f"\n  Log saved to: {log_file}")


if __name__ == "__main__":
    main()
