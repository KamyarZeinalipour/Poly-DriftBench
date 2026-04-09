#!/usr/bin/env python3
"""
Regenerate English-Only Conversations (Phase 1)
=================================================
Regenerates the Poly-DriftBench dataset with improved pipeline fixes:
  - Rolling context summary (prevents hallucinations in long convs)
  - Strict L4 [Source:] validation 
  - Deterministic DDM force-fix
  - Coherence + topic repetition checks
  - AI-chat framing (no phone greetings)

Translation (Phase 2) is skipped — run separately after QA.

Usage:
    # Full production run (25 short + 50 medium + 25 long):
    python scripts/regenerate_en.py

    # Dry run (2 per tier):
    python scripts/regenerate_en.py --dry-run

    # Custom counts:
    python scripts/regenerate_en.py --short 5 --medium 10 --long 5
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime

# ── Environment ──
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ──
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = PROJECT_ROOT / f"regenerate_en_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger("regenerate_en")


# ── Tier Definitions (matching production_summary.json) ──
TIERS = {
    "short": {
        "description": "Control — no drift expected",
        "num_conversations": 25,
        "min_turns": 10,
        "max_turns": 15,
    },
    "medium": {
        "description": "Drift onset — moderate context",
        "num_conversations": 50,
        "min_turns": 30,
        "max_turns": 50,
    },
    "long": {
        "description": "Deep drift — long context stress test",
        "num_conversations": 25,
        "min_turns": 80,
        "max_turns": 120,
    },
}


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Regenerate English conversations")
    parser.add_argument("--dry-run", action="store_true", help="2 convs per tier")
    parser.add_argument("--short", type=int, default=None, help="Num short conversations")
    parser.add_argument("--medium", type=int, default=None, help="Num medium conversations")
    parser.add_argument("--long", type=int, default=None, help="Num long conversations")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel conversations")
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "data" / "production"),
        help="Output directory",
    )
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override tier counts
    if args.dry_run:
        for tier in TIERS:
            TIERS[tier]["num_conversations"] = 2
    if args.short is not None:
        TIERS["short"]["num_conversations"] = args.short
    if args.medium is not None:
        TIERS["medium"]["num_conversations"] = args.medium
    if args.long is not None:
        TIERS["long"]["num_conversations"] = args.long

    total = sum(t["num_conversations"] for t in TIERS.values())
    logger.info("=" * 70)
    logger.info("POLY-DRIFTBENCH — English Regeneration (NO Translation)")
    logger.info("=" * 70)
    for tier_name, tier_cfg in TIERS.items():
        logger.info(
            f"  {tier_name:8s}: {tier_cfg['num_conversations']} convs "
            f"({tier_cfg['min_turns']}-{tier_cfg['max_turns']} turns) "
            f"— {tier_cfg['description']}"
        )
    logger.info(f"  Total:    {total} conversations")
    logger.info(f"  Output:   {args.output_dir}")
    logger.info(f"  Parallel: {args.parallel}")
    logger.info(f"  Log:      {log_file}")
    logger.info("=" * 70 + "\n")

    # Initialize pipeline (translation disabled)
    from src.data_gen.pipeline import DataFactory

    factory = DataFactory(config)

    start_time = time.time()
    all_results = {}

    for tier_name, tier_cfg in TIERS.items():
        tier_start = time.time()
        n = tier_cfg["num_conversations"]
        logger.info(f"\n{'─' * 50}")
        logger.info(f"TIER: {tier_name.upper()} — {n} conversations")
        logger.info(f"{'─' * 50}")

        output_dir = Path(args.output_dir) / tier_name
        saved = factory.produce_dataset(
            output_dir=output_dir,
            num_conversations=n,
            min_turns=tier_cfg["min_turns"],
            max_turns=tier_cfg["max_turns"],
            translate=False,  # ← English only!
            parallel_conversations=args.parallel,
        )

        tier_elapsed = time.time() - tier_start
        all_results[tier_name] = {
            "conversations_generated": len(saved),
            "conversations_requested": n,
            "time_seconds": round(tier_elapsed, 1),
            "avg_time_per_conv": round(tier_elapsed / max(len(saved), 1), 1),
            "files": [str(p) for p in saved],
        }
        logger.info(
            f"  ✅ {tier_name}: {len(saved)}/{n} generated in {tier_elapsed:.0f}s"
        )

    total_elapsed = time.time() - start_time

    # Save summary
    summary = {
        "regeneration_config": {
            "tiers": [
                {"name": k, **v} for k, v in TIERS.items()
            ],
            "parallel_conversations": args.parallel,
            "model": factory.model,
            "start_time": datetime.utcnow().isoformat() + "Z",
            "total_time_seconds": round(total_elapsed, 1),
            "total_conversations": total,
            "translation": False,
            "pipeline_version": "v7-catd-l5-dual-track",
        },
        "tier_results": all_results,
    }

    summary_path = Path(args.output_dir) / "regeneration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"DONE — {total} conversations in {total_elapsed:.0f}s")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"Log:     {log_file}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
