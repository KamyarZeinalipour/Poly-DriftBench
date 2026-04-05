#!/usr/bin/env python3
"""
Poly-DriftBench Production — 3-Tier Length Strategy
=====================================================

Generates 100 conversations across 3 context-length tiers
to capture the full drift onset curve for paper figures.

Tiers:
    Short  (25 convs): 10-15 turns → ~5K-8K tokens  (control)
    Medium (50 convs): 30-50 turns → 12K-20K tokens (drift onset)
    Long   (25 convs): 80-120 turns → 30K-50K tokens (deep drift)

Usage:
    python scripts/run_production.py
    python scripts/run_production.py --parallel 5
    python scripts/run_production.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data_gen.pipeline import DataFactory


# ──────────────────────────────────────────────────────────
# 3-Tier Configuration
# ──────────────────────────────────────────────────────────

TIERS = [
    {
        "name": "short",
        "description": "Control — no drift expected",
        "num_conversations": 25,
        "min_turns": 10,
        "max_turns": 15,
        "context_range": "~5K-8K tokens",
    },
    {
        "name": "medium",
        "description": "Drift onset — moderate context",
        "num_conversations": 50,
        "min_turns": 30,
        "max_turns": 50,
        "context_range": "~12K-20K tokens",
    },
    {
        "name": "long",
        "description": "Deep drift — long context stress test",
        "num_conversations": 25,
        "min_turns": 80,
        "max_turns": 120,
        "context_range": "~30K-50K tokens",
    },
]


def print_plan():
    """Pretty-print the production plan."""
    total = sum(t["num_conversations"] for t in TIERS)
    print("=" * 65)
    print("🏭 Poly-DriftBench Production — 3-Tier Length Strategy")
    print("=" * 65)
    print(f"  Total conversations: {total}")
    print(f"  Languages: EN + IT, ES, FR, DE (5 total)")
    print()
    for t in TIERS:
        print(f"  📏 {t['name'].upper():8s} | {t['num_conversations']:3d} convs | "
              f"{t['min_turns']}-{t['max_turns']} turns | {t['context_range']}")
        print(f"              └─ {t['description']}")
    print("=" * 65)


def run_production(parallel: int = 3, output_dir: str = "data/production",
                   dry_run: bool = False):
    """Run the full 3-tier production pipeline."""
    print_plan()

    if dry_run:
        print("\n🔍 DRY RUN — no data will be generated.")
        return

    # Load config
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    factory = DataFactory(config=config)

    # Global stats
    all_stats = {
        "production_config": {
            "tiers": TIERS,
            "parallel_conversations": parallel,
            "model": factory.model,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "tier_results": {},
    }

    total_start = time.time()
    conv_counter = 0  # Global conversation counter for unique IDs

    for tier in TIERS:
        tier_name = tier["name"]
        tier_start = time.time()

        print(f"\n{'='*65}")
        print(f"🔲 Starting Tier: {tier_name.upper()} "
              f"({tier['num_conversations']} conversations, "
              f"{tier['min_turns']}-{tier['max_turns']} turns)")
        print(f"{'='*65}\n")

        # Override the min/max turns for this tier
        saved_files = factory.produce_dataset(
            output_dir=output_dir / tier_name,
            num_conversations=tier["num_conversations"],
            min_turns=tier["min_turns"],
            max_turns=tier["max_turns"],
            translate=True,
            parallel_conversations=parallel,
        )

        tier_elapsed = time.time() - tier_start

        # Collect tier-level stats
        tier_stats = {
            "conversations_generated": len(saved_files),
            "conversations_requested": tier["num_conversations"],
            "time_seconds": round(tier_elapsed, 1),
            "avg_time_per_conv": round(tier_elapsed / max(len(saved_files), 1), 1),
            "files": [str(f) for f in saved_files],
        }

        # Aggregate quality scores from saved files
        scores = []
        api_calls_total = 0
        tokens_total = 0
        for fpath in saved_files:
            with open(fpath) as f:
                data = json.load(f)
            scores.append(data["quality"]["overall"])
            stats = data.get("metadata", {}).get("pipeline_stats", {})
            api_calls_total += stats.get("api_calls_total", 0)
            tokens_total += stats.get("tokens", {}).get("total", 0)

        if scores:
            import numpy as np
            tier_stats["quality"] = {
                "mean": round(float(np.mean(scores)), 2),
                "median": round(float(np.median(scores)), 2),
                "min": round(float(np.min(scores)), 2),
                "max": round(float(np.max(scores)), 2),
                "approved_rate": round(sum(1 for s in scores if s >= 7.0) / len(scores), 3),
            }
            tier_stats["total_api_calls"] = api_calls_total
            tier_stats["total_tokens"] = tokens_total

        all_stats["tier_results"][tier_name] = tier_stats
        conv_counter += len(saved_files)

        print(f"\n✅ Tier {tier_name.upper()} done: {len(saved_files)} conversations "
              f"in {tier_elapsed:.0f}s ({tier_elapsed/60:.1f}min)")

    total_elapsed = time.time() - total_start
    all_stats["production_config"]["end_time"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    all_stats["production_config"]["total_time_seconds"] = round(total_elapsed, 1)
    all_stats["production_config"]["total_conversations"] = conv_counter

    # Save production summary
    summary_path = output_dir / "production_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "=" * 65)
    print("🏆 PRODUCTION COMPLETE")
    print("=" * 65)
    print(f"  Total conversations: {conv_counter}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    for tier_name, ts in all_stats["tier_results"].items():
        q = ts.get("quality", {})
        print(f"  {tier_name.upper():8s}: {ts['conversations_generated']}/{ts['conversations_requested']} "
              f"| Mean={q.get('mean', '?')}/10 "
              f"| Approved={q.get('approved_rate', '?'):.0%} "
              f"| API calls={ts.get('total_api_calls', '?')}")
    print(f"\n  📄 Summary saved: {summary_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poly-DriftBench 3-Tier Production")
    parser.add_argument("--parallel", type=int, default=3,
                        help="Parallel conversations per tier (default: 3)")
    parser.add_argument("--output", default="data/production",
                        help="Output directory (default: data/production)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without generating")
    args = parser.parse_args()

    run_production(
        parallel=args.parallel,
        output_dir=args.output,
        dry_run=args.dry_run,
    )
