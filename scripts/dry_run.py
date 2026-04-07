#!/usr/bin/env python3
"""
Dry Run — End-to-End Validation
================================
Tests the full pipeline on a tiny data subset with 1 small model tokenizer
to verify data loading, fertility profiling, DDM evaluation, and all outputs.

Usage:
    python scripts/dry_run.py
"""

import sys
import json
import os
import tempfile
from pathlib import Path

# Set HF cache before importing transformers
os.environ["HF_HOME"] = "/home4/kamyar/italian_detox/hf_cache/hub"

import numpy as np

# ──────────────────────────────────────────────────────────
# 1. Test Data Loading (3-tier structure)
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Testing Data Loader (3-tier directory structure)")
print("=" * 60)

from src.tokenizer.fertility import load_parallel_texts

data_dir = Path("data/production")
parallel_data = load_parallel_texts(data_dir)

print(f"  Languages found: {list(parallel_data.keys())}")
for lang, texts in parallel_data.items():
    print(f"  {lang}: {len(texts)} texts, avg length = {np.mean([len(t) for t in texts]):.0f} chars")

# Verify parallel alignment
en_count = len(parallel_data.get("en", []))
for lang, texts in parallel_data.items():
    if lang != "en":
        assert len(texts) == en_count, (
            f"ALIGNMENT ERROR: {lang} has {len(texts)} texts vs EN has {en_count}"
        )
print(f"\n  ✅ All languages aligned at {en_count} texts each\n")

# ──────────────────────────────────────────────────────────
# 2. Test Fertility Profiling (1 small model, subset of data)
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Testing Fertility Profiler (Qwen2.5-3B, 50 texts)")
print("=" * 60)

from src.tokenizer.fertility import FertilityProfiler

# Use only 1 small model for speed
test_model_configs = [
    {"name": "qwen2.5-3b", "hf_id": "Qwen/Qwen2.5-3B-Instruct"},
]

# Take only first 50 texts for speed
SUBSET = 50
subset_data = {lang: texts[:SUBSET] for lang, texts in parallel_data.items()}
print(f"  Using {SUBSET} texts per language for dry run")

profiler = FertilityProfiler(test_model_configs)
results = profiler.profile_dataset(subset_data)

print(f"\n  Fertility Results ({len(results)} language-model pairs):")
df = profiler.results_to_dataframe(results)
for _, row in df.iterrows():
    print(
        f"    {row['model']:>15s} | {row['language']:>3s}: "
        f"TFR = {row['tfr']:.4f} ({row['overhead_pct']:+.1f}% overhead), "
        f"tokens: {int(row['total_tokens'])} vs EN {int(row['en_total_tokens'])}"
    )

# Save to temp dir
tmp_dir = Path("results/dry_run/fertility")
tmp_dir.mkdir(parents=True, exist_ok=True)
profiler.save_results(results, tmp_dir)
print(f"\n  ✅ Fertility results saved to {tmp_dir}/")
print(f"     Files: {[f.name for f in tmp_dir.iterdir()]}\n")

# ──────────────────────────────────────────────────────────
# 3. Test DDM Evaluation (on actual production data)
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Testing DDM Evaluator (5 conversations × 5 languages)")
print("=" * 60)

from src.evaluation.ddm import DDMEvaluator

# Load a few actual conversations and evaluate them
test_convs = []
conv_dir = data_dir / "short" / "parallel"

NUM_CONVS = 5
languages = ["en", "it", "es", "fr", "de"]

for lang in languages:
    lang_dir = conv_dir / lang
    if not lang_dir.exists():
        print(f"  ⚠️  Skipping {lang}: directory not found")
        continue

    jsonl_files = sorted(lang_dir.glob("*.jsonl"))[:NUM_CONVS]

    evaluator = DDMEvaluator(language=lang, strict_citations=False)

    for jsonl_file in jsonl_files:
        # Read all messages
        messages = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))

        # Extract assistant responses
        responses = [m["text"] for m in messages if m.get("role") == "assistant"]
        conv_id = jsonl_file.stem

        if not responses:
            continue

        result = evaluator.evaluate_conversation(
            responses=responses,
            conversation_id=conv_id,
            language=lang,
            model_name="data-gen",
        )
        test_convs.append(result)

print(f"\n  Evaluated {len(test_convs)} conversations total")

# Show summary for each
print(f"\n  {'Conv ID':<40s} {'Lang':>4s} {'Turns':>5s} {'DDM':>6s} {'AUC':>6s} {'DOP':>5s} {'sDOP':>5s} {'τ½':>5s} {'Recov':>7s}")
print("  " + "-" * 90)
for r in test_convs:
    short_id = r.conversation_id[-30:] if len(r.conversation_id) > 30 else r.conversation_id
    print(
        f"  {short_id:<40s} {r.language:>4s} {r.total_turns:>5d} "
        f"{r.mean_ddm:>6.3f} {r.auc:>6.3f} "
        f"{str(r.drift_onset_point):>5s} {str(r.sustained_dop):>5s} "
        f"{str(r.half_life):>5s} {r.recovery_rate:>6.1%}"
    )

# Show per-level decay for first conversation
if test_convs:
    print(f"\n  Per-Level Decay (first conv: {test_convs[0].conversation_id}):")
    for level, decay in test_convs[0].per_level_decay.items():
        scores_preview = [f"{s:.1f}" for s in decay.scores[:10]]
        trail = "..." if len(decay.scores) > 10 else ""
        print(
            f"    {level:15s}: mean={decay.mean_score:.3f} AUC={decay.auc:.3f} "
            f"onset={decay.decay_onset} scores=[{', '.join(scores_preview)}{trail}]"
        )

# Save DDM results
ddm_dir = Path("results/dry_run/drift_curves")
ddm_dir.mkdir(parents=True, exist_ok=True)
evaluator_save = DDMEvaluator(language="en")
evaluator_save.save_results(test_convs, ddm_dir)
print(f"\n  ✅ DDM results saved to {ddm_dir}/")
print(f"     Files: {[f.name for f in ddm_dir.iterdir()]}")

# ──────────────────────────────────────────────────────────
# 4. Verify Output Files
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Verifying Output Files")
print("=" * 60)

import pandas as pd

# Fertility
fert_csv = tmp_dir / "fertility_ratios.csv"
fert_df = pd.read_csv(fert_csv)
print(f"\n  fertility_ratios.csv: {len(fert_df)} rows, columns = {list(fert_df.columns)}")

fert_json = tmp_dir / "fertility_detailed.json"
with open(fert_json) as f:
    fert_detail = json.load(f)
print(f"  fertility_detailed.json: {len(fert_detail)} entries")

# Drift
drift_csv = ddm_dir / "drift_results.csv"
drift_df = pd.read_csv(drift_csv)
print(f"  drift_results.csv: {len(drift_df)} rows, columns = {list(drift_df.columns)}")

summary_csv = ddm_dir / "drift_summary.csv"
summary_df = pd.read_csv(summary_csv)
print(f"  drift_summary.csv: {len(summary_df)} rows, columns = {list(summary_df.columns)}")

agg_csv = ddm_dir / "drift_aggregated.csv"
agg_df = pd.read_csv(agg_csv)
print(f"  drift_aggregated.csv: {len(agg_df)} rows")

ci_csv = ddm_dir / "drift_ci95.csv"
ci_df = pd.read_csv(ci_csv)
print(f"  drift_ci95.csv: {len(ci_df)} rows, columns = {list(ci_df.columns)}")

decay_json = ddm_dir / "per_level_decay.json"
with open(decay_json) as f:
    decay = json.load(f)
print(f"  per_level_decay.json: {len(decay)} conversation entries")

# ──────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DRY RUN COMPLETE ✅")
print("=" * 60)
print(f"""
  Data Loading:      {len(parallel_data)} languages, {en_count} texts each
  Fertility:         {len(results)} results (1 model × {len(results)} languages)
  DDM Evaluation:    {len(test_convs)} conversations evaluated
  Output Files:      All {len(list(tmp_dir.iterdir())) + len(list(ddm_dir.iterdir()))} files written and verified

  Ready for full production run:
    HF_HOME=/home4/kamyar/italian_detox/hf_cache/hub python -m src.cli fertility --data-dir data/production
""")
