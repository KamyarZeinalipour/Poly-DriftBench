"""
Experiment 11 — Tier Effect Analysis
======================================
Compares drift metrics across Short (10-15 turns), Medium (30-50), Long (80-120).
Tests whether drift scales linearly or has a tipping point.

Key Questions:
    - Does the short tier show no drift (control)?
    - Is there a non-linear jump between medium and long?
    - Does normalized DOP (DOP/total_turns) change across tiers?
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def classify_tier(model_or_tier: str) -> str:
    """Extract tier from model name like 'data-gen-short'."""
    if isinstance(model_or_tier, str):
        for tier in ["short", "medium", "long"]:
            if tier in model_or_tier.lower():
                return tier
    return "unknown"


def run_experiment_11(drift_summary_csv: str, output_dir: str) -> pd.DataFrame:
    """
    Run Experiment 11: Tier Effect Analysis.
    
    Args:
        drift_summary_csv: Path to conversation-level drift summary.
        output_dir: Output directory.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 11: Tier Effect Analysis")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(drift_summary_csv)
    
    # Extract tier from model name
    df["tier"] = df["model"].apply(classify_tier)
    
    # Compute normalized DOP
    df["normalized_dop"] = df.apply(
        lambda r: r["drift_onset"] / r["total_turns"] if pd.notna(r["drift_onset"]) and r["total_turns"] > 0 else None,
        axis=1,
    )
    
    # Fraction of conversations with any drift
    df["has_drift"] = df["drift_onset"].notna()
    
    # Tier × Language summary
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tier_lang = df.groupby(["tier", "language"]).agg({
            "mean_ddm": ["mean", "std"],
            "auc": ["mean", "std"],
            "drift_onset": ["mean", "median"],
            "normalized_dop": ["mean"],
            "has_drift": ["mean"],  # Fraction with drift
            "half_life": ["mean"],
            "recovery_rate": ["mean"],
            "total_turns": ["mean"],
        }).reset_index()
    tier_lang.columns = ["_".join(c).strip("_") for c in tier_lang.columns]
    tier_lang.to_csv(output_dir / "tier_language_summary.csv", index=False)
    
    # Tier-only summary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tier_summary = df.groupby("tier").agg({
            "mean_ddm": ["mean", "std"],
            "auc": ["mean"],
            "has_drift": ["mean", "sum"],
            "normalized_dop": ["mean", "std"],
            "total_turns": ["mean"],
        }).reset_index()
    tier_summary.columns = ["_".join(c).strip("_") for c in tier_summary.columns]
    tier_summary.to_csv(output_dir / "tier_summary.csv", index=False)
    
    # Statistical test: Kruskal-Wallis across tiers for mean_ddm
    tier_groups = [
        group["mean_ddm"].dropna().values
        for _, group in df.groupby("tier")
    ]
    if len(tier_groups) >= 2 and all(len(g) > 0 for g in tier_groups):
        h_stat, h_p = stats.kruskal(*tier_groups)
    else:
        h_stat, h_p = 0.0, 1.0
    
    # Test for non-linearity: compare short→medium vs medium→long effect sizes
    tier_means = df.groupby("tier")["mean_ddm"].mean()
    tier_order = ["short", "medium", "long"]
    effect_sizes = {}
    for i in range(len(tier_order) - 1):
        t1, t2 = tier_order[i], tier_order[i + 1]
        if t1 in tier_means and t2 in tier_means:
            g1 = df[df["tier"] == t1]["mean_ddm"].dropna()
            g2 = df[df["tier"] == t2]["mean_ddm"].dropna()
            if len(g1) > 1 and len(g2) > 1:
                # Cohen's d
                pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
                d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0
                effect_sizes[f"{t1}_to_{t2}"] = float(d)
    
    summary = {
        "kruskal_wallis_h": float(h_stat),
        "kruskal_wallis_p": float(h_p),
        "significant": bool(h_p < 0.05),
        "effect_sizes_cohens_d": effect_sizes,
        "tier_mean_ddm": {t: float(m) for t, m in tier_means.items()},
    }
    
    with open(output_dir / "tier_effect_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"  Kruskal-Wallis: H={h_stat:.4f}, p={h_p:.6f}")
    logger.info(f"  Effect sizes: {effect_sizes}")
    logger.info(f"  Saved to {output_dir}/")
    
    return df
