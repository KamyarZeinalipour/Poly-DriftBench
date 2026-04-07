"""
Experiment 9 — Drift Velocity Analysis
========================================
Measures the RATE of DDM score decay (slope of the drift curve).
Does drift happen gradually or catastrophically? Do some languages collapse faster?

Metrics:
    - Mean velocity: Average ΔDDM/Δturn across the conversation
    - Max velocity: Steepest single-turn decline
    - Rolling velocity: 5-turn sliding window velocity
    - Velocity at DOP: How steep the decline is at drift onset
    - Statistical: ANOVA on velocities across languages
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_drift_velocity(ddm_scores: list[float], window: int = 5) -> dict:
    """
    Compute drift velocity metrics from a sequence of DDM scores.
    
    Args:
        ddm_scores: Per-turn DDM scores (0.0 to 1.0).
        window: Rolling window size for velocity smoothing.
    
    Returns:
        Dict with velocity metrics.
    """
    scores = np.array(ddm_scores)
    n = len(scores)
    
    if n < 2:
        return {"mean_velocity": 0.0, "max_velocity": 0.0}
    
    # Per-turn velocity (negative = declining)
    deltas = np.diff(scores)
    
    # Rolling velocity (smoothed)
    if n > window:
        rolling_vel = np.convolve(deltas, np.ones(window) / window, mode="valid")
    else:
        rolling_vel = deltas
    
    # Find velocity at DOP
    dop_velocity = 0.0
    for i, s in enumerate(scores):
        if s < 1.0 and i > 0:
            dop_velocity = float(deltas[i - 1])
            break
    
    return {
        "mean_velocity": float(np.mean(deltas)),
        "max_velocity": float(np.min(deltas)),  # Most negative = steepest decline
        "std_velocity": float(np.std(deltas)),
        "min_rolling_velocity": float(np.min(rolling_vel)) if len(rolling_vel) > 0 else 0.0,
        "velocity_at_dop": dop_velocity,
        "num_declining_turns": int(np.sum(deltas < 0)),
        "num_improving_turns": int(np.sum(deltas > 0)),
        "num_stable_turns": int(np.sum(deltas == 0)),
    }


def run_experiment_9(drift_results_csv: str, output_dir: str) -> pd.DataFrame:
    """
    Run Experiment 9: Drift Velocity Analysis.
    
    Args:
        drift_results_csv: Path to per-turn drift results from Exp 2.
        output_dir: Output directory for velocity results.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 9: Drift Velocity Analysis")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(drift_results_csv)
    
    velocity_rows = []
    
    # Group by conversation
    for (conv_id, lang, model), group in df.groupby(
        ["conversation_id", "language", "model"]
    ):
        scores = group.sort_values("turn")["ddm_score"].tolist()
        vel = compute_drift_velocity(scores)
        vel.update({
            "conversation_id": conv_id,
            "language": lang,
            "model": model,
            "total_turns": len(scores),
        })
        velocity_rows.append(vel)
    
    vel_df = pd.DataFrame(velocity_rows)
    
    # Save per-conversation velocity
    vel_df.to_csv(output_dir / "drift_velocity.csv", index=False)
    
    # Aggregate by language
    lang_vel = vel_df.groupby("language").agg({
        "mean_velocity": ["mean", "std"],
        "max_velocity": ["mean", "std"],
        "velocity_at_dop": ["mean"],
    }).reset_index()
    lang_vel.columns = ["_".join(c).strip("_") for c in lang_vel.columns]
    lang_vel.to_csv(output_dir / "velocity_by_language.csv", index=False)
    
    # ANOVA: Are velocities significantly different across languages?
    lang_groups = [
        group["mean_velocity"].values
        for _, group in vel_df.groupby("language")
    ]
    if len(lang_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*lang_groups)
        anova_result = {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "n_languages": len(lang_groups),
        }
        with open(output_dir / "velocity_anova.json", "w") as f:
            json.dump(anova_result, f, indent=2)
        logger.info(f"  ANOVA: F={f_stat:.4f}, p={p_value:.6f}")
    
    logger.info(f"  Saved velocity results to {output_dir}/")
    return vel_df
