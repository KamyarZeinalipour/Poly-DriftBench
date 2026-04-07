"""
Experiment 10 — Cross-Model Consistency
=========================================
Tests whether all 12 models rank languages in the same drift order.
If Kendall's W (concordance) is high, the Token Squeeze effect is model-independent.

Metrics:
    - Kendall's τ between all model pairs
    - Kendall's W (concordance coefficient) across all models
    - Spearman's ρ for rank correlation
"""

import json
import logging
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_kendalls_w(rankings: np.ndarray) -> tuple[float, float]:
    """
    Compute Kendall's W (coefficient of concordance).
    
    Args:
        rankings: Array of shape (n_judges, n_items) where each row is a ranking.
    
    Returns:
        (W, p_value) tuple.
    """
    k, n = rankings.shape  # k judges, n items
    
    # Rank each judge's scores
    ranked = np.zeros_like(rankings, dtype=float)
    for i in range(k):
        ranked[i] = stats.rankdata(rankings[i])
    
    # Sum of ranks for each item
    R = ranked.sum(axis=0)
    R_mean = R.mean()
    
    # S = sum of squared deviations
    S = np.sum((R - R_mean) ** 2)
    
    # Kendall's W
    W = (12 * S) / (k ** 2 * (n ** 3 - n))
    
    # Chi-squared approximation for p-value
    chi2 = k * (n - 1) * W
    p_value = 1 - stats.chi2.cdf(chi2, n - 1)
    
    return float(W), float(p_value)


def run_experiment_10(drift_summary_csv: str, output_dir: str) -> dict:
    """
    Run Experiment 10: Cross-Model Consistency.
    
    Args:
        drift_summary_csv: Path to conversation-level drift summary from Exp 2.
        output_dir: Output directory.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 10: Cross-Model Consistency")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(drift_summary_csv)
    
    # Compute mean DOP per model × language
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pivot = df.groupby(["model", "language"])["mean_ddm"].mean().unstack("language")
    
    models = pivot.index.tolist()
    languages = pivot.columns.tolist()
    
    logger.info(f"  Models: {len(models)}, Languages: {len(languages)}")
    
    # Pairwise Kendall's τ between all model pairs
    tau_results = []
    for m1, m2 in itertools.combinations(models, 2):
        scores1 = pivot.loc[m1].values
        scores2 = pivot.loc[m2].values
        
        # Remove NaN
        mask = ~(np.isnan(scores1) | np.isnan(scores2))
        if mask.sum() < 3:
            continue
        
        tau, p = stats.kendalltau(scores1[mask], scores2[mask])
        tau_results.append({
            "model_1": m1,
            "model_2": m2,
            "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
            "p_value": float(p) if not np.isnan(p) else 1.0,
        })
    
    tau_df = pd.DataFrame(tau_results)
    tau_df.to_csv(output_dir / "pairwise_kendall_tau.csv", index=False)
    
    # Kendall's W (concordance) across all models
    rankings = pivot.values  # shape: (n_models, n_languages)
    valid_mask = ~np.isnan(rankings).any(axis=0)
    if valid_mask.sum() >= 3:
        W, W_p = compute_kendalls_w(rankings[:, valid_mask])
    else:
        W, W_p = 0.0, 1.0
    
    # Spearman's ρ pairwise
    rho_results = []
    for m1, m2 in itertools.combinations(models, 2):
        scores1 = pivot.loc[m1].values
        scores2 = pivot.loc[m2].values
        mask = ~(np.isnan(scores1) | np.isnan(scores2))
        if mask.sum() < 3:
            continue
        rho, p = stats.spearmanr(scores1[mask], scores2[mask])
        rho_results.append({
            "model_1": m1, "model_2": m2,
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "p_value": float(p) if not np.isnan(p) else 1.0,
        })
    
    rho_df = pd.DataFrame(rho_results)
    rho_df.to_csv(output_dir / "pairwise_spearman_rho.csv", index=False)
    
    # Save summary
    summary = {
        "kendalls_w": W,
        "kendalls_w_p_value": W_p,
        "kendalls_w_significant": bool(W_p < 0.05),
        "n_models": len(models),
        "n_languages": len(languages),
        "mean_pairwise_tau": float(tau_df["kendall_tau"].mean()) if len(tau_df) > 0 else 0.0,
        "mean_pairwise_rho": float(rho_df["spearman_rho"].mean()) if len(rho_df) > 0 else 0.0,
        "language_drift_ranking": pivot.mean().sort_values().index.tolist(),
    }
    
    with open(output_dir / "cross_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"  Kendall's W = {W:.4f} (p={W_p:.6f})")
    logger.info(f"  Mean pairwise τ = {summary['mean_pairwise_tau']:.4f}")
    logger.info(f"  Language drift ranking: {summary['language_drift_ranking']}")
    logger.info(f"  Saved to {output_dir}/")
    
    return summary
