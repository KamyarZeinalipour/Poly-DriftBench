"""
Experiment 7 — Context Budget Analysis
=========================================
Tracks DDM score as a function of context window UTILIZATION (%) not turn number.
Tests whether all languages start drifting at the same utilization threshold.

Hypothesis: Non-English languages hit the drift threshold earlier because
TFR > 1.0 means they consume more context per turn.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def compute_context_budget_metrics(
    ddm_scores: list[float],
    context_lengths: list[int],
    context_window: int,
) -> dict:
    """
    Compute context budget metrics for a single conversation.
    
    Args:
        ddm_scores: Per-turn DDM scores.
        context_lengths: Per-turn total context length in tokens.
        context_window: Model's context window size.
    """
    utilizations = [c / context_window for c in context_lengths]
    
    # Find utilization at DOP
    dop_utilization = None
    for i, score in enumerate(ddm_scores):
        if score < 1.0:
            dop_utilization = utilizations[i]
            break
    
    # Find utilization at half-life
    halflife_utilization = None
    for i, score in enumerate(ddm_scores):
        if score <= 0.5:
            halflife_utilization = utilizations[i]
            break
    
    # Max utilization reached
    max_utilization = max(utilizations) if utilizations else 0.0
    
    return {
        "dop_utilization": dop_utilization,
        "halflife_utilization": halflife_utilization,
        "max_utilization": max_utilization,
        "final_utilization": utilizations[-1] if utilizations else 0.0,
        "utilizations": utilizations,
        "ddm_scores": ddm_scores,
    }


def run_experiment_7(
    drift_results_csv: str,
    model_configs: list[dict],
    output_dir: str,
    context_lengths_json: str = None,
) -> pd.DataFrame:
    """
    Run Experiment 7: Context Budget Analysis.
    
    This experiment can run in two modes:
    1. With pre-computed context lengths from Exp 2 inference (preferred)
    2. With estimated context lengths from token counts in the data
    
    Args:
        drift_results_csv: Per-turn drift results from Exp 2.
        model_configs: Model configurations with context_window sizes.
        output_dir: Output directory.
        context_lengths_json: Optional pre-computed context lengths from inference.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 7: Context Budget Analysis")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(drift_results_csv)
    
    # Build model context window lookup
    ctx_windows = {m["name"]: m.get("context_window", 8192) for m in model_configs}
    
    # Load pre-computed context lengths if available
    ctx_lengths = {}
    if context_lengths_json and Path(context_lengths_json).exists():
        with open(context_lengths_json) as f:
            ctx_lengths = json.load(f)
    
    budget_rows = []
    
    for (conv_id, lang, model), group in df.groupby(
        ["conversation_id", "language", "model"]
    ):
        group_sorted = group.sort_values("turn")
        scores = group_sorted["ddm_score"].tolist()
        
        # Get context window for this model
        ctx_window = ctx_windows.get(model, 8192)
        
        # Get context lengths (from Exp 2 or estimate)
        key = f"{model}|{lang}|{conv_id}"
        if key in ctx_lengths:
            lengths = ctx_lengths[key]
        else:
            # Estimate: assume ~100 tokens per turn cumulative
            lengths = [100 * (i + 1) for i in range(len(scores))]
        
        metrics = compute_context_budget_metrics(scores, lengths, ctx_window)
        
        budget_rows.append({
            "conversation_id": conv_id,
            "language": lang,
            "model": model,
            "context_window": ctx_window,
            "dop_utilization": metrics["dop_utilization"],
            "halflife_utilization": metrics["halflife_utilization"],
            "max_utilization": metrics["max_utilization"],
            "final_utilization": metrics["final_utilization"],
        })
    
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(output_dir / "context_budget.csv", index=False)
    
    # Summary by language
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lang_budget = budget_df.groupby("language").agg({
            "dop_utilization": ["mean", "median", "std"],
            "halflife_utilization": ["mean", "median"],
            "max_utilization": ["mean"],
        }).reset_index()
    lang_budget.columns = ["_".join(c).strip("_") for c in lang_budget.columns]
    lang_budget.to_csv(output_dir / "budget_by_language.csv", index=False)
    
    # Test: Do all languages start drifting at the same utilization?
    dop_groups = [
        group["dop_utilization"].dropna().values
        for _, group in budget_df.groupby("language")
    ]
    if len(dop_groups) >= 2 and all(len(g) > 0 for g in dop_groups):
        h_stat, h_p = scipy_stats.kruskal(*dop_groups)
        test_result = {
            "test": "kruskal_wallis",
            "h_statistic": float(h_stat),
            "p_value": float(h_p),
            "same_threshold": bool(h_p > 0.05),  # Fail to reject = same threshold
        }
    else:
        test_result = {"test": "insufficient_data"}
    
    with open(output_dir / "budget_test.json", "w") as f:
        json.dump(test_result, f, indent=2)
    
    logger.info(f"  Context budget analysis saved to {output_dir}/")
    return budget_df
