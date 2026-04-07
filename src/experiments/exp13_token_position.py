"""
Experiment 13 — Token Position Analysis
==========================================
Tracks the system prompt's relative position in the context as conversations grow.
Higher TFR languages push the system prompt to a smaller fraction of the total context.

Hypothesis: When the system prompt drops below X% of the total context, drift starts.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_token_position_metrics(
    system_prompt_tokens: int,
    context_lengths: list[int],
    ddm_scores: list[float],
    dop: int = None,
) -> dict:
    """
    Compute system prompt token position metrics.
    
    Args:
        system_prompt_tokens: Number of tokens in the system prompt.
        context_lengths: Total context length at each turn.
        ddm_scores: DDM scores at each turn.
        dop: Drift onset point (1-indexed).
    """
    ratios = [system_prompt_tokens / max(c, 1) for c in context_lengths]
    
    result = {
        "initial_ratio": ratios[0] if ratios else 0.0,
        "final_ratio": ratios[-1] if ratios else 0.0,
        "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "ratio_at_dop": None,
        "ratios": ratios,
    }
    
    if dop is not None and dop > 0 and dop <= len(ratios):
        result["ratio_at_dop"] = ratios[dop - 1]
    
    # Correlation: ratio vs DDM score
    n = min(len(ratios), len(ddm_scores))
    if n >= 3:
        corr, corr_p = stats.pearsonr(ratios[:n], ddm_scores[:n])
        result["ratio_ddm_correlation"] = float(corr)
        result["ratio_ddm_correlation_p"] = float(corr_p)
    
    return result


def run_experiment_13(
    inference_results: list,
    drift_results: list,
    system_prompt: str,
    model_configs: list[dict],
    output_dir: str,
) -> pd.DataFrame:
    """
    Run Experiment 13: Token Position Analysis.
    
    Can piggyback on SPAR analysis (Exp 5) or use context lengths from Exp 2.
    
    Args:
        inference_results: List of InferenceResult objects with context lengths.
        drift_results: List of ConversationDriftResult objects.
        system_prompt: The system prompt text.
        model_configs: Model configs for tokenizer access.
        output_dir: Output directory.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 13: Token Position Analysis")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    for inf_result, drift_result in zip(inference_results, drift_results):
        if not inf_result.context_lengths_tokens:
            continue
        
        # Estimate system prompt token count (approximate)
        # In practice, this should be computed from the tokenizer
        sys_tokens = 200  # Rough estimate for the DDM system prompt
        
        scores = [t.ddm_score for t in drift_result.turn_results]
        
        metrics = compute_token_position_metrics(
            sys_tokens,
            inf_result.context_lengths_tokens,
            scores,
            drift_result.drift_onset_point,
        )
        
        metrics.update({
            "conversation_id": drift_result.conversation_id,
            "language": drift_result.language,
            "model": drift_result.model_name,
            "drift_onset": drift_result.drift_onset_point,
        })
        
        # Remove the full ratios list for CSV
        ratios_list = metrics.pop("ratios", [])
        rows.append(metrics)
    
    pos_df = pd.DataFrame(rows)
    pos_df.to_csv(output_dir / "token_position.csv", index=False)
    
    # Summary by language
    if not pos_df.empty:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lang_pos = pos_df.groupby("language").agg({
                "ratio_at_dop": ["mean", "std"],
                "initial_ratio": "mean",
                "final_ratio": "mean",
            }).reset_index()
        lang_pos.columns = ["_".join(c).strip("_") for c in lang_pos.columns]
        lang_pos.to_csv(output_dir / "token_position_by_language.csv", index=False)
        
        # Is the ratio at DOP consistent across languages?
        dop_groups = [
            group["ratio_at_dop"].dropna().values
            for _, group in pos_df.groupby("language")
        ]
        if len(dop_groups) >= 2 and all(len(g) > 0 for g in dop_groups):
            h_stat, h_p = stats.kruskal(*dop_groups)
            logger.info(
                f"  Critical ratio at DOP consistent across languages: "
                f"H={h_stat:.4f}, p={h_p:.6f}"
            )
    
    logger.info(f"  Token position analysis saved to {output_dir}/")
    return pos_df
