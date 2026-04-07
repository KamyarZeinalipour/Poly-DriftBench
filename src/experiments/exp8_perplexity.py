"""
Experiment 8 — Perplexity at Drift Onset
==========================================
Measures model perplexity when instruction-following starts degrading.
Key question: Do models become "confused" (high PPL) or "confidently wrong" (low PPL)?
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def analyze_perplexity_at_drift(
    perplexities: list[float],
    ddm_scores: list[float],
    dop: int = None,
) -> dict:
    """
    Analyze perplexity relative to drift onset.
    
    Args:
        perplexities: Per-turn perplexity values.
        ddm_scores: Per-turn DDM scores.
        dop: Drift onset point (1-indexed).
    """
    if not perplexities or not ddm_scores:
        return {}
    
    ppls = np.array(perplexities)
    scores = np.array(ddm_scores)
    n = min(len(ppls), len(scores))
    ppls = ppls[:n]
    scores = scores[:n]
    
    result = {
        "mean_perplexity": float(np.mean(ppls)),
        "std_perplexity": float(np.std(ppls)),
    }
    
    if dop is not None and dop > 1 and dop <= n:
        dop_idx = dop - 1  # 0-indexed
        
        pre_drift_ppls = ppls[:dop_idx]
        post_drift_ppls = ppls[dop_idx:]
        
        result["pre_drift_mean_ppl"] = float(np.mean(pre_drift_ppls)) if len(pre_drift_ppls) > 0 else None
        result["post_drift_mean_ppl"] = float(np.mean(post_drift_ppls)) if len(post_drift_ppls) > 0 else None
        result["ppl_at_dop"] = float(ppls[dop_idx])
        
        # Is perplexity higher or lower after drift?
        if len(pre_drift_ppls) > 1 and len(post_drift_ppls) > 1:
            t_stat, p_value = stats.ttest_ind(pre_drift_ppls, post_drift_ppls)
            result["ppl_change_t_stat"] = float(t_stat)
            result["ppl_change_p_value"] = float(p_value)
            result["confident_drift"] = float(np.mean(post_drift_ppls)) <= float(np.mean(pre_drift_ppls))
    
    # Correlation: DDM score vs perplexity
    if n >= 3:
        corr, corr_p = stats.pearsonr(scores, ppls)
        result["ddm_ppl_correlation"] = float(corr)
        result["ddm_ppl_correlation_p"] = float(corr_p)
    
    return result


def run_experiment_8(
    inference_results: list,
    drift_results: list,
    output_dir: str,
) -> pd.DataFrame:
    """
    Run Experiment 8: Perplexity at Drift Onset.
    
    Args:
        inference_results: List of InferenceResult objects with perplexities.
        drift_results: List of ConversationDriftResult objects.
        output_dir: Output directory.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 8: Perplexity at Drift Onset")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    for inf_result, drift_result in zip(inference_results, drift_results):
        if not inf_result.response_perplexities:
            continue
        
        analysis = analyze_perplexity_at_drift(
            inf_result.response_perplexities,
            [t.ddm_score for t in drift_result.turn_results],
            drift_result.drift_onset_point,
        )
        
        analysis.update({
            "conversation_id": drift_result.conversation_id,
            "language": drift_result.language,
            "model": drift_result.model_name,
            "drift_onset": drift_result.drift_onset_point,
        })
        rows.append(analysis)
    
    ppl_df = pd.DataFrame(rows)
    ppl_df.to_csv(output_dir / "perplexity_analysis.csv", index=False)
    
    # Summary by language
    if not ppl_df.empty:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lang_ppl = ppl_df.groupby("language").agg({
                "pre_drift_mean_ppl": "mean",
                "post_drift_mean_ppl": "mean",
                "ppl_at_dop": "mean",
                "confident_drift": "mean",
            }).reset_index()
        lang_ppl.to_csv(output_dir / "perplexity_by_language.csv", index=False)
        
        # Overall: Do models drift confidently?
        confident_pct = ppl_df["confident_drift"].mean() * 100 if "confident_drift" in ppl_df else 0
        logger.info(f"  Models drift confidently {confident_pct:.1f}% of the time")
    
    logger.info(f"  Perplexity analysis saved to {output_dir}/")
    return ppl_df
