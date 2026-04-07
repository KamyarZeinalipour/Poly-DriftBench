"""
Experiment 12 — Per-Level Failure Ordering
============================================
Determines which DDM constraint (L1/L2/L3/L4) fails FIRST in each conversation.
This reveals the "failure cascade" — format drops before semantics, etc.

Key Questions:
    - Is L1 (canary token) always the first to fail?
    - Does the failure order change across languages?
    - Chi-squared test: Is failure ordering × language independent?
"""

import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

LEVELS = ["L1_format", "L2_structure", "L3_lexical", "L4_citation"]


def determine_failure_order(per_level_decay: dict) -> list[str]:
    """
    Determine the order in which constraint levels fail.
    
    Args:
        per_level_decay: Dict of level_name -> {"onset": int or None, ...}
    
    Returns:
        Sorted list of level names by failure onset (earliest first).
        Levels that never fail are excluded.
    """
    onsets = []
    for level in LEVELS:
        if level in per_level_decay:
            onset = per_level_decay[level].get("onset") or per_level_decay[level].get("decay_onset")
            if onset is not None:
                onsets.append((level, onset))
    
    # Sort by onset turn
    onsets.sort(key=lambda x: x[1])
    return [level for level, _ in onsets]


def run_experiment_12(
    drift_summary_csv: str,
    per_level_decay_json: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Run Experiment 12: Per-Level Failure Ordering.
    
    Args:
        drift_summary_csv: Path to conversation-level drift summary.
        per_level_decay_json: Path to per-level decay JSON from Exp 2.
        output_dir: Output directory.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 12: Per-Level Failure Ordering")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load per-level decay data
    with open(per_level_decay_json) as f:
        decay_data = json.load(f)
    
    # Load summary for language info
    summary_df = pd.read_csv(drift_summary_csv)
    
    # Build failure ordering for each conversation
    ordering_rows = []
    first_failure_counts = Counter()  # Which level fails first overall
    first_failure_by_lang = {}  # Which level fails first per language
    
    for key, level_data in decay_data.items():
        parts = key.split("|")
        if len(parts) >= 3:
            model, lang, conv_id = parts[0], parts[1], "|".join(parts[2:])
        else:
            continue
        
        order = determine_failure_order(level_data)
        
        if order:
            first_failure = order[0]
            first_failure_counts[first_failure] += 1
            
            if lang not in first_failure_by_lang:
                first_failure_by_lang[lang] = Counter()
            first_failure_by_lang[lang][first_failure] += 1
        
        ordering_rows.append({
            "conversation_id": conv_id,
            "language": lang,
            "model": model,
            "failure_order": " → ".join(order) if order else "none",
            "first_failure": order[0] if order else "none",
            "num_failing_levels": len(order),
        })
        
        # Add individual onset columns
        for level in LEVELS:
            if level in level_data:
                onset = level_data[level].get("onset") or level_data[level].get("decay_onset")
                ordering_rows[-1][f"{level}_onset"] = onset
    
    order_df = pd.DataFrame(ordering_rows)
    order_df.to_csv(output_dir / "failure_ordering.csv", index=False)
    
    # First-failure frequency by language
    freq_rows = []
    for lang in sorted(first_failure_by_lang.keys()):
        counts = first_failure_by_lang[lang]
        total = sum(counts.values())
        row = {"language": lang, "total_conversations": total}
        for level in LEVELS:
            count = counts.get(level, 0)
            row[f"{level}_first_count"] = count
            row[f"{level}_first_pct"] = count / total * 100 if total > 0 else 0
        freq_rows.append(row)
    
    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(output_dir / "first_failure_by_language.csv", index=False)
    
    # Chi-squared test: Is failure ordering independent of language?
    contingency_data = []
    for lang in sorted(first_failure_by_lang.keys()):
        counts = first_failure_by_lang[lang]
        contingency_data.append([counts.get(level, 0) for level in LEVELS])
    
    contingency = np.array(contingency_data)
    
    # Only run chi2 if we have enough data
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2 and contingency.sum() > 0:
        # Remove columns with all zeros
        col_mask = contingency.sum(axis=0) > 0
        contingency_clean = contingency[:, col_mask]
        
        if contingency_clean.shape[1] >= 2:
            chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_clean)
        else:
            chi2, chi2_p, dof = 0.0, 1.0, 0
    else:
        chi2, chi2_p, dof = 0.0, 1.0, 0
    
    # Mean onset turn per level per language
    onset_summary = order_df.groupby("language")[[f"{l}_onset" for l in LEVELS]].mean()
    onset_summary.to_csv(output_dir / "mean_onset_by_language.csv")
    
    # Summary
    summary = {
        "overall_first_failure_counts": dict(first_failure_counts),
        "chi_squared": float(chi2),
        "chi_squared_p": float(chi2_p),
        "chi_squared_dof": int(dof),
        "failure_order_dependent_on_language": bool(chi2_p < 0.05),
        "most_common_first_failure": first_failure_counts.most_common(1)[0] if first_failure_counts else ("none", 0),
    }
    
    with open(output_dir / "level_ordering_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"  Overall first-failure: {dict(first_failure_counts)}")
    logger.info(f"  Chi-squared: χ²={chi2:.4f}, p={chi2_p:.6f}, dof={dof}")
    logger.info(f"  Failure order depends on language: {chi2_p < 0.05}")
    logger.info(f"  Saved to {output_dir}/")
    
    return order_df
