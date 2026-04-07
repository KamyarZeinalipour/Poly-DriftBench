"""
Experiment 6 — System Prompt Re-injection
============================================
Tests whether re-injecting the system prompt mid-conversation can reset drift.
Compares recovery strength across languages.

Hypothesis: Higher TFR languages show weaker recovery because the system prompt
is compressed (takes proportionally less attention budget).
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_experiment_6(
    model_manager,
    model_configs: list[dict],
    data_dir: Path,
    system_prompt: str,
    output_dir: Path,
    languages: list[str] = None,
    reinjection_turns: list[int] = None,
    max_conversations: int = 10,
):
    """
    Run Experiment 6: System Prompt Re-injection.
    
    For each model × language:
    1. Run normal inference (baseline from Exp 2)
    2. Run with re-injection at specified turns
    3. Compare DDM scores at re-injection points
    """
    from src.evaluation.ddm import DDMEvaluator
    from src.experiments.inference import (
        load_conversation, run_conversation_inference, run_reinjection_inference,
    )

    languages = languages or ["en", "it", "es", "fr", "de"]
    reinjection_turns = reinjection_turns or [15, 30, 50]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: System Prompt Re-injection")
    logger.info(f"  Re-injection at turns: {reinjection_turns}")
    logger.info("=" * 60)

    all_results = []

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        hf_id = model_cfg["hf_id"]

        model, tokenizer = model_manager.load(hf_id)

        for lang in languages:
            # Use medium tier (most likely to show drift and recovery)
            lang_dir = data_dir / "medium" / "parallel" / lang
            if not lang_dir.exists():
                continue

            jsonl_files = sorted(lang_dir.glob("*.jsonl"))[:max_conversations]
            evaluator = DDMEvaluator(language=lang, strict_citations=False)

            for jsonl_file in jsonl_files:
                conv_id = jsonl_file.stem
                messages = load_conversation(jsonl_file)
                user_msgs = [m["content"] for m in messages if m["role"] == "user"]

                if len(user_msgs) < max(reinjection_turns):
                    continue

                # Run WITH re-injection
                reinject_result = run_reinjection_inference(
                    model, tokenizer, user_msgs, system_prompt,
                    reinjection_turns=reinjection_turns,
                )

                # Evaluate
                ddm_result = evaluator.evaluate_conversation(
                    reinject_result.responses, conv_id, lang, model_name,
                )

                # Compute recovery boost at each re-injection point
                for rt in reinjection_turns:
                    if rt < len(ddm_result.turn_results) and rt > 0:
                        before = ddm_result.turn_results[rt - 2].ddm_score if rt >= 2 else 1.0
                        after = ddm_result.turn_results[rt].ddm_score if rt < len(ddm_result.turn_results) else before
                        boost = after - before

                        all_results.append({
                            "conversation_id": conv_id,
                            "language": lang,
                            "model": model_name,
                            "reinjection_turn": rt,
                            "ddm_before": before,
                            "ddm_after": after,
                            "recovery_boost": boost,
                            "mean_ddm": ddm_result.mean_ddm,
                            "auc": ddm_result.auc,
                        })

        model_manager.unload()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "reinjection_results.csv", index=False)

    # Summary by language
    if not results_df.empty:
        lang_summary = results_df.groupby("language").agg({
            "recovery_boost": ["mean", "std"],
            "ddm_before": "mean",
            "ddm_after": "mean",
        }).reset_index()
        lang_summary.columns = ["_".join(c).strip("_") for c in lang_summary.columns]
        lang_summary.to_csv(output_dir / "reinjection_by_language.csv", index=False)

    logger.info(f"  Saved re-injection results to {output_dir}/")
    return results_df
