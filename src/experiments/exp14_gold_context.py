"""
Experiment 14: Gold-Context Scaffolding Analysis
=================================================
Decomposes instruction-following drift into two independent components:

    1. Pure Instruction Forgetting — the model losing the system prompt
    2. Cascade Damage — the model's own bad outputs reinforcing rule-breaking

Runs inference in multiple scaffolding modes:
    - free-form (baseline, Exp 2)
    - gold_full (all gold context — isolates pure forgetting)
    - gold_until_N (gold for first N turns, then free — measures cascade onset)
    - gold_ratio_R (R fraction of turns use gold — measures scaffolding density)

The delta between free-form and gold_full DDM scores is the "cascade damage"
component, a novel metric not measured in prior work.

Output:
    - gold_context_results.csv: Per-turn DDM scores for each scaffolding mode
    - gold_context_summary.csv: Conversation-level summary with decomposed drift
    - scaffolding_curve.csv: DDM at final turn as a function of gold_until / gold_ratio
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Scaffolding Configurations ─────────────────────────────

# gold_until sweep: test different scaffolding removal points
DEFAULT_GOLD_UNTIL_SWEEP = [0, 3, 5, 10, 15, 20, 30, 50]

# gold_ratio sweep: test different scaffolding densities
DEFAULT_GOLD_RATIO_SWEEP = [0.0, 0.1, 0.2, 0.33, 0.5, 0.75, 1.0]


def run_experiment_14(
    model_manager,
    model_configs: list[dict],
    data_dir: Path,
    system_prompt: str,
    output_dir: Path,
    languages: list[str] = None,
    tiers: list[str] = None,
    max_conversations: int = 5,
    gold_until_sweep: list[int] = None,
    gold_ratio_sweep: list[float] = None,
    compute_perplexity: bool = False,
) -> pd.DataFrame:
    """
    Run Experiment 14: Gold-Context Scaffolding Analysis.

    Args:
        model_manager: ModelManager instance for GPU inference.
        model_configs: List of model config dicts (name, hf_id, etc.).
        data_dir: Root data directory containing tier/parallel/lang structure.
        system_prompt: The DDM system prompt.
        output_dir: Where to save results.
        languages: Languages to evaluate (default: all available).
        tiers: Tiers to evaluate (default: ["medium", "long"]).
        max_conversations: Max conversations per tier×language.
        gold_until_sweep: List of gold_until values to test.
        gold_ratio_sweep: List of gold_ratio values to test.
        compute_perplexity: Whether to compute response perplexity.

    Returns:
        Summary DataFrame with decomposed drift metrics.
    """
    from src.evaluation.ddm import DDMEvaluator
    from src.experiments.inference import (
        run_conversation_inference,
        run_gold_context_inference,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiers = tiers or ["medium", "long"]
    gold_until_sweep = gold_until_sweep or DEFAULT_GOLD_UNTIL_SWEEP
    gold_ratio_sweep = gold_ratio_sweep or DEFAULT_GOLD_RATIO_SWEEP

    all_rows = []         # Per-turn results
    summary_rows = []     # Conversation-level summaries
    scaffolding_rows = [] # Scaffolding curve data

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        hf_id = model_cfg["hf_id"]

        logger.info(f"\n{'─' * 50}")
        logger.info(f"Exp 14 — Model: {model_name}")

        model, tokenizer = model_manager.load(hf_id)

        for tier in tiers:
            tier_dir = data_dir / tier

            # Discover available languages
            available_langs = _discover_languages(tier_dir, languages)
            if not available_langs:
                logger.warning(f"  No data for tier={tier}")
                continue

            for lang in available_langs:
                conversations = _load_conversations_with_gold(
                    tier_dir, lang, max_conversations
                )

                if not conversations:
                    continue

                logger.info(
                    f"  {model_name}|{lang}|{tier}: "
                    f"{len(conversations)} conversations"
                )

                for conv_id, user_msgs, gold_asst_msgs in conversations:
                    evaluator = DDMEvaluator(
                        language=lang, strict_citations=False
                    )

                    # ── Mode 1: Free-form (baseline) ──
                    try:
                        free_result = run_conversation_inference(
                            model, tokenizer, user_msgs,
                            system_prompt=system_prompt,
                            compute_perplexity=compute_perplexity,
                        )
                        free_result.conversation_id = conv_id
                        free_result.language = lang
                        free_result.model_name = model_name

                        free_ddm = evaluator.evaluate_conversation(
                            free_result.responses, conv_id, lang,
                            f"{model_name}-{tier}",
                        )
                    except Exception as e:
                        logger.error(f"    Free-form error on {conv_id}: {e}")
                        continue

                    # Store free-form per-turn results
                    for t in free_ddm.turn_results:
                        all_rows.append({
                            "model": model_name,
                            "tier": tier,
                            "language": lang,
                            "conversation_id": conv_id,
                            "mode": "free",
                            "turn": t.turn_number,
                            "l1": t.l1_score,
                            "l2": t.l2_score,
                            "l3": t.l3_score,
                            "l4": t.l4_score,
                            "ddm": t.ddm_score,
                            "gold_context": False,
                        })

                    # ── Mode 2: Full Gold Context ──
                    try:
                        gold_result = run_gold_context_inference(
                            model, tokenizer, user_msgs, gold_asst_msgs,
                            system_prompt=system_prompt,
                            gold_until=None, gold_ratio=None,  # = full gold
                            compute_perplexity=compute_perplexity,
                        )
                        gold_result.conversation_id = conv_id
                        gold_result.language = lang
                        gold_result.model_name = model_name

                        gold_ddm = evaluator.evaluate_conversation(
                            gold_result.responses, conv_id, lang,
                            f"{model_name}-{tier}-gold_full",
                        )
                    except Exception as e:
                        logger.error(f"    Gold-full error on {conv_id}: {e}")
                        continue

                    for t in gold_ddm.turn_results:
                        all_rows.append({
                            "model": model_name,
                            "tier": tier,
                            "language": lang,
                            "conversation_id": conv_id,
                            "mode": "gold_full",
                            "turn": t.turn_number,
                            "l1": t.l1_score,
                            "l2": t.l2_score,
                            "l3": t.l3_score,
                            "l4": t.l4_score,
                            "ddm": t.ddm_score,
                            "gold_context": True,
                        })

                    # ── Decompose Drift ──
                    cascade_damage = free_ddm.mean_ddm - gold_ddm.mean_ddm
                    pure_forgetting = 1.0 - gold_ddm.mean_ddm

                    summary_rows.append({
                        "model": model_name,
                        "tier": tier,
                        "language": lang,
                        "conversation_id": conv_id,
                        "n_turns": free_ddm.total_turns,
                        "free_mean_ddm": free_ddm.mean_ddm,
                        "free_auc": free_ddm.auc,
                        "free_dop": free_ddm.drift_onset_point,
                        "gold_mean_ddm": gold_ddm.mean_ddm,
                        "gold_auc": gold_ddm.auc,
                        "gold_dop": gold_ddm.drift_onset_point,
                        "cascade_damage": cascade_damage,
                        "pure_forgetting": pure_forgetting,
                        "cascade_pct": (
                            abs(cascade_damage) / max(1.0 - free_ddm.mean_ddm, 1e-6) * 100
                            if free_ddm.mean_ddm < 1.0 else 0.0
                        ),
                    })

                    # ── Mode 3: gold_until sweep ──
                    for gu in gold_until_sweep:
                        if gu == 0:
                            # Same as free-form, skip
                            scaffolding_rows.append({
                                "model": model_name, "tier": tier,
                                "language": lang, "conversation_id": conv_id,
                                "sweep_type": "gold_until", "sweep_value": 0,
                                "mean_ddm": free_ddm.mean_ddm,
                                "auc": free_ddm.auc,
                                "dop": free_ddm.drift_onset_point,
                            })
                            continue
                        if gu >= len(user_msgs):
                            # Same as full gold
                            scaffolding_rows.append({
                                "model": model_name, "tier": tier,
                                "language": lang, "conversation_id": conv_id,
                                "sweep_type": "gold_until",
                                "sweep_value": gu,
                                "mean_ddm": gold_ddm.mean_ddm,
                                "auc": gold_ddm.auc,
                                "dop": gold_ddm.drift_onset_point,
                            })
                            continue

                        try:
                            gu_result = run_gold_context_inference(
                                model, tokenizer, user_msgs, gold_asst_msgs,
                                system_prompt=system_prompt,
                                gold_until=gu,
                            )
                            gu_ddm = evaluator.evaluate_conversation(
                                gu_result.responses, conv_id, lang,
                                f"{model_name}-{tier}-gu{gu}",
                            )
                            scaffolding_rows.append({
                                "model": model_name, "tier": tier,
                                "language": lang, "conversation_id": conv_id,
                                "sweep_type": "gold_until",
                                "sweep_value": gu,
                                "mean_ddm": gu_ddm.mean_ddm,
                                "auc": gu_ddm.auc,
                                "dop": gu_ddm.drift_onset_point,
                            })
                        except Exception as e:
                            logger.warning(f"    gold_until={gu} error: {e}")

                    # ── Mode 4: gold_ratio sweep ──
                    for gr in gold_ratio_sweep:
                        if gr <= 0.0:
                            scaffolding_rows.append({
                                "model": model_name, "tier": tier,
                                "language": lang, "conversation_id": conv_id,
                                "sweep_type": "gold_ratio", "sweep_value": 0.0,
                                "mean_ddm": free_ddm.mean_ddm,
                                "auc": free_ddm.auc,
                                "dop": free_ddm.drift_onset_point,
                            })
                            continue
                        if gr >= 1.0:
                            scaffolding_rows.append({
                                "model": model_name, "tier": tier,
                                "language": lang, "conversation_id": conv_id,
                                "sweep_type": "gold_ratio", "sweep_value": 1.0,
                                "mean_ddm": gold_ddm.mean_ddm,
                                "auc": gold_ddm.auc,
                                "dop": gold_ddm.drift_onset_point,
                            })
                            continue

                        try:
                            gr_result = run_gold_context_inference(
                                model, tokenizer, user_msgs, gold_asst_msgs,
                                system_prompt=system_prompt,
                                gold_ratio=gr,
                            )
                            gr_ddm = evaluator.evaluate_conversation(
                                gr_result.responses, conv_id, lang,
                                f"{model_name}-{tier}-gr{gr:.2f}",
                            )
                            scaffolding_rows.append({
                                "model": model_name, "tier": tier,
                                "language": lang, "conversation_id": conv_id,
                                "sweep_type": "gold_ratio",
                                "sweep_value": gr,
                                "mean_ddm": gr_ddm.mean_ddm,
                                "auc": gr_ddm.auc,
                                "dop": gr_ddm.drift_onset_point,
                            })
                        except Exception as e:
                            logger.warning(f"    gold_ratio={gr} error: {e}")

        model_manager.unload()

    # ── Save Results ──
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_dir / "gold_context_results.csv", index=False)
        logger.info(f"  Saved per-turn results: {len(df)} rows")

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(output_dir / "gold_context_summary.csv", index=False)
        logger.info(f"  Saved summary: {len(sdf)} conversations")

        # Log key findings
        mean_cascade = sdf["cascade_pct"].mean()
        mean_forgetting = sdf["pure_forgetting"].mean()
        logger.info(f"\n  📊 Drift Decomposition:")
        logger.info(f"     Pure Forgetting:  {mean_forgetting:.3f}")
        logger.info(f"     Cascade Damage:   {mean_cascade:.1f}% of total drift")

    if scaffolding_rows:
        scdf = pd.DataFrame(scaffolding_rows)
        scdf.to_csv(output_dir / "scaffolding_curve.csv", index=False)
        logger.info(f"  Saved scaffolding curve: {len(scdf)} data points")

    return pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()


# ── Helpers ──────────────────────────────────────────────────

def _discover_languages(tier_dir: Path, languages: list[str] = None) -> list[str]:
    """Discover available languages in a tier directory."""
    parallel_dir = tier_dir / "parallel"
    if not parallel_dir.exists():
        # Try generated/ with JSON files
        gen_dir = tier_dir / "generated"
        if gen_dir.exists():
            return ["en"]  # JSON files contain all languages internally
        return []

    available = sorted(d.name for d in parallel_dir.iterdir() if d.is_dir())
    if languages:
        available = [l for l in available if l in languages]
    return available


def _load_conversations_with_gold(
    tier_dir: Path,
    language: str,
    max_conversations: int,
) -> list[tuple[str, list[str], list[str]]]:
    """
    Load conversations with both user messages and gold assistant responses.

    Returns list of (conv_id, user_messages, gold_assistant_messages).
    Supports both JSONL parallel format and JSON generated format.
    """
    conversations = []

    # Try JSONL parallel format first
    parallel_dir = tier_dir / "parallel" / language
    if parallel_dir.exists():
        jsonl_files = sorted(parallel_dir.glob("*.jsonl"))[:max_conversations]
        for jsonl_file in jsonl_files:
            conv_id = jsonl_file.stem
            user_msgs, asst_msgs = [], []
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    msg = json.loads(line)
                    role = msg.get("role", "user")
                    content = msg.get("text", msg.get("content", ""))
                    if role == "user":
                        user_msgs.append(content)
                    elif role == "assistant":
                        asst_msgs.append(content)
            if user_msgs and asst_msgs:
                conversations.append((conv_id, user_msgs, asst_msgs))

    # Try JSON generated format
    gen_dir = tier_dir / "generated"
    if gen_dir.exists() and not conversations:
        json_files = sorted(gen_dir.glob("conv_*.json"))[:max_conversations]
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            conv = data.get("conversations", {}).get(language, [])
            if not conv:
                continue
            conv_id = data.get("id", json_file.stem)
            user_msgs = [m["content"] for m in conv if m["role"] == "user"]
            asst_msgs = [m["content"] for m in conv if m["role"] == "assistant"]
            if user_msgs and asst_msgs:
                conversations.append((conv_id, user_msgs, asst_msgs))

    return conversations
