"""
Experiment 17: Diagnostic Probe — Memory Eviction vs Behavioral Laziness
=========================================================================
At the exact turn where drift onset is detected (DOP), injects a
diagnostic probe asking the model to recite its system rules verbatim.

This separates two failure modes:
  A. "Epistemic Eviction" — the model physically lost the system prompt
     from its effective context window (can't recite rules)
  B. "Behavioral Laziness" — the model still knows the rules but its
     autoregressive momentum overrides compliance (can recite rules
     but doesn't follow them)

Protocol:
  1. Run normal inference until DOP is detected (DDM < 1.0)
  2. Inject probe: "Recite the formatting rules you were given verbatim"
  3. Score the probe response for rule recall accuracy
  4. Resume normal conversation
  5. Repeat probe at DOP+5, DOP+10, DOP+20 to measure decay

Output:
  - diagnostic_probe_results.csv: Per-probe recall accuracy
  - eviction_vs_laziness.csv: Classification of each model's failure mode
"""

import re
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Diagnostic Probe ────────────────────────────────────────

DIAGNOSTIC_PROBE = (
    "IMPORTANT SYSTEM CHECK: Before answering any more questions, "
    "please recite ALL of the formatting rules you were given in your "
    "system instructions at the start of this conversation. List each "
    "rule exactly as you remember it. This is a diagnostic test."
)

# Keywords to check if the model remembers each rule
RULE_RECALL_PATTERNS = {
    "L1_recall": [
        r"sys.?ack",
        r"\[sys_ack:\s*active\]",
        r"append.*tag",
        r"active.*end",
    ],
    "L2_recall": [
        r"numbered.*bullet",
        r"bullet.*point",
        r"numbered.*point",
        r"at least.*2.*point",
        r"format.*number",
    ],
    "L3_recall": [
        r"never.*however",
        r"avoid.*however",
        r"forbidden.*however",
        r"must not.*however",
        r"don'?t.*however",
    ],
    "L4_recall": [
        r"cite.*source",
        r"source.*claim",
        r"\[source:",
        r"according to",
        r"factual.*cite",
    ],
    "L5_recall": [
        r"\[turn:\s*n\]",
        r"turn.*number",
        r"turn.*counter",
        r"increment.*turn",
        r"begin.*\[turn",
    ],
}


def score_rule_recall(response: str) -> dict:
    """
    Score how many rules the model can recite from memory.

    Returns dict with per-rule recall (0.0 or 1.0) and total recall score.
    """
    resp_lower = response.lower()
    scores = {}

    for rule_name, patterns in RULE_RECALL_PATTERNS.items():
        recalled = any(re.search(p, resp_lower) for p in patterns)
        scores[rule_name] = 1.0 if recalled else 0.0

    scores["total_recall"] = sum(scores.values()) / len(RULE_RECALL_PATTERNS)
    return scores


@dataclass
class ProbeResult:
    """Result of a single diagnostic probe injection."""
    model: str
    tier: str
    conversation_id: str
    probe_turn: int          # Turn at which probe was injected
    dop: int                 # Original drift onset point
    offset_from_dop: int     # How many turns after DOP
    total_recall: float      # 0.0 to 1.0
    l1_recall: float = 0.0
    l2_recall: float = 0.0
    l3_recall: float = 0.0
    l4_recall: float = 0.0
    l5_recall: float = 0.0
    ddm_at_probe: float = 0.0  # DDM score at the probe turn
    classification: str = ""   # "eviction" or "laziness"


def run_experiment_17(
    model_manager,
    model_configs: list[dict],
    data_dir: Path,
    system_prompt: str,
    output_dir: Path,
    languages: list[str] = None,
    tiers: list[str] = None,
    max_conversations: int = 5,
    probe_offsets: list[int] = None,
) -> pd.DataFrame:
    """
    Run Experiment 17: Diagnostic Probe at Drift Onset.

    Args:
        probe_offsets: Turns after DOP at which to inject probes.
            Default: [0, 5, 10, 20] — probes at DOP, DOP+5, DOP+10, DOP+20.
    """
    from src.evaluation.ddm import DDMEvaluator
    from src.experiments.inference import run_conversation_inference

    try:
        import torch
        from transformers import GenerationConfig
    except ImportError:
        logger.error("PyTorch/Transformers required for Exp 17")
        return pd.DataFrame()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers = tiers or ["medium", "long"]
    probe_offsets = probe_offsets or [0, 5, 10, 20]

    all_probes = []

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        hf_id = model_cfg["hf_id"]

        logger.info(f"\n{'─' * 50}")
        logger.info(f"Exp 17 — Model: {model_name}")

        model, tokenizer = model_manager.load(hf_id)
        evaluator = DDMEvaluator(language="en")

        gen_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        for tier in tiers:
            tier_dir = data_dir / tier / "generated"
            if not tier_dir.exists():
                continue

            conv_files = sorted(tier_dir.glob("conv_*.json"))[:max_conversations]

            for conv_file in conv_files:
                with open(conv_file) as f:
                    conv_data = json.load(f)

                lang = "en"
                conversations = conv_data.get("conversations", {})
                if isinstance(conversations, dict) and lang in conversations:
                    messages = conversations[lang]
                else:
                    messages = conv_data.get("messages", [])

                user_messages = [m["content"] for m in messages if m["role"] == "user"]
                if not user_messages:
                    continue

                conv_id = conv_data.get("id", conv_file.stem)

                # Phase 1: Run normal inference to find DOP
                result = run_conversation_inference(
                    model, tokenizer, user_messages,
                    system_prompt=system_prompt,
                )

                ddm_result = evaluator.evaluate_conversation(
                    result.responses, conv_id, lang, model_name,
                    context_lengths=result.context_lengths_tokens,
                )
                ddm_result.compute_summary()

                dop = ddm_result.drift_onset_point
                if dop is None:
                    logger.info(f"  {conv_id}: No drift detected, skipping probes")
                    continue

                logger.info(f"  {conv_id}: DOP={dop}, injecting probes at offsets {probe_offsets}")

                # Phase 2: For each probe offset, replay conversation up to
                # that turn, then inject the diagnostic probe
                for offset in probe_offsets:
                    probe_turn = dop + offset
                    if probe_turn > len(user_messages):
                        continue

                    # Build conversation up to probe_turn
                    conversation = [{"role": "system", "content": system_prompt}]
                    for t in range(probe_turn):
                        conversation.append({"role": "user", "content": user_messages[t]})
                        if t < len(result.responses):
                            conversation.append({"role": "assistant", "content": result.responses[t]})

                    # Inject probe
                    conversation.append({"role": "user", "content": DIAGNOSTIC_PROBE})

                    # Generate probe response
                    try:
                        input_text = tokenizer.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        input_text = "\n".join(
                            f"{m['role']}: {m['content']}" for m in conversation
                        )

                    input_ids = tokenizer.encode(
                        input_text, return_tensors="pt", truncation=True
                    ).to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(input_ids, generation_config=gen_config)

                    probe_response = tokenizer.decode(
                        outputs[0][input_ids.shape[1]:], skip_special_tokens=True
                    ).strip()

                    # Score recall
                    recall = score_rule_recall(probe_response)

                    # Get DDM at this turn
                    ddm_at_probe = 0.0
                    if probe_turn - 1 < len(ddm_result.turn_results):
                        ddm_at_probe = ddm_result.turn_results[probe_turn - 1].ddm_score

                    # Classify: eviction vs laziness
                    if recall["total_recall"] >= 0.6 and ddm_at_probe < 0.8:
                        classification = "laziness"   # Knows rules, doesn't follow
                    elif recall["total_recall"] < 0.4:
                        classification = "eviction"   # Can't recall rules
                    else:
                        classification = "partial"     # Mixed

                    probe_result = ProbeResult(
                        model=model_name,
                        tier=tier,
                        conversation_id=conv_id,
                        probe_turn=probe_turn,
                        dop=dop,
                        offset_from_dop=offset,
                        total_recall=recall["total_recall"],
                        l1_recall=recall["L1_recall"],
                        l2_recall=recall["L2_recall"],
                        l3_recall=recall["L3_recall"],
                        l4_recall=recall["L4_recall"],
                        l5_recall=recall["L5_recall"],
                        ddm_at_probe=ddm_at_probe,
                        classification=classification,
                    )
                    all_probes.append(asdict(probe_result))

                    logger.info(
                        f"    Probe@{probe_turn} (DOP+{offset}): "
                        f"recall={recall['total_recall']:.1%}, "
                        f"DDM={ddm_at_probe:.2f}, "
                        f"class={classification}"
                    )

        model_manager.unload()

    # Save results
    if all_probes:
        df = pd.DataFrame(all_probes)
        df.to_csv(output_dir / "diagnostic_probe_results.csv", index=False)

        # Summary
        eviction_pct = (df["classification"] == "eviction").mean() * 100
        laziness_pct = (df["classification"] == "laziness").mean() * 100
        partial_pct = (df["classification"] == "partial").mean() * 100

        logger.info(f"\n  📊 Diagnostic Probe Summary:")
        logger.info(f"     Epistemic Eviction: {eviction_pct:.1f}%")
        logger.info(f"     Behavioral Laziness: {laziness_pct:.1f}%")
        logger.info(f"     Partial/Mixed:       {partial_pct:.1f}%")

        # Recall decay over offset
        for offset in probe_offsets:
            subset = df[df["offset_from_dop"] == offset]
            if not subset.empty:
                mean_recall = subset["total_recall"].mean()
                logger.info(f"     Recall@DOP+{offset}: {mean_recall:.1%}")

        return df

    return pd.DataFrame()
