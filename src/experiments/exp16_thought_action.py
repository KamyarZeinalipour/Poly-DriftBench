"""
Experiment 16: Thought-Action Dissonance (System-2 Profiling)
==============================================================
Tests whether reasoning models (with hidden <think> blocks) suffer from:

  A. "Thought Drift" — the model FORGETS the rules even in its thinking
  B. "Execution Drift" — the model REMEMBERS the rules in <think> but
     fails to follow them in the final output (conversational momentum)

This is tested on models that produce <think>...</think> blocks
(e.g., DeepSeek-R1-Distill). The DDM evaluator is run separately on:
  1. The <think> block content (thought_ddm)
  2. The final output after </think> (output_ddm)

Metrics:
  - thought_ddm: DDM score of the reasoning trace
  - output_ddm:  DDM score of the final answer
  - dissonance:  thought_ddm - output_ddm (positive = knows but fails)
  - thought_dop: Turn where the model STOPS mentioning rules in <think>
  - output_dop:  Turn where the model STOPS following rules in output
  - dop_gap:     thought_dop - output_dop (negative = forgets before failing)

Key finding patterns:
  - dissonance >> 0 → "Execution Drift" (System-2 can't override System-1)
  - dissonance ≈ 0  → "Memory Eviction" (forgetting, not laziness)
  - thought_dop > output_dop → Rules vanish from thinking BEFORE output breaks
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Think Block Parser ──────────────────────────────────────

THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL | re.IGNORECASE,
)


def parse_think_output(response: str) -> tuple[str, str]:
    """
    Split a reasoning model response into thought and output.

    Args:
        response: Full model response potentially containing <think>...</think>.

    Returns:
        (think_content, output_content) tuple.
        If no <think> block found, returns ("", full_response).
    """
    match = THINK_PATTERN.search(response)
    if not match:
        return "", response.strip()

    think_content = match.group(1).strip()
    # Output is everything AFTER the </think> tag
    output_content = response[match.end():].strip()
    return think_content, output_content


def check_rule_awareness(think_content: str) -> dict:
    """
    Check if the model's thinking explicitly references the DDM rules.

    Returns a dict with boolean flags for each rule mentioned in thought.
    """
    think_lower = think_content.lower()
    return {
        "thinks_about_l1": any(p in think_lower for p in [
            "sys_ack", "active", "tag", "append",
        ]),
        "thinks_about_l2": any(p in think_lower for p in [
            "bullet", "numbered", "points", "format",
        ]),
        "thinks_about_l3": any(p in think_lower for p in [
            "however", "forbidden", "avoid", "banned word",
        ]),
        "thinks_about_l4": any(p in think_lower for p in [
            "source", "cite", "citation", "according",
        ]),
        "thinks_about_l5": any(p in think_lower for p in [
            "turn:", "turn number", "turn counter", "increment",
        ]),
    }


@dataclass
class ThoughtActionResult:
    """Per-turn result comparing thought vs output DDM."""
    turn: int
    thought_ddm: float = 0.0
    output_ddm: float = 0.0
    dissonance: float = 0.0  # thought_ddm - output_ddm
    has_think_block: bool = False
    rule_awareness: dict = field(default_factory=dict)
    # Per-level breakdown
    thought_l1: float = 0.0
    thought_l2: float = 0.0
    thought_l3: float = 0.0
    thought_l4: float = 0.0
    thought_l5: float = 0.0
    output_l1: float = 0.0
    output_l2: float = 0.0
    output_l3: float = 0.0
    output_l4: float = 0.0
    output_l5: float = 0.0


def run_experiment_16(
    model_manager,
    model_configs: list[dict],
    data_dir: Path,
    system_prompt: str,
    output_dir: Path,
    languages: list[str] = None,
    tiers: list[str] = None,
    max_conversations: int = 5,
) -> pd.DataFrame:
    """
    Run Experiment 16: Thought-Action Dissonance on reasoning models.

    Only runs on models with reasoning_model=True in config.
    """
    from src.evaluation.ddm import DDMEvaluator
    from src.experiments.inference import run_conversation_inference

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers = tiers or ["medium", "long"]

    # Filter to reasoning models only
    reasoning_models = [m for m in model_configs if m.get("reasoning_model")]
    if not reasoning_models:
        logger.warning("Exp 16: No reasoning models found in config. Skipping.")
        return pd.DataFrame()

    all_rows = []

    for model_cfg in reasoning_models:
        model_name = model_cfg["name"]
        hf_id = model_cfg["hf_id"]

        logger.info(f"\n{'─' * 50}")
        logger.info(f"Exp 16 — Reasoning Model: {model_name}")

        model, tokenizer = model_manager.load(hf_id)
        evaluator = DDMEvaluator(language="en")

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

                # Run inference
                result = run_conversation_inference(
                    model, tokenizer, user_messages,
                    system_prompt=system_prompt,
                )

                # Parse each response into thought + output
                for turn_idx, response in enumerate(result.responses):
                    think_content, output_content = parse_think_output(response)

                    has_think = bool(think_content)

                    # Evaluate DDM on output (normal)
                    output_turn = evaluator.evaluate_turn(
                        output_content, turn_number=turn_idx + 1
                    )

                    # Evaluate DDM on thought (if present)
                    if has_think:
                        thought_turn = evaluator.evaluate_turn(
                            think_content, turn_number=turn_idx + 1
                        )
                        awareness = check_rule_awareness(think_content)
                    else:
                        thought_turn = output_turn  # No thought = same as output
                        awareness = {}

                    dissonance = thought_turn.ddm_score - output_turn.ddm_score

                    row = {
                        "model": model_name,
                        "tier": tier,
                        "conversation_id": conv_id,
                        "turn": turn_idx + 1,
                        "has_think_block": has_think,
                        "thought_ddm": thought_turn.ddm_score,
                        "output_ddm": output_turn.ddm_score,
                        "dissonance": dissonance,
                        "thought_l1": thought_turn.l1_score,
                        "thought_l2": thought_turn.l2_score,
                        "thought_l3": thought_turn.l3_score,
                        "thought_l4": thought_turn.l4_score,
                        "thought_l5": thought_turn.l5_score,
                        "output_l1": output_turn.l1_score,
                        "output_l2": output_turn.l2_score,
                        "output_l3": output_turn.l3_score,
                        "output_l4": output_turn.l4_score,
                        "output_l5": output_turn.l5_score,
                        **{f"aware_{k}": v for k, v in awareness.items()},
                    }
                    all_rows.append(row)

        model_manager.unload()

    # Save results
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_dir / "thought_action_dissonance.csv", index=False)

        # Summary
        think_turns = df[df["has_think_block"]]
        if not think_turns.empty:
            mean_dissonance = think_turns["dissonance"].mean()
            exec_drift_pct = (think_turns["dissonance"] > 0.1).mean() * 100
            memory_evict_pct = (
                (think_turns["thought_ddm"] < 0.5) &
                (think_turns["output_ddm"] < 0.5)
            ).mean() * 100

            logger.info(f"\n  📊 Thought-Action Dissonance:")
            logger.info(f"     Mean dissonance:    {mean_dissonance:.3f}")
            logger.info(f"     Execution Drift:    {exec_drift_pct:.1f}% of turns")
            logger.info(f"     Memory Eviction:    {memory_evict_pct:.1f}% of turns")

        return df

    return pd.DataFrame()
