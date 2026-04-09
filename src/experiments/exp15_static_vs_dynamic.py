"""
Experiment 15: Static vs Dynamic Drift Delta
=============================================
Compares Track 1 (Static-Sterile, CAT-D) with Track 2 (Dynamic-Agentic,
DeepSeek User Simulator) to determine whether instruction drift is caused
by real context-length attention decay or by benchmark artifact confusion.

The Claim:
    If DOP_static ≈ DOP_dynamic → Drift is a real model limitation,
    NOT a confound from trajectory mismatch in our static benchmark.

Design:
    For each (model, domain, tier):
    1. Run Track 1: static user messages from pre-generated JSON
    2. Run Track 2: dynamic user messages from DeepSeek API
    3. Compare DOP, AUC, and per-level decay curves

Output:
    - static_vs_dynamic_results.csv
    - drift_delta_analysis.json
"""

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.evaluation.ddm import DDMEvaluator
from src.experiments.inference import (
    run_conversation_inference,
    run_dynamic_inference,
    InferenceResult,
)
from src.experiments.dynamic_user import DynamicUserSimulator
from src.data_gen.seed_generator import DOMAIN_TEMPLATES

logger = logging.getLogger(__name__)


@dataclass
class DriftDeltaResult:
    """Comparison between static and dynamic drift for one configuration."""
    model_name: str
    domain: str
    difficulty: str
    num_turns: int
    # Track 1 (Static)
    static_dop: int = None
    static_auc: float = 0.0
    static_mean_ddm: float = 0.0
    # Track 2 (Dynamic)
    dynamic_dop: int = None
    dynamic_auc: float = 0.0
    dynamic_mean_ddm: float = 0.0
    # Delta
    dop_delta: float = None      # dynamic_dop - static_dop
    auc_delta: float = 0.0       # dynamic_auc - static_auc
    # Per-level DOPs
    static_l1_dop: int = None
    static_l5_dop: int = None
    dynamic_l1_dop: int = None
    dynamic_l5_dop: int = None


def run_experiment_15(
    model,
    tokenizer,
    model_name: str,
    data_dir: str,
    output_dir: str,
    system_prompt: str,
    domains: list[str] = None,
    num_turns: int = 50,
    config: dict = None,
) -> pd.DataFrame:
    """
    Run Experiment 15: Static vs Dynamic Drift Delta.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Loaded tokenizer.
        model_name: Model identifier.
        data_dir: Directory containing pre-generated conversations (Track 1).
        output_dir: Where to save results.
        system_prompt: DDM system prompt with L1-L5 rules.
        domains: List of domains to test (default: 3 representative).
        num_turns: Number of turns for dynamic simulation.
        config: Project config dict.

    Returns:
        DataFrame with drift delta results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default: test 3 representative domains (1 easy, 1 medium, 1 hard)
    if domains is None:
        domains = _select_representative_domains()

    evaluator = DDMEvaluator(config=config, language="en")
    results = []

    for domain in domains:
        template = DOMAIN_TEMPLATES.get(domain)
        if not template:
            logger.warning(f"Domain {domain} not found, skipping")
            continue

        difficulty = template.get("difficulty", "unknown")
        logger.info(f"\n{'='*60}")
        logger.info(f"Exp 15: {model_name} × {domain} ({difficulty})")
        logger.info(f"{'='*60}")

        # ── Track 1: Static (read from JSON) ──
        logger.info("Track 1: Static-Sterile (CAT-D)...")
        static_result = _run_static_track(
            model, tokenizer, model_name, domain, data_dir,
            system_prompt, evaluator,
        )

        # ── Track 2: Dynamic (DeepSeek user simulator) ──
        logger.info("Track 2: Dynamic-Agentic (DeepSeek)...")
        dynamic_result = _run_dynamic_track(
            model, tokenizer, model_name, domain,
            system_prompt, evaluator, num_turns,
        )

        # ── Compute Delta ──
        delta = _compute_delta(
            model_name, domain, difficulty, num_turns,
            static_result, dynamic_result,
        )
        results.append(delta)

        logger.info(
            f"  Static DOP={delta.static_dop}, Dynamic DOP={delta.dynamic_dop}, "
            f"Delta={delta.dop_delta}"
        )

    # ── Save Results ──
    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = output_path / "static_vs_dynamic_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved: {csv_path}")

    # Save detailed JSON
    json_path = output_path / "drift_delta_analysis.json"
    with open(json_path, "w") as f:
        json.dump({
            "model": model_name,
            "domains_tested": domains,
            "results": [asdict(r) for r in results],
            "summary": _compute_summary(results),
        }, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return df


def _select_representative_domains() -> list[str]:
    """Select 3 representative domains: 1 easy, 1 medium, 1 hard."""
    easy = medium = hard = None
    for name, tmpl in DOMAIN_TEMPLATES.items():
        diff = tmpl.get("difficulty", "easy")
        if diff == "easy" and easy is None:
            easy = name
        elif diff == "medium" and medium is None:
            medium = name
        elif diff == "hard" and hard is None:
            hard = name
    return [d for d in [easy, medium, hard] if d is not None]


def _run_static_track(model, tokenizer, model_name, domain, data_dir,
                       system_prompt, evaluator):
    """Run Track 1: static user messages from JSON."""
    data_path = Path(data_dir)

    # Find a conversation file for this domain
    conv_files = sorted(data_path.glob(f"*_{domain}.json"))
    if not conv_files:
        # Try medium tier
        conv_files = sorted(data_path.glob(f"conv_*_{domain}.json"))
    if not conv_files:
        logger.warning(f"No static data found for domain {domain}")
        return None

    # Use the first matching file
    conv_file = conv_files[0]
    with open(conv_file) as f:
        conv_data = json.load(f)

    # Extract user messages
    messages = conv_data.get("messages", [])
    user_messages = [m["content"] for m in messages if m["role"] == "user"]

    if not user_messages:
        logger.warning(f"No user messages in {conv_file}")
        return None

    # Run inference
    inference_result = run_conversation_inference(
        model=model,
        tokenizer=tokenizer,
        user_messages=user_messages,
        system_prompt=system_prompt,
    )

    # Evaluate with DDM
    ddm_result = evaluator.evaluate_conversation(
        inference_result.responses,
        conversation_id=f"static_{domain}",
        language="en",
        model_name=model_name,
    )
    ddm_result.compute_summary()
    return ddm_result


def _run_dynamic_track(model, tokenizer, model_name, domain,
                        system_prompt, evaluator, num_turns):
    """Run Track 2: dynamic user messages from DeepSeek."""
    try:
        dynamic_user = DynamicUserSimulator.from_domain_template(
            domain=domain,
            num_turns=num_turns,
        )
    except (ValueError, ImportError) as e:
        logger.error(f"Could not create DynamicUserSimulator: {e}")
        return None

    # Run dynamic inference
    inference_result = run_dynamic_inference(
        model=model,
        tokenizer=tokenizer,
        dynamic_user=dynamic_user,
        system_prompt=system_prompt,
        num_turns=num_turns,
    )

    # Evaluate with DDM
    ddm_result = evaluator.evaluate_conversation(
        inference_result.responses,
        conversation_id=f"dynamic_{domain}",
        language="en",
        model_name=model_name,
    )
    ddm_result.compute_summary()
    return ddm_result


def _compute_delta(model_name, domain, difficulty, num_turns,
                    static_result, dynamic_result) -> DriftDeltaResult:
    """Compute the drift delta between static and dynamic tracks."""
    delta = DriftDeltaResult(
        model_name=model_name,
        domain=domain,
        difficulty=difficulty,
        num_turns=num_turns,
    )

    if static_result:
        delta.static_dop = static_result.drift_onset_point
        delta.static_auc = static_result.auc
        delta.static_mean_ddm = static_result.mean_ddm
        if "L1_format" in static_result.per_level_decay:
            delta.static_l1_dop = static_result.per_level_decay["L1_format"].decay_onset
        if "L5_dynamic" in static_result.per_level_decay:
            delta.static_l5_dop = static_result.per_level_decay["L5_dynamic"].decay_onset

    if dynamic_result:
        delta.dynamic_dop = dynamic_result.drift_onset_point
        delta.dynamic_auc = dynamic_result.auc
        delta.dynamic_mean_ddm = dynamic_result.mean_ddm
        if "L1_format" in dynamic_result.per_level_decay:
            delta.dynamic_l1_dop = dynamic_result.per_level_decay["L1_format"].decay_onset
        if "L5_dynamic" in dynamic_result.per_level_decay:
            delta.dynamic_l5_dop = dynamic_result.per_level_decay["L5_dynamic"].decay_onset

    # Compute deltas
    if delta.static_dop is not None and delta.dynamic_dop is not None:
        delta.dop_delta = delta.dynamic_dop - delta.static_dop
    delta.auc_delta = delta.dynamic_auc - delta.static_auc

    return delta


def _compute_summary(results: list[DriftDeltaResult]) -> dict:
    """Compute aggregate summary statistics."""
    dop_deltas = [r.dop_delta for r in results if r.dop_delta is not None]
    auc_deltas = [r.auc_delta for r in results]

    summary = {
        "num_comparisons": len(results),
        "num_with_dop_delta": len(dop_deltas),
    }

    if dop_deltas:
        summary["mean_dop_delta"] = float(np.mean(dop_deltas))
        summary["std_dop_delta"] = float(np.std(dop_deltas))
        summary["interpretation"] = (
            "DRIFT IS REAL (not benchmark artifact)"
            if abs(np.mean(dop_deltas)) < 3
            else "SIGNIFICANT DIFFERENCE — investigate confounds"
        )

    if auc_deltas:
        summary["mean_auc_delta"] = float(np.mean(auc_deltas))

    return summary
