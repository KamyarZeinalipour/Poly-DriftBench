#!/usr/bin/env python3
"""
Multi-GPU Full Experiment Runner
==================================
Splits models across GPU 2 and GPU 3 for 2x speedup.

GPU 2: Models 0-5 (llama-3.1-8b, llama-3.2-3b, llama-3.2-1b, mistral-7b, mistral-nemo-12b, mistral-small-24b)
GPU 3: Models 6-11 (qwen2.5-3b, qwen2.5-7b, qwen2.5-14b, qwen2.5-32b, gemma-2-9b, phi-3.5-mini)

Usage:
    python scripts/run_multi_gpu.py
"""

import os
import sys
import json
import gc
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Environment ──
os.environ.setdefault("HF_HUB_CACHE", "/home4/kamyar/italian_detox/hf_cache/hub")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Args ──
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True, help="GPU index (2 or 3)")
parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "production"))
parser.add_argument("--max-convs", type=int, default=None, help="Max convs per tier (None=all)")
args = parser.parse_args()

GPU_ID = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# ── Logging ──
log_file = PROJECT_ROOT / f"experiment_gpu{GPU_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s │ GPU{GPU_ID} │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(f"gpu{GPU_ID}")

# ── Import after env setup ──
import torch
from src.experiments.inference import (
    ModelManager, load_conversation, run_conversation_inference,
)
from src.evaluation.ddm import DDMEvaluator, build_system_prompt

DDM_SYSTEM_PROMPT = build_system_prompt()


def load_config():
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def run_exp2_for_models(model_configs, data_dir, output_dir, max_convs=None):
    """Run Experiment 2 (Baseline Drift) for a subset of models."""
    data_path = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = ["en", "it", "es", "fr", "de"]
    tiers = ["short", "medium", "long"]

    manager = ModelManager(device="cuda:0")  # Maps to actual GPU via CUDA_VISIBLE_DEVICES
    all_turn_rows = []
    all_summary_rows = []
    all_per_level = {}
    context_lengths_data = {}

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        hf_id = model_cfg["hf_id"]
        context_window = model_cfg.get("context_window", 8192)

        logger.info(f"\n{'━' * 60}")
        logger.info(f"MODEL: {model_name} ({hf_id})")
        logger.info(f"Context window: {context_window:,}")
        logger.info(f"{'━' * 60}")

        try:
            model, tokenizer = manager.load(hf_id)
        except Exception as e:
            logger.error(f"  ❌ Failed to load {model_name}: {e}")
            continue

        for tier in tiers:
            for lang in languages:
                lang_dir = data_path / tier / "parallel" / lang
                if not lang_dir.exists():
                    logger.warning(f"  Missing: {lang_dir}")
                    continue

                evaluator = DDMEvaluator(language=lang, strict_citations=False)
                jsonl_files = sorted(lang_dir.glob("*.jsonl"))

                if max_convs:
                    jsonl_files = jsonl_files[:max_convs]

                for jsonl_file in tqdm(
                    jsonl_files,
                    desc=f"  {model_name}|{lang}|{tier}",
                    leave=False,
                ):
                    conv_id = jsonl_file.stem
                    messages = load_conversation(jsonl_file)
                    user_msgs = [m["content"] for m in messages if m["role"] == "user"]

                    if not user_msgs:
                        continue

                    try:
                        inf_result = run_conversation_inference(
                            model, tokenizer, user_msgs,
                            system_prompt=DDM_SYSTEM_PROMPT,
                            compute_perplexity=True,
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"    OOM: {conv_id} ({tier}/{lang})")
                        torch.cuda.empty_cache()
                        continue
                    except Exception as e:
                        logger.error(f"    Error: {conv_id}: {e}")
                        continue

                    # Evaluate DDM
                    model_tier = f"{model_name}-{tier}"
                    ddm_result = evaluator.evaluate_conversation(
                        inf_result.responses, conv_id, lang, model_tier,
                    )

                    # Collect per-turn rows
                    for tr in ddm_result.turn_results:
                        all_turn_rows.append({
                            "model": model_tier,
                            "language": lang,
                            "conversation_id": conv_id,
                            "turn": tr.turn_number,
                            "ddm_score": tr.ddm_score,
                            "l1_score": tr.l1_score,
                            "l2_score": tr.l2_score,
                            "l3_score": tr.l3_score,
                            "l4_score": tr.l4_score,
                        })

                    # Collect summary row
                    all_summary_rows.append({
                        "model": model_tier,
                        "language": lang,
                        "conversation_id": conv_id,
                        "mean_ddm": ddm_result.mean_ddm,
                        "auc": ddm_result.auc,
                        "drift_onset": ddm_result.drift_onset_point,
                        "sustained_dop": ddm_result.sustained_dop,
                        "half_life": ddm_result.half_life,
                        "recovery_rate": ddm_result.recovery_rate,
                        "total_turns": len(ddm_result.turn_results),
                    })

                    # Context lengths for Exp 7
                    key = f"{model_tier}|{lang}|{conv_id}"
                    context_lengths_data[key] = inf_result.context_lengths_tokens

                    # Per-level decay for Exp 12
                    if hasattr(ddm_result, 'per_level_decay'):
                        all_per_level[key] = ddm_result.per_level_decay

        manager.unload()

    # ── Save Results ──
    logger.info(f"\n{'═' * 60}")
    logger.info(f"SAVING RESULTS — GPU {GPU_ID}")
    logger.info(f"{'═' * 60}")

    if all_turn_rows:
        pd.DataFrame(all_turn_rows).to_csv(
            output_dir / f"drift_results_gpu{GPU_ID}.csv", index=False
        )
    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_csv(
            output_dir / f"drift_summary_gpu{GPU_ID}.csv", index=False
        )
    if context_lengths_data:
        with open(output_dir / f"context_lengths_gpu{GPU_ID}.json", "w") as f:
            json.dump(context_lengths_data, f)
    if all_per_level:
        with open(output_dir / f"per_level_decay_gpu{GPU_ID}.json", "w") as f:
            json.dump(all_per_level, f, default=str)

    logger.info(f"  ✅ {len(all_summary_rows)} conversations evaluated")
    logger.info(f"  ✅ {len(all_turn_rows)} turns recorded")
    logger.info(f"  Results: {output_dir}/")

    return all_summary_rows


def main():
    config = load_config()
    all_models = config["models"]

    # Split models between GPU 2 and GPU 3
    mid = len(all_models) // 2
    if GPU_ID == 2:
        my_models = all_models[:mid]  # First 6
    else:
        my_models = all_models[mid:]  # Last 6

    model_names = [m["name"] for m in my_models]
    logger.info(f"{'═' * 60}")
    logger.info(f"POLY-DRIFTBENCH — GPU {GPU_ID}")
    logger.info(f"{'═' * 60}")
    logger.info(f"  Models: {model_names}")
    logger.info(f"  Data: {args.data_dir}")
    logger.info(f"  Max convs/tier: {args.max_convs or 'ALL'}")
    logger.info(f"  Log: {log_file}")
    logger.info(f"{'═' * 60}\n")

    t0 = time.time()

    # Phase 1: Exp 2 (Drift Measurement) — the core experiment
    output_dir = PROJECT_ROOT / "results" / "drift_curves"
    run_exp2_for_models(
        my_models,
        args.data_dir,
        output_dir,
        max_convs=args.max_convs,
    )

    elapsed = (time.time() - t0) / 60
    logger.info(f"\n{'═' * 60}")
    logger.info(f"GPU {GPU_ID} COMPLETE — {elapsed:.1f} minutes")
    logger.info(f"{'═' * 60}")


if __name__ == "__main__":
    main()
