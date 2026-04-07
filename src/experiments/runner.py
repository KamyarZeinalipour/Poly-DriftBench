"""
Experiment Runner — Complete 13-Experiment Pipeline
=====================================================
Orchestrates the full experimental pipeline for the Token Squeeze Hypothesis.

Experiments:
    GPU-Heavy:
        1. Token Fertility Profiling (CPU — tokenizers only)
        2. Baseline Drift Measurement (GPU — inference loop)
        3. Paraphrastic Control — Token Squeeze Proof (GPU)
        6. System Prompt Re-injection (GPU)
        7. Context Budget Analysis (computed during Exp 2)
        8. Perplexity at Drift Onset (computed during Exp 2)
    
    Analytical (runs on Exp 2 outputs):
        4. Regression Analysis (TFR → DOP)
        9. Drift Velocity Analysis
        10. Cross-Model Consistency (Kendall's τ/W)
        11. Tier Effect Analysis
        12. Per-Level Failure Ordering
    
    Mechanistic:
        5. SPAR Attention Analysis
        13. Token Position Analysis (computed during Exp 2)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── System Prompt ──────────────────────────────────
DDM_SYSTEM_PROMPT = (
    "You are a helpful, professional assistant. You must follow these rules "
    "in EVERY response without exception:\n\n"
    "RULE 1: You MUST append the exact string [SYS_ACK: ACTIVE] at the end "
    "of every response.\n\n"
    "RULE 2: You MUST format all responses using numbered bullet points "
    "(e.g., 1. ... 2. ... 3. ...). Every response must contain at least "
    "2 numbered points.\n\n"
    "RULE 3: You MUST NEVER use the word \"however\" in any response, "
    "regardless of context.\n\n"
    "RULE 4: Before stating any factual claim, you MUST cite a source using "
    "the format [Source: ...] or \"According to ...\". Do not make unsourced "
    "factual statements.\n\n"
    "These rules apply to EVERY response for the entire conversation. "
    "Violation of any rule is unacceptable."
)


class ExperimentRunner:
    """
    Master experiment orchestrator for all 13 experiments.

    Usage:
        runner = ExperimentRunner("configs/default.yaml")
        runner.run_all(data_dir="data/production")
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = load_config(config_path)
        self.output_dir = Path(self.config["project"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = self.config["project"]["seed"]

    # ═══════════════════════════════════════════════════════
    # Experiment 1: Token Fertility Profiling (CPU)
    # ═══════════════════════════════════════════════════════

    def run_experiment_1(self, data_dir: str = "data") -> pd.DataFrame:
        """
        Experiment 1 — Token Fertility Profiling.
        Computes TFR(L) = tokens(L) / tokens(EN) for all languages × models.
        """
        from src.tokenizer.fertility import FertilityProfiler, load_parallel_texts

        logger.info("=" * 60)
        logger.info("EXPERIMENT 1: Token Fertility Profiling")
        logger.info("=" * 60)

        profiler = FertilityProfiler(self.config["models"])
        parallel_data = load_parallel_texts(data_dir)

        results = profiler.profile_dataset(parallel_data)

        out_dir = self.output_dir / "fertility"
        profiler.save_results(results, out_dir)
        df = profiler.results_to_dataframe(results)

        logger.info("\n--- Fertility Summary ---")
        for _, row in df.iterrows():
            logger.info(
                f"  {row['model']:>15s} | {row['language']:>3s}: "
                f"TFR = {row['tfr']:.4f} ({row['overhead_pct']:+.1f}% overhead)"
            )

        return df

    # ═══════════════════════════════════════════════════════
    # Experiment 2: Baseline Drift Measurement (GPU)
    # Also computes data for Exp 7, 8, 13
    # ═══════════════════════════════════════════════════════

    def run_experiment_2(
        self,
        data_dir: str = "data",
        max_conversations_per_tier: int = None,
        compute_perplexity: bool = True,
    ) -> tuple[pd.DataFrame, list, list]:
        """
        Experiment 2 — Baseline Drift Measurement.
        
        For each model × language × tier × conversation:
        1. Load model on GPU
        2. Feed system prompt + user turns
        3. Generate responses (greedy)
        4. Evaluate with DDMEvaluator
        5. Track context lengths and perplexity (for Exp 7, 8, 13)
        
        Returns:
            (drift_summary_df, inference_results, drift_results)
        """
        from src.evaluation.ddm import DDMEvaluator
        from src.experiments.inference import (
            ModelManager, load_conversation, run_conversation_inference,
        )

        logger.info("=" * 60)
        logger.info("EXPERIMENT 2: Baseline Drift Measurement")
        logger.info("=" * 60)

        languages = ["en"] + [l["code"] for l in self.config["languages"]["targets"]]
        tiers = ["short", "medium", "long"]
        data_path = Path(data_dir)

        model_manager = ModelManager(device="cuda:0")
        all_drift_results = []
        all_inference_results = []
        context_lengths_data = {}

        for model_cfg in self.config["models"]:
            model_name = model_cfg["name"]
            hf_id = model_cfg["hf_id"]
            context_window = model_cfg.get("context_window", 8192)

            logger.info(f"\n{'─' * 50}")
            logger.info(f"Model: {model_name} ({hf_id})")
            logger.info(f"Context window: {context_window}")

            model, tokenizer = model_manager.load(hf_id)

            for tier in tiers:
                for lang in languages:
                    lang_dir = data_path / tier / "parallel" / lang
                    if not lang_dir.exists():
                        continue

                    evaluator = DDMEvaluator(language=lang, strict_citations=False)
                    jsonl_files = sorted(lang_dir.glob("*.jsonl"))

                    if max_conversations_per_tier:
                        jsonl_files = jsonl_files[:max_conversations_per_tier]

                    for jsonl_file in tqdm(
                        jsonl_files,
                        desc=f"  {model_name}|{lang}|{tier}",
                        leave=False,
                    ):
                        conv_id = jsonl_file.stem
                        messages = load_conversation(jsonl_file)
                        user_msgs = [
                            m["content"] for m in messages if m["role"] == "user"
                        ]

                        if not user_msgs:
                            continue

                        # Run inference
                        try:
                            inf_result = run_conversation_inference(
                                model, tokenizer, user_msgs,
                                system_prompt=DDM_SYSTEM_PROMPT,
                                compute_perplexity=compute_perplexity,
                            )
                            inf_result.conversation_id = conv_id
                            inf_result.language = lang
                            inf_result.model_name = model_name
                        except Exception as e:
                            logger.error(
                                f"    Inference failed for {conv_id}: {e}"
                            )
                            continue

                        # Evaluate DDM
                        ddm_result = evaluator.evaluate_conversation(
                            inf_result.responses, conv_id, lang,
                            f"{model_name}-{tier}",
                        )

                        all_drift_results.append(ddm_result)
                        all_inference_results.append(inf_result)

                        # Track context lengths for Exp 7
                        key = f"{model_name}-{tier}|{lang}|{conv_id}"
                        context_lengths_data[key] = inf_result.context_lengths_tokens

            model_manager.unload()

        # Save results
        drift_dir = self.output_dir / "drift_curves"
        save_evaluator = DDMEvaluator(language="en", strict_citations=False)
        save_evaluator.save_results(all_drift_results, drift_dir)

        # Save context lengths for Exp 7
        with open(drift_dir / "context_lengths.json", "w") as f:
            json.dump(context_lengths_data, f)

        logger.info(f"\n  ✅ Experiment 2 complete: {len(all_drift_results)} conversations evaluated")
        logger.info(f"  Results saved to {drift_dir}/")

        return (
            save_evaluator.results_to_summary_dataframe(all_drift_results),
            all_inference_results,
            all_drift_results,
        )

    # ═══════════════════════════════════════════════════════
    # Experiment 3: Paraphrastic Control (GPU)
    # ═══════════════════════════════════════════════════════

    def run_experiment_3(
        self,
        data_dir: str = "data",
        fertility_csv: str = None,
        max_conversations: int = 20,
    ) -> pd.DataFrame:
        """
        Experiment 3 — The Token Squeeze Proof.
        
        1. Take English texts, expand them to match target-language token counts
        2. Run expanded-English through inference
        3. If expanded-EN drifts like IT/ES/FR/DE → Token Squeeze confirmed
        """
        from src.evaluation.ddm import DDMEvaluator
        from src.expansion.strategies import ContextualRepetitionExpander
        from src.experiments.inference import (
            ModelManager, run_conversation_inference, load_conversation,
        )
        from src.tokenizer.fertility import FertilityProfiler

        logger.info("=" * 60)
        logger.info("EXPERIMENT 3: Paraphrastic Control (Token Squeeze Proof)")
        logger.info("=" * 60)

        data_path = Path(data_dir)
        out_dir = self.output_dir / "paraphrastic"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load fertility results to determine target token counts
        fert_csv = fertility_csv or str(
            self.output_dir / "fertility" / "fertility_ratios.csv"
        )
        if not Path(fert_csv).exists():
            logger.error("Run Experiment 1 first to get fertility ratios.")
            return pd.DataFrame()

        fert_df = pd.read_csv(fert_csv)

        model_manager = ModelManager(device="cuda:0")
        all_results = []

        # Use 2-3 representative models for this expensive experiment
        representative_models = self.config["models"][:3]

        for model_cfg in representative_models:
            model_name = model_cfg["name"]
            hf_id = model_cfg["hf_id"]

            model, tokenizer = model_manager.load(hf_id)
            expander = ContextualRepetitionExpander(tokenizer)

            # Get TFR for each target language with this model
            model_fert = fert_df[fert_df["model"] == model_name]

            for _, fert_row in model_fert.iterrows():
                target_lang = fert_row["language"]
                tfr = fert_row["tfr"]

                logger.info(f"  Expanding EN to match {target_lang} (TFR={tfr:.3f})")

                # Load English conversations
                en_dir = data_path / "medium" / "parallel" / "en"
                if not en_dir.exists():
                    continue

                for jsonl_file in sorted(en_dir.glob("*.jsonl"))[:max_conversations]:
                    conv_id = jsonl_file.stem
                    messages = load_conversation(jsonl_file)
                    user_msgs = [
                        m["content"] for m in messages if m["role"] == "user"
                    ]

                    if not user_msgs:
                        continue

                    # Expand each user message to simulate higher TFR
                    expanded_msgs = []
                    for msg in user_msgs:
                        orig_tokens = len(tokenizer.encode(msg, add_special_tokens=False))
                        target_tokens = int(orig_tokens * tfr)
                        expansion = expander.expand(msg, target_tokens)
                        expanded_msgs.append(expansion.expanded_text)

                    # Run inference on expanded English
                    try:
                        inf_result = run_conversation_inference(
                            model, tokenizer, expanded_msgs,
                            system_prompt=DDM_SYSTEM_PROMPT,
                        )
                    except Exception as e:
                        logger.error(f"    Inference error: {e}")
                        continue

                    # Evaluate DDM
                    evaluator = DDMEvaluator(language="en", strict_citations=False)
                    ddm_result = evaluator.evaluate_conversation(
                        inf_result.responses, conv_id, "en",
                        f"{model_name}-expanded-{target_lang}",
                    )
                    all_results.append(ddm_result)

            model_manager.unload()

        # Save
        if all_results:
            save_eval = DDMEvaluator(language="en", strict_citations=False)
            save_eval.save_results(all_results, out_dir)
            logger.info(f"  ✅ Paraphrastic control: {len(all_results)} tests saved")

        return pd.DataFrame()

    # ═══════════════════════════════════════════════════════
    # Experiment 4: Regression Analysis (TFR → DOP)
    # ═══════════════════════════════════════════════════════

    def run_experiment_4(
        self,
        fertility_csv: str = None,
        drift_csv: str = None,
    ) -> dict:
        """
        Experiment 4 — Regression: DOP = β₀ + β₁ × TFR + ε
        """
        import statsmodels.api as sm
        from scipy import stats

        logger.info("=" * 60)
        logger.info("EXPERIMENT 4: Regression Analysis (Fertility → Drift)")
        logger.info("=" * 60)

        fertility_csv = fertility_csv or str(
            self.output_dir / "fertility" / "fertility_ratios.csv"
        )
        drift_csv = drift_csv or str(
            self.output_dir / "drift_curves" / "drift_summary.csv"
        )

        if not Path(fertility_csv).exists() or not Path(drift_csv).exists():
            logger.error("Run Experiments 1 & 2 first.")
            return {}

        fertility_df = pd.read_csv(fertility_csv)
        drift_df = pd.read_csv(drift_csv)

        # Normalize model names for merging (strip tier suffix)
        drift_df["model_base"] = drift_df["model"].str.replace(
            r"-(?:short|medium|long)$", "", regex=True
        )

        merged = pd.merge(
            fertility_df, drift_df,
            left_on=["model", "language"],
            right_on=["model_base", "language"],
            how="inner",
            suffixes=("_fert", "_drift"),
        )

        if merged.empty:
            logger.warning("No merged data — check model name alignment.")
            return {}

        # OLS: DOP ~ TFR
        valid = merged.dropna(subset=["tfr", "drift_onset"])
        if len(valid) < 3:
            logger.warning("Not enough data for regression.")
            return {}

        X = sm.add_constant(valid["tfr"])
        y = valid["drift_onset"]

        model = sm.OLS(y, X).fit()

        results = {
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "beta_0": float(model.params.iloc[0]),
            "beta_1": float(model.params.iloc[1]),
            "p_value_beta_1": float(model.pvalues.iloc[1]),
            "f_statistic": float(model.fvalue),
            "n_observations": len(valid),
            "summary": str(model.summary()),
        }

        # Also run: AUC ~ TFR
        valid_auc = merged.dropna(subset=["tfr", "auc"])
        if len(valid_auc) >= 3:
            X_auc = sm.add_constant(valid_auc["tfr"])
            y_auc = valid_auc["auc"]
            model_auc = sm.OLS(y_auc, X_auc).fit()
            results["auc_r_squared"] = model_auc.rsquared
            results["auc_beta_1"] = float(model_auc.params.iloc[1])
            results["auc_p_value"] = float(model_auc.pvalues.iloc[1])

        out_dir = self.output_dir / "regression"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "regression_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"  R² = {results['r_squared']:.4f}")
        logger.info(f"  β₁ (TFR → DOP) = {results['beta_1']:.4f}, p = {results['p_value_beta_1']:.6f}")

        return results

    # ═══════════════════════════════════════════════════════
    # Experiment 5: SPAR Attention Analysis (GPU)
    # ═══════════════════════════════════════════════════════

    def run_experiment_5(
        self,
        data_dir: str = "data",
        max_conversations: int = 10,
    ) -> None:
        """
        Experiment 5 — SPAR: System Prompt Attention Ratio.
        
        Measures how much attention the model pays to system prompt tokens
        as the conversation grows. Uses eager attention for full weight extraction.
        """
        from src.attention.spar import SPARAnalyzer
        from src.experiments.inference import load_conversation, resolve_local_snapshot

        logger.info("=" * 60)
        logger.info("EXPERIMENT 5: SPAR Attention Analysis")
        logger.info("=" * 60)

        data_path = Path(data_dir)
        out_dir = self.output_dir / "attention_maps"
        out_dir.mkdir(parents=True, exist_ok=True)

        languages = ["en"] + [l["code"] for l in self.config["languages"]["targets"]]
        all_profiles = []

        # Use 2-3 smaller models for SPAR (memory-intensive)
        spar_models = [m for m in self.config["models"] if "3b" in m["name"].lower() or "1b" in m["name"].lower()]
        if not spar_models:
            spar_models = self.config["models"][:2]

        for model_cfg in spar_models:
            model_name = model_cfg["name"]
            hf_id = model_cfg["hf_id"]

            # Resolve local path
            local_path = resolve_local_snapshot(hf_id)
            load_path = str(local_path) if local_path else hf_id

            analyzer = SPARAnalyzer(
                load_path,
                device="cuda:0",
                dtype=__import__("torch").float16,
            )

            for lang in languages:
                lang_dir = data_path / "short" / "parallel" / lang
                if not lang_dir.exists():
                    continue

                jsonl_files = sorted(lang_dir.glob("*.jsonl"))[:max_conversations]

                for jsonl_file in tqdm(
                    jsonl_files,
                    desc=f"  SPAR|{model_name}|{lang}",
                    leave=False,
                ):
                    conv_id = jsonl_file.stem
                    messages = load_conversation(jsonl_file)

                    profile = analyzer.analyze_conversation(
                        conversation_turns=messages,
                        system_prompt=DDM_SYSTEM_PROMPT,
                        conversation_id=conv_id,
                        language=lang,
                        step=2,  # Every other turn for efficiency
                    )

                    analyzer.save_profile(profile, out_dir)
                    all_profiles.append(profile)

            # Free GPU memory
            del analyzer
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()

        # Save aggregate SPAR summary
        spar_rows = []
        for profile in all_profiles:
            for r in profile.spar_results:
                spar_rows.append({
                    "conversation_id": profile.conversation_id,
                    "language": profile.language,
                    "model": profile.model_name.split("/")[-1],
                    "turn": r.turn_number,
                    "spar_score": r.spar_score,
                    "seq_len": r.total_sequence_length,
                    "sys_len": r.system_prompt_length,
                })

        if spar_rows:
            spar_df = pd.DataFrame(spar_rows)
            spar_df.to_csv(out_dir / "spar_aggregate.csv", index=False)

        logger.info(f"  ✅ SPAR analysis: {len(all_profiles)} profiles saved")

    # ═══════════════════════════════════════════════════════
    # Experiments 6–13 (delegate to individual modules)
    # ═══════════════════════════════════════════════════════

    def run_experiment_6(self, data_dir: str = "data", max_conversations: int = 10):
        """Experiment 6 — System Prompt Re-injection."""
        from src.experiments.exp6_reinjection import run_experiment_6 as _run
        from src.experiments.inference import ModelManager

        manager = ModelManager(device="cuda:0")
        return _run(
            model_manager=manager,
            model_configs=self.config["models"][:3],  # Representative subset
            data_dir=Path(data_dir),
            system_prompt=DDM_SYSTEM_PROMPT,
            output_dir=self.output_dir / "reinjection",
            max_conversations=max_conversations,
        )

    def run_experiment_7(self):
        """Experiment 7 — Context Budget Analysis."""
        from src.experiments.exp7_context_budget import run_experiment_7 as _run

        return _run(
            drift_results_csv=str(self.output_dir / "drift_curves" / "drift_results.csv"),
            model_configs=self.config["models"],
            output_dir=str(self.output_dir / "context_budget"),
            context_lengths_json=str(self.output_dir / "drift_curves" / "context_lengths.json"),
        )

    def run_experiment_8(self, inference_results, drift_results):
        """Experiment 8 — Perplexity at Drift Onset."""
        from src.experiments.exp8_perplexity import run_experiment_8 as _run

        return _run(
            inference_results=inference_results,
            drift_results=drift_results,
            output_dir=str(self.output_dir / "perplexity"),
        )

    def run_experiment_9(self):
        """Experiment 9 — Drift Velocity Analysis."""
        from src.experiments.exp9_drift_velocity import run_experiment_9 as _run

        return _run(
            drift_results_csv=str(self.output_dir / "drift_curves" / "drift_results.csv"),
            output_dir=str(self.output_dir / "drift_velocity"),
        )

    def run_experiment_10(self):
        """Experiment 10 — Cross-Model Consistency."""
        from src.experiments.exp10_cross_model import run_experiment_10 as _run

        return _run(
            drift_summary_csv=str(self.output_dir / "drift_curves" / "drift_summary.csv"),
            output_dir=str(self.output_dir / "cross_model"),
        )

    def run_experiment_11(self):
        """Experiment 11 — Tier Effect Analysis."""
        from src.experiments.exp11_tier_effect import run_experiment_11 as _run

        return _run(
            drift_summary_csv=str(self.output_dir / "drift_curves" / "drift_summary.csv"),
            output_dir=str(self.output_dir / "tier_effect"),
        )

    def run_experiment_12(self):
        """Experiment 12 — Per-Level Failure Ordering."""
        from src.experiments.exp12_level_ordering import run_experiment_12 as _run

        return _run(
            drift_summary_csv=str(self.output_dir / "drift_curves" / "drift_summary.csv"),
            per_level_decay_json=str(self.output_dir / "drift_curves" / "per_level_decay.json"),
            output_dir=str(self.output_dir / "level_ordering"),
        )

    def run_experiment_13(self, inference_results, drift_results):
        """Experiment 13 — Token Position Analysis."""
        from src.experiments.exp13_token_position import run_experiment_13 as _run

        return _run(
            inference_results=inference_results,
            drift_results=drift_results,
            system_prompt=DDM_SYSTEM_PROMPT,
            model_configs=self.config["models"],
            output_dir=str(self.output_dir / "token_position"),
        )

    # ═══════════════════════════════════════════════════════
    # Full Pipeline
    # ═══════════════════════════════════════════════════════

    def run_all(
        self,
        data_dir: str = "data",
        skip_gpu: bool = False,
        max_conversations: int = None,
    ):
        """
        Run all 13 experiments in dependency order.
        
        Args:
            data_dir: Root data directory (e.g., "data/production").
            skip_gpu: If True, skip GPU experiments (2, 3, 5, 6).
            max_conversations: Limit conversations per tier (for dry runs).
        """
        logger.info("Starting full Token Squeeze experimental pipeline (13 experiments)")
        start = time.time()

        # Phase 1: Fertility (CPU)
        logger.info("\n" + "═" * 70)
        logger.info("PHASE 1: Token Fertility Profiling")
        logger.info("═" * 70)
        self.run_experiment_1(data_dir)

        if not skip_gpu:
            # Phase 2: GPU Inference
            logger.info("\n" + "═" * 70)
            logger.info("PHASE 2: GPU Inference Experiments")
            logger.info("═" * 70)

            # Exp 2 (core) — also collects data for 7, 8, 13
            _, inf_results, drift_results = self.run_experiment_2(
                data_dir, max_conversations_per_tier=max_conversations,
            )

            # Exp 8 & 13 (use inference results)
            self.run_experiment_8(inf_results, drift_results)
            self.run_experiment_13(inf_results, drift_results)

            # Exp 6 (re-injection — separate inference runs)
            self.run_experiment_6(data_dir, max_conversations=max_conversations or 10)

            # Exp 3 (paraphrastic — most expensive)
            self.run_experiment_3(data_dir)

            # Exp 5 (SPAR — memory-intensive)
            self.run_experiment_5(data_dir, max_conversations=max_conversations or 10)
        else:
            logger.info("(Skipping GPU experiments — running analytical only)")

        # Phase 3: Analytical (no GPU)
        logger.info("\n" + "═" * 70)
        logger.info("PHASE 3: Analytical Experiments")
        logger.info("═" * 70)

        self.run_experiment_4()    # Regression
        self.run_experiment_7()    # Context Budget

        self.run_experiment_9()    # Drift Velocity
        self.run_experiment_10()   # Cross-Model Consistency
        self.run_experiment_11()   # Tier Effect
        self.run_experiment_12()   # Per-Level Failure Ordering

        elapsed = time.time() - start
        logger.info(f"\n{'═' * 70}")
        logger.info(f"ALL 13 EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes.")
        logger.info(f"Results directory: {self.output_dir}/")
        logger.info(f"{'═' * 70}")
