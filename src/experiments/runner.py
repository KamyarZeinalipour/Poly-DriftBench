"""
Experiment Runner
=================
Orchestrates the full experimental pipeline for the Token Squeeze Hypothesis.

Experiments:
    1. Token Fertility Profiling
    2. Baseline Drift Measurement (5 languages)
    3. Paraphrastic Control (Token Squeeze Proof)
    4. Regression Analysis (Fertility → Drift)
    5. Mechanistic Validation (SPAR)
    6. Mitigation Cost-Benefit
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


class ExperimentRunner:
    """
    Master experiment orchestrator.

    Usage:
        runner = ExperimentRunner("configs/default.yaml")
        runner.run_experiment_1()  # Fertility profiling
        runner.run_experiment_2()  # Baseline drift
        ...
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = load_config(config_path)
        self.output_dir = Path(self.config["project"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = self.config["project"]["seed"]

        # Will be lazily initialized
        self._fertility_profiler = None
        self._ddm_evaluator = None
        self._spar_analyzer = None

    # ─── Experiment 1: Token Fertility Profiling ──────────

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

        # Save
        out_dir = self.output_dir / "fertility"
        profiler.save_results(results, out_dir)
        df = profiler.results_to_dataframe(results)

        # Log summary
        logger.info("\n--- Fertility Summary ---")
        for _, row in df.iterrows():
            logger.info(
                f"  {row['model']:>15s} | {row['language']:>3s}: "
                f"TFR = {row['tfr']:.4f} ({row['overhead_pct']:+.1f}% overhead)"
            )

        return df

    # ─── Experiment 2: Baseline Drift Measurement ────────

    def run_experiment_2(
        self,
        data_dir: str = "data",
        num_runs: int = None,
    ) -> pd.DataFrame:
        """
        Experiment 2 — Baseline Drift Measurement.

        Run standard EN, IT, ES, FR, DE datasets and plot decay curves.
        """
        from src.evaluation.ddm import DDMEvaluator, build_system_prompt

        logger.info("=" * 60)
        logger.info("EXPERIMENT 2: Baseline Drift Measurement")
        logger.info("=" * 60)

        num_runs = num_runs or self.config["experiments"]["num_runs"]
        evaluator = DDMEvaluator(self.config)
        system_prompt = build_system_prompt()
        all_languages = ["en"] + [
            l["code"] for l in self.config["languages"]["targets"]
        ]

        all_results = []

        for run_idx in range(num_runs):
            seed = self.seed + run_idx
            logger.info(f"\n--- Run {run_idx + 1}/{num_runs} (seed={seed}) ---")

            for lang in all_languages:
                for model_cfg in self.config["models"]:
                    model_name = model_cfg["name"]
                    logger.info(f"  Running: {model_name} × {lang}")

                    # TODO: Load conversations and run inference
                    # responses = run_inference(model_cfg, conversations, system_prompt)
                    # result = evaluator.evaluate_conversation(
                    #     responses, conv_id, lang, model_name
                    # )
                    # all_results.append(result)

        # df = evaluator.results_to_dataframe(all_results)
        # evaluator.save_results(all_results, self.output_dir / "drift_curves")
        logger.info("Experiment 2: Stub — implement inference loop")
        return pd.DataFrame()

    # ─── Experiment 3: Paraphrastic Control ───────────────

    def run_experiment_3(self, data_dir: str = "data") -> pd.DataFrame:
        """
        Experiment 3 — The Token Squeeze Proof.

        Run BTE, CPI, CRI expanded English datasets and compare decay
        curves against target language curves.
        """
        logger.info("=" * 60)
        logger.info("EXPERIMENT 3: Paraphrastic Control (Token Squeeze Proof)")
        logger.info("=" * 60)

        # TODO: For each target language and strategy:
        # 1. Load expanded English dataset
        # 2. Run inference
        # 3. Evaluate DDM
        # 4. Compare decay curves with target language
        logger.info("Experiment 3: Stub — implement after Exp 1 & 2")
        return pd.DataFrame()

    # ─── Experiment 4: Regression Analysis ────────────────

    def run_experiment_4(
        self,
        fertility_csv: str = None,
        drift_csv: str = None,
    ) -> dict:
        """
        Experiment 4 — Regression Analysis.

        Fits: DOP = β₀ + β₁ × TFR + ε
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

        fertility_df = pd.read_csv(fertility_csv)
        drift_df = pd.read_csv(drift_csv)

        # Merge on model + language
        merged = pd.merge(
            fertility_df, drift_df,
            on=["model", "language"],
            how="inner",
        )

        if merged.empty:
            logger.warning("No merged data — run Experiments 1 & 2 first.")
            return {}

        # OLS Regression: DOP ~ TFR
        X = sm.add_constant(merged["tfr"])
        y = merged["drift_onset"]

        model = sm.OLS(y, X).fit()

        results = {
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "beta_0": float(model.params.iloc[0]),
            "beta_1": float(model.params.iloc[1]),
            "p_value_beta_1": float(model.pvalues.iloc[1]),
            "f_statistic": float(model.fvalue),
            "n_observations": len(merged),
            "summary": str(model.summary()),
        }

        # Save
        out_dir = self.output_dir / "regression"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "regression_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n--- Regression Results ---")
        logger.info(f"  R² = {results['r_squared']:.4f}")
        logger.info(f"  β₁ (TFR coef) = {results['beta_1']:.4f}")
        logger.info(f"  p-value = {results['p_value_beta_1']:.6f}")

        return results

    # ─── Experiment 5: Mechanistic Validation (SPAR) ─────

    def run_experiment_5(self, data_dir: str = "data") -> None:
        """
        Experiment 5 — Mechanistic Validation.

        Extract SPAR scores and generate layer-wise heatmaps.
        """
        from src.attention.spar import SPARAnalyzer

        logger.info("=" * 60)
        logger.info("EXPERIMENT 5: Mechanistic Validation (SPAR)")
        logger.info("=" * 60)

        out_dir = self.output_dir / "attention_maps"

        for model_cfg in self.config["models"]:
            model_name = model_cfg["hf_id"]
            analyzer = SPARAnalyzer(model_name)

            # TODO: For each language × conversation:
            # profile = analyzer.analyze_conversation(
            #     turns, system_prompt, conv_id, lang
            # )
            # analyzer.save_profile(profile, out_dir)

        logger.info("Experiment 5: Stub — implement after Exp 2")

    # ─── Full Pipeline ───────────────────────────────────

    def run_all(self, data_dir: str = "data"):
        """Run all experiments in sequence."""
        logger.info("Starting full Token Squeeze experimental pipeline")
        start = time.time()

        self.run_experiment_1(data_dir)
        self.run_experiment_2(data_dir)
        self.run_experiment_3(data_dir)
        self.run_experiment_4()
        self.run_experiment_5(data_dir)

        elapsed = time.time() - start
        logger.info(f"\nAll experiments complete in {elapsed/60:.1f} minutes.")
