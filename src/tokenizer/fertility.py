"""
Token Fertility Profiler
========================
Computes the Token Fertility Ratio (TFR) for each language relative to English
across multiple tokenizers/models.

TFR(L) = tokens(L) / tokens(EN)

A TFR > 1.0 means language L requires more tokens than English for the
same semantic content — the core measurement behind the Token Squeeze Hypothesis.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class FertilityResult:
    """Result of fertility analysis for one language-model pair."""
    language: str
    model_name: str
    total_tokens: int
    en_total_tokens: int
    tfr: float  # Token Fertility Ratio
    per_turn_tfrs: list[float] = field(default_factory=list)
    mean_tfr: float = 0.0
    std_tfr: float = 0.0
    median_tfr: float = 0.0
    max_tfr: float = 0.0
    min_tfr: float = 0.0

    def __post_init__(self):
        if self.per_turn_tfrs:
            arr = np.array(self.per_turn_tfrs)
            self.mean_tfr = float(np.mean(arr))
            self.std_tfr = float(np.std(arr))
            self.median_tfr = float(np.median(arr))
            self.max_tfr = float(np.max(arr))
            self.min_tfr = float(np.min(arr))


class FertilityProfiler:
    """
    Profiles token fertility across languages and models.

    Usage:
        profiler = FertilityProfiler(model_configs)
        results = profiler.profile_dataset(parallel_data)
        profiler.save_results(results, output_dir)
    """

    def __init__(self, model_configs: list[dict]):
        """
        Args:
            model_configs: List of dicts with 'name' and 'hf_id' keys.
                Example: [{"name": "llama3-8b", "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct"}]
        """
        self.model_configs = model_configs
        self.tokenizers: dict[str, AutoTokenizer] = {}
        self._load_tokenizers()

    def _load_tokenizers(self):
        """Load all tokenizers (lightweight, no GPU needed)."""
        for cfg in self.model_configs:
            name = cfg["name"]
            hf_id = cfg["hf_id"]
            logger.info(f"Loading tokenizer: {name} ({hf_id})")
            try:
                self.tokenizers[name] = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=True
                )
                logger.info(f"  Vocab size: {self.tokenizers[name].vocab_size}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer {name}: {e}")
                raise

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens for a given text using a specific model's tokenizer."""
        tokenizer = self.tokenizers[model_name]
        return len(tokenizer.encode(text, add_special_tokens=False))

    def compute_fertility(
        self,
        en_texts: list[str],
        target_texts: list[str],
        model_name: str,
        target_lang: str,
    ) -> FertilityResult:
        """
        Compute Token Fertility Ratio between English and a target language.

        Args:
            en_texts: List of English texts (one per turn/message).
            target_texts: Parallel list of target language texts.
            model_name: Which model's tokenizer to use.
            target_lang: Language code (e.g., 'it', 'es').

        Returns:
            FertilityResult with aggregate and per-turn statistics.
        """
        assert len(en_texts) == len(target_texts), (
            f"Parallel texts must have same length: {len(en_texts)} vs {len(target_texts)}"
        )

        per_turn_tfrs = []
        total_en = 0
        total_target = 0

        for en_text, target_text in zip(en_texts, target_texts):
            en_tokens = self.count_tokens(en_text, model_name)
            target_tokens = self.count_tokens(target_text, model_name)
            total_en += en_tokens
            total_target += target_tokens

            if en_tokens > 0:
                per_turn_tfrs.append(target_tokens / en_tokens)

        overall_tfr = total_target / total_en if total_en > 0 else 0.0

        return FertilityResult(
            language=target_lang,
            model_name=model_name,
            total_tokens=total_target,
            en_total_tokens=total_en,
            tfr=overall_tfr,
            per_turn_tfrs=per_turn_tfrs,
        )

    def profile_dataset(
        self,
        parallel_data: dict[str, list[str]],
    ) -> list[FertilityResult]:
        """
        Profile an entire parallel dataset across all models and languages.

        Args:
            parallel_data: Dict mapping language codes to lists of texts.
                Must include 'en' as the baseline.
                Example: {"en": [...], "it": [...], "es": [...], ...}

        Returns:
            List of FertilityResult objects (one per language-model pair).
        """
        assert "en" in parallel_data, "English ('en') must be present as baseline."
        en_texts = parallel_data["en"]
        target_langs = [lang for lang in parallel_data if lang != "en"]

        results = []
        for model_name in tqdm(self.tokenizers, desc="Models"):
            for lang in tqdm(target_langs, desc="Languages", leave=False):
                result = self.compute_fertility(
                    en_texts=en_texts,
                    target_texts=parallel_data[lang],
                    model_name=model_name,
                    target_lang=lang,
                )
                results.append(result)
                logger.info(
                    f"  {model_name} | {lang}: TFR = {result.tfr:.4f} "
                    f"(mean={result.mean_tfr:.4f} ± {result.std_tfr:.4f})"
                )

        return results

    def results_to_dataframe(self, results: list[FertilityResult]) -> pd.DataFrame:
        """Convert results to a DataFrame for analysis."""
        rows = []
        for r in results:
            rows.append({
                "model": r.model_name,
                "language": r.language,
                "tfr": r.tfr,
                "mean_tfr": r.mean_tfr,
                "std_tfr": r.std_tfr,
                "median_tfr": r.median_tfr,
                "total_tokens": r.total_tokens,
                "en_total_tokens": r.en_total_tokens,
                "token_overhead": r.total_tokens - r.en_total_tokens,
                "overhead_pct": (r.tfr - 1.0) * 100,
            })
        return pd.DataFrame(rows)

    def save_results(
        self,
        results: list[FertilityResult],
        output_dir: str | Path,
    ) -> Path:
        """Save fertility results to CSV and JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as DataFrame
        df = self.results_to_dataframe(results)
        csv_path = output_dir / "fertility_ratios.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved fertility CSV: {csv_path}")

        # Save detailed JSON
        json_path = output_dir / "fertility_detailed.json"
        detailed = [asdict(r) for r in results]
        with open(json_path, "w") as f:
            json.dump(detailed, f, indent=2)
        logger.info(f"Saved fertility JSON: {json_path}")

        return csv_path


def load_parallel_texts(data_dir: str | Path) -> dict[str, list[str]]:
    """
    Load parallel texts from the data directory structure.

    Expects:
        data_dir/
            parallel/
                en/ *.jsonl
                it/ *.jsonl
                es/ *.jsonl
                ...

    Each JSONL file contains one JSON object per line with a "text" field.
    """
    data_dir = Path(data_dir)
    parallel_dir = data_dir / "parallel"
    parallel_data = {}

    for lang_dir in sorted(parallel_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        texts = []
        for jsonl_file in sorted(lang_dir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    texts.append(obj["text"])
        if texts:
            parallel_data[lang] = texts
            logger.info(f"Loaded {len(texts)} texts for language '{lang}'")

    return parallel_data
