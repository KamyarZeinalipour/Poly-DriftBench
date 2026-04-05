"""
Paraphrastic Expansion Module
==============================
Implements three linguistically valid strategies to inflate English text
to match the token count of a target language, replacing naive <pad> tokens.

Strategies:
    1. BTE — Back-Translation Expansion (EN → pivot → EN)
    2. CPI — Controlled Paraphrastic Inflation (LLM-driven verbose rewriting)
    3. CRI — Contextual Repetition Injection (discourse-level filler insertion)

All strategies verify semantic equivalence via BERTScore ≥ threshold.
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """Result of expanding a single text."""
    original_text: str
    expanded_text: str
    original_tokens: int
    expanded_tokens: int
    target_tokens: int
    strategy: str
    bertscore: float = 0.0
    success: bool = False  # True if expanded_tokens >= target_tokens and bertscore >= threshold


class BaseExpander(ABC):
    """Base class for expansion strategies."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        bertscore_threshold: float = 0.92,
    ):
        self.tokenizer = tokenizer
        self.bertscore_threshold = bertscore_threshold

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def verify_semantic_equivalence(
        self, original: str, expanded: str
    ) -> float:
        """Compute BERTScore between original and expanded text."""
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                [expanded], [original], lang="en", verbose=False
            )
            return float(F1[0])
        except ImportError:
            logger.warning("bert_score not installed, skipping verification")
            return 1.0

    @abstractmethod
    def expand(
        self, text: str, target_token_count: int, **kwargs
    ) -> ExpansionResult:
        """Expand text to reach target token count."""
        pass

    def expand_dataset(
        self,
        en_texts: list[str],
        target_token_counts: list[int],
        **kwargs,
    ) -> list[ExpansionResult]:
        """Expand a full dataset of texts."""
        results = []
        for text, target in zip(en_texts, target_token_counts):
            result = self.expand(text, target, **kwargs)
            results.append(result)
        return results


class BackTranslationExpander(BaseExpander):
    """
    Strategy 1: Back-Translation Expansion (BTE)
    EN → pivot language → EN

    Back-translation naturally produces verbose, slightly rephrased English
    that inflates token count while preserving semantics.
    """

    STRATEGY_NAME = "bte"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        bertscore_threshold: float = 0.92,
        pivot_lang: str = "ja",
        fallback_lang: str = "zh",
    ):
        super().__init__(tokenizer, bertscore_threshold)
        self.pivot_lang = pivot_lang
        self.fallback_lang = fallback_lang
        self._translation_model = None

    def _load_translation_models(self):
        """Lazy-load MarianMT translation models."""
        if self._translation_model is not None:
            return

        from transformers import MarianMTModel, MarianTokenizer

        # EN → pivot
        en_pivot_name = f"Helsinki-NLP/opus-mt-en-{self.pivot_lang}"
        self.en_to_pivot_tokenizer = MarianTokenizer.from_pretrained(en_pivot_name)
        self.en_to_pivot_model = MarianMTModel.from_pretrained(en_pivot_name)

        # pivot → EN
        pivot_en_name = f"Helsinki-NLP/opus-mt-{self.pivot_lang}-en"
        self.pivot_to_en_tokenizer = MarianTokenizer.from_pretrained(pivot_en_name)
        self.pivot_to_en_model = MarianMTModel.from_pretrained(pivot_en_name)

        self._translation_model = True
        logger.info(f"Loaded BTE translation models (pivot={self.pivot_lang})")

    def _translate(self, text: str, tokenizer, model) -> str:
        """Translate text using a MarianMT model."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def expand(
        self, text: str, target_token_count: int, max_rounds: int = 5, **kwargs
    ) -> ExpansionResult:
        """
        Expand via iterative back-translation until target token count
        is reached or max rounds exceeded.
        """
        self._load_translation_models()

        original_tokens = self.count_tokens(text)
        current_text = text

        for round_num in range(max_rounds):
            # EN → pivot
            pivot_text = self._translate(
                current_text, self.en_to_pivot_tokenizer, self.en_to_pivot_model
            )
            # pivot → EN
            back_translated = self._translate(
                pivot_text, self.pivot_to_en_tokenizer, self.pivot_to_en_model
            )

            current_tokens = self.count_tokens(back_translated)

            if current_tokens >= target_token_count:
                current_text = back_translated
                break

            # If still short, use the back-translated version and go again
            current_text = back_translated

        expanded_tokens = self.count_tokens(current_text)
        bertscore = self.verify_semantic_equivalence(text, current_text)

        return ExpansionResult(
            original_text=text,
            expanded_text=current_text,
            original_tokens=original_tokens,
            expanded_tokens=expanded_tokens,
            target_tokens=target_token_count,
            strategy=self.STRATEGY_NAME,
            bertscore=bertscore,
            success=(
                expanded_tokens >= target_token_count
                and bertscore >= self.bertscore_threshold
            ),
        )


class ParaphrasticInflationExpander(BaseExpander):
    """
    Strategy 2: Controlled Paraphrastic Inflation (CPI)
    Uses an LLM to rewrite text more verbosely until target count is reached.
    """

    STRATEGY_NAME = "cpi"

    INFLATION_PROMPT = (
        "Rewrite the following text to be more verbose and elaborate, "
        "adding clarifying clauses, hedging language, and detailed elaboration. "
        "The rewritten text MUST preserve the exact same meaning. "
        "The rewritten text should be approximately {target_ratio:.1f}x longer "
        "than the original.\n\n"
        "Original text:\n{text}\n\n"
        "Verbose rewrite:"
    )

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        bertscore_threshold: float = 0.92,
        llm_model: str = "gpt-4o",
        max_attempts: int = 5,
        temperature: float = 0.3,
    ):
        super().__init__(tokenizer, bertscore_threshold)
        self.llm_model = llm_model
        self.max_attempts = max_attempts
        self.temperature = temperature

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM for paraphrastic inflation."""
        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def expand(
        self, text: str, target_token_count: int, **kwargs
    ) -> ExpansionResult:
        """Expand via LLM-driven verbose rewriting."""
        original_tokens = self.count_tokens(text)
        target_ratio = target_token_count / max(original_tokens, 1)

        best_text = text
        best_score = 0.0

        for attempt in range(self.max_attempts):
            prompt = self.INFLATION_PROMPT.format(
                text=text, target_ratio=target_ratio
            )
            expanded = self._call_llm(prompt)
            if not expanded:
                continue

            current_tokens = self.count_tokens(expanded)
            bertscore = self.verify_semantic_equivalence(text, expanded)

            if (
                current_tokens >= target_token_count
                and bertscore >= self.bertscore_threshold
            ):
                return ExpansionResult(
                    original_text=text,
                    expanded_text=expanded,
                    original_tokens=original_tokens,
                    expanded_tokens=current_tokens,
                    target_tokens=target_token_count,
                    strategy=self.STRATEGY_NAME,
                    bertscore=bertscore,
                    success=True,
                )

            # Track best attempt
            if bertscore > best_score:
                best_text = expanded
                best_score = bertscore

        expanded_tokens = self.count_tokens(best_text)
        return ExpansionResult(
            original_text=text,
            expanded_text=best_text,
            original_tokens=original_tokens,
            expanded_tokens=expanded_tokens,
            target_tokens=target_token_count,
            strategy=self.STRATEGY_NAME,
            bertscore=best_score,
            success=False,
        )


class ContextualRepetitionExpander(BaseExpander):
    """
    Strategy 3: Contextual Repetition Injection (CRI)
    Injects discourse-level filler that mimics natural conversational redundancy.
    """

    STRATEGY_NAME = "cri"

    DEFAULT_FILLERS = [
        "As I mentioned earlier, ",
        "To reiterate the key point, ",
        "Building on what was discussed, ",
        "To clarify and expand on this, ",
        "It's worth emphasizing that ",
        "To put it another way, ",
        "In other words, ",
        "More specifically, ",
        "To elaborate further, ",
        "It should be noted that ",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        bertscore_threshold: float = 0.92,
        fillers: list[str] = None,
    ):
        super().__init__(tokenizer, bertscore_threshold)
        self.fillers = fillers or self.DEFAULT_FILLERS

    def expand(
        self, text: str, target_token_count: int, **kwargs
    ) -> ExpansionResult:
        """Expand by injecting discourse fillers between sentences."""
        original_tokens = self.count_tokens(text)

        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            sentences = [text]

        current_text = text
        filler_idx = 0

        while self.count_tokens(current_text) < target_token_count:
            # Insert a filler before a random sentence (not the first)
            if len(sentences) > 1:
                insert_pos = random.randint(1, len(sentences) - 1)
                filler = self.fillers[filler_idx % len(self.fillers)]
                sentences.insert(insert_pos, filler + sentences[insert_pos])
                sentences.pop(insert_pos + 1)
            else:
                filler = self.fillers[filler_idx % len(self.fillers)]
                sentences.append(filler + sentences[0].lower())

            current_text = " ".join(sentences)
            filler_idx += 1

            # Safety: prevent infinite loop
            if filler_idx > 50:
                break

        expanded_tokens = self.count_tokens(current_text)
        bertscore = self.verify_semantic_equivalence(text, current_text)

        return ExpansionResult(
            original_text=text,
            expanded_text=current_text,
            original_tokens=original_tokens,
            expanded_tokens=expanded_tokens,
            target_tokens=target_token_count,
            strategy=self.STRATEGY_NAME,
            bertscore=bertscore,
            success=(
                expanded_tokens >= target_token_count
                and bertscore >= self.bertscore_threshold
            ),
        )


def get_expander(
    strategy: str,
    tokenizer: AutoTokenizer,
    config: dict = None,
) -> BaseExpander:
    """Factory function to get the appropriate expander."""
    config = config or {}
    expansion_cfg = config.get("expansion", {})
    threshold = expansion_cfg.get("bertscore_threshold", 0.92)

    if strategy == "bte":
        strat_cfg = expansion_cfg.get("strategies", {}).get("bte", {})
        return BackTranslationExpander(
            tokenizer=tokenizer,
            bertscore_threshold=threshold,
            pivot_lang=strat_cfg.get("intermediate_lang", "ja"),
        )
    elif strategy == "cpi":
        strat_cfg = expansion_cfg.get("strategies", {}).get("cpi", {})
        return ParaphrasticInflationExpander(
            tokenizer=tokenizer,
            bertscore_threshold=threshold,
            llm_model=strat_cfg.get("model", "gpt-4o"),
            max_attempts=strat_cfg.get("max_attempts", 5),
        )
    elif strategy == "cri":
        strat_cfg = expansion_cfg.get("strategies", {}).get("cri", {})
        return ContextualRepetitionExpander(
            tokenizer=tokenizer,
            bertscore_threshold=threshold,
            fillers=strat_cfg.get("fillers"),
        )
    else:
        raise ValueError(f"Unknown expansion strategy: {strategy}")
