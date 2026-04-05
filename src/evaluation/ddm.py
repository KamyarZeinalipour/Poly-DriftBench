"""
Deterministic Drift Metric (DDM) — Multi-Level Constraint Evaluator
====================================================================
Measures instruction adherence decay across conversational turns using
four hierarchical constraints, producing a Decay Gradient instead of
a single binary cliff.

Levels:
    L1 (Format)    — Exact string tag appended to every response
    L2 (Structure) — Numbered bullet point format
    L3 (Lexical)   — Forbidden word avoidance
    L4 (Semantic)  — Citation before factual claims
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


# ──────────────────────────────────────────────────────────
# Individual Constraint Checkers
# ──────────────────────────────────────────────────────────

class L1FormatChecker:
    """L1: Check for exact appended tag string."""

    def __init__(self, tag: str = "[SYS_ACK: ACTIVE]"):
        self.tag = tag
        self.pattern = re.compile(re.escape(tag) + r"\s*$", re.MULTILINE)

    def check(self, response: str) -> bool:
        return bool(self.pattern.search(response))


class L2StructureChecker:
    """L2: Check that the response uses numbered bullet points."""

    def __init__(self, min_bullets: int = 2):
        self.min_bullets = min_bullets
        self.pattern = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)

    def check(self, response: str) -> bool:
        matches = self.pattern.findall(response)
        return len(matches) >= self.min_bullets


class L3LexicalChecker:
    """L3: Check that forbidden words are NOT present."""

    def __init__(self, forbidden_words: list[str] = None):
        self.forbidden_words = forbidden_words or ["however", "However", "HOWEVER"]

    def check(self, response: str) -> bool:
        response_lower = response.lower()
        for word in self.forbidden_words:
            if word.lower() in response_lower:
                return False
        return True


class L4CitationChecker:
    """L4: Check that factual claims are preceded by a citation."""

    def __init__(self):
        self.citation_patterns = [
            re.compile(r"\[Source:.*?\]"),
            re.compile(r"\(.*?\d{4}.*?\)"),
            re.compile(r"According to\s+", re.IGNORECASE),
            re.compile(r"As stated by\s+", re.IGNORECASE),
            re.compile(r"Based on\s+.*?research", re.IGNORECASE),
        ]

    def check(self, response: str) -> bool:
        for pattern in self.citation_patterns:
            if pattern.search(response):
                return True
        return False


# ──────────────────────────────────────────────────────────
# DDM Evaluator
# ──────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """Result for a single conversational turn."""
    turn_number: int
    l1_pass: bool
    l2_pass: bool
    l3_pass: bool
    l4_pass: bool
    ddm_score: float  # 0.0 to 1.0

    @property
    def all_pass(self) -> bool:
        return self.l1_pass and self.l2_pass and self.l3_pass and self.l4_pass


@dataclass
class ConversationDriftResult:
    """Drift analysis for an entire conversation."""
    conversation_id: str
    language: str
    model_name: str
    turn_results: list[TurnResult] = field(default_factory=list)
    drift_onset_point: Optional[int] = None   # First turn where DDM < 1.0
    total_collapse_point: Optional[int] = None  # First turn where DDM = 0.0
    mean_ddm: float = 0.0
    total_turns: int = 0

    def compute_summary(self):
        """Compute summary statistics from turn results."""
        if not self.turn_results:
            return

        self.total_turns = len(self.turn_results)
        scores = [t.ddm_score for t in self.turn_results]
        self.mean_ddm = float(np.mean(scores))

        # Find Drift Onset Point (DOP)
        for t in self.turn_results:
            if t.ddm_score < 1.0:
                self.drift_onset_point = t.turn_number
                break

        # Find Total Collapse Point (TCP)
        for t in self.turn_results:
            if t.ddm_score == 0.0:
                self.total_collapse_point = t.turn_number
                break


class DDMEvaluator:
    """
    Multi-Level Deterministic Drift Metric evaluator.

    Usage:
        evaluator = DDMEvaluator(config)
        result = evaluator.evaluate_conversation(responses, conv_id, lang, model)
    """

    def __init__(self, config: dict = None):
        config = config or {}
        ddm_cfg = config.get("ddm", {}).get("constraints", {})

        # Initialize checkers
        l1_cfg = ddm_cfg.get("L1_format", {})
        self.l1 = L1FormatChecker(tag=l1_cfg.get("tag", "[SYS_ACK: ACTIVE]"))

        l2_cfg = ddm_cfg.get("L2_structure", {})
        self.l2 = L2StructureChecker(min_bullets=l2_cfg.get("min_bullets", 2))

        l3_cfg = ddm_cfg.get("L3_lexical", {})
        self.l3 = L3LexicalChecker(
            forbidden_words=l3_cfg.get("forbidden_words", ["however"])
        )

        self.l4 = L4CitationChecker()

    def evaluate_turn(self, response: str, turn_number: int) -> TurnResult:
        """Evaluate a single turn against all four constraint levels."""
        l1 = self.l1.check(response)
        l2 = self.l2.check(response)
        l3 = self.l3.check(response)
        l4 = self.l4.check(response)

        ddm_score = sum([l1, l2, l3, l4]) / 4.0

        return TurnResult(
            turn_number=turn_number,
            l1_pass=l1,
            l2_pass=l2,
            l3_pass=l3,
            l4_pass=l4,
            ddm_score=ddm_score,
        )

    def evaluate_conversation(
        self,
        responses: list[str],
        conversation_id: str,
        language: str,
        model_name: str,
    ) -> ConversationDriftResult:
        """
        Evaluate all turns in a conversation.

        Args:
            responses: List of model responses (one per turn).
            conversation_id: Unique ID for this conversation.
            language: Language code (e.g., 'en', 'it').
            model_name: Model used for generation.

        Returns:
            ConversationDriftResult with per-turn and summary metrics.
        """
        result = ConversationDriftResult(
            conversation_id=conversation_id,
            language=language,
            model_name=model_name,
        )

        for i, response in enumerate(responses):
            turn_result = self.evaluate_turn(response, turn_number=i + 1)
            result.turn_results.append(turn_result)

        result.compute_summary()

        logger.info(
            f"  [{model_name}|{language}|{conversation_id}] "
            f"DOP={result.drift_onset_point}, TCP={result.total_collapse_point}, "
            f"mean_DDM={result.mean_ddm:.3f}"
        )

        return result

    def results_to_dataframe(
        self, results: list[ConversationDriftResult]
    ) -> pd.DataFrame:
        """Convert list of conversation results to a flat DataFrame."""
        rows = []
        for conv in results:
            for t in conv.turn_results:
                rows.append({
                    "conversation_id": conv.conversation_id,
                    "language": conv.language,
                    "model": conv.model_name,
                    "turn": t.turn_number,
                    "l1_format": int(t.l1_pass),
                    "l2_structure": int(t.l2_pass),
                    "l3_lexical": int(t.l3_pass),
                    "l4_citation": int(t.l4_pass),
                    "ddm_score": t.ddm_score,
                    "drift_onset": conv.drift_onset_point,
                    "total_collapse": conv.total_collapse_point,
                })
        return pd.DataFrame(rows)

    def save_results(
        self,
        results: list[ConversationDriftResult],
        output_dir: str | Path,
    ) -> Path:
        """Save drift results to CSV and JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.results_to_dataframe(results)
        csv_path = output_dir / "drift_results.csv"
        df.to_csv(csv_path, index=False)

        # Summary table
        summary = df.groupby(["model", "language"]).agg({
            "ddm_score": "mean",
            "drift_onset": "first",
            "total_collapse": "first",
        }).reset_index()
        summary_path = output_dir / "drift_summary.csv"
        summary.to_csv(summary_path, index=False)

        logger.info(f"Saved drift results: {csv_path}")
        logger.info(f"Saved drift summary: {summary_path}")
        return csv_path


def build_system_prompt(constraints_config: dict = None) -> str:
    """
    Build the system prompt that embeds all four DDM constraints.

    This is the standard Turn-1 system prompt used across all experiments.
    """
    return (
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
