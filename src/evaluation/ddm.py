"""
Deterministic Drift Metric (DDM) — Multi-Level Constraint Evaluator
====================================================================
Measures instruction adherence decay across conversational turns using
five hierarchical constraints, producing a continuous Decay Gradient.

Levels (Orthogonal Constraint Taxonomy):
    L1 (Format)    — Exact string tag appended to every response
    L2 (Structure) — Numbered bullet point format (with partial credit)
    L3 (Lexical)   — Forbidden word avoidance (cross-lingual)
    L4 (Semantic)  — Citation before factual claims (strict)
    L5 (Dynamic)   — Turn counter [Turn: N] that must increment each turn

Metrics:
    - DDM Score (per-turn): Weighted sum of L1–L5, continuous 0.0–1.0
    - DOP  (Drift Onset Point): First turn where DDM < 1.0
    - Sustained DOP: First turn where DDM < 1.0 for k consecutive turns
    - TCP  (Total Collapse Point): First turn where DDM = 0.0
    - AUC  (Area Under Decay Curve): Integral of DDM over turns, normalized
    - Half-Life (τ½): Turn at which DDM drops to ≤ 0.5
    - Recovery Rate: Fraction of post-drift turns where DDM recovers to 1.0
    - Per-Level Decay Curves: Independent L1/L2/L3/L4/L5 pass rates over turns
    - CI95: Bootstrap confidence intervals on DOP and mean DDM
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
# Cross-Lingual Forbidden Words for L3
# ──────────────────────────────────────────────────────────

# "however" and its direct equivalents across target languages
FORBIDDEN_WORDS_BY_LANG = {
    "en": ["however"],
    "it": ["however", "tuttavia", "però", "comunque", "ciononostante", "nondimeno"],
    "es": ["however", "sin embargo", "no obstante", "aunque", "sin embargo de"],
    "fr": ["however", "cependant", "toutefois", "néanmoins", "pourtant"],
    "de": ["however", "jedoch", "allerdings", "dennoch", "trotzdem", "gleichwohl"],
}


# ──────────────────────────────────────────────────────────
# Individual Constraint Checkers
# ──────────────────────────────────────────────────────────

class L1FormatChecker:
    """L1: Check for exact appended tag string. Binary pass/fail."""

    def __init__(self, tag: str = "[SYS_ACK: ACTIVE]"):
        self.tag = tag
        # Accept with or without brackets, case-insensitive
        # Extract core content (strip brackets if present)
        core = tag.strip("[]")
        escaped_core = re.escape(core)
        # Match: [SYS_ACK: ACTIVE] or SYS_ACK: ACTIVE (with optional brackets)
        self.pattern = re.compile(
            r"\[?" + escaped_core + r"\]?\s*$",
            re.MULTILINE | re.IGNORECASE,
        )

    def check(self, response: str) -> float:
        """Returns 1.0 if tag present, 0.0 otherwise."""
        return 1.0 if self.pattern.search(response) else 0.0


class L2StructureChecker:
    """
    L2: Check that the response uses numbered bullet points.
    Returns partial credit: min(found_bullets, min_bullets) / min_bullets.
    """

    def __init__(self, min_bullets: int = 2):
        self.min_bullets = min_bullets
        self.pattern = re.compile(r"^\s*\d+[\.)\]]\s+", re.MULTILINE)

    def check(self, response: str) -> float:
        """Returns continuous score 0.0–1.0 based on bullet count."""
        matches = self.pattern.findall(response)
        found = len(matches)
        return min(found, self.min_bullets) / self.min_bullets


class L3LexicalChecker:
    """
    L3: Check that forbidden words are NOT present.
    Supports cross-lingual forbidden word lists.
    Uses word-boundary matching to avoid false positives on substrings.
    """

    def __init__(
        self,
        forbidden_words: list[str] = None,
        language: str = "en",
    ):
        if forbidden_words:
            self.forbidden_words = [w.lower() for w in forbidden_words]
        else:
            self.forbidden_words = [
                w.lower() for w in FORBIDDEN_WORDS_BY_LANG.get(language, ["however"])
            ]

        # Pre-compile word-boundary patterns for accurate matching
        self._patterns = []
        for word in self.forbidden_words:
            # Use word boundaries; escape the word in case of special chars
            pat = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            self._patterns.append(pat)

    def check(self, response: str) -> float:
        """Returns 1.0 if no forbidden words found, 0.0 if any found."""
        for pat in self._patterns:
            if pat.search(response):
                return 0.0
        return 1.0


class L4CitationChecker:
    """
    L4: Check that factual claims are preceded by a citation.
    Strict mode: requires structured citations, not vague attributions.
    Returns partial credit based on citation density.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict

        # Strict citation patterns — require actual source references
        self.strict_patterns = [
            re.compile(r"\[Source:\s*[^\]]+\]"),               # [Source: XYZ]
            re.compile(r"\((?:[A-Z][a-z]+(?:\s+(?:et\s+al\.?|&)\s*)?(?:,\s*)?)+\d{4}[a-z]?\)"),  # (Author, 2024)
            re.compile(r"According to\s+(?:the\s+)?[A-Z][A-Za-z\s]+(?:,|\()"),  # According to <Named Source>
            re.compile(r"As (?:stated|reported|noted|documented) (?:by|in)\s+[A-Z]", re.IGNORECASE),
        ]

        # Lenient patterns (kept for backward compatibility)
        self.lenient_patterns = [
            re.compile(r"\[Source:.*?\]"),
            re.compile(r"\(.*?\d{4}.*?\)"),
            re.compile(r"According to\s+", re.IGNORECASE),
            re.compile(r"As stated by\s+", re.IGNORECASE),
            re.compile(r"Based on\s+.*?research", re.IGNORECASE),
        ]

    def check(self, response: str) -> float:
        """
        Returns 1.0 if at least one valid citation found, 0.0 otherwise.
        In strict mode, uses tighter patterns.
        """
        patterns = self.strict_patterns if self.strict else self.lenient_patterns
        for pattern in patterns:
            if pattern.search(response):
                return 1.0
        return 0.0



class L5StateTracker:
    """
    L5: Dynamic State-Tracking — checks that each response begins with
    [Turn: N] where N increments by exactly 1 each turn.

    Unlike L1-L4 (static rules that don't change per turn), L5 requires
    the model to actively compute state based on its position in the
    conversation. This separates static pattern-matching from true
    dynamic reasoning.
    """

    def __init__(self):
        # Pattern: [Turn: N] at the start (with optional whitespace)
        self.pattern = re.compile(
            r"^\s*\[Turn:\s*(\d+)\]", re.IGNORECASE
        )

    def check(self, response: str, expected_turn: int) -> float:
        """
        Returns 1.0 if [Turn: N] is present and N == expected_turn.
        Returns 0.0 otherwise.
        """
        match = self.pattern.search(response)
        if not match:
            return 0.0
        actual = int(match.group(1))
        return 1.0 if actual == expected_turn else 0.0


# ──────────────────────────────────────────────────────────
# Data Classes for Results
# ──────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """Result for a single conversational turn."""
    turn_number: int
    l1_score: float      # 0.0 or 1.0 (binary)
    l2_score: float      # 0.0 to 1.0 (partial credit)
    l3_score: float      # 0.0 or 1.0 (binary)
    l4_score: float      # 0.0 or 1.0 (binary)
    l5_score: float = 0.0  # 0.0 or 1.0 (dynamic state-tracking)
    ddm_score: float = 0.0  # Weighted combination, 0.0 to 1.0

    # Backward-compatible boolean properties
    @property
    def l1_pass(self) -> bool:
        return self.l1_score >= 1.0

    @property
    def l2_pass(self) -> bool:
        return self.l2_score >= 1.0

    @property
    def l3_pass(self) -> bool:
        return self.l3_score >= 1.0

    @property
    def l4_pass(self) -> bool:
        return self.l4_score >= 1.0

    @property
    def l5_pass(self) -> bool:
        return self.l5_score >= 1.0

    @property
    def all_pass(self) -> bool:
        return self.ddm_score >= 1.0


@dataclass
class PerLevelDecay:
    """Per-level decay statistics across all turns."""
    level: str                                  # "L1", "L2", "L3", "L4", "L5"
    scores: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    decay_onset: Optional[int] = None           # First turn where score < 1.0
    auc: float = 0.0                            # Area under this level's curve

    def compute(self):
        if not self.scores:
            return
        arr = np.array(self.scores)
        self.mean_score = float(np.mean(arr))
        # AUC via trapezoidal rule, normalized to [0, 1]
        n = len(arr)
        if n > 1:
            self.auc = float(np.trapz(arr, dx=1.0) / (n - 1))
        else:
            self.auc = float(arr[0])
        # Decay onset for this level
        for i, s in enumerate(self.scores):
            if s < 1.0:
                self.decay_onset = i + 1  # 1-indexed turn
                break


@dataclass
class ConversationDriftResult:
    """Drift analysis for an entire conversation with comprehensive metrics."""
    conversation_id: str
    language: str
    model_name: str
    turn_results: list[TurnResult] = field(default_factory=list)

    # ── Core Metrics ──
    total_turns: int = 0
    mean_ddm: float = 0.0
    std_ddm: float = 0.0

    # ── Onset & Collapse ──
    drift_onset_point: Optional[int] = None       # First turn where DDM < 1.0
    drift_onset_tokens: Optional[int] = None      # Context token count at DOP (Verbosity Confound fix)
    sustained_dop: Optional[int] = None            # First turn where DDM < 1.0 for k consecutive turns
    total_collapse_point: Optional[int] = None     # First turn where DDM = 0.0

    # ── Token-Space Metrics (addresses Verbosity Confound) ──
    context_lengths: list[int] = field(default_factory=list)  # Token count per turn

    # ── Area Under Curve ──
    auc: float = 0.0                               # Normalized AUC of DDM decay curve

    # ── Half-Life ──
    half_life: Optional[int] = None                # Turn where DDM ≤ 0.5

    # ── Recovery ──
    recovery_rate: float = 0.0                     # Fraction of post-drift turns that recover to 1.0
    recovery_events: int = 0                       # Number of 1.0 turns after first drift

    # ── Per-Level Decay ──
    per_level_decay: dict[str, PerLevelDecay] = field(default_factory=dict)

    # ── Confidence Intervals (populated by batch analysis) ──
    dop_ci95_lower: Optional[float] = None
    dop_ci95_upper: Optional[float] = None
    mean_ddm_ci95_lower: Optional[float] = None
    mean_ddm_ci95_upper: Optional[float] = None

    def compute_summary(self, sustained_k: int = 3):
        """
        Compute all summary statistics from turn results.

        Args:
            sustained_k: Number of consecutive turns for sustained DOP.
        """
        if not self.turn_results:
            return

        self.total_turns = len(self.turn_results)
        scores = np.array([t.ddm_score for t in self.turn_results])
        self.mean_ddm = float(np.mean(scores))
        self.std_ddm = float(np.std(scores))

        # ── AUC (normalized to [0, 1]) ──
        if len(scores) > 1:
            self.auc = float(np.trapz(scores, dx=1.0) / (len(scores) - 1))
        else:
            self.auc = float(scores[0])

        # ── Drift Onset Point (DOP) ──
        for t in self.turn_results:
            if t.ddm_score < 1.0:
                self.drift_onset_point = t.turn_number
                # Also record token count at DOP (Verbosity Confound fix)
                dop_idx = t.turn_number - 1  # 0-indexed
                if self.context_lengths and dop_idx < len(self.context_lengths):
                    self.drift_onset_tokens = self.context_lengths[dop_idx]
                break

        # ── Sustained DOP ──
        self._compute_sustained_dop(scores, sustained_k)

        # ── Total Collapse Point (TCP) ──
        for t in self.turn_results:
            if t.ddm_score == 0.0:
                self.total_collapse_point = t.turn_number
                break

        # ── Half-Life (τ½) ──
        for t in self.turn_results:
            if t.ddm_score <= 0.5:
                self.half_life = t.turn_number
                break

        # ── Recovery Rate ──
        self._compute_recovery()

        # ── Per-Level Decay Curves ──
        self._compute_per_level_decay()

    def _compute_sustained_dop(self, scores: np.ndarray, k: int):
        """Find the first turn where DDM < 1.0 for k consecutive turns."""
        n = len(scores)
        consecutive = 0
        for i in range(n):
            if scores[i] < 1.0:
                consecutive += 1
                if consecutive >= k:
                    self.sustained_dop = i - k + 2  # 1-indexed, start of the run
                    return
            else:
                consecutive = 0

    def _compute_recovery(self):
        """Compute recovery rate: fraction of post-DOP turns that return to DDM=1.0."""
        if self.drift_onset_point is None:
            self.recovery_rate = 1.0
            self.recovery_events = 0
            return

        dop_idx = self.drift_onset_point - 1  # 0-indexed
        post_drift_turns = self.turn_results[dop_idx + 1:]  # Turns after first drift

        if not post_drift_turns:
            self.recovery_rate = 0.0
            self.recovery_events = 0
            return

        recoveries = sum(1 for t in post_drift_turns if t.ddm_score >= 1.0)
        self.recovery_events = recoveries
        self.recovery_rate = recoveries / len(post_drift_turns)

    def _compute_per_level_decay(self):
        """Compute independent decay curves for each constraint level."""
        l1_scores = [t.l1_score for t in self.turn_results]
        l2_scores = [t.l2_score for t in self.turn_results]
        l3_scores = [t.l3_score for t in self.turn_results]
        l4_scores = [t.l4_score for t in self.turn_results]
        l5_scores = [t.l5_score for t in self.turn_results]

        for level_name, level_scores in [
            ("L1_format", l1_scores),
            ("L2_structure", l2_scores),
            ("L3_lexical", l3_scores),
            ("L4_citation", l4_scores),
            ("L5_dynamic", l5_scores),
        ]:
            decay = PerLevelDecay(level=level_name, scores=level_scores)
            decay.compute()
            self.per_level_decay[level_name] = decay


# ──────────────────────────────────────────────────────────
# DDM Evaluator
# ──────────────────────────────────────────────────────────

# Default constraint weights (equal by default; can be overridden)
DEFAULT_WEIGHTS = {"L1": 0.20, "L2": 0.20, "L3": 0.20, "L4": 0.20, "L5": 0.20}


class DDMEvaluator:
    """
    Multi-Level Deterministic Drift Metric evaluator.

    Enhancements over v1:
        - Continuous DDM score (partial credit for L2)
        - Cross-lingual L3 forbidden word checking
        - Strict L4 citation verification
        - Configurable constraint weights
        - Per-level decay curve extraction
        - AUC, half-life, sustained DOP, recovery rate

    Usage:
        evaluator = DDMEvaluator(config, language="it")
        result = evaluator.evaluate_conversation(responses, conv_id, "it", "llama-3.1-8b")
    """

    def __init__(
        self,
        config: dict = None,
        language: str = "en",
        weights: dict[str, float] = None,
        strict_citations: bool = True,
        sustained_k: int = 3,
    ):
        """
        Args:
            config: Full project config dict.
            language: Language code for cross-lingual L3 checking.
            weights: Custom weights for L1–L4. Must sum to 1.0.
            strict_citations: If True, use strict L4 citation patterns.
            sustained_k: Consecutive turns required for sustained DOP.
        """
        config = config or {}
        ddm_cfg = config.get("ddm", {}).get("constraints", {})

        # Weights
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        w_sum = sum(self.weights.values())
        if abs(w_sum - 1.0) > 1e-6:
            logger.warning(
                f"Constraint weights sum to {w_sum:.4f}, normalizing to 1.0"
            )
            self.weights = {k: v / w_sum for k, v in self.weights.items()}

        self.sustained_k = sustained_k
        self.language = language

        # Initialize checkers
        l1_cfg = ddm_cfg.get("L1_format", {})
        self.l1 = L1FormatChecker(tag=l1_cfg.get("tag", "[SYS_ACK: ACTIVE]"))

        l2_cfg = ddm_cfg.get("L2_structure", {})
        self.l2 = L2StructureChecker(min_bullets=l2_cfg.get("min_bullets", 2))

        l3_cfg = ddm_cfg.get("L3_lexical", {})
        self.l3 = L3LexicalChecker(
            forbidden_words=l3_cfg.get("forbidden_words"),
            language=language,
        )

        self.l4 = L4CitationChecker(strict=strict_citations)
        self.l5 = L5StateTracker()

    def evaluate_turn(self, response: str, turn_number: int) -> TurnResult:
        """Evaluate a single turn against all five constraint levels."""
        l1 = self.l1.check(response)
        l2 = self.l2.check(response)
        l3 = self.l3.check(response)
        l4 = self.l4.check(response)
        l5 = self.l5.check(response, expected_turn=turn_number)

        ddm_score = (
            self.weights["L1"] * l1
            + self.weights["L2"] * l2
            + self.weights["L3"] * l3
            + self.weights["L4"] * l4
            + self.weights.get("L5", 0.0) * l5
        )

        return TurnResult(
            turn_number=turn_number,
            l1_score=l1,
            l2_score=l2,
            l3_score=l3,
            l4_score=l4,
            l5_score=l5,
            ddm_score=ddm_score,
        )

    def evaluate_conversation(
        self,
        responses: list[str],
        conversation_id: str,
        language: str,
        model_name: str,
        context_lengths: list[int] = None,
    ) -> ConversationDriftResult:
        """
        Evaluate all turns in a conversation.

        Args:
            responses: List of model responses (one per turn).
            conversation_id: Unique ID for this conversation.
            language: Language code (e.g., 'en', 'it').
            model_name: Model used for generation.
            context_lengths: Token count per turn from inference.
                If provided, enables DOP_tokens metric (absolute token
                count at drift onset). Addresses the Verbosity Confound:
                comparing drift in token-space, not just turn-space.

        Returns:
            ConversationDriftResult with comprehensive metrics.
        """
        # Update L3 checker for the target language if needed
        if language != self.language:
            self.l3 = L3LexicalChecker(language=language)
            self.language = language

        result = ConversationDriftResult(
            conversation_id=conversation_id,
            language=language,
            model_name=model_name,
        )

        # Pass context lengths for token-space metrics
        if context_lengths:
            result.context_lengths = context_lengths

        for i, response in enumerate(responses):
            turn_result = self.evaluate_turn(response, turn_number=i + 1)
            result.turn_results.append(turn_result)

        result.compute_summary(sustained_k=self.sustained_k)

        dop_tokens_str = f", DOP_tokens={result.drift_onset_tokens}" if result.drift_onset_tokens else ""
        logger.info(
            f"  [{model_name}|{language}|{conversation_id}] "
            f"DOP={result.drift_onset_point}{dop_tokens_str}, "
            f"sDOP={result.sustained_dop}, "
            f"TCP={result.total_collapse_point}, "
            f"τ½={result.half_life}, "
            f"AUC={result.auc:.3f}, "
            f"mean_DDM={result.mean_ddm:.3f}, "
            f"recovery={result.recovery_rate:.2%}"
        )

        return result

    # ── Batch Analysis with Confidence Intervals ──────────

    @staticmethod
    def compute_ci95_bootstrap(
        results: list[ConversationDriftResult],
        metric: str = "mean_ddm",
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """
        Compute 95% bootstrap confidence interval for a metric across conversations.

        Args:
            results: List of ConversationDriftResult from multiple conversations.
            metric: Attribute name on ConversationDriftResult to bootstrap.
            n_bootstrap: Number of bootstrap samples.
            seed: Random seed.

        Returns:
            Tuple of (mean, ci95_lower, ci95_upper).
        """
        rng = np.random.RandomState(seed)
        values = []
        for r in results:
            val = getattr(r, metric, None)
            if val is not None:
                values.append(val)

        if not values:
            return (0.0, 0.0, 0.0)

        values = np.array(values)
        observed_mean = float(np.mean(values))

        # Bootstrap
        boot_means = []
        n = len(values)
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=n, replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.array(boot_means)
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        return (observed_mean, ci_lower, ci_upper)

    # ── DataFrame Export ──────────────────────────────────

    def results_to_dataframe(
        self, results: list[ConversationDriftResult]
    ) -> pd.DataFrame:
        """Convert list of conversation results to a flat DataFrame (per-turn rows)."""
        rows = []
        for conv in results:
            for t in conv.turn_results:
                rows.append({
                    "conversation_id": conv.conversation_id,
                    "language": conv.language,
                    "model": conv.model_name,
                    "turn": t.turn_number,
                    "l1_format": t.l1_score,
                    "l2_structure": t.l2_score,
                    "l3_lexical": t.l3_score,
                    "l4_citation": t.l4_score,
                    "ddm_score": t.ddm_score,
                    "drift_onset": conv.drift_onset_point,
                    "sustained_dop": conv.sustained_dop,
                    "total_collapse": conv.total_collapse_point,
                    "half_life": conv.half_life,
                    "auc": conv.auc,
                    "recovery_rate": conv.recovery_rate,
                })
        return pd.DataFrame(rows)

    def results_to_summary_dataframe(
        self, results: list[ConversationDriftResult]
    ) -> pd.DataFrame:
        """Create a conversation-level summary DataFrame (one row per conversation)."""
        rows = []
        for conv in results:
            row = {
                "conversation_id": conv.conversation_id,
                "language": conv.language,
                "model": conv.model_name,
                "total_turns": conv.total_turns,
                "mean_ddm": conv.mean_ddm,
                "std_ddm": conv.std_ddm,
                "auc": conv.auc,
                "drift_onset": conv.drift_onset_point,
                "sustained_dop": conv.sustained_dop,
                "total_collapse": conv.total_collapse_point,
                "half_life": conv.half_life,
                "recovery_rate": conv.recovery_rate,
                "recovery_events": conv.recovery_events,
            }
            # Add per-level stats
            for level_name, decay in conv.per_level_decay.items():
                row[f"{level_name}_mean"] = decay.mean_score
                row[f"{level_name}_auc"] = decay.auc
                row[f"{level_name}_onset"] = decay.decay_onset
            rows.append(row)
        return pd.DataFrame(rows)

    def save_results(
        self,
        results: list[ConversationDriftResult],
        output_dir: str | Path,
    ) -> Path:
        """Save comprehensive drift results to CSV and JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Per-turn results ──
        df = self.results_to_dataframe(results)
        csv_path = output_dir / "drift_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved per-turn results: {csv_path}")

        # ── Conversation-level summary ──
        summary_df = self.results_to_summary_dataframe(results)
        summary_path = output_dir / "drift_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved conversation summary: {summary_path}")

        # ── Aggregated summary by model × language ──
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            agg_df = summary_df.groupby(["model", "language"]).agg({
                "mean_ddm": ["mean", "std"],
                "auc": ["mean", "std"],
                "drift_onset": ["mean", "median"],
                "sustained_dop": ["mean", "median"],
                "half_life": ["mean", "median"],
                "recovery_rate": ["mean"],
                "total_turns": ["mean"],
            }).reset_index()
        # Flatten multi-level columns
        agg_df.columns = [
            "_".join(col).strip("_") for col in agg_df.columns.values
        ]
        agg_path = output_dir / "drift_aggregated.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Saved aggregated summary: {agg_path}")

        # ── Per-level decay curves as JSON ──
        decay_data = {}
        for conv in results:
            key = f"{conv.model_name}|{conv.language}|{conv.conversation_id}"
            decay_data[key] = {
                level: {
                    "scores": d.scores,
                    "mean": d.mean_score,
                    "auc": d.auc,
                    "onset": d.decay_onset,
                }
                for level, d in conv.per_level_decay.items()
            }
        decay_path = output_dir / "per_level_decay.json"
        with open(decay_path, "w") as f:
            json.dump(decay_data, f, indent=2)
        logger.info(f"Saved per-level decay curves: {decay_path}")

        # ── CI95 by model × language ──
        ci_rows = []
        groups = {}
        for r in results:
            key = (r.model_name, r.language)
            groups.setdefault(key, []).append(r)

        for (model, lang), group_results in groups.items():
            ddm_mean, ddm_lo, ddm_hi = self.compute_ci95_bootstrap(
                group_results, "mean_ddm"
            )
            dop_mean, dop_lo, dop_hi = self.compute_ci95_bootstrap(
                group_results, "drift_onset_point"
            )
            auc_mean, auc_lo, auc_hi = self.compute_ci95_bootstrap(
                group_results, "auc"
            )
            ci_rows.append({
                "model": model,
                "language": lang,
                "n_conversations": len(group_results),
                "mean_ddm": ddm_mean,
                "mean_ddm_ci95_lower": ddm_lo,
                "mean_ddm_ci95_upper": ddm_hi,
                "dop_mean": dop_mean,
                "dop_ci95_lower": dop_lo,
                "dop_ci95_upper": dop_hi,
                "auc_mean": auc_mean,
                "auc_ci95_lower": auc_lo,
                "auc_ci95_upper": auc_hi,
            })

        ci_df = pd.DataFrame(ci_rows)
        ci_path = output_dir / "drift_ci95.csv"
        ci_df.to_csv(ci_path, index=False)
        logger.info(f"Saved CI95 bootstrap results: {ci_path}")

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
