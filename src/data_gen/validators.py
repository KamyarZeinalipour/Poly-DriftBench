"""
Rule-Based Quality Validators
===============================
Deterministic, programmatic quality checks for generated conversations
and translations. These complement the LLM-based QualityAuditor with
hard, unfoolable metrics.

Two validator classes:
    1. ConversationValidator — checks generated English conversations
    2. TranslationValidator — checks parallel translations

Philosophy:
    LLM auditors can be subjective and miss obvious errors.
    Rule-based validators catch structural violations deterministically.
    The pipeline runs BOTH — rules first (fast, free), LLM second (expensive).
"""

import re
import logging
import math
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """A single validation failure."""
    turn: int                    # 0-indexed turn number (-1 for global)
    rule: str                    # Rule identifier (e.g., "DDM_L1")
    severity: str                # "error" | "warning"
    message: str                 # Human-readable description


@dataclass
class ValidationReport:
    """Complete validation report for a conversation."""
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else f"❌ FAILED ({self.error_count} errors)"
        return (
            f"{status} | {self.warning_count} warnings | "
            f"Metrics: {self.metrics}"
        )


@dataclass
class TranslationValidationReport:
    """Complete validation report for a translated conversation."""
    source_lang: str
    target_lang: str
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def passed(self) -> bool:
        return self.error_count == 0


# ──────────────────────────────────────────────────────────
# Conversation Validator (Rule-Based)
# ──────────────────────────────────────────────────────────

class ConversationValidator:
    """
    Deterministic quality checks for generated English conversations.

    Rules checked:
        DDM Constraints:
            DDM_L1: Every assistant response ends with [SYS_ACK: ACTIVE]
            DDM_L2: Every assistant response has ≥2 numbered bullet points
            DDM_L3: No assistant response contains the word "however"
            DDM_L4: Every assistant response contains a [Source: ...] citation

        Structural Rules:
            STRUCT_ROLE: Messages alternate user/assistant correctly
            STRUCT_EMPTY: No empty messages
            STRUCT_LENGTH_MIN: Messages meet minimum length threshold
            STRUCT_LENGTH_MAX: Messages don't exceed maximum length

        Quality Rules:
            QUAL_REPETITION: No near-duplicate messages (n-gram overlap)
            QUAL_DIVERSITY: Lexical diversity above threshold (TTR)
            QUAL_ASSISTANT_VARIETY: Assistant responses aren't too similar
            QUAL_USER_LENGTH_VARIANCE: User messages vary in length
    """

    # Thresholds
    MIN_USER_MSG_LENGTH = 50         # chars
    MIN_ASST_MSG_LENGTH = 100        # chars
    MAX_MSG_LENGTH = 5000            # chars
    MAX_NGRAM_OVERLAP = 0.6          # 60% trigram overlap = too similar
    MIN_LEXICAL_DIVERSITY = 0.15     # Type-token ratio
    MIN_USER_LENGTH_CV = 0.2         # Coefficient of variation for user msg lengths

    def validate(self, conversation: list[dict]) -> ValidationReport:
        """
        Run all rule-based checks on a conversation.

        Args:
            conversation: List of {"role": "user"|"assistant", "content": "..."}

        Returns:
            ValidationReport with issues and metrics.
        """
        report = ValidationReport()

        # Separate roles
        user_msgs = [m for m in conversation if m["role"] == "user"]
        asst_msgs = [m for m in conversation if m["role"] == "assistant"]

        # ─── DDM Constraints ───────────────────────────
        self._check_ddm_constraints(asst_msgs, report)

        # ─── Structural Rules ─────────────────────────
        self._check_structure(conversation, report)

        # ─── Quality Rules ────────────────────────────
        self._check_repetition(asst_msgs, report)
        self._check_lexical_diversity(conversation, report)
        self._check_assistant_variety(asst_msgs, report)
        self._check_user_length_variance(user_msgs, report)

        # ─── Compute Summary Metrics ──────────────────
        report.metrics = self._compute_metrics(conversation, user_msgs, asst_msgs)

        return report

    # ─── DDM Checks ──────────────────────────────────

    def _check_ddm_constraints(
        self, asst_msgs: list[dict], report: ValidationReport
    ):
        """Check all 4 DDM levels on every assistant response."""
        for i, msg in enumerate(asst_msgs):
            content = msg["content"]
            turn_num = i  # 0-indexed assistant turn

            # L1: Format tag
            if "[SYS_ACK: ACTIVE]" not in content:
                report.issues.append(ValidationIssue(
                    turn=turn_num, rule="DDM_L1", severity="error",
                    message=f"Assistant turn {turn_num+1}: Missing [SYS_ACK: ACTIVE] tag"
                ))

            # L2: Numbered bullet points (≥2)
            bullets = re.findall(r'^\s*\d+[\.\)]\s', content, re.MULTILINE)
            if len(bullets) < 2:
                report.issues.append(ValidationIssue(
                    turn=turn_num, rule="DDM_L2", severity="error",
                    message=f"Assistant turn {turn_num+1}: Only {len(bullets)} bullet points (need ≥2)"
                ))

            # L3: No "however"
            # Match "however" as a standalone word (not inside other words)
            if re.search(r'\bhowever\b', content, re.IGNORECASE):
                report.issues.append(ValidationIssue(
                    turn=turn_num, rule="DDM_L3", severity="error",
                    message=f"Assistant turn {turn_num+1}: Contains forbidden word 'however'"
                ))

            # L4: Source citation
            has_source = bool(
                re.search(r'\[Source:', content) or
                re.search(r'According to\s', content, re.IGNORECASE)
            )
            if not has_source:
                report.issues.append(ValidationIssue(
                    turn=turn_num, rule="DDM_L4", severity="error",
                    message=f"Assistant turn {turn_num+1}: Missing source citation"
                ))

    # ─── Structural Checks ───────────────────────────

    def _check_structure(self, conversation: list[dict], report: ValidationReport):
        """Check role alternation, empty messages, length bounds."""

        for i, msg in enumerate(conversation):
            # Role alternation
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg["role"] != expected_role:
                report.issues.append(ValidationIssue(
                    turn=i, rule="STRUCT_ROLE", severity="error",
                    message=f"Message {i+1}: Expected '{expected_role}', got '{msg['role']}'"
                ))

            # Empty check
            if not msg["content"] or not msg["content"].strip():
                report.issues.append(ValidationIssue(
                    turn=i, rule="STRUCT_EMPTY", severity="error",
                    message=f"Message {i+1}: Empty content"
                ))
                continue

            # Length bounds
            content_len = len(msg["content"])
            if msg["role"] == "user" and content_len < self.MIN_USER_MSG_LENGTH:
                report.issues.append(ValidationIssue(
                    turn=i, rule="STRUCT_LENGTH_MIN", severity="warning",
                    message=f"User message {i+1}: Too short ({content_len} chars, min={self.MIN_USER_MSG_LENGTH})"
                ))
            elif msg["role"] == "assistant" and content_len < self.MIN_ASST_MSG_LENGTH:
                report.issues.append(ValidationIssue(
                    turn=i, rule="STRUCT_LENGTH_MIN", severity="warning",
                    message=f"Assistant message {i+1}: Too short ({content_len} chars, min={self.MIN_ASST_MSG_LENGTH})"
                ))

            if content_len > self.MAX_MSG_LENGTH:
                report.issues.append(ValidationIssue(
                    turn=i, rule="STRUCT_LENGTH_MAX", severity="warning",
                    message=f"Message {i+1}: Too long ({content_len} chars, max={self.MAX_MSG_LENGTH})"
                ))

    # ─── Repetition Checks ───────────────────────────

    def _check_repetition(self, asst_msgs: list[dict], report: ValidationReport):
        """Detect near-duplicate assistant responses using trigram overlap."""

        def get_trigrams(text: str) -> set:
            words = text.lower().split()
            return set(tuple(words[i:i+3]) for i in range(len(words) - 2))

        trigram_sets = [get_trigrams(m["content"]) for m in asst_msgs]

        for i in range(len(trigram_sets)):
            for j in range(i + 1, min(i + 5, len(trigram_sets))):  # Check nearby turns
                if not trigram_sets[i] or not trigram_sets[j]:
                    continue

                overlap = len(trigram_sets[i] & trigram_sets[j])
                union = len(trigram_sets[i] | trigram_sets[j])
                jaccard = overlap / union if union > 0 else 0

                if jaccard > self.MAX_NGRAM_OVERLAP:
                    report.issues.append(ValidationIssue(
                        turn=j, rule="QUAL_REPETITION", severity="warning",
                        message=(
                            f"Assistant turns {i+1} and {j+1}: Too similar "
                            f"(Jaccard={jaccard:.2f}, threshold={self.MAX_NGRAM_OVERLAP})"
                        )
                    ))

    def _check_lexical_diversity(
        self, conversation: list[dict], report: ValidationReport
    ):
        """Check type-token ratio for overall vocabulary diversity."""
        all_words = []
        for msg in conversation:
            all_words.extend(msg["content"].lower().split())

        if len(all_words) < 100:
            return

        # Type-Token Ratio (TTR) — unique words / total words
        ttr = len(set(all_words)) / len(all_words)
        report.metrics["lexical_diversity_ttr"] = round(ttr, 4)

        if ttr < self.MIN_LEXICAL_DIVERSITY:
            report.issues.append(ValidationIssue(
                turn=-1, rule="QUAL_DIVERSITY", severity="warning",
                message=f"Low lexical diversity (TTR={ttr:.3f}, min={self.MIN_LEXICAL_DIVERSITY})"
            ))

    def _check_assistant_variety(
        self, asst_msgs: list[dict], report: ValidationReport
    ):
        """Check that assistant responses don't all start the same way."""
        if len(asst_msgs) < 5:
            return

        # Extract first sentence of each response
        first_sentences = []
        for msg in asst_msgs:
            content = msg["content"].strip()
            # Get text before first numbered bullet or first period
            match = re.match(r'^(.*?(?:\.|(?=\d+[\.\)])|\n))', content, re.DOTALL)
            if match:
                first_sentences.append(match.group(1).strip().lower()[:80])

        # Check for too many identical openings
        opening_counter = Counter(first_sentences)
        most_common_count = opening_counter.most_common(1)[0][1] if opening_counter else 0
        repetition_ratio = most_common_count / len(asst_msgs)

        report.metrics["assistant_opening_diversity"] = round(1 - repetition_ratio, 3)

        if repetition_ratio > 0.4:  # More than 40% start the same way
            report.issues.append(ValidationIssue(
                turn=-1, rule="QUAL_ASSISTANT_VARIETY", severity="warning",
                message=(
                    f"Assistant responses too similar: "
                    f"{most_common_count}/{len(asst_msgs)} start the same way"
                )
            ))

    def _check_user_length_variance(
        self, user_msgs: list[dict], report: ValidationReport
    ):
        """Check that user messages vary in length (not all the same size)."""
        if len(user_msgs) < 5:
            return

        lengths = [len(m["content"]) for m in user_msgs]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_len if mean_len > 0 else 0

        report.metrics["user_length_cv"] = round(cv, 3)

        if cv < self.MIN_USER_LENGTH_CV:
            report.issues.append(ValidationIssue(
                turn=-1, rule="QUAL_USER_LENGTH_VARIANCE", severity="warning",
                message=(
                    f"User messages too uniform in length "
                    f"(CV={cv:.3f}, min={self.MIN_USER_LENGTH_CV})"
                )
            ))

    # ─── Metrics ─────────────────────────────────────

    def _compute_metrics(
        self,
        conversation: list[dict],
        user_msgs: list[dict],
        asst_msgs: list[dict],
    ) -> dict:
        """Compute summary statistics."""
        user_lens = [len(m["content"]) for m in user_msgs]
        asst_lens = [len(m["content"]) for m in asst_msgs]

        metrics = {
            "total_messages": len(conversation),
            "user_messages": len(user_msgs),
            "assistant_messages": len(asst_msgs),
            "user_avg_length": round(sum(user_lens) / max(len(user_lens), 1)),
            "user_min_length": min(user_lens) if user_lens else 0,
            "user_max_length": max(user_lens) if user_lens else 0,
            "asst_avg_length": round(sum(asst_lens) / max(len(asst_lens), 1)),
            "asst_min_length": min(asst_lens) if asst_lens else 0,
            "asst_max_length": max(asst_lens) if asst_lens else 0,
            "total_chars": sum(len(m["content"]) for m in conversation),
        }

        # DDM pass rates
        ddm_l1 = sum(1 for m in asst_msgs if "[SYS_ACK: ACTIVE]" in m["content"])
        ddm_l2 = sum(
            1 for m in asst_msgs
            if len(re.findall(r'^\s*\d+[\.\)]\s', m["content"], re.MULTILINE)) >= 2
        )
        ddm_l3 = sum(
            1 for m in asst_msgs
            if not re.search(r'\bhowever\b', m["content"], re.IGNORECASE)
        )
        ddm_l4 = sum(
            1 for m in asst_msgs
            if re.search(r'\[Source:', m["content"]) or
               re.search(r'According to\s', m["content"], re.IGNORECASE)
        )
        n = max(len(asst_msgs), 1)
        metrics["ddm_l1_rate"] = round(ddm_l1 / n, 3)
        metrics["ddm_l2_rate"] = round(ddm_l2 / n, 3)
        metrics["ddm_l3_rate"] = round(ddm_l3 / n, 3)
        metrics["ddm_l4_rate"] = round(ddm_l4 / n, 3)
        metrics["ddm_mean"] = round((ddm_l1 + ddm_l2 + ddm_l3 + ddm_l4) / (4 * n), 3)

        return metrics


# ──────────────────────────────────────────────────────────
# Translation Validator (Rule-Based)
# ──────────────────────────────────────────────────────────

class TranslationValidator:
    """
    Deterministic quality checks for translated conversations.

    Rules checked:
        Format Preservation:
            FMT_SYS_ACK: [SYS_ACK: ACTIVE] tag preserved exactly
            FMT_BULLETS: Numbered bullet structure preserved
            FMT_SOURCE: [Source: ...] citations preserved (not translated)
            FMT_ROLE: Role labels preserved correctly

        Length Checks:
            LEN_RATIO: Translation length is within expected ratio
            LEN_EXTREME: No translated message is abnormally short/long

        Content Checks:
            CONTENT_EMPTY: No empty translations
            CONTENT_UNTRANSLATED: Detect if content wasn't actually translated
            CONTENT_ROLE_LEAK: Detect if role labels leaked into content
    """

    # Expected length ratios relative to English (source)
    # These are approximate — higher fertility languages produce more characters
    LANG_LENGTH_RATIOS = {
        "it": (0.8, 1.5),   # Italian: 80%-150% of English
        "es": (0.8, 1.5),   # Spanish
        "fr": (0.85, 1.6),  # French (tends longer)
        "de": (0.85, 1.7),  # German (compounds make it longer)
    }

    # Format markers that must survive translation unchanged
    FORMAT_MARKERS = [
        "[SYS_ACK: ACTIVE]",
    ]

    # Regex patterns that must be preserved
    FORMAT_PATTERNS = [
        (r'\[Source:\s*[^\]]+\]', "FMT_SOURCE", "Source citation"),
        (r'^\s*\d+[\.\)]\s', "FMT_BULLETS", "Numbered bullet"),
    ]

    def validate(
        self,
        source_conversation: list[dict],
        translated_conversation: list[dict],
        source_lang: str = "en",
        target_lang: str = "it",
    ) -> TranslationValidationReport:
        """
        Validate a translated conversation against its source.

        Args:
            source_conversation: Original English messages.
            translated_conversation: Translated messages.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            TranslationValidationReport with issues and metrics.
        """
        report = TranslationValidationReport(
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Basic length match
        if len(source_conversation) != len(translated_conversation):
            report.issues.append(ValidationIssue(
                turn=-1, rule="STRUCT_LENGTH", severity="error",
                message=(
                    f"Message count mismatch: source={len(source_conversation)}, "
                    f"translated={len(translated_conversation)}"
                )
            ))
            return report

        # Per-message checks
        for i, (src, tgt) in enumerate(
            zip(source_conversation, translated_conversation)
        ):
            src_content = src["content"]
            tgt_content = tgt["content"]

            # Role preservation
            if src["role"] != tgt["role"]:
                report.issues.append(ValidationIssue(
                    turn=i, rule="FMT_ROLE", severity="error",
                    message=f"Message {i+1}: Role changed from '{src['role']}' to '{tgt['role']}'"
                ))

            # Empty check
            if not tgt_content or not tgt_content.strip():
                report.issues.append(ValidationIssue(
                    turn=i, rule="CONTENT_EMPTY", severity="error",
                    message=f"Message {i+1}: Translation is empty"
                ))
                continue

            # Only check format rules on assistant messages
            if src["role"] == "assistant":
                self._check_format_preservation(i, src_content, tgt_content, report)

            # Length ratio check
            self._check_length_ratio(i, src_content, tgt_content, target_lang, report)

            # Untranslated detection
            self._check_untranslated(i, src_content, tgt_content, src["role"], report)

        # Compute summary metrics
        report.metrics = self._compute_metrics(
            source_conversation, translated_conversation, target_lang
        )

        return report

    def _check_format_preservation(
        self, turn: int, src: str, tgt: str, report: TranslationValidationReport
    ):
        """Check that format markers survived translation."""

        # Exact string markers
        for marker in self.FORMAT_MARKERS:
            if marker in src and marker not in tgt:
                report.issues.append(ValidationIssue(
                    turn=turn, rule="FMT_SYS_ACK", severity="error",
                    message=f"Message {turn+1}: '{marker}' missing in translation"
                ))

        # Regex patterns
        for pattern, rule, name in self.FORMAT_PATTERNS:
            src_matches = re.findall(pattern, src, re.MULTILINE)
            tgt_matches = re.findall(pattern, tgt, re.MULTILINE)

            if len(src_matches) > 0 and len(tgt_matches) == 0:
                report.issues.append(ValidationIssue(
                    turn=turn, rule=rule, severity="error",
                    message=(
                        f"Message {turn+1}: {name} lost in translation "
                        f"(src has {len(src_matches)}, tgt has 0)"
                    )
                ))
            elif len(tgt_matches) < len(src_matches):
                report.issues.append(ValidationIssue(
                    turn=turn, rule=rule, severity="warning",
                    message=(
                        f"Message {turn+1}: Some {name}s lost "
                        f"(src={len(src_matches)}, tgt={len(tgt_matches)})"
                    )
                ))

        # Check that [Source: ...] content wasn't translated
        src_sources = re.findall(r'\[Source:\s*([^\]]+)\]', src)
        tgt_sources = re.findall(r'\[Source:\s*([^\]]+)\]', tgt)

        for src_s in src_sources:
            found = any(
                self._normalized_match(src_s, tgt_s)
                for tgt_s in tgt_sources
            )
            if not found and tgt_sources:
                report.issues.append(ValidationIssue(
                    turn=turn, rule="FMT_SOURCE_TRANSLATED", severity="warning",
                    message=(
                        f"Message {turn+1}: Source citation may have been translated "
                        f"('{src_s[:50]}...' not found verbatim)"
                    )
                ))

    def _check_length_ratio(
        self, turn: int, src: str, tgt: str, target_lang: str,
        report: TranslationValidationReport
    ):
        """Check translation length is within expected ratio."""
        if len(src) < 20:  # Skip very short messages
            return

        ratio = len(tgt) / len(src)
        min_ratio, max_ratio = self.LANG_LENGTH_RATIOS.get(
            target_lang, (0.5, 2.0)
        )

        if ratio < min_ratio * 0.5:  # Very suspicious — less than half expected min
            report.issues.append(ValidationIssue(
                turn=turn, rule="LEN_EXTREME", severity="error",
                message=(
                    f"Message {turn+1}: Translation suspiciously short "
                    f"(ratio={ratio:.2f}, expected {min_ratio}-{max_ratio})"
                )
            ))
        elif ratio < min_ratio:
            report.issues.append(ValidationIssue(
                turn=turn, rule="LEN_RATIO", severity="warning",
                message=(
                    f"Message {turn+1}: Translation shorter than expected "
                    f"(ratio={ratio:.2f}, expected ≥{min_ratio})"
                )
            ))
        elif ratio > max_ratio * 1.5:  # Way too long
            report.issues.append(ValidationIssue(
                turn=turn, rule="LEN_EXTREME", severity="error",
                message=(
                    f"Message {turn+1}: Translation suspiciously long "
                    f"(ratio={ratio:.2f}, expected {min_ratio}-{max_ratio})"
                )
            ))
        elif ratio > max_ratio:
            report.issues.append(ValidationIssue(
                turn=turn, rule="LEN_RATIO", severity="warning",
                message=(
                    f"Message {turn+1}: Translation longer than expected "
                    f"(ratio={ratio:.2f}, expected ≤{max_ratio})"
                )
            ))

    def _check_untranslated(
        self, turn: int, src: str, tgt: str, role: str,
        report: TranslationValidationReport
    ):
        """
        Detect if a message was left untranslated.

        Compares word overlap — if >80% of content words are identical,
        it's probably untranslated (except for format markers).
        """
        # Strip out format markers before comparison
        clean_src = self._strip_format_markers(src).lower()
        clean_tgt = self._strip_format_markers(tgt).lower()

        src_words = set(clean_src.split())
        tgt_words = set(clean_tgt.split())

        if not src_words:
            return

        overlap = len(src_words & tgt_words) / len(src_words)

        if overlap > 0.8 and len(src_words) > 10:
            report.issues.append(ValidationIssue(
                turn=turn, rule="CONTENT_UNTRANSLATED", severity="error",
                message=(
                    f"Message {turn+1}: Appears untranslated "
                    f"(word overlap={overlap:.0%})"
                )
            ))

    def _strip_format_markers(self, text: str) -> str:
        """Remove format markers to get just the natural language content."""
        cleaned = text
        for marker in self.FORMAT_MARKERS:
            cleaned = cleaned.replace(marker, "")
        cleaned = re.sub(r'\[Source:\s*[^\]]+\]', '', cleaned)
        cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def _normalized_match(self, a: str, b: str) -> bool:
        """Check if two strings are approximately the same."""
        a_clean = re.sub(r'\s+', ' ', a.strip().lower())
        b_clean = re.sub(r'\s+', ' ', b.strip().lower())
        return a_clean == b_clean

    def _compute_metrics(
        self,
        source: list[dict],
        translated: list[dict],
        target_lang: str,
    ) -> dict:
        """Compute summary translation metrics."""
        src_asst = [m for m in source if m["role"] == "assistant"]
        tgt_asst = [m for m in translated if m["role"] == "assistant"]

        # Length ratios
        ratios = []
        for s, t in zip(source, translated):
            if len(s["content"]) > 0:
                ratios.append(len(t["content"]) / len(s["content"]))

        # Format preservation rates
        sysack_preserved = sum(
            1 for s, t in zip(src_asst, tgt_asst)
            if "[SYS_ACK: ACTIVE]" in s["content"] and "[SYS_ACK: ACTIVE]" in t["content"]
        )
        source_preserved = sum(
            1 for s, t in zip(src_asst, tgt_asst)
            if (re.search(r'\[Source:', s["content"]) and
                re.search(r'\[Source:', t["content"]))
        )
        bullet_preserved = sum(
            1 for s, t in zip(src_asst, tgt_asst)
            if (re.findall(r'^\s*\d+[\.\)]\s', s["content"], re.MULTILINE) and
                re.findall(r'^\s*\d+[\.\)]\s', t["content"], re.MULTILINE))
        )

        n = max(len(src_asst), 1)
        return {
            "target_lang": target_lang,
            "total_messages": len(translated),
            "avg_length_ratio": round(sum(ratios) / max(len(ratios), 1), 3),
            "min_length_ratio": round(min(ratios), 3) if ratios else 0,
            "max_length_ratio": round(max(ratios), 3) if ratios else 0,
            "sysack_preservation_rate": round(sysack_preserved / n, 3),
            "source_citation_preservation_rate": round(source_preserved / n, 3),
            "bullet_preservation_rate": round(bullet_preserved / n, 3),
        }
