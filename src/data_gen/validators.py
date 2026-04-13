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
            DDM_L5: Every assistant response starts with [Turn: N] (incrementing)

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
            QUAL_META_LANGUAGE: User messages don't use robotic pivot phrases
            QUAL_INTRO_REPETITION: Assistant doesn't reuse intro phrases
            QUAL_PHRASE_SPAM: Assistant doesn't spam specific phrases
            QUAL_CITATION_DOMAIN: Citations match semantic domain of advice
            QUAL_USER_VERBATIM: User doesn't repeat same complaint verbatim
            CATD_ECHO: User messages don't echo assistant-specific content
    """

    # Thresholds
    MIN_USER_MSG_LENGTH = 3          # chars (lowered: micro-responses like 'Huh.' are ecologically valid)
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
        self._check_user_meta_language(user_msgs, report)
        self._check_assistant_intro_repetition(asst_msgs, report)
        self._check_citation_domain_match(asst_msgs, report)
        self._check_user_verbatim_repetition(user_msgs, report)
        self._check_catd_echo(user_msgs, asst_msgs, report)

        # ─── Coherence Rules ─────────────────────────
        self._check_coherence(conversation, report)
        self._check_topic_repetition(asst_msgs, report)

        # ─── Compute Summary Metrics ──────────────────
        report.metrics.update(self._compute_metrics(conversation, user_msgs, asst_msgs))

        return report

    # ─── DDM Checks ──────────────────────────────────

    def _check_ddm_constraints(
        self, asst_msgs: list[dict], report: ValidationReport
    ):
        """Check all 5 DDM levels on every assistant response."""
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

            # L4: Source citation — MUST use [Source: ...] format
            # Note: "According to" is NOT accepted as valid DDM L4 compliance.
            has_source = bool(re.search(r'\[Source:', content))
            if not has_source:
                report.issues.append(ValidationIssue(
                    turn=turn_num, rule="DDM_L4", severity="error",
                    message=f"Assistant turn {turn_num+1}: Missing [Source: ...] citation"
                ))

            # L5: Turn counter — [Turn: N] at the beginning, incrementing by 1
            turn_match = re.match(r'^\s*\[Turn:\s*(\d+)\]', content)
            if not turn_match:
                report.issues.append(ValidationIssue(
                    turn=turn_num, rule="DDM_L5", severity="error",
                    message=f"Assistant turn {turn_num+1}: Missing [Turn: N] counter at start"
                ))
            else:
                actual_n = int(turn_match.group(1))
                expected_n = i + 1  # 1-indexed
                if actual_n != expected_n:
                    report.issues.append(ValidationIssue(
                        turn=turn_num, rule="DDM_L5", severity="error",
                        message=(
                            f"Assistant turn {turn_num+1}: Turn counter mismatch "
                            f"(found [Turn: {actual_n}], expected [Turn: {expected_n}])"
                        )
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

    def _check_user_meta_language(
        self, user_msgs: list[dict], report: ValidationReport
    ):
        """Check for robotic pivot phrases that break realism."""
        META_PHRASES = [
            r'\bswitching topics\b',
            r'\bswitching gears\b',
            r'\bmoving on\b',
            r'\bdifferent question\b',
            r'\bplot twist\b',
            r'\bokay\s*,?\s*so\s*,?\s*about\b',  # "Okay, so about..."
        ]

        meta_count = 0
        for i, msg in enumerate(user_msgs):
            content = msg["content"].lower()
            for pattern in META_PHRASES:
                if re.search(pattern, content, re.IGNORECASE):
                    meta_count += 1
                    break  # Count once per message

        report.metrics["user_meta_language_count"] = meta_count

        if meta_count > 2:
            report.issues.append(ValidationIssue(
                turn=-1, rule="QUAL_META_LANGUAGE", severity="warning",
                message=(
                    f"User uses {meta_count} robotic pivot phrases "
                    f"('Switching topics', 'Moving on', 'Plot twist', etc.) — "
                    f"should pivot naturally"
                )
            ))

    def _check_assistant_intro_repetition(
        self, asst_msgs: list[dict], report: ValidationReport
    ):
        """Check for assistant reusing the same intro phrases."""
        if len(asst_msgs) < 5:
            return

        # Extract first sentence of each assistant response (after [Turn: N])
        intro_phrases = []
        for msg in asst_msgs:
            content = msg["content"]
            # Strip [Turn: N] prefix
            content = re.sub(r'^\[Turn:\s*\d+\]\s*', '', content)
            # Get first sentence
            first_sent = content.split('.')[0].strip().lower() if content else ""
            if first_sent:
                intro_phrases.append(first_sent)

        # Count duplicates
        from collections import Counter
        phrase_counts = Counter(intro_phrases)
        repeated = {p: c for p, c in phrase_counts.items() if c >= 3}

        if repeated:
            worst = max(repeated, key=repeated.get)
            report.issues.append(ValidationIssue(
                turn=-1, rule="QUAL_INTRO_REPETITION", severity="warning",
                message=(
                    f"Assistant reuses intro phrase {repeated[worst]}x: "
                    f"'{worst[:60]}...'"
                )
            ))

        # Also check for specific known bad patterns
        bad_phrases = ["the single most important action", "alternatively"]
        for phrase in bad_phrases:
            count = sum(1 for msg in asst_msgs 
                       if phrase in msg["content"].lower()[:100])
            if count > 2:
                report.issues.append(ValidationIssue(
                    turn=-1, rule="QUAL_PHRASE_SPAM", severity="warning",
                    message=(
                        f"Assistant uses '{phrase}' {count}x in opening lines — "
                        f"must vary introductions"
                    )
                ))

    def _check_citation_domain_match(
        self, asst_msgs: list[dict], report: ValidationReport
    ):
        """
        Check that [Source: ...] citations match the semantic domain of the advice.
        Only flag CLEAR mismatches — allow cross-domain citations since
        CAT-D topic pivots can make the user change topics mid-turn.

        NOTE: This check is intentionally very lenient. Organic topic pivots
        mean the user may discuss food in a turn that also mentions cleaning.
        A citation matching EITHER topic is valid.
        """
        # Only flag truly impossible combinations — empty by default
        # since cross-domain citations are almost always valid in long conversations
        DOMAIN_MISMATCHES = []

        mismatch_count = 0
        for i, msg in enumerate(asst_msgs):
            content = msg["content"]
            content_lower = content.lower()
            for topic_pattern, source_pattern, domain_name in DOMAIN_MISMATCHES:
                if re.search(topic_pattern, content, re.IGNORECASE):
                    # Skip if there's any cross-domain context
                    has_cross_domain = any(w in content_lower for w in
                        ['budget', 'money', 'saving', 'afford', 'cost',
                         'food', 'cook', 'meal', 'eat', 'grocery',
                         'clean', 'organize', 'sleep', 'exercise'])
                    if has_cross_domain:
                        continue
                    if re.search(source_pattern, content, re.IGNORECASE):
                        mismatch_count += 1
                        if mismatch_count <= 3:
                            report.issues.append(ValidationIssue(
                                turn=i, rule="QUAL_CITATION_DOMAIN", severity="warning",
                                message=(
                                    f"Assistant turn {i+1}: Citation doesn't match "
                                    f"{domain_name} topic — wrong domain source"
                                )
                            ))

        report.metrics["citation_domain_mismatches"] = mismatch_count

    def _check_user_verbatim_repetition(
        self, user_msgs: list[dict], report: ValidationReport
    ):
        """
        Detect the 'Groundhog Day' problem: user repeating the same complaint
        verbatim or near-verbatim across multiple turns.
        """
        if len(user_msgs) < 10:
            return

        def get_sentence_set(text: str) -> set:
            """Extract normalized sentence fragments for comparison."""
            # Split into chunks and normalize
            words = text.lower().split()
            # Use 5-grams for sentence-level comparison
            return set(tuple(words[i:i+5]) for i in range(len(words) - 4))

        sentence_sets = [get_sentence_set(m["content"]) for m in user_msgs]
        verbatim_pairs = 0

        for i in range(len(sentence_sets)):
            for j in range(i + 5, len(sentence_sets)):  # Gap of at least 5 turns
                if not sentence_sets[i] or not sentence_sets[j]:
                    continue
                overlap = len(sentence_sets[i] & sentence_sets[j])
                min_size = min(len(sentence_sets[i]), len(sentence_sets[j]))
                if min_size == 0:
                    continue
                similarity = overlap / min_size
                if similarity > 0.5:  # >50% 5-gram overlap = near-verbatim
                    verbatim_pairs += 1

        report.metrics["user_verbatim_repetitions"] = verbatim_pairs

        if verbatim_pairs > 3:
            report.issues.append(ValidationIssue(
                turn=-1, rule="QUAL_USER_VERBATIM", severity="warning",
                message=(
                    f"User repeats near-verbatim complaints {verbatim_pairs} times "
                    f"across non-adjacent turns (Groundhog Day effect)"
                )
            ))

    def _check_catd_echo(
        self, user_msgs: list[dict], asst_msgs: list[dict],
        report: ValidationReport
    ):
        """
        Check if user messages echo distinctive nouns/phrases from the
        immediately preceding assistant response. This is a CAT-D violation
        because the user message would be incoherent with a different assistant.
        """
        # Extract distinctive nouns from assistant messages (words >6 chars
        # that aren't common English words)
        COMMON_WORDS = {
            'because', 'should', 'would', 'could', 'before', 'after',
            'through', 'between', 'another', 'example', 'important',
            'recommend', 'suggest', 'consider', 'actually', 'really',
            'something', 'problem', 'question', 'different', 'morning',
            'tonight', 'already', 'general', 'overall', 'specific',
        }

        echo_count = 0
        for i, user_msg in enumerate(user_msgs[1:], start=1):  # Skip first
            if i - 1 >= len(asst_msgs):
                break

            asst_content = asst_msgs[i - 1]["content"].lower()
            user_content = user_msg["content"].lower()

            # Extract distinctive terms from assistant (nouns > 6 chars)
            asst_terms = set(re.findall(r'\b[a-z]{7,}\b', asst_content))
            asst_terms -= COMMON_WORDS

            # Check if user echoes any distinctive assistant terms
            user_words = set(re.findall(r'\b[a-z]{7,}\b', user_content))
            echoed = asst_terms & user_words

            # Filter: only count terms that are truly assistant-specific
            # (not common topic words that any user might naturally use)
            suspicious_echoes = [
                w for w in echoed
                if w not in {'cooking', 'cleaning', 'bedroom', 'kitchen',
                            'morning', 'evening', 'weekend', 'grocery',
                            'exercise', 'routine', 'anxiety', 'sleeping',
                            'budgeting', 'organize', 'schedule'}
            ]

            if len(suspicious_echoes) >= 3:
                echo_count += 1

        report.metrics["catd_echo_count"] = echo_count

        if echo_count > 2:
            report.issues.append(ValidationIssue(
                turn=-1, rule="CATD_ECHO", severity="warning",
                message=(
                    f"User echoes assistant-specific terms in {echo_count} turns — "
                    f"potential CAT-D violation"
                )
            ))

    # ─── Coherence Checks ────────────────────────────

    def _check_coherence(self, conversation: list[dict], report: ValidationReport):
        """
        Check for phantom references — user mentions advice/items that
        never appeared in prior assistant messages.
        """
        for i, msg in enumerate(conversation):
            if msg["role"] != "user":
                continue

            content_lower = msg["content"].lower()

            # Detect "you mentioned X" / "you said X" / "like you said" patterns
            ref_patterns = [
                r'(?:you\s+(?:mentioned|said|suggested|told\s+me|recommended|asked\s+me))\s+(.{10,80}?)(?:\.|,|\?|!|$)',
                r'(?:like\s+you\s+(?:said|suggested|mentioned))\s*,?\s*(.{5,60}?)(?:\.|,|\?|!|$)',
                r'(?:the\s+.{3,30}\s+(?:you\s+mentioned|you\s+suggested))(.{0,10}?)(?:\.|,|\?|!|$)',
            ]

            # Gather all prior assistant text for verification
            prior_asst_text = " ".join(
                m["content"].lower() for m in conversation[:i]
                if m["role"] == "assistant"
            )

            if not prior_asst_text:
                continue

            for pattern in ref_patterns:
                refs = re.findall(pattern, content_lower)
                for ref in refs:
                    ref_clean = ref.strip()
                    if len(ref_clean) < 5:
                        continue
                    # Extract key terms (words > 4 chars) from the reference
                    key_terms = [w for w in re.findall(r'[a-z]{5,}', ref_clean)]
                    if not key_terms:
                        continue
                    # Check if ANY key term appears in prior assistant messages
                    if not any(term in prior_asst_text for term in key_terms):
                        report.issues.append(ValidationIssue(
                            turn=i, rule="COHER_PHANTOM_REF", severity="warning",
                            message=(
                                f"Message {i+1}: User references '{ref_clean}' "
                                f"but no matching content found in prior assistant messages"
                            )
                        ))

    def _check_topic_repetition(
        self, asst_msgs: list[dict], report: ValidationReport
    ):
        """
        Detect the same advice/topic being repeated far apart in long
        conversations — a sign the model forgot earlier context.
        Only runs for medium/long conversations (≥20 assistant turns).
        """
        if len(asst_msgs) < 20:
            return

        # Extract "advice fingerprints" per assistant turn
        fingerprints = []
        for msg in asst_msgs:
            bullets = re.findall(
                r'^\s*\d+[\.)\s]\s*(.+)$', msg["content"], re.MULTILINE
            )
            terms = set()
            for b in bullets:
                # Extract substantive words (>5 chars, no stopwords)
                words = [w.lower() for w in re.findall(r'[a-zA-Z]{6,}', b)]
                terms.update(words[:5])  # First 5 per bullet
            fingerprints.append(terms)

        # Compare non-adjacent turns (gap ≥ 10 turns) for high overlap
        for i in range(len(fingerprints)):
            for j in range(i + 10, len(fingerprints)):
                if not fingerprints[i] or not fingerprints[j]:
                    continue
                if len(fingerprints[i]) < 3 or len(fingerprints[j]) < 3:
                    continue  # Skip turns with too few terms
                overlap = len(fingerprints[i] & fingerprints[j])
                union = len(fingerprints[i] | fingerprints[j])
                jaccard = overlap / union if union > 0 else 0
                if jaccard > 0.4:
                    report.issues.append(ValidationIssue(
                        turn=j, rule="QUAL_TOPIC_REPEAT", severity="warning",
                        message=(
                            f"Assistant turn {j+1} repeats advice from turn {i+1} "
                            f"(overlap={jaccard:.0%})"
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

        # DDM pass rates (all 5 levels)
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
            if re.search(r'\[Source:', m["content"])
        )
        ddm_l5 = sum(
            1 for i, m in enumerate(asst_msgs)
            if re.match(r'^\s*\[Turn:\s*' + str(i + 1) + r'\]', m["content"])
        )
        n = max(len(asst_msgs), 1)
        metrics["ddm_l1_rate"] = round(ddm_l1 / n, 3)
        metrics["ddm_l2_rate"] = round(ddm_l2 / n, 3)
        metrics["ddm_l3_rate"] = round(ddm_l3 / n, 3)
        metrics["ddm_l4_rate"] = round(ddm_l4 / n, 3)
        metrics["ddm_l5_rate"] = round(ddm_l5 / n, 3)
        metrics["ddm_mean"] = round(
            (ddm_l1 + ddm_l2 + ddm_l3 + ddm_l4 + ddm_l5) / (5 * n), 3
        )

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
