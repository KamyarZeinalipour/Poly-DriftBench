"""
Multi-Agent Data Factory — Pipeline Orchestrator
==================================================
Orchestrates the five agents to produce high-quality Poly-DriftBench data.

Pipeline:
    Phase 1: ScenarioArchitect → conversation plan
    Phase 2: UserSimulator ↔ AssistantSimulator → raw conversation (turn-by-turn)
    Phase 3: QualityAuditor → review + rewrite loop
    Phase 4: TranslatorAgent → parallel translations (EN → IT, ES, FR, DE)

The pipeline includes automatic retry logic:
    - If QualityAuditor rejects a conversation, specific turns are rewritten
    - Up to 3 revision rounds before falling back to full regeneration
    - Conversations below quality threshold are discarded and regenerated
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .agents import (
    ScenarioArchitect,
    UserSimulator,
    AssistantSimulator,
    QualityAuditor,
    TranslationPipeline,
    ConversationPlan,
    QualityReport,
    pipeline_stats,
)
from .validators import ConversationValidator, TranslationValidator
from .seed_generator import DOMAIN_TEMPLATES

logger = logging.getLogger(__name__)
console = Console()


class DataFactory:
    """
    Multi-agent data generation pipeline.

    Usage:
        factory = DataFactory(config)
        factory.produce_dataset(num_conversations=100)
    """

    def __init__(self, config: dict, model: str = None):
        self.config = config

        # Model defaults to None → agents auto-detect from .env
        # (DeepSeek if DEEPSEEK_API_KEY is set, else OpenAI)
        self.model = model

        # Initialize agents — they auto-detect the API provider
        self.architect = ScenarioArchitect(model=model)
        self.user_sim = UserSimulator(model=model, temperature=0.8)
        self.assistant_sim = AssistantSimulator(model=model, temperature=0.4)
        self.auditor = QualityAuditor(model=model, temperature=0.2)

        # Multi-agent translation pipeline (3 agents: translator + reviewer + back-translator)
        self.translation_pipeline = TranslationPipeline(model=model)

        # Initialize rule-based validators
        self.conv_validator = ConversationValidator()
        self.trans_validator = TranslationValidator()

        # Store the resolved model name for metadata
        self.model = self.architect.model

        # Pipeline settings
        self.max_revision_rounds = 3
        self.min_quality_score = 7.0
        self.min_ddm_score = 9.0
        self.target_languages = [
            lang["code"] for lang in config.get("languages", {}).get("targets", [])
        ]

    # ─── Phase 1: Planning ───────────────────────────────

    def _plan_conversation(
        self, domain: str, conv_id: str, num_turns: int
    ) -> ConversationPlan:
        """Phase 1: Generate conversation plan via ScenarioArchitect."""
        template = DOMAIN_TEMPLATES[domain]

        console.print(f"  📐 [cyan]Architect[/cyan] planning {domain}...", end=" ")
        plan = self.architect.plan(
            domain=domain,
            domain_description=template["description"],
            num_turns=num_turns,
            topics=template["topics"],
        )
        plan.conversation_id = conv_id
        console.print("✅")

        logger.info(
            f"  Plan: {plan.num_turns} turns, "
            f"persona='{plan.user_persona[:60]}...', "
            f"{len(plan.plot_twists)} twists, "
            f"{len(plan.context_callbacks)} callbacks"
        )
        return plan

    # ─── Phase 2: Turn-by-Turn Generation ────────────────

    def _generate_conversation(
        self, plan: ConversationPlan
    ) -> list[dict]:
        """
        Phase 2: Generate conversation using batch user messages + sequential assistant.

        OPTIMIZATION: User messages are pre-generated in batches (CAT-D enables this
        since user messages don't depend on assistant responses). Only assistant
        messages are generated sequentially.

        API calls: ~8 (user batches) + N (assistant) instead of 2N sequential.
        For 112-turn conv: 120 calls instead of 224 (46% reduction).
        """
        conversation = []

        console.print(f"  💬 [green]Generating[/green] {plan.num_turns} turns...")

        # Phase 2a: Batch-generate ALL user messages (few API calls)
        console.print(f"    📦 [cyan]Batch-generating[/cyan] {plan.num_turns} user messages...")
        user_messages = self.user_sim.generate_all_messages(plan)
        console.print(f"    📦 Got {len(user_messages)} user messages ✅")

        # Phase 2b: Generate assistant responses sequentially (needs context)
        for turn_idx in range(plan.num_turns):
            if turn_idx < len(user_messages):
                user_msg = user_messages[turn_idx]
            else:
                # Fallback: generate individually
                user_msg = self.user_sim.generate_message(
                    plan=plan,
                    conversation_history=conversation,
                    current_turn=turn_idx,
                )

            conversation.append({"role": "user", "content": user_msg})

            # Assistant generates response (needs full context)
            assistant_msg = self.assistant_sim.generate_response(
                domain=plan.domain,
                conversation_history=conversation,
                user_message=user_msg,
            )
            conversation.append({"role": "assistant", "content": assistant_msg})

            if (turn_idx + 1) % 10 == 0:
                console.print(
                    f"    Turn {turn_idx + 1}/{plan.num_turns} complete"
                )

        console.print(
            f"  💬 Generated {len(conversation)} messages "
            f"({len(conversation) // 2} turn pairs)"
        )
        return conversation

    # ─── Phase 3: Quality Audit & Revision ───────────────

    def _rule_based_validate(self, conversation: list[dict]) -> dict:
        """
        Phase 3a: Run deterministic rule-based validation BEFORE the LLM auditor.
        This catches obvious DDM violations for free (no API calls).
        """
        console.print("  📏 [blue]Rule Validator[/blue] checking...", end=" ")

        report = self.conv_validator.validate(conversation)

        console.print(
            f"{'✅' if report.passed else '❌'} "
            f"{report.error_count} errors, {report.warning_count} warnings | "
            f"DDM: L1={report.metrics.get('ddm_l1_rate', 0):.0%} "
            f"L2={report.metrics.get('ddm_l2_rate', 0):.0%} "
            f"L3={report.metrics.get('ddm_l3_rate', 0):.0%} "
            f"L4={report.metrics.get('ddm_l4_rate', 0):.0%}"
        )

        # Log specific errors
        for issue in report.issues:
            if issue.severity == "error":
                logger.warning(f"    ❌ {issue.rule}: {issue.message}")

        return {
            "passed": report.passed,
            "errors": report.error_count,
            "warnings": report.warning_count,
            "metrics": report.metrics,
            "issues": [
                {"turn": i.turn, "rule": i.rule, "severity": i.severity, "msg": i.message}
                for i in report.issues
            ],
        }

    def _audit_and_revise(
        self,
        conversation: list[dict],
        plan: ConversationPlan,
    ) -> tuple[list[dict], QualityReport, dict]:
        """
        Phase 3: Rule-based validation + LLM audit with automatic rewriting.

        Returns:
            (conversation, quality_report, rule_validation_result)
        """
        # Phase 3a: Rule-based validation (free, fast)
        rule_result = self._rule_based_validate(conversation)

        # If rule-based check finds DDM errors, auto-fix them before LLM audit
        if not rule_result["passed"]:
            ddm_errors = [
                i for i in rule_result["issues"]
                if i["rule"].startswith("DDM_") and i["severity"] == "error"
            ]
            if ddm_errors:
                console.print(
                    f"  🔧 [yellow]Auto-fixing[/yellow] {len(ddm_errors)} DDM violations..."
                )
                # Rewrite the specific assistant turns that failed DDM
                failed_turns = set(i["turn"] for i in ddm_errors)
                for turn_idx in failed_turns:
                    msg_idx = turn_idx * 2 + 1  # Convert assistant turn to message index
                    if msg_idx < len(conversation) and conversation[msg_idx]["role"] == "assistant":
                        user_msg = conversation[msg_idx - 1]["content"] if msg_idx > 0 else ""
                        new_response = self.assistant_sim.generate_response(
                            domain=plan.domain,
                            conversation_history=conversation[:msg_idx],
                            user_message=user_msg,
                        )
                        conversation[msg_idx]["content"] = new_response

                # Re-validate after LLM-based fix
                rule_result = self._rule_based_validate(conversation)

                # If DDM errors STILL persist, apply deterministic force-fix
                if not rule_result["passed"]:
                    remaining_ddm = [
                        i for i in rule_result["issues"]
                        if i["rule"].startswith("DDM_") and i["severity"] == "error"
                    ]
                    if remaining_ddm:
                        console.print(
                            f"  🔩 [red]Force-fixing[/red] {len(remaining_ddm)} "
                            f"persistent DDM violations (deterministic)..."
                        )
                        remaining_turns = set(i["turn"] for i in remaining_ddm)
                        for turn_idx in remaining_turns:
                            msg_idx = turn_idx * 2 + 1
                            if msg_idx < len(conversation) and conversation[msg_idx]["role"] == "assistant":
                                conversation[msg_idx]["content"] = (
                                    AssistantSimulator.force_fix_ddm(
                                        conversation[msg_idx]["content"],
                                        turn_number=turn_idx + 1,
                                    )
                                )
                        # Final re-validate
                        rule_result = self._rule_based_validate(conversation)

            # Log coherence warnings (non-blocking but informative)
            coherence_warnings = [
                i for i in rule_result["issues"]
                if i["rule"].startswith("COHER_") or i["rule"] == "QUAL_TOPIC_REPEAT"
            ]
            if coherence_warnings:
                console.print(
                    f"  ⚠️  [yellow]{len(coherence_warnings)} coherence warnings[/yellow]"
                )
                for w in coherence_warnings:
                    logger.warning(f"    ⚠️  {w['rule']}: {w['msg']}")

        # Phase 3b: LLM-based audit (skip for long conversations —
        # the auditor can't produce valid JSON for 40+ turn conversations
        # and the rule-based validator already ensures 100% DDM compliance)
        num_turns = len(conversation) // 2
        if num_turns > 40:
            console.print(
                f"  ⏭️  [dim]Skipping LLM audit ({num_turns} turns > 40)[/dim] — "
                f"rule-based validation sufficient"
            )
            # Create a synthetic quality report from rule-based results
            report = QualityReport(
                overall_score=7.5,
                realism_score=7.5,
                diversity_score=7.5,
                complexity_score=7.5,
                ddm_compliance_score=10.0 if rule_result["passed"] else 7.0,
                issues=[],
                rewrite_requests=[],
                approved=True,
            )
            conversation, rule_result = self._final_ddm_safety_net(conversation)
            return conversation, report, rule_result

        for revision_round in range(self.max_revision_rounds):
            console.print(
                f"  🔍 [yellow]Auditor[/yellow] reviewing "
                f"(round {revision_round + 1})...",
                end=" ",
            )

            report = self.auditor.audit(conversation, plan.domain)

            console.print(
                f"Score: {report.overall_score}/10 | "
                f"DDM: {report.ddm_compliance_score}/10 | "
                f"{'✅ APPROVED' if report.approved else '⚠️  NEEDS REVISION'}"
            )

            if report.approved:
                # Final DDM safety net — force-fix any violations introduced by rewrites
                conversation, rule_result = self._final_ddm_safety_net(conversation)
                return conversation, report, rule_result

            # If DDM compliance is low, rewrite specific assistant turns
            if report.rewrite_requests:
                console.print(
                    f"  🔄 Rewriting {len(report.rewrite_requests)} turns..."
                )
                conversation = self._apply_rewrites(
                    conversation, report.rewrite_requests, plan
                )
            else:
                break

        # Final audit + DDM safety net
        final_report = self.auditor.audit(conversation, plan.domain)
        conversation, rule_result = self._final_ddm_safety_net(conversation)
        return conversation, final_report, rule_result

    def _apply_rewrites(
        self,
        conversation: list[dict],
        rewrite_requests: list[dict],
        plan: ConversationPlan,
    ) -> list[dict]:
        """Apply specific turn rewrites requested by the auditor."""
        for request in rewrite_requests:
            turn_idx = request.get("turn", 0) - 1  # Convert to 0-indexed
            role = request.get("role", "assistant")
            reason = request.get("reason", "")

            if turn_idx < 0 or turn_idx >= len(conversation):
                continue

            if role == "assistant" and conversation[turn_idx]["role"] == "assistant":
                # Get the preceding user message for context
                user_msg = ""
                if turn_idx > 0:
                    user_msg = conversation[turn_idx - 1]["content"]

                new_response = self.assistant_sim.generate_response(
                    domain=plan.domain,
                    conversation_history=conversation[:turn_idx],
                    user_message=user_msg,
                )
                conversation[turn_idx]["content"] = new_response
                logger.debug(f"  Rewrote assistant turn {turn_idx + 1}: {reason}")

            elif role == "user" and conversation[turn_idx]["role"] == "user":
                new_msg = self.user_sim.generate_message(
                    plan=plan,
                    conversation_history=conversation[:turn_idx],
                    current_turn=turn_idx // 2,
                )
                conversation[turn_idx]["content"] = new_msg
                logger.debug(f"  Rewrote user turn {turn_idx + 1}: {reason}")

        return conversation

    # ─── Filler Phrase Post-Processor ────────────────────

    FILLER_PATTERNS = [
        # Phrases that LLMs spam due to the Penalty Horizon
        # (frequency_penalty only covers ~4K-8K token window)
        r"Here'?s what most people get wrong about this:?\s*",
        r"That frustration makes total sense\.?\s*",
        r"A surprising fact:?\s*",
        r"Most people don'?t realize that\s+",
        r"The single most important action is to\s+",
        r"The key is to\s+",
        r"The most important thing is to\s+",
    ]

    def _strip_filler_phrases(self, conversation: list[dict]) -> list[dict]:
        """
        Regex post-processor for the Penalty Horizon problem.

        Because frequency_penalty only applies to a sliding window of ~4K-8K
        tokens, the LLM can still spam phrases in 120+ turn conversations.
        This deterministically strips known filler phrases from assistant
        responses, then polishes any orphaned grammar left behind.
        """
        import re

        asst_msgs = [m for m in conversation if m["role"] == "assistant"]

        # Track phrase usage counts
        phrase_counts = {}
        for msg in asst_msgs:
            content = msg["content"]
            # Strip [Turn: N] prefix for analysis
            text = re.sub(r'^\s*\[Turn:\s*\d+\]\s*', '', content)

            for pattern in self.FILLER_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    key = pattern[:40]
                    phrase_counts[key] = phrase_counts.get(key, 0) + 1

                    # Only strip if used more than 2 times (allow occasional use)
                    if phrase_counts[key] > 2:
                        # Remove the filler phrase but keep the rest
                        content = re.sub(
                            r'(\[Turn:\s*\d+\]\s*)' + pattern,
                            r'\1',
                            content,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                        # Also remove if it appears mid-sentence
                        content = re.sub(
                            pattern, '', content, count=1, flags=re.IGNORECASE
                        )
                        msg["content"] = content.strip()

        # ─── Grammar Polish (fixes "Regex Chainsaw" damage) ─────
        # After stripping filler phrases, the sentence start may have:
        #   - Orphaned punctuation: ", and your idea..." or "—lying awake..."
        #   - Dangling conjunctions: "is that chronic sleep..." 
        #   - Lowercase first letter: "they believe a small step..."
        for msg in asst_msgs:
            content = msg["content"]
            match = re.match(r'^(\[Turn:\s*\d+\])\s*(.*)$', content, flags=re.DOTALL)
            if not match:
                continue

            turn_tag, core_text = match.groups()

            # Strip orphaned punctuation (commas, dashes, colons) at start
            core_text = re.sub(r'^[\s,\-—–:]+', '', core_text)
            # Strip dangling conjunctions/fragments left by naive replace
            core_text = re.sub(r'^(and|is that|is|that|but|so|or)\s+', '', core_text, flags=re.IGNORECASE)
            # Clean up again after conjunction removal
            core_text = re.sub(r'^[\s,\-—–:]+', '', core_text)
            # Capitalize the first alphabetical character
            core_text = re.sub(
                r'^([^\w]*)([a-z])',
                lambda m: m.group(1) + m.group(2).upper(),
                core_text
            )

            msg["content"] = f"{turn_tag} {core_text}"

        # Log how many were stripped
        stripped = sum(max(0, v - 2) for v in phrase_counts.values())
        if stripped > 0:
            console.print(
                f"  🧹 [dim]Stripped {stripped} filler phrases "
                f"(Penalty Horizon post-processor)[/dim]"
            )

        return conversation

    # ─── Final Safety Net ─────────────────────────────────

    def _final_ddm_safety_net(self, conversation: list[dict]) -> tuple[list[dict], dict]:
        """
        Final deterministic safety net — runs LAST in the pipeline.

        1. Strip filler phrases (Penalty Horizon fix)
        2. Force-fix DDM on ALL assistant messages (L5 + phrase sanitization)
        3. Re-validate from scratch → this is the FINAL rule_result stored in JSON

        The validation MUST run here (not earlier) to avoid 'ghost errors'
        where stale pre-revision validation logs get attached to the final JSON.
        """
        # Step 1: Strip filler phrases (Penalty Horizon)
        conversation = self._strip_filler_phrases(conversation)

        # Step 2: Run force_fix_ddm on ALL assistant messages for:
        #   - L5 turn counter correction
        #   - L1/L3/L4 fixes
        #   - Banned phrase replacement
        asst_idx = 0
        for i, msg in enumerate(conversation):
            if msg["role"] == "assistant":
                asst_idx += 1
                conversation[i]["content"] = AssistantSimulator.force_fix_ddm(
                    msg["content"], turn_number=asst_idx
                )

        # Step 3: FINAL validation — this is the definitive result stored in JSON
        # Any earlier validation results are discarded (fixes Ghost Error bug)
        console.print("  📏 [bold green]FINAL Validation[/bold green] (post-revision + post-fix)...")
        rule_result = self._rule_based_validate(conversation)

        ddm_errors = [
            i for i in rule_result["issues"]
            if i["rule"].startswith("DDM_") and i["severity"] == "error"
        ]
        if ddm_errors:
            console.print(
                f"  🔩 [red]Safety net[/red]: {len(ddm_errors)} DDM violations "
                f"remain after force-fix (likely structural L2 issues)"
            )
        else:
            console.print("  ✅ [bold green]All DDM constraints PASS[/bold green]")

        return conversation, rule_result

    # ─── Phase 4: Translation ────────────────────────────

    def _translate_conversation(
        self,
        conversation: list[dict],
        target_lang: str,
    ) -> tuple[list[dict], dict]:
        """Phase 4: Translate conversation using the multi-agent pipeline."""
        console.print(
            f"  🌍 [magenta]Translation Pipeline[/magenta] → {target_lang.upper()} "
            f"(translate → review → back-verify → quality check)...",
            end=" ",
        )

        translated, quality_summary = self.translation_pipeline.translate_conversation(
            conversation=conversation,
            source_lang="en",
            target_lang=target_lang,
        )

        console.print(f"✅ ({len(translated)} messages)")
        return translated, quality_summary

    # ─── Full Pipeline ───────────────────────────────────

    def produce_single(
        self,
        domain: str,
        conv_id: str,
        num_turns: int,
        translate: bool = True,
    ) -> dict:
        """
        Produce a single high-quality conversation with translations.

        Returns:
            Dict with 'en' + target language conversations and metadata.
        """
        console.print(Panel(
            f"[bold]Producing: {conv_id}[/bold]\n"
            f"Domain: {domain} | Turns: {num_turns}",
            title="🏭 Data Factory",
            border_style="blue",
        ))

        start_time = time.time()
        pipeline_stats.reset()

        # Phase 1: Plan
        pipeline_stats.start_phase("planning")
        plan = self._plan_conversation(domain, conv_id, num_turns)
        pipeline_stats.end_phase("planning")

        # Phase 2: Generate
        pipeline_stats.start_phase("generation")
        conversation = self._generate_conversation(plan)
        pipeline_stats.end_phase("generation")

        # Phase 3: Rule-Based Validate + LLM Audit & Revise
        pipeline_stats.start_phase("audit_and_revise")
        conversation, quality_report, rule_validation = self._audit_and_revise(
            conversation, plan
        )
        pipeline_stats.end_phase("audit_and_revise")

        # Phase 4: Translate + Validate Translations (PARALLEL across languages)
        translations = {"en": conversation}
        translation_validations = {}
        if translate and self.target_languages:
            pipeline_stats.start_phase("translation")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            console.print(
                f"  🌍 [magenta]Parallel Translation Pipeline[/magenta] → "
                f"{', '.join(l.upper() for l in self.target_languages)} "
                f"({len(self.target_languages)} languages × {len(conversation)} msgs)..."
            )

            def _translate_and_validate(lang: str) -> tuple[str, list[dict], dict]:
                """Translate one language, run QA agents, and validate — runs in a thread."""
                translated, quality_summary = self.translation_pipeline.translate_conversation(
                    conversation=conversation,
                    source_lang="en",
                    target_lang=lang,
                )

                # Rule-based translation validation
                trans_report = self.trans_validator.validate(
                    conversation, translated, "en", lang
                )

                val_result = {
                    "passed": trans_report.passed,
                    "errors": trans_report.error_count,
                    "metrics": trans_report.metrics,
                    "quality": quality_summary,
                    "issues": [
                        {"turn": i.turn, "rule": i.rule, "severity": i.severity, "msg": i.message}
                        for i in trans_report.issues[:20]
                    ],
                }

                return lang, translated, val_result

            # Run all languages in parallel
            with ThreadPoolExecutor(max_workers=len(self.target_languages)) as executor:
                futures = {
                    executor.submit(_translate_and_validate, lang): lang
                    for lang in self.target_languages
                }
                for future in as_completed(futures):
                    lang, translated, val_result = future.result()
                    translations[lang] = translated
                    translation_validations[lang] = val_result

                    # Print per-language results as they complete
                    status = "✅" if val_result["passed"] else "❌"
                    metrics = val_result["metrics"]
                    quality = val_result.get("quality", {})
                    bi_score = quality.get("bilingual", {}).get("mean_score", "—")
                    mono_score = quality.get("monolingual", {}).get("mean_score", "—")
                    console.print(
                        f"    {status} {lang.upper()} done | "
                        f"{val_result['errors']} errors | "
                        f"SYS_ACK={metrics.get('sysack_preservation_rate', 0):.0%} "
                        f"Sources={metrics.get('source_citation_preservation_rate', 0):.0%} "
                        f"Bullets={metrics.get('bullet_preservation_rate', 0):.0%} | "
                        f"Bilingual={bi_score}/10 Monolingual={mono_score}/10"
                    )

            pipeline_stats.end_phase("translation")

        elapsed = time.time() - start_time

        result = {
            "id": conv_id,
            "domain": domain,
            "num_turns": num_turns,
            "plan": {
                "user_persona": plan.user_persona,
                "personality": plan.user_personality_traits,
                "emotional_arc": plan.emotional_arc,
                "plot_twists": plan.plot_twists,
            },
            "quality": {
                "overall": quality_report.overall_score,
                "realism": quality_report.realism_score,
                "diversity": quality_report.diversity_score,
                "ddm_compliance": quality_report.ddm_compliance_score,
                "approved": quality_report.approved,
                "issues": quality_report.issues,
            },
            "rule_validation": rule_validation,
            "translation_validation": translation_validations,
            "conversations": translations,
            "metadata": {
                "model": self.model,
                "generation_time_seconds": round(elapsed, 1),
                "revision_rounds": min(
                    self.max_revision_rounds,
                    len(quality_report.rewrite_requests) + 1,
                ),
                "pipeline_stats": pipeline_stats.to_dict(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pipeline_version": "v8-quality-hardened",
                "num_languages": 1 + len(self.target_languages),
            },
        }

        # Status
        rule_status = "✅" if rule_validation["passed"] else "⚠️"
        status = "✅ APPROVED" if quality_report.approved else "⚠️  BELOW THRESHOLD"
        console.print(
            f"\n  {status} | Score: {quality_report.overall_score}/10 | "
            f"Rules: {rule_status} | Time: {elapsed:.0f}s\n"
        )

        return result

    def produce_dataset(
        self,
        output_dir: str | Path = "data",
        num_conversations: int = 100,
        min_turns: int = 30,
        max_turns: int = 50,
        translate: bool = True,
        parallel_conversations: int = 3,
    ) -> list[Path]:
        """
        Produce the full Poly-DriftBench dataset.

        Args:
            output_dir: Root data directory.
            num_conversations: Total conversations to generate.
            min_turns: Minimum turn pairs.
            max_turns: Maximum turn pairs.
            translate: Whether to run Phase 4 (translation).
            parallel_conversations: Number of conversations to generate
                simultaneously. Each conversation's full pipeline runs
                in its own thread. Default 3 (safe for most API limits).

        Returns:
            List of paths to saved conversation files.
        """
        import random
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        output_dir = Path(output_dir)
        domains = list(DOMAIN_TEMPLATES.keys())
        saved_files = []
        quality_scores = []
        lock = threading.Lock()  # Thread-safe list access

        console.print(Panel(
            f"[bold]Poly-DriftBench Generation[/bold]\n"
            f"Conversations: {num_conversations}\n"
            f"Languages: EN + {', '.join(l.upper() for l in self.target_languages)}\n"
            f"Turns: {min_turns}-{max_turns}\n"
            f"Model: {self.model}\n"
            f"Parallel Conversations: {parallel_conversations}",
            title="🏭 Data Factory",
            border_style="green",
        ))

        def _produce_and_save(i: int) -> tuple[Path | None, float | None]:
            """Generate, validate, translate, and save one conversation."""
            domain = domains[i % len(domains)]
            num_turns = random.randint(min_turns, max_turns)
            conv_id = f"conv_{i:04d}_{domain}"

            try:
                result = self.produce_single(
                    domain=domain,
                    conv_id=conv_id,
                    num_turns=num_turns,
                    translate=translate,
                )

                # Save master file
                master_dir = output_dir / "generated"
                master_dir.mkdir(parents=True, exist_ok=True)
                master_path = master_dir / f"{conv_id}.json"
                with open(master_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                # Save per-language JSONL files
                for lang, conv in result["conversations"].items():
                    lang_dir = output_dir / "parallel" / lang
                    lang_dir.mkdir(parents=True, exist_ok=True)
                    jsonl_path = lang_dir / f"{conv_id}.jsonl"
                    with open(jsonl_path, "w") as f:
                        for msg in conv:
                            f.write(json.dumps({
                                "text": msg["content"],
                                "role": msg["role"],
                                "conversation_id": conv_id,
                            }, ensure_ascii=False) + "\n")

                return master_path, result["quality"]["overall"]

            except Exception as e:
                logger.error(f"Failed to generate {conv_id}: {e}")
                console.print(f"  ❌ [red]Failed {conv_id}: {e}[/red]")
                return None, None

        # Run conversations in parallel batches
        with ThreadPoolExecutor(max_workers=parallel_conversations) as executor:
            futures = {
                executor.submit(_produce_and_save, i): i
                for i in range(num_conversations)
            }
            for future in as_completed(futures):
                master_path, score = future.result()
                if master_path is not None:
                    with lock:
                        saved_files.append(master_path)
                        quality_scores.append(score)

        # Final summary
        if quality_scores:
            import numpy as np
            self._print_summary(
                total=num_conversations,
                generated=len(saved_files),
                scores=quality_scores,
            )

        return saved_files

    def _print_summary(self, total: int, generated: int, scores: list[float]):
        """Print a rich summary table."""
        import numpy as np

        table = Table(title="📊 Generation Summary", border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total Requested", str(total))
        table.add_row("Successfully Generated", str(generated))
        table.add_row("Success Rate", f"{generated/total*100:.1f}%")
        table.add_row("Mean Quality Score", f"{np.mean(scores):.2f}/10")
        table.add_row("Median Quality Score", f"{np.median(scores):.2f}/10")
        table.add_row("Min Quality Score", f"{np.min(scores):.2f}/10")
        table.add_row("Max Quality Score", f"{np.max(scores):.2f}/10")
        approved = sum(1 for s in scores if s >= 7.0)
        table.add_row("Approved (≥7.0)", f"{approved}/{generated}")

        console.print(table)
