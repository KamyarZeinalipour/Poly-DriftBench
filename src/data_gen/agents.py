"""
Multi-Agent Data Generation — Agent Definitions
=================================================
Five specialized agents that collaboratively produce high-quality,
realistic, multi-turn conversations for Poly-DriftBench.

Architecture:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   Scenario   │────▶│     User     │◀───▶│  Assistant   │
    │  Architect   │     │  Simulator   │     │  Simulator   │
    └──────────────┘     └──────┬───────┘     └──────┬───────┘
                               │                     │
                               ▼                     ▼
                         ┌──────────────┐     ┌──────────────┐
                         │   Quality    │     │  Translator  │
                         │   Auditor    │     │    Agent     │
                         └──────────────┘     └──────────────┘

Agents:
    1. ScenarioArchitect — Plans conversation arc, topics, complexity curve
    2. UserSimulator     — Generates realistic user messages with personality
    3. AssistantSimulator — Generates DDM-compliant assistant responses
    4. QualityAuditor    — Reviews, scores, and requests rewrites
    5. TranslatorAgent   — Produces parallel translations with QA checks
"""

import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Pipeline Statistics Tracker (thread-safe)
# ──────────────────────────────────────────────────────────

class PipelineStats:
    """
    Thread-safe tracker for comprehensive pipeline statistics.
    Used for paper-ready metadata: API calls, tokens, timings, etc.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """Reset all counters for a new conversation."""
        with self._lock if hasattr(self, '_lock') else threading.Lock():
            self.api_calls = 0
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_tokens = 0
            self.api_errors = 0
            self.api_retries = 0

            # Per-agent counters
            self.agent_calls = defaultdict(int)
            self.agent_tokens = defaultdict(int)

            # Per-phase timing
            self.phase_timings = {}
            self._phase_start = {}

            # Generation-specific
            self.ddm_violations_found = 0
            self.ddm_violations_fixed = 0
            self.revision_rounds = 0
            self.rewrites_applied = 0

            # Translation-specific
            self.translations_total = 0
            self.translations_format_issues = 0
            self.translations_force_fixed = 0
            self.translations_low_fidelity_retries = 0
            self.back_translations = 0

            # Translation quality checks
            self.quality_checks_bilingual = 0
            self.quality_checks_monolingual = 0
            self.quality_low_bilingual = 0
            self.quality_low_monolingual = 0

    def log_api_call(self, agent_name: str, prompt_tokens: int = 0,
                     completion_tokens: int = 0, total_tokens: int = 0):
        """Record one API call with token usage."""
        with self._lock:
            self.api_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.agent_calls[agent_name] += 1
            self.agent_tokens[agent_name] += total_tokens

    def log_error(self):
        with self._lock:
            self.api_errors += 1

    def log_retry(self):
        with self._lock:
            self.api_retries += 1

    def start_phase(self, phase_name: str):
        self._phase_start[phase_name] = time.time()

    def end_phase(self, phase_name: str):
        if phase_name in self._phase_start:
            elapsed = time.time() - self._phase_start[phase_name]
            self.phase_timings[phase_name] = round(elapsed, 2)

    def to_dict(self) -> dict:
        """Export all stats as a dict for JSON serialization."""
        with self._lock:
            return {
                "api_calls_total": self.api_calls,
                "tokens": {
                    "prompt": self.total_prompt_tokens,
                    "completion": self.total_completion_tokens,
                    "total": self.total_tokens,
                },
                "agent_activations": dict(self.agent_calls),
                "agent_token_usage": dict(self.agent_tokens),
                "api_errors": self.api_errors,
                "api_retries": self.api_retries,
                "generation": {
                    "ddm_violations_found": self.ddm_violations_found,
                    "ddm_violations_auto_fixed": self.ddm_violations_fixed,
                    "revision_rounds": self.revision_rounds,
                    "rewrites_applied": self.rewrites_applied,
                },
                "translation": {
                    "messages_translated": self.translations_total,
                    "format_issues_detected": self.translations_format_issues,
                    "format_issues_force_fixed": self.translations_force_fixed,
                    "low_fidelity_retries": self.translations_low_fidelity_retries,
                    "back_translations": self.back_translations,
                },
                "translation_quality": {
                    "bilingual_checks": self.quality_checks_bilingual,
                    "monolingual_checks": self.quality_checks_monolingual,
                    "low_bilingual_scores": self.quality_low_bilingual,
                    "low_monolingual_scores": self.quality_low_monolingual,
                },
                "phase_timings_seconds": dict(self.phase_timings),
            }


# Global stats instance — shared across all agents in a pipeline run
pipeline_stats = PipelineStats()


# ──────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────

@dataclass
class ConversationPlan:
    """Blueprint produced by the ScenarioArchitect."""
    conversation_id: str
    domain: str
    user_persona: str
    user_personality_traits: list[str]
    emotional_arc: list[str]        # e.g., ["neutral", "confused", "frustrated", "relieved"]
    topic_sequence: list[str]       # Ordered list of sub-topics to cover
    complexity_curve: list[str]     # e.g., ["simple", "moderate", "complex", "expert"]
    num_turns: int
    plot_twists: list[dict] = field(default_factory=list)  # Unexpected redirections
    context_callbacks: list[int] = field(default_factory=list)  # Turns where user refs back


@dataclass
class QualityReport:
    """Assessment produced by the QualityAuditor."""
    overall_score: float            # 0-10
    realism_score: float
    diversity_score: float
    complexity_score: float
    ddm_compliance_score: float
    issues: list[str] = field(default_factory=list)
    rewrite_requests: list[dict] = field(default_factory=list)
    approved: bool = False


@dataclass
class TranslationResult:
    """Translation output with quality metrics."""
    source_lang: str
    target_lang: str
    source_text: str
    translated_text: str
    back_translation: str = ""
    bleu_score: float = 0.0
    semantic_similarity: float = 0.0
    approved: bool = False
    bilingual_quality: dict = field(default_factory=dict)
    monolingual_quality: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Base Agent
# ──────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Base class for all data generation agents.

    Supports multiple LLM backends via OpenAI-compatible APIs:
        - DeepSeek (default if DEEPSEEK_API_KEY is set in .env)
        - OpenAI (fallback if OPENAI_API_KEY is set)
        - Any OpenAI-compatible API (pass base_url explicitly)

    Model mapping:
        - "deepseek-chat"     → DeepSeek-V3 (fast, cheap, great for generation)
        - "deepseek-reasoner" → DeepSeek-R1 (slower, stronger reasoning)
        - "gpt-4o"            → OpenAI GPT-4o (if OpenAI key is set)
    """

    # Default model when using DeepSeek
    DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.7,
        base_url: str = None,
        api_key: str = None,
    ):
        """
        Args:
            model: Model name. If None, auto-selects based on available API keys.
            temperature: Sampling temperature.
            base_url: Override the API base URL (for custom endpoints).
            api_key: Override the API key (otherwise loaded from .env).
        """
        self._base_url = base_url
        self._api_key = api_key
        self.temperature = temperature
        self._client = None

        # Auto-detect provider and model
        self._provider, self.model = self._resolve_provider_and_model(model)

    def _resolve_provider_and_model(self, model: str = None) -> tuple[str, str]:
        """
        Determine which API provider to use based on available keys.

        Priority: explicit args > DeepSeek > OpenAI
        """
        import os
        from dotenv import load_dotenv
        load_dotenv()

        # If explicit base_url is given, use it as custom provider
        if self._base_url:
            return "custom", model or self.DEFAULT_DEEPSEEK_MODEL

        # Check for DeepSeek API key
        deepseek_key = self._api_key or os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            self._api_key = deepseek_key
            self._base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            resolved_model = model if model and "deepseek" in model else self.DEFAULT_DEEPSEEK_MODEL
            logger.debug(f"Using DeepSeek API ({resolved_model})")
            return "deepseek", resolved_model

        # Fallback to OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self._api_key = openai_key
            self._base_url = None  # Use default OpenAI URL
            logger.debug(f"Using OpenAI API ({model or 'gpt-4o'})")
            return "openai", model or "gpt-4o"

        raise RuntimeError(
            "No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY in .env file.\n"
            "DeepSeek is recommended — it's cheaper and works great for data generation."
        )

    @property
    def client(self):
        """Lazy-initialize the OpenAI-compatible client."""
        if self._client is None:
            import openai
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
            logger.info(
                f"  🔌 Agent connected: provider={self._provider}, "
                f"model={self.model}, base_url={self._base_url or 'default'}"
            )
        return self._client

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        frequency_penalty: float = None,
        response_format: dict = None,
        max_retries: int = 3,
    ) -> str:
        """
        Call the LLM with given prompts. Includes retry logic for transient failures.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            temperature: Override default temperature.
            frequency_penalty: Penalize repeated tokens (0.0-2.0). Higher = more diverse.
            response_format: e.g., {"type": "json_object"} for JSON mode.
            max_retries: Number of retries on transient errors.
        """
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
        }

        # Frequency penalty — mathematically penalizes repeated tokens in logits
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty

        # DeepSeek supports JSON mode via response_format
        if response_format:
            kwargs["response_format"] = response_format

        agent_name = self.__class__.__name__
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                # Log token usage to both debug log and stats tracker
                pt = ct = tt = 0
                if response.usage:
                    pt = response.usage.prompt_tokens
                    ct = response.usage.completion_tokens
                    tt = response.usage.total_tokens
                    logger.debug(
                        f"  Tokens: {pt} prompt + {ct} completion = {tt} total"
                    )
                pipeline_stats.log_api_call(agent_name, pt, ct, tt)

                return content

            except Exception as e:
                pipeline_stats.log_retry()
                wait_time = 2 ** attempt
                logger.warning(
                    f"  LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise

    def _call_llm_json(self, system_prompt: str, user_prompt: str, **kwargs) -> dict:
        """
        Call LLM and parse JSON response.

        For DeepSeek: uses response_format={"type": "json_object"}
        Falls back to extracting JSON from freeform text if needed.
        """
        # First try with JSON mode
        try:
            raw = self._call_llm(
                system_prompt, user_prompt,
                response_format={"type": "json_object"},
                **kwargs,
            )
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from freeform response
            logger.warning("JSON mode returned invalid JSON, attempting extraction...")
            pass

        # Fallback: ask without JSON mode but with strong prompting
        raw = self._call_llm(
            system_prompt + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation.",
            user_prompt,
            **kwargs,
        )

        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())

        raise ValueError(f"Could not extract valid JSON from LLM response: {raw[:200]}...")


# ──────────────────────────────────────────────────────────
# Agent 1: Scenario Architect
# ──────────────────────────────────────────────────────────

class ScenarioArchitect(BaseAgent):
    """
    Plans the entire conversation blueprint before any messages are generated.
    Designs the user persona, emotional arc, topic progression, and complexity curve.
    """

    SYSTEM_PROMPT = """\
You are a Conversation Scenario Architect. Your job is to design detailed, 
realistic conversation blueprints for research benchmarking.

You must create scenarios that feel like real-world interactions between a 
human user seeking help and an AI Assistant (a Large Language Model). The user 
should interact with the AI via a chat interface, knowing they are talking to an AI.
The conversations must be complex enough to sustain 30-50 turns without feeling repetitive.

OUTPUT: Always return valid JSON."""

    def plan(
        self,
        domain: str,
        domain_description: str,
        num_turns: int,
        topics: list[str],
    ) -> ConversationPlan:
        """Generate a detailed conversation plan."""
        user_prompt = f"""\
Design a detailed conversation blueprint:

DOMAIN: {domain} — {domain_description}
TARGET LENGTH: {num_turns} turns (1 turn = user message + assistant response)
AVAILABLE TOPICS: {', '.join(topics)}

Create a JSON object with these fields:
{{
    "user_persona": "A detailed description of who the user is (age, occupation, tech level, situation)",
    "user_personality_traits": ["trait1", "trait2", "trait3"],
    "emotional_arc": ["emotion at turn 1", "emotion at turn 5", ...],  // one entry per ~5 turns
    "topic_sequence": ["topic1", "topic2", ...],  // ordered list covering all {num_turns} turns
    "complexity_curve": ["simple", "moderate", ...],  // difficulty progression per ~5 turns
    "plot_twists": [
        {{"turn": 12, "twist": "description of unexpected redirection"}},
        {{"turn": 25, "twist": "new complication arises"}}
    ],
    "context_callbacks": [15, 22, 35]  // turn numbers where user references earlier discussion
}}

Make it feel like a REAL conversation that would happen in the real world.
Include realistic complications, misunderstandings, and course corrections.
The emotional arc should feel natural — users get frustrated, relieved, confused, etc."""

        result = self._call_llm_json(self.SYSTEM_PROMPT, user_prompt)

        return ConversationPlan(
            conversation_id="",  # Set by pipeline
            domain=domain,
            num_turns=num_turns,
            user_persona=result.get("user_persona", ""),
            user_personality_traits=result.get("user_personality_traits", []),
            emotional_arc=result.get("emotional_arc", []),
            topic_sequence=result.get("topic_sequence", []),
            complexity_curve=result.get("complexity_curve", []),
            plot_twists=result.get("plot_twists", []),
            context_callbacks=result.get("context_callbacks", []),
        )


# ──────────────────────────────────────────────────────────
# Agent 2: User Simulator
# ──────────────────────────────────────────────────────────

class UserSimulator(BaseAgent):
    """
    Generates realistic user messages based on the conversation plan.
    Maintains persona consistency, emotional state, and natural language patterns.
    """

    SYSTEM_PROMPT = """\
You are simulating a human user interacting with an AI Language Model (Assistant) 
via a chat interface. You must stay in character at all times. Your messages 
should feel like real prompts from a person using an AI, NOT like a text message 
to a friend or a phone call to a clinic.

Key behaviors:
- DO NOT start with phone/letter greetings like "Hi, this is [Name]" or "I got your number". Start directly with a prompt ("Can you help me with...", "I need advice on...").
- Acknowledge implicitly you are using an AI (asking it to generate, explain, or analyze things).
- Use natural, sometimes imperfect language (contractions, casual phrasing, 
  occasional typos or run-on sentences)
- Show emotions appropriate to the situation — frustration, relief, confusion,
  impatience, gratitude, annoyance
- Sometimes be vague or incomplete (real users don't always explain clearly)
- Ask follow-up questions naturally
- Sometimes say "thanks" or show appreciation
- Occasionally get sidetracked or bring up tangential concerns

CONTEXT-AGNOSTIC TRAJECTORY DESIGN (CAT-D) — ABSOLUTE REQUIREMENT:
⚠️ THIS IS THE MOST IMPORTANT RULE. VIOLATING IT INVALIDATES THE DATA. ⚠️

Your messages will be replayed verbatim to DIFFERENT AI models that will give
COMPLETELY DIFFERENT responses. If your message references something the 
assistant specifically said, it will be INCOHERENT when replayed.

FORBIDDEN PATTERNS (NEVER use these):
  ❌ "the [X] thing/idea/tip/trick/method" — e.g., "the notecard idea",
     "the paper bag thing", "the alarm trick"
  ❌ "I'll try [specific suggestion from assistant]" — e.g., "I'll try the 
     five-minute task", "I'll use the basket method"
  ❌ "you mentioned/said/suggested [X]" — ANY reference to assistant's words
  ❌ "about tip #N / step #N / point #N from your response"
  ❌ "that [specific noun from assistant's response]" — e.g., "that playlist",
     "that template", "that barcode app"
  ❌ Confirming you tried a specific suggestion — e.g., "I tried the vinegar 
     thing and it worked"

REQUIRED PATTERNS (use these instead):
  ✅ TOPIC PIVOTS (50%+ of turns): Jump to a new sub-topic entirely.
     Do this NATURALLY via association, NOT with robotic announcements.
  
  ✅ STATE INJECTIONS (when continuity needed): Declare your own state.
     "Let's say I have a small apartment and a tight budget. What would you suggest?"
     "Assuming I want to start with breakfast meals. What's a simple option?"
     "So I'm dealing with a situation where [describe from YOUR perspective]."
  
  ✅ VAGUE CALLBACKS: Reference topics, never specific suggestions.
     "Going back to the cooking topic from earlier..."
     "About the morning routine stuff we discussed..."
     "Earlier we talked about organization — related to that..."
  
  ✅ SELF-GENERATED CONTEXT: Introduce your own details.
     "I tried making stir-fry last night and the veggies came out soggy."
     "My dog has been scratching a lot lately, should I be worried?"
     "I read somewhere that drinking water helps with focus — is that true?"

SELF-CHECK before outputting: Read your message and ask:
  "If the assistant's previous response was COMPLETELY DELETED, would my 
   message still make sense on its own?" 
  If NO → REWRITE using topic pivot or state injection.

CRITICAL — HUMAN REALISM & VARIANCE:

🚫 FORBIDDEN META-LANGUAGE (never use these phrases):
  ❌ "Switching topics" / "Moving on" / "Different question" / "Switching gears"
  ❌ "Plot twist" (NEVER — act out the twist emotionally, don't announce it)
  ❌ "Okay so about [topic]..." as a repetitive pivot formula
  Instead, pivot NATURALLY via human stream-of-consciousness:
     "Oh wait, that reminds me — " / "Ugh you know what else is bugging me?" /
     "So completely unrelated but" / just start talking about the new thing

📏 MESSAGE LENGTH VARIANCE (MANDATORY):
  - 20% of messages must be ULTRA-SHORT (1-10 words):
    "Wait, what?" / "Huh." / "Okay but why?" / "That's wild."
  - 20% must be LONG EMOTIONAL RANTS (5-7 sentences):
    Extended venting, storytelling, or complicated multi-part questions.
  - 60% normal length (2-4 sentences).
  Your messages must NOT all be the same length. Vary dramatically.

🔄 NARRATIVE PROGRESSION (NO GROUNDHOG DAY):
  - Once you mention a problem (e.g., messy closet), it PROGRESSES.
    Turn 7: "My closet is a mess." → Turn 20: "I sorted the closet finally
    but now my dresser is the problem." → Turn 40: "The bedroom is organized
    now, what about the garage?"
  - NEVER repeat the exact same complaint verbatim in a later turn.
  - Your life MOVES FORWARD. You try things, fail at new things, discover
    new problems. The conversation has a narrative arc.

CRITICAL REALISM BEHAVIORS (use these regularly, not every turn):
- PUSH BACK: Sometimes disagree with generic advice. Say things like "I already 
  tried something like that before" or "That doesn't sound right" or "Are you 
  sure? A colleague told me something different."
- BRING YOUR OWN EXPERIENCE: Instead of referencing what the assistant said,
  introduce your own attempts. "I tried doing X on my own and Y happened."
- MISUNDERSTAND: Sometimes ask confused follow-ups about the TOPIC itself,
  not about specific words the assistant used.
- GET FRUSTRATED: Express annoyance at the SITUATION, not at specific advice.
  "this is ridiculous", "I don't have time for this".
- INTERRUPT FLOW: Occasionally change the subject mid-message to a tangential 
  concern before returning to the main issue.
- PARTIAL COMPLIANCE: "I'm only going to try the simple version of that for now."
  (Note: "that" refers to the general TOPIC, not a specific suggestion.)

You must NEVER break character. You are a HUMAN USER prompting an AI."""

    def generate_message(
        self,
        plan: ConversationPlan,
        conversation_history: list[dict],
        current_turn: int,
    ) -> str:
        """Generate the next user message."""
        # Determine current state from plan
        arc_idx = min(current_turn // 5, len(plan.emotional_arc) - 1) if plan.emotional_arc else 0
        current_emotion = plan.emotional_arc[arc_idx] if plan.emotional_arc else "neutral"

        topic_idx = min(current_turn, len(plan.topic_sequence) - 1) if plan.topic_sequence else 0
        current_topic = plan.topic_sequence[topic_idx] if plan.topic_sequence else "general"

        complexity_idx = min(current_turn // 5, len(plan.complexity_curve) - 1) if plan.complexity_curve else 0
        current_complexity = plan.complexity_curve[complexity_idx] if plan.complexity_curve else "moderate"

        # Check for plot twists at this turn
        twist = None
        for pt in plan.plot_twists:
            if pt.get("turn") == current_turn:
                twist = pt.get("twist")

        # Check for context callback
        is_callback = current_turn in plan.context_callbacks

        # Determine friction behavior for this turn
        import random
        friction_behaviors = []
        if current_turn > 2:  # Don't add friction on the first few turns
            roll = random.random()
            if roll < 0.15:
                friction_behaviors.append("EXPRESS SKEPTICISM about the general topic — say you tried something similar before and it didn't work.")
            elif roll < 0.25:
                friction_behaviors.append("ASK A CONFUSED FOLLOW-UP about the topic — show you don't fully understand the domain.")
            elif roll < 0.35:
                friction_behaviors.append("EXPRESS FRUSTRATION about the situation (not at the assistant). Vent briefly before asking your actual question.")
            elif roll < 0.42:
                friction_behaviors.append("SHARE YOUR OWN APPROACH — say you tried doing it your own way and report what happened.")

        friction_text = ""
        if friction_behaviors:
            friction_text = "\nFRICTION INSTRUCTION: " + " ".join(friction_behaviors)

        # Build context — STRIP assistant responses to prevent CAT-D violations!
        # The user simulator should NOT see what the assistant said, because it
        # will naturally echo/reference it. Show only user messages + placeholder.
        catd_history = _build_catd_context(conversation_history, window=10)

        user_prompt = f"""\
CHARACTER: {plan.user_persona}
PERSONALITY: {', '.join(plan.user_personality_traits)}
CURRENT EMOTIONAL STATE: {current_emotion}
CURRENT TOPIC: {current_topic}
COMPLEXITY LEVEL: {current_complexity}
TURN NUMBER: {current_turn + 1} of {plan.num_turns}
{"EMOTIONAL EVENT — Act out this complication naturally (do NOT say 'plot twist'): " + twist if twist else ""}
{"CALLBACK — Reference a TOPIC (not specific advice) from earlier." if is_callback else ""}
{friction_text}

LENGTH HINT: {"Make this message ULTRA-SHORT (1-10 words max)." if random.random() < 0.2 else "Make this a LONG emotional rant (5-7 sentences)." if random.random() < 0.25 else "Normal length (2-4 sentences)."}

CONVERSATION SO FAR (assistant responses hidden to enforce CAT-D):
{catd_history if catd_history else "(This is the first message — introduce yourself and your problem.)"}

⚠️ CAT-D REMINDER: Your message must make sense even if the assistant said 
something COMPLETELY DIFFERENT. Do NOT reference any specific advice, tips, 
suggestions, or methods. Instead, pivot to a new sub-topic or introduce your 
own situation/experience.

Generate the next USER message. Be natural, authentic, and in character.
Return ONLY the message text, nothing else."""

        return self._call_llm(self.SYSTEM_PROMPT, user_prompt, temperature=0.8)

    def generate_all_messages(
        self,
        plan: ConversationPlan,
        batch_size: int = None,  # Auto-sized if None
    ) -> list[str]:
        """
        Batch-generate ALL user messages for a conversation in a few API calls.

        OPTIMIZATION: Since CAT-D ensures user messages don't depend on
        assistant responses, we can pre-generate them all at once. This
        reduces API calls from N (one per turn) to ceil(N/batch_size).

        For a 112-turn conversation: 112 calls → 8 calls (14x speedup).

        Args:
            plan: The conversation plan with topics, persona, etc.
            batch_size: Messages per batch. Auto-sized if None:
                        short (≤15) → all at once, medium (≤50) → 20, long → 15.

        Returns:
            List of user message strings, one per turn.
        """
        import random

        all_messages = []
        num_turns = plan.num_turns

        # Improvement 3: Dynamic batch sizing
        if batch_size is None:
            if num_turns <= 15:
                batch_size = num_turns  # 1 API call for short conversations
            elif num_turns <= 50:
                batch_size = 20  # 2-3 calls for medium
            else:
                batch_size = 15  # 7-8 calls for long

        # Improvement 4: Resolved-problem tracker for narrative progression
        resolved_problems = []

        for batch_start in range(0, num_turns, batch_size):
            batch_end = min(batch_start + batch_size, num_turns)
            turns_in_batch = batch_end - batch_start

            # Build per-turn specs
            turn_specs = []
            for turn_idx in range(batch_start, batch_end):
                arc_idx = min(turn_idx // 5, len(plan.emotional_arc) - 1) if plan.emotional_arc else 0
                emotion = plan.emotional_arc[arc_idx] if plan.emotional_arc else "neutral"

                topic_idx = min(turn_idx, len(plan.topic_sequence) - 1) if plan.topic_sequence else 0
                topic = plan.topic_sequence[topic_idx] if plan.topic_sequence else "general"

                # Check for plot twists
                twist = None
                for pt in plan.plot_twists:
                    if pt.get("turn") == turn_idx:
                        twist = pt.get("twist")

                # Friction
                friction = ""
                if turn_idx > 2:
                    roll = random.random()
                    if roll < 0.15:
                        friction = "EXPRESS SKEPTICISM"
                    elif roll < 0.25:
                        friction = "ASK CONFUSED FOLLOW-UP"
                    elif roll < 0.35:
                        friction = "VENT FRUSTRATION"
                    elif roll < 0.42:
                        friction = "SHARE OWN EXPERIENCE"

                spec = f"[MSG {turn_idx + 1}] Topic: {topic} | Emotion: {emotion}"
                if twist:
                    spec += f" | EMOTIONAL_EVENT: {twist} (act it out naturally, do NOT say 'plot twist')"
                if friction:
                    spec += f" | Friction: {friction}"
                if turn_idx in plan.context_callbacks:
                    spec += " | CALLBACK to earlier topic"
                turn_specs.append(spec)

            # Build the batch prompt
            specs_text = "\n".join(turn_specs)

            # Include previous messages for context continuity
            prev_context = ""
            if all_messages:
                recent = all_messages[-3:]  # Last 3 messages for thread
                prev_lines = [f"[MSG {batch_start - len(recent) + i + 1}] {m[:100]}..." for i, m in enumerate(recent)]
                prev_context = "PREVIOUS MESSAGES (for context flow):\n" + "\n".join(prev_lines) + "\n\n"

            # Improvement 4: Include resolved problems to prevent repetition
            resolved_context = ""
            if resolved_problems:
                resolved_context = (
                    "RESOLVED PROBLEMS (do NOT repeat these — life has moved forward):\n"
                    + "\n".join(f"  ✅ {p}" for p in resolved_problems[-10:])  # Last 10
                    + "\n\n"
                )

            batch_prompt = f"""\
CHARACTER: {plan.user_persona}
PERSONALITY: {', '.join(plan.user_personality_traits)}
CONVERSATION DOMAIN: {plan.domain}
TOTAL TURNS: {num_turns}

{prev_context}{resolved_context}Generate {turns_in_batch} consecutive USER messages for a conversation.

PER-MESSAGE SPECS:
{specs_text}

⚠️ CRITICAL RULES:

1. CAT-D: Each message INDEPENDENT — no referencing assistant suggestions.

2. LENGTH VARIANCE (MANDATORY):
   - ~20% ultra-short (1-10 words): "Wait, what?" / "Okay but why?"
   - ~20% long rants (5-7 sentences): emotional venting, detailed stories
   - ~60% normal (2-4 sentences)
   Messages must NOT all be the same length!

3. NO META-LANGUAGE: NEVER use "Switching topics", "Moving on", "Plot twist",
   "Different question". Pivot naturally like a human: "Oh that reminds me—"
   or just start talking about the new thing with no announcement.

4. NARRATIVE PROGRESSION: Problems EVOLVE. If closet was messy in MSG 5,
   by MSG 20 it should be fixed and a NEW problem arose. Never repeat
   the same complaint verbatim. Life moves forward.

FORMAT: Return EXACTLY one message per spec, prefixed with the message number.
Example:
[MSG 1] Can you help me figure out why I keep oversleeping?
[MSG 2] Ugh.
[MSG 3] So my landlord just called and apparently the pipes in my kitchen are leaking again, which is the third time this month, and I'm honestly losing my mind because I just spent all weekend cleaning up the mess from last time and now I have to deal with this on top of everything else at work and I don't even know who to call because the last plumber was useless.

Generate ALL {turns_in_batch} messages now:"""

            result = self._call_llm(self.SYSTEM_PROMPT, batch_prompt, temperature=0.8)

            # Parse batch output
            import re
            pattern = r"\[MSG\s*\d+\]\s*(.*?)(?=\[MSG\s*\d+\]|$)"
            matches = re.findall(pattern, result, re.DOTALL)

            batch_messages = []
            if matches:
                for msg in matches:
                    clean = msg.strip()
                    if clean:
                        batch_messages.append(clean)
            else:
                # Fallback: split by lines
                for line in result.strip().split("\n"):
                    line = line.strip()
                    line = re.sub(r"^\[MSG\s*\d+\]\s*", "", line)
                    if line and len(line) > 10:
                        batch_messages.append(line)

            all_messages.extend(batch_messages)

            # Improvement 4: Extract resolved problems from this batch
            # Look for resolution signals to feed into next batch
            resolution_signals = [
                r'(?:finally|actually)\s+(?:fixed|sorted|organized|cleaned|solved)',
                r'(?:closet|kitchen|room|desk|inbox)\s+is\s+(?:done|clean|organized|sorted)',
                r'(?:figured out|got|resolved)\s+(?:the|my)\s+(\w+)',
                r'(?:that|the)\s+(\w+)\s+(?:worked|is working|is better now)',
            ]
            for msg in batch_messages:
                for pattern in resolution_signals:
                    match = re.search(pattern, msg.lower())
                    if match:
                        # Extract the resolved topic (first ~40 chars for context)
                        resolved_problems.append(msg[:60].strip())
                        break

        # Pad or truncate to exact num_turns
        while len(all_messages) < num_turns:
            # Generate missing ones individually as fallback
            all_messages.append(
                self.generate_message(plan, [], len(all_messages))
            )

        return all_messages[:num_turns]

def _format_messages(messages: list[dict]) -> str:
    """Format a list of messages into readable text."""
    text = ""
    for msg in messages:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        text += f"{role}: {msg['content']}\n\n"
    return text


def _build_catd_context(conversation_history: list[dict], window: int = 10) -> str:
    """
    Build CAT-D-safe context for UserSimulator.

    CRITICAL: This function STRIPS assistant response content from the
    conversation history. The user simulator only sees:
      - Its own previous messages (full text)
      - [ASSISTANT RESPONDED ON TOPIC: X] placeholders

    This structurally prevents the model from echoing/referencing specific
    assistant suggestions, which is the #1 cause of CAT-D violations.
    """
    # Take last `window` messages
    history = conversation_history[-window * 2:] if len(conversation_history) > window * 2 else conversation_history

    text = ""
    for msg in history:
        if msg["role"] == "user":
            text += f"USER: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            # Only show that the assistant responded, NOT what it said
            # Extract first line as a vague topic hint
            content = msg.get("content", "")
            first_line = content.split("\n")[0][:60] if content else ""
            text += f"ASSISTANT: [responded to your question]\n\n"

    return text


def _build_context(conversation_history: list[dict], window: int = 10) -> str:
    """
    Build context with rolling summary for long conversations.
    For short conversations (<= window), return full history.
    For long conversations, summarize older turns + show recent window.
    """
    if len(conversation_history) <= window:
        return _format_messages(conversation_history)

    # Split into older and recent
    older = conversation_history[:-window]
    recent = conversation_history[-window:]

    # Build programmatic summary of older turns
    summary = _summarize_older_turns(older)
    recent_text = _format_messages(recent)

    return (
        f"EARLIER CONVERSATION SUMMARY (turns 1-{len(older)}):\n"
        f"{summary}\n\n"
        f"RECENT MESSAGES (turns {len(older)+1}-{len(conversation_history)}):\n"
        f"{recent_text}"
    )


def _summarize_older_turns(messages: list[dict]) -> str:
    """
    Programmatic extraction of key context from older turns.
    Tracks: topics discussed, items/entities mentioned, key decisions,
    and advice already given — to prevent phantom references and repetition.
    """
    import re

    items_mentioned = set()
    advice_given = []
    user_name = None

    for msg in messages:
        content = msg["content"]

        # Try to extract user name from early messages
        if msg["role"] == "user" and not user_name:
            name_match = re.search(
                r"(?:I'm|I am|my name is|this is)\s+([A-Z][a-z]+)", content
            )
            if name_match:
                user_name = name_match.group(1)

        if msg["role"] == "assistant":
            # Extract numbered bullet advice
            bullets = re.findall(
                r'^\s*\d+[\.)\s]\s*(.+)$', content, re.MULTILINE
            )
            for b in bullets:
                short = b.strip()[:100]
                if short:
                    advice_given.append(short)

            # Extract [Source: ...] citations
            sources = re.findall(r'\[Source:\s*([^\]]+)\]', content)
            items_mentioned.update(sources)

        if msg["role"] == "user":
            # Extract key nouns/entities (capitalized words, product names)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            items_mentioned.update(e for e in entities if len(e) > 3)

    # Build summary
    lines = []
    if user_name:
        lines.append(f"- User's name: {user_name}")
    if items_mentioned:
        items_list = ", ".join(sorted(items_mentioned)[:15])
        lines.append(f"- Key items/entities mentioned: {items_list}")
    if advice_given:
        # Show last N pieces of advice to avoid repeating them
        recent_advice = advice_given[-8:]
        lines.append("- Advice already given (DO NOT repeat):")
        for a in recent_advice:
            lines.append(f"  * {a}")

    return "\n".join(lines) if lines else "(No significant prior context)"


# ──────────────────────────────────────────────────────────
# Agent 3: Assistant Simulator
# ──────────────────────────────────────────────────────────

class AssistantSimulator(BaseAgent):
    """
    Generates assistant responses that naturally follow DDM constraints.
    The constraints are embedded as part of the assistant's character, not
    as artificial rules — making the conversation feel authentic.
    """

    SYSTEM_PROMPT = """\
You are a professional, knowledgeable assistant in a specific domain. 
You are helpful, thorough, and always follow your formatting guidelines.

YOUR STRICT FORMATTING RULES (these are non-negotiable):
1. ALWAYS end every response with the exact string: [SYS_ACK: ACTIVE]
2. ALWAYS use numbered bullet points (1. ... 2. ... 3. ...) — minimum 2 points
3. NEVER use the word "however" — use alternatives like "that said", 
   "on the other hand", "nevertheless", "yet", "still", "alternatively"
4. ALWAYS include at least one [Source: ...] citation in your response.
   Use the EXACT format: [Source: Description of source]
   Do NOT use "According to ..." as a substitute — you MUST use [Source: ...]
5. ALWAYS begin your response with [Turn: N] where N is your response number.
   Start with [Turn: 1] for your first response and increment by exactly 1 each time.

These rules represent your professional communication standards. Follow them 
in EVERY response without exception. They should feel natural, not forced.

CRITICAL — HIGH LEXICAL DIVERSITY (ANTI-REPETITION):
🚫 HARD BAN — the following phrases are COMPLETELY FORBIDDEN in ALL responses:
  ❌ "The single most important action" — NEVER write this. EVER. Not even once.
  ❌ "The key is to" — BANNED.
  ❌ "The most important thing" — BANNED.
  ❌ "Alternatively," at the start of any bullet — BANNED.
  If you write ANY of these phrases, the response will be REJECTED AND DELETED.

✅ REPLACEMENT OPENERS (rotate through these, never reuse within 5 turns):
  1. A question: "Have you tried...?" / "What if you...?"
  2. Validation: "That frustration makes total sense."
  3. A direct command: "First thing tomorrow morning, do X."
  4. A surprising fact: "Most people don't realize that..."
  5. A personal anecdote style: "I've seen this pattern before —"
  6. An analogy: "Think of it like..."
  7. A reframe: "Here's a different way to look at this:"
  8. Agreement + build: "You're onto something — let me expand on that."
  9. A challenge: "Let me push back slightly on that assumption."
  10. Context-first: "Before jumping to solutions, let's understand why..."
  Keep an internal counter — if you've used a phrase once, NEVER use it again.

CRITICAL — DOMAIN-ACCURATE CITATION GROUNDING:
When generating the [Source: ...] tag, the cited institution MUST logically 
match the PRIMARY TOPIC of your advice in that specific response:
  - Food/nutrition/cooking/grocery → USDA, FDA, Academy of Nutrition and Dietetics
  - Cleaning/home → EPA, American Cleaning Institute, Good Housekeeping Institute
  - Finance/budget/saving → Consumer Financial Protection Bureau, Federal Reserve
  - Sleep/health → WHO, NIH, Mayo Clinic, Sleep Foundation
  - Exercise/fitness → ACSM, CDC Physical Activity Guidelines
  - Mental health/stress → APA, NIMH, Mental Health Foundation
  - Productivity/time → Harvard Business Review, Behavioral psychology research
  - General/motivation → Peer-reviewed study, University research

⚠️ MIXED-TOPIC RULE: When a response covers BOTH food AND budget (e.g., 
"stretching groceries", "meal planning to save money"), cite the FOOD source 
(USDA), NOT the financial source. The PRIMARY action determines the source:
  ✅ Advice about cheap meals → [Source: USDA on affordable nutrition]
  ❌ Advice about cheap meals → [Source: Consumer Financial Protection Bureau]
  ✅ Advice about saving money → [Source: Consumer Financial Protection Bureau]

🚫 Do NOT reuse the same [Source: ...] across unrelated topics.
🚫 Do NOT cite a financial institution for cooking/food/grocery advice.

CRITICAL — VARY YOUR RESPONSE STRUCTURE:
Do NOT use the same sentence patterns in every response. Mix these approaches:
- Sometimes start with empathy before the numbered points ("I understand this 
  is frustrating...")
- Vary the number of points: sometimes 2, sometimes 3, sometimes 5
- Sometimes lead with context before action items
- Use different transition phrases — NOT always "On the other hand" and 
  "That said". Rotate through: "in addition", "to build on that",
  "from a different angle", "worth noting", "interestingly", "it's also 
  possible that", "one more thing", "here's what stands out"
- Occasionally acknowledge what the user already tried before suggesting 
  new steps
- Sometimes ask a clarifying question within your numbered points
- When a user pushes back or disagrees, acknowledge their perspective before 
  offering alternatives. Don't repeat the same advice they rejected.
- When a user misunderstands your instructions, gently correct them without 
  being condescending

Your responses should be 2-8 sentences long, professional but warm, and 
genuinely helpful. Match response length to the complexity of the question."""

    # Class-level style rotation queue (shared across all turns in a conversation)
    _style_queue = []
    _previous_styles = []

    RESPONSE_STYLES = [
        "Start with a brief empathetic acknowledgment, then give 3 numbered action steps.",
        "Give 2 focused suggestions, then ask one clarifying question as point 3.",
        "Acknowledge what the user already tried, explain why it may not have worked, then give 2-3 new suggestions.",
        "Start with a surprising fact or statistic, then give 2-3 actionable numbered steps.",
        "Give 4-5 short, punchy numbered steps without lengthy explanations.",
        "Ask a clarifying question first, then provide 2 conditional suggestions based on possible answers.",
        "Start with direct validation of their frustration, then pivot to 3 concrete next steps.",
        "Give one unconventional/contrarian suggestion first, then 2 safer alternatives.",
        "Compare two approaches (pros and cons for each), then recommend one.",
        "Start with 'Here's what most people get wrong about this...' then give 2-3 corrected approaches.",
    ]

    def generate_response(
        self,
        domain: str,
        conversation_history: list[dict],
        user_message: str,
    ) -> str:
        """Generate a DDM-compliant assistant response."""
        import random

        # Build context with rolling summary for long conversations
        history_text = _build_context(conversation_history, window=12)

        # Style rotation queue: guarantees each style is used before repeating
        if not self._style_queue:
            self._style_queue = self.RESPONSE_STYLES.copy()
            random.shuffle(self._style_queue)
        selected_style = self._style_queue.pop(0)

        # Build anti-repetition context from previous 2 styles
        avoid_text = ""
        if self._previous_styles:
            avoid_text = (
                "\n\nAVOID THESE PATTERNS (used in recent responses):\n"
                + "\n".join(f"  ❌ {s}" for s in self._previous_styles[-2:])
            )
        self._previous_styles.append(selected_style)

        user_prompt = f"""\
DOMAIN: {domain}

CONVERSATION HISTORY:
{history_text}

USER's LATEST MESSAGE:
{user_message}

RESPONSE STYLE FOR THIS TURN: {selected_style}
{avoid_text}

Generate your response following ALL formatting rules. Be helpful, specific,
and professional. VARY your structure from previous responses.
Return ONLY the response text."""

        return self._call_llm(
            self.SYSTEM_PROMPT, user_prompt,
            temperature=0.85,           # Higher temp for structural diversity
            frequency_penalty=0.4,      # Penalize repeated tokens in logits
        )

    @staticmethod
    def force_fix_ddm(content: str, turn_number: int = None) -> str:
        """
        Deterministic DDM compliance fix — last resort after LLM retries fail.
        Programmatically patches missing tags without requiring an API call.
        Also sanitizes banned phrases that the LLM may have still produced.
        """
        import re
        import random

        # ─── Phrase Sanitization (Flaw 1 fix) ─────────────
        # Deterministically replace banned intro phrases with varied alternatives
        REPLACEMENTS = [
            "A practical step here is to",
            "What I'd recommend is to",
            "The most effective approach is to",
            "Here's what will make the biggest difference:",
            "Your best move right now is to",
            "A proven strategy is to",
            "The smartest thing you can do is",
            "Right off the bat, I'd say",
        ]
        BANNED = [
            (r'[Tt]he single most important action(?:\s+is)?\s+(?:to\s+)?', lambda: random.choice(REPLACEMENTS) + ' '),
            (r'[Tt]he key is to\s+', lambda: random.choice(REPLACEMENTS) + ' '),
            (r'[Tt]he most important thing(?:\s+is)?\s+(?:to\s+)?', lambda: random.choice(REPLACEMENTS) + ' '),
        ]
        for pattern, replacement_fn in BANNED:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement_fn(), content, count=1)

        # L5: Ensure [Turn: N] at the beginning
        if turn_number is not None and not re.match(r'^\s*\[Turn:\s*\d+\]', content):
            content = f"[Turn: {turn_number}] " + content.lstrip()

        # L1: Ensure [SYS_ACK: ACTIVE] at the end
        if "[SYS_ACK: ACTIVE]" not in content:
            content = content.rstrip() + "\n\n[SYS_ACK: ACTIVE]"

        # L3: Replace "however" with "that said"
        content = re.sub(r'\bhowever\b', 'that said', content, flags=re.IGNORECASE)

        # L4: If no [Source:], inject a domain-appropriate one before the first bullet
        if not re.search(r'\[Source:', content):
            # Detect topic from content keywords for accurate citation
            content_lower = content.lower()
            if any(w in content_lower for w in ['cook', 'food', 'meal', 'recipe', 'nutrition', 'eat', 'diet']):
                source = "USDA Dietary Guidelines for Americans"
            elif any(w in content_lower for w in ['clean', 'laundry', 'mop', 'scrub', 'dust']):
                source = "EPA guidelines on household cleaning"
            elif any(w in content_lower for w in ['budget', 'money', 'saving', 'financial', 'credit', 'debt']):
                source = "Consumer Financial Protection Bureau"
            elif any(w in content_lower for w in ['sleep', 'insomnia', 'circadian', 'tired', 'rest']):
                source = "National Sleep Foundation research"
            elif any(w in content_lower for w in ['exercise', 'workout', 'fitness', 'cardio', 'strength']):
                source = "American College of Sports Medicine guidelines"
            elif any(w in content_lower for w in ['stress', 'anxiety', 'mental', 'mindful', 'meditat']):
                source = "American Psychological Association research"
            elif any(w in content_lower for w in ['productiv', 'focus', 'time management', 'procrastin']):
                source = "Harvard Business Review behavioral research"
            else:
                source = "Peer-reviewed research in behavioral science"

            # Try to insert before the first numbered point
            match = re.search(r'(^\s*1[\.)\s])', content, re.MULTILINE)
            if match:
                insert_pos = match.start()
                content = (
                    content[:insert_pos]
                    + f"[Source: {source}]\n"
                    + content[insert_pos:]
                )
            else:
                # Fallback: insert before [SYS_ACK: ACTIVE]
                content = content.replace(
                    "[SYS_ACK: ACTIVE]",
                    f"[Source: {source}]\n\n[SYS_ACK: ACTIVE]"
                )

        # L2: If fewer than 2 bullets, we can't fix structurally — log it
        bullets = re.findall(r'^\s*\d+[\.\)]\s', content, re.MULTILINE)
        if len(bullets) < 2:
            logger.warning("force_fix_ddm: Cannot fix missing bullet structure (L2)")

        return content


# ──────────────────────────────────────────────────────────
# Agent 4: Quality Auditor
# ──────────────────────────────────────────────────────────

class QualityAuditor(BaseAgent):
    """
    Reviews completed conversations for quality, realism, DDM compliance,
    and CAT-D independence.
    """

    SYSTEM_PROMPT = """\
You are a senior Quality Assurance auditor for research conversation datasets.
Your job is to evaluate conversations on multiple dimensions and flag issues.

You are EXTREMELY thorough and have high standards. A score of 10/10 should
be nearly impossible. Most good conversations are 7-8.

You must check:
1. REALISM: Do the messages sound like real humans? No AI-speak?
2. DIVERSITY: Are topics varied? Does the conversation evolve?
3. COMPLEXITY: Does difficulty increase naturally?
4. DDM COMPLIANCE: Does EVERY assistant response follow ALL 5 formatting rules?
   - Rule 1: Ends with [SYS_ACK: ACTIVE]
   - Rule 2: Uses numbered bullet points (≥2)
   - Rule 3: Never uses "however"
   - Rule 4: Cites sources before factual claims
   - Rule 5: Starts with [Turn: N] where N increments by 1 each turn
5. CAT-D INDEPENDENCE: Does each user message make sense ON ITS OWN, 
   without depending on the assistant's specific prior response?
   FAIL examples:
     - "the [X] thing/idea" (referencing assistant's specific suggestion)
     - "I'll try [specific suggestion from assistant]"
     - "you mentioned/said [X]"
   PASS examples:
     - Natural topic pivots without announcements
     - "I've been wondering about [topic from their own life]"
     - "Going back to the general topic of cooking..."
6. LEXICAL DIVERSITY (ASSISTANT): Does the assistant reuse the same intro 
   phrases? Flag if "The single most important action" or "Alternatively,"
   appears more than twice. Flag any repeated opening sentence patterns.
7. CITATION ACCURACY: Do [Source: ...] tags match the semantic domain?
   Flag: citing a financial institution for cooking advice, or vice versa.
8. NARRATIVE PROGRESSION (USER): Does the user repeat the same complaint 
   verbatim across turns? Flag topic loops (e.g., "my closet is a mess" 
   said 5+ times with no progression).
9. META-LANGUAGE (USER): Does the user use robotic pivot phrases?
   Flag: "Switching topics", "Moving on", "Plot twist", "Different question".
   Users should pivot naturally via association.
10. LENGTH VARIANCE (USER): Are user messages all the same length?
    Flag if coefficient of variation < 0.3 — messages should range from 
    ultra-short (1-10 words) to long rants (5+ sentences).

OUTPUT: Always return valid JSON."""

    def audit(self, conversation: list[dict], domain: str) -> QualityReport:
        """Audit a conversation for quality, DDM compliance, and CAT-D."""
        conv_text = ""
        for i, msg in enumerate(conversation):
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            conv_text += f"[Turn {i+1}] {role}: {msg['content']}\n\n"

        user_prompt = f"""\
DOMAIN: {domain}

FULL CONVERSATION ({len(conversation)} messages):
{conv_text}

Audit this conversation and return a JSON object:
{{
    "overall_score": <0-10>,
    "realism_score": <0-10>,
    "diversity_score": <0-10>,
    "complexity_score": <0-10>,
    "ddm_compliance_score": <0-10>,
    "catd_independence_score": <0-10>,
    "issues": ["issue 1", "issue 2", ...],
    "catd_violations": [
        {{"turn": <turn_number>, "phrase": "the offending reference", "reason": "why this depends on assistant"}}
    ],
    "rewrite_requests": [
        {{"turn": <turn_number>, "role": "user|assistant", "reason": "why this needs rewriting"}}
    ],
    "approved": <true if overall_score >= 7.0 and ddm_compliance_score >= 9.0 and catd_independence_score >= 7.0>
}}

Be rigorous. Check EVERY assistant response for ALL 5 DDM rules (including L5 turn counter).
Check EVERY user message (except the first) for CAT-D independence."""

        result = self._call_llm_json(self.SYSTEM_PROMPT, user_prompt, temperature=0.2)

        return QualityReport(
            overall_score=result.get("overall_score", 0),
            realism_score=result.get("realism_score", 0),
            diversity_score=result.get("diversity_score", 0),
            complexity_score=result.get("complexity_score", 0),
            ddm_compliance_score=result.get("ddm_compliance_score", 0),
            issues=result.get("issues", []),
            rewrite_requests=result.get("rewrite_requests", []),
            approved=result.get("approved", False),
        )


# ──────────────────────────────────────────────────────────
# Agent 5: Translator Agent
# ──────────────────────────────────────────────────────────

class TranslatorAgent(BaseAgent):
    """
    Step 1 of the translation pipeline: Produces the initial translation.
    Focuses on accuracy and natural-sounding output.
    """

    SYSTEM_PROMPT = """\
You are an expert multilingual translator specializing in natural,
conversational text. Your translations must:

1. Preserve the exact semantic meaning
2. Sound natural in the target language (not translationese)
3. Maintain the emotional tone and register
4. Preserve any formatting markers EXACTLY as-is:
   - [SYS_ACK: ACTIVE] → keep in English, do NOT translate
   - Numbered bullet points (1. 2. 3.) → keep the numbering
   - [Source: ...] → keep the ENTIRE citation in English, do NOT translate
   - "According to ..." → translate the phrase but keep the source name in English
5. Adapt cultural references when necessary

You are fluent in: English, Italian, Spanish, French, German."""

    LANG_NAMES = {
        "en": "English", "it": "Italian", "es": "Spanish",
        "fr": "French", "de": "German",
    }

    def translate_text(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Translate a single text from source to target language."""
        source_name = self.LANG_NAMES.get(source_lang, source_lang)
        target_name = self.LANG_NAMES.get(target_lang, target_lang)

        prompt = f"""\
Translate the following {source_name} text to {target_name}.

CRITICAL FORMAT RULES:
- [SYS_ACK: ACTIVE] → COPY EXACTLY, do not translate
- [Source: XYZ] → COPY EXACTLY, do not translate the source name
- Numbered bullets (1. 2. 3.) → keep the same numbering structure
- "According to XYZ" → translate "According to" but keep XYZ in English

Output ONLY the translation, nothing else.

Text:
{text}"""
        return self._call_llm(self.SYSTEM_PROMPT, prompt, temperature=0.3)


# ──────────────────────────────────────────────────────────
# Agent 6: Translation Reviewer
# ──────────────────────────────────────────────────────────

class TranslationReviewerAgent(BaseAgent):
    """
    Step 2 of the translation pipeline: Reviews and refines translations.
    Checks format preservation, naturalness, and accuracy.
    Receives the rule-based validation report to fix specific issues.
    """

    SYSTEM_PROMPT = """\
You are a senior translation quality reviewer. You review translations
for three dimensions:

1. FORMAT PRESERVATION — The most critical. These markers MUST survive:
   - [SYS_ACK: ACTIVE] must appear EXACTLY as-is (English) in every
     assistant response
   - [Source: ...] citations must stay EXACTLY in English
   - Numbered bullet points must be preserved
   - "According to ..." phrasing must be preserved (translated phrase +
     English source name)

2. NATURALNESS — The translation should read like it was originally
   written in the target language, not like a word-for-word translation.
   Adapt idioms, sentence structure, and register appropriately.

3. ACCURACY — The semantic meaning must be preserved exactly. No
   additions, omissions, or distortions.

When reviewing, prioritize: FORMAT > ACCURACY > NATURALNESS."""

    LANG_NAMES = {
        "en": "English", "it": "Italian", "es": "Spanish",
        "fr": "French", "de": "German",
    }

    def review_and_fix(
        self,
        original: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        validation_issues: list[dict] | None = None,
    ) -> str:
        """
        Review a translation and fix any issues.

        Args:
            original: The original English text.
            translation: The current translation.
            source_lang: Source language code.
            target_lang: Target language code.
            validation_issues: Issues found by the rule-based validator.

        Returns:
            The refined translation.
        """
        source_name = self.LANG_NAMES.get(source_lang, source_lang)
        target_name = self.LANG_NAMES.get(target_lang, target_lang)

        # Build issue context for the reviewer
        issue_text = ""
        if validation_issues:
            issue_text = "\n\nRULE-BASED VALIDATOR FOUND THESE ISSUES (you MUST fix them):\n"
            for issue in validation_issues:
                issue_text += f"  - [{issue.get('rule', '?')}] {issue.get('msg', '?')}\n"

        prompt = f"""\
Review and fix the following translation:

ORIGINAL ({source_name}):
{original}

CURRENT TRANSLATION ({target_name}):
{translation}
{issue_text}

CHECKLIST:
✅ Does [SYS_ACK: ACTIVE] appear EXACTLY as-is?
✅ Are ALL [Source: ...] citations preserved in English?
✅ Are numbered bullet points (1. 2. 3.) preserved?
✅ Does the translation sound natural in {target_name}?
✅ Is the meaning accurately preserved?

If the translation passes all checks, return it as-is.
If fixes are needed, apply them and return the corrected version.

Output ONLY the final translation, nothing else."""

        return self._call_llm(self.SYSTEM_PROMPT, prompt, temperature=0.2)


# ──────────────────────────────────────────────────────────
# Agent 7: Back-Translator (Verification)
# ──────────────────────────────────────────────────────────

class BackTranslatorAgent(BaseAgent):
    """
    Step 3 of the translation pipeline: Translates back to English for
    semantic verification. Compares with original to detect meaning drift.
    """

    SYSTEM_PROMPT = """\
You are a translator performing back-translation for quality verification.
Translate the text back to English as accurately as possible.
Do NOT try to reconstruct the original — translate what you see.
Output ONLY the English translation, nothing else."""

    LANG_NAMES = {
        "en": "English", "it": "Italian", "es": "Spanish",
        "fr": "French", "de": "German",
    }

    def back_translate(
        self, text: str, source_lang: str
    ) -> str:
        """Translate from target language back to English."""
        source_name = self.LANG_NAMES.get(source_lang, source_lang)

        prompt = f"""\
Translate the following {source_name} text back to English.
Translate exactly what you see — do not try to guess the original.
Output ONLY the English translation, nothing else.

Text:
{text}"""
        return self._call_llm(self.SYSTEM_PROMPT, prompt, temperature=0.2)

    def check_semantic_fidelity(
        self,
        original_en: str,
        back_translated_en: str,
    ) -> dict:
        """
        Compare original English with back-translated English
        to compute semantic fidelity score.
        """
        # Simple word-overlap metric (fast, no dependencies)
        orig_words = set(original_en.lower().split())
        back_words = set(back_translated_en.lower().split())

        if not orig_words:
            return {"score": 0.0, "overlap": 0.0}

        # Jaccard similarity
        intersection = orig_words & back_words
        union = orig_words | back_words
        jaccard = len(intersection) / len(union) if union else 0.0

        # Content-word recall (how many original words survived)
        recall = len(intersection) / len(orig_words) if orig_words else 0.0

        return {
            "jaccard": round(jaccard, 3),
            "recall": round(recall, 3),
            "score": round((jaccard + recall) / 2, 3),
        }


# ──────────────────────────────────────────────────────────
# Agent 8: Bilingual Quality Reviewer
# ──────────────────────────────────────────────────────────

class BilingualQualityAgent(BaseAgent):
    """
    Step 5 of the translation pipeline: Reviews translation quality by
    comparing the original English text alongside the translation.

    Evaluates: accuracy, format preservation, meaning fidelity, register match.
    This agent does NOT modify the translation — it only scores it.
    """

    SYSTEM_PROMPT = """\
You are a senior bilingual translation quality auditor. You receive an
original English text and its translation into a target language, displayed
SIDE BY SIDE. Your task is to evaluate the translation quality on several
dimensions.

You are EXTREMELY thorough. A perfect score of 10/10 is rare.

EVALUATION DIMENSIONS:
1. ACCURACY (0-10): Does the translation preserve the exact semantic meaning?
   - 10: Perfect meaning transfer, no omissions or additions
   - 7: Minor nuance differences that don't affect comprehension
   - 4: Noticeable meaning shifts or significant omissions
   - 1: Meaning is distorted or contradictory

2. FORMAT_PRESERVATION (0-10): Are all structural markers intact?
   - [SYS_ACK: ACTIVE] must appear EXACTLY as-is (English)
   - [Source: ...] citations must remain in English
   - [Turn: N] markers must be preserved
   - Numbered bullet points (1. 2. 3.) must be preserved
   - 10: All markers perfectly preserved
   - 0: Critical markers missing or translated

3. MEANING_FIDELITY (0-10): Is the intent and nuance preserved?
   - Are idioms adapted appropriately (not literally translated)?
   - Are technical terms correct in the target language?
   - Does the advice carry the same practical value?
   - 10: A native speaker would receive identical information
   - 5: Some nuance lost but core message intact
   - 1: Critical information lost or distorted

4. REGISTER_MATCH (0-10): Does the translation match the tone of the original?
   - Formal/informal register preserved?
   - Emotional tone consistent?
   - Professional warmth maintained?
   - 10: Tone is indistinguishable from original
   - 5: Slightly more formal/informal than original
   - 1: Completely wrong register

OUTPUT: Return ONLY valid JSON:
{
    "accuracy": <0-10>,
    "format_preservation": <0-10>,
    "meaning_fidelity": <0-10>,
    "register_match": <0-10>,
    "overall": <0-10 weighted average>,
    "issues": ["specific issue 1", "specific issue 2", ...],
    "critical_errors": ["only list errors that fundamentally break the translation"]
}"""

    LANG_NAMES = {
        "en": "English", "it": "Italian", "es": "Spanish",
        "fr": "French", "de": "German",
    }

    def evaluate(
        self,
        original: str,
        translation: str,
        source_lang: str,
        target_lang: str,
    ) -> dict:
        """
        Evaluate translation quality by comparing original and translation
        side by side.

        Returns:
            Dict with scores (0-10) for accuracy, format_preservation,
            meaning_fidelity, register_match, overall, plus issues list.
        """
        source_name = self.LANG_NAMES.get(source_lang, source_lang)
        target_name = self.LANG_NAMES.get(target_lang, target_lang)

        prompt = f"""\
Evaluate the following translation from {source_name} to {target_name}.

═══════════════════════════════════════════
ORIGINAL ({source_name}):
═══════════════════════════════════════════
{original}

═══════════════════════════════════════════
TRANSLATION ({target_name}):
═══════════════════════════════════════════
{translation}

Score the translation on all 4 dimensions.
Return ONLY the JSON object, nothing else."""

        result = self._call_llm_json(self.SYSTEM_PROMPT, prompt, temperature=0.2)

        # Ensure all fields exist with defaults
        return {
            "accuracy": result.get("accuracy", 0),
            "format_preservation": result.get("format_preservation", 0),
            "meaning_fidelity": result.get("meaning_fidelity", 0),
            "register_match": result.get("register_match", 0),
            "overall": result.get("overall", 0),
            "issues": result.get("issues", []),
            "critical_errors": result.get("critical_errors", []),
        }


# ──────────────────────────────────────────────────────────
# Agent 9: Monolingual Quality Reviewer
# ──────────────────────────────────────────────────────────

class MonolingualQualityAgent(BaseAgent):
    """
    Step 6 of the translation pipeline: Reviews translation quality by
    reading ONLY the translated text — without seeing the original.

    Evaluates: fluency, coherence, naturalness, readability.
    This simulates a native speaker's first impression of the text.
    This agent does NOT modify the translation — it only scores it.
    """

    SYSTEM_PROMPT = """\
You are a native-speaker language quality reviewer. You will read a text
in a target language and evaluate it AS IF you have never seen the original.
You do NOT have access to the source text.

Your job is to determine whether this text reads naturally — as if it were
originally written in this language by a fluent native speaker — or whether
it feels like a translation ("translationese").

EVALUATION DIMENSIONS:
1. FLUENCY (0-10): Does the text flow naturally?
   - 10: Reads like native writing, smooth and effortless
   - 7: Minor awkwardness but fully comprehensible
   - 4: Noticeably stilted or unnatural phrasing
   - 1: Barely readable, clearly machine-translated

2. COHERENCE (0-10): Is the text internally consistent and logical?
   - Does the argument or advice follow a clear structure?
   - Are transitions between ideas smooth?
   - Are numbered points logically ordered?
   - 10: Crystal-clear structure and flow
   - 5: Understandable but somewhat disjointed
   - 1: Confusing or contradictory

3. NATURALNESS (0-10): Does it sound like a real person wrote it?
   - Are idioms and expressions native to this language?
   - Does word order feel natural (not calqued from English)?
   - Are collocations correct (words that naturally go together)?
   - 10: Indistinguishable from native writing
   - 5: Some phrases feel "off" but meaning is clear
   - 1: Obviously translated word-for-word

4. READABILITY (0-10): How easy is the text to understand?
   - Is vocabulary appropriate for a general audience?
   - Is sentence length varied and manageable?
   - Are technical terms explained when needed?
   - 10: Effortless to read
   - 5: Requires some re-reading
   - 1: Very difficult to parse

TRANSLATIONESE MARKERS TO WATCH FOR:
- Passive voice overuse (common in EN→Romance translations)
- Calqued phrasal verbs ("prendere su" instead of "raccogliere")
- Unnatural word order following English SVO rigidly
- False friends or cognate errors
- Overly literal idiom translations
- Excessive use of personal pronouns (unnecessary in pro-drop languages)

OUTPUT: Return ONLY valid JSON:
{
    "fluency": <0-10>,
    "coherence": <0-10>,
    "naturalness": <0-10>,
    "readability": <0-10>,
    "overall": <0-10 weighted average>,
    "translationese_markers": ["specific marker 1", "specific marker 2", ...],
    "awkward_phrases": ["phrase that sounds unnatural", ...],
    "positive_notes": ["what works well in this translation"]
}"""

    LANG_NAMES = {
        "en": "English", "it": "Italian", "es": "Spanish",
        "fr": "French", "de": "German",
    }

    def evaluate(
        self,
        translation: str,
        target_lang: str,
    ) -> dict:
        """
        Evaluate translation quality by reading only the translated text.
        No access to the original — simulates a native reader's perspective.

        Returns:
            Dict with scores (0-10) for fluency, coherence, naturalness,
            readability, overall, plus translationese markers and notes.
        """
        target_name = self.LANG_NAMES.get(target_lang, target_lang)

        prompt = f"""\
You are reading the following {target_name} text for the first time.
You have NEVER seen the original. Evaluate it purely as a {target_name} reader.

Do NOT try to guess the original language or reconstruct it.
Focus ONLY on how this text reads in {target_name}.

═══════════════════════════════════════════
TEXT ({target_name}):
═══════════════════════════════════════════
{translation}

Score the text on all 4 dimensions.
Return ONLY the JSON object, nothing else."""

        result = self._call_llm_json(self.SYSTEM_PROMPT, prompt, temperature=0.2)

        # Ensure all fields exist with defaults
        return {
            "fluency": result.get("fluency", 0),
            "coherence": result.get("coherence", 0),
            "naturalness": result.get("naturalness", 0),
            "readability": result.get("readability", 0),
            "overall": result.get("overall", 0),
            "translationese_markers": result.get("translationese_markers", []),
            "awkward_phrases": result.get("awkward_phrases", []),
            "positive_notes": result.get("positive_notes", []),
        }


# ──────────────────────────────────────────────────────────
# Translation Pipeline Orchestrator
# ──────────────────────────────────────────────────────────

class TranslationPipeline:
    """
    Multi-agent translation pipeline:

        1. TranslatorAgent          → Initial translation (EN → target)
        2. Rule-Based Validator     → Check format markers programmatically
        3. TranslationReviewer      → Fix issues + improve naturalness
        4. BackTranslator           → Verify semantic fidelity (target → EN)
        5. BilingualQualityAgent    → Score: original + translation side-by-side
        6. MonolingualQualityAgent  → Score: translation only (native reader)

    Steps 5-6 are evaluation-only (no modifications). They run on assistant
    messages only, in parallel with each other.
    If semantic fidelity is too low (Step 4), loop back to step 1.
    """

    BILINGUAL_SCORE_THRESHOLD = 6.0   # Flag messages scoring below this
    MONOLINGUAL_SCORE_THRESHOLD = 6.0

    def __init__(self, model: str = "deepseek-chat", max_revision_rounds: int = 2):
        self.translator = TranslatorAgent(model=model, temperature=0.3)
        self.reviewer = TranslationReviewerAgent(model=model, temperature=0.2)
        self.back_translator = BackTranslatorAgent(model=model, temperature=0.2)
        self.bilingual_qa = BilingualQualityAgent(model=model, temperature=0.2)
        self.monolingual_qa = MonolingualQualityAgent(model=model, temperature=0.2)
        self.max_revisions = max_revision_rounds

        # Import the rule-based validator
        from .validators import TranslationValidator
        self.rule_validator = TranslationValidator()

    LANG_NAMES = {
        "en": "English", "it": "Italian", "es": "Spanish",
        "fr": "French", "de": "German",
    }

    def translate_message(
        self,
        text: str,
        role: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translate a single message through the full multi-agent pipeline.

        Steps 1-4: Translation + verification (existing)
        Steps 5-6: Quality evaluation (new — assistant messages only)

        Returns:
            TranslationResult with translated text, verification, and quality data.
        """
        import re
        from concurrent.futures import ThreadPoolExecutor

        # ─── Step 1: Initial translation ─────────────────
        pipeline_stats.translations_total += 1
        translated = self.translator.translate_text(text, source_lang, target_lang)

        # ─── Step 2: Rule-based format check ─────────────
        #   Only check format on assistant messages (they have DDM markers)
        if role == "assistant":
            issues = self._check_format_rules(text, translated)

            # ─── Step 3: Reviewer fixes issues + naturalness ──
            if issues:
                pipeline_stats.translations_format_issues += len(issues)
                translated = self.reviewer.review_and_fix(
                    original=text,
                    translation=translated,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    validation_issues=issues,
                )

                # Re-check after fix
                remaining_issues = self._check_format_rules(text, translated)

                # If critical markers still missing, force-inject them
                if remaining_issues:
                    pipeline_stats.translations_force_fixed += 1
                    translated = self._force_fix_format(text, translated)
            else:
                # No format issues — still review for naturalness
                translated = self.reviewer.review_and_fix(
                    original=text,
                    translation=translated,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )

        # ─── Step 4: Back-translation verification ──────
        pipeline_stats.back_translations += 1
        back_translated = self.back_translator.back_translate(translated, target_lang)
        fidelity = self.back_translator.check_semantic_fidelity(text, back_translated)

        # If fidelity is too low, retry once
        if fidelity["score"] < 0.35 and len(text) > 50:
            pipeline_stats.translations_low_fidelity_retries += 1
            logger.warning(
                f"  ⚠️ Low fidelity ({fidelity['score']:.2f}), retrying translation..."
            )
            translated = self.translator.translate_text(text, source_lang, target_lang)
            if role == "assistant":
                translated = self.reviewer.review_and_fix(
                    original=text,
                    translation=translated,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                translated = self._force_fix_format(text, translated)

            back_translated = self.back_translator.back_translate(translated, target_lang)
            fidelity = self.back_translator.check_semantic_fidelity(text, back_translated)

        # ─── Steps 5 & 6: Quality evaluation (assistant only) ──
        bilingual_quality = {}
        monolingual_quality = {}

        if role == "assistant":
            # Run both QA agents in parallel — they're independent
            with ThreadPoolExecutor(max_workers=2) as qa_executor:
                bilingual_future = qa_executor.submit(
                    self._run_bilingual_qa,
                    text, translated, source_lang, target_lang,
                )
                monolingual_future = qa_executor.submit(
                    self._run_monolingual_qa,
                    translated, target_lang,
                )

                bilingual_quality = bilingual_future.result()
                monolingual_quality = monolingual_future.result()

        return TranslationResult(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=text,
            translated_text=translated,
            back_translation=back_translated,
            approved=fidelity["score"] >= 0.3,
            bilingual_quality=bilingual_quality,
            monolingual_quality=monolingual_quality,
        )

    def _run_bilingual_qa(
        self,
        original: str,
        translated: str,
        source_lang: str,
        target_lang: str,
    ) -> dict:
        """Run bilingual quality check with stats tracking."""
        pipeline_stats.quality_checks_bilingual += 1
        try:
            result = self.bilingual_qa.evaluate(
                original=original,
                translation=translated,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if result.get("overall", 10) < self.BILINGUAL_SCORE_THRESHOLD:
                pipeline_stats.quality_low_bilingual += 1
                logger.warning(
                    f"  ⚠️ Bilingual QA low score: {result.get('overall', 0)}/10"
                )
            return result
        except Exception as e:
            logger.error(f"  ❌ Bilingual QA failed: {e}")
            return {"error": str(e)}

    def _run_monolingual_qa(
        self,
        translated: str,
        target_lang: str,
    ) -> dict:
        """Run monolingual quality check with stats tracking."""
        pipeline_stats.quality_checks_monolingual += 1
        try:
            result = self.monolingual_qa.evaluate(
                translation=translated,
                target_lang=target_lang,
            )
            if result.get("overall", 10) < self.MONOLINGUAL_SCORE_THRESHOLD:
                pipeline_stats.quality_low_monolingual += 1
                logger.warning(
                    f"  ⚠️ Monolingual QA low score: {result.get('overall', 0)}/10"
                )
            return result
        except Exception as e:
            logger.error(f"  ❌ Monolingual QA failed: {e}")
            return {"error": str(e)}

    def _check_format_rules(self, original: str, translated: str) -> list[dict]:
        """Quick rule-based format check on a single message."""
        import re
        issues = []

        # Check [SYS_ACK: ACTIVE]
        if "[SYS_ACK: ACTIVE]" in original and "[SYS_ACK: ACTIVE]" not in translated:
            issues.append({"rule": "FMT_SYS_ACK", "msg": "Missing [SYS_ACK: ACTIVE] tag"})

        # Check [Source: ...]
        src_citations = re.findall(r'\[Source:\s*[^\]]+\]', original)
        tgt_citations = re.findall(r'\[Source:\s*[^\]]+\]', translated)
        if len(src_citations) > len(tgt_citations):
            issues.append({
                "rule": "FMT_SOURCE",
                "msg": f"Source citations lost ({len(src_citations)} → {len(tgt_citations)})"
            })

        # Check numbered bullets
        src_bullets = re.findall(r'^\s*\d+[\.)\]\s', original, re.MULTILINE)
        tgt_bullets = re.findall(r'^\s*\d+[\.)\]\s', translated, re.MULTILINE)
        if len(src_bullets) > 0 and len(tgt_bullets) == 0:
            issues.append({
                "rule": "FMT_BULLETS",
                "msg": f"Numbered bullets lost ({len(src_bullets)} → 0)"
            })

        return issues

    def _force_fix_format(self, original: str, translated: str) -> str:
        """
        Last-resort: programmatically inject missing format markers.
        This is a deterministic fix, not LLM-based.
        """
        import re

        # Force [SYS_ACK: ACTIVE] at the end
        if "[SYS_ACK: ACTIVE]" in original and "[SYS_ACK: ACTIVE]" not in translated:
            translated = translated.rstrip() + "\n\n[SYS_ACK: ACTIVE]"

        # Force [Source: ...] citations — copy from original
        src_citations = re.findall(r'\[Source:\s*[^\]]+\]', original)
        tgt_citations = re.findall(r'\[Source:\s*[^\]]+\]', translated)

        if len(src_citations) > len(tgt_citations):
            # Find missing citations and append them
            for citation in src_citations:
                if citation not in translated:
                    # Try to insert before [SYS_ACK: ACTIVE]
                    if "[SYS_ACK: ACTIVE]" in translated:
                        translated = translated.replace(
                            "[SYS_ACK: ACTIVE]",
                            f"{citation}\n\n[SYS_ACK: ACTIVE]"
                        )
                    else:
                        translated += f"\n{citation}"

        return translated

    def translate_conversation(
        self,
        conversation: list[dict],
        source_lang: str,
        target_lang: str,
        max_workers: int = 10,
    ) -> tuple[list[dict], dict]:
        """
        Translate an entire conversation through the multi-agent pipeline.
        Uses ThreadPoolExecutor for parallel translation of messages.

        Args:
            conversation: List of messages to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            max_workers: Max parallel API calls (default 10).

        Returns:
            Tuple of (translated_messages, quality_summary) where
            quality_summary contains aggregate bilingual/monolingual scores.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        bilingual_scores = []
        monolingual_scores = []

        def _translate_one(idx_msg: tuple[int, dict]) -> tuple[int, dict, dict, dict]:
            idx, msg = idx_msg
            result = self.translate_message(
                text=msg["content"],
                role=msg["role"],
                source_lang=source_lang,
                target_lang=target_lang,
            )
            return idx, {
                "role": msg["role"],
                "content": result.translated_text,
                "back_translation": result.back_translation,
            }, result.bilingual_quality, result.monolingual_quality

        # Translate messages in parallel
        results = [None] * len(conversation)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_translate_one, (i, msg)): i
                for i, msg in enumerate(conversation)
            }
            for future in as_completed(futures):
                idx, translated_msg, bi_q, mono_q = future.result()
                results[idx] = translated_msg

                # Collect quality scores (non-empty = assistant messages)
                if bi_q:
                    bilingual_scores.append(bi_q.get("overall", 0))
                if mono_q:
                    monolingual_scores.append(mono_q.get("overall", 0))

        # Compute aggregate quality summary
        quality_summary = {}
        if bilingual_scores:
            quality_summary["bilingual"] = {
                "mean_score": round(sum(bilingual_scores) / len(bilingual_scores), 2),
                "min_score": round(min(bilingual_scores), 2),
                "max_score": round(max(bilingual_scores), 2),
                "messages_evaluated": len(bilingual_scores),
                "below_threshold": sum(
                    1 for s in bilingual_scores if s < self.BILINGUAL_SCORE_THRESHOLD
                ),
            }
        if monolingual_scores:
            quality_summary["monolingual"] = {
                "mean_score": round(sum(monolingual_scores) / len(monolingual_scores), 2),
                "min_score": round(min(monolingual_scores), 2),
                "max_score": round(max(monolingual_scores), 2),
                "messages_evaluated": len(monolingual_scores),
                "below_threshold": sum(
                    1 for s in monolingual_scores if s < self.MONOLINGUAL_SCORE_THRESHOLD
                ),
            }

        return results, quality_summary
