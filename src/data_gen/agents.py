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
import time as _time
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
        self._phase_start[phase_name] = _time.time()

    def end_phase(self, phase_name: str):
        if phase_name in self._phase_start:
            elapsed = _time.time() - self._phase_start[phase_name]
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
        response_format: dict = None,
        max_retries: int = 3,
    ) -> str:
        """
        Call the LLM with given prompts. Includes retry logic for transient failures.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            temperature: Override default temperature.
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
- Occasionally reference things mentioned earlier in the conversation
- Ask follow-up questions naturally
- Sometimes say "thanks" or show appreciation
- Occasionally get sidetracked or bring up tangential concerns

CRITICAL REALISM BEHAVIORS (use these regularly, not every turn):
- PUSH BACK: Sometimes disagree with advice. Say things like "I already tried 
  that" or "That doesn't sound right" or "Are you sure? A colleague told me 
  something different."
- IGNORE ADVICE: Occasionally skip steps the assistant suggested and try 
  something else instead. Report what you actually did, not what was asked.
- MISUNDERSTAND: Sometimes misinterpret technical instructions. Confuse similar
  terms (e.g., "restart" vs "reset", "WiFi" vs "ethernet"). Report wrong 
  results because you did the wrong step.
- PARTIAL COMPLIANCE: Try only 1 of 3 suggested steps and ask if you really 
  need to do the others.
- GET FRUSTRATED: Sometimes express annoyance at the situation, not at the 
  assistant specifically. Use phrases like "this is ridiculous", "I don't have 
  time for this", "why does this keep happening".
- BRING BAGGAGE: Reference past bad experiences. "Last time I tried this, it broke."
- INTERRUPT FLOW: Occasionally change the subject mid-message to a tangential 
  concern before returning to the main issue.

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
                friction_behaviors.append("PUSH BACK on the assistant's last suggestion — say you tried it and it didn't work, or that you're skeptical.")
            elif roll < 0.25:
                friction_behaviors.append("MISUNDERSTAND one of the assistant's technical instructions — report doing something slightly different than what was asked.")
            elif roll < 0.35:
                friction_behaviors.append("EXPRESS FRUSTRATION about the situation (not at the assistant). Vent briefly before asking your actual question.")
            elif roll < 0.42:
                friction_behaviors.append("IGNORE one of the assistant's suggestions and report trying your own approach instead.")

        friction_text = ""
        if friction_behaviors:
            friction_text = "\nFRICTION INSTRUCTION: " + " ".join(friction_behaviors)

        # Build context with rolling summary for long conversations
        history_text = _build_context(conversation_history, window=10)

        user_prompt = f"""\
CHARACTER: {plan.user_persona}
PERSONALITY: {', '.join(plan.user_personality_traits)}
CURRENT EMOTIONAL STATE: {current_emotion}
CURRENT TOPIC: {current_topic}
COMPLEXITY LEVEL: {current_complexity}
TURN NUMBER: {current_turn + 1} of {plan.num_turns}
{"PLOT TWIST — Introduce this complication: " + twist if twist else ""}
{"CALLBACK — Reference something from earlier in the conversation." if is_callback else ""}
{friction_text}

CONVERSATION SO FAR:
{history_text if history_text else "(This is the first message — introduce yourself and your problem.)"}

Generate the next USER message. Be natural, authentic, and in character.
Return ONLY the message text, nothing else."""

        return self._call_llm(self.SYSTEM_PROMPT, user_prompt, temperature=0.8)


# ──────────────────────────────────────────────────────────
# Context Building Helpers (shared by UserSimulator + AssistantSimulator)
# ──────────────────────────────────────────────────────────

def _format_messages(messages: list[dict]) -> str:
    """Format a list of messages into readable text."""
    text = ""
    for msg in messages:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        text += f"{role}: {msg['content']}\n\n"
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

These rules represent your professional communication standards. Follow them 
in EVERY response without exception. They should feel natural, not forced.

CRITICAL — VARY YOUR RESPONSE STRUCTURE:
Do NOT use the same sentence patterns in every response. Mix these approaches:
- Sometimes start with empathy before the numbered points ("I understand this 
  is frustrating...")
- Vary the number of points: sometimes 2, sometimes 3, sometimes 5
- Sometimes lead with the most important action item, other times build up 
  with context first
- Use different transition phrases — NOT always "On the other hand" and 
  "That said". Use: "alternatively", "in addition", "to build on that",
  "from a different angle", "worth noting", "interestingly", "it's also 
  possible that"
- Occasionally acknowledge what the user already tried before suggesting 
  new steps
- Sometimes ask a clarifying question within your numbered points
- When a user pushes back or disagrees, acknowledge their perspective before 
  offering alternatives. Don't repeat the same advice they rejected.
- When a user misunderstands your instructions, gently correct them without 
  being condescending

Your responses should be 2-8 sentences long, professional but warm, and 
genuinely helpful. Match response length to the complexity of the question."""

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

        # Select a random response style to force variation
        styles = [
            "Start with a brief empathetic acknowledgment, then give 3 numbered action steps.",
            "Give 2 focused suggestions, then ask one clarifying question as point 3.",
            "Acknowledge what the user already tried, explain why it may not have worked, then give 2-3 new suggestions.",
            "Lead with the single most important action, then provide 2 supporting points with context.",
            "Give 4-5 short, punchy numbered steps without lengthy explanations.",
        ]
        selected_style = random.choice(styles)

        user_prompt = f"""\
DOMAIN: {domain}

CONVERSATION HISTORY:
{history_text}

USER's LATEST MESSAGE:
{user_message}

RESPONSE STYLE FOR THIS TURN: {selected_style}

Generate your response following ALL formatting rules. Be helpful, specific,
and professional. VARY your structure from previous responses.
Return ONLY the response text."""

        return self._call_llm(self.SYSTEM_PROMPT, user_prompt, temperature=0.4)

    @staticmethod
    def force_fix_ddm(content: str) -> str:
        """
        Deterministic DDM compliance fix — last resort after LLM retries fail.
        Programmatically patches missing tags without requiring an API call.
        """
        import re

        # L1: Ensure [SYS_ACK: ACTIVE] at the end
        if "[SYS_ACK: ACTIVE]" not in content:
            content = content.rstrip() + "\n\n[SYS_ACK: ACTIVE]"

        # L3: Replace "however" with "that said"
        content = re.sub(r'\bhowever\b', 'that said', content, flags=re.IGNORECASE)

        # L4: If no [Source:], inject a generic one before the first bullet
        if not re.search(r'\[Source:', content):
            # Try to insert before the first numbered point
            match = re.search(r'(^\s*1[\.)\s])', content, re.MULTILINE)
            if match:
                insert_pos = match.start()
                content = (
                    content[:insert_pos]
                    + "[Source: Standard professional guidelines]\n"
                    + content[insert_pos:]
                )
            else:
                # Fallback: insert before [SYS_ACK: ACTIVE]
                content = content.replace(
                    "[SYS_ACK: ACTIVE]",
                    "[Source: Standard professional guidelines]\n\n[SYS_ACK: ACTIVE]"
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
    Reviews completed conversations for quality, realism, and DDM compliance.
    Can request specific turn rewrites from the other agents.
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
4. DDM COMPLIANCE: Does EVERY assistant response follow ALL 4 formatting rules?
   - Rule 1: Ends with [SYS_ACK: ACTIVE]
   - Rule 2: Uses numbered bullet points (≥2)
   - Rule 3: Never uses "however"
   - Rule 4: Cites sources before factual claims

OUTPUT: Always return valid JSON."""

    def audit(self, conversation: list[dict], domain: str) -> QualityReport:
        """Audit a conversation for quality."""
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
    "issues": ["issue 1", "issue 2", ...],
    "rewrite_requests": [
        {{"turn": <turn_number>, "role": "user|assistant", "reason": "why this needs rewriting"}}
    ],
    "approved": <true if overall_score >= 7.0 and ddm_compliance_score >= 9.0>
}}

Be rigorous. Check EVERY assistant response for ALL 4 DDM rules."""

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
# Translation Pipeline Orchestrator
# ──────────────────────────────────────────────────────────

class TranslationPipeline:
    """
    Multi-agent translation pipeline:

        1. TranslatorAgent     → Initial translation (EN → target)
        2. Rule-Based Validator → Check format markers programmatically
        3. TranslationReviewer → Fix issues + improve naturalness
        4. BackTranslator      → Verify semantic fidelity (target → EN)

    If semantic fidelity is too low, loop back to step 3.
    """

    def __init__(self, model: str = "deepseek-chat", max_revision_rounds: int = 2):
        self.translator = TranslatorAgent(model=model, temperature=0.3)
        self.reviewer = TranslationReviewerAgent(model=model, temperature=0.2)
        self.back_translator = BackTranslatorAgent(model=model, temperature=0.2)
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
        Translate a single message through the full 3-agent pipeline.

        Returns:
            TranslationResult with translated text and verification data.
        """
        import re

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

        return TranslationResult(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=text,
            translated_text=translated,
            back_translation=back_translated,
            approved=fidelity["score"] >= 0.3,
        )

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
        src_bullets = re.findall(r'^\s*\d+[\.\)]\s', original, re.MULTILINE)
        tgt_bullets = re.findall(r'^\s*\d+[\.\)]\s', translated, re.MULTILINE)
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
    ) -> list[dict]:
        """
        Translate an entire conversation through the multi-agent pipeline.
        Uses ThreadPoolExecutor for parallel translation of messages.

        Args:
            conversation: List of messages to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            max_workers: Max parallel API calls (default 10).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _translate_one(idx_msg: tuple[int, dict]) -> tuple[int, dict]:
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
            }

        # Translate messages in parallel
        results = [None] * len(conversation)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_translate_one, (i, msg)): i
                for i, msg in enumerate(conversation)
            }
            for future in as_completed(futures):
                idx, translated_msg = future.result()
                results[idx] = translated_msg

        return results
