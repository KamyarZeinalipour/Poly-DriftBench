"""
Dynamic User Simulator — Track 2 Agentic Mode
================================================
Uses the DeepSeek API as a "Driver Model" to play the user role
dynamically during inference. The driver reads the test model's actual
output and generates contextually appropriate follow-up questions.

This eliminates Trajectory Mismatch entirely: the user messages are
always coherent with what the test model actually said.

Usage:
    sim = DynamicUserSimulator(
        domain="pet_care",
        goal_state="You want advice on raising a new puppy",
        num_turns=50,
    )
    next_msg = sim.generate_next_message(conversation_history)
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Default DeepSeek model — cheapest option
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class DynamicUserSimulator:
    """
    Uses DeepSeek API to dynamically generate user messages during inference.

    The simulator is given a hidden goal state and domain. It reads the test
    model's actual responses and generates natural follow-up questions,
    ensuring perfect conversational coherence.

    This is Track 2 of the Dual-Track Evaluation Framework.
    """

    SYSTEM_PROMPT = """\
You are simulating a human user interacting with an AI assistant via chat.
You must stay in character as a real person. Generate natural, casual messages.

YOUR HIDDEN GOAL: {goal_state}
DOMAIN: {domain_description}
AVAILABLE TOPICS: {topics}
TARGET TURNS: {num_turns}

RULES:
1. Be natural — use contractions, casual phrasing, occasionally be vague.
2. Introduce topics from the AVAILABLE TOPICS list throughout the conversation.
3. React naturally to what the assistant ACTUALLY said — ask follow-ups,
   express opinions, push back sometimes, show gratitude occasionally.
4. DO NOT break character. You are a human, not an AI.
5. Keep messages SHORT (1-3 sentences typically). Real users don't write essays.
6. Vary your style: sometimes ask a direct question, sometimes share context
   first, sometimes express frustration, sometimes just say "thanks, also..."
7. DO NOT mention your hidden goal or the number of turns.
8. Occasionally push back ("I already tried that", "are you sure?")
9. Every 5-8 turns, pivot to a new topic from the list.

Generate ONLY the user message. No quotes, no labels, just the raw message."""

    def __init__(
        self,
        domain: str,
        domain_description: str,
        goal_state: str,
        topics: list[str],
        num_turns: int = 50,
        model: str = DEFAULT_MODEL,
        api_key: str = None,
        base_url: str = DEFAULT_BASE_URL,
    ):
        """
        Initialize the dynamic user simulator.

        Args:
            domain: Domain key (e.g., "pet_care").
            domain_description: Human-readable domain description.
            goal_state: Hidden goal that drives user behavior.
            topics: List of topics the user should cover.
            num_turns: Target number of turns.
            model: DeepSeek model to use.
            api_key: DeepSeek API key (or from DEEPSEEK_API_KEY env var).
            base_url: DeepSeek API base URL.
        """
        self.domain = domain
        self.goal_state = goal_state
        self.topics = topics
        self.num_turns = num_turns
        self.model = model
        self.current_turn = 0

        # Format system prompt with goal
        self.system_prompt = self.SYSTEM_PROMPT.format(
            goal_state=goal_state,
            domain_description=domain_description,
            topics=", ".join(topics),
            num_turns=num_turns,
        )

        # Initialize API client
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY env var "
                "or pass api_key parameter."
            )

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def generate_first_message(self) -> str:
        """Generate the opening user message to start the conversation."""
        self.current_turn = 1

        prompt = (
            f"Generate the FIRST message from a user who wants help with "
            f"{self.domain}. Their goal: {self.goal_state}. "
            f"Start with a natural greeting and initial question. "
            f"Keep it casual and realistic (1-2 sentences)."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return f"Hi, I need help with {self.goal_state}. Can you assist?"

    def generate_next_message(
        self,
        conversation_history: list[dict],
    ) -> str:
        """
        Generate the next user message based on the conversation so far.

        Args:
            conversation_history: List of {"role": ..., "content": ...} dicts
                containing the full conversation including the test model's
                actual responses.

        Returns:
            The next user message as a string.
        """
        self.current_turn += 1

        # Build the messages for DeepSeek
        # System prompt + conversation history + instruction to generate next msg
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history (keep last 20 messages to stay in context)
        recent = conversation_history[-20:] if len(conversation_history) > 20 else conversation_history
        for msg in recent:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

        # Add instruction to generate the next user message
        messages.append({
            "role": "user",
            "content": (
                f"[INSTRUCTION: You are the USER, not the assistant. "
                f"This is turn {self.current_turn} of {self.num_turns}. "
                f"Based on the assistant's last response above, generate your "
                f"next natural message. Remember your hidden goal and the "
                f"available topics. Generate ONLY the message text.]"
            ),
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"DeepSeek API error on turn {self.current_turn}: {e}")
            # Fallback: pick a random topic
            import random
            topic = random.choice(self.topics)
            return f"Moving on — what about {topic}?"

    @classmethod
    def from_domain_template(
        cls,
        domain: str,
        num_turns: int = 50,
        **kwargs,
    ) -> "DynamicUserSimulator":
        """
        Create a DynamicUserSimulator from a domain template.

        Args:
            domain: Domain key from DOMAIN_TEMPLATES.
            num_turns: Target number of turns.
            **kwargs: Additional arguments passed to __init__.
        """
        from src.data_gen.seed_generator import DOMAIN_TEMPLATES

        template = DOMAIN_TEMPLATES[domain]
        goal_state = (
            f"You are {template['user_persona']}. Ask questions about "
            f"{template['description']}."
        )

        return cls(
            domain=domain,
            domain_description=template["description"],
            goal_state=goal_state,
            topics=template["topics"],
            num_turns=num_turns,
            **kwargs,
        )
