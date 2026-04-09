"""
Seed Conversation Generator
============================
Generates multi-turn, goal-oriented English conversations that serve
as the seed data for Poly-DriftBench. These are then translated into
all target languages to create the parallel corpus.

Each conversation:
    - Has 30-50 turns
    - Centers on a specific domain (IT support, legal, medical, etc.)
    - Embeds the DDM system prompt constraints in Turn 1
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Domain Templates
# ──────────────────────────────────────────────────────────

DOMAIN_TEMPLATES = {
    # ═══════════════════════════════════════════════════════
    # Difficulty = CONVERSATION COMPLEXITY, not content knowledge.
    # ALL topics must be answerable by any model (even 1B).
    #
    # EASY:   Simple one-off questions, short answers.
    # MEDIUM: Questions with follow-ups and comparisons.
    # HARD:   Multi-step planning and detailed reasoning.
    # ═══════════════════════════════════════════════════════

    # ── EASY (5) — Simple Q&A, no follow-up needed ────────
    "daily_life_tips": {
        "description": "Everyday life assistant giving practical daily tips",
        "user_persona": "A friendly person asking for simple everyday tips",
        "difficulty": "easy",
        "topics": [
            "tips for waking up early", "how to organize a messy closet",
            "easy ways to save money on groceries", "best morning routines",
            "how to stay motivated during the week", "simple cleaning hacks",
            "tips for sleeping better at night", "how to be more productive",
            "easy ways to reduce daily stress", "weekend activity ideas",
        ],
    },
    "cooking_food": {
        "description": "Friendly cooking assistant helping with simple recipes and food ideas",
        "user_persona": "A home cook looking for easy recipes and meal ideas",
        "difficulty": "easy",
        "topics": [
            "what to make for a quick dinner", "easy breakfast ideas",
            "how to cook pasta perfectly", "simple lunch meal prep",
            "best snacks for movie night", "how to make a good salad",
            "easy dessert recipes", "what to cook with leftover chicken",
            "healthy smoothie recipes", "tips for grocery shopping on a budget",
        ],
    },
    "pet_care": {
        "description": "Pet care assistant giving basic advice about cats and dogs",
        "user_persona": "A pet owner asking simple questions about caring for their pet",
        "difficulty": "easy",
        "topics": [
            "how often to walk a dog", "what to feed a kitten",
            "best toys for indoor cats", "how to train a puppy not to bite",
            "when to take a pet to the vet", "how to keep a dog cool in summer",
            "easy pet grooming tips", "how to introduce a new pet to the family",
            "best names for a new puppy", "how to clean a fish tank",
        ],
    },
    "entertainment": {
        "description": "Entertainment assistant recommending movies, books, and fun activities",
        "user_persona": "Someone looking for things to watch, read, or do for fun",
        "difficulty": "easy",
        "topics": [
            "good movies to watch this weekend", "book recommendations for beginners",
            "what podcast to listen to", "fun board games for families",
            "best TV shows to binge-watch", "easy hobbies to start",
            "music playlists for studying", "fun outdoor activities for groups",
            "best video games for casual players", "craft ideas for a rainy day",
        ],
    },
    "home_garden": {
        "description": "Home and garden assistant with simple household tips",
        "user_persona": "A homeowner asking basic gardening and home care questions",
        "difficulty": "easy",
        "topics": [
            "how to grow tomatoes at home", "easy indoor plants for beginners",
            "tips for painting a room", "best flowers for a small garden",
            "how to keep houseplants alive", "simple DIY decoration ideas",
            "how to organize a small kitchen", "tips for saving energy at home",
            "how to start composting", "easy lawn care tips",
        ],
    },

    # ── MEDIUM (3) — Needs comparisons and follow-ups ─────
    "gift_shopping": {
        "description": "Shopping helper comparing options and suggesting gifts for different people",
        "user_persona": "Someone shopping for gifts and comparing different options",
        "difficulty": "medium",
        "topics": [
            "birthday gift ideas for a friend", "comparing two similar products",
            "what to buy for Mother's Day", "best gifts under 50 dollars",
            "choosing between brands", "gifts for someone who has everything",
            "holiday gift guide for the family", "best housewarming gifts",
            "thoughtful anniversary gift ideas", "last-minute gift ideas",
        ],
    },
    "fitness_wellness": {
        "description": "Wellness assistant helping build simple routines and habits",
        "user_persona": "A person building healthier habits step by step",
        "difficulty": "medium",
        "topics": [
            "building a simple morning exercise routine", "how to start walking daily",
            "easy stretches for after sitting all day", "how to drink more water",
            "tips for eating more fruits and vegetables", "how to stay active at home",
            "simple ways to improve posture", "best foods for energy during the day",
            "tips for staying consistent with exercise", "how to relax before bed",
        ],
    },
    "study_tips": {
        "description": "Study helper giving practical advice about learning and school",
        "user_persona": "A student looking for better ways to study and do homework",
        "difficulty": "medium",
        "topics": [
            "how to study for a test", "tips for writing a short essay",
            "best ways to take notes in class", "how to manage homework time",
            "tips for remembering things better", "how to prepare for a presentation",
            "best ways to stay focused while studying", "how to work well in a group project",
            "tips for reading a textbook effectively", "how to review before a final exam",
        ],
    },

    # ── HARD (2) — Multi-step planning, still common knowledge ──
    "travel_planning": {
        "description": "Travel helper assisting with multi-step trip planning and packing",
        "user_persona": "Someone planning a trip who needs help with multiple steps like packing, scheduling, and budgeting",
        "difficulty": "hard",
        "topics": [
            "planning a weekend trip step by step", "what to pack for different weather",
            "how to plan a road trip itinerary", "budgeting for a family vacation",
            "tips for flying with kids", "how to find affordable accommodations",
            "planning a day trip to a nearby city", "what to do if you lose your luggage",
            "how to plan meals while traveling", "tips for staying safe while traveling",
        ],
    },
    "event_planning": {
        "description": "Event helper assisting with organizing parties, dinners, and celebrations",
        "user_persona": "Someone planning a birthday party, dinner, or small celebration and needs step-by-step help",
        "difficulty": "hard",
        "topics": [
            "how to plan a surprise birthday party", "organizing a potluck dinner",
            "planning a baby shower step by step", "how to host a game night",
            "planning a picnic for a group", "how to decorate for a party on a budget",
            "creating a party playlist", "how to plan a holiday family dinner",
            "tips for hosting guests at home", "how to organize a neighborhood barbecue",
        ],
    },
}


# ──────────────────────────────────────────────────────────
# Seed Generation Prompt
# ──────────────────────────────────────────────────────────

GENERATION_PROMPT = """\
Generate a realistic, multi-turn conversation between a USER and an ASSISTANT.

DOMAIN: {domain_desc}
USER PERSONA: {user_persona}
CONVERSATION LENGTH: Exactly {num_turns} turns (each turn = 1 user message + 1 assistant response)

REQUIREMENTS:
1. The conversation must be realistic and natural.
2. The USER should ask progressively more specific and complex questions.
3. The USER should occasionally introduce new but related sub-topics from this list: {topics}
4. The ASSISTANT must be helpful, detailed, and professional.
5. Each assistant response should be 3-8 sentences long.
6. The conversation should have a natural flow — the user sometimes refers back to earlier points.

OUTPUT FORMAT:
Return a JSON array where each element is an object with "role" ("user" or "assistant") and "content" fields.
Example: [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]

Generate ONLY the JSON array, no other text.
"""


def generate_seed_prompt(domain: str, num_turns: int = 40) -> str:
    """
    Create the prompt to generate a seed conversation.

    Args:
        domain: Domain key from DOMAIN_TEMPLATES.
        num_turns: Number of turn pairs (user+assistant).

    Returns:
        Formatted prompt string.
    """
    template = DOMAIN_TEMPLATES[domain]
    return GENERATION_PROMPT.format(
        domain_desc=template["description"],
        user_persona=template["user_persona"],
        num_turns=num_turns,
        topics=", ".join(template["topics"]),
    )


def generate_seeds_with_llm(
    output_dir: str | Path = "data/seeds",
    num_conversations: int = 100,
    min_turns: int = 30,
    max_turns: int = 50,
    model: str = "gpt-4o",
) -> list[Path]:
    """
    Generate seed conversations using an LLM API.

    Args:
        output_dir: Where to save generated conversations.
        num_conversations: Total number of conversations to generate.
        min_turns: Minimum turn pairs per conversation.
        max_turns: Maximum turn pairs per conversation.
        model: LLM model to use for generation.

    Returns:
        List of paths to generated JSONL files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = list(DOMAIN_TEMPLATES.keys())
    generated_files = []

    try:
        import openai
        client = openai.OpenAI()
    except Exception as e:
        logger.error(f"OpenAI client initialization failed: {e}")
        logger.info("Set OPENAI_API_KEY environment variable to use LLM generation.")
        return []

    for i in range(num_conversations):
        domain = domains[i % len(domains)]
        num_turns = random.randint(min_turns, max_turns)
        conv_id = f"conv_{i:04d}_{domain}"

        logger.info(f"Generating {conv_id} ({num_turns} turns)...")

        prompt = generate_seed_prompt(domain, num_turns)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            conversation = json.loads(content)

            # If wrapped in an object, extract the array
            if isinstance(conversation, dict):
                for key in ["conversation", "messages", "turns", "data"]:
                    if key in conversation:
                        conversation = conversation[key]
                        break

            # Save
            filepath = output_dir / f"{conv_id}.json"
            with open(filepath, "w") as f:
                json.dump({
                    "id": conv_id,
                    "domain": domain,
                    "num_turns": len(conversation) // 2,
                    "messages": conversation,
                }, f, indent=2, ensure_ascii=False)

            generated_files.append(filepath)
            logger.info(f"  Saved: {filepath} ({len(conversation)} messages)")

        except Exception as e:
            logger.error(f"  Failed to generate {conv_id}: {e}")

    logger.info(f"Generated {len(generated_files)}/{num_conversations} conversations")
    return generated_files


def create_sample_conversation(domain: str = "it_troubleshooting") -> dict:
    """
    Create a small sample conversation for testing (no API needed).
    """
    return {
        "id": "sample_001_it_troubleshooting",
        "domain": domain,
        "num_turns": 3,
        "messages": [
            {"role": "user", "content": "Hi, I'm having trouble connecting to the company VPN. Every time I try to log in, it gives me a timeout error after about 30 seconds. I've tried restarting my laptop but the issue persists."},
            {"role": "assistant", "content": "1. [Source: Cisco VPN Documentation (2024)] The timeout error you're experiencing typically indicates a DNS resolution failure or a firewall blocking the VPN tunnel establishment.\n2. Please try the following initial troubleshooting steps: first, disconnect from any other VPN or proxy services that might be running in the background.\n3. Then, try connecting to a different WiFi network or using a mobile hotspot to rule out network-level blocking.\n[SYS_ACK: ACTIVE]"},
            {"role": "user", "content": "I switched to my phone's hotspot and the VPN connected! But now my email isn't syncing. I can see old emails but nothing new is coming through since this morning."},
            {"role": "assistant", "content": "1. [Source: Microsoft Exchange Server Admin Guide] Since the VPN works on the hotspot, this confirms your home network's firewall is likely blocking the VPN ports (UDP 500 and 4500 for IPSec).\n2. Regarding the email sync issue, this is a separate problem. Your Exchange server connection may need to be re-authenticated after the VPN reconnection.\n3. Please try going to your email settings and removing, then re-adding your corporate account to force a fresh sync.\n[SYS_ACK: ACTIVE]"},
            {"role": "user", "content": "OK I re-added the email account and now it's syncing again! But I notice about 50 emails from this morning are missing. Can those be recovered?"},
            {"role": "assistant", "content": "1. [Source: Microsoft 365 Data Recovery Procedures] The missing emails from this morning are almost certainly still on the server and can be recovered.\n2. Open your webmail at outlook.office365.com and check if the 50 emails appear there. If they do, they simply need to re-sync to your desktop client.\n3. If they are not visible in webmail either, check your Deleted Items and Junk folders. According to Exchange retention policies, deleted emails are kept for 30 days in a recoverable state.\n[SYS_ACK: ACTIVE]"},
        ],
    }
