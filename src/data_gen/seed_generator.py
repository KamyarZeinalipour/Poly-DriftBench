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
    "it_troubleshooting": {
        "description": "IT support for enterprise software issues",
        "user_persona": "A non-technical employee experiencing email and VPN issues",
        "topics": [
            "email sync failures", "VPN connection drops", "printer setup",
            "password resets", "software installation", "network latency",
            "file permission errors", "calendar sync issues",
        ],
    },
    "legal_document_review": {
        "description": "Legal assistant reviewing contracts and agreements",
        "user_persona": "A small business owner seeking contract review help",
        "topics": [
            "NDA clauses", "liability limitations", "payment terms",
            "intellectual property rights", "termination conditions",
            "force majeure", "confidentiality scope", "dispute resolution",
        ],
    },
    "customer_support": {
        "description": "E-commerce customer support handling orders and returns",
        "user_persona": "A frustrated customer with delivery and product issues",
        "topics": [
            "missing packages", "wrong item received", "refund status",
            "size exchange", "warranty claims", "billing errors",
            "loyalty points", "subscription cancellation",
        ],
    },
    "travel_planning": {
        "description": "Travel agent assisting with trip planning",
        "user_persona": "A couple planning a two-week European vacation",
        "topics": [
            "flight bookings", "hotel recommendations", "visa requirements",
            "travel insurance", "local transportation", "restaurant reservations",
            "activity planning", "budget tracking",
        ],
    },
    "medical_consultation": {
        "description": "Medical information assistant (non-diagnostic)",
        "user_persona": "A patient seeking information about managing a chronic condition",
        "topics": [
            "medication side effects", "dietary guidelines", "exercise recommendations",
            "symptom tracking", "appointment scheduling", "test result interpretation",
            "insurance coverage", "specialist referrals",
        ],
    },
    "financial_advisory": {
        "description": "Financial planning assistant for personal finance",
        "user_persona": "A young professional starting to invest",
        "topics": [
            "budget planning", "retirement accounts", "index funds vs ETFs",
            "tax optimization", "emergency fund", "debt repayment strategy",
            "real estate investment", "risk assessment",
        ],
    },
    "academic_tutoring": {
        "description": "Academic tutor helping with university coursework",
        "user_persona": "An undergraduate studying for exams",
        "topics": [
            "concept explanation", "problem solving", "study strategies",
            "essay structuring", "citation formatting", "exam preparation",
            "research methodology", "statistics help",
        ],
    },
    "recipe_instruction": {
        "description": "Cooking assistant guiding through complex recipes",
        "user_persona": "A home cook attempting advanced dishes",
        "topics": [
            "ingredient substitutions", "cooking techniques", "timing coordination",
            "plating presentation", "dietary modifications", "batch cooking",
            "kitchen equipment", "flavor pairing",
        ],
    },
    "real_estate": {
        "description": "Real estate assistant helping with property search",
        "user_persona": "A first-time homebuyer navigating the market",
        "topics": [
            "neighborhood research", "mortgage pre-approval", "home inspection",
            "offer strategy", "closing costs", "property tax",
            "renovation estimates", "market trends",
        ],
    },
    "insurance_claims": {
        "description": "Insurance claims assistant for auto and home",
        "user_persona": "A policyholder filing a claim after property damage",
        "topics": [
            "claim filing process", "documentation requirements", "adjuster visits",
            "coverage limits", "deductible explanation", "repair estimates",
            "timeline expectations", "appeal process",
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
