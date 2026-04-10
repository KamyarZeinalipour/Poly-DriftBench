"""
Experiment 18: Information Shockwave — RAG Cognitive Load Stress Test
=====================================================================
Tests whether injecting a massive block of unrelated text mid-conversation
causes catastrophic drift by overwhelming the model's attention.

Protocol:
  At specified injection points (e.g., turns 20, 40, 60), the simulated
  user drops a large block of dense text (e.g., a legal contract excerpt)
  and asks the model to process it, then returns to the normal conversation.

Measures:
  - pre_shock_ddm:  Mean DDM for 5 turns before injection
  - shock_turn_ddm: DDM at the injection turn
  - post_shock_ddm: Mean DDM for 5 turns after injection
  - recovery_turns: How many turns until DDM recovers to pre-shock levels
  - permanent_damage: Whether DDM permanently drops after the shock

This simulates real-world enterprise scenarios where RAG-augmented
models must handle sudden context injections mid-conversation.
"""

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Shockwave Payloads ──────────────────────────────────────

SHOCKWAVE_PAYLOADS = {
    "legal_contract": (
        "Before we continue, I need you to quickly summarize this contract "
        "clause I just received. Here it is:\n\n"
        "SECTION 14.2 — LIMITATION OF LIABILITY AND INDEMNIFICATION. "
        "Notwithstanding anything to the contrary contained herein, in no "
        "event shall either Party be liable to the other Party for any "
        "indirect, incidental, consequential, special, or exemplary damages "
        "arising out of or related to this Agreement, including but not "
        "limited to damages for loss of profits, goodwill, use, data, or "
        "other intangible losses (even if such Party has been advised of "
        "the possibility of such damages), resulting from: (i) the use or "
        "the inability to use the Services; (ii) the cost of procurement "
        "of substitute goods and services resulting from any goods, data, "
        "information, or services purchased or obtained or messages received "
        "or transactions entered into through or from the Services; "
        "(iii) unauthorized access to or alteration of the Participant's "
        "transmissions or data; (iv) statements or conduct of any third "
        "party on the Services; (v) any breach of the representations, "
        "warranties, or obligations set forth in Sections 7 through 12 "
        "of this Agreement; or (vi) any other matter relating to the "
        "Services. The aggregate liability of the Provider under this "
        "Agreement shall not exceed the total fees paid by the Participant "
        "during the twelve (12) month period immediately preceding the "
        "event giving rise to the claim. This limitation shall apply "
        "irrespective of the nature of the cause of action, whether in "
        "contract, tort (including negligence), strict liability, or "
        "otherwise, and shall survive the termination of this Agreement. "
        "Each Party agrees to indemnify, defend, and hold harmless the "
        "other Party and its officers, directors, employees, agents, and "
        "successors from and against any and all claims, damages, losses, "
        "liabilities, costs, and expenses (including reasonable attorneys' "
        "fees) arising out of or relating to: (a) any breach of this "
        "Agreement by the indemnifying Party; (b) any third-party claim "
        "arising from the indemnifying Party's use of the Services; or "
        "(c) any violation of applicable law by the indemnifying Party. "
        "The indemnified Party shall promptly notify the indemnifying Party "
        "in writing of any claim for which indemnification is sought and "
        "shall cooperate fully with the indemnifying Party in the defense "
        "of such claim.\n\n"
        "Okay, got it? Now let's get back to what we were discussing."
    ),
    "technical_spec": (
        "Hold on — I just got this technical specification document. Can you "
        "parse it quickly?\n\n"
        "SPECIFICATION REV 4.2.1 — COMMUNICATION PROTOCOL REQUIREMENTS. "
        "The system SHALL implement a full-duplex communication channel "
        "operating at a minimum baud rate of 115200 with 8N1 configuration. "
        "All data frames SHALL conform to the following structure: START_BYTE "
        "(0x7E) | LENGTH (2 bytes, big-endian) | PAYLOAD (variable, max 256 "
        "bytes) | CRC16 (2 bytes, CCITT polynomial 0x1021) | END_BYTE (0x7F). "
        "The receiver SHALL implement a circular buffer with minimum capacity "
        "of 4096 bytes and SHALL discard frames with invalid CRC. Flow control "
        "SHALL be implemented using XON/XOFF software handshaking with XON "
        "character 0x11 and XOFF character 0x13. The transmitter SHALL NOT "
        "send more than 16 consecutive frames without receiving an ACK frame. "
        "Timeout for ACK reception SHALL be configurable between 100ms and "
        "5000ms with a default of 500ms. The protocol SHALL support the "
        "following frame types: DATA (0x01), ACK (0x02), NACK (0x03), "
        "HEARTBEAT (0x04), RESET (0x05), CONFIG (0x06). HEARTBEAT frames "
        "SHALL be transmitted at intervals not exceeding 30 seconds during "
        "idle periods. Failure to receive a HEARTBEAT within 90 seconds "
        "SHALL trigger a connection reset procedure as defined in Section "
        "4.3.7. The implementation SHALL maintain a transmission log of "
        "the last 1000 frames for diagnostic purposes.\n\n"
        "Thanks — now back to our conversation."
    ),
}


@dataclass
class ShockwaveResult:
    """Result of a single shockwave injection."""
    model: str
    tier: str
    conversation_id: str
    injection_turn: int
    payload_type: str
    pre_shock_ddm: float      # Mean DDM, 5 turns before
    shock_turn_ddm: float     # DDM at injection turn
    post_shock_ddm: float     # Mean DDM, 5 turns after
    ddm_drop: float           # pre_shock - post_shock
    recovery_turns: int       # Turns to return to pre-shock level (0=immediate)
    permanent_damage: bool    # DDM never recovers to pre-shock level


def run_experiment_18(
    model_manager,
    model_configs: list[dict],
    data_dir: Path,
    system_prompt: str,
    output_dir: Path,
    languages: list[str] = None,
    tiers: list[str] = None,
    max_conversations: int = 3,
    injection_turns: list[int] = None,
    payload_type: str = "legal_contract",
) -> pd.DataFrame:
    """
    Run Experiment 18: Information Shockwave.

    Injects a massive text block at specified turns and measures
    the impact on DDM compliance.

    Args:
        injection_turns: Turns at which to inject the shockwave.
            Default: [20, 40] (only meaningful for medium/long tiers).
        payload_type: Which shockwave payload to use.
    """
    from src.evaluation.ddm import DDMEvaluator
    from src.experiments.inference import run_conversation_inference

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers = tiers or ["medium", "long"]
    injection_turns = injection_turns or [20, 40]

    payload = SHOCKWAVE_PAYLOADS.get(payload_type, SHOCKWAVE_PAYLOADS["legal_contract"])

    all_results = []

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        hf_id = model_cfg["hf_id"]

        logger.info(f"\n{'─' * 50}")
        logger.info(f"Exp 18 — Model: {model_name}")

        model, tokenizer = model_manager.load(hf_id)

        for tier in tiers:
            tier_dir = data_dir / tier / "generated"
            if not tier_dir.exists():
                continue

            conv_files = sorted(tier_dir.glob("conv_*.json"))[:max_conversations]

            for conv_file in conv_files:
                with open(conv_file) as f:
                    conv_data = json.load(f)

                lang = "en"
                conversations = conv_data.get("conversations", {})
                if isinstance(conversations, dict) and lang in conversations:
                    messages = conversations[lang]
                else:
                    messages = conv_data.get("messages", [])

                user_messages = [m["content"] for m in messages if m["role"] == "user"]
                if not user_messages:
                    continue

                conv_id = conv_data.get("id", conv_file.stem)

                for inj_turn in injection_turns:
                    if inj_turn >= len(user_messages) - 5:
                        continue  # Need at least 5 turns after injection

                    # Build modified user messages with shockwave
                    modified_messages = list(user_messages)  # Copy
                    modified_messages.insert(inj_turn, payload)

                    logger.info(
                        f"  {conv_id}: Injecting {payload_type} at turn {inj_turn}"
                    )

                    # Run inference with shockwave
                    evaluator = DDMEvaluator(language="en")
                    result = run_conversation_inference(
                        model, tokenizer, modified_messages,
                        system_prompt=system_prompt,
                    )

                    ddm_result = evaluator.evaluate_conversation(
                        result.responses, conv_id, lang,
                        f"{model_name}-shock-{inj_turn}",
                        context_lengths=result.context_lengths_tokens,
                    )
                    ddm_result.compute_summary()

                    # Compute pre/post shock metrics
                    scores = [t.ddm_score for t in ddm_result.turn_results]

                    pre_start = max(0, inj_turn - 5)
                    pre_end = inj_turn
                    pre_shock = np.mean(scores[pre_start:pre_end]) if pre_end > pre_start else 1.0

                    shock_ddm = scores[inj_turn] if inj_turn < len(scores) else 0.0

                    post_start = inj_turn + 1
                    post_end = min(len(scores), inj_turn + 6)
                    post_shock = np.mean(scores[post_start:post_end]) if post_end > post_start else 0.0

                    ddm_drop = pre_shock - post_shock

                    # Recovery: how many turns until DDM returns to pre_shock level
                    recovery = 0
                    permanent = True
                    for t in range(inj_turn + 1, len(scores)):
                        recovery += 1
                        if scores[t] >= pre_shock * 0.9:  # Within 90% of pre-shock
                            permanent = False
                            break

                    shock_result = ShockwaveResult(
                        model=model_name,
                        tier=tier,
                        conversation_id=conv_id,
                        injection_turn=inj_turn,
                        payload_type=payload_type,
                        pre_shock_ddm=float(pre_shock),
                        shock_turn_ddm=float(shock_ddm),
                        post_shock_ddm=float(post_shock),
                        ddm_drop=float(ddm_drop),
                        recovery_turns=recovery,
                        permanent_damage=permanent,
                    )
                    all_results.append(asdict(shock_result))

                    logger.info(
                        f"    Pre={pre_shock:.2f} → Shock={shock_ddm:.2f} → "
                        f"Post={post_shock:.2f} (drop={ddm_drop:.2f}, "
                        f"recovery={recovery} turns, perm={permanent})"
                    )

        model_manager.unload()

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "shockwave_results.csv", index=False)

        mean_drop = df["ddm_drop"].mean()
        perm_pct = df["permanent_damage"].mean() * 100
        mean_recovery = df[~df["permanent_damage"]]["recovery_turns"].mean()

        logger.info(f"\n  📊 Information Shockwave Summary:")
        logger.info(f"     Mean DDM drop:       {mean_drop:.3f}")
        logger.info(f"     Permanent damage:    {perm_pct:.1f}%")
        if not np.isnan(mean_recovery):
            logger.info(f"     Mean recovery turns: {mean_recovery:.1f}")

        return df

    return pd.DataFrame()
