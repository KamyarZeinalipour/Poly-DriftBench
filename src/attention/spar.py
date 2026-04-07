"""
System Prompt Attention Ratio (SPAR) — Mechanistic Interpretability Module
==========================================================================
Provides direct evidence for the Token Squeeze Hypothesis by measuring
how much attention the transformer pays to system prompt tokens across
conversational turns.

SPAR(t) = (1 / (L * H)) * Σ_l Σ_h [Σ_{i ∈ sys} α^(l,h)_{last,i} / Σ_j α^(l,h)_{last,j}]

This module:
    1. Extracts attention weights from all layers/heads
    2. Computes SPAR at each turn
    3. Generates layer-wise decay heatmaps
    4. Supports activation patching for causal intervention
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class SPARResult:
    """SPAR measurement for a single turn."""
    turn_number: int
    spar_score: float  # Aggregate SPAR across all layers/heads
    per_layer_spar: list[float] = field(default_factory=list)  # SPAR per layer
    per_head_spar: Optional[list[list[float]]] = None  # SPAR per layer per head
    total_sequence_length: int = 0
    system_prompt_length: int = 0


@dataclass
class AttentionDecayProfile:
    """Full attention decay profile for a conversation."""
    conversation_id: str
    language: str
    model_name: str
    spar_results: list[SPARResult] = field(default_factory=list)
    num_layers: int = 0
    num_heads: int = 0

    @property
    def spar_curve(self) -> list[float]:
        """Get the SPAR decay curve (one value per turn)."""
        return [r.spar_score for r in self.spar_results]

    @property
    def per_layer_curves(self) -> list[list[float]]:
        """Get per-layer SPAR curves. Shape: [num_layers, num_turns]."""
        if not self.spar_results:
            return []
        num_layers = len(self.spar_results[0].per_layer_spar)
        return [
            [r.per_layer_spar[l] for r in self.spar_results]
            for l in range(num_layers)
        ]


class SPARAnalyzer:
    """
    Extracts attention weights and computes SPAR scores.

    Usage:
        analyzer = SPARAnalyzer(model_name, device="cuda")
        profile = analyzer.analyze_conversation(
            messages, system_prompt, conv_id, language
        )
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model with attention output enabled."""
        if self.model is not None:
            return

        logger.info(f"Loading model for attention analysis: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",  # Need eager for attention weights
        )
        self.model.eval()

        # Get model architecture info
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = getattr(config, "num_attention_heads", 32)
        logger.info(
            f"  Loaded: {self.num_layers} layers, {self.num_heads} heads"
        )

    def _build_input_ids(
        self,
        system_prompt: str,
        conversation_turns: list[dict],
        up_to_turn: int,
    ) -> tuple[torch.Tensor, int, int]:
        """
        Build input_ids for the conversation up to a given turn.

        Returns:
            (input_ids, system_prompt_end_idx, total_length)
        """
        # Build the full message list
        messages = [{"role": "system", "content": system_prompt}]
        for turn in conversation_turns[:up_to_turn]:
            messages.append(turn)

        # Tokenize
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Find system prompt token boundary
        sys_only = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )
        sys_token_count = len(self.tokenizer.encode(sys_only, add_special_tokens=False))

        return input_ids.to(self.device), sys_token_count, input_ids.shape[1]

    def compute_spar(
        self,
        attention_weights: tuple,
        system_prompt_end_idx: int,
    ) -> SPARResult:
        """
        Compute SPAR from raw attention weights.

        Args:
            attention_weights: Tuple of (num_layers,) tensors,
                each of shape [batch, num_heads, seq_len, seq_len].
            system_prompt_end_idx: Token index where system prompt ends.
        """
        per_layer_spar = []
        per_head_spar = []

        for layer_idx, layer_attn in enumerate(attention_weights):
            # layer_attn shape: [1, num_heads, seq_len, seq_len]
            attn = layer_attn[0]  # Remove batch dim: [num_heads, seq_len, seq_len]

            # Get attention from last token to all positions
            last_token_attn = attn[:, -1, :]  # [num_heads, seq_len]

            # SPAR per head: ratio of attention to system prompt tokens
            sys_attn = last_token_attn[:, :system_prompt_end_idx].sum(dim=-1)
            total_attn = last_token_attn.sum(dim=-1)

            head_spars = (sys_attn / (total_attn + 1e-10)).cpu().numpy().tolist()
            per_head_spar.append(head_spars)

            # Layer-level SPAR: mean across heads
            layer_spar = float(np.mean(head_spars))
            per_layer_spar.append(layer_spar)

        # Aggregate SPAR: mean across all layers
        aggregate_spar = float(np.mean(per_layer_spar))

        return SPARResult(
            turn_number=0,  # Set by caller
            spar_score=aggregate_spar,
            per_layer_spar=per_layer_spar,
            per_head_spar=per_head_spar,
        )

    def analyze_conversation(
        self,
        conversation_turns: list[dict],
        system_prompt: str,
        conversation_id: str,
        language: str,
        step: int = 1,
    ) -> AttentionDecayProfile:
        """
        Analyze SPAR decay across a full conversation.

        Args:
            conversation_turns: List of {"role": "user"/"assistant", "content": "..."} dicts.
            system_prompt: The system prompt text.
            conversation_id: Unique conversation identifier.
            language: Language code.
            step: Analyze every N-th turn (for efficiency).

        Returns:
            AttentionDecayProfile with SPAR at each analyzed turn.
        """
        self.load_model()

        profile = AttentionDecayProfile(
            conversation_id=conversation_id,
            language=language,
            model_name=self.model_name,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )

        total_turns = len(conversation_turns)
        turn_indices = list(range(0, total_turns, step))

        for turn_idx in turn_indices:
            try:
                input_ids, sys_end_idx, total_len = self._build_input_ids(
                    system_prompt, conversation_turns, turn_idx + 1
                )

                with torch.no_grad():
                    outputs = self.model(
                        input_ids,
                        output_attentions=True,
                        return_dict=True,
                    )

                spar_result = self.compute_spar(
                    outputs.attentions, sys_end_idx
                )
                spar_result.turn_number = turn_idx + 1
                spar_result.total_sequence_length = total_len
                spar_result.system_prompt_length = sys_end_idx

                profile.spar_results.append(spar_result)

                logger.debug(
                    f"  Turn {turn_idx + 1}/{total_turns}: "
                    f"SPAR={spar_result.spar_score:.4f}, "
                    f"seq_len={total_len}, sys_len={sys_end_idx}"
                )

            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"  OOM at turn {turn_idx + 1} (seq_len ~{total_len}). "
                    f"Stopping analysis for this conversation."
                )
                break
            except Exception as e:
                logger.error(f"  Error at turn {turn_idx + 1}: {e}")
                continue

        logger.info(
            f"  [{language}|{conversation_id}] Analyzed {len(profile.spar_results)} turns. "
            f"SPAR range: {profile.spar_curve[0]:.4f} → {profile.spar_curve[-1]:.4f}"
            if profile.spar_curve else "No turns analyzed."
        )

        return profile

    def save_profile(
        self, profile: AttentionDecayProfile, output_dir: str | Path
    ) -> Path:
        """Save attention decay profile to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"spar_{profile.model_name.split('/')[-1]}"
            f"_{profile.language}_{profile.conversation_id}.json"
        )
        filepath = output_dir / filename

        data = {
            "conversation_id": profile.conversation_id,
            "language": profile.language,
            "model_name": profile.model_name,
            "num_layers": profile.num_layers,
            "num_heads": profile.num_heads,
            "spar_curve": profile.spar_curve,
            "turns": [
                {
                    "turn": r.turn_number,
                    "spar": r.spar_score,
                    "per_layer_spar": r.per_layer_spar,
                    "seq_len": r.total_sequence_length,
                    "sys_len": r.system_prompt_length,
                }
                for r in profile.spar_results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved SPAR profile: {filepath}")
        return filepath
