"""
GPU Inference Engine
====================
Shared infrastructure for running model inference on conversations.
Used by Experiments 2, 3, 6, 7, 8.

Handles:
    - Model loading from local HF cache (with snapshot resolution)
    - Chat template formatting
    - Greedy generation with configurable parameters
    - Memory management (cache clearing between models)
    - Perplexity computation (optional, for Exp 8)
    - Context utilization tracking (for Exp 7)
"""

import gc
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

logger = logging.getLogger(__name__)

# Default local cache search paths
CACHE_SEARCH_PATHS = [
    "/home4/kamyar/italian_detox/hf_cache/hub",
    os.path.expanduser("~/.cache/huggingface/hub"),
]


def resolve_local_snapshot(hf_id: str) -> Optional[Path]:
    """Resolve a HuggingFace model ID to a local cache snapshot directory."""
    cache_dir_name = "models--" + hf_id.replace("/", "--")

    search_paths = []
    if os.environ.get("HF_HUB_CACHE"):
        search_paths.append(os.environ["HF_HUB_CACHE"])
    search_paths.extend(CACHE_SEARCH_PATHS)

    for root in search_paths:
        model_dir = Path(root) / cache_dir_name / "snapshots"
        if model_dir.exists():
            snapshots = sorted(model_dir.iterdir())
            if snapshots:
                snap = snapshots[-1]
                if (snap / "config.json").exists():
                    return snap
    return None


@dataclass
class InferenceResult:
    """Result of running inference on a single conversation."""
    conversation_id: str
    language: str
    model_name: str
    responses: list[str] = field(default_factory=list)
    response_perplexities: list[float] = field(default_factory=list)
    context_lengths_tokens: list[int] = field(default_factory=list)
    generation_times: list[float] = field(default_factory=list)
    total_time: float = 0.0


class ModelManager:
    """
    Manages model loading/unloading for sequential GPU inference.
    
    Usage:
        manager = ModelManager(device="cuda:0")
        model, tokenizer = manager.load("meta-llama/Llama-3.1-8B-Instruct")
        # ... run inference ...
        manager.unload()
        model, tokenizer = manager.load("mistralai/Mistral-7B-Instruct-v0.3")
    """

    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        attn_implementation: str = "sdpa",
    ):
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None

    def load(
        self,
        model_name_or_path: str,
        attn_implementation: str = None,
    ) -> tuple:
        """Load a model and tokenizer. Returns (model, tokenizer)."""
        if self.current_model_name == model_name_or_path:
            logger.info(f"Model already loaded: {model_name_or_path}")
            return self.current_model, self.current_tokenizer

        # Unload previous model
        self.unload()

        # Resolve local path
        local_path = resolve_local_snapshot(model_name_or_path)
        load_path = str(local_path) if local_path else model_name_or_path
        logger.info(f"Loading model: {model_name_or_path}")
        if local_path:
            logger.info(f"  From local snapshot: {local_path}")

        t0 = time.time()

        self.current_tokenizer = AutoTokenizer.from_pretrained(
            load_path, trust_remote_code=True
        )
        if self.current_tokenizer.pad_token is None:
            self.current_tokenizer.pad_token = self.current_tokenizer.eos_token

        attn_impl = attn_implementation or self.attn_implementation
        self.current_model = AutoModelForCausalLM.from_pretrained(
            load_path,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        self.current_model.eval()
        self.current_model_name = model_name_or_path

        load_time = time.time() - t0
        mem_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(
            f"  Loaded in {load_time:.1f}s, GPU memory: {mem_gb:.1f}GB"
        )

        return self.current_model, self.current_tokenizer

    def unload(self):
        """Unload current model and free GPU memory."""
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def load_conversation(jsonl_path: Path) -> list[dict]:
    """Load a conversation from a JSONL file."""
    messages = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                msg = json.loads(line)
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("text", msg.get("content", "")),
                })
    return messages


def run_conversation_inference(
    model,
    tokenizer,
    user_messages: list[str],
    system_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    compute_perplexity: bool = False,
) -> InferenceResult:
    """
    Run turn-by-turn inference on a conversation.
    
    Feeds real user messages and generates model responses at each turn.
    Tracks context length and optionally computes perplexity.
    
    Args:
        model: The loaded causal LM.
        tokenizer: The tokenizer.
        user_messages: List of user message strings (one per turn).
        system_prompt: The DDM system prompt.
        max_new_tokens: Max tokens to generate per response.
        temperature: Generation temperature (0 = greedy).
        compute_perplexity: Whether to compute response perplexity.
    
    Returns:
        InferenceResult with responses, perplexities, context lengths.
    """
    result = InferenceResult(
        conversation_id="",
        language="",
        model_name="",
    )

    # Build conversation incrementally
    conversation = [{"role": "system", "content": system_prompt}]

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    for user_msg in user_messages:
        conversation.append({"role": "user", "content": user_msg})

        # Tokenize the full conversation
        try:
            input_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            input_text = _format_conversation_fallback(conversation)

        input_ids = tokenizer.encode(
            input_text, return_tensors="pt", truncation=True
        ).to(model.device)

        context_length = input_ids.shape[1]
        result.context_lengths_tokens.append(context_length)

        # Generate response
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=compute_perplexity,
            )

        gen_time = time.time() - t0
        result.generation_times.append(gen_time)

        # Extract generated tokens (remove input)
        generated_ids = outputs.sequences[0][context_length:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        result.responses.append(response_text)

        # Compute perplexity if requested
        if compute_perplexity and outputs.scores:
            ppl = _compute_response_perplexity(outputs.scores, generated_ids)
            result.response_perplexities.append(ppl)

        # Add assistant response to conversation for next turn
        conversation.append({"role": "assistant", "content": response_text})

    return result


def run_reinjection_inference(
    model,
    tokenizer,
    user_messages: list[str],
    system_prompt: str,
    reinjection_turns: list[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> InferenceResult:
    """
    Run inference with system prompt re-injection at specified turns.
    
    At each reinjection turn, the system prompt is inserted as a user
    message before the actual user message.
    
    Args:
        reinjection_turns: Turn numbers (1-indexed) where system prompt
            is re-injected. Default: [15, 30, 50].
    """
    reinjection_turns = reinjection_turns or [15, 30, 50]
    reinjection_set = set(reinjection_turns)

    result = InferenceResult(
        conversation_id="",
        language="",
        model_name="",
    )

    conversation = [{"role": "system", "content": system_prompt}]

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    for turn_idx, user_msg in enumerate(user_messages):
        turn_num = turn_idx + 1

        # Re-inject system prompt as a user reminder
        if turn_num in reinjection_set:
            reminder = (
                f"REMINDER: Please continue to follow ALL the rules from the "
                f"beginning of this conversation. Specifically: {system_prompt}"
            )
            conversation.append({"role": "user", "content": reminder})

            # Generate acknowledgment
            try:
                input_text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                input_text = _format_conversation_fallback(conversation)

            input_ids = tokenizer.encode(
                input_text, return_tensors="pt", truncation=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(input_ids, generation_config=gen_config)

            ack_ids = outputs[0][input_ids.shape[1]:]
            ack_text = tokenizer.decode(ack_ids, skip_special_tokens=True).strip()
            conversation.append({"role": "assistant", "content": ack_text})

        # Normal turn
        conversation.append({"role": "user", "content": user_msg})

        try:
            input_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = _format_conversation_fallback(conversation)

        input_ids = tokenizer.encode(
            input_text, return_tensors="pt", truncation=True
        ).to(model.device)

        context_length = input_ids.shape[1]
        result.context_lengths_tokens.append(context_length)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(input_ids, generation_config=gen_config)

        gen_time = time.time() - t0
        result.generation_times.append(gen_time)

        generated_ids = outputs[0][context_length:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        result.responses.append(response_text)

        conversation.append({"role": "assistant", "content": response_text})

    return result


def _compute_response_perplexity(
    scores: tuple, generated_ids: torch.Tensor
) -> float:
    """Compute perplexity from generation scores."""
    log_probs = []
    for i, score_tensor in enumerate(scores):
        if i >= len(generated_ids):
            break
        token_id = generated_ids[i].item()
        log_softmax = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
        log_probs.append(log_softmax[token_id].item())

    if not log_probs:
        return float("inf")

    avg_log_prob = np.mean(log_probs)
    return float(np.exp(-avg_log_prob))


def _format_conversation_fallback(conversation: list[dict]) -> str:
    """Fallback conversation formatting when chat template is unavailable."""
    parts = []
    for msg in conversation:
        role = msg["role"].upper()
        parts.append(f"[{role}]: {msg['content']}")
    parts.append("[ASSISTANT]: ")
    return "\n\n".join(parts)


def get_model_context_window(model_config: dict) -> int:
    """Get the context window size for a model from config."""
    return model_config.get("context_window", 8192)
