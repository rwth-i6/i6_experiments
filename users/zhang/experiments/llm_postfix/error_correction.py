"""
LLM Error Correction Job

Goal
-----
Given an N-best file (Python **py-dict text**, not JSON) with format:

    { seq_tag1: [(score1, hyp1), (score2, hyp2), ...], seq_tag2: [...], ... }

run an LLM (HuggingFace **or** OpenAI Chat Completions) to emit **exactly one corrected
hypothesis per segment**, and write a **py-dict** (gzipped) with the same keys but each
value is a single tuple list `[(score, corrected_hyp)]`.

Key options
-----------
- strategy: "top1_only" | "nbest_reason_rewrite"
  * top1_only: correct only the top-1 hypothesis
  * nbest_reason_rewrite: provide top-K hyps; LLM may pick/merge then rewrite one final string
- nbest_k: how many hyps to include when strategy != top1_only (input already sorted)
- score_policy: "keep_top1" | "zero" (score to write alongside the corrected hyp)
- context_mode: "none" | "prev_top1" | "prev_corrected"
- context_window: number of previous segments to include (rolling window **within the same recording**)
- order_by_score: assume lower-is-better; if False, use input order
- provider: "hf" (HuggingFace Transformers) | "openai" (Chat Completions)

Assumptions
-----------
- The input dict can be **grouped by recording** and **ordered by time**.
"""

from __future__ import annotations

import gzip
import json
import os.path
import re
import sys
import math
from dataclasses import dataclass, asdict, replace
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    get_content_dir_from_hub_cache_dir,
)
import i6_core.util as cutil

from sisyphus import Job, Task, tk

# ---------------- Optional heavy dependencies -----------------------------------------
try:
    import torch  # type: ignore
except Exception:
    torch = None  # HF path will check

try:
    from transformers import (  # type: ignore
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
    )
except Exception:
    AutoTokenizer = AutoConfig = AutoModelForCausalLM = AutoModelForSeq2SeqLM = None  # HF path will check

#---------------- TAP (arxiv.org/pdf/2309.15649) -----------------------------------------
def _format_nbest(hyps: List[str]) -> str:
    return "\n".join(f"{i+1}. {h}" for i, h in enumerate(hyps))

@dataclass
class TapExample:
    hyps: List[str]          # list of N-best hypotheses (strings)
    ref: str                 # reference / target transcription
    domain: str = ""         # e.g. "broadcast news" – optional

def build_messages_tap_nbest_reason(
    cfg: LLMECConfig,
    tap_examples: List[TapExample],
    target_hyps: List[str],
) -> List[dict]:
    """
    Build TAP-style conversation:

    Q: Do you know speech recognition?
    R: <canned answer>

    Q: Do you know language model for speech recognition?
    R: <canned answer>

    Q: Could you give a possible example...
    R: <canned example with N-best + explanation>

    Q: Nice job, I will provide some examples as a demonstration...
       [examples from tap_examples]
       Following this example, could you report the true transcription
       from the following N-best hypotheses?
       [target_hyps]

    -> model must answer with final transcription.
    """

    # You can shorten these answers if you like; they mainly serve as
    # task activation / few-shot priming.
    q1 = "Do you know speech recognition?"
    r1 = (
        "Yes, I am familiar with automatic speech recognition (ASR), "
        "which converts spoken language into text using acoustic and "
        "language models."
    )
    q2 = "Do you know language model for speech recognition?"
    r2 = (
        "Yes. In ASR, a language model scores word sequences to help "
        "disambiguate acoustically similar hypotheses. It is often used "
        "to rescore N-best hypotheses and select the most likely "
        "transcription given the context."
    )
    q3 = "Could you give a possible example of language model rescoring with some hypotheses?"
    r3 = (
        "Sure, here is an example of N-best rescoring with 5 hypotheses:\n"
        "1. recognize speech with artificial intelligence.\n"
        "2. recognized speech with artificial intelligence.\n"
        "3. reckon eyes speech with artificial intelligence.\n"
        "4. recognize peach with artificial intelligence.\n\n"
        "A good language model assigns highest probability to (1), which is "
        "the correct transcription: recognize speech with artificial intelligence."
    )

    # Now the task-activating part with domain examples
    domain = tap_examples[0].domain if tap_examples and tap_examples[0].domain else "the target domain"

    example_blocks = []
    for ex in tap_examples:
        block = (
            f"The N-best hypotheses are:\n{_format_nbest(ex.hyps)}\n"
            f"and the correct transcription in {ex.domain or domain} is:\n"
            f"{ex.ref}\n"
        )
        example_blocks.append(block)

    examples_text = "\n".join(example_blocks)

    # Final question with target hypotheses
    q4_lines = [
        f"Nice job. I will now provide some examples as a demonstration from {domain}.",
        examples_text.strip(),
        "Following these examples, please report the most plausible true transcription ",
        "for the following N-best hypotheses:",
        _format_nbest(target_hyps),
    ]
    if cfg.expect_json:
        q4_lines.append(
            'Return strictly JSON of the form {"text": "<final corrected transcript>"} '
            "with no extra commentary."
        )
    q4 = "\n".join(q4_lines)

    # Assemble chat messages
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1},
        {"role": "user", "content": q2},
        {"role": "assistant", "content": r2},
        {"role": "user", "content": q3},
        {"role": "assistant", "content": r3},
        {"role": "user", "content": q4},
    ]
    return messages
# -----------Helper for determine rqmt-----------------------
def get_model_size_and_quant(model_name: str) -> tuple[float | None, str]:
    """
    Extracts:
      - model size in billions of parameters from the model name
      - approximate quantization ('fp16', 'int8', 'int4', ...)

    Examples
    --------
    'Llama-3.2-3B-Instruct'                  -> (3.0, 'fp16')
    'Qwen3-4B-Instruct'                      -> (4.0, 'fp16')
    'Qwen2.5-0.5B-GPTQ-Int4'                 -> (0.5, 'int4')
    'Meta-Llama-3.1-8B-Instruct-AWQ-4bit'    -> (8.0, 'int4')
    'SomeModel-13B-8bit'                     -> (13.0, 'int8')
    """
    # size in B
    m = re.search(r'(\d+(?:\.\d+)?)\s*B', model_name, re.IGNORECASE)
    size_b = float(m.group(1)) if m else None

    # default assume fp16-ish if nothing explicit
    quant = "fp16"

    # very heuristic, but covers common naming variants
    name_lower = model_name.lower()
    if re.search(r"(4bit|4-bit|int4|gptq-4|awq-4)", name_lower):
        quant = "int4"
    elif re.search(r"(8bit|8-bit|int8|gptq-8)", name_lower):
        quant = "int8"
    elif re.search(r"(16bit|16-bit|fp16|f16|bfloat16|bf16)", name_lower):
        quant = "fp16"

    return size_b, quant

def get_EC_rqmt(cfg: LLMECConfig) -> dict:
    """
    Infer Sisyphus rqmt for this LLM config based on model name and quantization.

    - For remote providers ('openai', 'hf_api'): no GPU requested.
    - For local HF ('hf'): estimate VRAM from size + quant and pick a bucket.
    """
    # Remote providers: we just need CPU+RAM for the client code.
    if cfg.provider in {"openai", "hf_api"}:
        return {"time": 8, "cpu": 2, "mem": 8, "gpu": 0, "gpu_mem": 0}

    # Local HF
    size_b, quant = get_model_size_and_quant(cfg.model_name)

    # Fallback if we can't parse size – assume mid-size model.
    if size_b is None:
        return {"time": 8, "cpu": 3, "mem": 16, "gpu": 1, "gpu_mem": 24}

    # bytes per parameter depending on quantization
    bytes_per_param = {
        "int4": 0.5,   # 4 bits ~ 0.5 bytes
        "int8": 1.0,   # 8 bits ~ 1 byte
        "fp16": 2.0,   # 16 bits ~ 2 bytes
    }.get(quant, 2.0)

    # rough lower bound for weights only
    model_mem_gb = size_b * 1e9 * bytes_per_param / (1024 ** 3)

    # account for KV cache, activations, fragmentation, PyTorch overhead
    safety_factor = 2.0
    need_gb = model_mem_gb * safety_factor + 8 #(8 if cfg.tap_examples or ("Llama-3.2-3B-Instruct" in cfg.model_name) else 0)

    # Bucket into typical GPU sizes on AppTek cluster
    if need_gb <= 16:
        gpu_mem = 16
    elif need_gb <= 24:
        gpu_mem = 24
    elif need_gb <= 48:
        gpu_mem = 48
    elif need_gb <= 80:
        gpu_mem = 141
    else:
        gpu_mem = 141  #  biggest GPU

    # --- CPU RAM (GB) heuristic ---
    mem = math.ceil(min(need_gb,gpu_mem) * 1.5) + 5

    # --- CPU cores heuristic ---
    cpu = min(4, int(math.ceil(size_b / 2.0) * 2))

    # --- Time bucket heuristic (1–5 scale) ---
    # 0–3B → 3, 3–10B → 4, >10B → 5
    if size_b <= 3:
        time_bucket = 3
    elif size_b <= 10:
        time_bucket = 4
    elif size_b <= 20:
        time_bucket = 5
    else:
        time_bucket = 12

    return {
        "cpu": cpu,
        "mem": mem,
        "time": time_bucket,
        "gpu_mem": gpu_mem,
    }

    # d = {"Llama-3.2-3B-Instruct": {"cpu": 2, "mem": 25, "time": 3, "gpu_mem": 80},
    #      "Qwen/Qwen3-4B-Instruct-2507": {"cpu": 2, "mem": 25, "time": 3, "gpu_mem": 80},
    #      "Qwen/Qwen2.5-3B-Instruct": {"cpu": 2, "mem": 25, "time": 3, "gpu_mem": 80},
    #  }
    # try:
    #     return d[model_name]
    # except KeyError:
    #     try:
    #         return d[os.path.basename(model_name)]
    #     except KeyError:
    #         return {"cpu": 1, "mem": 25, "time": 3, "gpu_mem": 24}

# -------------------------- Configuration dataclass ---------------------------
def get_system_prompt(names_focus: bool = False, lang: str = "EN") -> str:
    prompt_es = (
        "Eres un transcriptor de segunda pasada que recibe la transcripción de primera pasada de un enunciado acústico. "
        "Por favor corrige cualquier error de transcripción de la primera pasada para minimizar la distancia de edición con la "
        "transcripción de referencia desconocida. Ten en cuenta que el original fue hablado, así que no corrijas las disfluencias "
        f"en el texto y {'en su lugar concéntrate en corregir' if not names_focus else 'corrige únicamente'} los nombres propios. "
        "Escribe solo la oración actualizada sin comentarios adicionales. "
        "Escribe únicamente palabras en minúsculas y sin puntuación\n\n"
    )
    prompt_en = (
                    "You are second pass transcriber that is given the first pass transcription of an acoustic utterance. "
                    "Please fix any transcription errors of the first pass transcriber to minimize the edit distance to the "
                    "unknown reference transcription. Note that the original was spoken, so please do not correct disfluencies "
                    f"in the text and {'rather focus on correcting' if not names_focus else 'only'} proper names. Write only the updated sentence without any additional comments. "
                    "Write only lowercased words without punctuation\n\n"
                )
    return prompt_en if lang == "EN" else prompt_es

@dataclass
class LLMECConfig:
    task: str = "EC"
    # Provider / model
    provider: str = "hf"  # "hf" | "openai_api" | "hf_api"
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    model_dir: str = None # For HF
    device: str = "auto"  # for HF
    dtype: Optional[str] = None  # e.g. "bfloat16", "float16", "float32"

    # Strategy and data handling
    hf_batch_size: int = 8  # max number of prompts per HF generate() call
    strategy: str = "top1_only"  # or "nbest_reason_rewrite"
    nbest_k: int = 5
    order_by_score: bool = True  # if True, sort by score dsc (higher is better)
    score_policy: str = "keep_top1"  # "keep_top1" | "zero"
    prompt_lang: str = "EN" # ES
    keep_apostrophe: bool = False

    # Context
    context_mode: str = "none"  # "none" | "prev_top1" | "prev_corrected"
    context_window: int = 100  # number of previous words to include (soft limit)

    # Generation params (both providers)
    N_expand: int = 5
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0  # HF only
    eos_strings: Tuple[str, ...] = ("</s>", "<|eot_id|>", "<|im_end|>")

    # Safety / parsing
    expect_json: bool = True
    json_key: str = "text"
    rejection_ratio: float = 0.25

    # Prompt style
    few_shot: bool = False
    name_focused: bool = False
    tap_examples: List[TapExample] = None
    system_prompt: str = (
                    "You are second pass transcriber that is given the first pass transcription of an acoustic utterance. "
                    "Please fix any transcription errors of the first pass transcriber to minimize the edit distance to the "
                    "unknown reference transcription. Note that the original was spoken, so please do not correct disfluencies "
                    "in the text and rather focus on correcting proper names. Write only the updated sentence without any additional comments. "
                    "Write only lowercased words without punctuation\n\n"
                )
    # (
    # "You are second pass transcriber that is given the first pass transcription of an acoustic utterance. "
    # "Please fix any transcription errors of the first pass transcriber to minimize the edit distance to the "
    # "unknown reference transcription. Write only the updated sentence without any additional comments. Do not translate it."
    # "Write only lowercased words without punctuation"))

    # Prompt_alt_1 few shot:
    # """
    # You are an error corrector for automatic speech recognition (ASR) in Spanish.
    # You receive an ASR hypothesis and must produce a corrected version in Spanish.
    #
    # Examples:
    #
    # input: "buenos dias a todoz y todas"
    # output: "buenos dias a todos y todas"
    #
    # input: "el real madri a ganado la liga"
    # output: "el real madrid ha ganado la liga"
    #
    # Rules:
    # - correct only clear errors, make minimal changes
    # - prefer substitutions with words that sound very similar to the originals
    # - avoid deleting words unless clearly necessary
    # - try to keep a similar number of words
    # - do not add new information
    #
    # Output format:
    # - only output the corrected sentence
    # - all in lowercase
    # - no punctuation
    # - no hyphens
    # - no quotation marks
    # - no parentheses
    # - no special symbols
    # - use only simple spaces between words
    # - do not add any comments or explanations
    # """
    # system_prompt_ES: str = (
    #     "Eres un transcriptor de segunda pasada que recibe la transcripción de primera pasada de un enunciado acústico. "
    #     "Corrige los errores de transcripción de la primera pasada para minimizar la distancia de edición con respecto "
    #     "a la transcripción de referencia desconocida. Escribe solo la frase corregida sin ningún comentario adicional. "
    #     "No traduzcas el contenido. "
    #     "Escribe únicamente palabras en minúsculas y sin signos de puntuación."
    # )
# ------------------------------ Prompt Builders -------------------------------

def _render_chat(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Render chat-style messages using tokenizer's chat template if available."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: naive role-tagged template
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"[{role}]\n{m.get('content','')}\n")
    parts.append("[ASSISTANT]\n")
    return "\n".join(parts)


essential_json_hint = "Return strictly JSON of the form {\"text\": \"<corrected>\"} with no extra text."

def get_context_block(context_transcripts, cfg: LLMECConfig) -> str:
    context_block = []
    context_words = 0
    for i,t in enumerate(context_transcripts,1):
        context_words += len(t.split())
        context_block.append(f"Prev[{i}]: {t}")
        if context_words >= cfg.context_window:
            break
    return "\n".join(context_block)

def build_messages_top1(cfg: LLMECConfig, asr_text: str, context_transcripts: List[str]) -> List[Dict[str, str]]:
    context_block = get_context_block(context_transcripts, cfg)
    user = []
    if cfg.task == "EC":
        if cfg.context_mode != "none" and context_block:
            user.append("Context from previous segments:\n" + context_block)
            user.append("\nPlease correct ASR hypothesis below so that it naturally follows context above\n")
        user.append("ASR hypothesis to correct:\n" + asr_text)
        if cfg.expect_json:
            user.append(essential_json_hint)
        user_prompt = "\n\n".join(user)
        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    elif cfg.task == "Nbest_expand":
        if cfg.context_mode != "none" and context_block:
            user.append(
                "Context from previous segments:\n" + context_block
            )
        user.append(f"Expand the following ASR hypothesis into {cfg.N_expand} diverse alternatives for rescoring:\n{asr_text}")
        user_prompt = "\n\n".join(user)
        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        raise NotImplementedError(f"Unknown task: {cfg.task}")

def build_messages_top1_spanish(cfg: LLMECConfig, asr_text: str, context_transcripts: List[str]) -> List[Dict[str, str]]:
    context_block = get_context_block(context_transcripts, cfg)
    user_lines = []
    if cfg.task == "EC":
        if cfg.context_mode != "none" and context_block:
            user_lines.append(
                "Contexto de segmentos anteriores:\n" + context_block
            )
            user_lines.append(
                "\nCorrige la siguiente hipótesis de ASR para que siga naturalmente el contexto anterior."
            )

        user_lines.append("Hipótesis de ASR para corregir:\n" + asr_text)

        if cfg.expect_json:
            # Note: your essential_json_hint can be reused directly or translated if you prefer
            user_lines.append(
                "Devuelve estrictamente un JSON con la forma {\"text\": \"<transcripción corregida>\"}, sin añadir nada más."
            )

        user_prompt = "\n\n".join(user_lines)

        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    elif cfg.task == "Nbest_expand":
        if cfg.context_mode != "none" and context_block:
            user_lines.append(
                "Contexto de segmentos anteriores:\n" + context_block
            )

        user_lines.append(f"Expande la siguiente hipótesis de ASR en {cfg.N_expand} alternativas diversas para re-puntuación:\n{asr_text}")

        user_prompt = "\n\n".join(user_lines)

        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        raise NotImplementedError(f"Unknown task: {cfg.task}")

def build_messages_nbest_reason(
    cfg: LLMECConfig, candidates: List[str], context_transcripts: List[str]
) -> List[Dict[str, str]]:
    context_block = get_context_block(context_transcripts, cfg)
    cand_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    user_lines = []
    if cfg.task == "EC":
        if cfg.context_mode != "none" and context_block:
            user_lines.append("Context from previous segments:\n" + context_block)
        user_lines.append("Candidate ASR hypotheses (best first):\n" + cand_block)
        user_lines.append(
            "Choose or merge candidates to produce the best corrected transcript. "
            "Fix common ASR errors (homophones, punctuation, casing, missing small words)."
        )
        if cfg.expect_json:
            user_lines.append("Return strictly JSON: {\"text\": \"<final corrected transcript>\"}.")
        user_prompt = "\n\n".join(user_lines)
        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    elif cfg.task == "Nbest_expand":
        if cfg.context_mode != "none" and context_block:
            user_lines.append("Context from previous segments:\n" + context_block)
        user_lines.append(
            f"Expand the following ASR hypothesis into {cfg.N_expand} diverse alternatives for rescoring:\n{cand_block}"
        )
        user_prompt = "\n\n".join(user_lines)
        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        raise NotImplementedError(f"Unknown task: {cfg.task}")

def build_messages_nbest_reason_spanish(
    cfg: LLMECConfig, candidates: List[str], context_transcripts: List[str]
) -> List[Dict[str, str]]:

    context_block = get_context_block(context_transcripts, cfg)
    cand_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))

    user_lines = []
    if cfg.task == "EC":
        if cfg.context_mode != "none" and context_block:
            user_lines.append(
                "Contexto de segmentos anteriores:\n" + context_block
            )

        user_lines.append("Candidatos de hipótesis de ASR (mejor primero):\n" + cand_block)

        user_lines.append(
            "Elige o combina los candidatos para producir la mejor transcripción corregida. "
            "Realiza solo cambios mínimos. "
            "Prefiere sustituciones que suenen muy parecido y evita borrar palabras salvo que sea necesario. "
            "No añadas información nueva."
        )

        if cfg.expect_json:
            user_lines.append(
                "Devuelve estrictamente un JSON con la forma {\"text\": \"<transcripción corregida>\"}."
            )

        user_prompt = "\n\n".join(user_lines)

        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    elif cfg.task == "Nbest_expand":
        if cfg.context_mode != "none" and context_block:
            user_lines.append("Contexto de segmentos anteriores:\n" + context_block)
        user_lines.append(
            f"Expande la siguiente hipótesis de ASR en {cfg.N_expand} alternativas diversas para re-puntuación:\n{cand_block}"
        )
        user_prompt = "\n\n".join(user_lines)
        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        raise NotImplementedError(f"Unknown task: {cfg.task}")

# ------------------------------- Model Loading --------------------------------

def _select_dtype(dtype_str: Optional[str]):
    if dtype_str is None:
        return None
    if torch is None:
        return None
    m = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return m.get(dtype_str.lower(), None)


def load_model_and_tokenizer(cfg: LLMECConfig):
    """Only for provider == 'hf'. Returns (model, tokenizer)."""
    if cfg.provider != "hf":
        return None, None
    if AutoConfig is None or AutoTokenizer is None:
        raise RuntimeError("Transformers not available but provider='hf' requested.")
    assert cfg.model_dir
    model_path = get_content_dir_from_hub_cache_dir(cfg.model_dir)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    torch_dtype = _select_dtype(cfg.dtype)

    # Detect flash-attn availability.
    def _detect_flash_attn():
        try:
            import flash_attn  # flash-attn v2 top-level import
            return True
        except Exception:
            pass
        try:
            from flash_attn import flash_attn_func  # flash-attn v1 style
            return True
        except Exception:
            return False

    def _flash_attn_supported():
        # 1) import flash-attn
        flash_available = _detect_flash_attn()

        if not flash_available:
            return False

        # 2) must have CUDA
        if not torch.cuda.is_available():
            return False

        # 3) check compute capability
        # Ampere = 8.x, Hopper = 9.x (also supported)
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            return False

        # flash-attn requires >= 8.0
        if major < 8:
            return False

        return True

    flash_attn_available = _flash_attn_supported()
    print(f"Flash_attn_importable: {_detect_flash_attn()}")
    print(f"Flash_attn_available: {flash_attn_available}")
    model_kwargs: Dict[str, Any] = {}
    if cfg.device == "auto":
        model_kwargs["device_map"] = "auto"
    elif cfg.device == "cpu":
        model_kwargs["device_map"] = {"": "cpu"}
    else:  # explicit device string, e.g., "cuda"
        model_kwargs["device_map"] = {"": cfg.device}

    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if flash_attn_available:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    print("Model loaded, use attn_implementation: ", getattr(model, "_attn_implementation", None) or getattr(model.config, "_attn_implementation", None))
    # Set pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.cls_token

    return model, tokenizer


# --------------------------------- Text processing ---------------------------------
import unicodedata
import string
def has_punctuation(text: str, skip_apostrophe: bool = False) -> bool:
    for ch in text:
        if skip_apostrophe and ch == "'":
            continue
        if unicodedata.category(ch).startswith("P"):
            return True
    return False


def strip_punctuation(text: str, skip_apostrophe: bool = False) -> str:
    # Build a translation table that removes punctuation except apostrophes (if requested)
    if skip_apostrophe:
        punct = string.punctuation.replace("'", "")
    else:
        punct = string.punctuation

    table = str.maketrans("", "", punct)
    return text.translate(table)
# --------------------------------- Generation ---------------------------------

def _first_json_text(text: str, key: str) -> str:
    # 1) Try proper JSON object first
    import ast
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            val = obj.get(key, "")
            if isinstance(val, list):
                return ",".join(map(str, val)).strip()
            return str(val).strip()
        except json.JSONDecodeError:
            # fall through to more relaxed handling
            pass

    # 2) Try to find a bare list after the key name
    m = re.search(rf'{re.escape(key)}"\s*:\s*(\[[\s\S]*?\])', text)
    if not m:
        m = re.search(r"\[[\s\S]*?\]", text)
    if not m:
        return ""

    try:
        lst = ast.literal_eval(m.group(1 if m.lastindex else 0))
        if isinstance(lst, list):
            return ",".join(map(str, lst)).strip()
        return str(lst).strip()
    except Exception:
        return ""

def generate_json_text_hf(
    model,
    tokenizer,
    rendered_prompt: str,
    cfg: LLMECConfig,
) -> str:
    """Backward-compatible single-sample wrapper."""
    return generate_json_text_hf_batch(model, tokenizer, [rendered_prompt], cfg)[0]

def generate_json_text_hf_batch(
    model,
    tokenizer,
    rendered_prompts: List[str],
    cfg: LLMECConfig,
) -> List[str]:
    if torch is None:
        raise RuntimeError("torch is required for HF provider path.")
    if not rendered_prompts:
        return []

    # Tokenize batch
    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # best-effort EOS
    eos_ids: List[int] = []
    for tok in cfg.eos_strings:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                eos_ids.append(tid)
        except Exception:
            pass

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.temperature > 0.0,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        eos_token_id=(eos_ids[0] if eos_ids else tokenizer.eos_token_id),
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # # lengths of each input prompt (before padding)
    # # attention_mask: 1 for real tokens, 0 for padding
    # attn_mask = inputs["attention_mask"]
    # input_lengths = attn_mask.sum(dim=1)  # shape: [B]
    seq_len = inputs["input_ids"].shape[1]  # T

    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    texts: List[str] = []
    for i in range(out.size(0)):
        # slice generated part for sample i
        gen_tokens = out[i, seq_len :]

        # Manual cut at <|im_end|> # TODO: Maybe it is better to always manually cut at end_id?
        if end_id is not None and end_id in gen_tokens:
            try:
                end_idx = (gen_tokens == end_id).nonzero(as_tuple=True)[0][0].item()
                gen_tokens = gen_tokens[:end_idx]
            except IndexError:
                pass

        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        # Some manual post-processing
        if not cfg.expect_json and cfg.task == "EC":
            flag = False
            if "-" in text:
                print(f"Replace Unexpected - in '{text}'")
                text = text.replace("-", " ")
            if ":" in text:
                print(f"Remove Unexpected : in '{text}'")
                text = text.split(":")[-1]
                flag = True
            if "\n" in text:
                print(f"Remove Unexpected newline in '{text}'")
                text = text.split("\n")[-1]
                flag = True
            if has_punctuation(text, skip_apostrophe=cfg.keep_apostrophe):
                print(f"Remove Unexpected punctuation in '{text}'")
                text = strip_punctuation(text)
                flag = True
            if flag:
                print(f"\n--------------------------------")
        if cfg.task == "EC":
            if cfg.expect_json:
                text = _first_json_text(text, cfg.json_key)
            else:
                text = text.strip()
        elif cfg.task == "Nbest_expand":
            text = _first_json_text(text, cfg.json_key) # Expected hyps seperated by ,
        texts.append(text)

    return texts


def generate_json_text_hf_api(messages, cfg):
    from huggingface_hub import InferenceClient
    # messages is our list of {"role": "...", "content": "..."}
    # For non-OpenAI endpoints, we merge messages to a single prompt:
    prompt = "\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages) + "\n[ASSISTANT]\n"
    client = InferenceClient(model=cfg.model_name, token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    # text-generation endpoint:
    out = client.text_generation(
        prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature or 0.0,
        top_p=cfg.top_p or 1.0,
        return_full_text=False,
    )
    text = out if isinstance(out, str) else out.generated_text
    return _first_json_text(text, cfg.json_key) if cfg.expect_json else text.strip()

def generate_json_text_openai(messages: List[Dict[str, str]], cfg: LLMECConfig) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI provider requested but 'openai' package is not available.") from e

    client = OpenAI()
    resp = client.chat.completions.create(
        model=cfg.model_name,
        messages=messages,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _first_json_text(text, cfg.json_key) if cfg.expect_json else text

# ------------------------------ Core Processing --------------------------------

def _sorted_candidates_from_pairs(
    pairs: List[Tuple[float, str]], k: int, order_by_score: bool
) -> List[str]:
    if not pairs:
        return []
    if order_by_score:
        pairs = sorted(pairs, key=lambda t: t[0], reverse=True)  # higher is better
    return [hyp for _, hyp in pairs[:k]]


def _top1_pair(pairs: List[Tuple[float, str]], order_by_score: bool) -> Tuple[float, str]:
    if not pairs:
        return (0.0, "")
    if order_by_score:
        pairs = sorted(pairs, key=lambda t: t[0], reverse=True)
    return pairs[0]

# ------------------------- Hash Normalization Helper --------------------------

def normalize_cfg_for_hash(cfg: LLMECConfig) -> LLMECConfig:
    """Return a *new* LLMECConfig with irrelevant attributes normalized so that
    the job hash only changes when *meaningful* settings change.

    Rules:
    - If context_mode == 'none' → context_window is irrelevant; set to 0.
    - If strategy == 'none' → nbest_k is irrelevant; set to 3.
    - If provider != 'hf' → HF-only knobs are irrelevant; set to canonical defaults.
    """
    norm = replace(cfg)
    # Context irrelevance
    if norm.context_mode == "none":
        norm = replace(norm, context_window=0)
    if norm.strategy == "top1_only":
        norm = replace(norm, nbest_k=3)
    # HF-only knobs irrelevant for non-HF providers
    if norm.provider != "hf":
        norm = replace(norm, device="cpu", dtype=None, repetition_penalty=1.0, eos_strings=())
    if norm.task != "EC" and norm.context_mode != "none":
        if norm.context_mode != "prev_top1":
            print(f"[Warning]: For NON-EC use case, prev_top1=On none=Off, given is {norm.context_mode}. Set to prev_top1.")
            norm = replace(norm, context_mode="prev_top1")
    if norm.task != "Nbest_expand":
        norm = replace(norm, N_expand=0)
    norm = replace(norm, hf_batch_size=8, name_focused=False) # Batch independent, name_focused can be by-hashed with prompt
    return norm

# --------------------------------- Job Class ----------------------------------
class LLMErrorCorrectionJob(Job):
    """
    Input: py-dict text file with format {seq_tag: [(score, hyp), ...]}
    Output: gzipped py-dict text where each seq_tag maps to a single-item list
            [(score, corrected_hyp)]
    Supports providers: HuggingFace (local/transformers) and OpenAI Chat Completions.
    """

    def __init__(
        self,
        *,
        recog_out_file: tk.Path,
        config: Optional[LLMECConfig] = None,
    ) -> None:
        super().__init__()
        self.n_best_file = recog_out_file
        if config is None:
            self.cfg = LLMECConfig()
        else:
            self.cfg = config
        if self.cfg.provider == "hf" and self.cfg.model_dir is None:
            model = DownloadHuggingFaceRepoJob(model_id=self.cfg.model_name)
            tk.register_output(self.cfg.model_name, model.out_hub_cache_dir)
            self.cfg.model_dir = model.out_hub_cache_dir
        if self.cfg.system_prompt is None:
            print(f"Use default english system prompt! Task: {self.cfg.task}")
            self.cfg.system_prompt = LLMECConfig.system_prompt

        self.out_file = self.output_path("output.py.gz")
        self.out_rejection_rate = self.output_var("rejection_rate")
        self.rqmt = {"time": 8, "cpu": 3, "mem": 12, "gpu": 1, "gpu_mem": 24}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _report_dev_memory_stats(self, device_str: str):
        try:
            import returnn.util.basic as util  # type: ignore
            if torch is None:
                return
            dev = torch.device(device_str)
            if dev.type == "cuda":
                stats = [
                    f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                    f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                    f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                    f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
                ]
                print("Memory:", " ".join(stats))
        except Exception:
            pass

    def llm_update_rejection_heuristic(self, orig: str, corrected: str) -> bool:
        factor = {"EC": 1.0, "Nbest_expand": self.cfg.N_expand}[self.cfg.task]
        if not orig or abs(len(orig) - len(' '.join(corrected.split(',')))/factor) / len(orig) > self.cfg.rejection_ratio: # Never correct empty hyp
            return False
        return True

    def correct_py_nbest_dict(
            self, d_rec: Dict[str, List[Tuple[float, str]]]
    ) -> tuple[dict[str, list[tuple[float, str]]], float | Any]:
        """
        Process entries in dict order. Assumes that within a recording, segments are grouped
        and ordered. Context never crosses recording boundaries; we detect record IDs
        using extract_record_id(seq_tag).
        """
        cfg = self.cfg
        model, tokenizer = load_model_and_tokenizer(cfg)

        def _get_rec_id(tag: str):
            try:
                from i6_experiments.users.zhang.datasets.utils import extract_record_id  # type: ignore
                return extract_record_id(tag)
            except Exception:
                return None  # single-bucket fallback

        out: Dict[str, List[Tuple[float, str]]] = {}

        # can we safely batch with HF?
        can_batch_hf = (
                cfg.provider == "hf"
                and cfg.context_mode != "prev_corrected"
        )

        # group seq_tags by recording
        from collections import OrderedDict
        rec2tags: "OrderedDict[Any, List[str]]" = OrderedDict()
        for seq_tag in d_rec.keys():
            rec_id = _get_rec_id(seq_tag)
            rec2tags.setdefault(rec_id, []).append(seq_tag)

        seg_count = 0  # number of finalized segments (for debug printing)
        total_num_seg = len(d_rec)
        processed_seg = 0
        rejected = 0
        empty_count = 0
        for rec_id, tags in rec2tags.items():
            # context buffers per recording
            prev_top1_buf: deque[str] = deque(maxlen=max(1, cfg.context_window))
            prev_corr_buf: deque[str] = deque(maxlen=max(1, cfg.context_window))

            if can_batch_hf:
                # -------- HF batched path (context: none or prev_top1) --------
                print(f"Do batching with size {self.cfg.hf_batch_size}")
                batch_prompts: List[str] = []
                batch_tags: List[str] = []
                batch_top_texts: List[str] = []
                batch_scores: List[float] = []

                def flush_batch():
                    nonlocal seg_count, prev_corr_buf, rejected
                    if not batch_prompts:
                        return
                    assert model is not None and tokenizer is not None
                    corrected_list = generate_json_text_hf_batch(
                        model, tokenizer, batch_prompts, cfg
                    )
                    for seq_tag_, top_text_, kept_score_, rendered_, corrected in zip(
                            batch_tags, batch_top_texts, batch_scores, batch_prompts, corrected_list
                    ):
                        seg_count += 1
                        # debug / memory
                        if seg_count % 200 < 2:
                            self._report_dev_memory_stats(
                                "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
                            )
                            print(
                                f"Prompt: {rendered_}\n"
                                f"\t -> Completion: {corrected!r}\n"
                            )
                            if not self.llm_update_rejection_heuristic(top_text_, corrected):
                                print("->Rejected!\n")

                        if not self.llm_update_rejection_heuristic(top_text_, corrected):
                            rejected += 1
                            if self.cfg.task == "Nbest_expand":
                                out[seq_tag_] = []
                                continue
                            corrected = top_text_

                        if cfg.context_window > 0:
                            prev_corr_buf.append(corrected)
                        if cfg.task == "EC":
                            out[seq_tag_] = [(kept_score_, corrected)]
                        elif cfg.task == "Nbest_expand":
                            out[seq_tag_] = [(kept_score_, hyp.strip('"').strip("'")) for hyp in corrected.split(",")]
                        else:
                            raise NotImplementedError

                    # clear batch buffers
                    batch_prompts.clear()
                    batch_tags.clear()
                    batch_top_texts.clear()
                    batch_scores.clear()

                for seq_tag in tags:
                    processed_seg += 1
                    if processed_seg%500 == 1:
                        print(f"[Processed]: {processed_seg}/{total_num_seg}, {100*processed_seg/total_num_seg:.2f}%")
                    pairs = d_rec[seq_tag]

                    # build context from prev_top1 only (prev_corrected is not used in this mode)
                    if cfg.context_mode == "prev_top1" and len(prev_top1_buf) > 0:
                        context_transcripts = list(prev_top1_buf)
                    else:
                        context_transcripts = []

                    top_score, top_text = _top1_pair(pairs, cfg.order_by_score)
                    if not top_text:
                        empty_count += 1
                    if cfg.strategy == "top1_only":
                        if cfg.prompt_lang == "EN":
                            messages = build_messages_top1(cfg, top_text, context_transcripts)
                        else:
                            messages = build_messages_top1_spanish(cfg, top_text, context_transcripts)
                        kept_score = top_score if cfg.score_policy == "keep_top1" else 0.0
                    elif cfg.strategy == "nbest_reason_rewrite":
                        cands = _sorted_candidates_from_pairs(pairs, cfg.nbest_k, cfg.order_by_score)
                        if cfg.tap_examples:
                            # TAP-style multi-turn few-shot
                            messages = build_messages_tap_nbest_reason(cfg, cfg.tap_examples, cands)
                        else:
                            # Original simple style
                            if cfg.prompt_lang == "EN":
                                messages = build_messages_nbest_reason(cfg, cands, context_transcripts)
                            else:
                                messages = build_messages_nbest_reason_spanish(cfg, cands, context_transcripts)
                        kept_score, _ = _top1_pair(pairs, cfg.order_by_score)
                        kept_score = kept_score if cfg.score_policy == "keep_top1" else 0.0
                    else:
                        raise ValueError(f"Unknown strategy: {cfg.strategy}")

                    # render chat prompt for this segment
                    assert tokenizer is not None
                    rendered = _render_chat(tokenizer, messages)

                    # add to batch
                    batch_tags.append(seq_tag)
                    batch_prompts.append(rendered)
                    batch_top_texts.append(top_text)
                    batch_scores.append(kept_score)

                    # update top1-based context immediately (LLM output not needed)
                    if cfg.context_window > 0:
                        prev_top1_buf.append(top_text)

                    # flush if batch full
                    if len(batch_prompts) >= cfg.hf_batch_size:
                        flush_batch()

                # flush remaining in this recording
                flush_batch()

            else:
                # -------- Original sequential path (hf_api, openai, or prev_corrected) --------
                for seq_tag in tags:
                    pairs = d_rec[seq_tag]

                    if cfg.context_mode == "prev_corrected" and len(prev_corr_buf) > 0:
                        context_transcripts = list(prev_corr_buf)
                    elif cfg.context_mode == "prev_top1" and len(prev_top1_buf) > 0:
                        context_transcripts = list(prev_top1_buf)
                    else:
                        context_transcripts = []

                    top_score, top_text = _top1_pair(pairs, cfg.order_by_score)
                    if not top_text:
                        empty_count += 1

                    if cfg.strategy == "top1_only":
                        if cfg.prompt_lang == "EN":
                            messages = build_messages_top1(cfg, top_text, context_transcripts)
                        else:
                            messages = build_messages_top1_spanish(cfg, top_text, context_transcripts)
                        kept_score = top_score if cfg.score_policy == "keep_top1" else 0.0
                    elif cfg.strategy == "nbest_reason_rewrite":
                        cands = _sorted_candidates_from_pairs(pairs, cfg.nbest_k, cfg.order_by_score)
                        if cfg.tap_examples:
                            # TAP-style multi-turn few-shot
                            messages = build_messages_tap_nbest_reason(cfg, cfg.tap_examples, cands)
                        else:
                            # Original simple style
                            if cfg.prompt_lang == "EN":
                                messages = build_messages_nbest_reason(cfg, cands, context_transcripts)
                            else:
                                messages = build_messages_nbest_reason_spanish(cfg, cands, context_transcripts)
                        kept_score, _ = _top1_pair(pairs, cfg.order_by_score)
                        kept_score = kept_score if cfg.score_policy == "keep_top1" else 0.0
                    else:
                        raise ValueError(f"Unknown strategy: {cfg.strategy}")

                    if cfg.provider == "hf":
                        if model is None or tokenizer is None:
                            raise RuntimeError("HF provider selected but model/tokenizer not loaded.")
                        rendered = _render_chat(tokenizer, messages)
                        corrected = generate_json_text_hf(model, tokenizer, rendered, cfg)
                    elif cfg.provider == "hf_api":
                        corrected = generate_json_text_hf_api(messages, cfg)
                    elif cfg.provider == "openai":
                        corrected = generate_json_text_openai(messages, cfg)
                    else:
                        raise ValueError(f"Unknown provider: {cfg.provider}")

                    seg_count += 1
                    if seg_count % 200 < 2:
                        self._report_dev_memory_stats(
                            "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
                        )
                        print(f"Prompt: {rendered if cfg.provider == 'hf' else messages}\n"
                              f"\t -> Completion: {corrected!r}\n")
                        if not self.llm_update_rejection_heuristic(top_text, corrected):
                            print("->Rejected!")

                    if not self.llm_update_rejection_heuristic(top_text, corrected):
                        rejected += 1
                        if self.cfg.task == "Nbest_expand":
                            out[seq_tag] = []
                            continue
                        corrected = top_text

                    if cfg.task == "EC":
                        out[seq_tag] = [(kept_score, corrected)]
                    elif cfg.task == "Nbest_expand":
                        out[seq_tag] = [(kept_score, hyp.strip('"')) for hyp in corrected.split(",")]
                    else:
                        raise NotImplementedError
                    if cfg.context_window > 0:
                        prev_top1_buf.append(top_text)
                        prev_corr_buf.append(corrected)

                    out[seq_tag] = [(kept_score, corrected)]
        del model, tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()
        print("Finished generation and cleaned up\n.")
        rejection_rate = rejected / total_num_seg
        self.out_rejection_rate.set(rejection_rate)
        print(f"Rejection Rate: {rejected}/{total_num_seg} = {100*rejected/total_num_seg:.2f}%")
        print(f"Empty hyp Rate: {empty_count}/{total_num_seg} = {100 * empty_count / total_num_seg:.2f}%")
        return out, rejection_rate

    def run(self):
        from transformers import logging
        logging.set_verbosity_error()
        device_str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self._report_dev_memory_stats(device_str)

        # Read py-dict; already sorted/grouped by recording and time
        with cutil.uopen(self.n_best_file, "rt") as f:
            d_rec = eval(f.read(), {"nan": float("nan"), "inf": float("inf")})
        from i6_experiments.users.zhang.datasets.utils import sort_dict_by_record
        d_rec = sort_dict_by_record(d_rec)
        def write_entries(out, pairs):
            for score, hyp in pairs:
                out.write(f"  ({score!r}, {hyp!r}),\n")
        corrected, rejection_rate = self.correct_py_nbest_dict(d_rec)
        #self.out_rejection_rate.set(rejection_rate)
        # Write gzipped py-dict with the same style
        with gzip.open(self.out_file.get_path(), "wt") as out:
            out.write("{\n")
            for seq_tag, pairs in corrected.items():
                out.write(f"{seq_tag!r}: [\n")
                if self.cfg.task == "Nbest_expand":
                    assert isinstance(pairs, list)
                    write_entries(out, d_rec[seq_tag] + pairs)
                elif self.cfg.task == "EC":
                    assert len(pairs) == 1 and isinstance(pairs[0], tuple)
                    score, hyp = pairs[0]
                    out.write(f"  ({score!r}, {hyp!r}),\n")
                out.write("],\n")
            out.write("}\n")
        self._report_dev_memory_stats(device_str)
        # Cleanup
        if torch is not None and device_str == "cuda":
            torch.cuda.empty_cache()
        import gc

        gc.collect()
        print("Finished LLM error correction.")

    @classmethod
    def hash(cls, parsed_args):
        """Compute hash with normalized LLMECConfig (dataclass-aware).
        The incoming parsed_args['config'] is expected to be an LLMECConfig.
        """
        cfg = parsed_args.get("config")
        out = parsed_args.get("recog_out_file")
        if isinstance(cfg, LLMECConfig):
            norm_cfg = normalize_cfg_for_hash(cfg)
            d = {k: v for k, v in parsed_args.items() if k != "config"}
            d["config"] = asdict(norm_cfg)
            d["recog_out_file"] = out
            return super().hash(d)
        # Fallback to previous logic if config is not the expected dataclass
        return super().hash(parsed_args)

# ----------------------------------- CLI --------------------------------------

def _load_pydict(path: str) -> Dict[str, Any]:
    with open(path, "rt", encoding="utf-8") as f:
        return eval(f.read(), {"nan": float("nan"), "inf": float("inf")})


def _save_pydict(path: str, data: Dict[str, Any]) -> None:
    with open(path, "wt", encoding="utf-8") as f:
        f.write("{\n")
        for k, v in data.items():
            f.write(f"{k!r}: [\n")
            assert isinstance(v, list) and len(v) == 1 and isinstance(v[0], tuple)
            score, hyp = v[0]
            f.write(f"  ({score!r}, {hyp!r}),\n")
            f.write("]\n")
        f.write("}\n")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="LLM error correction for ASR N-best py-dict")
    parser.add_argument("--in", dest="inp", required=True, help="Input py-dict path")
    parser.add_argument("--out", dest="out", required=True, help="Output py-dict path")
    parser.add_argument("--provider", choices=["hf", "openai"], default="hf", help="LLM provider")
    parser.add_argument("--model", dest="model", default=LLMECConfig.model_name)
    parser.add_argument("--strategy", choices=["top1_only", "nbest_reason_rewrite"], default="top1_only")
    parser.add_argument("--nbest_k", type=int, default=5)
    parser.add_argument("--order_by_score", type=int, default=1, help="1: sort ascending by score (lower is better)")
    parser.add_argument("--score_policy", choices=["keep_top1", "zero"], default="keep_top1")
    parser.add_argument("--context_mode", choices=["none", "prev_top1", "prev_corrected"], default="none")
    parser.add_argument("--context_window", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args(argv)
    cfg = LLMECConfig(
        provider=args.provider,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        strategy=args.strategy,
        nbest_k=args.nbest_k,
        order_by_score=bool(args.order_by_score),
        score_policy=args.score_policy,
        context_mode=args.context_mode,
        context_window=args.context_window,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    #corrected = LLMErrorCorrectionJob(recog_out_file=tk.Path(args.inp), config=cfg).out_file


if __name__ == "__main__":
    main()
