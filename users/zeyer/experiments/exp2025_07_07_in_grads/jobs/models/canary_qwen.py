"""Canary-Qwen (NVIDIA NeMo SALM) model adapter.

First pass: recog-only. Forward/log_probs (for grad extraction) need access
to SALM's internal Canary encoder output as a grad-able tensor -- non-trivial
to expose via the SALM wrapper -- left as TODO.

Same overlay-activation pattern as Voxtral: ``sys.path.insert(0, OVERLAY_PATH)``
in ``__init__`` before any ``nemo`` import; other model classes are unaffected.
"""

from __future__ import annotations

from typing import Optional, Union, Any, Sequence, List, Dict
import os
import sys
import tempfile
import time

import numpy as np
import torch

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from .base import BaseModelInterface, ForwardOutput


OVERLAY_PATH = "/home/az668407/work/canary-qwen-overlay"


def _activate_overlay() -> None:
    """Prepend the canary-qwen overlay site-packages so the in-process
    ``nemo`` import resolves to the overlay's copy (env may or may not
    have nemo at a different version)."""
    if OVERLAY_PATH not in sys.path:
        sys.path.insert(0, OVERLAY_PATH)


class CanaryQwen(BaseModelInterface):
    """NVIDIA Canary-Qwen 2.5B (Canary-1B-flash encoder + Qwen3-1.7B + LoRA).

    Loaded via NeMo's :class:`SALM` wrapper. See module docstring for the
    overlay activation pattern.

    For now: recog only. ``forward`` raises NotImplementedError pending a
    grad-able audio-embedding hook into SALM internals.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        llm_model_dir: str,
        speech_prompt: str = "Transcribe the following:",
    ):
        """
        :param model_dir: hub cache dir for ``nvidia/canary-qwen-2.5b``
            (via DownloadHuggingFaceRepoJobV2).
        :param llm_model_dir: hub cache dir for ``Qwen/Qwen3-1.7B`` (the base
            LLM SALM resolves via AutoTokenizer at init time). Required
            because compute nodes are offline.
        :param speech_prompt: user-turn text that precedes the audio_locator
            tag. Per the model card, "Transcribe the following: <audio>"
            is the documented ASR prompt.
        """
        super().__init__()
        _activate_overlay()

        self.device = device
        self.model_dir = model_dir
        self.llm_model_dir = llm_model_dir
        self.speech_prompt = speech_prompt

        # Merge canary + qwen hub_cache dirs by symlink so SALM's AutoTokenizer
        # (which only honors HF_HUB_CACHE / HF_HOME) can resolve both repos.
        merged_cache = tempfile.mkdtemp(prefix="hf_hub_merged_")
        for src in (self.model_dir, self.llm_model_dir):
            src_path = src if isinstance(src, str) else str(src)
            for name in os.listdir(src_path):
                if name.startswith("."):
                    continue
                link = os.path.join(merged_cache, name)
                if os.path.lexists(link):
                    continue
                os.symlink(os.path.join(src_path, name), link)
        os.environ["HF_HUB_CACHE"] = merged_cache
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        print(f"HF_HUB_CACHE merged: {merged_cache}")

        print("Import NeMo / SALM (from overlay)...")
        start_time = time.time()
        import nemo
        from nemo.collections.speechlm2.models import SALM
        print(f"  nemo={nemo.__version__} from {nemo.__file__}")
        print(f"  ({time.time() - start_time:.1f}s)")

        model_dir_str = get_content_dir_from_hub_cache_dir(self.model_dir)
        print(f"Loading SALM from {model_dir_str}...")
        start_time = time.time()
        self.model = SALM.from_pretrained(model_dir_str)
        self.model.to(device)
        self.model.eval()
        print(f"  ({time.time() - start_time:.1f}s)")
        print("model type:", type(self.model).__name__)
        try:
            self.audio_locator_tag = self.model.audio_locator_tag
            print(f"  audio_locator_tag={self.audio_locator_tag!r}")
        except AttributeError:
            self.audio_locator_tag = "<audio>"  # fallback; will likely break recog
            print("  WARN: model.audio_locator_tag missing, using fallback")

    # ---- Recog ----------------------------------------------------------

    def _save_audio_tmp(self, audio: torch.Tensor, sample_rate: int) -> str:
        import soundfile as sf
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, audio.cpu().numpy().astype(np.float32), sample_rate)
        return path

    def recog(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        raw_input_seq_lens: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> List[List[str]]:
        """Greedy recognition via SALM's documented ASR-mode API."""
        assert len(raw_inputs) == 1
        audio_path = self._save_audio_tmp(raw_inputs[0], raw_inputs_sample_rate)
        try:
            prompts = [
                [
                    {
                        "role": "user",
                        "content": f"{self.speech_prompt} {self.audio_locator_tag}",
                        "audio": [audio_path],
                    }
                ]
            ]
            with torch.no_grad():
                answer_ids = self.model.generate(prompts=prompts, max_new_tokens=max_new_tokens)
            hyp_text = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        return [hyp_text.strip().split()]

    # ---- Forward / log_probs (TODO) -------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "CanaryQwen forward (forced alignment) requires a grad-able hook"
            " into SALM's perception module. Not implemented yet."
        )

    def log_probs(self, *args, **kwargs):
        raise NotImplementedError("CanaryQwen log_probs not implemented")
