"""Canary-Qwen (NVIDIA NeMo SALM) model adapter.

Forward path (forced alignment) implementation
-----------------------------------------------
SALM exposes the audio encoder as ``self.model.perception(audios, audio_lens)``
returning ``(audio_embeds[B,T,H], audio_embed_lens[B])`` -- the embeds tensor
is grad-able. We:

1. Build ``input_ids`` via NeMo's :class:`PromptFormatter` (Qwen chat format)
   with a user turn that contains the audio_locator_tag and an assistant turn
   pre-filled with the reference transcription.
2. Call ``self.model.perception`` on the raw audio; slice to the real length
   and ``requires_grad_(True).retain_grad()`` so backward gives
   ``d log p / d audio_embeds``.
3. Splice audio embeds into text embeds at the ``audio_locator_tag_id``
   positions via ``replace_placeholders_and_build_targets``.
4. Run ``self.model.llm(inputs_embeds=...)`` and keep the last hidden state for
   ``log_probs`` to re-apply ``lm_head`` at sliced positions.

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

from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from ..logits_transform import make_logits_transform
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
    forward-path design.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        llm_model_dir: str,
        speech_prompt: str = "Transcribe the following:",
        logits_transform: Union[None, str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]] = None,
    ):
        """
        :param model_dir: hub cache dir for ``nvidia/canary-qwen-2.5b``.
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
        self.logits_transform = make_logits_transform(logits_transform)

        # Merge canary + qwen hub_cache dirs by symlink so SALM's
        # AutoTokenizer (which only honors HF_HUB_CACHE / HF_HOME) can
        # resolve both repos.
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

        self.audio_locator_tag = self.model.audio_locator_tag
        self.audio_locator_tag_id = int(self.model.audio_locator_tag_id)
        self.assistant_end_token_id = int(self.model.text_eos_id)
        self.text_pad_id = int(self.model.text_pad_id)
        print(
            f"  audio_locator_tag={self.audio_locator_tag!r} (id={self.audio_locator_tag_id}); "
            f"eos={self.assistant_end_token_id}; pad={self.text_pad_id}"
        )

        from nemo.collections.common.prompts import PromptFormatter
        self.formatter = PromptFormatter.resolve(self.model.cfg.prompt_format)(self.model.tokenizer)
        print(f"  prompt_format={self.model.cfg.prompt_format!r}")

    # ---- Helpers --------------------------------------------------------

    def _save_audio_tmp(self, audio: torch.Tensor, sample_rate: int) -> str:
        import soundfile as sf
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, audio.cpu().numpy().astype(np.float32), sample_rate)
        return path

    def _build_chat_input_ids(self, *, transcription: Optional[str]):
        """Build chat-formatted input_ids via NeMo's PromptFormatter.

        Returns (input_ids[1, T], dst_text_start) -- dst_text_start is the
        index in input_ids at which the assistant transcription tokens
        begin (None for open recog).
        """
        user_turn = {
            "role": "user",
            "content": f"{self.speech_prompt} {self.audio_locator_tag}",
        }
        user_only = self.formatter.encode_dialog([user_turn])["input_ids"]
        if transcription is None:
            return user_only.unsqueeze(0), None
        full = self.formatter.encode_dialog(
            [user_turn, {"role": "assistant", "content": transcription}]
        )["input_ids"]
        return full.unsqueeze(0), int(user_only.shape[0])

    # ---- Forward (forced alignment) -------------------------------------

    def forward(
        self,
        *,
        raw_inputs: Union[np.ndarray, torch.Tensor, List[List[str]]],
        raw_inputs_sample_rate: Optional[int] = None,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
        omitted_prev_context: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        from nemo.collections.speechlm2.models.salm import replace_placeholders_and_build_targets

        assert (len(raw_inputs),) == raw_input_seq_lens.shape == (len(raw_targets),) == raw_target_seq_lens.shape
        assert len(raw_inputs) == 1, "CanaryQwen wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("CanaryQwen chunked context not implemented yet")

        dev = self.device
        words = raw_targets[0]
        transcription = " ".join(words)
        print(f"[fwd] start; words={len(words)} transcription={transcription!r}", flush=True)

        input_ids, dst_text_start = self._build_chat_input_ids(transcription=transcription)
        input_ids = input_ids.to(dev)
        print(
            f"[fwd] chat_inputs ok; dst_text_start={dst_text_start} "
            f"input_ids.shape={tuple(input_ids.shape)}",
            flush=True,
        )

        # Audio encoder.
        audios = raw_inputs.to(dev).float()
        audio_lens = raw_input_seq_lens.to(dev).long()
        audio_embeds_batched, audio_embed_lens = self.model.perception(audios, audio_lens)
        torch.cuda.synchronize()
        n_audio_total = int(audio_embeds_batched.shape[1])
        n_audio_real = int(audio_embed_lens[0])
        # Build the [1, T_real, H] tensor as the grad leaf and consume it via
        # [0] in the graph. autograd.grad(loss, inputs) needs ``inputs`` to be
        # the exact tensor used downstream -- a post-hoc unsqueeze view is a
        # sibling branch, not an ancestor of the loss.
        audio_embeds_b = audio_embeds_batched[:, :n_audio_real].contiguous()  # (1, T_real, H)
        audio_embeds_b.requires_grad_(True)
        audio_embeds_b.retain_grad()
        print(
            f"[fwd] perception ok; full={n_audio_total} real={n_audio_real} "
            f"hidden={audio_embeds_b.shape[-1]} dtype={audio_embeds_b.dtype}",
            flush=True,
        )

        # Splice audio into text embeds at audio_locator_tag positions.
        input_ids_to_embed = torch.where(input_ids == self.audio_locator_tag_id, 0, input_ids)
        text_embeds = self.model.embed_tokens(input_ids_to_embed)
        input_embs, _, attention_mask = replace_placeholders_and_build_targets(
            input_ids=input_ids,
            embeds=text_embeds,
            padding_id=self.text_pad_id,
            placeholder_id=self.audio_locator_tag_id,
            replacements=[audio_embeds_b[0]],
            target_ids=None,
        )
        torch.cuda.synchronize()
        print(
            f"[fwd] splice ok; input_embs.shape={tuple(input_embs.shape)} "
            f"attn.shape={tuple(attention_mask.shape)}",
            flush=True,
        )

        # LLM forward; keep last hidden state for log_probs.
        res = self.model.llm(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        last_out = res.hidden_states[-1]
        del res
        print(
            f"[fwd] llm forward ok; last_out.shape={tuple(last_out.shape)}",
            flush=True,
        )

        # After expansion the assistant text in input_embs starts at
        # dst_text_start + (n_audio_real - 1) -- the single audio_locator
        # token in the user prompt expanded into n_audio_real frames.
        embs_dst_text_start = dst_text_start + n_audio_real - 1

        # Targets in input_ids coords: assistant text tokens, drop trailing
        # chat-end marker, re-append our EOS for chunk-exit log_prob lookups.
        dst_text_end = input_ids.shape[1] - 1
        targets = input_ids[:, dst_text_start:dst_text_end]
        n_targets = int(targets.shape[1])
        targets = torch.cat(
            [
                targets,
                torch.tensor(
                    [[self.assistant_end_token_id]], device=targets.device, dtype=targets.dtype
                ),
            ],
            dim=1,
        )

        # Per-word target_start_end via incremental decode.
        tokenizer = self.model.tokenizer
        words_start_end: List[List[int]] = []
        words_: List[str] = []
        for t in range(n_targets):
            tid = int(targets[0, t].item())
            s = tokenizer.ids_to_text([tid])
            if t == 0 or s.startswith(" "):
                words_start_end.append([t, t + 1])
                words_.append(s.lstrip(" "))
            else:
                words_[-1] += s
                words_start_end[-1][1] = t + 1
        assert len(words_start_end) == len(words_) == len(words), (
            f"word-grouping mismatch: target_decoded={words_!r} ref_words={words!r}"
        )
        assert words_ == words, f"target_decoded={words_!r} ref_words={words!r}"
        words_start_end = words_start_end + [[n_targets, n_targets + 1]]  # EOS slot

        # Audio frame -> raw sample mapping.
        n_samples = int(raw_input_seq_lens[0])
        input_slice = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([n_audio_real], dtype=torch.int64),
        )
        edges = torch.arange(n_audio_real + 1, dtype=torch.float64) * (n_samples / max(n_audio_real, 1))
        input_raw_start_end = torch.stack(
            [edges[:-1].round().long(), edges[1:].round().long()], dim=-1
        ).unsqueeze(0)  # [1, n_audio_real, 2]

        print(
            f"[fwd] returning ForwardOutput (n_audio_real={n_audio_real}, n_targets={n_targets})",
            flush=True,
        )
        return ForwardOutput(
            inputs=audio_embeds_b,
            input_seq_lens=torch.tensor([n_audio_real]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(words_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(embs_dst_text_start=embs_dst_text_start, last_out=last_out),
        )

    # ---- log_probs -------------------------------------------------------

    def log_probs(
        self,
        *,
        forward_output: ForwardOutput,
        start: Union[int, torch.Tensor],
        end: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        last_out = forward_output.outputs["last_out"]
        embs_dst_text_start = forward_output.outputs["embs_dst_text_start"]
        last_out = batch_slice(
            last_out,
            (embs_dst_text_start + start - 1, embs_dst_text_start + end - 1),
        )
        logits = self.model.llm.lm_head(last_out).float()
        for f in self.logits_transform:
            logits = f(logits)
        return logits.log_softmax(-1)

    # ---- Recog (open recognition, greedy) -------------------------------

    def _save_audio_for_generate(self, audio: torch.Tensor, sample_rate: int) -> str:
        return self._save_audio_tmp(audio, sample_rate)

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
        audio_path = self._save_audio_for_generate(raw_inputs[0], raw_inputs_sample_rate)
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
