"""Voxtral model adapter for the grad-based forced-alignment pipeline.

Mirrors the :class:`Phi4MM` interface (forward → ForwardOutput, log_probs,
recog) but for Mistral AI's Voxtral (HF ``VoxtralForConditionalGeneration``).

Notes
-----
- Voxtral lives in transformers >= 4.56, while our main env is pinned to
  4.51. We work around this with a per-job overlay at ``OVERLAY_PATH``:
  ``sys.path.insert(0, OVERLAY_PATH)`` inside ``__init__`` BEFORE the
  ``from transformers import ...`` line. Other model classes (Phi4MM) keep
  the env's transformers untouched because they never trigger this insert.

- Audio embedding access: Voxtral's standard forward path splices audio
  features into ``inputs_embeds`` internally via ``masked_scatter`` at
  ``audio_token_id`` positions. To expose a grad-able audio tensor we
  reproduce that splice ourselves: call ``model.get_audio_features`` ->
  ``requires_grad_(True).retain_grad()`` -> manually do the
  ``masked_scatter`` -> call ``model(inputs_embeds=merged)``. Backward then
  gives ``d log p / d audio_features``.

- Audio frame rate: Voxtral uses a Whisper-style encoder at ~50 Hz (20 ms
  per audio token), coarser than Phi4-MM's 10 ms. Audio is padded to the
  next 30 s chunk, so we slice ``input_slice_start_end`` to the actual
  audio span.

- Forced-target prompting: we build a chat conversation with the assistant
  turn pre-filled with the reference transcription, then locate the
  transcription tokens by running ``apply_chat_template`` twice (once
  user-only + ``add_generation_prompt=True``, once full conversation) and
  using the length difference. Less elegant than Phi4-MM's literal prompt
  string but isolates us from chat-template churn.
"""

from __future__ import annotations

from typing import Optional, Union, Any, Sequence, List, Dict
import os
import sys
import tempfile
import time

import numpy as np
import torch

from i6_experiments.users.zeyer.torch.report_dev_memory_stats import report_dev_memory_stats
from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from ..logits_transform import make_logits_transform
from .base import BaseModelInterface, ForwardOutput


OVERLAY_PATH = "/home/az668407/work/voxtral-overlay"


def _activate_overlay() -> None:
    """Prepend the overlay site-packages so newer transformers/tokenizers/
    mistral_common shadow the env versions for this process only."""
    if OVERLAY_PATH not in sys.path:
        sys.path.insert(0, OVERLAY_PATH)


class Voxtral(BaseModelInterface):
    """Voxtral (Mistral AI) audio LM. See module docstring for design notes."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        speech_prompt: str = "Transcribe the audio clip into text.",
        grad_wrt: str = "speech_embeddings",
        logits_transform: Union[None, str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]] = None,
    ):
        """
        :param model_dir: HF hub cache dir (e.g. via DownloadHuggingFaceRepoJobV2).
            Expected repo: ``mistralai/Voxtral-Mini-3B-2507`` or larger.
        :param speech_prompt: text-only part of the user turn (the transcription
            request); not yet meaningfully used by Voxtral's transcription path
            but kept for parity with Phi4MM.
        """
        super().__init__()
        _activate_overlay()

        self.device = device
        self.model_dir = model_dir
        self.speech_prompt = speech_prompt
        self.grad_wrt = grad_wrt
        self.logits_transform = make_logits_transform(logits_transform)

        print("Import Voxtral / transformers (from overlay)...")
        start_time = time.time()
        import transformers as _tf
        from transformers import AutoProcessor, VoxtralForConditionalGeneration
        print(f"  transformers={_tf.__version__} from {_tf.__file__}")
        print(f"  ({time.time() - start_time:.1f}s)")

        model_dir_str = get_content_dir_from_hub_cache_dir(self.model_dir)
        print("Loading Voxtral...")
        start_time = time.time()
        self.processor = AutoProcessor.from_pretrained(model_dir_str)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_dir_str, dtype=torch.bfloat16, device_map=str(device)
        ).to(device)
        print(f"  ({time.time() - start_time:.1f}s)")
        print(self.model)
        print("model.dtype:", self.model.dtype)

        # Special token IDs we need.
        self.audio_token_id = int(self.model.config.audio_token_id)
        tokenizer = self.processor.tokenizer
        # Mistral tokenizers expose eos_token_id directly.
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            eos = self.model.config.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = int(eos[0])
        self.assistant_end_token_id = int(eos)

    # ---- Helpers --------------------------------------------------------

    def _save_audio_tmp(self, audio: torch.Tensor, sample_rate: int) -> str:
        """Save a 1D audio tensor to a temporary WAV file. Returns the path.

        We do this because Voxtral's chat template content items expect
        ``{"type": "audio", "path": ...}``. The processor will re-load from
        disk via soundfile / torchaudio. Cheap for TIMIT-scale utterances
        (~50 KB per 3-second clip).
        """
        import soundfile as sf
        path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(path, audio.cpu().numpy().astype(np.float32), sample_rate)
        return path

    def _build_chat_inputs(self, *, audio_path: str, transcription: Optional[str]):
        """Build processor inputs for a single utterance.

        If ``transcription`` is None: open-recog mode -- the conversation
        ends after the user turn with ``add_generation_prompt=True``, ready
        for ``model.generate``.

        If ``transcription`` is given: forced mode -- adds an assistant turn
        pre-filled with the transcription. Returns a tuple
        ``(inputs_full, dst_text_start)`` where ``dst_text_start`` is the
        token index at which the transcription begins inside ``input_ids``.
        """
        user_msg = {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": self.speech_prompt},
            ],
        }
        inputs_user = self.processor.apply_chat_template([user_msg], add_generation_prompt=True)
        if transcription is None:
            return inputs_user, None

        # Forced-target: VoxtralProcessor.apply_chat_template won't accept a
        # trailing assistant message (mistral_common's validator rejects it,
        # and both ``continue_final_message=True`` and per-message
        # ``prefix=True`` get dropped by the processor's kwarg filter /
        # dict->AssistantMessage conversion). Instead, get the user-only
        # prompt + audio features above, tokenize the transcription with the
        # raw tokenizer, and concatenate the IDs manually. The assistant turn
        # is conventionally prefixed with a space so per-token decoding sees
        # word-start markers consistently.
        dst_text_start = int(inputs_user["input_ids"].shape[1])
        transc_text = transcription if transcription.startswith(" ") else " " + transcription
        transc_ids = self.processor.tokenizer(
            transc_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        eos_col = torch.tensor(
            [[self.assistant_end_token_id]], dtype=transc_ids.dtype
        )
        transc_ids = torch.cat([transc_ids, eos_col], dim=1)
        inputs_user["input_ids"] = torch.cat(
            [inputs_user["input_ids"], transc_ids], dim=1
        )
        if "attention_mask" in inputs_user:
            inputs_user["attention_mask"] = torch.cat(
                [inputs_user["attention_mask"], torch.ones_like(transc_ids)], dim=1
            )
        return inputs_user, dst_text_start

    def _splice_audio_into_embeds(self, inputs: Dict[str, torch.Tensor]):
        """Return (audio_embeds, inputs_embeds) where audio_embeds has
        ``requires_grad=True`` and ``inputs_embeds`` is the merged text +
        audio embedding sequence ready for ``model(inputs_embeds=...)``.
        """
        # Compute audio embeddings explicitly to expose them as a grad-able
        # tensor (default forward hides them inside masked_scatter).
        print(f"  [splice] input_features shape={tuple(inputs['input_features'].shape)} dtype={inputs['input_features'].dtype}", flush=True)
        audio_embeds = self.model.get_audio_features(inputs["input_features"])
        # Newer Voxtral wraps the encoder output in a ModelOutput; unwrap.
        if not isinstance(audio_embeds, torch.Tensor):
            audio_embeds = getattr(audio_embeds, "last_hidden_state", None) or audio_embeds[0]
        torch.cuda.synchronize()
        print(f"  [splice] get_audio_features -> shape={tuple(audio_embeds.shape)} dtype={audio_embeds.dtype}", flush=True)
        # get_audio_features returns shape [num_audio_tokens, hidden_size];
        # transformers reshapes internally before scatter.
        if audio_embeds.dim() == 3:
            # Some versions return [B, T, H]; flatten for masked_scatter.
            audio_embeds = audio_embeds.reshape(-1, audio_embeds.shape[-1])
        audio_embeds.requires_grad_(True)
        audio_embeds.retain_grad()
        print(f"  [splice] retain_grad ok; audio_embeds.requires_grad={audio_embeds.requires_grad}", flush=True)

        text_embeds = self.model.get_input_embeddings()(inputs["input_ids"])  # [B, T, H]
        torch.cuda.synchronize()
        print(f"  [splice] text_embeds shape={tuple(text_embeds.shape)} dtype={text_embeds.dtype}", flush=True)
        mask = (inputs["input_ids"] == self.audio_token_id).unsqueeze(-1).expand_as(text_embeds)
        n_audio_slots = int(mask[..., 0].sum().item())
        print(f"  [splice] audio mask: {n_audio_slots} slots, audio_embeds has {audio_embeds.shape[0]} rows", flush=True)
        merged = text_embeds.masked_scatter(mask, audio_embeds.to(text_embeds.dtype))
        torch.cuda.synchronize()
        print(f"  [splice] masked_scatter ok; merged shape={tuple(merged.shape)}", flush=True)
        return audio_embeds, merged

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
        assert raw_inputs_sample_rate is not None
        assert (len(raw_inputs),) == raw_input_seq_lens.shape == (len(raw_targets),) == raw_target_seq_lens.shape
        assert len(raw_inputs) == 1, "Voxtral wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        assert raw_input_seq_lens[0] == raw_inputs.shape[1]

        dev = self.device
        words = raw_targets[0]
        transcription = " ".join(words)
        # Note: ``omitted_prev_context`` from chunked datasets is not yet
        # supported for Voxtral -- TIMIT is single-chunk so this is fine for
        # the first pass; Buckeye long-form would need similar handling to
        # Phi4MM's "... " prefix.
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("Voxtral chunked context not implemented yet")

        print(f"[fwd] start; words={len(words)} transcription={transcription!r}", flush=True)
        audio_path = self._save_audio_tmp(raw_inputs[0], raw_inputs_sample_rate)
        try:
            inputs, dst_text_start = self._build_chat_inputs(
                audio_path=audio_path, transcription=transcription
            )
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        print(f"[fwd] chat_inputs ok; dst_text_start={dst_text_start} input_ids.shape={tuple(inputs['input_ids'].shape)}", flush=True)
        inputs = inputs.to(dev)
        input_ids = inputs["input_ids"]
        assert input_ids.shape[0] == 1
        dst_text_end = input_ids.shape[1] - 1  # exclude trailing EOS
        print(f"[fwd] inputs.to(dev) ok; dst_text_end={dst_text_end}", flush=True)

        audio_embeds, inputs_embeds = self._splice_audio_into_embeds(inputs)
        attention_mask = inputs.get("attention_mask")
        print(f"[fwd] splice ok; about to run language model forward (inputs_embeds.shape={tuple(inputs_embeds.shape)})", flush=True)

        report_dev_memory_stats(dev)
        res = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        torch.cuda.synchronize()
        print(f"[fwd] model(...) returned; hidden_states={len(res.hidden_states)} layers, last shape={tuple(res.hidden_states[-1].shape)}", flush=True)
        last_out = res.hidden_states[-1]  # [B, T, H]
        del res
        assert last_out.shape[:2] == input_ids.shape
        report_dev_memory_stats(dev)
        print(f"[fwd] returning ForwardOutput (n_audio_total={int(audio_embeds.shape[0])}, target_len={int(input_ids.shape[1] - dst_text_start)})", flush=True)

        targets = input_ids[:, dst_text_start:dst_text_end]
        n_targets = int(targets.shape[1])
        targets = torch.cat(
            [targets, torch.tensor([[self.assistant_end_token_id]], device=targets.device, dtype=targets.dtype)],
            dim=1,
        )  # [B, T'+1] -- EOS appended for chunk-exit log_prob lookups

        # Per-word target_start_end via incremental decode. Mistral's
        # SentencePiece-like tokenizer marks word starts with a leading
        # space in the decoded token text.
        tokenizer = self.processor.tokenizer
        words_start_end: List[List[int]] = []
        words_: List[str] = []
        for t in range(n_targets):
            s = tokenizer.decode(targets[0, t : t + 1].tolist(), skip_special_tokens=True)
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

        # Slice audio_embeds to the real-audio span (drop Whisper-style
        # 30-s padding). Whisper encoder: 16 kHz input, 10 ms log-mel hop,
        # 2x downsample = 320 samples per output frame.
        n_audio_total = int(audio_embeds.shape[0])
        n_samples = int(raw_input_seq_lens[0])
        n_audio_real = min(n_audio_total, (n_samples + 319) // 320)
        input_slice = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([n_audio_real], dtype=torch.int64),
        )

        edges = torch.arange(n_audio_real + 1, dtype=torch.float64) * (n_samples / max(n_audio_real, 1))
        input_raw_start_end = torch.stack(
            [edges[:-1].round().long(), edges[1:].round().long()], dim=-1
        ).unsqueeze(0)  # [1, n_audio_real, 2]

        # ``inputs`` in ForwardOutput is [B, T_in, F_in]. audio_embeds is
        # already [num_audio_tokens, hidden] -- reshape to [1, T, H].
        audio_embeds_b = audio_embeds.unsqueeze(0)  # [1, n_audio_total, hidden]

        return ForwardOutput(
            inputs=audio_embeds_b,
            input_seq_lens=torch.tensor([n_audio_total]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(words_start_end, dtype=torch.int64).unsqueeze(0),
            outputs=dict(dst_text_start=dst_text_start, last_out=last_out),
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
        dst_text_start = forward_output.outputs["dst_text_start"]
        last_out = batch_slice(last_out, (dst_text_start + start - 1, dst_text_start + end - 1))
        logits = self.model.lm_head(last_out).float()
        for f in self.logits_transform:
            logits = f(logits)
        return logits.log_softmax(-1)

    # ---- Recog (open recognition, greedy) -------------------------------

    def recog(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        raw_input_seq_lens: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> List[List[str]]:
        """Greedy generation with no pre-filled assistant turn. Returns the
        decoded hyp text whitespace-split into words. Normalization for WER
        is left to the caller (apply ``text_dict_normalize_file`` etc. as a
        post-proc step)."""
        assert len(raw_inputs) == 1
        audio_path = self._save_audio_tmp(raw_inputs[0], raw_inputs_sample_rate)
        try:
            inputs, _ = self._build_chat_inputs(audio_path=audio_path, transcription=None)
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        inputs = inputs.to(self.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=self.assistant_end_token_id,
                pad_token_id=self.assistant_end_token_id,
            )
        hyp_ids = output_ids[0, prompt_len:]
        eos_idx = (hyp_ids == self.assistant_end_token_id).nonzero(as_tuple=False)
        if eos_idx.numel() > 0:
            hyp_ids = hyp_ids[: int(eos_idx[0, 0])]
        hyp_text = self.processor.tokenizer.decode(hyp_ids, skip_special_tokens=True)
        return [hyp_text.strip().split()]
