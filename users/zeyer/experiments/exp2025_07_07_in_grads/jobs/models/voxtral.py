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
        recog_mode: str = "chat",
        forward_mode: str = "chat",
        transcription_model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        grad_wrt: str = "speech_embeddings",
        audio_time_stretch: float = 1.0,
        ensure_audio_long_enough: bool = False,
        char_level: bool = False,
        char_level_sep: Optional[str] = None,
        logits_transform: Union[None, str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]] = None,
        version: int = 1,
    ):
        """
        :param model_dir: HF hub cache dir (e.g. via DownloadHuggingFaceRepoJobV2).
            Expected repo: ``mistralai/Voxtral-Mini-3B-2507`` or larger.
        :param speech_prompt: text-only part of the user turn (the transcription
            request). Used in chat-mode recog and in the forced-alignment
            forward path.
        :param recog_mode: ``"transcription"`` uses Voxtral's dedicated
            transcription request (``processor.apply_transcription_request``),
            which is what the OpenASR leaderboard uses and avoids instruct-mode
            paraphrasing. ``"chat"`` uses the apply_chat_template path (kept
            for comparison; tends to paraphrase -> inflated WER). The
            forced-alignment ``forward`` always uses the chat/forced format
            regardless of this flag.
        :param transcription_model_id: hub model id passed to
            ``apply_transcription_request`` (template selection only, no
            network access). Must match the loaded model family.
        """
        super().__init__()
        _activate_overlay()

        self.device = device
        self.model_dir = model_dir
        self.speech_prompt = speech_prompt
        self.recog_mode = recog_mode
        self.forward_mode = forward_mode
        self.transcription_model_id = transcription_model_id
        self.grad_wrt = grad_wrt
        self.audio_time_stretch = float(audio_time_stretch)
        assert self.audio_time_stretch > 0, audio_time_stretch
        self.ensure_audio_long_enough = bool(ensure_audio_long_enough)
        self._char_level = bool(char_level)
        self._char_level_sep = char_level_sep
        self.logits_transform = make_logits_transform(logits_transform)
        assert version >= 3, (
            "version 1: buggy splice (pre-projection encoder states). "
            "version 2: wrong n_audio_real (Whisper encoder frame count, not projected-token count; "
            "4x too many padding frames included in the grad slice). "
            "version >= 3: correct n_audio_real = (n_samples + 1279) // 1280. "
            "(version=1/2 defaults exist only for hash stability of old finished jobs.)"
        )
        assert grad_wrt in ("speech_embeddings", "log_mel", "encoder_conv1_out"), grad_wrt
        if grad_wrt != "speech_embeddings":
            assert version >= 4, (
                "version >= 4 required when grad_wrt != 'speech_embeddings' (log_mel grad path enabled)."
            )
        if grad_wrt == "encoder_conv1_out":
            assert version >= 7, (
                "version >= 7 required when grad_wrt == 'encoder_conv1_out' "
                "(v6: retain_grad bug fixed with requires_grad_ first)."
            )
        if self._char_level:
            assert version >= 7, "version >= 7 required when char_level=True (per-char vocab lookup for input_ids)."
        if self.ensure_audio_long_enough:
            assert version >= 8, (
                "version >= 8 required when ensure_audio_long_enough=True "
                "(per-seq just-enough time-stretch when S > T_default)."
            )
        if self.audio_time_stretch != 1.0:
            assert version >= 5, (
                "version >= 5 required when audio_time_stretch != 1.0 (time-stretch audio preprocessing enabled)."
            )
        self.version = version

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
        # raw tokenizer, and concatenate the IDs manually. Return a plain
        # dict -- mutating the BatchFeature wrapper doesn't update its
        # internal tensor list, so a later ``.to(dev)`` would leave the
        # replaced tensors on CPU.
        dst_text_start = int(inputs_user["input_ids"].shape[1])
        transc_text = transcription if transcription.startswith(" ") else " " + transcription
        transc_ids = self.processor.tokenizer(transc_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        eos_col = torch.tensor([[self.assistant_end_token_id]], dtype=transc_ids.dtype)
        transc_ids = torch.cat([transc_ids, eos_col], dim=1)
        new_input_ids = torch.cat([inputs_user["input_ids"], transc_ids], dim=1)
        out = {
            "input_ids": new_input_ids,
            "input_features": inputs_user["input_features"],
        }
        if "attention_mask" in inputs_user:
            out["attention_mask"] = torch.cat([inputs_user["attention_mask"], torch.ones_like(transc_ids)], dim=1)
        return out, dst_text_start

    def _build_transcription_inputs(
        self,
        *,
        audio_path: str,
        transcription: Optional[str],
        transcription_ids: Optional[torch.Tensor] = None,
    ):
        """Forced-target inputs in Voxtral's native transcription mode.

        ``apply_transcription_request`` builds the prefix ``[start, <audio>,
        lang:en, TRANSCRIBE]`` -- the exact context the model uses for ASR
        (no instruct wrapper, so audio attention/gradients localize better
        than chat mode). For forced alignment we tokenize the reference and
        append it after the prefix, mirroring :meth:`_build_chat_inputs`.

        If ``transcription_ids`` is provided (shape ``[1, n_tokens]``), it is
        used verbatim instead of tokenizing ``transcription``. This bypasses
        the BPE merger so each char gets its own token in char-level mode
        (same idea as in :class:`CanaryQwen`).
        """
        pre = self.processor.apply_transcription_request(
            language="en",
            audio=audio_path,
            model_id=self.transcription_model_id,
            return_tensors="pt",
        )
        if transcription is None:
            return pre, None
        dst_text_start = int(pre["input_ids"].shape[1])
        if transcription_ids is not None:
            transc_ids = transcription_ids
        else:
            transc_text = transcription if transcription.startswith(" ") else " " + transcription
            transc_ids = self.processor.tokenizer(transc_text, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ]
        eos_col = torch.tensor([[self.assistant_end_token_id]], dtype=transc_ids.dtype)
        transc_ids = torch.cat([transc_ids, eos_col], dim=1)
        new_input_ids = torch.cat([pre["input_ids"], transc_ids], dim=1)
        out = {
            "input_ids": new_input_ids,
            "input_features": pre["input_features"],
        }
        if "attention_mask" in pre:
            out["attention_mask"] = torch.cat([pre["attention_mask"], torch.ones_like(transc_ids)], dim=1)
        return out, dst_text_start

    def _splice_audio_into_embeds(self, inputs: Dict[str, torch.Tensor]):
        """Return (grad_leaf, inputs_embeds).

        ``grad_leaf`` is a [1, T, F] tensor with ``requires_grad=True`` and
        ``retain_grad()`` set; differentiation target for per-token grads.
        ``inputs_embeds`` is the merged text + audio embedding sequence ready
        for ``model(inputs_embeds=...)``.

        Two variants depending on ``self.grad_wrt``:

        - ``speech_embeddings`` (default): grad leaf is the post-projection
            audio embeds [1, n_projected, H_proj]. Encoder + projector run
            without grad (detached). T resolution = 1280 samples/token
            (~12.5 Hz). Fast, but coarse.
        - ``log_mel``: grad leaf is the log-mel input feature tensor
            transposed to [1, n_mel_frames, n_mel_bins]. The encoder + projector
            stay in the forward graph so grad flows all the way back. T resolution
            = 160 samples/frame (100 Hz). Slower backward but ~8x finer time
            resolution -- needed for char-level alignment when projected
            T < #chars.
        """
        # Replicate VoxtralForConditionalGeneration.forward's audio path:
        #   inputs_embeds[input_ids == audio_token_id] = get_audio_features(feat)
        # get_audio_features ALREADY applies the multimodal projector (encoder
        # 1500 frames -> 375 projected speech tokens via 4x downsample).
        input_features = inputs["input_features"]  # [1, n_mel, n_mel_frames]

        if self.grad_wrt == "log_mel":
            # Grad leaf is the log-mel features, exposed as [1, T, F] = [1, n_mel_frames, n_mel].
            # We requires_grad on the transposed view (which is the leaf),
            # then transpose back for the encoder. Grad flows: input_features_T
            # -> transpose -> encoder -> projector -> LLM -> loss.
            input_features_T = input_features.transpose(1, 2).contiguous().requires_grad_(True)
            input_features_T.retain_grad()
            input_features_for_enc = input_features_T.transpose(1, 2)
            audio_out = self.model.get_audio_features(input_features_for_enc)
            if not isinstance(audio_out, torch.Tensor):
                audio_2d = audio_out.pooler_output
            else:
                audio_2d = audio_out.reshape(-1, audio_out.shape[-1])
            audio_embeds = audio_2d.unsqueeze(0)  # NO detach -- grad flows through encoder.
            grad_leaf = input_features_T
        elif self.grad_wrt == "encoder_conv1_out":
            # Grad leaf is the activation after Whisper encoder's first conv
            # (stride=1, output at 100 Hz in d_model space, e.g. 1280 dim).
            # Captured via a forward hook on conv1. Same 10 ms time resolution
            # as log_mel but in the model's learned feature space -- expected
            # to be cleaner than raw log-mel.
            # Model params are frozen, so conv1's output is NOT in the autograd
            # graph on its own. The hook detaches it into a fresh leaf (in the
            # [1, T_mel, d_model] output layout) and feeds the [1, d_model, T_mel]
            # view back downstream, so conv2..projector..LM (and the loss) depend
            # on the leaf -- the same input-leaf trick as the log_mel path.
            captured: List[torch.Tensor] = []

            def _conv1_hook(_module, _inp, out):
                leaf = out.detach().transpose(1, 2).contiguous().requires_grad_(True)
                leaf.retain_grad()
                captured.append(leaf)
                return leaf.transpose(1, 2)

            conv1_mod = self.model.audio_tower.conv1
            handle = conv1_mod.register_forward_hook(_conv1_hook)
            try:
                audio_out = self.model.get_audio_features(input_features)
            finally:
                handle.remove()
            assert len(captured) == 1, f"expected 1 conv1 call, got {len(captured)}"
            if not isinstance(audio_out, torch.Tensor):
                audio_2d = audio_out.pooler_output
            else:
                audio_2d = audio_out.reshape(-1, audio_out.shape[-1])
            audio_embeds = audio_2d.unsqueeze(0)  # grad flows through encoder to the conv1 leaf.
            grad_leaf = captured[0]  # [1, T_mel, d_model], differentiation target
        else:  # speech_embeddings (default)
            audio_out = self.model.get_audio_features(input_features)
            if not isinstance(audio_out, torch.Tensor):
                audio_2d = audio_out.pooler_output
            else:
                audio_2d = audio_out.reshape(-1, audio_out.shape[-1])
            # Build [1, n_projected, H] grad leaf. detach() so grad flows
            # through projected speech tokens only (not back into Whisper).
            audio_embeds = audio_2d.detach().unsqueeze(0).requires_grad_(True)
            audio_embeds.retain_grad()
            grad_leaf = audio_embeds

        text_embeds = self.model.get_input_embeddings()(inputs["input_ids"])  # [1, T, H]
        audio_token_mask = inputs["input_ids"] == self.audio_token_id  # [1, T]
        n_slots = int(audio_token_mask.sum().item())
        assert n_slots == audio_embeds.shape[1], (
            f"audio placeholder slots ({n_slots}) != projected audio embeds ({audio_embeds.shape[1]})"
        )
        merged = text_embeds.clone()
        merged[audio_token_mask] = audio_embeds[0].to(text_embeds.dtype)
        print(
            f"  [splice] grad_wrt={self.grad_wrt} grad_leaf {tuple(grad_leaf.shape)} "
            f"audio_embeds {tuple(audio_embeds.shape)} merged {tuple(merged.shape)}",
            flush=True,
        )
        return grad_leaf, merged

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

        # ORIGINAL audio length, kept for time-mapping in input_raw_start_end
        # (so word boundaries from the aligner come out in the original audio's
        # timeline -- not the stretched one). Encoder-side calculations use the
        # stretched length below.
        orig_n_samples = int(raw_input_seq_lens[0])

        # --- char-level explosion (variant 12) ----------------------------
        # Explode each word into individual characters, with optional inter-word
        # separator. Build per-char token IDs via direct vocab lookup so the
        # BPE merger doesn't recombine "s h e" -> "she" (same trick as in
        # CanaryQwen). word_char_ranges is used later to re-group per-char
        # target_start_end back to word-level boundaries for the WBE metric.
        word_char_ranges: Optional[List[tuple]] = None
        char_token_ids: Optional[torch.Tensor] = None
        chars: Optional[List[str]] = None
        if self._char_level:
            chars = []
            word_char_ranges = []
            for wi, word in enumerate(words):
                if self._char_level_sep and wi > 0:
                    chars.append(self._char_level_sep)
                cstart = len(chars)
                for ch in word:
                    chars.append(ch)
                word_char_ranges.append((cstart, len(chars)))
            tok = self.processor.tokenizer
            char_ids: List[int] = []
            for ch in chars:
                ids = tok.encode(ch, add_special_tokens=False)
                assert len(ids) == 1, (
                    f"char {ch!r} tokenizes to {len(ids)} tokens {ids} -- "
                    "per-char vocab lookup requires single-token chars"
                )
                char_ids.extend(ids)
            char_token_ids = torch.tensor([char_ids], dtype=torch.long)
            transcription = "".join(chars)  # for display / sanity print

        # --- audio time-stretch (fixed and/or per-seq just-enough) --------
        # Combined: ``audio_time_stretch`` sets a baseline; if
        # ``ensure_audio_long_enough`` and char_level, also bump factor when
        # S (#target tokens) > T_default (default encoder frame count).
        effective_stretch = self.audio_time_stretch
        if self.ensure_audio_long_enough and self._char_level:
            n_target = len(chars)
            samples_per_frame = 160 if self.grad_wrt in ("log_mel", "encoder_conv1_out") else 1280
            T_default = (orig_n_samples + samples_per_frame - 1) // samples_per_frame
            if n_target > T_default:
                required_factor = (n_target * 1.05) / T_default  # 5% margin
                if required_factor > effective_stretch:
                    effective_stretch = required_factor
                    print(
                        f"[fwd] ensure_audio_long_enough: S={n_target} > T_default={T_default} -> "
                        f"stretch={effective_stretch:.3f}",
                        flush=True,
                    )
        if effective_stretch != 1.0:
            import librosa

            audio_np = raw_inputs[0].detach().cpu().numpy().astype(np.float32)
            stretched = librosa.effects.time_stretch(audio_np, rate=1.0 / effective_stretch)
            raw_inputs = torch.tensor(stretched, dtype=raw_inputs.dtype).unsqueeze(0)
            raw_input_seq_lens = torch.tensor([raw_inputs.shape[1]], dtype=raw_input_seq_lens.dtype)
            print(
                f"[fwd] audio_time_stretch={effective_stretch:.3f}: "
                f"orig_samples={orig_n_samples} stretched_samples={raw_inputs.shape[1]}",
                flush=True,
            )
        # Note: ``omitted_prev_context`` from chunked datasets is not yet
        # supported for Voxtral -- TIMIT is single-chunk so this is fine for
        # the first pass; Buckeye long-form would need similar handling to
        # Phi4MM's "... " prefix.
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("Voxtral chunked context not implemented yet")

        print(f"[fwd] start; words={len(words)} transcription={transcription!r}", flush=True)
        audio_path = self._save_audio_tmp(raw_inputs[0], raw_inputs_sample_rate)
        try:
            if self.forward_mode == "transcription":
                inputs, dst_text_start = self._build_transcription_inputs(
                    audio_path=audio_path,
                    transcription=transcription,
                    transcription_ids=char_token_ids,
                )
            else:
                assert not self._char_level, "char_level only supported with forward_mode='transcription' for now"
                inputs, dst_text_start = self._build_chat_inputs(audio_path=audio_path, transcription=transcription)
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        print(
            f"[fwd] {self.forward_mode}_inputs ok; dst_text_start={dst_text_start} input_ids.shape={tuple(inputs['input_ids'].shape)}",
            flush=True,
        )
        # ``inputs`` is a plain dict (forced) or BatchFeature (recog).
        if hasattr(inputs, "to"):
            inputs = inputs.to(dev)
        else:
            inputs = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        assert input_ids.shape[0] == 1
        dst_text_end = input_ids.shape[1] - 1  # exclude trailing EOS
        print(f"[fwd] inputs.to(dev) ok; dst_text_end={dst_text_end}", flush=True)

        audio_embeds, inputs_embeds = self._splice_audio_into_embeds(inputs)
        attention_mask = inputs.get("attention_mask")
        print(
            f"[fwd] splice ok; about to run language model forward (inputs_embeds.shape={tuple(inputs_embeds.shape)})",
            flush=True,
        )

        report_dev_memory_stats(dev)
        res = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        torch.cuda.synchronize()
        print(
            f"[fwd] model(...) returned; hidden_states={len(res.hidden_states)} layers, last shape={tuple(res.hidden_states[-1].shape)}",
            flush=True,
        )
        last_out = res.hidden_states[-1]  # [B, T, H]
        del res
        assert last_out.shape[:2] == input_ids.shape
        report_dev_memory_stats(dev)
        print(
            f"[fwd] returning ForwardOutput (n_audio_total={int(audio_embeds.shape[1])}, target_len={int(input_ids.shape[1] - dst_text_start)})",
            flush=True,
        )

        targets = input_ids[:, dst_text_start:dst_text_end]
        n_targets = int(targets.shape[1])
        targets = torch.cat(
            [targets, torch.tensor([[self.assistant_end_token_id]], device=targets.device, dtype=targets.dtype)],
            dim=1,
        )  # [B, T'+1] -- EOS appended for chunk-exit log_prob lookups

        # Per-token target_start_end. In char_level mode, each token is one
        # char (we built the IDs via per-char vocab lookup), so we know the
        # grouping directly. Otherwise, use the incremental-decode trick:
        # Mistral's SentencePiece-like tokenizer marks word starts with a
        # leading space in the decoded token text.
        tokenizer = self.processor.tokenizer
        words_start_end: List[List[int]] = []
        words_: List[str] = []
        if self._char_level:
            words_start_end = [[t, t + 1] for t in range(n_targets)]
            words_ = list(chars)
            ref_token_list = chars
        else:
            for t in range(n_targets):
                s = tokenizer.decode(targets[0, t : t + 1].tolist(), skip_special_tokens=True)
                if t == 0 or s.startswith(" "):
                    words_start_end.append([t, t + 1])
                    words_.append(s.lstrip(" "))
                else:
                    words_[-1] += s
                    words_start_end[-1][1] = t + 1
            ref_token_list = words
        assert len(words_start_end) == len(words_) == len(ref_token_list), (
            f"word-grouping mismatch: target_decoded={words_!r} ref={ref_token_list!r}"
        )
        assert words_ == ref_token_list, f"target_decoded={words_!r} ref={ref_token_list!r}"
        # Re-group per-char target_start_end back to word level for the WBE
        # metric (which operates on word boundaries, not chars).
        if word_char_ranges is not None:
            regrouped = []
            for cstart, cend in word_char_ranges:
                t0 = int(words_start_end[cstart][0])
                t1 = int(words_start_end[cend - 1][1])
                regrouped.append([t0, t1])
            words_start_end = regrouped
        words_start_end = words_start_end + [[n_targets, n_targets + 1]]  # EOS slot

        # Slice the grad leaf to the real-audio span (drop Whisper-style
        # 30-s silence padding). Samples-per-frame depends on grad_wrt:
        #   speech_embeddings: 1280 samples/projected-token
        #     (10ms log-mel hop * 2x conv * 4x projector = 80ms = 1280 samples).
        #   log_mel: 160 samples/log-mel-frame (16 kHz * 10ms hop).
        n_audio_total = int(audio_embeds.shape[1])
        n_samples = int(raw_input_seq_lens[0])
        if self.grad_wrt in ("log_mel", "encoder_conv1_out"):
            # conv1 is stride=1, so its output is also at 100 Hz (160 samples/frame).
            samples_per_frame = 160
        else:
            samples_per_frame = 1280
        n_audio_real = min(n_audio_total, (n_samples + samples_per_frame - 1) // samples_per_frame)
        input_slice = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([n_audio_real], dtype=torch.int64),
        )

        # Use ORIGINAL audio length for time mapping so word boundaries from
        # the aligner come out in the original audio's timeline (relevant when
        # audio_time_stretch != 1.0 -- otherwise orig_n_samples == n_samples).
        edges = torch.arange(n_audio_real + 1, dtype=torch.float64) * (orig_n_samples / max(n_audio_real, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(
            0
        )  # [1, n_audio_real, 2]

        # ``audio_embeds`` is already the [1, N, H] grad leaf (built in
        # _splice_audio_into_embeds) and is the exact tensor consumed by the
        # forward graph, so it can be the differentiation target directly.
        return ForwardOutput(
            inputs=audio_embeds,
            input_seq_lens=torch.tensor([n_audio_total]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(words_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
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
        # Newer Voxtral has lm_head on the inner LlamaForCausalLM, not the
        # top-level VoxtralForConditionalGeneration.
        lm_head = getattr(self.model, "lm_head", None) or self.model.language_model.lm_head
        logits = lm_head(last_out).float()
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
        if self.recog_mode == "transcription":
            return self._recog_transcription(
                raw_inputs=raw_inputs,
                raw_inputs_sample_rate=raw_inputs_sample_rate,
                max_new_tokens=max_new_tokens,
            )
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

    def _recog_transcription(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        max_new_tokens: int = 100,
    ) -> List[List[str]]:
        """Voxtral's dedicated transcription request -- the mode the OpenASR
        leaderboard uses (``processor.apply_transcription_request``). Avoids
        instruct-mode paraphrasing. We pass the audio as a temp WAV path so
        the processor's ``is_str`` branch loads it directly (no format/
        sampling_rate juggling)."""
        audio_path = self._save_audio_tmp(raw_inputs[0], raw_inputs_sample_rate)
        try:
            inputs = self.processor.apply_transcription_request(
                language="en",
                audio=audio_path,
                model_id=self.transcription_model_id,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device, dtype=torch.bfloat16)
            prompt_len = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
            hyp_text = self.processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0]
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        return [hyp_text.strip().split()]
