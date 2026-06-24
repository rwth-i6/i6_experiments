"""Qwen2.5-Omni (Thinker) model adapter for the grad-based forced-alignment pipeline.

Mirrors :class:`Voxtral` (forward -> ForwardOutput, log_probs, recog),
but for Alibaba's Qwen2.5-Omni unified ASR+TTS speech LLM.
This wrapper covers the **ASR direction** (audio -> text):
we drive the *Thinker* sub-model only
(the text-producing half; the Talker speech-codec head is unused here).

Notes
-----
- Overlay: Qwen2.5-Omni lives in transformers >= 4.52,
  while our main env is pinned to 4.51.
  We reuse the existing ``voxtral-overlay``
  (transformers 5.9.0, which ships all ``Qwen2_5Omni*`` classes)
  via the same ``sys.path.insert(0, OVERLAY_PATH)`` trick as :class:`Voxtral`.

- Audio embedding access:
  the Thinker's standard forward splices audio features into ``inputs_embeds``
  via ``masked_scatter`` at ``audio_token_id`` positions.
  To expose a grad-able audio tensor we reproduce that splice ourselves:
  ``get_audio_features`` -> ``requires_grad_(True).retain_grad()``
  -> manual ``masked_scatter`` -> ``model(inputs_embeds=merged, ...)``.
  Backward then gives ``d log p / d audio_features``.

- TMRoPE:
  unlike Voxtral (1-D RoPE),
  Qwen2.5-Omni uses a multimodal positional encoding
  computed from ``input_ids`` + audio token lengths.
  So we pass ``input_ids`` AND ``feature_attention_mask``
  alongside our spliced ``inputs_embeds``;
  the forward then keeps our embeds
  (since ``inputs_embeds`` is not None and we pass no ``input_features``,
  the internal audio merge is skipped)
  but still derives correct positions via ``get_rope_index``.

- Audio frame rate:
  the Qwen audio encoder runs a 100 Hz log-mel front-end,
  then a conv (stride 2) + a pool (stride 2),
  i.e. ~25 Hz / 40 ms per audio token.
  Qwen does NOT pad audio to a fixed 30 s window
  (it chunks variable length, and ``get_audio_features`` returns only the real frames),
  so unlike Voxtral there is no silence-padding slice to undo.

- Forced-target prompting:
  we build the user turn (audio + a transcribe instruction)
  with the chat template + ``add_generation_prompt=True``,
  run it through the processor (which expands ``<|AUDIO|>`` into N audio tokens),
  then append the reference transcription token IDs manually --
  same idea as :meth:`Voxtral._build_chat_inputs`.
"""

from __future__ import annotations

from typing import Optional, Union, Any, Sequence, List, Dict
import sys
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
    """Prepend the overlay site-packages
    so the newer transformers (with the ``Qwen2_5Omni*`` classes)
    shadows the env version for this process only."""
    if OVERLAY_PATH not in sys.path:
        sys.path.insert(0, OVERLAY_PATH)


class QwenOmni(BaseModelInterface):
    """Qwen2.5-Omni Thinker (ASR direction). See module docstring for design notes."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        speech_prompt: str = "Transcribe the English audio into text without any punctuation.",
        grad_wrt: str = "speech_embeddings",
        char_level: bool = False,
        char_level_sep: Optional[str] = None,
        logits_transform: Union[None, str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]] = None,
        attn_implementation: Optional[str] = None,
        version: int = 1,
    ):
        """
        :param model_dir: HF hub cache dir (e.g. via DownloadHuggingFaceRepoJobV2).
            Expected repo: ``Qwen/Qwen2.5-Omni-3B`` (or 7B).
        :param speech_prompt: the text part of the user turn (transcription request).
        :param grad_wrt: ``"speech_embeddings"`` (default):
            grad leaf is the post-encoder/projector audio embeds (~25 Hz / 40 ms).
            ``"log_mel"``: grad leaf is the log-mel input feature (100 Hz),
            encoder stays in the graph for ~4x finer resolution.
        :param char_level: explode words into chars
            (per-char vocab lookup so the BPE merger does not recombine)
            for finer target granularity.
        """
        super().__init__()
        _activate_overlay()

        self.device = device
        self.model_dir = model_dir
        self.speech_prompt = speech_prompt
        self.grad_wrt = grad_wrt
        assert grad_wrt in ("speech_embeddings", "log_mel"), grad_wrt
        self._char_level = bool(char_level)
        self._char_level_sep = char_level_sep
        self.logits_transform = make_logits_transform(logits_transform)
        self.version = version

        print("Import Qwen2.5-Omni / transformers (from overlay)...")
        start_time = time.time()
        import transformers as _tf
        from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        print(f"  transformers={_tf.__version__} from {_tf.__file__}")
        print(f"  ({time.time() - start_time:.1f}s)")

        model_dir_str = get_content_dir_from_hub_cache_dir(self.model_dir)
        print("Loading Qwen2.5-Omni Thinker...")
        start_time = time.time()
        self.processor = AutoProcessor.from_pretrained(model_dir_str)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_dir_str,
            dtype=torch.bfloat16,
            device_map=str(device),
            # eager attention is required for output_attentions (self-attn alignment jobs)
            **({"attn_implementation": attn_implementation} if attn_implementation else {}),
        ).to(device)
        print(f"  ({time.time() - start_time:.1f}s)")
        print(self.model)
        print("model.dtype:", self.model.dtype)

        # Special token IDs.
        # The Thinker config exposes the audio placeholder id as ``audio_token_id``
        # (older configs as ``audio_token_index``).
        self.audio_token_id = int(
            getattr(self.model.config, "audio_token_id", None)
            if getattr(self.model.config, "audio_token_id", None) is not None
            else self.model.config.audio_token_index
        )
        eos = self.model.config.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = int(eos[0])
        self.assistant_end_token_id = int(eos)
        print(f"  audio_token_id={self.audio_token_id} eos={self.assistant_end_token_id}")

    # ---- Helpers --------------------------------------------------------

    def _build_inputs(self, *, audio_np: np.ndarray, sample_rate: int, transcription_ids: torch.Tensor):
        """Build forced-target inputs for one utterance.

        Returns ``(inputs, dst_text_start)``,
        where ``inputs`` is a plain dict with
        ``input_ids`` (audio already expanded + reference appended),
        ``input_features``, and ``feature_attention_mask``,
        and ``dst_text_start`` is the token index
        at which the reference transcription begins.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_np},
                    {"type": "text", "text": self.speech_prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        pre = self.processor(
            text=[text],
            audio=[audio_np],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        dst_text_start = int(pre["input_ids"].shape[1])
        eos_col = torch.tensor([[self.assistant_end_token_id]], dtype=transcription_ids.dtype)
        transc_ids = torch.cat([transcription_ids, eos_col], dim=1)
        new_input_ids = torch.cat([pre["input_ids"], transc_ids], dim=1)
        out = {
            "input_ids": new_input_ids,
            "input_features": pre["input_features"],
            "feature_attention_mask": pre["feature_attention_mask"],
        }
        return out, dst_text_start

    def _splice_audio_into_embeds(self, inputs: Dict[str, torch.Tensor]):
        """Return ``(grad_leaf, inputs_embeds)``.

        ``grad_leaf`` is the differentiation target:
        a [1, n_audio, H] (speech_embeddings)
        or [1, n_mel_frames, n_mel] (log_mel) tensor,
        with ``requires_grad=True`` + ``retain_grad()``.
        ``inputs_embeds`` is the merged text + audio embedding sequence,
        ready for ``model(inputs_embeds=...)``.
        """
        input_features = inputs["input_features"]  # [1, n_mel, n_mel_frames]
        feature_attention_mask = inputs["feature_attention_mask"]  # [1, n_mel_frames]

        if self.grad_wrt == "log_mel":
            # Grad leaf is the log-mel features [1, n_mel_frames, n_mel];
            # encoder + projector stay in the graph,
            # so grad flows back to the 100 Hz input.
            feat_T = input_features.transpose(1, 2).contiguous().requires_grad_(True)
            feat_T.retain_grad()
            audio_out = self.model.get_audio_features(
                feat_T.transpose(1, 2), feature_attention_mask=feature_attention_mask, return_dict=True
            ).last_hidden_state
            audio_embeds = audio_out.unsqueeze(0)  # NO detach -- grad flows through encoder.
            grad_leaf = feat_T
        else:  # speech_embeddings (default)
            audio_out = self.model.get_audio_features(
                input_features, feature_attention_mask=feature_attention_mask, return_dict=True
            ).last_hidden_state  # [n_audio, H] (real frames only, no padding)
            audio_embeds = audio_out.detach().unsqueeze(0).requires_grad_(True)
            audio_embeds.retain_grad()
            grad_leaf = audio_embeds

        text_embeds = self.model.get_input_embeddings()(inputs["input_ids"])  # [1, T, H]
        audio_token_mask = inputs["input_ids"] == self.audio_token_id  # [1, T]
        n_slots = int(audio_token_mask.sum().item())
        assert n_slots == audio_embeds.shape[1], (
            f"audio placeholder slots ({n_slots}) != audio embeds ({audio_embeds.shape[1]})"
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
        collect_attentions: Optional[list] = None,
    ) -> ForwardOutput:
        """See :class:`BaseModelInterface`."""
        assert raw_inputs_sample_rate is not None
        assert (len(raw_inputs),) == raw_input_seq_lens.shape == (len(raw_targets),) == raw_target_seq_lens.shape
        assert len(raw_inputs) == 1, "QwenOmni wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        assert raw_input_seq_lens[0] == raw_inputs.shape[1]
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("QwenOmni chunked context not implemented yet")

        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])

        sample_rate = int(raw_inputs_sample_rate)
        if sample_rate != 16000:
            import torchaudio

            raw_inputs = torchaudio.functional.resample(raw_inputs, sample_rate, 16000)
            sample_rate = 16000
        audio_np = raw_inputs[0].detach().cpu().numpy().astype(np.float32)

        tok = self.processor.tokenizer
        # --- target token IDs + per-word token ranges --------------------
        word_char_ranges: Optional[List[tuple]] = None
        if self._char_level:
            # Per-char vocab lookup (a char may map to >1 token);
            # record per-word [start,end) in TOKENS.
            char_ids: List[int] = []
            chars: List[str] = []
            word_char_ranges = []
            for wi, word in enumerate(words):
                if self._char_level_sep and wi > 0:
                    chars.append(self._char_level_sep)
                    char_ids.extend(tok.encode(self._char_level_sep, add_special_tokens=False))
                tstart = len(char_ids)
                for ch in word:
                    ids = tok.encode(ch, add_special_tokens=False)
                    assert len(ids) >= 1, f"char {ch!r} tokenizes to 0 tokens"
                    chars.append(ch)
                    char_ids.extend(ids)
                word_char_ranges.append((tstart, len(char_ids)))
            target_ids = torch.tensor([char_ids], dtype=torch.long)
            transcription = "".join(chars)
        else:
            transcription = " ".join(words)
            transc_text = transcription if transcription.startswith(" ") else " " + transcription
            target_ids = tok(transc_text, add_special_tokens=False, return_tensors="pt")["input_ids"]

        print(f"[fwd] start; words={len(words)} transcription={transcription!r}", flush=True)
        inputs, dst_text_start = self._build_inputs(
            audio_np=audio_np, sample_rate=sample_rate, transcription_ids=target_ids
        )
        inputs = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        assert input_ids.shape[0] == 1
        dst_text_end = input_ids.shape[1] - 1  # exclude trailing EOS
        print(
            f"[fwd] inputs ok; dst_text_start={dst_text_start} dst_text_end={dst_text_end} "
            f"input_ids.shape={tuple(input_ids.shape)}",
            flush=True,
        )

        audio_embeds, inputs_embeds = self._splice_audio_into_embeds(inputs)
        attention_mask = torch.ones_like(input_ids)
        print(
            f"[fwd] splice ok; about to run Thinker forward (inputs_embeds.shape={tuple(inputs_embeds.shape)})",
            flush=True,
        )

        report_dev_memory_stats(dev)
        res = self.model(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            feature_attention_mask=inputs["feature_attention_mask"],
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=collect_attentions is not None,
        )
        torch.cuda.synchronize()
        last_out = res.hidden_states[-1]  # [B, T, H]
        print(
            f"[fwd] model(...) returned; hidden_states={len(res.hidden_states)} layers, "
            f"last shape={tuple(last_out.shape)}",
            flush=True,
        )
        if collect_attentions is not None:
            audio_pos = (input_ids[0] == self.audio_token_id).nonzero(as_tuple=True)[0]
            a0, a1 = int(audio_pos[0]), int(audio_pos[-1]) + 1
            assert a1 - a0 == int(audio_pos.numel()), "audio token block not contiguous"
            n_tgt = int(input_ids.shape[1] - 1 - dst_text_start)
            rows = torch.arange(dst_text_start - 1, dst_text_start - 1 + n_tgt, device=input_ids.device)
            collect_attentions.append(
                dict(
                    attns=[a[0][:, rows][:, :, a0:a1].float().cpu() for a in res.attentions],
                    n_audio=a1 - a0,
                    n_audio_real=a1 - a0,
                )
            )
        del res
        assert last_out.shape[:2] == input_ids.shape
        report_dev_memory_stats(dev)

        targets = input_ids[:, dst_text_start:dst_text_end]
        n_targets = int(targets.shape[1])
        targets = torch.cat(
            [targets, torch.tensor([[self.assistant_end_token_id]], device=targets.device, dtype=targets.dtype)],
            dim=1,
        )  # [B, T'+1] -- EOS appended for chunk-exit log_prob lookups

        # Per-token -> per-word grouping.
        if self._char_level:
            assert word_char_ranges is not None and len(word_char_ranges) == len(words)
            words_start_end = [[int(a), int(b)] for a, b in word_char_ranges]
        else:
            words_start_end = []
            words_: List[str] = []
            for t in range(n_targets):
                s = tok.decode(targets[0, t : t + 1].tolist(), skip_special_tokens=True)
                if t == 0 or s.startswith(" "):
                    words_start_end.append([t, t + 1])
                    words_.append(s.lstrip(" "))
                else:
                    words_[-1] += s
                    words_start_end[-1][1] = t + 1
            assert words_ == words, f"target_decoded={words_!r} ref={words!r}"
        words_start_end = words_start_end + [[n_targets, n_targets + 1]]  # EOS slot

        # Qwen returns only real audio frames (no 30 s padding) -> full span.
        n_audio_total = int(audio_embeds.shape[1])
        if self.grad_wrt == "log_mel":
            # log-mel grad leaf is at 100 Hz -> use the real mel-frame count.
            n_audio_real = int(inputs["feature_attention_mask"].sum().item())
            n_audio_total = int(audio_embeds.shape[1])
            n_audio_real = min(n_audio_real, n_audio_total)
        else:
            n_audio_real = n_audio_total
        input_slice = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([n_audio_real], dtype=torch.int64),
        )
        edges = torch.arange(n_audio_real + 1, dtype=torch.float64) * (orig_n_samples / max(n_audio_real, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(
            0
        )  # [1, n_audio_real, 2]

        print(
            f"[fwd] returning ForwardOutput (n_audio_total={n_audio_total}, n_audio_real={n_audio_real}, "
            f"target_len={int(targets.shape[1])})",
            flush=True,
        )
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
        """Greedy generation with no pre-filled assistant turn. Returns hyp words."""
        assert len(raw_inputs) == 1
        sample_rate = int(raw_inputs_sample_rate)
        if sample_rate != 16000:
            import torchaudio

            raw_inputs = torchaudio.functional.resample(raw_inputs, sample_rate, 16000)
            sample_rate = 16000
        audio_np = raw_inputs[0].detach().cpu().numpy().astype(np.float32)
        conversation = [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_np}, {"type": "text", "text": self.speech_prompt}],
            }
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=[text], audio=[audio_np], sampling_rate=sample_rate, return_tensors="pt", padding=True
        ).to(self.device)
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
