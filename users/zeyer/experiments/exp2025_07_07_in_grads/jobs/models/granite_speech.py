"""IBM Granite-Speech model adapter (``ibm-granite/granite-speech-3.3-8b``).

Architecture: block-attention conformer encoder (4 s audio blocks, self-conditioned CTC)
-> 2-layer window q-former projector (10 Hz acoustic embedding rate)
-> Granite-3.3-8b LLM (RoPE, 128k context) with LoRA adapters for audio turns.
No fixed audio window, so long-form input is structurally unconstrained
(the interesting contrast to the Whisper family and Canary-1B).

Forward path (forced alignment):
the processor expands the single ``<|audio|>`` in the prompt
to one token per 10 Hz acoustic embedding,
and the model splices the projected audio embeddings at those positions internally,
so ``input_ids`` coordinates equal embedding coordinates
(unlike the Canary-Qwen adapter's embed-level splice).
Teacher-force the reference transcription as the assistant turn,
keep the last hidden state for ``log_probs``.

The env's transformers 4.51 predates GraniteSpeech,
so the adapter prepends a transformers 4.53 overlay (``pip install --target``)
before the first transformers import;
use this wrapper only in granite-only jobs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import os
import sys
import time

import numpy as np
import torch

from i6_experiments.users.zeyer.torch.batch_slice import batch_slice
from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from ..logits_transform import make_logits_transform
from .base import BaseModelInterface, ForwardOutput

OVERLAY_PATH = "/home/az668407/work/transformers-granite-overlay"


def _activate_overlay() -> None:
    """Prepend the granite transformers overlay (>=4.52 needed for GraniteSpeech).

    Whole-process switch: transformers must not be imported yet,
    else the 4.51 modules stay cached and the class lookup fails."""
    if OVERLAY_PATH not in sys.path:
        assert "transformers" not in sys.modules, (
            "granite overlay must be activated before any transformers import in this process"
        )
        sys.path.insert(0, OVERLAY_PATH)


class GraniteSpeech(BaseModelInterface):
    """IBM Granite-Speech (conformer encoder + q-former + Granite LLM with LoRA)."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        speech_prompt: str = "can you transcribe the speech into a written format?",
        logits_transform: Union[None, str, Dict[str, Any], Sequence[Union[str, Dict[str, Any]]]] = None,
        grad_wrt: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        collect_attn_heads: Optional[List[List[int]]] = None,
        version: int = 1,
    ):
        """
        :param model_dir: hub cache dir for ``ibm-granite/granite-speech-3.3-8b``.
        :param speech_prompt: user-turn text after the audio token
            (the model card's documented ASR prompt).
        :param grad_wrt: only ``None`` -- gradient extraction is not implemented for this adapter
            (the zoo uses it for chunk-align scoring and self-attention extraction only).
        :param attn_implementation: LLM attention override
            (``"eager"`` needed for ``collect_attentions``; SDPA returns no weights).
        :param collect_attn_heads: long-form, see :class:`CaptureSelectedAttn`.
        """
        super().__init__()
        _activate_overlay()

        assert grad_wrt is None, f"GraniteSpeech: gradient extraction not implemented ({grad_wrt=})"
        assert version >= 1
        self.device = device
        self.model_dir = model_dir
        self.speech_prompt = speech_prompt
        self.logits_transform = make_logits_transform(logits_transform)
        self.collect_attn_heads = collect_attn_heads
        self.version = version

        # Compute nodes are offline; the main model loads by path,
        # but the peft LoRA adapter resolves by repo id through the hub cache.
        os.environ["HF_HUB_CACHE"] = str(model_dir)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        print("Import transformers (from overlay) / load GraniteSpeech...")
        start_time = time.time()
        import transformers
        from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
        import peft

        print(f"  transformers={transformers.__version__} from {transformers.__file__}")
        print(f"  peft={peft.__version__}")

        model_dir_str = get_content_dir_from_hub_cache_dir(self.model_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir_str)
        self.tokenizer = self.processor.tokenizer
        self.model = GraniteSpeechForConditionalGeneration.from_pretrained(
            model_dir_str,
            torch_dtype=torch.bfloat16,
            **({"attn_implementation": attn_implementation} if attn_implementation else {}),
        )
        # the LoRA adapter must be loaded and active: audio turns are trained with it
        assert self.model._hf_peft_config_loaded, "expected the granite-speech LoRA adapter to be loaded"
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"  ({time.time() - start_time:.1f}s)")

        self.audio_token = self.processor.audio_token
        self.audio_token_id = int(self.model.config.audio_token_id)
        self.vocab_size = int(self.model.get_output_embeddings().out_features)
        # chunk-exit log-prob lookup slot = the assistant end-of-turn token (set in forward)
        self.assistant_end_token_id: Optional[int] = None
        print(f"  audio_token={self.audio_token!r} (id={self.audio_token_id}) vocab={self.vocab_size}")

    # ---- Helpers --------------------------------------------------------

    def _build_texts(self, transcription: str) -> tuple:
        """Chat prefix (up to and incl. the assistant tag) and the full teacher-forced text."""
        chat = [{"role": "user", "content": f"{self.audio_token}{self.speech_prompt}"}]
        prefix = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        full = self.tokenizer.apply_chat_template(
            chat + [{"role": "assistant", "content": transcription}], tokenize=False
        )
        assert full.startswith(prefix), f"chat template mismatch: {prefix=!r} {full[: len(prefix) + 20]=!r}"
        return prefix, full

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
        assert raw_inputs_sample_rate == 16000, "GraniteSpeech expects 16 kHz"
        assert len(raw_inputs) == 1, "GraniteSpeech wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        # Omitted prev context: "... " prefix as an unscored context marker (Phi4MM pattern).
        added_prefix = omitted_prev_context is not None and int(omitted_prev_context[0]) > 0

        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])
        transcription = " ".join(words)
        if added_prefix:
            transcription = "... " + transcription
        print(f"[fwd] start; words={len(words)} transcription={transcription!r}", flush=True)

        # Audio features + the number of audio tokens the processor would splice in.
        wav = raw_inputs[:, :orig_n_samples].float()
        audio_inputs = self.processor.audio_processor(wav)
        input_features = audio_inputs["input_features"].to(dev)
        input_features_mask = audio_inputs["input_features_mask"].to(dev)
        n_audio = int(audio_inputs["audio_embed_sizes"][0])

        # Expand the single audio token like the processor does,
        # for both the prefix and the full text,
        # so dst_text_start is exact (no token-boundary guesswork at the seam).
        prefix_text, full_text = self._build_texts(transcription)
        assert prefix_text.count(self.audio_token) == 1
        prefix_text = prefix_text.replace(self.audio_token, self.audio_token * n_audio, 1)
        full_text = full_text.replace(self.audio_token, self.audio_token * n_audio, 1)
        prefix_ids = self.tokenizer(prefix_text, return_tensors="pt")["input_ids"]
        input_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"].to(dev)
        dst_text_start = int(prefix_ids.shape[1])
        assert torch.equal(input_ids[:, :dst_text_start].cpu(), prefix_ids), "prefix ids not a prefix of full ids"
        print(
            f"[fwd] chat ok; n_audio={n_audio} dst_text_start={dst_text_start} "
            f"input_ids.shape={tuple(input_ids.shape)}",
            flush=True,
        )

        _cap = None
        if collect_attentions is not None and self.collect_attn_heads is not None:
            from .base import CaptureSelectedAttn, find_hf_decoder_layers

            # heads capture = attention extraction only, no grads:
            # all layers' retained weights + activations OOM on long-form input
            _cap = CaptureSelectedAttn(
                [_l.self_attn for _l in find_hf_decoder_layers(self.model)],
                [(int(_li), int(_hi)) for _li, _hi in self.collect_attn_heads],
            )
            with _cap, torch.no_grad():
                res = self.model(
                    input_ids=input_ids,
                    input_features=input_features,
                    input_features_mask=input_features_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                )
        else:
            with torch.no_grad():
                res = self.model(
                    input_ids=input_ids,
                    input_features=input_features,
                    input_features_mask=input_features_mask,
                    output_hidden_states=True,
                    output_attentions=collect_attentions is not None,
                    return_dict=True,
                )
        last_out = res.hidden_states[-1]  # [1, L, H]
        assert last_out.shape[:2] == input_ids.shape

        if collect_attentions is not None:
            # Rows = the query positions that predict each transcript token
            # (dst_text_start + u - 1), cols = the contiguous audio token block.
            audio_pos = (input_ids[0] == self.audio_token_id).nonzero(as_tuple=True)[0]
            a0, a1 = int(audio_pos[0]), int(audio_pos[-1]) + 1
            assert a1 - a0 == int(audio_pos.numel()) == n_audio, "audio token block not contiguous"
            n_tgt = int(input_ids.shape[1] - 1 - dst_text_start)
            if _cap is not None:
                # nested dict {layer: {head: [S, n_audio]}}; the extract job indexes attns[li][hi]
                rows_c = torch.arange(dst_text_start - 1, dst_text_start - 1 + n_tgt)
                attns = {
                    _li: {_hi: w[rows_c][:, a0:a1] for _hi, w in _hs.items()} for _li, _hs in _cap.captured.items()
                }
            else:
                assert res.attentions is not None and res.attentions[0] is not None, (
                    "no attention weights returned -- construct with attn_implementation='eager'"
                )
                rows = torch.arange(dst_text_start - 1, dst_text_start - 1 + n_tgt, device=dev)
                attns = [a[0][:, rows][:, :, a0:a1].float().cpu() for a in res.attentions]
            collect_attentions.append(dict(attns=attns, n_audio=n_audio, n_audio_real=n_audio))
        del res

        # Targets: assistant text tokens;
        # strip all trailing special tokens (end-of-turn marker, possibly more),
        # then re-append the first stripped one as the chunk-exit slot.
        # the template tail is <|end_of_text|> plus a plain "\n" token,
        # so strip trailing specials and whitespace-only tokens
        special_ids = set(self.tokenizer.all_special_ids) | set(self.tokenizer.get_added_vocab().values())

        def _is_tail_marker(tid: int) -> bool:
            return tid in special_ids or self.tokenizer.decode([tid]).strip() == ""

        dst_text_end = int(input_ids.shape[1])
        while dst_text_end > dst_text_start and _is_tail_marker(int(input_ids[0, dst_text_end - 1])):
            dst_text_end -= 1
        assert dst_text_start < dst_text_end < int(input_ids.shape[1]), (
            f"unexpected assistant-turn token structure ({dst_text_start=}, {dst_text_end=}, {input_ids.shape=})"
        )
        self.assistant_end_token_id = int(input_ids[0, dst_text_end])
        targets = input_ids[:, dst_text_start:dst_text_end]
        n_targets = int(targets.shape[1])
        targets = torch.cat(
            [targets, torch.tensor([[self.assistant_end_token_id]], device=targets.device, dtype=targets.dtype)],
            dim=1,
        )

        # Per-word grouping via the leading-space word-start marker,
        # decoded group-wise so byte-fallback tokens reconstruct (cf. the Canary-Qwen adapter).
        words_start_end: List[List[int]] = []
        words_token_ids: List[List[int]] = []
        for t in range(n_targets):
            tid = int(targets[0, t].item())
            s = self.tokenizer.decode([tid])
            if t == 0 or s.startswith(" "):
                words_start_end.append([t, t + 1])
                words_token_ids.append([tid])
            else:
                words_token_ids[-1].append(tid)
                words_start_end[-1][1] = t + 1
        words_ = [self.tokenizer.decode(ids).strip() for ids in words_token_ids]
        if added_prefix:
            # drop the "..." context marker (never scored; spans keep their token indices)
            assert words_ and words_[0] == "...", f"expected '...' prefix, got {words_[:1]!r}"
            words_ = words_[1:]
            words_start_end = words_start_end[1:]
        assert words_ == list(words), f"target_decoded={words_!r} ref_words={words!r}"
        words_start_end = words_start_end + [[n_targets, n_targets + 1]]  # exit slot

        # Audio token -> raw sample mapping (10 Hz acoustic embedding grid).
        input_slice = (torch.tensor([0], dtype=torch.int64), torch.tensor([n_audio], dtype=torch.int64))
        edges = torch.arange(n_audio + 1, dtype=torch.float64) * (orig_n_samples / max(n_audio, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(0)

        print(f"[fwd] returning ForwardOutput (n_audio={n_audio}, n_targets={n_targets})", flush=True)
        return ForwardOutput(
            inputs=input_features,
            input_seq_lens=torch.tensor([n_audio]),
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
        # Position dst_text_start + u - 1 predicts target token u.
        last_out = batch_slice(last_out, (dst_text_start + start - 1, dst_text_start + end - 1))
        logits = self.model.get_output_embeddings()(last_out).float()
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
        max_new_tokens: int = 512,
    ) -> List[List[str]]:
        assert len(raw_inputs) == 1 and raw_inputs_sample_rate == 16000
        chat = [{"role": "user", "content": f"{self.audio_token}{self.speech_prompt}"}]
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        wav = raw_inputs[:, : int(raw_input_seq_lens[0])].float()
        inputs = self.processor(text, wav, device=str(self.device)).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        hyp = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return [hyp.strip().split()]
