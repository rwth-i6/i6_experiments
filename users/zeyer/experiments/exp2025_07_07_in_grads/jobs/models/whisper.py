"""Whisper AED model adapter for the grad-based forced-alignment pipeline.
Adds a classic encoder-decoder AED (distinct from the LLM-decoder speech models).

Model: HF ``openai/whisper-base`` (``WhisperForConditionalGeneration``).
Encoder turns log-mel (100 Hz, 80 mel, 30 s padded) into ~50 Hz audio states;
the autoregressive decoder cross-attends them.
Per-token score is the *direct* autoregressive ``log p(y_i | y_<i, audio)`` (teacher-forced)
-- same as the other AED/LLM adapters.
Grad target = the log-mel input (100 Hz). Batch 1 only.
"""

from __future__ import annotations

from typing import Optional, Union, List
import time

import numpy as np
import torch

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from .base import BaseModelInterface, ForwardOutput


class Whisper(BaseModelInterface):
    """HF Whisper AED. See module docstring."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        language: str = "en",
        char_level: bool = False,
        char_level_sep: Optional[str] = None,
        char_level_case: Optional[str] = None,
        version: int = 1,
    ):
        super().__init__()
        assert version >= 1
        self.device = device
        self.model_dir = model_dir
        self.language = language
        self._char_level = char_level
        self._char_level_sep = char_level_sep
        assert char_level_case in (None, "lower", "upper", "title"), char_level_case
        self._char_level_case = char_level_case
        self.version = version

        print("Import / load Whisper...")
        start_time = time.time()
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        d = get_content_dir_from_hub_cache_dir(model_dir)
        self.processor = WhisperProcessor.from_pretrained(d)
        self.model = WhisperForConditionalGeneration.from_pretrained(d).to(device).eval()
        tok = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        self.prefix_ids = tok.convert_tokens_to_ids(
            ["<|startoftranscript|>", f"<|{language}|>", "<|transcribe|>", "<|notimestamps|>"]
        )
        self.eos_id = int(tok.eos_token_id)
        # log-mel hop = 160 samples (100 Hz at 16 kHz)
        self.hop = 160
        print(f"  ({time.time() - start_time:.1f}s) prefix={self.prefix_ids} eos={self.eos_id}")

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
        assert len(raw_inputs) == 1, "Whisper wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("Whisper chunked context not implemented yet")

        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])
        wav = raw_inputs[0].detach().cpu().numpy().astype(np.float32)

        # Log-mel features [1, 80, 3000] (30 s padded).
        # Grad leaf is the transposed [1, 3000, 80] (time x mel) so the extract reduces over mel.
        feats = self.feature_extractor(
            wav, sampling_rate=raw_inputs_sample_rate, return_tensors="pt"
        ).input_features.to(dev)  # [1, 80, 3000]
        leaf = feats.transpose(1, 2).contiguous().detach().requires_grad_(True)  # [1, 3000, 80]
        leaf.retain_grad()
        feats_for_model = leaf.transpose(1, 2)  # [1, 80, 3000]

        # Teacher-forced decoder input: prefix + transcription tokens.
        tok = self.processor.tokenizer
        if self._char_level:
            # Explode words into a flat char list
            # and look up each char's single token id directly
            # (do NOT tokenize the concatenated string:
            # Whisper BPE would merge "t h e" back into "the").
            # A separator token (e.g. space, id 220) precedes every word as autoregressive context;
            # it is never part of a word range, hence never scored -- only the per-char word tokens are.
            # Mirrors the Canary/Phi4 char-level adapters.
            chars: List[str] = []
            word_char_ranges: List[List[int]] = []
            for word in words:
                if self._char_level_case == "upper":
                    word = word.upper()
                elif self._char_level_case == "lower":
                    word = word.lower()
                elif self._char_level_case == "title":
                    word = word.title()
                if self._char_level_sep:
                    chars.append(self._char_level_sep)
                cstart = len(chars)
                for ch in word:
                    chars.append(ch)
                word_char_ranges.append([cstart, len(chars)])
            transc_ids = []
            for ch in chars:
                ids = tok.encode(ch, add_special_tokens=False)
                assert len(ids) == 1, (
                    f"char {ch!r} tokenizes to {len(ids)} tokens {ids} -- char-level needs single-token chars"
                )
                transc_ids.append(ids[0])
            n_targets = len(transc_ids)
            assert n_targets > 0, f"empty target for words={words!r}"
            words_start_end: List[List[int]] = [list(r) for r in word_char_ranges]
            assert len(words_start_end) == len(words), (
                f"char word-grouping mismatch: {len(words_start_end)} vs {len(words)} ({words!r})"
            )
        else:
            # Word-level: leading space so the first word gets its word-start BPE marker.
            transcription = " " + " ".join(words)
            transc_ids = tok(transcription, add_special_tokens=False).input_ids
            n_targets = len(transc_ids)
            assert n_targets > 0, f"empty target for words={words!r}"
        dec_in = torch.tensor([self.prefix_ids + transc_ids], dtype=torch.long, device=dev)
        dst_text_start = len(self.prefix_ids)

        with torch.enable_grad():
            out = self.model(input_features=feats_for_model, decoder_input_ids=dec_in, output_hidden_states=True)
            dec_hidden = out.decoder_hidden_states[-1]  # [1, P+U, H]
        del out

        if not self._char_level:
            # Per-word grouping: a token starts a word if its decoded text begins with a space
            # (Whisper BPE marks word starts with a leading space).
            words_start_end: List[List[int]] = []
            words_: List[str] = []
            for j in range(n_targets):
                s = tok.decode([transc_ids[j]])
                if j == 0 or s.startswith(" "):
                    words_start_end.append([j, j + 1])
                    words_.append(s.strip())
                else:
                    words_[-1] += s
                    words_start_end[-1][1] = j + 1
            assert len(words_start_end) == len(words), (
                f"word-grouping mismatch: {len(words_start_end)} groups ({words_!r}) vs {len(words)} words ({words!r})"
            )

        targets = torch.tensor(
            [transc_ids + [self.eos_id]], dtype=torch.long, device=dev
        )  # [1, U+1], EOS appended for chunk-exit lookups
        words_start_end = words_start_end + [[n_targets, n_targets + 1]]  # exit slot

        # Slice the log-mel grad leaf to the real audio span (drop 30 s padding).
        n_real = min(int(leaf.shape[1]), orig_n_samples // self.hop + 1)
        input_slice = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([n_real], dtype=torch.int64),
        )
        edges = torch.arange(n_real + 1, dtype=torch.float64) * (orig_n_samples / max(n_real, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(
            0
        )  # [1, n_real, 2]

        print(
            f"[fwd] words={len(words)} subwords={n_targets} n_real={n_real} text={' '.join(words)!r}",
            flush=True,
        )
        return ForwardOutput(
            inputs=leaf,
            input_seq_lens=torch.tensor([int(leaf.shape[1])]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(words_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(dec_hidden=dec_hidden, dst_text_start=dst_text_start),
        )

    def log_probs(
        self,
        *,
        forward_output: ForwardOutput,
        start: Union[int, torch.Tensor],
        end: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        from i6_experiments.users.zeyer.torch.batch_slice import batch_slice

        dec_hidden = forward_output.outputs["dec_hidden"]
        dst_text_start = forward_output.outputs["dst_text_start"]
        # Decoder position P+i-1 predicts target token i; slice [start-1, end-1].
        sl = batch_slice(dec_hidden, (dst_text_start + start - 1, dst_text_start + end - 1))
        logits = self.model.proj_out(sl).float()
        return logits.log_softmax(-1)
