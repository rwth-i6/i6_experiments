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
        param_noise_std: float = 0.0,
        param_noise_seed: int = 0,
        input_noise_std: float = 0.0,
        act_noise_std: float = 0.0,
        act_dropout: float = 0.0,
        perturb_seed: int = 0,
        attn_implementation: Optional[str] = None,
        grad_wrt: str = "log_mel",
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
        self.grad_wrt = grad_wrt
        self.version = version

        print("Import / load Whisper...")
        start_time = time.time()
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        d = get_content_dir_from_hub_cache_dir(model_dir)
        self.processor = WhisperProcessor.from_pretrained(d)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(
                d,
                # eager attention is required for output_attentions (cross-attn alignment jobs)
                **({"attn_implementation": attn_implementation} if attn_implementation else {}),
            )
            .to(device)
            .eval()
        )
        from ..param_noise import apply_param_noise

        apply_param_noise(self.model, param_noise_std, param_noise_seed)
        self._input_noise_std = input_noise_std
        self._perturb_seed = perturb_seed
        if act_noise_std or act_dropout:
            from ..perturb import install_activation_perturbation

            kind = "act_noise" if act_noise_std else "act_dropout"
            install_activation_perturbation(self.model, kind, act_noise_std or act_dropout, perturb_seed)
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
        collect_attentions: Optional[list] = None,
    ) -> ForwardOutput:
        """See :class:`BaseModelInterface`.

        :param collect_attentions: if given, the decoder runs with ``output_attentions=True``
            and a dict is appended:
            ``attns`` = per decoder layer CROSS-attention ``[H, n_targets, enc_frames]``
            (query = the position PREDICTING each transcript token),
            ``n_audio`` = padded encoder frames (1500),
            ``n_audio_real`` = encoder frames covering real audio (20 ms grid).
            Used by the attention alignment jobs (auto head selection).
        """
        assert raw_inputs_sample_rate is not None
        assert len(raw_inputs) == 1, "Whisper wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("Whisper chunked context not implemented yet")

        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])
        wav = raw_inputs[0].detach().cpu().numpy().astype(np.float32)
        if self._input_noise_std:
            from ..perturb import apply_input_noise

            wav = apply_input_noise(wav, self._input_noise_std, self._perturb_seed)

        # Log-mel features [1, 80, 3000] (30 s padded).
        feats = self.feature_extractor(
            wav, sampling_rate=raw_inputs_sample_rate, return_tensors="pt"
        ).input_features.to(dev)  # [1, 80, 3000]
        _enc_captured: List[torch.Tensor] = []
        _enc_hook = None
        if self.grad_wrt == "log_mel":
            # Grad leaf is the log-mel, transposed [1, 3000, 80] so the extract reduces over mel (10 ms grid).
            leaf = feats.transpose(1, 2).contiguous().detach().requires_grad_(True)  # [1, 3000, 80]
            leaf.retain_grad()
            feats_for_model = leaf.transpose(1, 2)  # [1, 80, 3000]
        else:
            # Grad leaf is an ENCODER-depth activation (20 ms grid): a forward hook leafifies it during the
            # model call (see _register_enc_grad_hook), so the per-token grad measures saliency at that
            # depth. The log-mel passes through ungradded.
            feats_for_model = feats
            _enc_hook = self._register_enc_grad_hook(_enc_captured)

        # Teacher-forced decoder input: prefix + transcription tokens.
        tok = self.processor.tokenizer
        if self._char_level:
            # Explode words into a flat char list and look up each char's token id(s) directly
            # (do NOT tokenize the concatenated string:
            # Whisper BPE would merge "t h e" back into "the").
            # A separator token (e.g. space, id 220) precedes every word as autoregressive context;
            # it is never part of a word range, hence never scored -- only the per-char word tokens are.
            # A char may map to >1 BPE token (e.g. the pound sign -> 2 byte-tokens, no merge in the
            # vocab); we keep all of them and track the word ranges in TOKEN units, so the multi-token
            # char is simply interior to its word and the per-token downstream stays correct.
            # Mirrors the Canary/Phi4 char-level adapters.
            chars: List[str] = []
            transc_ids: List[int] = []
            word_char_ranges: List[List[int]] = []  # per word, [token_start, token_end)
            for word in words:
                if self._char_level_case == "upper":
                    word = word.upper()
                elif self._char_level_case == "lower":
                    word = word.lower()
                elif self._char_level_case == "title":
                    word = word.title()
                if self._char_level_sep:
                    chars.append(self._char_level_sep)
                    transc_ids.extend(tok.encode(self._char_level_sep, add_special_tokens=False))
                tstart = len(transc_ids)
                for ch in word:
                    ids = tok.encode(ch, add_special_tokens=False)
                    assert len(ids) >= 1, f"char {ch!r} tokenizes to 0 tokens"
                    chars.append(ch)
                    transc_ids.extend(ids)
                word_char_ranges.append([tstart, len(transc_ids)])
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

        try:
            with torch.enable_grad():
                out = self.model(
                    input_features=feats_for_model,
                    decoder_input_ids=dec_in,
                    output_hidden_states=True,
                    output_attentions=collect_attentions is not None,
                )
                dec_hidden = out.decoder_hidden_states[-1]  # [1, P+U, H]
        finally:
            if _enc_hook is not None:
                _enc_hook.remove()
        if self.grad_wrt != "log_mel":
            assert len(_enc_captured) == 1, f"expected 1 encoder hook call, got {len(_enc_captured)}"
            leaf = _enc_captured[0]  # [1, T_enc, D] (20 ms grid)
        if collect_attentions is not None:
            # Rows = the query positions that PREDICT each transcript token; cols = encoder frames.
            n_enc = int(out.cross_attentions[0].shape[-1])
            n_enc_real = min(n_enc, orig_n_samples // (2 * self.hop) + 1)  # 20 ms grid
            rows_t = torch.arange(dst_text_start - 1, dst_text_start - 1 + n_targets, device=dev)
            collect_attentions.append(
                dict(
                    attns=[a[0][:, rows_t].float().cpu() for a in out.cross_attentions],
                    n_audio=n_enc,
                    n_audio_real=int(n_enc_real),
                )
            )
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

        # Slice the grad leaf to the real audio span (drop 30 s padding). The log-mel grid is 10 ms
        # (hop 160); the encoder grid is 20 ms (the two conv layers downsample by 2).
        _frame_hop = self.hop if self.grad_wrt == "log_mel" else 2 * self.hop
        n_real = min(int(leaf.shape[1]), orig_n_samples // _frame_hop + 1)
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

    def _register_enc_grad_hook(self, captured: List[torch.Tensor]):
        """Register a forward hook that leafifies the chosen encoder-depth activation (``self.grad_wrt``),
        so the per-token gradient is taken w.r.t. that depth instead of the log-mel input.
        ``enc_in`` = input to encoder layer 0 (after the conv subsampling, 20 ms grid);
        ``enc_L<N>`` = output of encoder layer N (1-indexed);
        ``enc_out`` = the final encoder output (after the encoder output layer-norm, what the decoder
        cross-attends). The leaf is appended to ``captured``."""
        enc = self.model.model.encoder
        n_layers = len(enc.layers)

        def _leafify(x: torch.Tensor) -> torch.Tensor:
            leaf = x.detach().requires_grad_(True)
            leaf.retain_grad()
            captured.append(leaf)
            return leaf

        gw = self.grad_wrt
        if gw == "enc_in":

            def _pre(_m, args, kwargs):
                return (_leafify(args[0]), *args[1:]), kwargs

            return enc.layers[0].register_forward_pre_hook(_pre, with_kwargs=True)
        if gw == "enc_out":

            def _ln(_m, _inp, out):
                return _leafify(out)

            return enc.layer_norm.register_forward_hook(_ln)
        assert gw.startswith("enc_L"), f"unknown grad_wrt {gw!r}"
        idx = int(gw[len("enc_L") :])
        assert 1 <= idx <= n_layers, f"{gw}: encoder layer {idx} out of range 1..{n_layers}"

        def _layer_out(_m, _inp, out):
            return (_leafify(out[0]), *out[1:])

        return enc.layers[idx - 1].register_forward_hook(_layer_out)

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

    def recog(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        raw_input_seq_lens: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> List[List[str]]:
        """Greedy transcription via HF generate (forced language/task prefix, no timestamps).
        Returns the decoded hyp text whitespace-split into words;
        normalization for WER/matching is left to the caller."""
        assert len(raw_inputs) == 1
        wav = raw_inputs[0].detach().cpu().numpy().astype(np.float32)
        feats = self.feature_extractor(
            wav, sampling_rate=raw_inputs_sample_rate, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                feats,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                language=self.language,
                task="transcribe",
            )
        text = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return [text.split()]
