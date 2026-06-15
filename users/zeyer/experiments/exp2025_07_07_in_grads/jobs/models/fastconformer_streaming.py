"""NeMo cache-aware streaming FastConformer (hybrid CTC + RNN-T) adapter for grad-based alignment.

Model: ``nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`` -- a STREAMING FastConformer
encoder (limited-context attention + causal convs, cache-aware; multi-lookahead) feeding BOTH a CTC
head and an RNN-T head. The streaming representative: one checkpoint gives a streaming CTC and a
streaming transducer, same FastConformer family as the offline Parakeet -> controlled
offline-vs-streaming comparison. Ref: Noroozi et al., arXiv:2312.17279 (cache-aware streaming
Conformer); Rekesh et al., ASRU 2023 (FastConformer).

``att_context_size = [left, right]`` (encoder look-ahead in 80 ms frames) picks the streaming latency
from the multi-lookahead set (e.g. ``[70, 13]``=1040 ms, ``[70, 6]``=480 ms, ``[70, 1]``=80 ms,
``[70, 0]``=0 ms). A single masked forward at the chosen size reproduces the streaming receptive
field. ``head`` selects the CTC or RNN-T decoder; scoring mirrors ParakeetCtc (``ctc_partial``) /
ParakeetRnnt (RNN-T prefix). Log-mel grad leaf (~100 Hz). Batch 1.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Optional, Union, List

import numpy as np
import torch

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from .base import BaseModelInterface, ForwardOutput


class FastConformerStreaming(BaseModelInterface):
    """NeMo hybrid RNN-T/CTC cache-aware streaming FastConformer. See module docstring."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        overlay_path: str,
        head: str = "rnnt",
        att_context_size: Optional[List[int]] = None,
        per_token_score: Optional[str] = None,
        version: int = 1,
    ):
        """:param head: ``"rnnt"`` (transducer prefix score) or ``"ctc"`` (CTC partial score).
        :param att_context_size: ``[left, right]`` encoder context in 80 ms frames; None keeps the
            checkpoint default (offline-ish). Smaller right = lower latency = more streaming delay.
        :param per_token_score: defaults to ``"prefix"`` (rnnt) / ``"prefix_fwd"`` (ctc).
        """
        super().__init__()
        assert version >= 1
        assert head in ("ctc", "rnnt"), head
        self.device = device
        self.model_dir = model_dir
        self.overlay_path = overlay_path
        self.head = head
        self.att_context_size = [int(x) for x in att_context_size] if att_context_size is not None else None
        if per_token_score is None:
            per_token_score = "prefix" if head == "rnnt" else "prefix_fwd"
        self.per_token_score = per_token_score
        self.version = version

        if overlay_path not in sys.path:
            sys.path.insert(0, overlay_path)
        print("Import NeMo / EncDecHybridRNNTCTCBPEModel (from overlay)...")
        start_time = time.time()
        import nemo
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

        print(f"  nemo={nemo.__version__} from {nemo.__file__}")
        content = get_content_dir_from_hub_cache_dir(model_dir)
        nemo_files = glob.glob(os.path.join(content, "**", "*.nemo"), recursive=True)
        assert len(nemo_files) == 1, f"expected exactly one .nemo under {content}, got {nemo_files}"
        print(f"Restoring hybrid streaming model from {nemo_files[0]}...")
        self.model = EncDecHybridRNNTCTCBPEModel.restore_from(nemo_files[0], map_location=device)
        self.model.to(device).eval()
        self.tokenizer = self.model.tokenizer

        # Pick the streaming look-ahead from the multi-lookahead set (cache-aware encoder).
        if self.att_context_size is not None:
            self.model.encoder.set_default_att_context_size(self.att_context_size)
        self.target_sr = int(self.model.cfg.sample_rate)

        if head == "rnnt":
            self.blank_idx = int(self.model.decoder.blank_idx)
            self.vocab_size = self.blank_idx + 1
        else:
            self.vocab_size = int(self.model.ctc_decoder.num_classes_with_blank)
            self.blank_idx = self.vocab_size - 1
        print(
            f"  ({time.time() - start_time:.1f}s) head={head} att_ctx={self.att_context_size} "
            f"vocab={self.vocab_size} blank_idx={self.blank_idx} sr={self.target_sr} score={self.per_token_score}"
        )

    def _tokenize_words(self, words: List[str]):
        """Encode to subwords and group into words via the '▁' (word-start) marker."""
        text = " ".join(w.lower() for w in words)
        ids = list(self.tokenizer.text_to_ids(text))
        pieces = self.tokenizer.ids_to_tokens(ids)
        word_ranges = []
        cur_start = 0
        for j, p in enumerate(pieces):
            if j > 0 and p.startswith("▁"):
                word_ranges.append((cur_start, j))
                cur_start = j
        word_ranges.append((cur_start, len(pieces)))
        if len(word_ranges) == len(words):
            return ids, word_ranges
        ids, word_ranges = [], []
        for w in words:
            wp = list(self.tokenizer.text_to_ids(w.lower()))
            assert wp, f"word {w!r} encoded to empty"
            word_ranges.append((len(ids), len(ids) + len(wp)))
            ids.extend(wp)
        return ids, word_ranges

    def forward(
        self,
        *,
        raw_inputs: Union[np.ndarray, torch.Tensor, List[List[str]]],
        raw_inputs_sample_rate: Optional[int] = None,
        raw_input_seq_lens: torch.Tensor,
        raw_targets: List[List[str]],
        raw_target_seq_lens: torch.Tensor,
        omitted_prev_context: Optional[torch.Tensor] = None,
        collect_lattice: Optional[list] = None,
    ) -> ForwardOutput:
        assert raw_inputs_sample_rate is not None
        assert len(raw_inputs) == 1 and isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("chunked context not implemented")
        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])
        wav = raw_inputs[0].to(dev).float()
        if raw_inputs_sample_rate != self.target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav[None], raw_inputs_sample_rate, self.target_sr)[0]
        sub_ids, word_ranges = self._tokenize_words(words)
        assert len(word_ranges) == len(words)
        s_len = len(sub_ids)
        assert s_len > 0, f"empty target for {words!r}"

        with torch.enable_grad():
            wav_len = torch.tensor([wav.shape[0]], device=dev, dtype=torch.long)
            proc, proc_len = self.model.preprocessor(input_signal=wav[None], length=wav_len)  # [1, F, T_mel]
            leaf = proc.detach().transpose(1, 2).contiguous().requires_grad_(True)  # [1, T_mel, F]
            leaf.retain_grad()
            t_feat = int(leaf.shape[1])
            enc, enc_len = self.model.encoder(audio_signal=leaf.transpose(1, 2), length=proc_len)  # [1, H, T_enc]
            t_enc = int(enc.shape[2])

            if self.head == "ctc":
                log_probs = self.model.ctc_decoder(encoder_output=enc)[0].float()  # [T_enc, V] (log_softmax)
                from .ctc_partial import ctc_partial_scores

                partial = ctc_partial_scores(log_probs, sub_ids, self.blank_idx, mode=self.per_token_score)  # [S]
            else:
                tgt = torch.tensor([sub_ids], dtype=torch.long, device=dev)
                tgt_len = torch.tensor([s_len], dtype=torch.long, device=dev)
                dec_out, _, _ = self.model.decoder(targets=tgt, target_length=tgt_len)  # [1, H, U+1]
                logits = self.model.joint.joint(enc.transpose(1, 2), dec_out.transpose(1, 2))  # [1, T, U+1, V]
                log_probs = torch.log_softmax(logits[0, ..., : self.vocab_size].float(), dim=-1)  # [T, U+1, V]
                if collect_lattice is not None:
                    collect_lattice.append(
                        dict(log_probs=log_probs.detach().cpu().numpy(), log_dur=None, durations=None)
                    )
                if self.per_token_score == "prefix":
                    from .parakeet_rnnt import ParakeetRnnt

                    partial = ParakeetRnnt._rnnt_prefix_scores(self, log_probs, sub_ids)  # [S]
                else:
                    ys = torch.tensor(sub_ids, dtype=torch.long, device=dev)
                    cols = log_probs[:, torch.arange(s_len, device=dev), ys]
                    partial = torch.logsumexp(cols, dim=0)  # [S]

        partial_padded = torch.cat([partial, partial.new_zeros(1)])  # [S+1], exit slot
        targets = torch.tensor([sub_ids + [self.blank_idx]], dtype=torch.long, device=dev)
        word_start_end = [[a, b] for (a, b) in word_ranges] + [[s_len, s_len + 1]]
        input_slice = (torch.tensor([0], dtype=torch.int64), torch.tensor([t_feat], dtype=torch.int64))
        edges = torch.arange(t_feat + 1, dtype=torch.float64) * (orig_n_samples / max(t_feat, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(0)
        print(
            f"[fwd] head={self.head} words={len(words)} subwords={s_len} T_feat={t_feat} T_enc={t_enc} "
            f"text={' '.join(w.lower() for w in words)!r}",
            flush=True,
        )
        return ForwardOutput(
            inputs=leaf,
            input_seq_lens=torch.tensor([t_feat]),
            input_slice_start_end=input_slice,
            input_raw_start_end=input_raw_start_end,
            targets=targets,
            target_seq_lens=torch.tensor([targets.shape[1]]),
            target_start_end=torch.tensor(word_start_end, dtype=torch.int64, device=dev).unsqueeze(0),
            outputs=dict(partial_padded=partial_padded, vocab_size=self.vocab_size),
        )

    def log_probs(
        self, *, forward_output: ForwardOutput, start: Union[int, torch.Tensor], end: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """Scatter the precomputed per-token scores into ``[1, n, V]`` at the target indices."""
        partial_padded = forward_output.outputs["partial_padded"]  # [S+1], grad-attached
        v = forward_output.outputs["vocab_size"]
        start_i = int(start[0]) if isinstance(start, torch.Tensor) else int(start)
        end_i = int(end[0]) if isinstance(end, torch.Tensor) else int(end)
        n = end_i - start_i
        device = partial_padded.device
        vals = partial_padded[start_i:end_i]
        targets_slice = forward_output.targets[0, start_i:end_i].to(device)
        out = partial_padded.new_zeros((1, n, v))
        out[0, torch.arange(n, device=device), targets_slice] = vals
        return out

    def recog(
        self,
        *,
        raw_inputs: torch.Tensor,
        raw_inputs_sample_rate: int,
        raw_input_seq_lens: torch.Tensor,
        max_new_tokens: int = 100,
    ) -> List[List[str]]:
        """Greedy decoding via NeMo ``transcribe`` using the selected head."""
        assert len(raw_inputs) == 1 and raw_inputs_sample_rate == 16000
        if self.head == "ctc":
            self.model.change_decoding_strategy(decoder_type="ctc")
        wav = raw_inputs[0].detach().cpu().numpy().astype("float32")
        with torch.no_grad():
            out = self.model.transcribe([wav], batch_size=1, verbose=False)
        h = out[0]
        text = h.text if hasattr(h, "text") else (h[0] if isinstance(h, (list, tuple)) else h)
        return [str(text).split()]

    def forced_align_words(self, *, audio: torch.Tensor, sample_rate: int, words: List[str]):
        """torchaudio CTC forced-alignment on the streaming CTC head's own emission (head='ctc' only).
        The 'posteriors' baseline for the streaming CTC, analogous to ParakeetCtc.forced_align_words."""
        assert self.head == "ctc", "forced_align_words is the CTC-head baseline"
        import torchaudio

        dev = self.device
        wav = audio.to(dev).float()
        dur = int(wav.shape[0]) / sample_rate
        if sample_rate != self.target_sr:
            wav = torchaudio.functional.resample(wav[None], sample_rate, self.target_sr)[0]
        sub_ids, word_ranges = self._tokenize_words(words)
        assert len(word_ranges) == len(words)
        with torch.no_grad():
            wav_len = torch.tensor([wav.shape[0]], device=dev, dtype=torch.long)
            proc, proc_len = self.model.preprocessor(input_signal=wav[None], length=wav_len)  # [1, F, T_mel]
            enc, enc_len = self.model.encoder(audio_signal=proc, length=proc_len)  # [1, H, T_enc]
            log_probs = self.model.ctc_decoder(encoder_output=enc).float()  # [1, T_enc, V] (log_softmax)
        n_frames = int(log_probs.shape[1])
        spf = dur / max(n_frames, 1)
        targets = torch.tensor([sub_ids], dtype=torch.int32, device=dev)
        aligned, scores = torchaudio.functional.forced_align(log_probs, targets, blank=self.blank_idx)
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=self.blank_idx)
        assert len(spans) == len(sub_ids), f"{len(spans)} vs {len(sub_ids)}"
        pred_sub_se = [(s.start * spf, s.end * spf) for s in spans]
        return [(pred_sub_se[a][0], pred_sub_se[b - 1][1]) for a, b in word_ranges]
