"""Parakeet RNN-T (NeMo FastConformer-Transducer) adapter for the grad-based
forced-alignment pipeline.

Adds a STRONG, leaderboard-grade transducer to the breadth.
Our other transducer (torchaudio Emformer RNN-T) is a small base model;
this is NVIDIA's FastConformer-Transducer (e.g. ``nvidia/parakeet-rnnt-1.1b``),
near the top of the Open ASR Leaderboard among transducers.

Per-token score: same as the Emformer adapter.
Teacher-force the reference subwords, run the prediction net + joint,
and score each token via either
the proper transducer prefix-score (RNN-T forward over the T x (U+1) lattice)
or the crude time-marginal emission ``logsumexp_t log p(y_u | t)``.
Grad target = the log-mel features (~100 Hz, FastConformer preprocessor),
mirroring the Canary log-mel grad surface.

NeMo is loaded from the same overlay Canary uses (no shared-env edits).
Batch size 1 only.
"""

from __future__ import annotations

from typing import Optional, Union, List
import sys
import time
import glob
import os

import numpy as np
import torch

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir
from .base import BaseModelInterface, ForwardOutput


class ParakeetRnnt(BaseModelInterface):
    """NeMo FastConformer RNN-T transducer. See module docstring."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_dir: str,
        overlay_path: str,
        per_token_score: str = "prefix",
        version: int = 1,
        char_level: bool = False,
    ):
        """:param overlay_path: NeMo env overlay to activate on sys.path (passed from the recipe)."""
        super().__init__()
        assert version >= 1
        assert per_token_score in ("prefix", "emission"), per_token_score
        self.device = device
        self.model_dir = model_dir
        self.overlay_path = overlay_path
        self.per_token_score = per_token_score
        self.version = version
        self.char_level = bool(char_level)

        if overlay_path not in sys.path:
            sys.path.insert(0, overlay_path)

        print("Import NeMo / EncDecRNNTBPEModel (from overlay)...")
        start_time = time.time()
        import nemo
        from nemo.collections.asr.models import EncDecRNNTBPEModel

        print(f"  nemo={nemo.__version__} from {nemo.__file__}")

        content = get_content_dir_from_hub_cache_dir(model_dir)
        nemo_files = glob.glob(os.path.join(content, "**", "*.nemo"), recursive=True)
        assert len(nemo_files) == 1, f"expected exactly one .nemo under {content}, got {nemo_files}"
        print(f"Restoring RNN-T from {nemo_files[0]}...")
        self.model = EncDecRNNTBPEModel.restore_from(nemo_files[0], map_location=device)
        self.model.to(device).eval()
        self.tokenizer = self.model.tokenizer

        # NeMo RNN-T blank is the last joint class (num_classes_with_blank - 1).
        self.blank_idx = int(self.model.decoder.blank_idx)
        self.vocab_size = self.blank_idx + 1
        self.target_sr = int(self.model.cfg.sample_rate)
        print(
            f"  ({time.time() - start_time:.1f}s) vocab={self.vocab_size} "
            f"blank_idx={self.blank_idx} sr={self.target_sr} score={self.per_token_score}"
        )

    # ---- Helpers --------------------------------------------------------

    def _rnnt_prefix_scores(self, logp: torch.Tensor, ys: List[int]) -> torch.Tensor:
        """Proper transducer per-token prefix score (analog of the CTC partial score).
        RNN-T forward over the T x (U+1) lattice,
        then the telescoping difference of the time-integrated occupancy.

        Vectorized as a wavefront over anti-diagonals ``d = t + u``:
        each cell is the same two-term ``logaddexp`` as the scalar recursion
        (see :func:`_rnnt_prefix_scores_loop`),
        with missing predecessors entering as ``-1e9``,
        which ``logaddexp`` absorbs exactly in fp.
        ~(T+U) vector graph nodes instead of T*U scalar nodes --
        the difference between ~50 s/seq and a few s/seq on real Buckeye segments.

        :param logp: ``[T, U+1, V]`` log-probs from the joint (teacher-forced).
        :param ys: length-U target subword ids.
        :return: ``[U]`` prefix scores ``Δ_u = accum(u) - accum(u-1)``,
            ``accum(u) = logsumexp_t α(t,u)``. Grad-attached.
        """
        T, U1, _ = logp.shape
        U = len(ys)
        dev = logp.device
        neg = -1.0e9  # finite log-zero (true -inf -> 0*inf=NaN grads)
        ys_t = torch.tensor(ys, device=dev, dtype=torch.long)
        logblank = logp[:, :, self.blank_idx]  # [T, U+1]: stay at u, advance t
        loglabel = logp[torch.arange(T, device=dev)[:, None], torch.arange(U, device=dev)[None, :], ys_t[None, :]]
        # ^ [T, U]: from predictor-position u emit ys[u] -> advance to u+1, stay t

        alpha_mat = logp.new_full((T, U1), neg)  # filled diagonal by diagonal
        prev_t_lo = 0
        prev_vec = None
        for d in range(T + U1 - 1):
            t_lo = max(0, d - (U1 - 1))
            t_hi = min(d, T - 1)
            ts = torch.arange(t_lo, t_hi + 1, device=dev)  # [n] cells (t, d-t)
            us = d - ts
            if d == 0:
                vec = logp.new_zeros(1)  # alpha[0, 0] = 0
            else:
                # blank predecessor (t-1, u): valid iff t > 0.
                bl = prev_vec[(ts - 1 - prev_t_lo).clamp(min=0)] + logblank[(ts - 1).clamp(min=0), us]
                bl = torch.where(ts > 0, bl, torch.full_like(bl, neg))
                # label predecessor (t, u-1): valid iff u > 0.
                la = prev_vec[(ts - prev_t_lo).clamp(max=prev_vec.shape[0] - 1)] + loglabel[ts, (us - 1).clamp(min=0)]
                la = torch.where(us > 0, la, torch.full_like(la, neg))
                vec = torch.logaddexp(bl, la)
            alpha_mat = alpha_mat.index_put((ts, us), vec)
            prev_t_lo, prev_vec = t_lo, vec
        accum = torch.logsumexp(alpha_mat, dim=0)  # [U+1]
        return accum[1:] - accum[:-1]  # [U]

    def _rnnt_prefix_scores_loop(self, logp: torch.Tensor, ys: List[int]) -> torch.Tensor:
        """Scalar reference implementation of :func:`_rnnt_prefix_scores`
        (the original per-cell recursion; kept for equivalence verification)."""
        T, U1, _ = logp.shape
        U = len(ys)
        dev = logp.device
        ys_t = torch.tensor(ys, device=dev, dtype=torch.long)
        logblank = logp[:, :, self.blank_idx]  # [T, U+1]: stay at u, advance t
        loglabel = logp[torch.arange(T, device=dev)[:, None], torch.arange(U, device=dev)[None, :], ys_t[None, :]]

        # alpha[t][u]: log prob of reaching (t,u) having emitted u labels.
        alpha = [[None] * U1 for _ in range(T)]
        alpha[0][0] = logp.new_zeros(())
        for t in range(T):
            for u in range(U1):
                if t == 0 and u == 0:
                    continue
                terms = []
                if t > 0:
                    terms.append(alpha[t - 1][u] + logblank[t - 1, u])  # blank from (t-1,u)
                if u > 0:
                    terms.append(alpha[t][u - 1] + loglabel[t, u - 1])  # emit ys[u-1] from (t,u-1)
                alpha[t][u] = terms[0] if len(terms) == 1 else torch.logaddexp(terms[0], terms[1])
        accum = torch.stack([torch.logsumexp(torch.stack([alpha[t][u] for t in range(T)]), 0) for u in range(U1)])
        return accum[1:] - accum[:-1]  # [U]

    def _tdt_prefix_scores(
        self, logp: torch.Tensor, logdur: torch.Tensor, ys: List[int], durations: List[int]
    ) -> torch.Tensor:
        """Duration-aware TDT prefix score (analog of :func:`_rnnt_prefix_scores`).
        TDT forward over the T x (U+1) lattice where every emission carries a duration
        ``d`` from ``durations``: a token advances ``(t, u) -> (t+d, u+1)``,
        a blank advances ``(t, u) -> (t+d, u)`` with ``d >= 1``
        (blank duration 0 is excluded, matching TDT decoding -- it would not terminate).

        Vectorized as a row loop over ``t``:
        cross-row transitions (``d >= 1``) are vector logaddexp accumulations;
        the in-row token-with-``d=0`` recurrence ``A_u = logaddexp(B_u, A_{u-1} + c_{u-1})``
        is solved exactly via ``logcumsumexp`` (``A = S + logcumsumexp(B - S)``,
        ``S_u = sum_{j<u} c_j``).

        :param logp: ``[T, U+1, V]`` token log-probs from the joint (teacher-forced).
        :param logdur: ``[T, U+1, D]`` duration log-probs from the joint.
        :param ys: length-U target subword ids.
        :param durations: the duration values (e.g. ``[0, 1, 2, 3, 4]``), index-aligned to ``logdur``.
        :return: ``[U]`` prefix scores, grad-attached (as in :func:`_rnnt_prefix_scores`).
        """
        T, U1, _ = logp.shape
        U = len(ys)
        dev = logp.device
        neg = -1.0e9  # finite log-zero (true -inf -> 0*inf=NaN grads)
        ys_t = torch.tensor(ys, device=dev, dtype=torch.long)
        logblank = logp[:, :, self.blank_idx]  # [T, U+1]
        loglabel = logp[torch.arange(T, device=dev)[:, None], torch.arange(U, device=dev)[None, :], ys_t[None, :]]
        # ^ [T, U]: from predictor-position u emit ys[u] -> u+1
        d0_i = durations.index(0) if 0 in durations else None

        alpha_rows: List[torch.Tensor] = []
        for t in range(T):
            base = logp.new_full((U1,), neg)
            if t == 0:
                base = base.index_put((torch.zeros(1, dtype=torch.long, device=dev),), logp.new_zeros(1))
            for d_i, d in enumerate(durations):
                if d < 1 or d > t:
                    continue
                prev = alpha_rows[t - d]  # [U+1]
                base = torch.logaddexp(base, prev + logblank[t - d] + logdur[t - d, :, d_i])  # blank, same u
                shifted = prev[:-1] + loglabel[t - d] + logdur[t - d, :-1, d_i]  # token u-1 -> u
                base = torch.cat([base[:1], torch.logaddexp(base[1:], shifted)])
            if d0_i is not None and U > 0:
                c = loglabel[t] + logdur[t, :-1, d0_i]  # [U] in-row token-with-d=0 weights
                s = torch.cat([logp.new_zeros(1), torch.cumsum(c, 0)])  # [U+1]
                alpha_rows.append(s + torch.logcumsumexp(base - s, 0))
            else:
                alpha_rows.append(base)
        accum = torch.logsumexp(torch.stack(alpha_rows), dim=0)  # [U+1]
        return accum[1:] - accum[:-1]  # [U]

    def _tdt_prefix_scores_loop(
        self, logp: torch.Tensor, logdur: torch.Tensor, ys: List[int], durations: List[int]
    ) -> torch.Tensor:
        """Scalar reference implementation of :func:`_tdt_prefix_scores`
        (per-cell recursion; kept for equivalence verification)."""
        T, U1, _ = logp.shape
        U = len(ys)
        dev = logp.device
        ys_t = torch.tensor(ys, device=dev, dtype=torch.long)
        logblank = logp[:, :, self.blank_idx]
        loglabel = logp[torch.arange(T, device=dev)[:, None], torch.arange(U, device=dev)[None, :], ys_t[None, :]]

        alpha = [[None] * U1 for _ in range(T)]
        for t in range(T):
            for u in range(U1):
                if t == 0 and u == 0:
                    alpha[0][0] = logp.new_zeros(())
                    continue
                terms = []
                for d_i, d in enumerate(durations):
                    if 1 <= d <= t:
                        terms.append(alpha[t - d][u] + logblank[t - d, u] + logdur[t - d, u, d_i])
                        if u > 0:
                            terms.append(alpha[t - d][u - 1] + loglabel[t - d, u - 1] + logdur[t - d, u - 1, d_i])
                    if d == 0 and u > 0:
                        terms.append(alpha[t][u - 1] + loglabel[t, u - 1] + logdur[t, u - 1, d_i])
                a = terms[0]
                for x in terms[1:]:
                    a = torch.logaddexp(a, x)
                alpha[t][u] = a
        accum = torch.stack([torch.logsumexp(torch.stack([alpha[t][u] for t in range(T)]), 0) for u in range(U1)])
        return accum[1:] - accum[:-1]  # [U]

    def _tokenize_words(self, words: List[str]):
        """Encode the transcription to subwords and group them into words
        via the ``▁`` (word-start) marker.

        Returns ``(subword_ids, word_ranges)`` where
        word_ranges[w] = (start, end) subword indices for word w (end exclusive).
        """
        if self.char_level:
            return self._tokenize_words_charlevel(words)
        text = " ".join(w.lower() for w in words)
        ids = list(self.tokenizer.text_to_ids(text))
        pieces = self.tokenizer.ids_to_tokens(ids)
        word_ranges = []
        cur_start = 0
        for j, p in enumerate(pieces):
            if j > 0 and p.startswith("▁"):  # word-start marker
                word_ranges.append((cur_start, j))
                cur_start = j
        word_ranges.append((cur_start, len(pieces)))
        if len(word_ranges) == len(words):
            return ids, word_ranges
        # Fallback: in-context grouping didn't yield one group per word (rare).
        # Encode each word separately so the word count is exact.
        ids, word_ranges = [], []
        for w in words:
            wp = list(self.tokenizer.text_to_ids(w.lower()))
            assert wp, f"word {w!r} encoded to empty"
            word_ranges.append((len(ids), len(ids) + len(wp)))
            ids.extend(wp)
        return ids, word_ranges

    def _tokenize_words_charlevel(self, words: List[str]):
        """Char-level retokenization, IDENTICAL in form to the AED / speech-LLM char-level path:
        each character is its own BARE target token and word boundaries are a separate ``▁``
        (SentencePiece word-boundary) token BETWEEN words, so detokenizing the sequence reconstructs
        the original transcript. The AED path emits bare chars + a standalone separator token
        (e.g. Voxtral: ``s h e <space> d o g``); the SentencePiece equivalent is ``s h e ▁ d o g``
        -- NOT the per-char ``▁s ▁h ▁e`` (which would decode to spurious inter-char spaces and is
        not a valid segmentation). The per-token grad is per character; the separator token belongs
        to no word (as in AED), so ``word_ranges`` cover only each word's characters.
        A rare char with no bare single-char piece falls back to SentencePiece byte encoding
        (still reconstructs).
        Returns ``(char_ids, word_ranges)`` with the same contract as :meth:`_tokenize_words`."""
        sp = getattr(self.tokenizer, "tokenizer", None)  # raw SentencePieceProcessor inside NeMo's tokenizer
        assert sp is not None and hasattr(sp, "piece_to_id"), "expected a SentencePiece-backed NeMo tokenizer"
        unk = sp.unk_id()
        sep_id = sp.piece_to_id("\u2581")  # the standalone word-boundary piece (AED's space-separator analog)
        assert sep_id != unk, "no standalone word-boundary piece in the SentencePiece vocab"
        ids: List[int] = []
        word_ranges = []
        for wi, w in enumerate(words):
            if wi > 0:
                ids.append(sep_id)  # inter-word separator, excluded from any word range (as in AED)
            start = len(ids)
            for ch in w.lower():
                pid = sp.piece_to_id(ch)
                if pid == unk:  # no bare single-char piece -> byte-fallback (still reconstructs)
                    ids.extend(int(i) for i in sp.encode(ch, out_type=int))
                else:
                    ids.append(int(pid))
            assert len(ids) > start, f"word {w!r} produced no char tokens"
            word_ranges.append((start, len(ids)))
        return ids, word_ranges

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
        collect_lattice: Optional[list] = None,
    ) -> ForwardOutput:
        """See :class:`BaseModelInterface`.

        :param collect_lattice: if given, a dict is appended with the teacher-forced
            joint outputs as numpy:
            ``log_probs`` [T, U+1, V] token log-probs,
            ``log_dur`` [T, U+1, D] duration log-probs (TDT; else None),
            ``durations`` (TDT; else None).
            Used by the native-Viterbi alignment baseline job.
        """
        assert raw_inputs_sample_rate is not None
        assert len(raw_inputs) == 1, "ParakeetRnnt wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("ParakeetRnnt chunked context not implemented yet")

        dev = self.device
        words = raw_targets[0]
        orig_n_samples = int(raw_input_seq_lens[0])

        wav = raw_inputs[0].to(dev).float()  # [T_samples]
        if raw_inputs_sample_rate != self.target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav[None], raw_inputs_sample_rate, self.target_sr)[0]

        sub_ids, word_ranges = self._tokenize_words(words)
        assert len(word_ranges) == len(words), (
            f"subword->word grouping mismatch: {len(word_ranges)} groups vs {len(words)} words "
            f"(text={' '.join(words)!r})"
        )
        s_len = len(sub_ids)
        assert s_len > 0, f"empty target for words={words!r}"

        with torch.enable_grad():
            # Preprocessor -> log-mel [1, F, T_mel]. The grad leaf is the
            # transposed [1, T_mel, F] so the extract reduces over the mel dim
            # and keeps the time axis as the ~100 Hz alignment grid.
            wav_len = torch.tensor([wav.shape[0]], device=dev, dtype=torch.long)
            proc, proc_len = self.model.preprocessor(input_signal=wav[None], length=wav_len)  # [1, F, T_mel]
            leaf = proc.detach().transpose(1, 2).contiguous().requires_grad_(True)  # [1, T_mel, F]
            leaf.retain_grad()
            t_feat = int(leaf.shape[1])

            enc, enc_len = self.model.encoder(audio_signal=leaf.transpose(1, 2), length=proc_len)  # [1, H, T_enc]
            # Prediction net teacher-forced on the ref subwords (NeMo prepends SOS=blank
            # internally, so the output has U+1 positions).
            tgt = torch.tensor([sub_ids], dtype=torch.long, device=dev)
            tgt_len = torch.tensor([s_len], dtype=torch.long, device=dev)
            dec_out, _, _ = self.model.decoder(targets=tgt, target_length=tgt_len)  # [1, H, U+1]
            logits = self.model.joint.joint(enc.transpose(1, 2), dec_out.transpose(1, 2))  # [1, T_enc, U+1, V(+dur)]
            # TDT (Token-and-Duration Transducer) joints append duration logits to
            # the token logits along the last axis. Softmax over the TOKEN part only
            # (the first vocab_size classes); RNN-T joints have no extra outputs, so
            # the slice is a no-op there.
            num_dur = int(logits.shape[-1]) - self.vocab_size
            log_probs = torch.log_softmax(logits[0, ..., : self.vocab_size].float(), dim=-1)  # [T_enc, U+1, V]

            if collect_lattice is not None:
                _cl_dur = None
                _cl_durs = None
                if num_dur > 0:
                    _cl_dur = (
                        torch.log_softmax(logits[0, ..., self.vocab_size :].float(), dim=-1).detach().cpu().numpy()
                    )
                    _cl_durs = [int(d) for d in self.model.cfg["decoding"]["durations"]]
                collect_lattice.append(
                    dict(log_probs=log_probs.detach().cpu().numpy(), log_dur=_cl_dur, durations=_cl_durs)
                )

            if self.per_token_score == "prefix" and num_dur > 0:
                # TDT: duration-aware prefix score (durations from the decoding cfg).
                durations = [int(d) for d in self.model.cfg["decoding"]["durations"]]
                assert len(durations) == num_dur, f"{durations=} vs {num_dur=}"
                log_dur = torch.log_softmax(logits[0, ..., self.vocab_size :].float(), dim=-1)  # [T, U+1, D]
                partial = self._tdt_prefix_scores(log_probs, log_dur, sub_ids, durations)  # [S]
            elif self.per_token_score == "prefix":
                # Proper transducer prefix score (RNN-T forward); best.
                partial = self._rnnt_prefix_scores(log_probs, sub_ids)  # [S]
            else:
                # Crude: s_u = logsumexp_t emission of token u (time-marginal).
                ys = torch.tensor(sub_ids, dtype=torch.long, device=dev)
                cols = log_probs[:, torch.arange(s_len, device=dev), ys]  # [T_enc, S]
                partial = torch.logsumexp(cols, dim=0)  # [S]

        partial_padded = torch.cat([partial, partial.new_zeros(1)])  # [S+1], exit slot
        targets = torch.tensor([sub_ids + [self.blank_idx]], dtype=torch.long, device=dev)  # [1, S+1]
        word_start_end = [[a, b] for (a, b) in word_ranges] + [[s_len, s_len + 1]]

        input_slice = (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([t_feat], dtype=torch.int64),
        )
        edges = torch.arange(t_feat + 1, dtype=torch.float64) * (orig_n_samples / max(t_feat, 1))
        input_raw_start_end = torch.stack([edges[:-1].round().long(), edges[1:].round().long()], dim=-1).unsqueeze(
            0
        )  # [1, T_feat, 2]

        print(
            f"[fwd] words={len(words)} subwords={s_len} T_feat={t_feat} T_enc={int(enc.shape[2])} "
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

    # ---- log_probs -------------------------------------------------------

    def log_probs(
        self,
        *,
        forward_output: ForwardOutput,
        start: Union[int, torch.Tensor],
        end: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Scatter the precomputed per-token scores into ``[B, n, V]`` at the
        target indices, so the gather-based extract loop reads s_u per token."""
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
        """Greedy RNN-T decoding via NeMo ``transcribe``. Returns the hyp text
        whitespace-split into words; normalization is left to the caller.
        (``max_new_tokens`` is unused -- NeMo decodes until end of audio.)"""
        assert len(raw_inputs) == 1
        assert raw_inputs_sample_rate == 16000, "NeMo transcribe expects 16 kHz"
        wav = raw_inputs[0].detach().cpu().numpy().astype("float32")
        with torch.no_grad():
            hyp = self.model.transcribe([wav], batch_size=1, verbose=False)[0]
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        return [text.split()]
