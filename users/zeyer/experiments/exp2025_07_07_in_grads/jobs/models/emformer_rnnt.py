"""Emformer RNN-T (transducer) model adapter for the grad-based forced-alignment pipeline.
Adds the TRANSDUCER architecture to the breadth (CTC + AED/LLM + RNN-T).

Model: ``torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH``
(self-contained, like MMS_FA was for CTC):
an Emformer encoder + RNN-T predictor + joiner, 4096 sentencepiece subwords.
Log-mel features at 100 Hz; encoder ~25 Hz.

Per-token score
---------------
Same pipeline as every other model: a scalar per-token score, grad'd w.r.t. the input.
For the transducer we teacher-force the target subwords
and take the time-marginal emission log-prob of each token from the joiner:

    s_u = logsumexp_t  log softmax(joiner_logits[t, u, :])[y_u]

i.e. "how strongly is token u emitted, summed over time".
Its gradient w.r.t. the log-mel input localizes where token u is in time
(flows through the Emformer encoder + joiner; the predictor/target path is fixed by teacher forcing).
This is a tractable transducer analog of the autoregressive ``log p(y_i|y_<i,x)`` / CTC partial score
(it marginalizes the alignment via the logsumexp over time rather than the full RNN-T forward lattice).

Grad target: the log-mel features (100 Hz). Batch size 1 only.
"""

from __future__ import annotations

from typing import Optional, Union, List
import time

import numpy as np
import torch

from .base import BaseModelInterface, ForwardOutput


class EmformerRnnt(BaseModelInterface):
    """torchaudio Emformer RNN-T transducer. See module docstring."""

    def __init__(self, *, device: torch.device, per_token_score: str = "emission", version: int = 1):
        super().__init__()
        assert version >= 1
        assert per_token_score in ("prefix", "emission"), per_token_score
        self.device = device
        self.per_token_score = per_token_score
        self.version = version

        print("Import torchaudio / Emformer RNN-T bundle...")
        start_time = time.time()
        import torchaudio  # noqa: F401
        from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH as bundle

        self._bundle = bundle
        self.model = bundle.get_decoder().model.to(device).eval()
        self.feature_extractor = bundle.get_feature_extractor()
        self.token_processor = bundle.get_token_processor()
        self._sp = self.token_processor.sp_model
        self.blank_idx = int(bundle._blank)
        self.target_sr = bundle.sample_rate
        self.vocab_size = self.blank_idx + 1  # joiner outputs subwords + blank
        print(f"  ({time.time() - start_time:.1f}s) vocab={self.vocab_size} blank_idx={self.blank_idx}")

    # ---- Helpers --------------------------------------------------------

    def _rnnt_prefix_scores(self, logp: torch.Tensor, ys: List[int]) -> torch.Tensor:
        """Proper transducer per-token prefix score (analog of the CTC partial score).
        RNN-T forward over the T x (U+1) lattice,
        then the telescoping difference of the time-integrated occupancy.

        :param logp: ``[T, U+1, V]`` log-probs from the joiner (teacher-forced).
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

    def _tokenize_words(self, words: List[str]):
        """Encode the transcription to sentencepiece subwords
        and group them into words via the ``▁`` (word-start) marker.

        Returns ``(subword_ids, word_ranges)``
        where word_ranges[w] = (start, end) subword indices for word w (end exclusive).
        """
        text = " ".join(w.lower() for w in words)
        ids = list(self._sp.encode(text))
        pieces = [self._sp.id_to_piece(i) for i in ids]
        word_ranges = []
        cur_start = 0
        for j, p in enumerate(pieces):
            if j > 0 and p.startswith("▁"):  # word-start marker
                word_ranges.append((cur_start, j))
                cur_start = j
        word_ranges.append((cur_start, len(pieces)))
        if len(word_ranges) == len(words):
            return ids, word_ranges
        # Fallback: the in-context grouping didn't yield one group per word (rare).
        # Encode each word separately so the word count is exact.
        ids, word_ranges = [], []
        for w in words:
            wp = list(self._sp.encode(w.lower())) or list(self._sp.encode("▁" + w.lower()))
            assert wp, f"word {w!r} encoded to empty"
            word_ranges.append((len(ids), len(ids) + len(wp)))
            ids.extend(wp)
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
    ) -> ForwardOutput:
        assert raw_inputs_sample_rate is not None
        assert len(raw_inputs) == 1, "EmformerRnnt wrapper supports batch size 1 only"
        assert isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2
        if omitted_prev_context is not None and int(omitted_prev_context[0]) > 0:
            raise NotImplementedError("EmformerRnnt chunked context not implemented yet")

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

        # Log-mel features are the grad leaf (100 Hz).
        # Grad flows feats -> Emformer encoder -> joiner -> per-token scores.
        feats, flen = self.feature_extractor(wav.cpu())  # [T_feat, 80] (extractor is CPU numpy-ish)
        feats = feats.to(dev)
        t_feat = int(feats.shape[0])
        grad_leaf = feats.detach().unsqueeze(0).requires_grad_(True)  # [1, T_feat, 80]
        grad_leaf.retain_grad()

        with torch.enable_grad():
            enc, enc_len = self.model.transcribe(grad_leaf, flen.reshape(1).to(dev))  # [1, T_enc, H]
            # Teacher-force: predictor input = blank(SOS) + subwords.
            tgt = torch.tensor([[self.blank_idx] + sub_ids], dtype=torch.int32, device=dev)
            tgt_len = torch.tensor([tgt.shape[1]], dtype=torch.int32, device=dev)
            pred, pred_len, _ = self.model.predict(tgt, tgt_len, None)  # [1, U+1, H]
            logits, _, _ = self.model.join(enc, enc_len, pred, pred_len)  # [1, T_enc, U+1, V]
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)  # [T_enc, U+1, V]

            if self.per_token_score == "prefix":
                # Proper transducer prefix score (RNN-T forward); best.
                partial = self._rnnt_prefix_scores(log_probs, sub_ids)  # [S]
            else:
                # Crude: s_u = logsumexp_t emission of token u (time-marginal).
                ys = torch.tensor(sub_ids, dtype=torch.long, device=dev)
                cols = log_probs[:, torch.arange(s_len, device=dev), ys]  # [T_enc, S]
                partial = torch.logsumexp(cols, dim=0)  # [S]

        partial_padded = torch.cat([partial, partial.new_zeros(1)])  # [S+1], exit slot
        targets = torch.tensor([sub_ids + [self.blank_idx]], dtype=torch.long, device=dev)  # [1, S+1]
        # Per-word target_start_end from the subword grouping, + exit slot.
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
            f"[fwd] words={len(words)} subwords={s_len} T_feat={t_feat} T_enc={int(enc.shape[1])} "
            f"text={' '.join(w.lower() for w in words)!r}",
            flush=True,
        )
        return ForwardOutput(
            inputs=grad_leaf,
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
