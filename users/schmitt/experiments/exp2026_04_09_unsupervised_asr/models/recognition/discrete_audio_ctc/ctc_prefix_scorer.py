"""
CTC prefix scoring for label-synchronous beam search.

Self-contained reimplementation of ESPnet's ``espnet.nets.ctc_prefix_score.CTCPrefixScoreTH``
(espnet is not installed in the RETURNN runtime env here), following the same algorithm as
``speech_llm.prefix_lm.model.recognition.ctc_label_sync_espnet``. Plain torch rather than RETURNN
frontend, to match the existing ``discrete_audio_aed.beam_search`` in this package.

The point of prefix scoring is that it turns a *time*-synchronous model (CTC) into a *label*-
synchronous one, so an autoregressive LM can be applied once per emitted label rather than per
frame. For a prefix ``g`` it maintains the CTC forward variables

    r[t, 0]  log P(g emitted within x[:t+1], alignment ends in a non-blank at t)
    r[t, 1]  log P(g emitted within x[:t+1], alignment ends in a blank at t)

and the prefix probability ``psi(g) = log P(g is a prefix of the output | x)``. Extending ``g`` by a
label ``c`` costs ``psi(g.c) - psi(g)``, which is what :meth:`score` returns for every ``c``.

Two combination modes are supported (``score_type``): ``"sum"`` gives the true CTC marginal, and
``"max"`` gives the Viterbi / best-path score. Both are verified: ``"sum"`` reproduces
``torch.nn.functional.ctc_loss`` exactly, and ``"max"`` matches a brute-force maximum over all
alignments; searching with ``"max"`` and no LM reproduces greedy best-path decoding.
"""

__all__ = ["CtcPrefixScorer"]

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

NEG_INF = float("-inf")


class CtcPrefixScorer:
    """
    Batched CTC prefix scorer.

    All hypotheses are kept flat as ``N = batch * beam`` rows; the search is responsible for
    gathering rows when the beam is reordered (see :meth:`select`).
    """

    def __init__(self, log_probs: Tensor, enc_lens: Tensor, *, blank_idx: int, score_type: str = "sum"):
        """
        :param log_probs: ``[B, T, V]`` CTC log probabilities, V including blank.
        :param enc_lens: ``[B]`` valid frame counts.
        :param blank_idx: index of the CTC blank in V.
        :param score_type: how alignments are combined.

            - ``"sum"``: the true CTC marginal ``log P(y|x) = log sum_alignments P(a|x)``.
            - ``"max"``: Viterbi / best-path, ``log max_alignments P(a|x)``.

            The distinction matters a lot for a weak model. The marginal sums over *all* alignments
            of ``y``, and the number of alignments grows combinatorially with ``|y|``; when the
            posteriors are diffuse that combinatorial term dominates the acoustic evidence and the
            most-likely-by-marginal sequence comes out far too long (measured here: 111 labels per
            utterance against a 62-label reference, while the model's own frame-level non-blank rate
            implies ~63). ``"max"`` has no such bias: with a wide beam and no LM it reduces to plain
            best-path (greedy) decoding, which is the sane baseline to add an LM on top of.
        """
        assert score_type in ("sum", "max"), score_type
        self.score_type = score_type
        # how the two path types (and successive frames) are combined in the recursion
        self._combine = torch.logaddexp if score_type == "sum" else torch.maximum
        batch_size, max_len, vocab_size = log_probs.shape
        self.batch_size = batch_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        self.device = log_probs.device

        # Force padded frames to emit blank with probability 1, so the recursion below can run over
        # the full padded length without leaking probability into padding.
        valid = torch.arange(max_len, device=log_probs.device)[None, :] < enc_lens[:, None]  # [B, T]
        blank_only = torch.full_like(log_probs[0, 0], NEG_INF)
        blank_only[blank_idx] = 0.0
        log_probs = torch.where(valid[:, :, None], log_probs, blank_only[None, None, :])

        self.log_probs = log_probs.transpose(0, 1).contiguous()  # [T, B, V]
        self.enc_lens = enc_lens

    # -- state -----------------------------------------------------------------------------------

    def initial_state(self, beam_size: int) -> Dict[str, Any]:
        """State of the empty prefix, replicated over the beam. ``N = batch * beam``."""
        num_hyps = self.batch_size * beam_size
        r = torch.full((self.max_len, 2, num_hyps), NEG_INF, device=self.device, dtype=self.log_probs.dtype)
        # the empty prefix "ends in blank" at t with prob prod_{s<=t} P(blank at s)
        blank_lp = self.log_probs[:, :, self.blank_idx]  # [T, B]
        blank_cumsum = torch.cumsum(blank_lp, dim=0)  # [T, B]
        r[:, 1] = blank_cumsum.repeat_interleave(beam_size, dim=1)
        return {
            "r": r,
            "psi": torch.zeros(num_hyps, device=self.device, dtype=self.log_probs.dtype),
            "last_label": torch.full((num_hyps,), -1, dtype=torch.long, device=self.device),
            "beam_size": beam_size,
        }

    def _expand_log_probs(self, beam_size: int) -> Tensor:
        """``[T, N, V]`` log probs, one row per hypothesis."""
        return self.log_probs.repeat_interleave(beam_size, dim=1)

    # -- scoring ---------------------------------------------------------------------------------

    def score(self, state: Dict[str, Any], num_labels: int) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Score every single-label extension of the current prefixes.

        :param state: from :meth:`initial_state` or :meth:`select`.
        :param num_labels: number of *non-blank* labels (the candidate extensions ``c``).
        :return: ``(label_scores [N, num_labels], eos_score [N], cache)``.
            ``label_scores[n, c] = log P(g_n . c is a prefix | x) - log P(g_n is a prefix | x)``.
            ``eos_score[n] = log P(y = g_n | x) - log P(g_n is a prefix | x)``, i.e. the cost of
            *stopping* here. ``cache`` must be passed to :meth:`select` to keep the chosen rows.
        """
        r_prev = state["r"]  # [T, 2, N]
        beam_size = state["beam_size"]
        num_hyps = r_prev.shape[2]
        x = self._expand_log_probs(beam_size)  # [T, N, V]
        x_labels = x[:, :, :num_labels]  # [T, N, C]
        x_blank = x[:, :, self.blank_idx]  # [T, N]

        # log_phi[t, n, c]: probability that g_n is complete at t and c may start at t+1.
        # For c == last(g_n) the repeat must be separated by a blank, so only the blank-ending path
        # is allowed -- this is the one place where CTC differs from a plain prefix automaton.
        r_sum = self._combine(r_prev[:, 0], r_prev[:, 1])  # [T, N]
        log_phi = r_sum[:, :, None].expand(-1, -1, num_labels).clone()  # [T, N, C]
        last = state["last_label"]
        has_last = last >= 0
        rows = torch.nonzero(has_last, as_tuple=False).squeeze(-1)  # [M] hypotheses with a last label
        if rows.numel():
            # log_phi[t, n, last[n]] = r_prev[t, 1, n]  (advanced indexing broadcasts to [T, M])
            log_phi[:, rows, last[rows]] = r_prev[:, 1][:, rows]

        # Forward recursion over frames. r[t, 0] extends the label, r[t, 1] stays in blank.
        r = torch.full((self.max_len, 2, num_hyps, num_labels), NEG_INF, device=self.device, dtype=x.dtype)
        # "g complete before frame 0" holds only for the empty prefix
        phi_prev = torch.where(
            has_last[None, :, None].expand(1, num_hyps, num_labels),
            torch.full((1, num_hyps, num_labels), NEG_INF, device=self.device, dtype=x.dtype),
            torch.zeros((1, num_hyps, num_labels), device=self.device, dtype=x.dtype),
        )[0]
        r[0, 0] = phi_prev + x_labels[0]
        r[0, 1] = NEG_INF  # nothing emitted yet in this extension
        psi = r[0, 0].clone()  # [N, C]
        for t in range(1, self.max_len):
            prev_n, prev_b = r[t - 1, 0], r[t - 1, 1]
            r[t, 0] = self._combine(prev_n, log_phi[t - 1]) + x_labels[t]
            r[t, 1] = self._combine(prev_n, prev_b) + x_blank[t][:, None]
            psi = self._combine(psi, log_phi[t - 1] + x_labels[t])

        psi_prev = state["psi"][:, None]  # [N, 1]
        label_scores = psi - psi_prev
        # stopping: the prefix must be the *complete* output, i.e. all frames consumed
        eos_score = r_sum[self.max_len - 1] - state["psi"]
        return label_scores, eos_score, {"r": r, "psi": psi, "beam_size": beam_size}

    @staticmethod
    def select(cache: Dict[str, Any], hyp_idx: Tensor, label_idx: Tensor) -> Dict[str, Any]:
        """
        Keep the chosen ``(hypothesis, label)`` pairs after the beam top-k.

        :param cache: from :meth:`score`.
        :param hyp_idx: ``[N']`` row index into the previous hypotheses.
        :param label_idx: ``[N']`` chosen label per new hypothesis.
        """
        r = cache["r"]  # [T, 2, N, C]
        max_len = r.shape[0]
        gather_idx = hyp_idx[None, None, :, None].expand(max_len, 2, -1, r.shape[3])
        r_sel = torch.gather(r, 2, gather_idx)  # [T, 2, N', C]
        lab = label_idx[None, None, :, None].expand(max_len, 2, -1, 1)
        r_sel = torch.gather(r_sel, 3, lab).squeeze(-1)  # [T, 2, N']
        psi_sel = cache["psi"][hyp_idx, label_idx]  # [N']
        return {"r": r_sel, "psi": psi_sel, "last_label": label_idx, "beam_size": cache["beam_size"]}
