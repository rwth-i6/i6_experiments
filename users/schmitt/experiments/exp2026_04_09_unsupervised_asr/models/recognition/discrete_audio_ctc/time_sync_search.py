"""
Time-synchronous CTC prefix beam search with phoneme-LM shallow fusion.

This is the classic CTC prefix beam search (Hannun et al. 2014), and it exists because the
label-synchronous variant in :mod:`.label_sync_search` is **not valid for standalone CTC decoding**.
There, all beam entries share a label count but not a frame count, and the running score
``psi(g) = max/sum over completion times`` is therefore compared across hypotheses that have consumed
different amounts of audio: a prefix that crams its labels into the first frames scores only those
frames, while the correct prefix spreads over the whole utterance and scores all of them. The beam
fills with "fast" prefixes that then keep extending. Measured on smooth synthetic posteriors, that
search sits 13.4 nats below a *verified* optimum and widening the beam 20x recovers 0.5 nats. (The
reference it was ported from uses CTC prefix scores only as an auxiliary term next to a dominant
speech-LLM decoder, which hides this.)

Here the beam advances one **frame** at a time, so every hypothesis has consumed exactly the same
audio and the scores are comparable by construction. Each prefix carries the two CTC forward
probabilities

    p_b   log P(prefix, alignment ends in blank at t)
    p_nb  log P(prefix, alignment ends in a non-blank at t)

and per frame each beam entry either *stays* (emits blank, or repeats its last label -- neither
changes the prefix) or *extends* by one label. The LM contributes exactly once per emitted label,
which is the point of doing prefix search at all.
"""

__all__ = ["ctc_lm_time_sync_search"]

from typing import Optional, Tuple

import torch
from torch import Tensor

NEG_INF = float("-inf")


def _lm_next_log_probs(lm, tokens: Tensor, lens: Tensor) -> Tensor:
    """
    ``[N, V_lm]`` LM distribution over the next label for each prefix.

    Recomputed from the full prefix rather than kept as an incremental state: in a time-synchronous
    search the beam entries have *different* label counts, so their KV caches would be ragged. The
    prefixes are short (tens of labels) and the LM is small, so the full recompute is cheap; it can
    be replaced by explicit cache management if it ever shows up in profiles.
    """
    num_hyps = tokens.shape[0]
    bos = torch.full((num_hyps, 1), lm.bos_idx, dtype=torch.long, device=tokens.device)
    inp = torch.cat([bos, tokens], dim=1)
    logits = lm.score_text(inp, (lens + 1).to(torch.int32))  # [N, L+1, V]
    last = logits[torch.arange(num_hyps, device=tokens.device), lens]  # [N, V]
    return last.float().log_softmax(dim=-1)


def _merge_duplicate_prefixes(p_b: Tensor, p_nb: Tensor, tokens: Tensor, lens: Tensor, beam_size: int,
                              score_type: str = "sum"):
    """
    Sum the probabilities of beam entries that represent the *same* prefix.

    Required by CTC prefix search: a prefix ``l.c`` can be reached both by extending ``l`` at this
    frame and by an entry created at an earlier frame, and those are the same hypothesis. Keeping
    them apart splits the probability mass and underestimates the prefix -- measured at ~18 nats on
    a synthetic case, which also distorts the beam ranking (a prefix reachable many ways gets
    systematically underrated).

    Duplicates are collapsed onto their lowest-indexed representative; the others are set to -inf so
    the next top-k drops them.
    """
    batch_size = p_b.shape[0] // beam_size
    max_lab = tokens.shape[1]
    tk = tokens.view(batch_size, beam_size, max_lab)
    ln = lens.view(batch_size, beam_size)
    idx = torch.arange(beam_size, device=p_b.device)

    if max_lab == 0:  # all prefixes empty -> all identical
        same = ln[:, :, None] == ln[:, None, :]
    else:
        pos = torch.arange(max_lab, device=p_b.device)
        beyond = pos[None, None, None, :] >= ln[:, :, None, None]  # positions past the prefix
        eq = tk[:, :, None, :] == tk[:, None, :, :]
        same = (ln[:, :, None] == ln[:, None, :]) & (eq | beyond).all(dim=-1)

    # representative = lowest index among identical prefixes
    first = torch.where(same, idx[None, :, None].expand_as(same), torch.full_like(same, beam_size, dtype=torch.long))
    first = first.min(dim=1).values  # [B, K] -> representative index of each entry
    group = first[:, None, :] == idx[None, :, None]  # [B, K(rep), K(member)]

    def _collapse(v: Tensor) -> Tensor:
        v = v.view(batch_size, beam_size)
        masked = torch.where(group, v[:, None, :], torch.full_like(v[:, None, :], NEG_INF))
        merged = torch.logsumexp(masked, dim=2) if score_type == "sum" else masked.max(dim=2).values
        keep = first == idx[None, :]
        return torch.where(keep, merged, torch.full_like(merged, NEG_INF)).reshape(-1)

    return _collapse(p_b), _collapse(p_nb)


def ctc_lm_time_sync_search(
    *,
    ctc_log_probs: Tensor,
    enc_lens: Tensor,
    lm,
    beam_size: int,
    blank_idx: int,
    num_labels: int,
    lm_scale: float = 0.0,
    ctc_scale: float = 1.0,
    length_reward: float = 0.0,
    score_type: str = "sum",
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    :param ctc_log_probs: ``[B, T, V]`` CTC log probs (V includes blank).
    :param enc_lens: ``[B]`` valid frame counts.
    :param lm: phoneme LM with ``score_text`` + ``bos_idx``/``eos_idx``; ``None`` -> pure CTC.
    :param beam_size: beam width (number of prefixes kept per sequence).
    :param lm_scale: shallow-fusion weight, applied once per emitted label.
    :param length_reward: per-label bonus, counteracting the LM's length bias.
    :param score_type: ``"sum"`` = CTC marginal (sums over alignments), ``"max"`` = Viterbi. The
        marginal favours longer sequences because the alignment count grows with ``|y|``; with a weak
        model that combinatorial term dominates. ``"max"`` has no such bias and converges to
        best-path (greedy) decoding as the beam widens.
    :return: ``(tokens [B, beam, L], scores [B, beam], lens [B, beam])``, best first.
    """
    assert score_type in ("sum", "max"), score_type
    combine = torch.logaddexp if score_type == "sum" else torch.maximum
    batch_size, max_len, _ = ctc_log_probs.shape
    device = ctc_log_probs.device
    num_hyps = batch_size * beam_size
    use_lm = lm is not None and lm_scale != 0.0

    # padded frames emit blank with probability 1, so they cannot change any prefix
    valid = torch.arange(max_len, device=device)[None, :] < enc_lens[:, None]
    blank_only = torch.full_like(ctc_log_probs[0, 0], NEG_INF)
    blank_only[blank_idx] = 0.0
    x_all = torch.where(valid[:, :, None], ctc_log_probs, blank_only[None, None, :])
    if ctc_scale != 1.0:
        x_all = ctc_scale * x_all  # 0 and -inf are fixed points, so padded frames stay blank-only
    x_all = x_all.repeat_interleave(beam_size, dim=0)  # [N, T, V]

    # empty prefix: trivially "ends in blank"; only beam 0 is alive so the beam does not start with
    # `beam_size` duplicates of it.
    p_b = torch.full((num_hyps,), NEG_INF, device=device)
    p_b.view(batch_size, beam_size)[:, 0] = 0.0
    p_nb = torch.full((num_hyps,), NEG_INF, device=device)
    tokens = torch.zeros((num_hyps, 0), dtype=torch.long, device=device)
    lens = torch.zeros(num_hyps, dtype=torch.long, device=device)
    lm_lp = _lm_next_log_probs(lm, tokens, lens) if use_lm else None

    batch_offset = (torch.arange(batch_size, device=device)[:, None] * beam_size)

    for t in range(max_len):
        x = x_all[:, t]  # [N, V]
        both = combine(p_b, p_nb)  # [N]

        # --- stay: blank, or a repeat of the last label. Neither changes the prefix. ------------
        stay_p_b = both + x[:, blank_idx]
        has_last = lens > 0
        last = tokens.gather(1, (lens - 1).clamp_min(0)[:, None]).squeeze(1) if tokens.shape[1] else torch.zeros_like(lens)
        rep = p_nb + x.gather(1, last[:, None]).squeeze(1)
        stay_p_nb = torch.where(has_last, rep, torch.full_like(rep, NEG_INF))
        stay_score = combine(stay_p_b, stay_p_nb)

        # --- extend by label c ------------------------------------------------------------------
        # a repeat (c == last) must be separated by a blank, so it may only come from p_b
        base = both[:, None].expand(num_hyps, num_labels).clone()
        rows = torch.nonzero(has_last, as_tuple=False).squeeze(-1)
        if rows.numel():
            base[rows, last[rows]] = p_b[rows]
        ext_p_nb = base + x[:, :num_labels] + length_reward
        if use_lm:
            ext_p_nb = ext_p_nb + lm_scale * lm_lp[:, :num_labels]

        cand = torch.cat([stay_score[:, None], ext_p_nb], dim=-1)  # [N, 1 + C]
        cand = cand.reshape(batch_size, beam_size * (num_labels + 1))
        _, flat = torch.topk(cand, k=beam_size, dim=-1)
        src = (flat // (num_labels + 1) + batch_offset).reshape(num_hyps)
        choice = (flat % (num_labels + 1)).reshape(num_hyps)  # 0 = stay, else label+1
        emitted = choice > 0
        label = (choice - 1).clamp_min(0)

        p_b = torch.where(emitted, torch.full_like(stay_p_b, NEG_INF), stay_p_b[src])
        p_nb = torch.where(emitted, ext_p_nb[src, label], stay_p_nb[src])

        tokens = tokens[src]
        lens = lens[src]
        if bool(emitted.any()):
            rows = torch.nonzero(emitted, as_tuple=False).squeeze(-1)
            if int(lens.max()) + 1 > tokens.shape[1]:  # widen only when a prefix outgrows the buffer
                tokens = torch.cat([tokens, torch.zeros((num_hyps, 1), dtype=torch.long, device=device)], dim=1)
            tokens[rows, lens[rows]] = label[rows]
            lens = lens + emitted.long()
            if use_lm:
                lm_lp = _lm_next_log_probs(lm, tokens, lens)
        elif use_lm:
            lm_lp = lm_lp[src]

        p_b, p_nb = _merge_duplicate_prefixes(p_b, p_nb, tokens, lens, beam_size, score_type)

    scores = combine(p_b, p_nb)
    if use_lm:  # the LM's own end-of-sequence probability, once, for the completed prefix
        scores = scores + lm_scale * lm_lp[:, lm.eos_idx]

    scores = scores.reshape(batch_size, beam_size)
    order = scores.argsort(dim=-1, descending=True)
    tokens = tokens.reshape(batch_size, beam_size, -1)
    lens = lens.reshape(batch_size, beam_size)
    tokens = torch.gather(tokens, 1, order[:, :, None].expand(-1, -1, tokens.shape[-1]))
    return tokens, torch.gather(scores, 1, order), torch.gather(lens, 1, order)
