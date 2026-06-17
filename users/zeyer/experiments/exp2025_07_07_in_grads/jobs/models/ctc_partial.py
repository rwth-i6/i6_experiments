"""Shared CTC per-token partial-score computation for the grad-align CTC wrappers.

``acc[k] = log Σ_t α_t(k)`` over the 2S+1 extended states (blank-interleaved) of the forced target,
computed exactly by the CTC forward. The per-token score selects a (telescoping) difference of these
accumulated states. The difference forms are the exact CTC prefix-score / per-label conditional --
``Δ_i = log p(y_i | y_<i, x)`` -- consistent with AED/LLM teacher-forced scoring and with the CTC
prefix term in label-synchronous search. Modes (``idx = 2i+1`` is token i's label state):

  raw_partial    acc[2i+1]                                  NON-telescoping (legacy parakeet/owsm bug)
  prefix_diff    acc[2i+1] - acc[2i-1]                      label i minus previous label
  inb_false      acc[2i+1] - acc[2i]
  inb_true       acc[2i+2] - acc[2i]                        (wav2vec2 include_next_blank=True)
  inb_both       logaddexp(acc[2i+1], acc[2i+2]) - acc[2i]
  inb_both_prev  logaddexp(acc[2i+1], acc[2i+2]) - logaddexp(acc[2i], acc[2i-1])

The legacy ``include_next_blank`` values map as: False->inb_false, True->inb_true, "both"->inb_both,
"both_prev"->inb_both_prev (see ``include_next_blank_to_mode``).
"""

from typing import List, Union
import torch

CTC_PARTIAL_SCORE_MODES = (
    "raw_partial",
    "prefix_diff",
    "inb_false",
    "inb_true",
    "inb_both",
    "inb_both_prev",
    "prefix_fwd",
)


def include_next_blank_to_mode(inb: Union[bool, str]) -> str:
    return {False: "inb_false", True: "inb_true", "both": "inb_both", "both_prev": "inb_both_prev"}[inb]


def ctc_prefix_forward_scores(lp: torch.Tensor, target_ids: List[int], blank: int) -> torch.Tensor:
    """The REAL CTC prefix score (Graves / ESPnet ``CTCPrefixScoreTH``, as used in label-sync beam
    search incl. DLM-sum): per-token conditional ``log p(y_i | y_<i, x)`` via the free-continuation
    prefix forward. Unlike the forced-target occupancy differences (``ctc_partial_scores``), this is
    the EXACT conditional (verified == brute-force) and is blank-aware by construction (robust to
    repeated labels). Differentiable. Cost O(S*T) over emission frames T (~12.5-50 Hz, so cheap).

    The per-prefix forward variables are exactly the forced-FSA forwards at the prefix's label/blank
    states, so we accumulate the prefix log-prob ``psi_i`` inside ONE vectorized forced-FSA forward
    (O(T) Python steps, vectorized over the S tokens; ~5-24x faster than the per-token scalar scan).
    Verified == brute-force enumeration (incl. repeated labels).
    """
    T = lp.shape[0]
    S = len(target_ids)
    if S == 0:
        return lp.new_zeros(0)
    neg = -1.0e30
    device = lp.device
    tgt = torch.tensor([int(c) for c in target_ids], device=device, dtype=torch.long)
    ext: List[int] = [blank]
    for c in target_ids:
        ext.append(int(c))
        ext.append(blank)
    sx = len(ext)
    ext_t = torch.tensor(ext, device=device, dtype=torch.long)
    emit = lp[:, ext_t]  # [T, 2S+1]
    odd = torch.arange(sx, device=device) % 2 == 1
    diff = torch.zeros(sx, dtype=torch.bool, device=device)
    diff[2:] = ext_t[2:] != ext_t[:-2]
    skip_ok = odd & diff
    label_emit = lp[:, tgt]  # [T, S]: emission of each token's label
    ar = torch.arange(S, device=device)
    prev_blank_idx = ar * 2  # state 2i = blank before y_i
    prev_label_idx = (ar * 2 - 1).clamp(min=0)  # state 2i-1 = prev label (invalid for i=0)
    repeat = torch.zeros(S, dtype=torch.bool, device=device)
    repeat[1:] = tgt[1:] == tgt[:-1]
    label_valid = (ar >= 1) & (~repeat)  # phi includes the prev-label term only if i>=1 and not a repeat
    alpha = lp.new_full((sx,), neg)
    alpha[0] = emit[0, 0]
    if sx > 1:
        alpha[1] = emit[0, 1]
    phi_m1 = lp.new_full((S,), neg)
    phi_m1[0] = 0.0  # empty prefix is "complete" before t=0, so token 0 can emit at t=0
    psi = phi_m1 + label_emit[0]  # t=0 contribution
    neg1 = lp.new_full((1,), neg)
    neg2 = lp.new_full((2,), neg)
    neg_s = lp.new_full((S,), neg)
    for t in range(1, T):
        phi_b = alpha[prev_blank_idx]  # alpha_{t-1}(2i)
        phi_l = torch.where(label_valid, alpha[prev_label_idx], neg_s)  # alpha_{t-1}(2i-1)
        phi = torch.logaddexp(phi_b, phi_l)  # phi_{t-1}
        psi = torch.logaddexp(psi, phi + label_emit[t])  # += phi_{t-1} + log y_{y_i}(t)
        a_prev = torch.cat([neg1, alpha[:-1]])
        a_skip = torch.where(skip_ok, torch.cat([neg2, alpha[:-2]]), lp.new_full((sx,), neg))
        alpha = emit[t] + torch.logsumexp(torch.stack([alpha, a_prev, a_skip], 0), 0)
    # psi[i] = log P(prefix y_0..y_i); per-token conditional = psi[i] - psi[i-1]
    return psi - torch.cat([lp.new_zeros(1), psi[:-1]])


def ctc_prefix_forward_scores_batched(
    lp: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor, blank: int, lengths: torch.Tensor
) -> torch.Tensor:
    """Batched :func:`ctc_prefix_forward_scores` over a B>1 batch of different-length seqs.

    :param lp: ``[B, Tm, V]`` log-probs, zero-padded in time.
    :param targets: ``[B, Sm]`` padded token ids.
    :param target_lengths: ``[B]`` = ``S_b``.
    :param lengths: ``[B]`` valid emission frames ``T_b``.
    :return: ``[B, Sm]`` per-token scores; mask each seq to its own ``S_b``.

    Same forced-FSA forward as the per-seq version, with a batch axis on the ``2S+1`` states:
    each seq is FROZEN once ``t >= T_b``, and padded states / tokens are ``-1e30``-masked.
    Verified bit-equal (max|Δ| = 0) to the per-seq version on a ragged batch.
    """
    neg = -1.0e30
    B, Tm, _ = lp.shape
    dev = lp.device
    s_b = target_lengths
    Sm = targets.shape[1]
    sxm = 2 * Sm + 1
    tgt = targets
    ext = torch.full((B, sxm), blank, device=dev, dtype=torch.long)
    ext[:, 1::2] = tgt  # even=blank, odd=label
    bi, tr = torch.arange(B, device=dev), torch.arange(Tm, device=dev)
    emit = lp[bi[:, None, None], tr[None, :, None], ext[:, None, :]]  # [B, Tm, sxm]
    label_emit = lp[bi[:, None, None], tr[None, :, None], tgt[:, None, :]]  # [B, Tm, Sm]
    sx_b = 2 * s_b + 1
    s_idx = torch.arange(sxm, device=dev)
    emit = torch.where((s_idx[None, :] < sx_b[:, None])[:, None, :], emit, torch.full_like(emit, neg))
    diff = torch.zeros(B, sxm, dtype=torch.bool, device=dev)
    diff[:, 2:] = ext[:, 2:] != ext[:, :-2]
    skip_ok = (s_idx % 2 == 1)[None, :] & diff
    ar = torch.arange(Sm, device=dev)
    pbi, pli = ar * 2, (ar * 2 - 1).clamp(min=0)
    repeat = torch.zeros(B, Sm, dtype=torch.bool, device=dev)
    repeat[:, 1:] = tgt[:, 1:] == tgt[:, :-1]
    label_valid = (ar[None, :] >= 1) & (~repeat) & (ar[None, :] < s_b[:, None])
    alpha = lp.new_full((B, sxm), neg)
    alpha[:, 0] = emit[:, 0, 0]
    alpha[:, 1] = torch.where(sx_b > 1, emit[:, 0, 1], lp.new_full((B,), neg))
    psi = lp.new_full((B, Sm), neg)
    psi[:, 0] = 0.0
    psi = psi + label_emit[:, 0]
    for t in range(1, Tm):
        active = (t < lengths)[:, None]
        phi = torch.logaddexp(
            alpha[bi[:, None], pbi[None, :]],
            torch.where(label_valid, alpha[bi[:, None], pli[None, :]], lp.new_full((B, Sm), neg)),
        )
        psi_n = torch.logaddexp(psi, phi + label_emit[:, t])
        a_prev = torch.cat([lp.new_full((B, 1), neg), alpha[:, :-1]], 1)
        a_skip = torch.where(
            skip_ok, torch.cat([lp.new_full((B, 2), neg), alpha[:, :-2]], 1), lp.new_full((B, sxm), neg)
        )
        alpha_n = emit[:, t] + torch.logsumexp(torch.stack([alpha, a_prev, a_skip], 0), 0)
        psi = torch.where(active, psi_n, psi)
        alpha = torch.where(active, alpha_n, alpha)
    return psi - torch.cat([lp.new_zeros(B, 1), psi[:, :-1]], 1)


def ctc_prefix_forward_scores_scan(
    lp: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor, blank: int, lengths: torch.Tensor
) -> torch.Tensor:
    """``torch.compile``-friendly twin of :func:`ctc_prefix_forward_scores_batched`, via ``torch.scan``.

    Identical ``[B, Sm]`` per-token scores and FSA-forward, but the T-loop is a ``scan`` HOP -- so it does NOT
    unroll (a Python ``for t in range(Tm)`` unrolls -> recompiles per length). Wrap in
    ``torch.compile(fullgraph=True)`` and ``maybe_mark_dynamic(lp/targets, 0)`` (batch) + ``mark_dynamic`` on
    Tm/Sm: then B+Tm+Sm are all dynamic, fused (inductor), correct grad, compiled ONCE for all shapes.

    Each non-obvious line works around a specific torch-2.12 scan/inductor limitation, tagged [WA: ...]:
    """
    from torch._higher_order_ops.scan import scan

    neg = -1.0e30
    B, Tm, _ = lp.shape
    dev = lp.device
    Sm = targets.shape[1]
    sxm = 2 * Sm + 1
    # [WA strided-assign] ext[:,1::2]=targets specializes Sm under compile -> interleave [blank,tok] via stack+flatten.
    blanks = torch.full((B, Sm), blank, device=dev, dtype=torch.long)
    ext = torch.cat(
        [torch.stack([blanks, targets], 2).flatten(1), torch.full((B, 1), blank, device=dev, dtype=torch.long)], 1
    )  # [B, sxm] = blank, t0, blank, t1, ..., blank
    # [WA advanced-index] lp[bi,tr,ext] specializes B -> torch.gather; expand(-1,..) keeps B/sxm symbolic.
    emit = lp.gather(2, ext[:, None, :].expand(-1, Tm, -1))  # [B, Tm, sxm]
    label_emit = lp.gather(2, targets[:, None, :].expand(-1, Tm, -1))  # [B, Tm, Sm]
    s_idx = torch.arange(sxm, device=dev)
    sx_b = 2 * target_lengths + 1
    emit = torch.where((s_idx[None, :] < sx_b[:, None])[:, None, :], emit, torch.full_like(emit, neg))
    diff = torch.zeros(B, sxm, dtype=torch.bool, device=dev)
    diff[:, 2:] = ext[:, 2:] != ext[:, :-2]
    skip_ok = (s_idx % 2 == 1)[None, :] & diff
    ar = torch.arange(Sm, device=dev)
    pbi, pli = ar * 2, (ar * 2 - 1).clamp(min=0)
    repeat = torch.zeros(B, Sm, dtype=torch.bool, device=dev)
    repeat[:, 1:] = targets[:, 1:] == targets[:, :-1]
    label_valid = (ar[None, :] >= 1) & (~repeat) & (ar[None, :] < target_lengths[:, None])
    neg_BSm, neg_Bsxm = lp.new_full((B, Sm), neg), lp.new_full((B, sxm), neg)
    neg_B1, neg_B2 = lp.new_full((B, 1), neg), lp.new_full((B, 2), neg)
    # [WA const-init] scan zeros the xs-grad unless the carry init REQUIRES grad -> build alpha0/psi0 from lp
    # (the real frame-0 state) via where (not in-place); t=0 is then a no-op inside the scan.
    alpha0 = torch.where(s_idx[None, :] < 2, emit[:, 0, :], neg_Bsxm)
    psi0 = torch.where(ar[None, :] == 0, label_emit[:, 0, :], neg_BSm)
    tr = torch.arange(Tm, device=dev)
    # [WA derived-scan-len] xs=emit[1:] makes the scan length Tm-1 (a DERIVED symbol) -> inductor
    # decompose_scan KeyError s84. Pass FULL-length xs (scan length = bound Tm); skip t=0 via is_first.
    is_first = (tr == 0)[:, None, None]  # [Tm,1,1]; broadcasts over B in combine (no B-size expand)
    active = tr[:, None, None] < lengths[None, :, None]  # [Tm, B, 1]
    xs = (
        emit.transpose(0, 1).contiguous(),
        label_emit.transpose(0, 1).contiguous(),
        active.contiguous(),
        is_first.contiguous(),
    )

    def combine(carry, x):
        alpha, psi = carry
        emt, lmt, act, isf = x
        # index_select (1-D index), not alpha[:,pbi] advanced indexing (dynamic-friendly).
        phi = torch.logaddexp(alpha.index_select(1, pbi), torch.where(label_valid, alpha.index_select(1, pli), neg_BSm))
        psi_n = torch.logaddexp(psi, phi + lmt)
        a_prev = torch.cat([neg_B1, alpha[:, :-1]], 1)
        a_skip = torch.where(skip_ok, torch.cat([neg_B2, alpha[:, :-2]], 1), neg_Bsxm)
        # [WA logsumexp] logsumexp(stack(...)) errors inside compiled scan -> chained logaddexp.
        a_n = emt + torch.logaddexp(torch.logaddexp(alpha, a_prev), a_skip)
        upd = act & (~isf)  # t=0 no-op (init already IS frame 0); t>=1 active frames recurse.
        a_o, psi_o = torch.where(upd, a_n, alpha), torch.where(upd, psi_n, psi)
        return (a_o, psi_o), psi_o.clone()  # [WA aliasing] clone y -> HOPs forbid carry<->output aliasing.

    (_, psi_f), _ = scan(combine, (alpha0, psi0), xs)
    return psi_f - torch.cat([lp.new_zeros(B, 1), psi_f[:, :-1]], 1)


def ctc_accum_states(lp: torch.Tensor, target_ids: List[int], blank: int) -> torch.Tensor:
    """``acc[k] = log Σ_t α_t(k)`` over the 2S+1 extended states. Differentiable w.r.t. ``lp`` ([T,V])."""
    device, dtype = lp.device, lp.dtype
    T = lp.shape[0]
    neg = -1.0e9  # finite log-zero: true -inf makes the where/logsumexp backward produce 0*inf=NaN.
    ext: List[int] = [blank]
    for c in target_ids:
        ext.append(int(c))
        ext.append(blank)
    sx = len(ext)  # 2S+1
    ext_t = torch.tensor(ext, device=device, dtype=torch.long)
    emit = lp[:, ext_t]  # [T, sx]
    # CTC skip s-2 -> s allowed only into a LABEL state (odd s) whose label DIFFERS from 2 back.
    odd = torch.arange(sx, device=device) % 2 == 1
    diff = torch.zeros(sx, dtype=torch.bool, device=device)
    diff[2:] = ext_t[2:] != ext_t[:-2]
    skip_ok = odd & diff
    alpha = torch.full((sx,), neg, device=device, dtype=dtype)
    alpha[0] = emit[0, 0]
    if sx > 1:
        alpha[1] = emit[0, 1]
    accum = alpha.clone()
    for t in range(1, T):
        a_prev = torch.cat([torch.full((1,), neg, device=device, dtype=dtype), alpha[:-1]])
        a_skip = torch.cat([torch.full((2,), neg, device=device, dtype=dtype), alpha[:-2]])
        a_skip = torch.where(skip_ok, a_skip, torch.full_like(a_skip, neg))
        alpha = emit[t] + torch.logsumexp(torch.stack([alpha, a_prev, a_skip], 0), 0)
        accum = torch.logaddexp(accum, alpha)
    return accum  # [2S+1]


def ctc_partial_scores(lp: torch.Tensor, target_ids: List[int], blank: int, mode: str = "prefix_diff") -> torch.Tensor:
    """Per-token CTC partial scores ``[S]`` for the given ``mode`` (see module docstring)."""
    assert mode in CTC_PARTIAL_SCORE_MODES, (
        f"invalid per_token_score {mode!r}, expected one of {CTC_PARTIAL_SCORE_MODES}"
    )
    S = len(target_ids)
    if S == 0:
        return lp.new_zeros(0)
    if mode == "prefix_fwd":
        return ctc_prefix_forward_scores(lp, target_ids, blank)
    acc = ctc_accum_states(lp, target_ids, blank)
    idx = torch.arange(S, device=acc.device) * 2 + 1  # label states 2i+1
    label = acc[idx]
    prev_blank = acc[idx - 1]  # 2i
    next_blank = acc[idx + 1]  # 2i+2
    # previous label state 2i-1; for i=0 the "prefix before token 0" is the leading blank acc[0].
    prev_label = acc[(idx - 2).clamp(min=0)]
    prev_label = torch.where(idx - 2 >= 0, prev_label, acc[0].expand_as(prev_label))
    if mode == "raw_partial":
        return label
    if mode == "prefix_diff":
        return label - prev_label
    if mode == "inb_false":
        return label - prev_blank
    if mode == "inb_true":
        return next_blank - prev_blank
    if mode == "inb_both":
        return torch.logaddexp(label, next_blank) - prev_blank
    # inb_both_prev
    return torch.logaddexp(label, next_blank) - torch.logaddexp(prev_blank, prev_label)
