"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/soft_scorer.py (``ViterbiPath``,
``viterbi_blankfree``, ``SegmentPosterior``, ``segment_conditionals`` and their helpers) with
``sae/emc/blankfree_sampler.py`` (``_collapse``, ``_members_of_group``).

The max-plus (Viterbi) pass of the blank-free lattice and the per-segment conditional that re-adds
the path's weight: the two pieces the genmarg posterior decode (``genmarg_steps.generative_decode``)
calls.  Copied verbatim; the soft straight-through scorer term, the derangement and the frozen
scorer of ``soft_scorer`` are not ported (flag-gated model branch removed from the package), nor
is the FFBS sampler of ``blankfree_sampler``.

THE MAX-PLUS FORWARD (source docstring).  The lattice has no Viterbi pass, so
:func:`_forward_step_max` is ``lattice._forward_step``'s ELEMENTWISE branch with
``logsumexp -> amax`` and ``logaddexp -> maximum``: the same lattice, the same arc weights, the
max-plus semiring.  It cannot take the D3 GEMM path (there is no max-plus GEMM), so the two
reductions of a frame are chunked over the symbol axis (history reduction) and over the successor
group axis (band reduction) to bound the materialised tensor -- ``SYMBOL_BLOCK`` / ``GROUP_BLOCK``
are MEMORY knobs and enter no measured quantity.  The forward table is checkpointed exactly as the
manual backward's is (``checkpoint``), and a replayed frame is bit-identical to the frame it
replaces because it runs the same ``_forward_step_max`` on the same context.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch

from ..model.lattice import (
    NEG_INF, LatticeConfig, PriorHistory, arc_weights, band_matrix, forward_ctx, scaled_prior_term,
    bigram_history, scaled_seg_pad,
)

__all__ = ["ViterbiPath", "SegmentPosterior", "viterbi_blankfree", "segment_conditionals",
           "SYMBOL_BLOCK", "GROUP_BLOCK"]

#: Memory chunks of the two max-plus reductions (IMPLEMENTATION; they change no number).  The
#: history reduction materialises ``[B, O, |h|, SYMBOL_BLOCK]`` and the band reduction
#: ``[B, K, GROUP_BLOCK, O, O]`` -- about 0.35 / 0.43 GiB in fp64 at B = 128, W = 25, |h| = 1681.
SYMBOL_BLOCK = 4
GROUP_BLOCK = 4


def _members_of_group(hist: PriorHistory) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(pred [G, P], valid [G, P])``: every history whose emit arcs land in group ``g``.

    ``PriorHistory`` says ``next[h, k] = succ[last(h)] * n_ctx + k``, so the predecessors of a
    destination ``h' = g * n_ctx + k`` are exactly the ``h = outer * n_ctx + l`` with
    ``succ[l] = g``, i.e. ``hist.members[g]`` crossed with the full ``outer`` axis (the same
    inversion ``candidates.py:70-73`` performs).  ``members`` pads with the column ``n_ctx``, which
    is not a history: those entries are flagged invalid instead of being clamped into a real one.
    """
    outer = torch.arange(hist.n_outer, device=hist.members.device)
    pred = hist.members[:, None, :] + outer[None, :, None] * hist.n_ctx  # [G, n_outer, S_max]
    valid = (hist.members[:, None, :] < hist.n_ctx).expand_as(pred).flatten(1)
    return pred.flatten(1).clamp(max=hist.n_hist - 1), valid


def _collapse(emit_phone: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[B, G, T]`` per-frame emits -> ``([B, G, U_max] padded string, [B, G] length)``.

    The blank-free collapsed string IS the sequence of EMITTED tokens in frame order: a repeat arc
    re-emits ``last(h)`` without consuming reverse frames and does not start a token
    (``lattice`` module docstring; the same collapse ``test_blankfree_lattice._oracle`` applies).
    """
    is_emit = emit_phone >= 0
    n_tokens = is_emit.sum(-1)
    u_max = int(n_tokens.max()) if n_tokens.numel() else 0
    b, g, _ = emit_phone.shape
    if u_max == 0:
        return emit_phone.new_full((b, g, 0), -1), n_tokens
    pos = is_emit.cumsum(-1) - 1
    idx = torch.where(is_emit, pos, torch.full_like(pos, u_max))
    buf = emit_phone.new_full((b, g, u_max + 1), -1)
    buf.scatter_(2, idx, emit_phone)
    return buf[:, :, :u_max].contiguous(), n_tokens


class ViterbiPath(NamedTuple):
    """The single max-weight path of the blank-free lattice, per utterance.  All tensors DETACHED."""

    emit_phone: torch.Tensor  # [B, T] long: k of the segment EMITTED at that frame, else -1
    emit_start: torch.Tensor  # [B, T] long: reverse frame s that segment starts at, else -1
    emit_duration: torch.Tensor  # [B, T] long: its duration d, else 0
    emit_hist: torch.Tensor  # [B, T] long: the SOURCE history h_u of that emit, else -1
    emit_flag: torch.Tensor  # [B, T] long: the SOURCE flag f of that emit, else -1
    frame_token: torch.Tensor  # [B, T] long: which emitted token the frame belongs to, -1 past T_b
    phones: torch.Tensor  # [B, U_max] long: the collapsed phone string, -1 padded
    n_tokens: torch.Tensor  # [B] long: its length
    log_w: torch.Tensor  # [B] the path's full log weight = max-plus "log Z" (NEG_INF at Z = 0)
    z_zero: torch.Tensor  # [B] bool: no path at all


def _last_max(x: torch.Tensor, hist: PriorHistory) -> torch.Tensor:
    """``lattice._last_logsumexp`` in the max-plus semiring: ``[B,O,n_ctx,K] -> [B,O,G,K]``."""
    b, o_n, _, k_n = x.shape
    if hist.n_groups == 1:
        return x.amax(dim=2, keepdim=True)
    sel = torch.cat([x, x.new_full((b, o_n, 1, k_n), NEG_INF)], dim=2)
    sel = sel.index_select(2, hist.members.reshape(-1))
    return sel.view(b, o_n, hist.n_groups, -1, k_n).amax(dim=3)


def _emit_source_max(
    fwd: torch.Tensor, prior_term: torch.Tensor, hist: PriorHistory, k_block: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``lattice._emit_context`` + ``_group_logsumexp`` in the max-plus semiring.

    ``(src [B, O, G, K], both [B, O, H])``; ``both = max(f0, f1)`` is the blank transition's source,
    exactly as ``logaddexp(f0, f1)`` is in the sum-product pass.  Chunked over the emitted symbol so
    the ``[B, O, |h|, K]`` intermediate the elementwise reduction needs is never materialised whole.
    """
    b, o_n, h_n, _ = fwd.shape
    k_n = prior_term.shape[-1]
    f0 = fwd[..., 0]
    both = torch.maximum(f0, fwd[..., 1])
    pt = prior_term.view(1, 1, h_n, k_n)
    sn = hist.same_nonsil.view(1, 1, h_n, k_n)
    parts = []
    for k0 in range(0, k_n, max(1, int(k_block))):
        k1 = min(k0 + max(1, int(k_block)), k_n)
        hk = torch.where(sn[..., k0:k1], f0.unsqueeze(-1), both.unsqueeze(-1)) + pt[..., k0:k1]
        x = hk if hist.n_outer == 1 else hk.view(
            b, o_n, hist.n_outer, hist.n_ctx, k1 - k0).amax(dim=2)
        parts.append(_last_max(x, hist))
    return (parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)), both


def _forward_step_max(fwd: torch.Tensor, t: int, cx: Dict) -> torch.Tensor:
    """One frame of the MAX-PLUS forward, ``fwd_t -> fwd_{t+1}``: ``lattice._forward_step``'s
    elementwise branch with ``logsumexp -> amax`` and ``logaddexp -> maximum``.

    The band reduction reuses ``lattice.band_matrix`` (the D3 gather, which is semiring-free: it
    only lays ``seg_pad`` out as ``[B*K, o_src, o_dst]``) and then maxes over ``o_src`` in blocks of
    the successor-group axis.
    """
    b, o_n, h_n, k_n = cx["b"], cx["o_n"], cx["h_n"], cx["k_n"]
    hist, g_n, stride = cx["hist"], cx["g_n"], cx["stride"]
    gwin = cx["seg_pad"][:, t * stride: t * stride + o_n]
    w_sym_t = cx["w_sym"][:, t, :k_n].view(b, 1, 1, k_n)
    src, both = _emit_source_max(fwd, cx["prior_term"], hist, cx["k_block"])
    band = band_matrix(gwin, cx["mm_gather"]).view(b, k_n, o_n, o_n)  # [B, K, o_src, o_dst]
    srcp = src.permute(0, 3, 2, 1)  # [B, K, G, o_src]
    emit_g = fwd.new_full((b, k_n, g_n, o_n), NEG_INF)  # [B, K, G, o_dst]
    gb = max(1, int(cx["g_block"]))
    for g0 in range(0, g_n, gb):
        g1 = min(g0 + gb, g_n)
        cand = srcp[:, :, g0:g1, :, None] + band[:, :, None, :, :]
        emit_g[:, :, g0:g1] = cand.amax(dim=3)
    emit_flat = (emit_g.permute(0, 3, 2, 1) + w_sym_t).reshape(b, o_n, g_n * k_n)
    emit = fwd.new_full((b, o_n, h_n), NEG_INF)
    emit.index_copy_(2, cx["dst_flat"], emit_flat)
    prev = torch.stack([both, fwd[..., 1]], dim=-1)
    nxt = (
        torch.cat([prev[:, stride:], cx["shift_tail"]], dim=1).view(
            b, o_n, cx["n_outer"], cx["n_ctx"], 2)
        + cx["w_pair"][:, t].view(b, 1, 1, cx["n_ctx"], 2)
    ).view(b, o_n, h_n, 2)
    nxt = torch.cat(
        [nxt[..., :1], torch.maximum(nxt[..., 1:], emit.unsqueeze(-1))], dim=-1
    ).clamp(min=NEG_INF)
    return torch.where(cx["active_all"][:, t], nxt, cx["neg"])


@torch.no_grad()
def viterbi_blankfree(
    log_q: torch.Tensor,
    seg_table: torch.Tensor,
    prior_log_bi: torch.Tensor,
    feat_lens: torch.Tensor,
    unit_lens: torch.Tensor,
    cfg: LatticeConfig,
    *,
    temperature: float,
    anchor_weight: float = 0.0,
    prior_weight: float = 1.0,
    log_q_init: Optional[torch.Tensor] = None,
    history: Optional[PriorHistory] = None,
    checkpoint: int = 0,
    seg_pad: Optional[torch.Tensor] = None,
    group_block: int = GROUP_BLOCK,
    symbol_block: int = SYMBOL_BLOCK,
) -> ViterbiPath:
    """The single MAX-WEIGHT path of the blank-free lattice, per utterance.

    The arguments are :func:`lattice.lattice_forward_backward`'s, with the same meaning and the same
    defaults, so a caller passes the training step's own ``dp`` dict: the path is then the mode of
    the posterior the objective is defined by, at that step's tau, anchor and prior weight.

    The forward is checkpointed exactly as the manual backward is (``checkpoint``, the bed's 32) and
    the backward WALK replays the frames between two checkpoints with the same ``_forward_step_max``
    on the same context, so a replayed frame is bit-identical to the frame it replaces.  Ties are
    broken by ``argmax``'s first index; a tie is a different max-weight path of the same weight,
    which is why the arrival state is asserted (the walk must land on the start state) and the
    caller re-adds the path's own weight and compares it with the forward's maximum.
    """
    log_q = log_q.detach()
    seg_table = seg_table.detach()
    prior_log_bi = prior_log_bi.detach()
    if log_q_init is not None:
        log_q_init = log_q_init.detach()
    device = log_q.device
    b, t_max, n_sym = log_q.shape
    assert n_sym == cfg.n_symbols, f"log_q has {n_sym} symbols, cfg wants {cfg.n_symbols}"
    assert cfg.topology == "blankfree", (
        f"this Viterbi is the BLANK-FREE one (topology {cfg.topology!r}); the blank arc is only "
        "carried because lattice's own frame is"
    )
    hist = (history or bigram_history(cfg)).to(device)
    o_n, h_n, k_n, w = cfg.n_offsets, hist.n_hist, cfg.n_phones, cfg.band
    d_cap, stride = cfg.d_cap, cfg.recognizer_stride
    assert prior_log_bi.shape == (h_n, k_n), (tuple(prior_log_bi.shape), hist.name)
    ck = int(checkpoint)
    assert ck >= 0, "checkpoint is a frame stride S >= 0 (0 = keep the whole forward table)"
    if seg_pad is None:
        seg_pad = scaled_seg_pad(seg_table, unit_lens, cfg, temperature)
    cx = forward_ctx(
        log_q, log_q_init, seg_pad, prior_log_bi, feat_lens, cfg, hist,
        temperature=temperature, anchor_weight=anchor_weight, prior_weight=prior_weight,
        use_mm=True,  # for cx["mm_gather"]; the max-plus step never takes a GEMM
    )
    cx["g_block"], cx["k_block"] = int(group_block), int(symbol_block)

    n_store = t_max if ck == 0 else (t_max + ck - 1) // ck
    table = log_q.new_full((b, n_store, o_n, h_n, 2), NEG_INF)
    lens, s_lens = feat_lens.to(device), unit_lens.to(device)
    end_offsets = (s_lens - stride * lens + w).long()
    end_frames, done_all = cx["end_frames"], cx["done_all"]
    batch_index = torch.arange(b, device=device)

    # --- the max-plus forward ---------------------------------------------------------------
    fwd = log_q.new_full((b, o_n, h_n, 2), NEG_INF)
    fwd[:, w, hist.start, 0] = 0.0  # start (t = 0, s = 0, h = (BOS, BOS), f = 0)
    best = log_q.new_full((b,), NEG_INF)
    for t in range(t_max):
        if ck <= 0 or t % ck == 0:
            table[:, t if ck <= 0 else t // ck] = fwd
        fwd = _forward_step_max(fwd, t, cx)
        if (t + 1) in end_frames:
            best = torch.where(
                done_all[:, t], fwd[batch_index, end_offsets].reshape(b, -1).amax(dim=1), best)
    best = best.clamp(min=NEG_INF)
    z_zero = best <= NEG_INF / 2

    # --- constants of the backward walk (blankfree_sampler's, without the G axis) -------------
    prior_t = cx["prior_term"].view(h_n, k_n)
    same_nonsil = hist.same_nonsil.view(h_n, k_n)
    pred_all, pred_valid = _members_of_group(hist)
    w_sym_all = cx["w_sym"][:, :, : hist.n_ctx]
    w_blank_all = cx["w_pair"][:, :, 0, 0]
    d_ax = torch.arange(1, d_cap + 1, device=device)
    two = torch.arange(2, device=device)
    n_seg_rows = seg_pad.shape[1]

    s_pos = torch.zeros((b,), dtype=torch.long, device=device)
    h_state = torch.full((b,), hist.start, dtype=torch.long, device=device)
    flag = torch.zeros((b,), dtype=torch.long, device=device)
    emit_phone = torch.full((b, t_max), -1, dtype=torch.long, device=device)
    emit_start = torch.full((b, t_max), -1, dtype=torch.long, device=device)
    emit_duration = torch.zeros((b, t_max), dtype=torch.long, device=device)
    emit_hist = torch.full((b, t_max), -1, dtype=torch.long, device=device)
    emit_flag = torch.full((b, t_max), -1, dtype=torch.long, device=device)
    frame_active = torch.zeros((b, t_max), dtype=torch.bool, device=device)

    chunk = None if ck == 0 else log_q.new_full((b, ck + 1, o_n, h_n, 2), NEG_INF)
    chunk_id = -1
    tail = _forward_step_max(table[:, t_max - 1], t_max - 1, cx) if ck == 0 else None

    for t in range(t_max - 1, -1, -1):
        if ck == 0:
            fwd = table[:, t]
            fwd_next = table[:, t + 1] if t + 1 < t_max else tail
        else:
            cid = t // ck
            if cid != chunk_id:
                chunk_id = cid
                start, stop = cid * ck, min((cid + 1) * ck, t_max)
                f = table[:, cid]
                for j in range(start, stop):
                    chunk[:, j - start] = f
                    f = _forward_step_max(f, j, cx)
                chunk[:, stop - start] = f
            fwd = chunk[:, t - chunk_id * ck]
            fwd_next = chunk[:, t + 1 - chunk_id * ck]

        if (t + 1) in end_frames:
            starting = (lens == t + 1) & ~z_zero
            term = fwd_next[batch_index, end_offsets].reshape(b, h_n * 2)
            pick = term.argmax(dim=-1)
            h_state = torch.where(starting, pick // 2, h_state)
            flag = torch.where(starting, pick % 2, flag)
            s_pos = torch.where(starting, s_lens, s_pos)

        active = (lens > t) & ~z_zero
        k_last = hist.last[h_state]
        phone = k_last.clamp(max=k_n - 1)
        grp = h_state // hist.n_ctx
        w_sym_t, w_blank_t = w_sym_all[:, t], w_blank_all[:, t]
        w_emit = w_sym_t[batch_index, k_last]

        o_keep = s_pos - stride * t + w
        keep_ok = (o_keep >= 0) & (o_keep < o_n)
        keep = fwd[batch_index, o_keep.clamp(0, o_n - 1), h_state]  # [B, 2]
        blank_log = (keep + w_blank_t.view(b, 1)).masked_fill(
            ~(keep_ok & (flag == 0)).unsqueeze(-1), NEG_INF)
        rep_log = (keep[..., 1] + w_emit).masked_fill(~(keep_ok & (flag == 1)), NEG_INF)

        src_s = s_pos.unsqueeze(-1) - d_ax  # [B, D]
        src_o = src_s - stride * t + w
        emit_ok = (
            ((flag == 1) & (k_last != cfg.bos_id)).unsqueeze(-1)
            & (src_s >= 0) & (src_o >= 0) & (src_o < o_n)
        )
        segment = seg_pad[
            batch_index.unsqueeze(-1), (src_s + w).clamp(0, n_seg_rows - 1),
            (d_ax - 1).view(1, -1), phone.unsqueeze(-1),
        ]  # [B, D]
        pred = pred_all[grp]  # [B, P]
        arc = fwd[batch_index[:, None, None], src_o.clamp(0, o_n - 1).unsqueeze(-1),
                  pred[:, None, :]]  # [B, D, P, 2]
        arc = arc + prior_t[pred, phone.unsqueeze(-1)][:, None, :, None]
        blocked = same_nonsil[pred, phone.unsqueeze(-1)]  # [B, P]
        arc = arc.masked_fill(
            (blocked[:, None, :, None] & (two == 1)) | ~pred_valid[grp][:, None, :, None], NEG_INF)
        emit_log = arc.flatten(2).amax(dim=-1) + segment + w_emit.unsqueeze(-1)
        emit_log = emit_log.masked_fill(~emit_ok | (segment <= NEG_INF / 2), NEG_INF)

        choice = torch.cat([blank_log, rep_log.unsqueeze(-1), emit_log], dim=-1).argmax(dim=-1)
        is_emit = choice >= 3
        is_blank = choice < 2
        di = (choice - 3).clamp(min=0)
        sel = torch.gather(arc, 1, di.view(b, 1, 1, 1).expand(b, 1, arc.shape[2], 2)).reshape(b, -1)
        pick = sel.argmax(dim=-1)
        prev_h = torch.gather(pred, 1, (pick // 2).unsqueeze(-1)).squeeze(-1)
        prev_flag = torch.where(
            is_emit, pick % 2, torch.where(choice == 2, torch.ones_like(choice), choice))

        duration = torch.where(is_emit, d_ax[di], torch.zeros_like(di))
        keep_emit = active & is_emit
        frame_active[:, t] = active
        emit_phone[:, t] = torch.where(keep_emit, phone, torch.full_like(phone, -1))
        emit_start[:, t] = torch.where(keep_emit, s_pos - duration, torch.full_like(phone, -1))
        emit_duration[:, t] = torch.where(keep_emit, duration, torch.zeros_like(duration))
        emit_hist[:, t] = torch.where(keep_emit, prev_h, torch.full_like(prev_h, -1))
        emit_flag[:, t] = torch.where(keep_emit, prev_flag, torch.full_like(prev_flag, -1))

        h_state = torch.where(active & is_emit, prev_h, h_state)
        s_pos = torch.where(active, s_pos - duration, s_pos)
        flag = torch.where(active, prev_flag, flag)

    live = (~z_zero) & (lens > 0)
    if bool(live.any()):
        # The walk terminates at the lattice's own start state by construction (the only finite
        # entry of fwd_0 is (o = W, h = start, f = 0)); asserting it catches an index or a stride
        # error that would otherwise show up only as a wrong path weight.
        assert bool((s_pos[live] == 0).all()), "the Viterbi walk did not consume the reverse frames"
        assert bool((h_state[live] == hist.start).all()), "the Viterbi walk left the start history"
        assert bool((flag[live] == 0).all()), "the Viterbi walk did not end on the start flag"

    phones, n_tokens = _collapse(emit_phone.unsqueeze(1))
    is_emit_all = emit_phone >= 0
    frame_token = torch.where(
        frame_active, is_emit_all.cumsum(dim=1) - 1, torch.full_like(emit_phone, -1))
    return ViterbiPath(
        emit_phone=emit_phone, emit_start=emit_start, emit_duration=emit_duration,
        emit_hist=emit_hist, emit_flag=emit_flag, frame_token=frame_token,
        phones=phones.squeeze(1), n_tokens=n_tokens.squeeze(1), log_w=best, z_zero=z_zero,
    )


def _pack(x: torch.Tensor, is_emit: torch.Tensor, u_max: int, fill: int) -> torch.Tensor:
    """``[B, T]`` per-frame values at the emit frames -> ``[B, U_max]`` per TOKEN, ``fill`` padded.

    The index expression is ``blankfree_sampler._collapse``'s, so every per-token array of this
    module is in the collapsed string's own order and the trash column absorbs the non-emit frames.
    """
    b, _ = x.shape
    pos = is_emit.cumsum(dim=1) - 1
    idx = torch.where(is_emit, pos, torch.full_like(pos, u_max))
    buf = x.new_full((b, u_max + 1), fill)
    buf.scatter_(1, idx, x)
    return buf[:, :u_max].contiguous()


class SegmentPosterior(NamedTuple):
    """The per-segment conditional over the 40 symbols and the checks that it is the right one."""

    log_f: torch.Tensor  # [B, U, K] the conditional's unnormalised log weight (NEG_INF = illegal)
    p: torch.Tensor  # [B, U, K] softmax of log_f -- the conditional posterior
    ids: torch.Tensor  # [B, U] long: the Viterbi symbol of each segment, -1 padded
    valid: torch.Tensor  # [B, U] bool: a real token of this utterance
    path_log_w: torch.Tensor  # [B] the path weight re-added from the per-token terms
    mismatch: torch.Tensor  # [] long: segments whose argmax is not the Viterbi symbol
    max_gain: torch.Tensor  # [] the largest such excess, in nats of the scaled conditional


def segment_conditionals(
    path: ViterbiPath,
    log_q: torch.Tensor,
    seg_table: torch.Tensor,
    prior_log_bi: torch.Tensor,
    cfg: LatticeConfig,
    *,
    temperature: float,
    anchor_weight: float = 0.0,
    prior_weight: float = 1.0,
    log_q_init: Optional[torch.Tensor] = None,
    history: Optional[PriorHistory] = None,
) -> SegmentPosterior:
    """The lattice's conditional posterior of each Viterbi segment's symbol, the neighbours fixed.

    For token ``u`` of the Viterbi path, substituting its symbol ``k_u`` by ``c`` while keeping the
    SEGMENTATION and every other symbol changes exactly these terms of the path's log weight::

      sum_{t in frames(u)} w_sym[b, t, c]          the recognizer gathers of its emit AND repeats
      + (1/tau) G[c, d_u - 1, s_u]                 its own segment score (illegal d -> NEG_INF)
      + prior_term[h_u, c]                         its own prior term
      + prior_term[next[h_u, c], k_{u+1}]          the NEXT token's prior term, whose history moved
      + prior_term[next[next[h_u, c], k_{u+1}], k_{u+2}]      and the one after it

    -- three prior terms, because a history keeps ``last`` and one ``outer`` symbol
    (:class:`lattice.PriorHistory`; at the trigram the fourth successor is already free of ``c``,
    and the expression above is written through ``hist.next`` so it holds for any history this
    module is given).  The repeat rule applies at each of the three arcs
    (``same_nonsil`` at the arc's own SOURCE history, forbidden only out of ``f = 1``), which is
    what forbids ``c = k_{u+1}`` for a non-SIL neighbour, and the duration legality of
    ``scaled_seg_pad`` applies to the substituted symbol (a 30-frame SIL segment admits no other
    symbol, so its conditional is a point mass -- as it should be).

    Everything is DIFFERENTIABLE in ``log_q`` and ``seg_table``; the path (the segmentation and the
    neighbours) is not, which is the disclosed straight-through approximation.  ``p``'s argmax is
    the Viterbi symbol because the joint maximum is a conditional maximum: the caller asserts it.
    """
    device = log_q.device
    b, t_max, _ = log_q.shape
    hist = (history or bigram_history(cfg)).to(device)
    k_n, h_n = cfg.n_phones, hist.n_hist
    u_max = int(path.phones.shape[1])
    scale = 1.0 / float(temperature)
    ids = path.phones
    valid = ids >= 0
    ids_c = ids.clamp(min=0)
    if u_max == 0:
        zero = log_q.new_zeros(())
        return SegmentPosterior(
            log_q.new_zeros((b, 0, k_n)), log_q.new_zeros((b, 0, k_n)), ids, valid,
            log_q.new_zeros((b,)), torch.zeros((), dtype=torch.long, device=device), zero)

    # --- the recognizer weights of the frames of each token (emit + its repeats) --------------
    _, w_sym = arc_weights(log_q, log_q_init, temperature, anchor_weight, cfg)  # [B, T, n_ctx]
    ft = path.frame_token
    frame_ok = ft >= 0
    contrib = torch.where(frame_ok.unsqueeze(-1), w_sym[:, :, :k_n], w_sym.new_zeros(()))
    rec = w_sym.new_zeros((b, u_max, k_n)).scatter_add(
        1, ft.clamp(min=0).unsqueeze(-1).expand(b, t_max, k_n), contrib)

    # --- the segment score of the token's (s_u, d_u) at every symbol --------------------------
    is_emit = path.emit_phone >= 0
    dur = _pack(path.emit_duration, is_emit, u_max, 0)
    start = _pack(path.emit_start, is_emit, u_max, 0)
    h_u = _pack(path.emit_hist, is_emit, u_max, 0).clamp(min=0)
    f_u = _pack(path.emit_flag, is_emit, u_max, 0).clamp(min=0)
    d_cap, s1 = seg_table.shape[2], seg_table.shape[3]
    d1 = (dur - 1).clamp(min=0, max=d_cap - 1)
    s0 = start.clamp(min=0, max=s1 - 1)
    flat = seg_table.reshape(b, -1)
    gidx = (torch.arange(k_n, device=device).view(1, 1, -1) * d_cap + d1.unsqueeze(-1)) * s1 \
        + s0.unsqueeze(-1)
    seg = scale * torch.gather(flat, 1, gidx.reshape(b, u_max * k_n)).view(b, u_max, k_n)
    d_ax = torch.arange(1, d_cap + 1, device=device)
    d_max_k = torch.tensor([cfg.type_d_max(k) for k in range(k_n)], device=device)
    dur_ok = (d_ax.view(1, -1) >= cfg.d_min) & (d_ax.view(1, -1) <= d_max_k.view(-1, 1))  # [K, D]
    seg_ok = dur_ok.t()[d1]  # [B, U, K]

    # --- the three prior terms the substitution moves -----------------------------------------
    prior_term = scaled_prior_term(prior_log_bi.to(log_q.dtype), scale, prior_weight)  # [H, K]
    p_flat = prior_term.reshape(-1)
    n_flat = hist.next.reshape(-1)
    s_flat = hist.same_nonsil.view(h_n, k_n).reshape(-1)
    nxt1 = hist.next[h_u]  # [B, U, K]: h_{u+1}(c)
    k1 = torch.cat([ids[:, 1:], ids.new_full((b, 1), -1)], dim=1)
    k2 = torch.cat([ids[:, 2:], ids.new_full((b, min(2, u_max)), -1)], dim=1)
    has1, has2 = (k1 >= 0) & valid, (k2 >= 0) & valid
    f1 = torch.cat([f_u[:, 1:], f_u.new_zeros((b, 1))], dim=1)
    f2 = torch.cat([f_u[:, 2:], f_u.new_zeros((b, min(2, u_max)))], dim=1)
    idx1 = nxt1 * k_n + k1.clamp(min=0).unsqueeze(-1)
    p2 = torch.where(has1.unsqueeze(-1), p_flat[idx1], p_flat.new_zeros(()))
    nxt2 = n_flat[idx1]
    idx2 = nxt2 * k_n + k2.clamp(min=0).unsqueeze(-1)
    p3 = torch.where(has2.unsqueeze(-1), p_flat[idx2], p_flat.new_zeros(()))
    blocked = (
        (hist.same_nonsil.view(h_n, k_n)[h_u] & (f_u == 1).unsqueeze(-1))
        | (s_flat[idx1] & ((f1 == 1) & has1).unsqueeze(-1))
        | (s_flat[idx2] & ((f2 == 1) & has2).unsqueeze(-1))
    )

    log_f = (rec + seg + prior_term[h_u] + p2 + p3).masked_fill(blocked | ~seg_ok, NEG_INF)
    p = torch.softmax(log_f, dim=-1)

    take = lambda x: torch.gather(x, 2, ids_c.unsqueeze(-1)).squeeze(-1)
    own = take(rec) + take(seg) + take(prior_term[h_u])
    path_log_w = (own * valid.to(own.dtype)).sum(dim=1)
    best_c = log_f.argmax(dim=-1)
    gain = log_f.gather(2, best_c.unsqueeze(-1)).squeeze(-1) - take(log_f)
    mismatch = ((best_c != ids_c) & valid).sum()
    max_gain = torch.where(valid, gain, gain.new_zeros(())).max()
    return SegmentPosterior(log_f, p, ids, valid, path_log_w, mismatch, max_gain.detach())

