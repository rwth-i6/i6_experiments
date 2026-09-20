"""SAE 4a step 4 -- how well the jointly trained reverse model phi of ONE GAN arm predicts the units.

Runs under the ``w2vu`` env python, GPU, spawned by ``rev_diag.W2vu2RevDiagJob``.  It is a DIAGNOSTIC
readout: it measures, it draws NO conclusion, and it defines no gate.  Every quantity below is
pre-registered here, with the code that produces it (memory: preregistration-lives-with-the-code).

THE TERM (identical to the training term and to ``rev_dev.py``, ``w2vu_rev.rev_term``):
tau = 1, beta = 0 (no prior factor), blank-free topology, d_min = 2, D = 25 / D_sil = 50, band 25,
recognizer stride 3 (T = ceil(S/3)), the enc50 K = 500 unit stream on the retained (VAD-masked)
50 Hz clock, the frozen per-utterance PCA-16 eta.  The generator runs in EVAL mode under no_grad,
on the pre-segmentation stride-3 logits restricted to the 40 phone columns and re-normalized
(``rev_term.log_q_from_dense``).  ALL dev utterances of both splits are read; nothing is skipped
except where a count is reported.

WHAT IS REPORTED, PER SPLIT.  Every headline is CORPUS level (sum over utterances / sum of frames);
the per-utterance mean of the same per-utterance quantity is reported beside it as ``mean_utt_*``.

(a) ``own`` -- the recognizer's own posterior:
    * ``logz_per_frame`` = sum_utt log Z_tau / sum_utt S.  HIGHER is better.  This is ``rev_dev.py``'s
      headline and it is asserted against the arm's registered ``rev_dev.json`` value (``--expect``).
    * ``viterbi`` -- the MAX-semiring path of the SAME lattice (same arcs, same weights, max instead
      of logsumexp), backtraced to a tiling of the S reverse frames by segments (token k, duration
      d, start s).  ``top1_acc`` is the per-frame top-1 unit accuracy of phi ALONG THAT PATH: the
      frame's prediction is ``argmax_u nu_phi(k, bucket(d), bucket(r), eta)[u]``, i.e. the emission
      the path itself uses at that frame (token identity, duration bucket, within-segment position
      bucket and the frozen eta), compared with the observed unit z_s.  Corpus level = correct
      frames / frames.
    * ``greedy`` -- the SAME accuracy along the greedy phone path instead: ``k_t = argmax_k log q_t``
      per recognizer frame, MIN-DURATION IGNORED, maximal runs of equal k expanded onto the unit
      clock by the stride-3 rule (recognizer frame t covers unit frames 3t .. 3t+2, clipped at S),
      so a run of L recognizer frames is ONE segment of d = 3L unit frames.  A greedy segment may be
      shorter than d_min or longer than D_k; the duration/position buckets are still those of
      ``reverse.duration_bucket`` / ``reverse.position_bounds``, which are defined for every d >= 1.
      The greedy path carries no lattice score, so only the accuracy is reported for it.
(b) ``deranged`` -- the destroyed-structure control: the same two readouts with the recognizer
    posterior DERANGED inside each utterance, i.e. ``log_q[t] <- log_q[pi(t)]`` for a permutation pi
    of the valid T frames with NO FIXED POINT.  pi is drawn from ``RandomState(seed ^ crc32(tag))``
    (``blankfree_permute.utterance_permutation``'s convention: seed and tag alone, no global RNG) and
    REDRAWN until it has no fixed point, which is the "never its own" convention of the derangement
    the cold bed's gap uses (``s1a_job.build_derangement``, via ``blankfree_eval_jobs``).  The units,
    eta and the lengths are untouched; only the recognizer's frame order is destroyed.  An utterance
    with T < 2 has no derangement and is counted in ``no_derangement``.
(c) ``gold`` -- phi WITHOUT the recognizer: the reverse model's likelihood of the observed units
    given the GOLD phone sequence of ``GoldPhonesJob``'s ``gold.json``, taken exactly as that file
    has it (it carries NO SIL token: 39 ARPAbet types).  The gold sequence is force-aligned to the S
    reverse frames under phi's own segment scores with the same min-duration rule
    (``reverse.SegmentalReverseModel``: d_min <= d <= D_k, the segments tile the S frames exactly):
    * ``forward_logp_per_frame`` -- the exact forward SUM over segmentations,
      ``reverse.SegmentalReverseModel.forward_logsum`` (the constrained/forced forward the reverse
      code already exposes), per frame;
    * ``viterbi_logp_per_frame`` -- the MAX over segmentations of the same DP, per frame;
    * ``top1_acc`` -- the per-frame top-1 unit accuracy along that Viterbi segmentation, the same
      emission convention as (a).
    A gold pair that cannot be aligned at all (``reverse.feasible``: U * d_min > S or S > sum_i D_k)
    is SKIPPED and counted in ``skipped_infeasible``; it is never averaged in (memory:
    impossible-scores-poison-means).  Both gold numbers are fp32, the reverse model's own log-space
    discipline; the lattice numbers of (a)/(b) are fp64, the training term's (``RevTermConfig``).
(d) ``floors`` -- the reference points for the accuracies and the per-frame log-likelihoods:
    * ``majority_unit_acc`` -- predict the single most frequent unit of the split, IN-SAMPLE (the
      unit is chosen on the same frames it is scored on);
    * ``unigram_logp_per_frame`` -- the unit unigram fitted on the split's own units, IN-SAMPLE (it
      is the negative empirical entropy of the split's unit distribution, an upper bound on any
      held-out unigram);
    * ``uniform_logp_per_frame`` = log(1 / 500).
    The floors are per-frame log-likelihoods of the UNIT STREAM alone.  They are NOT comparable with
    ``logz_per_frame`` of (a)/(b), which is a lattice marginal over (path, tokens, segmentation)
    carrying the recognizer's own term as well; they are the floor of the GOLD number of (c) and of
    nothing else (memory: magnitudes-are-per-arm).
(e) degeneracy: for every best path -- (a) own, (a) greedy, (b) and (c) -- the number of DISTINCT
    phone states used on it (over the whole split) and the top-10 share of the PREDICTED units (the
    share of frames whose predicted unit is one of the 10 most frequent predicted units).  The same
    share over the OBSERVED units of the same frames is reported beside it as the reference.

SELF-CHECKS (fatal): the own ``logz_per_frame`` reproduces ``--expect`` to ``--expect-tol``; every
backtraced path is RE-SCORED arc by arc against its Viterbi score (``max_rescore_diff``); every
Viterbi score is <= the forward log Z of the same utterance; every path tiles [0, S) exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import Counter

import numpy as np
import torch

from speech_llm.sae.emc.lattice import build_segment_table, scaled_seg_pad
from speech_llm.sae.emc.prior import PHONE2ID, PHONES
from speech_llm.sae.emc.reverse import duration_bucket, feasible, position_bounds
from speech_llm.sae.psi_align import NEG_INF

TOP_N_UNITS = 10  # (e): the share of the 10 most frequent predicted units


# ---------------------------------------------------------------------------------------------------
# (b) the derangement of the recognizer frame axis
# ---------------------------------------------------------------------------------------------------


def frame_derangement(tag, length, *, seed):
    """A permutation of ``length`` frames with NO FIXED POINT, from ``seed`` and ``tag`` alone.

    ``RandomState(seed ^ crc32(tag))`` is ``blankfree_permute.utterance_permutation``'s convention
    (the same permutation every time the utterance is seen, drawn from no global RNG); the draw is
    REPEATED until the permutation has no fixed point, which is the "never its own" rule of
    ``s1a_job.build_derangement``.  Rejection from the uniform permutations is uniform over the
    derangements.  ``length < 2`` has no derangement and returns ``None``.
    """
    n = int(length)
    assert n >= 0, n
    if n < 2:
        return None
    state = np.random.RandomState((int(seed) ^ zlib.crc32(tag.encode("utf-8"))) & 0xFFFFFFFF)
    ax = np.arange(n)
    for _ in range(1000):
        perm = state.permutation(n)
        if not bool((perm == ax).any()):
            return perm.astype(np.int64)
    raise RuntimeError(f"no derangement drawn for {tag!r} (n = {n})")


# ---------------------------------------------------------------------------------------------------
# segments -> per-frame predicted units (the emission convention of (a), (b) and (c))
# ---------------------------------------------------------------------------------------------------


def top1_unit_table(phi, eta):
    """``[B, n_types, n_dur_buckets, n_pos_buckets]`` = ``argmax_u nu_phi(k, c, j, eta)[u]``."""
    return phi.emission_log_probs(eta).argmax(dim=-1)


def predicted_units(segments, s_len, top1_b, cfg_rev):
    """Per reverse frame, the unit phi predicts on a path given as ``[(s0, d, k, t_emit)]``."""
    pred = np.full(int(s_len), -1, dtype=np.int64)
    for s0, d, k, _t in segments:
        c = duration_bucket(int(d), cfg_rev)
        for j, (lo, hi) in enumerate(position_bounds(int(d), cfg_rev.n_pos_buckets)):
            if hi > lo:
                pred[s0 + lo: s0 + hi] = int(top1_b[int(k), c, j])
    assert bool((pred >= 0).all()), "the path does not cover every reverse frame"
    return pred


def assert_tiling(segments, s_len):
    """The segments of a path must tile ``[0, S)`` exactly, in order."""
    s = 0
    for s0, d, _k, _t in sorted(segments):
        assert s0 == s, (s0, s)
        s += int(d)
    assert s == int(s_len), (s, s_len)


def greedy_segments(labels_t, s_len, stride):
    """The greedy phone path as segments on the unit clock (min-duration ignored; see the docstring)."""
    segs = []
    t_len = len(labels_t)
    start = 0
    for t in range(1, t_len + 1):
        if t == t_len or int(labels_t[t]) != int(labels_t[start]):
            s0 = start * stride
            s1 = min(t * stride, int(s_len))
            if s1 > s0:
                segs.append((s0, s1 - s0, int(labels_t[start]), start))
            start = t
    return segs


# ---------------------------------------------------------------------------------------------------
# (a)/(b) the lattice, in the MAX semiring, with a backtrace
# ---------------------------------------------------------------------------------------------------


def lattice_best_path(
    *, log_q, seg_pad, feat_lens, unit_lens, cfg, hist, temperature, mode="max", paths=True
):
    """The blank-free product lattice of ``emc.lattice`` with ``max`` (or ``logsumexp``) per frame.

    Same states ``(t, s, h, f)``, same three arcs and the same weights as
    ``lattice.forward_log_z`` -- ``mode="sum"`` reproduces its ``log_z`` exactly and is what
    ``test_rev_diag.py`` checks this recursion against; ``mode="max"`` is the Viterbi score, and its
    backtrace returns the winning path as ``[(s_start, d, k, t_emit)]`` plus the repeat arcs' count.

    Only the BIGRAM history is supported (``h' = k``), which is the history the ported term runs
    (``rev_term.reverse_term``: ``bigram_history``); a larger history is rejected rather than
    silently mis-indexed.
    """
    assert hist.n_outer == 1, "lattice_best_path implements the bigram history only"
    assert cfg.topology == "blankfree", cfg.topology
    device = log_q.device
    b, t_max, n_sym = log_q.shape
    assert n_sym == cfg.n_symbols, (n_sym, cfg.n_symbols)
    o_n, h_n, k_n, w, stride = cfg.n_offsets, hist.n_hist, cfg.n_phones, cfg.band, cfg.recognizer_stride
    hist = hist.to(device)
    scale = 1.0 / float(temperature)
    dtype = log_q.dtype
    neg = torch.full((), NEG_INF, device=device, dtype=dtype)

    w_sym = scale * log_q                                          # [B, T, K]
    w_rep = torch.cat([w_sym, w_sym.new_full((b, t_max, 1), NEG_INF)], dim=-1)  # [B, T, n_ctx]
    o_src_ax = torch.arange(o_n, device=device).view(-1, 1)
    o_dst_ax = torch.arange(o_n, device=device).view(1, -1)
    d_gather = (o_dst_ax - o_src_ax + stride - 1).clamp(min=0)     # [O_src, O_dst] -> d - 1
    d_gather = d_gather.view(1, o_n, o_n, 1).expand(b, o_n, o_n, k_n)
    same = hist.same_nonsil.view(1, 1, h_n, k_n)
    lens = feat_lens.to(device)
    active = (torch.arange(t_max, device=device).view(1, -1) < lens.view(-1, 1))
    end_off = (unit_lens.to(device) - stride * lens + w).long()
    assert bool(((end_off >= 0) & (end_off < o_n)).all()), end_off

    fwd = log_q.new_full((b, o_n, h_n, 2), NEG_INF)
    fwd[:, w, hist.start, 0] = 0.0
    score = log_q.new_full((b,), NEG_INF)
    h_end = torch.zeros(b, dtype=torch.long, device=device)
    f_end = torch.zeros(b, dtype=torch.long, device=device)
    bp = torch.full((t_max, b, o_n, h_n), -1, dtype=torch.int32, device=device) if paths else None
    rows = torch.arange(b, device=device)

    for t in range(t_max):
        gwin = seg_pad[:, t * stride: t * stride + o_n]             # [B, O_src, D_pad, K]
        g_band = torch.gather(gwin, 2, d_gather)                    # [B, O_src, O_dst, K]
        f0, f1 = fwd[..., 0], fwd[..., 1]
        if mode == "max":
            both = torch.maximum(f0, f1)
        else:
            both = torch.logaddexp(f0, f1)
        hk = torch.where(same, f0.unsqueeze(-1), both.unsqueeze(-1))  # [B, O, H, K]; beta = 0
        if mode == "max":
            src, h_arg = hk.max(dim=2)                              # [B, O, K]
            cand = src.unsqueeze(2) + g_band                        # [B, O_src, O_dst, K]
            emit_v, o_arg = cand.max(dim=1)                         # [B, O_dst, K]
        else:
            src = torch.logsumexp(hk, dim=2)
            emit_v = torch.logsumexp(src.unsqueeze(2) + g_band, dim=1)
            h_arg = o_arg = None
        emit_v = emit_v + w_sym[:, t].view(b, 1, k_n)
        emit = fwd.new_full((b, o_n, h_n), NEG_INF)
        emit.index_copy_(2, hist.dst.reshape(-1), emit_v)           # h' = k (bigram)

        prev = torch.stack([both, f1], dim=-1)                      # (blank src, repeat src)
        shifted = torch.cat([prev[:, stride:], prev.new_full((b, stride, h_n, 2), NEG_INF)], dim=1)
        w_pair = torch.stack(
            [w_rep.new_full((b, hist.n_ctx), NEG_INF), w_rep[:, t]], dim=-1
        ).view(b, 1, hist.n_ctx, 2)                                  # blank is impossible (blank-free)
        nxt = (shifted.view(b, o_n, hist.n_ctx, 2) + w_pair).view(b, o_n, h_n, 2)
        if mode == "max":
            take_emit = emit > nxt[..., 1]
            nxt = torch.cat(
                [nxt[..., :1], torch.maximum(nxt[..., 1:], emit.unsqueeze(-1))], dim=-1
            ).clamp(min=NEG_INF)
        else:
            nxt = torch.cat(
                [nxt[..., :1], torch.logaddexp(nxt[..., 1:], emit.unsqueeze(-1))], dim=-1
            ).clamp(min=NEG_INF)
        if paths and mode == "max":
            # the emit arc's source state, encoded as ((o_src * H + h_src) * 2 + f_src); -1 = repeat
            h_src = torch.gather(h_arg, 1, o_arg)                   # [B, O_dst, K]
            fsel = (~same) & (f1 > f0).unsqueeze(-1)                # [B, O, H, K] which f won
            fsel_h = torch.gather(fsel, 2, h_arg.unsqueeze(2)).squeeze(2).long()
            f_src = torch.gather(fsel_h, 1, o_arg)
            code = ((o_arg * h_n + h_src) * 2 + f_src).to(torch.int32)
            code_h = torch.full((b, o_n, h_n), -1, dtype=torch.int32, device=device)
            code_h.index_copy_(2, hist.dst.reshape(-1), code)
            bp[t] = torch.where(take_emit, code_h, torch.full_like(code_h, -1))
        fwd = torch.where(active[:, t].view(b, 1, 1, 1), nxt, neg)

        done = lens == (t + 1)
        if bool(done.any()):
            final = fwd[rows, end_off]                               # [B, H, 2]
            flat = final.reshape(b, -1)
            if mode == "max":
                best, arg = flat.max(dim=1)
            else:
                best, arg = torch.logsumexp(flat, dim=1), torch.zeros(b, dtype=torch.long, device=device)
            score = torch.where(done, best, score)
            h_end = torch.where(done, torch.div(arg, 2, rounding_mode="floor"), h_end)
            f_end = torch.where(done, arg % 2, f_end)

    out = {"score": score.clamp(min=NEG_INF)}
    if not (paths and mode == "max"):
        return out
    out.update(_backtrace(bp.cpu().numpy(), score.detach().cpu().numpy(),
                          h_end.cpu().numpy(), f_end.cpu().numpy(),
                          feat_lens.cpu().numpy(), unit_lens.cpu().numpy(),
                          cfg, hist.n_hist, hist.n_ctx, int(hist.start)))
    return out


def _backtrace(bp, score, h_end, f_end, feat_lens, unit_lens, cfg, h_n, n_ctx, start):
    """Walk the backpointers of :func:`lattice_best_path` back to the start state, per utterance.

    -> ``paths`` ``[(s_start, d, k, t_emit)]`` of the EMIT arcs and ``repeats`` ``[(t, k)]`` of the
    repeat arcs; together they are every arc of the winning path, which is what re-scores it.
    """
    w, stride = cfg.band, cfg.recognizer_stride
    paths, repeats, ok = [], [], []
    for i in range(len(feat_lens)):
        t_len, s_len = int(feat_lens[i]), int(unit_lens[i])
        if not np.isfinite(score[i]) or score[i] <= NEG_INF / 2:
            paths.append(None); repeats.append(None); ok.append(False); continue
        o = int(s_len - stride * t_len + w)
        h, f = int(h_end[i]), int(f_end[i])
        segs, reps = [], []
        for t in range(t_len - 1, -1, -1):
            code = int(bp[t, i, o, h])
            assert f == 1, f"a blank arc on a blank-free path (t = {t}, f = {f})"
            if code < 0:
                reps.append((t, h % n_ctx))          # repeat: re-emits last(h), s unchanged
                o += stride
                continue
            o_src, rem = divmod(code, h_n * 2)
            h_src, f_src = divmod(rem, 2)
            segs.append((o_src + stride * t - w, o - o_src + stride, h % n_ctx, t))
            o, h, f = o_src, h_src, f_src
        assert (o, h, f) == (w, start, 0), ((o, h, f), (w, start, 0))
        paths.append(sorted(segs)); repeats.append(sorted(reps)); ok.append(True)
    return {"paths": paths, "repeats": repeats, "ok": np.asarray(ok)}


def rescore_path(segments, repeats, log_q_b, seg_pad_b, cfg, temperature):
    """Sum the arc weights of a backtraced path -- the check that it IS the DP's own best path.

    ``seg_pad_b`` is this utterance's :func:`lattice.scaled_seg_pad` table, so the reverse term is
    read exactly where the DP read it (row ``s + W``, column ``d - 1``).
    """
    scale = 1.0 / float(temperature)
    total = 0.0
    for s0, d, k, t in segments:
        total += float(scale * log_q_b[t, k]) + float(seg_pad_b[s0 + cfg.band, d - 1, k])
    for t, k in repeats:
        total += float(scale * log_q_b[t, k])
    return total


# ---------------------------------------------------------------------------------------------------
# (c) the gold forced alignment
# ---------------------------------------------------------------------------------------------------


def gold_forced_align(phi, z, y, eta, s_lens, u_lens):
    """Forward log-sum AND Viterbi of ``p_phi(z | y, eta)`` over segmentations with d_min <= d <= D_k.

    The forward is ``reverse.SegmentalReverseModel.forward_logsum`` (the forced forward the reverse
    code exposes); the Viterbi is the SAME recursion with ``max`` instead of ``logsumexp``, with a
    backtrace of the winning segmentation.  fp32, the reverse model's own log space.
    """
    cfg = phi.cfg
    device = z.device
    b, s_max = z.shape
    u_max = y.shape[1]
    seg = build_segment_table(phi, z, eta)                          # [B, K, D, S+1] with log p(d|k)
    forward = phi.forward_logsum(z, y, eta, s_lens, u_lens)

    s_ax = torch.arange(s_max + 1, device=device).view(1, -1)
    d_ax = torch.arange(cfg.d_cap, device=device).view(-1, 1) + 1
    shift = (s_ax - d_ax + cfg.d_cap).clamp(min=0).view(1, cfg.d_cap, s_max + 1).expand(b, -1, -1)
    rows = torch.arange(b, device=device)
    alpha = torch.full((b, s_max + 1), NEG_INF, device=device, dtype=torch.float32)
    alpha[:, 0] = 0.0
    bp = torch.zeros((u_max, b, s_max + 1), dtype=torch.int16, device=device)
    scores = []
    for i in range(u_max):
        base = alpha.unsqueeze(1) + seg[rows, y[:, i]]               # [B, D, S+1]
        padded = torch.cat([base.new_full((b, cfg.d_cap, cfg.d_cap), NEG_INF), base], dim=2)
        shifted = torch.gather(padded, 2, shift)
        alpha, arg = shifted.max(dim=1)
        alpha = alpha.clamp(min=NEG_INF)
        bp[i] = arg.to(torch.int16)
        scores.append(alpha)
    table = torch.stack(scores, dim=1) if u_max else None            # [B, U, S+1]
    viterbi = torch.full((b,), NEG_INF, device=device, dtype=torch.float32)
    if u_max:
        viterbi = table[rows, u_lens - 1, s_lens]
    bp_np = bp.cpu().numpy()
    paths = []
    for i in range(b):
        s, u = int(s_lens[i]), int(u_lens[i])
        if not np.isfinite(float(viterbi[i])) or float(viterbi[i]) <= NEG_INF / 2:
            paths.append(None)
            continue
        segs = []
        for j in range(u - 1, -1, -1):
            d = int(bp_np[j, i, s]) + 1
            s -= d
            segs.append((s, d, int(y[i, j]), j))
        assert s == 0, s
        paths.append(sorted(segs))
    return forward, viterbi, paths


# ---------------------------------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------------------------------


class PathAcc(object):
    """Per-frame accuracy, path score and degeneracy counts of one readout, accumulated per split."""

    def __init__(self):
        self.frames = 0
        self.correct = 0
        self.utts = 0
        self.utt_acc = 0.0
        self.score = 0.0
        self.utt_score = 0.0
        self.tokens = 0
        self.repeats = 0
        self.phones = Counter()
        self.pred_units = Counter()
        self.obs_units = Counter()
        self.max_rescore_diff = None     # set only where a path is re-scored (the lattice paths)

    def add(self, *, pred, obs, segments, score=None, repeats=None, rescore_diff=None):
        n = len(obs)
        hit = int((pred == obs).sum())
        self.frames += n
        self.correct += hit
        self.utts += 1
        self.utt_acc += hit / n
        self.tokens += len(segments)
        self.repeats += len(repeats or [])
        for _s0, _d, k, _t in segments:
            self.phones[int(k)] += 1
        self.pred_units.update(Counter(pred.tolist()))
        self.obs_units.update(Counter(obs.tolist()))
        if score is not None:
            self.score += float(score)
            self.utt_score += float(score) / n
        if rescore_diff is not None:
            self.max_rescore_diff = max(self.max_rescore_diff or 0.0, float(rescore_diff))

    def report(self, *, with_score=True):
        if not self.utts:
            return None
        top = sum(c for _u, c in self.pred_units.most_common(TOP_N_UNITS))
        top_obs = sum(c for _u, c in self.obs_units.most_common(TOP_N_UNITS))
        rec = {
            "utts": self.utts,
            "frames": self.frames,
            "top1_acc": self.correct / self.frames,
            "mean_utt_top1_acc": self.utt_acc / self.utts,
            "tokens": self.tokens,
            "tokens_per_frame": self.tokens / self.frames,
            "repeat_arcs": self.repeats,
            "distinct_phone_states": len(self.phones),
            "distinct_predicted_units": len(self.pred_units),
            "top10_predicted_unit_share": top / self.frames,
            "top10_observed_unit_share": top_obs / self.frames,
            "top_predicted_units": [[int(u), int(c)] for u, c in self.pred_units.most_common(TOP_N_UNITS)],
            "phone_state_counts": {PHONES[k]: int(c) for k, c in sorted(self.phones.items())},
        }
        if with_score:
            rec["score_per_frame"] = self.score / self.frames
            rec["mean_utt_score_per_frame"] = self.utt_score / self.utts
            if self.max_rescore_diff is not None:
                rec["max_rescore_diff"] = self.max_rescore_diff
        return rec


class LogZAcc(object):
    """``rev_dev.py``'s convention, verbatim: sum log Z / sum S, and the mean of log Z / S."""

    def __init__(self):
        self.frames = 0
        self.logz = 0.0
        self.utt = 0.0
        self.kept = 0
        self.z_zero = 0

    def add(self, logz_per_frame, n_frames, keep):
        if not keep:
            self.z_zero += 1
            return
        self.kept += 1
        self.frames += int(n_frames)
        self.logz += float(logz_per_frame) * int(n_frames)
        self.utt += float(logz_per_frame)

    def report(self):
        return {
            "kept": self.kept,
            "z_zero": self.z_zero,
            "frames": self.frames,
            "logz_per_frame": self.logz / self.frames if self.frames else None,
            "mean_utt_logz_per_frame": self.utt / self.kept if self.kept else None,
        }


# ---------------------------------------------------------------------------------------------------
# the readout
# ---------------------------------------------------------------------------------------------------


def _splits_of(gold):
    return {utt: split for split, d in gold.items() for utt in d}


def _floor_report(counts):
    """(d): the majority-unit accuracy and the in-sample unit unigram, from a split's unit counts."""
    total = int(counts.sum())
    p = counts[counts > 0] / total
    return {
        "frames": total,
        "majority_unit": int(counts.argmax()),
        "majority_unit_acc": float(counts.max()) / total,
        "unigram_logp_per_frame": float((p * np.log(p)).sum()),
        "unigram_is_in_sample": True,
        "uniform_logp_per_frame": float(-np.log(len(counts))),
        "distinct_units": int((counts > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-data", required=True)
    ap.add_argument("--feats", required=True)        # valid.npy of the feature dump
    ap.add_argument("--rev-units", required=True)    # W2vu2RevUnitsJob output dir
    ap.add_argument("--rev-split", default="valid")  # which {split}.rev500 / {split}.eta.npy
    ap.add_argument("--gold", required=True)         # GoldPhonesJob json: the dev phones and splits
    ap.add_argument("--user-dir", action="append", default=[])
    ap.add_argument("--derange-seed", type=int, default=0)
    ap.add_argument("--expect", default="")          # {split: rev_dev.json logz_per_frame}
    ap.add_argument("--expect-tol", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-txt", default="")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_per import _load_feats, _load_model

    model, dictionary = _load_model(args.ckpt, args.data, args.text_data,
                                    "cuda" if torch.cuda.is_available() else "cpu",
                                    user_dirs=args.user_dir)
    device = next(model.parameters()).device
    for d in args.user_dir:
        sys.path.insert(0, os.path.dirname(os.path.abspath(d)))
    from w2vu_rev import rev_term as rt
    from speech_llm.sae.emc.lattice import bigram_history

    feats, offsets, ids = _load_feats(args.feats)
    gold_all = json.load(open(args.gold))
    id2split = _splits_of(gold_all)
    units = [np.array(line.split(), dtype=np.int64)
             for line in open(os.path.join(args.rev_units, f"{args.rev_split}.rev500"))]
    eta = np.load(os.path.join(args.rev_units, f"{args.rev_split}.eta.npy"))
    assert len(units) == len(ids) == eta.shape[0], (len(units), len(ids), eta.shape)
    for i, u in enumerate(units):
        assert len(u) == offsets[i + 1] - offsets[i], f"row {i}: units vs features length"

    phi = getattr(model, "phi", None)
    assert phi is not None, "this readout is about the arm's OWN phi; the checkpoint carries none"
    phi.eval()
    rev_cfg = getattr(model, "rev_cfg", rt.RevTermConfig())
    lat_cfg = rt.REV_LATTICE_CFG
    hist = bigram_history(lat_cfg)
    cfg_rev = phi.cfg
    assert cfg_rev.d_min == lat_cfg.d_min == 2, (cfg_rev.d_min, lat_cfg.d_min)
    cols = torch.tensor(rt.phone_columns(dictionary.index, dictionary.unk()),
                        dtype=torch.long, device=device)
    expect = json.loads(args.expect) if args.expect else {}

    splits = sorted(set(id2split.values()))
    logz = {s: {"own": LogZAcc(), "deranged": LogZAcc()} for s in splits}
    path = {s: {k: PathAcc() for k in ("own", "own_greedy", "deranged", "deranged_greedy", "gold")}
            for s in splits}
    gold_acc = {s: {"forward": 0.0, "viterbi": 0.0, "frames": 0, "utts": 0, "utt_forward": 0.0,
                    "utt_viterbi": 0.0, "skipped_infeasible": 0, "sil_tokens": 0, "tokens": 0}
                for s in splits}
    unit_counts = {s: np.zeros(cfg_rev.n_units, dtype=np.int64) for s in splits}
    no_derangement = 0
    vit_above_logz = 0.0

    order = [u for u in range(len(ids)) if ids[u] in id2split]
    if args.limit:
        order = order[: args.limit]

    with torch.no_grad():
        for start in range(0, len(order), args.batch):
            chunk = order[start: start + args.batch]
            sizes = [int(offsets[u + 1] - offsets[u]) for u in chunk]
            s_max = max(sizes)
            x = torch.zeros(len(chunk), s_max, feats.shape[1], dtype=torch.float32)
            pad = torch.ones(len(chunk), s_max, dtype=torch.bool)
            uu = torch.zeros(len(chunk), s_max, dtype=torch.long)
            for i, (u, size) in enumerate(zip(chunk, sizes)):
                x[i, :size] = torch.from_numpy(
                    np.asarray(feats[offsets[u]:offsets[u + 1]], dtype=np.float32)
                )
                pad[i, :size] = False
                uu[i, :size] = torch.from_numpy(units[u])
            x, pad, uu = x.to(device), pad.to(device), uu.to(device)
            ee = torch.from_numpy(eta[chunk]).to(device)

            gen = model.generator(x, None, pad)
            dense_x, dense_pad = gen["dense_x"], gen["dense_padding_mask"]
            feat_lens = (~dense_pad).sum(-1).long()
            unit_lens = torch.tensor(sizes, dtype=torch.long, device=device)
            assert torch.equal(feat_lens, torch.div(unit_lens + 2, 3, rounding_mode="floor"))
            log_q = rt.log_q_from_dense(dense_x, cols)

            # the segment table and the top-1 emission table: the units and eta, never the posterior
            seg = build_segment_table(phi, uu, ee.to(log_q.dtype))
            seg_dp = seg.double() if rev_cfg.float64 else seg
            seg_pad = scaled_seg_pad(seg_dp, unit_lens, lat_cfg, rev_cfg.tau)
            seg_pad_np = seg_pad.cpu().numpy()
            top1 = top1_unit_table(phi, ee.to(log_q.dtype)).cpu().numpy()

            der = log_q.clone()
            for i, u in enumerate(chunk):
                perm = frame_derangement(ids[u], int(feat_lens[i]), seed=args.derange_seed)
                if perm is None:
                    no_derangement += 1
                    continue
                der[i, : len(perm)] = log_q[i, torch.as_tensor(perm, device=device)]

            for name, lq in (("own", log_q), ("deranged", der)):
                out = rt.reverse_term(log_q=lq, units=uu, unit_lens=unit_lens, feat_lens=feat_lens,
                                      eta=ee.to(lq.dtype), phi=phi, cfg=rev_cfg)
                lq_dp = lq.double() if rev_cfg.float64 else lq
                vit = lattice_best_path(log_q=lq_dp, seg_pad=seg_pad, feat_lens=feat_lens,
                                        unit_lens=unit_lens, cfg=lat_cfg, hist=hist,
                                        temperature=rev_cfg.tau, mode="max")
                labels = lq.argmax(dim=-1).cpu().numpy()
                lq_np = lq_dp.cpu().numpy()
                for i, u in enumerate(chunk):
                    tag, split, n = ids[u], id2split[ids[u]], sizes[i]
                    z = units[u]
                    keep = bool(out.keep[i])
                    logz[split][name].add(-float(out.per_utt[i]), n, keep)
                    segs, reps = vit["paths"][i], vit["repeats"][i]
                    if segs is not None:
                        assert_tiling(segs, n)
                        got = float(vit["score"][i])
                        ref = rescore_path(segs, reps, lq_np[i], seg_pad_np[i], lat_cfg, rev_cfg.tau)
                        if keep:
                            vit_above_logz = max(vit_above_logz, got - (-float(out.per_utt[i]) * n))
                        path[split][name].add(pred=predicted_units(segs, n, top1[i], cfg_rev),
                                              obs=z, segments=segs, score=got, repeats=reps,
                                              rescore_diff=abs(got - ref))
                    gsegs = greedy_segments(labels[i][: int(feat_lens[i])], n, lat_cfg.recognizer_stride)
                    assert_tiling(gsegs, n)
                    path[split][name + "_greedy"].add(
                        pred=predicted_units(gsegs, n, top1[i], cfg_rev), obs=z, segments=gsegs)
                    if name == "own":
                        unit_counts[split] += np.bincount(z, minlength=cfg_rev.n_units)

            # --- (c) the gold oracle: phi alone, the gold phone string force-aligned to the frames
            keep_idx, ys = [], []
            for i, u in enumerate(chunk):
                tag, split = ids[u], id2split[ids[u]]
                y = [PHONE2ID[p] for p in gold_all[split][tag]]
                if feasible(sizes[i], y, cfg_rev):
                    keep_idx.append(i)
                    ys.append(y)
                else:
                    gold_acc[split]["skipped_infeasible"] += 1
            if keep_idx:
                u_max = max(len(y) for y in ys)
                yy = torch.zeros(len(keep_idx), u_max, dtype=torch.long, device=device)
                for j, y in enumerate(ys):
                    yy[j, : len(y)] = torch.tensor(y, dtype=torch.long, device=device)
                sel = torch.tensor(keep_idx, dtype=torch.long, device=device)
                u_lens = torch.tensor([len(y) for y in ys], dtype=torch.long, device=device)
                fwd_g, vit_g, paths_g = gold_forced_align(
                    phi, uu.index_select(0, sel), yy, ee.index_select(0, sel).to(log_q.dtype),
                    unit_lens.index_select(0, sel), u_lens)
                for j, i in enumerate(keep_idx):
                    u = chunk[i]
                    split, n = id2split[ids[u]], sizes[i]
                    segs = paths_g[j]
                    if segs is None:
                        gold_acc[split]["skipped_infeasible"] += 1
                        continue
                    assert_tiling(segs, n)
                    a = gold_acc[split]
                    a["utts"] += 1
                    a["frames"] += n
                    a["forward"] += float(fwd_g[j])
                    a["viterbi"] += float(vit_g[j])
                    a["utt_forward"] += float(fwd_g[j]) / n
                    a["utt_viterbi"] += float(vit_g[j]) / n
                    a["tokens"] += len(ys[j])
                    a["sil_tokens"] += sum(1 for k in ys[j] if k == cfg_rev.sil_id)
                    path[split]["gold"].add(pred=predicted_units(segs, n, top1[i], cfg_rev),
                                            obs=units[u], segments=segs, score=float(vit_g[j]))
            if start % (args.batch * 50) == 0:
                print(f"{start}/{len(order)}", flush=True)

    report = {
        "checkpoint": args.ckpt,
        "readout": "diagnostic: it measures phi's unit prediction; it draws no conclusion",
        "tau": rev_cfg.tau, "prior_weight": rev_cfg.prior_weight, "float64_dp": rev_cfg.float64,
        "band": lat_cfg.band, "d_min": lat_cfg.d_min, "d_max": lat_cfg.d_max,
        "d_max_sil": lat_cfg.d_max_sil, "topology": lat_cfg.topology,
        "recognizer_stride": lat_cfg.recognizer_stride, "n_units": cfg_rev.n_units,
        "history": hist.name, "derange_seed": args.derange_seed,
        "utts_without_derangement": no_derangement,
        "max_viterbi_minus_logz": vit_above_logz,
        "expect_logz_per_frame": expect, "expect_tol": args.expect_tol,
        "conventions": {
            "logz_per_frame": "sum_utt log Z_tau / sum_utt S, the rev_dev.py headline (higher better)",
            "viterbi": "max-semiring path of the same lattice; score_per_frame = sum score / sum S",
            "top1_acc": "frames whose argmax_u nu_phi(k, bucket(d), bucket(r), eta) equals z_s, / frames",
            "greedy": "argmax_k log q per recognizer frame, min-duration ignored, runs expanded by stride 3",
            "deranged": "log_q frame axis permuted inside each utterance with no fixed point, seed "
                        f"{args.derange_seed}; units, eta and lengths untouched",
            "gold": "gold phones force-aligned to the frames under phi alone (d_min <= d <= D_k); "
                    "fp32, reverse.forward_logsum for the sum and its max variant for the Viterbi",
            "floors": "majority unit and the unit unigram are IN-SAMPLE on the split's own units; "
                      "they are the floor of the gold number only, never of logz_per_frame",
            "top10_predicted_unit_share": "share of frames whose PREDICTED unit is in the 10 most "
                                          "frequent predicted units (observed share beside it)",
        },
        "splits": {},
    }
    for split in splits:
        n_utts = logz[split]["own"].kept + logz[split]["own"].z_zero
        if not n_utts:                      # only reachable under --limit; a full run reads both
            continue
        rec = {"utts": n_utts, "floors": _floor_report(unit_counts[split])}
        for name in ("own", "deranged"):
            entry = logz[split][name].report()
            entry["viterbi"] = path[split][name].report()
            entry["greedy"] = path[split][name + "_greedy"].report(with_score=False)
            rec[name] = entry
        a = gold_acc[split]
        g = {"utts": a["utts"], "frames": a["frames"], "tokens": a["tokens"],
             "sil_tokens_in_gold": a["sil_tokens"], "skipped_infeasible": a["skipped_infeasible"],
             "forward_logp_per_frame": a["forward"] / a["frames"] if a["frames"] else None,
             "viterbi_logp_per_frame": a["viterbi"] / a["frames"] if a["frames"] else None,
             "mean_utt_forward_logp_per_frame": a["utt_forward"] / a["utts"] if a["utts"] else None,
             "mean_utt_viterbi_logp_per_frame": a["utt_viterbi"] / a["utts"] if a["utts"] else None,
             "viterbi": path[split]["gold"].report()}
        rec["gold"] = g
        report["splits"][split] = rec

    # --- self-check: the own headline must reproduce the arm's registered rev_dev.json value
    checks = {}
    for split, want in expect.items():
        assert split in report["splits"], (split, sorted(report["splits"]))
        got = report["splits"][split]["own"]["logz_per_frame"]
        checks[split] = {"got": got, "want": want, "abs_diff": abs(got - want)}
    report["logz_check"] = checks
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2), flush=True)
    if args.out_txt:
        with open(args.out_txt, "w") as fh:
            fh.write(_summary(report))
        print(_summary(report), flush=True)
    for split, c in checks.items():
        assert c["abs_diff"] <= args.expect_tol, (split, c)
    assert vit_above_logz <= 1e-3, f"a Viterbi score exceeds its own log Z by {vit_above_logz}"
    for split in report["splits"]:
        for name in ("own", "deranged"):
            d = report["splits"][split][name]["viterbi"]["max_rescore_diff"]
            assert d <= 1e-3, f"{split}/{name}: backtraced path re-scores {d} off its Viterbi score"


def _summary(report):
    """The txt readout: one block per split, every headline with its convention named."""
    lines = ["rev_diag -- phi's unit prediction on dev (diagnostic; no conclusion is drawn here)",
             f"checkpoint: {report['checkpoint']}",
             "tau=%s beta=%s d_min=%s D=%s/%s band=%s stride=%s K=%s history=%s derange_seed=%s"
             % (report["tau"], report["prior_weight"], report["d_min"], report["d_max"],
                report["d_max_sil"], report["band"], report["recognizer_stride"],
                report["n_units"], report["history"], report["derange_seed"]),
             f"logz self-check vs rev_dev.json: {json.dumps(report['logz_check'])}",
             f"max(viterbi - logZ) = {report['max_viterbi_minus_logz']:.3e} (must be <= 0 up to fp)",
             ""]
    for split, rec in sorted(report["splits"].items()):
        f = rec["floors"]
        lines.append(f"[{split}] utts={rec['utts']} frames={f['frames']}")
        for name in ("own", "deranged"):
            e = rec[name]
            v, g = e["viterbi"], e["greedy"]
            lines.append(
                "  %-9s logZ/frame %+8.5f (mean utt %+8.5f, z_zero %d)" %
                (name, e["logz_per_frame"], e["mean_utt_logz_per_frame"], e["z_zero"]))
            lines.append(
                "            viterbi: score/frame %+8.5f  top1 acc %.4f (mean utt %.4f)  "
                "tokens/frame %.4f  phones %d  top10 pred share %.3f (obs %.3f)" %
                (v["score_per_frame"], v["top1_acc"], v["mean_utt_top1_acc"], v["tokens_per_frame"],
                 v["distinct_phone_states"], v["top10_predicted_unit_share"],
                 v["top10_observed_unit_share"]))
            lines.append(
                "            greedy : top1 acc %.4f (mean utt %.4f)  tokens/frame %.4f  phones %d  "
                "top10 pred share %.3f" %
                (g["top1_acc"], g["mean_utt_top1_acc"], g["tokens_per_frame"],
                 g["distinct_phone_states"], g["top10_predicted_unit_share"]))
        gd = rec["gold"]
        gv = gd["viterbi"]
        if gv is None:
            lines.append("  gold      no utterance could be aligned (skipped %d)"
                         % gd["skipped_infeasible"])
            lines.append("")
            continue
        lines.append(
            "  gold      forward logp/frame %+8.5f  viterbi logp/frame %+8.5f  top1 acc %.4f "
            "(mean utt %.4f)  phones %d  top10 pred share %.3f  skipped %d" %
            (gd["forward_logp_per_frame"], gd["viterbi_logp_per_frame"], gv["top1_acc"],
             gv["mean_utt_top1_acc"], gv["distinct_phone_states"],
             gv["top10_predicted_unit_share"], gd["skipped_infeasible"]))
        lines.append(
            "  floors    majority unit %d acc %.4f | unigram logp/frame %+8.5f (in-sample) | "
            "uniform %+8.5f" %
            (f["majority_unit"], f["majority_unit_acc"], f["unigram_logp_per_frame"],
             f["uniform_logp_per_frame"]))
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
