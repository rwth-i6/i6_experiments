"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/rate_term.py.

Port note: the pieces the blank-free train step reads -- the non-SIL tilt / expected count,
the FD defaults, ``fd_batch_fits``, ``_stacked_fd_call`` and ``_fd_passes``.  Cut: section (A)
(``WORDS_PER_SEC``, the T_phi paths, ``RhoReading``, ``compute_rho`` and the rho cache, which
imported another ``users/`` package), section (B) (``TiltSolution``, ``solve_rate_tilt``) and
``RateStats`` / ``rate_loss`` (the train step differentiates the term itself).  rho reaches the
model as the ``rate_rho_hz`` model arg.  The text below is the source's and still names them.

emc.rate_term -- the LABEL-FREE rate term of SAE_4A.md "S3b-R" (registered 2026-09-15).

S3 collapsed from a flat start to an all-blank output (SAE_4A.md "S3b": blank share 0.994, 0.000
emitted phones / s at every fixed-point iteration), and nothing in ``L_tau`` prices an empty
output -- ``d_min`` and ``D_sil`` price segments that EXIST.  The remedy under test adds, per
utterance,

    L_rate = ((E_q[N] / T - rho) / rho)^2

with ``N`` the expected number of EMITTED NON-SIL tokens under the tempered posterior (the sum of
``LatticeOutput.seg_post`` over the non-SIL types, their durations and their start frames -- NOT
``expected_tokens``, which counts SIL as the ordinary symbol the lattice treats it as), ``T`` the
utterance's frames, and ``rho`` the target rate in tokens PER FRAME.

rho is LABEL-FREE (SAE_4A.md "S3b-R")
-------------------------------------
``rho_hz = phones_per_word(T_phi) * WORDS_PER_SEC`` and ``rho = rho_hz / 50`` (both clocks are the
50 Hz enc50 / L15 rate, INTERFACES.md).  ``phones_per_word`` is a property of the PHONEMIZED TEXT
``T_phi`` (``TextToPhonemeJob.THKMON3k9LJQ``, SAE_4A.md Artifacts) -- the same text side the prior
and ``L_agg`` are built from -- and ``WORDS_PER_SEC = 2.7`` is a DISCLOSED CONSTANT for read
English speech, not a measurement of this corpus' audio.  No transcript, no alignment and no
recognizer output enters it: :func:`compute_rho` has no gold/transcript/alignment parameter at all
(asserted by signature in ``test_rate_term.py``).

The dev-side gold phone rate (9.8 / 9.4 per second, SAE_1f.md:533-536) is REPORTING-ONLY and must
never enter this module, a loss or a selection.  It appears in no function here.

The gradient (the user's ``E_{q_theta}``)
-----------------------------------------
Under the lattice's Gibbs measure the path score is scaled by ``1/tau``, so with ``b`` a per-token
TILT added to every non-SIL entry of the segment table (one ``b`` per emitted non-SIL token on any
path, because a token contributes exactly one ``G[k, s, d]``),

    d log Z / d b            = (1/tau) * E[N]
    d log Z / d log_q[t, c]  = (1/tau) * post_q[t, c]
    => d E[N] / d log_q[t, c] = d post_q[t, c] / d b            (a symmetric second derivative)

which is what :func:`rate_loss` differentiates: the derivative on the right is taken by a CENTRAL
FINITE DIFFERENCE in ``b`` -- two extra tilted passes at ``seg_table +/- eps`` on the non-SIL types,
run as ONE stacked DP CALL wherever the lattice's batch budget allows (``batched``, below) --
and the per-utterance surrogate

    2 (E[N]/T - rho) / (rho^2 T) * sum_{t,c} g[t, c] * log_q[t, c],  g = (post_q(+eps) - post_q(-eps)) / (2 eps)

carries the whole gradient to theta.  ``phi`` gets NO gradient from this term (the value depends on
phi too, but the user's form differentiates ``E_{q_theta}`` only); ``seg_table`` is detached in
every extra pass.  The value is re-attached exactly as ``lattice.lattice_loss`` does
(``surrogate + (value - surrogate).detach()``), and ``Z = 0`` utterances are kept out of both, never
folded in as a zero (memory: impossible-scores-poison-means).

ONE-SIDED HINGE (``hinge = True``) -- A DISCLOSED DEVIATION
-----------------------------------------------------------
The user's stated form is TWO-SIDED and is the default here.  ``hinge = True`` replaces it, in the
same normalized units, by

    L_rate = (max(0, rho - E_q[N]/T) / rho)^2

which is IDENTICAL to the two-sided form below rho (value and gradient: ``-2 (rho - r)`` is
``2 (r - rho)`` there) and EXACTLY ZERO above it.  Why it is offered: during the tau anneal the
expectation starts ABOVE rho -- at sub-epochs 1-3 of S3 it is ~2x rho even with SIL's share removed
-- so the two-sided term is active in the direction "emit FEWER tokens", which is the direction of
the collapse it exists to prevent (design review 2026-09-15, F1, remedy (b)).  The hinge cannot
push emissions down; it can only fail to push them up.

This is a DEVIATION from the form the user stated, and it is a real one: the hinged term no longer
prices an output that emits too MUCH, so an arm that runs it has no term holding the rate from
above and must be read with that in mind.  It is off by default, it is per arm, and the arms that
use it say so in their tag.  The finite difference is UNCHANGED by it -- ``g = d post_q / d b`` is a
property of the lattice, not of the loss shape -- so both FD paths (the stacked call and the
sequential passes) carry the hinge through the single coefficient ``d value / d E[N]``, and the
term costs exactly what the two-sided one costs.

FREE CONSISTENCY CHECK.  The same two passes give ``tau * (log Z(+eps) - log Z(-eps)) / (2 eps)``,
which must equal ``E[N]`` at ``b = 0`` up to the O(eps^2) FD error.  Its relative error is reported
as ``emc/rate_fd_check`` -- pre-registered convention (memory: preregistration-lives-with-the-code):

    rate_fd_check = sum_kept |E_fd - E[N]| / max(sum_kept E[N], 1e-3)

a POOLED relative deviation, so a collapsed batch (E[N] -> 0) cannot make it explode by dividing a
rounding error by zero.  It is a monitor, never a gate.

COST.  The value costs nothing (it is read off the caller's own pass).  The gradient costs two
tilted passes at ``mode = "central"`` and one at ``"forward"``.  The DP is LAUNCH-BOUND -- one
Python loop over frames per CALL, everything inside a frame vectorized over the batch -- so the two
tilts are stacked along the batch axis and run as ONE call: utterances never interact in this DP,
so a call of ``2B`` is the same arithmetic as two calls of ``B`` for one frame loop instead of two.
It holds twice the per-pass activations, so it is taken only where ``lattice``'s own two caps
(``MAX_UTTS_PER_BATCH``, ``MAX_BATCH_FRAMES``) still hold for the stacked batch; otherwise the
sequential passes run and :class:`RateStats` carries the reason.  ``batched = False`` forces the
sequential path at the memory of a single pass.

This module adds no code to ``lattice.py``: the DP is hand-written and its backward is pinned
bit-for-bit, so every helper the rate term needs lives here and calls the lattice's own public
entry points.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from .lattice import (
    MAX_BATCH_FRAMES,
    MAX_UTTS_PER_BATCH,
    LatticeConfig,
    LatticeOutput,
    estimate_peak_gib,
    lattice_forward_backward,
)

__all__ = [
    "DEFAULT_FD_EPS",
    "DEFAULT_FD_MODE",
    "DEFAULT_FD_BATCHED",
    "DEFAULT_HINGE",
    "fd_batch_fits",
    "nonsil_type_mask",
    "tilt_segment_table",
    "expected_nonsil_tokens",
]

# ===================================================================================================
# the non-SIL tilt b and the non-SIL expected count
# ===================================================================================================


def nonsil_type_mask(cfg: LatticeConfig, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """``[n_phones]``: 1 on every non-SIL type, 0 on SIL.  The tilt's support, once."""
    mask = torch.ones(cfg.n_phones, device=device, dtype=dtype)
    mask[cfg.sil_id] = 0.0
    return mask


def tilt_segment_table(seg_table: torch.Tensor, b: float, cfg: LatticeConfig) -> torch.Tensor:
    """``seg_table + b`` on the NON-SIL types only; SIL (and blank, which has no segment) untouched.

    ``seg_table`` is ``[B, K, D, S+1]``, so the tilt broadcasts over ``(d, s)``: every path pays
    ``b`` exactly once per emitted non-SIL token, which is what makes ``d log Z / d b`` the expected
    non-SIL token count.  Illegal entries are ``reverse``'s finite ``NEG_INF`` (-1e30) and stay
    there under a finite shift; the DP re-floors them anyway (``lattice._scaled_seg_pad``).
    Out of place: the caller's table may carry the graph to phi.
    """
    if b == 0.0:
        return seg_table
    mask = nonsil_type_mask(cfg, device=seg_table.device, dtype=seg_table.dtype)
    return seg_table + float(b) * mask.view(1, -1, 1, 1)


def expected_nonsil_tokens(out: LatticeOutput, cfg: LatticeConfig) -> torch.Tensor:
    """``[B]`` E[number of emitted NON-SIL tokens] -- ``seg_post`` summed over the non-SIL types,
    their durations and their start frames.

    NOT ``out.expected_tokens``: that one sums ``mass_k`` over EVERY type, SIL included (lattice.py
    :1139/:1174), because SIL is an ordinary emitted symbol in this topology.  The two differ by
    E[#SIL tokens], which is exactly the quantity a blank/SIL collapse inflates.
    """
    mass = out.seg_post.sum(dim=(2, 3))  # [B, K]
    return mass.sum(dim=1) - mass[:, cfg.sil_id]


# ===================================================================================================
# (C) the training term
# ===================================================================================================

DEFAULT_FD_EPS = 0.25    # nats per non-SIL token (orchestrator brief 2026-09-15)
DEFAULT_FD_MODE = "central"
# Run the tilted passes as ONE DP call on a stacked batch (orchestrator brief 2026-09-15): the DP is
# a Python loop over frames with small kernels, so its cost rides on the number of CALLS (T launches
# each), not on B.  ``False`` is the sequential fallback, which costs one call per tilt and the
# memory of one pass.  The stacked call is used only where it fits the lattice's OWN published
# budget (:func:`fd_batch_fits`); it is never forced past it.
DEFAULT_FD_BATCHED = True
# The two-sided form of the user's brief is the default; ``True`` is the ONE-SIDED hinge
# max(0, rho - r)^2 (design review 2026-09-15, remedy (b)), a DISCLOSED DEVIATION -- see the
# "ONE-SIDED HINGE" section of the module docstring.
DEFAULT_HINGE = False


class _FdResult(NamedTuple):
    post_plus: torch.Tensor
    post_minus: Optional[torch.Tensor]
    log_z: torch.Tensor             # [n_tilts, B], in the order the tilts were requested
    n_passes: int
    n_dp_calls: int
    batched: bool
    note: str


def fd_batch_fits(batch: int, t_max: int, n_tilts: int) -> Tuple[bool, str]:
    """Can ``n_tilts`` tilts run as ONE DP call of batch ``n_tilts * B``, under lattice's own caps?

    The stacked call is the whole point of the batched path -- one Python frame loop instead of
    ``n_tilts`` -- but it also holds ``n_tilts`` times the per-pass activations, and the two caps of
    :func:`lattice.check_batch_budget` (``MAX_UTTS_PER_BATCH``, ``MAX_BATCH_FRAMES``) are the only
    published statement of what this module is allowed to allocate.  They are NOT overridden here:
    a stacked batch that exceeds either one falls back to the sequential passes and says so.
    """
    stacked = int(n_tilts) * int(batch)
    msgs = []
    if stacked > MAX_UTTS_PER_BATCH:
        msgs.append(f"{n_tilts} x B = {stacked} > MAX_UTTS_PER_BATCH = {MAX_UTTS_PER_BATCH}")
    if stacked * int(t_max) > MAX_BATCH_FRAMES:
        msgs.append(
            f"{n_tilts} x B x T_max = {stacked * int(t_max)} > MAX_BATCH_FRAMES = {MAX_BATCH_FRAMES}"
        )
    if not msgs:
        return True, ""
    return False, (
        "; ".join(msgs)
        + f" -- the stacked call would peak at ~{estimate_peak_gib(stacked, t_max):.2f} GiB "
        f"(one pass: ~{estimate_peak_gib(batch, t_max):.2f} GiB); running the tilts sequentially"
    )


def _stacked_fd_call(
    log_q: torch.Tensor,
    base: torch.Tensor,
    dp_kwargs: Dict,
    cfg: LatticeConfig,
    tilts: Sequence[float],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """The tilts as ONE DP call on a batch of ``len(tilts) * B``; ``(post_q per tilt, log_z stack)``.

    Utterances are INDEPENDENT in this DP -- the forward table, the band, the segment posteriors and
    log Z all carry the batch axis and nothing reduces across it -- so stacking the tilts along B is
    the same arithmetic as running them one after another, for the price of one Python frame loop.
    Everything that carries a batch axis is repeated in the SAME order (``repeat`` on dim 0, tilt by
    tilt), and ``prior_log_bi`` / ``history`` are batch-free tables shared by every row.
    """
    n = len(tilts)
    kw = dict(dp_kwargs)
    prior = kw["prior_log_bi"]
    assert prior.dim() == 2, (
        f"prior_log_bi is the batch-free [|h|, K] table; got shape {tuple(prior.shape)}, which would "
        "have to be repeated along the batch for the stacked tilt call"
    )
    kw["feat_lens"] = dp_kwargs["feat_lens"].repeat(n)
    kw["unit_lens"] = dp_kwargs["unit_lens"].repeat(n)
    if dp_kwargs.get("log_q_init") is not None:
        kw["log_q_init"] = dp_kwargs["log_q_init"].detach().repeat(n, 1, 1)
    log_q_s = log_q.detach().repeat(n, 1, 1)
    seg_s = torch.cat([tilt_segment_table(base, float(b), cfg) for b in tilts], dim=0)
    out = lattice_forward_backward(log_q_s, seg_s, **kw)
    b_n = log_q.shape[0]
    posts = [out.post_q[i * b_n:(i + 1) * b_n] for i in range(n)]
    log_z = torch.stack([out.log_z[i * b_n:(i + 1) * b_n] for i in range(n)])
    return posts, log_z


def _fd_passes(
    log_q: torch.Tensor,
    seg_table: torch.Tensor,
    dp_kwargs: Dict,
    cfg: LatticeConfig,
    *,
    eps: float,
    mode: str,
    batched: bool = DEFAULT_FD_BATCHED,
) -> _FdResult:
    """The tilted passes of the finite difference in ``b``.

    The extra passes are IDENTICAL to the caller's own (same cfg, temperature, anchor, prior,
    log_q_init, history and DP knobs) except for the tilt on the non-SIL segment entries, and they
    run under ``no_grad`` on a DETACHED table: their post_q is a CONSTANT of the surrogate.

    ``batched`` runs the two central-difference tilts as ONE DP call on a stacked batch instead of
    two sequential calls -- the DP is launch-bound (one Python frame loop per CALL), so two calls
    cost about twice one call however small B is, while the stacked call costs one loop and
    ``n_tilts`` times the activations.  It is used only where it fits the lattice's own batch budget
    (:func:`fd_batch_fits`); otherwise the sequential path runs and the reason is reported.
    ``mode = "forward"`` has a single tilted pass, so there is nothing to stack.
    """
    kw = dict(dp_kwargs)
    kw["collect_stats"] = True  # the brief: the extra passes collect the same statistics
    base = seg_table.detach()
    tilts = [+eps, -eps] if mode == "central" else [+eps]
    fits, note = (False, "one tilted pass only")
    if batched and len(tilts) > 1:
        fits, note = fd_batch_fits(log_q.shape[0], log_q.shape[1], len(tilts))
    with torch.no_grad():
        if fits:
            posts, log_z = _stacked_fd_call(log_q, base, kw, cfg, tilts)
            return _FdResult(posts[0], posts[1], log_z, len(tilts), 1, True, "")
        outs = [
            lattice_forward_backward(log_q.detach(), tilt_segment_table(base, float(b), cfg), **kw)
            for b in tilts
        ]
    log_z = torch.stack([o.log_z for o in outs])
    minus = outs[1].post_q if len(outs) > 1 else None
    return _FdResult(outs[0].post_q, minus, log_z, len(tilts), len(tilts), False, note)
