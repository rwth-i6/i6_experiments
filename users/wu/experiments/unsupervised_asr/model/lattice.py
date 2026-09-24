"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lattice.py.

Port note: ``NEG_INF`` is imported from ``.reverse`` (the source imported the same -1.0e30
from ``sae/psi_align``); the ``__main__`` bench imports ``.prior`` relatively, so it runs as
``python -m <package>.model.lattice``.  Every other line is the source's.

emc.lattice -- the exact-marginal cycle objective L_tau of SAE §4a, stage S0b (SAE_4A.md:132-140).

The objective
-------------
The latent variable is the PAIR (``path``, ``segmentation``): a CTC path ``path`` over T recognizer
frames, and a segmentation of the S reverse frames into the ``|B(path)|`` tokens of the collapsed
transcript. Per utterance (orchestrator brief 2026-09-15, the formalization of SAE_4A.md:15-19)

    L_tau = -log sum_{path, seg} exp( (1/tau) * [   sum_t log q_theta(path_t | x_t)
                                                  + alpha * sum_t log q_init(path_t | x_t)
                                                  + beta * log P_psi(B(path))
                                                  + log p_phi(z, seg | B(path), eta) ] )

with T = S (both 50 Hz; the train step asserts EQUALITY per utterance, never a ratio --
design review 2026-09-15, SAE_3E1.md:1371), ``q_init`` a frozen copy of the init recognizer
(``alpha = 0`` disables it), ``beta = 1`` by default, and the band ``|s_end(u) - t_emit(u)| <= W``
(W = 25, SAE_4A.md:76) coupling the reverse-segment END frame of token u to the CTC frame at which
token u is first emitted.

Naming in this module (no Greek letters anywhere, memory: no-greek-letters): ``tau`` is
``temperature``, ``alpha`` is ``anchor_weight``, ``beta`` is ``prior_weight``; ``fwd`` / ``bwd`` are
the two DP tables (the plan's DP alpha/beta), never the scalars above.

The lattice
-----------
Everything in the summand factorizes per arc, so the sum is an exact log-semiring forward over the
states ``(t, s, h, f)``: ``t`` the CTC frame, ``s`` the reverse frame reached (segments END at s),
``h`` the PHONE HISTORY the prior conditions on, ``f`` the CTC flag "the last frame emitted the last
phone of h" (the repeat rule). The band keeps ``s`` inside ``[t - W, t + W]``, which is stored as the
offset ``o = s - t + W`` in ``[0, 2W]``.

``h`` is a knob, not the bigram it started as (:class:`PriorHistory`): ``|h| = n_outer * n_ctx`` with
``h = (outer, last)``, ``last`` the last emitted phone and ``outer`` whatever the history keeps of
the phone before it -- nothing (bigram, ``n_outer = 1``, M = 82 ``(h, f)`` pairs), its class
(class trigram, ``n_outer = C``) or its identity (trigram, ``n_outer = n_ctx``). The prior term is
then ``log_prior[h, k]`` over a ``(|h|, K)`` table and the emit destination is
``h' = next_h[h, k] = succ[last(h)] * n_ctx + k``.

Three transitions out of ``(t, s, h, f)``:

1. **blank** -> ``(t+1, s, h, 0)``, weight ``(1/tau)[log q(blank|x_t) + alpha log q_init(blank|x_t)]``.
2. **repeat** (``f = 1`` only) -> ``(t+1, s, h, 1)``, weight
   ``(1/tau)[log q(h|x_t) + alpha log q_init(h|x_t)]``. It does **not** advance ``s`` and adds no
   prior and no reverse term (design review 2026-09-15: "the CTC repeat transition must not advance s").
3. **emit** ``k`` with duration ``d`` (``d_min <= d <= D_k``) -> ``(t+1, s+d, k, 1)``, weight
   ``(1/tau)[log q(k|x_t) + alpha log q_init(k|x_t) + beta log P_psi(k|h) + G[k, s, d]]``, allowed
   only if ``|s + d - (t+1)| <= W``. SIL may follow SIL as a new token; a non-SIL ``k`` whose
   ``k == last(h)`` requires ``f = 0`` (orchestrator brief 2026-09-15).

**How the history advances** (the rule is the bigram's, unchanged at every order): ONLY the emit arc
moves ``h``, to ``next_h[h, k]``, so ``last`` becomes the emitted ``k`` and ``outer`` becomes what
``succ`` keeps of the previous ``last``. The blank and the repeat arc leave ``h`` exactly as it is --
a blank does not reset the context and a repeat re-emits the SAME token, which is one token and
therefore one prior term. SIL is a symbol like any other on that path: it is emitted, it enters
``h``, and a SIL following a SIL is a second token with its own ``log P_psi(SIL | h)``.

Start ``(0, 0, BOS, 0)``; final states ``(T, S, h, f)`` for every ``(h, f)``. **No end-of-sentence
prior term**: ``prior.PhoneNgramPrior`` pre-registers its convention as "no end-of-sequence term"
(prior.py docstring), so there is no ``log P_psi(EOS | h)`` to add and none is invented here.

``G[k, s, d] = log p_phi(z_s..z_{s+d-1} | k, d, position-dependent emissions, eta) + log p(d | k)``
comes from ``reverse.SegmentalReverseModel`` (``segment_scores`` + ``duration_log_probs``); see
:func:`build_segment_table`.

Two documented properties of the topology, both following the brief rather than strict CTC:

* The transition rule ``|s + d - (t+1)| <= W`` is exactly the state invariant ``|s - t| <= W`` at
  every state, i.e. the band on the segment END is read one CTC frame after the emission frame
  (``|s_end(u) - (t_emit(u) + 1)| <= W``). The design review notes the built-in offset of the same
  comparison (design_review_4a_2026-09-15.md:103-105); the one-frame convention is stated here so the
  brute-force oracle and the DP agree by construction.
* Because a SIL may follow a SIL without an intervening blank, one CTC path can collapse to several
  token strings. The latent is therefore ``(path, token string consistent with path, segmentation)``
  and the sum runs over all three, which is what an exact marginal wants.

Gradients
---------
The DP runs on DETACHED tables and its backward is written by hand: autograd through the expanded
per-step transition tensor (``band x M x D x |V|``, ~4e6 floats) would keep it for every t, ~9 GB per
utterance (memory: lattice-dp-autograd-memory; design review item 3). The manual log-semiring
backward yields the arc posteriors ``post_q[t, c]`` (mass of arcs emitting symbol c at frame t,
blanks and repeats included) and ``seg_post[k, s, d]``; the autograd surrogate is then

    -(1/tau) * ( sum post_q * log_q_with_grad + sum seg_post * seg_table_with_grad )

whose gradient is exactly ``dL/dtheta`` and ``dL/dphi``. Autograd is used only as the TEST ORACLE on
tiny shapes (:func:`forward_log_z` with grad-carrying inputs, ``test_lattice.py``).

How a frame is computed (this changes no value of the objective, only its cost)
------------------------------------------------------------------------------
``reduction`` picks how the two sums inside a frame are taken: elementwise (materialise the reduced
axis, ``[B, O, |h|, K]`` and ``[B, O_src, G, O_dst, K]``) or as log-semiring MATRIX PRODUCTS against
the constant masked prior and the frame's band matrix (D3 of
reports/estimate_prior_order_2026-09-15.full.md). The matrix path never materialises a reduced axis,
so a frame's traffic carries ``|h|`` instead of ``|h| x K`` and its kernel count does not grow with
``|h|`` at all. ``"auto"`` keeps the elementwise path at the bigram, whose floating point is banked
bit-for-bit, and takes the matrix path at every larger history. ``checkpoint = S`` (D4) keeps only
every S-th forward frame resident and recomputes the rest inside the backward, which is what makes
a ``|h| = 1681`` forward table (56 GiB at B = 125, T = 704) fit at all.

``Z_tau = 0`` (no path at all: the band excludes every path, or no segmentation of S frames exists)
is detected PER UTTERANCE on the forward table at the final states and reported in ``z_zero``; it is
never masked into the batch sum (design review 2026-09-15; memory: impossible-scores-poison-means).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import torch

from .reverse import NEG_INF  # a finite -inf: logsumexp of an all-masked row stays NaN-free

__all__ = [
    "LatticeConfig",
    "LatticeOutput",
    "PriorHistory",
    "bigram_history",
    "class_trigram_history",
    "trigram_history",
    "build_prior_history",
    "build_segment_table",
    "forward_log_z",
    "lattice_forward_backward",
    "lattice_loss",
    "MAX_UTTS_PER_BATCH",
    "MAX_BATCH_FRAMES",
    "estimate_peak_gib",
    "check_batch_budget",
    "arc_weights",
    "band_matrix",
    "forward_ctx",
    "scaled_prior_term",
]

# ---------------------------------------------------------------------------------------------------
# batch budget -- the two caps the S2 config must respect
# ---------------------------------------------------------------------------------------------------

# Resident memory of one DP call scales with B * T_max, the PADDED utterance-frames of the batch,
# not with its real frame count: the forward table is ``[B, T, 2W+1, n_ctx, 2]`` fp32 and the padded
# segment table ``[B, S+1+2W, D_pad, K]`` fp32. RETURNN's ``batch_size`` counts REAL frames, so it
# alone does not bound this: one long utterance batched with many short ones can blow the DP up
# (review_s0b_core_2026-09-15.md, CONCERN 4b). The DP therefore checks both caps itself instead of
# trusting the config, and the S2 config must respect them.
# Both caps are MEASURED shapes, never an extrapolation to the device limit. bench() on a GH200
# (2026-09-15; W = 25, K = 40, D = 25 / D_sil = 50, fp32, T = S = 500) reports 0.55 / 1.08 / 2.16 /
# 4.33 GiB peak and 25 / 52 / 98 / 122 utt/s at B = 16 / 32 / 64 / 128, i.e. 6.8e-5 GiB per padded
# utterance-frame, and B = 128 is the largest shape benched. The S1b/S2 operating point is
# max_seqs = 128 / batch_size = 88000 padded frames (orchestrator brief 2026-09-15), so these caps
# are a backstop that only fires when a config ignores it.
MAX_UTTS_PER_BATCH = 128  # B
# B * T_max, PADDED. Raised from 64,000 (= the benched 128 x 500) to 128,000 for the 2026-09-15
# max_seqs 64 -> 128 change: the benched GiB/frame is linear, so 128,000 frames is 8.7 GiB of DP
# peak on a 96 GiB GH200, and the config's own batch_size (88,000) is what binds in practice.
MAX_BATCH_FRAMES = 128_000
PEAK_GIB_PER_UTT_FRAME = 6.8e-5  # measured peak / (B * T_max), linear in B * T_max (bench 2026-09-15)


def estimate_peak_gib(batch: int, t_max: int) -> float:
    """Peak allocation of one forward+backward DP call, GiB. Linear in the padded utterance-frames.

    Fitted on the GH200 bench of 2026-09-15 (W = 25, K = 40, D_sil = 50, fp32, T = S); it is an
    estimate of THIS module's peak only, not of the whole training step.
    """
    return PEAK_GIB_PER_UTT_FRAME * float(batch) * float(t_max)


def check_batch_budget(batch: int, t_max: int, *, strict: bool = True) -> None:
    """Raise (``strict``) or warn unless the batch respects both caps of the batch budget."""
    msgs = []
    if batch > MAX_UTTS_PER_BATCH:
        msgs.append(f"B = {batch} > MAX_UTTS_PER_BATCH = {MAX_UTTS_PER_BATCH} (set RETURNN max_seqs)")
    if batch * t_max > MAX_BATCH_FRAMES:
        msgs.append(
            f"B * T_max = {batch * t_max} > MAX_BATCH_FRAMES = {MAX_BATCH_FRAMES} "
            f"(set RETURNN batch_size, in FRAMES)"
        )
    if not msgs:
        return
    text = (
        "lattice batch budget exceeded: "
        + "; ".join(msgs)
        + f" -- estimated DP peak {estimate_peak_gib(batch, t_max):.2f} GiB"
    )
    if strict:
        raise ValueError(text)
    print("WARNING: " + text, flush=True)


@dataclass
class LatticeConfig:
    """Topology of the product lattice. Every experimental constant carries its source."""

    n_phones: int = 40  # 39 ARPAbet + SIL, prior.PHONES (SAE_4A.md:40-41)
    sil_id: int = 39  # SIL is the last type (prior.SIL_ID)
    band: int = 25  # W = 25 frames = 0.5 s at 50 Hz (SAE_4A.md:76)
    d_min: int = 2  # standing constraint, never below 2 (SAE_4A.md:53-56; SAE_3E1.md:161)
    d_max: int = 25  # D = 25 frames for phones (SAE_4A.md:53)
    d_max_sil: int = 50  # D_sil = 50 frames for SIL (SAE_4A.md:53)
    frame_rate_hz: float = 50.0  # reverse enc50 / L15 clock (SAE_4A.md:43-49)
    topology: str = "ctc"
    recognizer_stride: int = 1

    def __post_init__(self):
        assert self.d_min >= 2, "d_min >= 2 is standing (SAE_4A.md:53-56); it is not a knob below 2"
        assert self.d_max >= self.d_min and self.d_max_sil >= self.d_min
        assert 0 <= self.sil_id < self.n_phones
        assert self.band >= 1
        assert self.topology in ("ctc", "blankfree")
        assert self.recognizer_stride >= 1

    @property
    def n_ctx(self) -> int:
        """History values h: the phones plus BOS (prior.BOS_ID = n_phones), context only."""
        return self.n_phones + 1

    @property
    def bos_id(self) -> int:
        return self.n_phones

    @property
    def d_cap(self) -> int:
        return max(self.d_max, self.d_max_sil)

    @property
    def n_offsets(self) -> int:
        return 2 * self.band + 1

    @property
    def d_pad(self) -> int:
        """Padded duration axis of the DP tables: long enough that a band gather cannot alias.

        With ``D_pad >= 2W + 1`` the largest in-band offset step ``d - 1 = 2W`` still has its own
        column, so an out-of-range gather lands on a NEG_INF column instead of on a legal duration
        -- which is what lets both per-frame masks go (see :func:`_band_index`).
        """
        return max(self.d_cap, self.n_offsets + self.recognizer_stride - 1)

    @property
    def n_symbols(self) -> int:
        """Recognizer width: CTC has blank 0; blank-free phones occupy indices 0..K-1."""
        return self.n_phones + (self.topology == "ctc")

    def type_d_max(self, k: int) -> int:
        return self.d_max_sil if k == self.sil_id else self.d_max


# ---------------------------------------------------------------------------------------------------
# the history axis h -- the order of the prior INSIDE the objective
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorHistory:
    """The DP's history axis ``h``: ``|h|``, the successor map ``h' = f(h, k)`` and the repeat symbol.

    The lattice conditions its prior on ``h`` and nothing else, so the ORDER of the phone LM inside
    the objective is exactly this object (user ruling 2026-09-15: a bigram term in the objective is
    not to be launched again; reports/estimate_prior_order_2026-09-15.full.md, D2 and section 6).

    Layout, required of every history so the DP stays cheap::

        h  = outer * n_ctx + last            |h| = n_outer * n_ctx
        next_h[h, k] = succ[last(h)] * n_ctx + k
        last(h) = h % n_ctx                  (the phone a repeat arc re-emits)

    ``last`` is the last emitted phone (``n_phones`` = BOS, which can never be repeated: its
    ``w_sym`` column is NEG_INF) and ``outer`` is whatever the history keeps of the phone BEFORE it.
    The emitted ``k`` always becomes the new ``last``, so ``succ`` -- a map of the 41 context symbols
    onto the ``n_outer`` values -- is the whole of the history rule:

    =============== ========= ================================ ===================
    history         n_outer   succ                             |h| at n_ctx = 41
    =============== ========= ================================ ===================
    ``bigram``      1         ``0``                            41
    ``class_trigram`` C       ``class(l)``                     C * 41
    ``trigram``     n_ctx     ``l``                            1681
    =============== ========= ================================ ===================

    **Blank / repeat / SIL** follow the bigram path exactly (module docstring): only the emit arc
    advances ``h``; blank and repeat leave it untouched; SIL is an ordinary symbol that enters ``h``
    and pays its own prior term each time it is emitted. The repeat rule (a non-SIL ``k`` may not be
    emitted from ``f = 1`` when ``k == last(h)``) is carried by :attr:`same_nonsil`.

    Every table is a CONSTANT of the run (built once, on the CPU, moved to the DP's device by
    :meth:`to`), and ``next`` is materialised in full so the brute-force oracle and the DP read the
    same successor map instead of two copies of the same rule.

    The two batch caps of this module are fitted at the bigram (``bench()`` 2026-09-15); a larger
    ``|h|`` multiplies the resident forward table ``[B, T, O, |h|, 2]`` by ``|h| / 41`` and has to be
    re-measured before it is launched -- :func:`estimate_peak_gib` does NOT know about ``|h|``.
    """

    name: str
    n_outer: int
    n_ctx: int
    start: int  # h of the start state, (BOS, BOS)
    succ: torch.Tensor  # [n_ctx] long: outer index of the successor of a history ending in l
    last: torch.Tensor  # [H] long: the phone h ends in (n_phones = BOS)
    next: torch.Tensor  # [H, K] long: h' = f(h, k), the successor map
    group: torch.Tensor  # [H] long: succ[last(h)], the successor group of h = the outer of h'
    members: torch.Tensor  # [G, S_max] long: the l of each group, padded with the index n_ctx
    dst: torch.Tensor  # [G, K] long: the h' of group g emitting k
    same_nonsil: torch.Tensor  # [1, 1, H, K] bool: the CTC repeat diagonal

    @property
    def n_hist(self) -> int:
        return self.n_outer * self.n_ctx

    @property
    def n_groups(self) -> int:
        """Distinct emit destinations' outer index = ``n_outer`` (``dst[g, k] = g * n_ctx + k``)."""
        return self.n_outer

    def to(self, device) -> "PriorHistory":
        if self.succ.device == device:
            return self
        return PriorHistory(
            self.name, self.n_outer, self.n_ctx, self.start,
            *(t.to(device) for t in
              (self.succ, self.last, self.next, self.group, self.members, self.dst, self.same_nonsil)),
        )


def _prior_history(name: str, cfg: "LatticeConfig", succ, n_outer: int, start_outer: int) -> PriorHistory:
    """Build the tables of a history from its ``succ`` map alone (see :class:`PriorHistory`)."""
    n_ctx, k_n = cfg.n_ctx, cfg.n_phones
    succ = torch.as_tensor(succ, dtype=torch.long).flatten()
    assert succ.shape == (n_ctx,), (
        f"succ must give one outer index for each of the {n_ctx} context symbols (BOS last), got "
        f"{tuple(succ.shape)}"
    )
    assert int(succ.min()) >= 0 and int(succ.max()) < n_outer, (int(succ.min()), int(succ.max()), n_outer)
    assert 0 <= start_outer < n_outer, (start_outer, n_outer)
    h_n = n_outer * n_ctx
    last = torch.arange(h_n) % n_ctx
    group = succ[last]
    k_ax = torch.arange(k_n)
    nxt = group.view(-1, 1) * n_ctx + k_ax.view(1, -1)
    dst = torch.arange(n_outer).view(-1, 1) * n_ctx + k_ax.view(1, -1)
    s_max = max(1, int(torch.bincount(succ, minlength=n_outer).max()))
    members = torch.full((n_outer, s_max), n_ctx, dtype=torch.long)  # n_ctx = the NEG_INF pad column
    for g in range(n_outer):
        idx = (succ == g).nonzero().flatten()
        members[g, : idx.numel()] = idx
    same_nonsil = (last.view(-1, 1) == k_ax.view(1, -1))
    if cfg.topology == "ctc":
        same_nonsil = same_nonsil & (k_ax.view(1, -1) != cfg.sil_id)
    same_nonsil = same_nonsil.view(1, 1, h_n, k_n)
    return PriorHistory(name, int(n_outer), n_ctx, int(start_outer) * n_ctx + cfg.bos_id,
                        succ, last, nxt, group, members, dst, same_nonsil)


def bigram_history(cfg: "LatticeConfig") -> PriorHistory:
    """``h = last emitted phone`` (41 values with BOS) -- the history every stage up to S2c ran."""
    return _prior_history("bigram", cfg, torch.zeros(cfg.n_ctx, dtype=torch.long), 1, 0)


def class_trigram_history(cfg: "LatticeConfig", classes) -> PriorHistory:
    """``h = (class(p_-2), p_-1)``: the affordable order-3 history (estimate 2026-09-15, D2).

    :param classes: one class id per CONTEXT symbol, ``n_ctx`` of them with BOS last. The class of
        BOS is part of the model (``prior.manner8_class_ids()`` gives it its own class, which is the
        convention the held-out class-trigram perplexity was measured under,
        ``analysis/prior_order_ppl.py:220,264,275``); it is never invented here.
    """
    succ = torch.as_tensor(list(classes), dtype=torch.long)
    assert succ.numel() == cfg.n_ctx, (
        f"prior_classes needs one class id per context symbol ({cfg.n_ctx}, BOS last), got "
        f"{succ.numel()}"
    )
    return _prior_history("class_trigram", cfg, succ, int(succ.max()) + 1, int(succ[cfg.bos_id]))


def trigram_history(cfg: "LatticeConfig") -> PriorHistory:
    """``h = (p_-2, p_-1)``, |h| = 1681 at n_ctx = 41.

    CORRECTNESS ONLY: ~14x the step and ~70 GiB at the S2 shape (estimate 2026-09-15, section 1), so
    it exists to let the brute-force oracle check the general path at order 3 on tiny shapes. It is
    not an operating point on this code path.
    """
    return _prior_history("trigram", cfg, torch.arange(cfg.n_ctx), cfg.n_ctx, cfg.bos_id)


def build_prior_history(
    cfg: "LatticeConfig", history: str = "bigram", classes=None
) -> PriorHistory:
    """``history`` in ``{"bigram", "class_trigram", "trigram"}`` -> its :class:`PriorHistory`."""
    if history == "bigram":
        assert classes is None, "the bigram history takes no classes"
        return bigram_history(cfg)
    if history == "class_trigram":
        assert classes is not None, "class_trigram needs prior_classes (one class id per context symbol)"
        return class_trigram_history(cfg, classes)
    if history == "trigram":
        assert classes is None, "the full trigram history takes no classes"
        return trigram_history(cfg)
    raise ValueError(f"unknown prior history {history!r} (bigram / class_trigram / trigram)")


class LatticeOutput(NamedTuple):
    """Everything one DP pass produces. All tensors are detached (the surrogate re-attaches)."""

    log_z: torch.Tensor  # [B] log Z_tau; NEG_INF-floored for a Z = 0 utterance
    z_zero: torch.Tensor  # [B] bool: no path at all -- reported, never averaged in
    post_q: torch.Tensor  # [B, T, n_symbols] arc posteriors per (frame, emitted symbol)
    seg_post: torch.Tensor  # [B, n_phones, d_cap, S+1] arc posteriors per (token, start, duration)
    expected_tokens: torch.Tensor  # [B] expected number of EMITTED tokens (repeats excluded)
    expected_reverse: torch.Tensor  # [B] E[sum_u G[k_u, s_u, d_u]] under the posterior
    expected_prior: torch.Tensor  # [B] E[sum_u log P_psi(k_u | h_u)], the RAW prior term


# ---------------------------------------------------------------------------------------------------
# the segment table G
# ---------------------------------------------------------------------------------------------------


def build_segment_table(
    reverse_model, units: torch.Tensor, eta: torch.Tensor, *, detach: bool = False
) -> torch.Tensor:
    """``G[b, k, d-1, s] = log p_phi(z_s..z_{s+d-1} | k, d, eta) + log p(d | k)``, ``[B, K, D, S+1]``.

    Straight from ``reverse.SegmentalReverseModel``; this module never re-implements the emission or
    the duration head. The returned table is the DIFFERENTIABLE one (it carries the graph to phi);
    the DP detaches and masks its own copy, and the surrogate multiplies this one by ``seg_post``.
    Illegal ``(k, d)`` already carry ``reverse``'s NEG_INF from the pre-softmax duration mask; the
    lattice re-floors them, so the two do not stack into -2e30.
    """
    seg = reverse_model.segment_scores(units, eta)  # [B, K, D, S+1]
    g = seg + reverse_model.duration_log_probs().unsqueeze(0).unsqueeze(-1)
    return g.detach() if detach else g


def _legality_mask(cfg: LatticeConfig, unit_lens: torch.Tensor, s_max: int, device) -> torch.Tensor:
    """``[B, K, D, S+1]`` bool: may a segment of type k, duration d start at reverse frame s?

    ``d_min <= d <= D_k`` and ``s + d <= S_b``. The second half also removes the padded columns of a
    short utterance, whose ``segment_scores`` entries read padded units and are meaningless.
    """
    b = unit_lens.shape[0]
    d_ax = torch.arange(1, cfg.d_cap + 1, device=device).view(1, -1, 1)  # [1, D, 1]
    s_ax = torch.arange(s_max + 1, device=device).view(1, 1, -1)  # [1, 1, S+1]
    d_max_k = torch.tensor(
        [cfg.type_d_max(k) for k in range(cfg.n_phones)], device=device
    ).view(-1, 1, 1)
    dur_ok = (d_ax >= cfg.d_min) & (d_ax <= d_max_k)  # [K, D, 1]
    fits = (s_ax + d_ax) <= unit_lens.to(device).view(b, 1, 1, 1)  # [B, 1, D, S+1]
    return dur_ok.unsqueeze(0) & fits


# ---------------------------------------------------------------------------------------------------
# band index helpers
# ---------------------------------------------------------------------------------------------------


# The blank and the repeat keep s while t advances, so both move the offset o = s - t + W down by
# one: the forward reads ``out[o] = fwd[o + 1]`` and the backward the same arc from its source,
# ``out[o] = bwd[o - 1]``. Both are a slice and a NEG_INF pad, done inline on the [.., H, 2] tensor
# that already holds the two arcs, so the pair costs one cat instead of four kernels.


def _band_index(cfg: LatticeConfig, device) -> Dict[str, torch.Tensor]:
    """Constant index tensors of the band, built once per DP call (never per frame).

    Neither gather needs a companion mask, which is what removes one ``masked_fill`` on a
    ``[B, O, D, K]`` tensor from every frame of every pass:

    * forward, ``(o, d) -> o' = o + d - stride``: an offset step below the minimum duration
      reads the ``d = 1``
      column and a step above ``D`` reads a padded column; both are NEG_INF in ``_scaled_seg_pad``
      (``d_min >= 2``, and ``D_pad >= 2W + 1`` leaves every in-band step its own column);
    * backward, the same arc read from its source: the destination index ``o + d - stride`` runs past the
      band, so the backward table is NEG_INF-padded to ``O + D_pad - 1`` offsets before the gather.
    """
    o_n, d_pad = cfg.n_offsets, cfg.d_pad
    o_src = torch.arange(o_n, device=device).view(-1, 1)
    o_dst = torch.arange(o_n, device=device).view(1, -1)
    d_idx = (o_dst - o_src + cfg.recognizer_stride - 1).clamp(min=0)
    d_ax = torch.arange(d_pad, device=device).view(1, -1)  # = d - 1
    dst_of = o_src + d_ax + 1 - cfg.recognizer_stride
    # The diagonal of the CTC repeat rule lives on the history axis and therefore on the history
    # itself (``PriorHistory.same_nonsil``), not here.
    return {
        "fwd_gather": d_idx.view(1, o_n, 1, o_n, 1),
        "bwd_gather": (o_src + d_ax).view(1, o_n, d_pad, 1),
        # D3: the same two steps with (o_src, o_dst) as the matrix axes -- ``mm_gather`` builds the
        # band matrix, ``mm_dst`` reads the (o, d) layout back out of an (o_src, o_dst) product and
        # ``mm_oob`` is the destination-past-the-band mask that the NEG_INF tail carries there.
        "mm_gather": d_idx.view(1, 1, o_n, o_n),
        "mm_dst": dst_of.clamp(min=0, max=o_n - 1).view(1, 1, o_n, d_pad),
        "mm_oob": ((dst_of < 0) | (dst_of >= o_n)).view(1, 1, o_n, d_pad),
    }


def _arc_weights(
    log_q: torch.Tensor,
    log_q_init: Optional[torch.Tensor],
    temperature: float,
    anchor_weight: float,
    cfg: LatticeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(w_blank [B, T], w_sym [B, T, H])`` -- the per-frame recognizer part of every arc weight.

    ``w_sym`` is padded to H = n_ctx columns with NEG_INF so the BOS history can never repeat-emit.
    """
    scale = 1.0 / float(temperature)
    acoustic = log_q
    if anchor_weight and log_q_init is not None:
        acoustic = acoustic + float(anchor_weight) * log_q_init
    if cfg.topology == "ctc":
        w_blank = scale * acoustic[..., 0]
        w_sym = scale * acoustic[..., 1:]
    else:
        w_blank = acoustic.new_full(acoustic.shape[:-1], NEG_INF)
        w_sym = scale * acoustic
    pad = w_sym.new_full(w_sym.shape[:-1] + (1,), NEG_INF)
    return w_blank, torch.cat([w_sym, pad], dim=-1)


def _group_logsumexp(hk: torch.Tensor, hist: PriorHistory) -> torch.Tensor:
    """``[B, O, H, K] -> [B, O, G, K]``: the emit source mass of each SUCCESSOR GROUP of h.

    All ``h`` of a group send ``k`` to the same ``h'``, so the h axis may be summed away inside a
    group before the band/duration gather -- which is what keeps the gather's cost at G, not H.
    Two stages, because the group of ``h`` depends on ``last(h)`` alone
    (``group = succ[last]``, :class:`PriorHistory`):

    1. sum over ``outer`` -- exact, and the only place the full ``|h|`` axis is reduced;
    2. sum the ``n_ctx`` remaining ``last`` values into their ``G`` groups, by a padded gather whose
       pad column is NEG_INF.

    At ``n_outer = 1`` (the bigram) stage 1 is skipped and stage 2 is the single
    ``logsumexp(hk, dim=2)`` of the original bigram DP -- the same kernel on the same buffer, so the
    bigram path stays bit-for-bit what it was.
    """
    b, o_n, _, k_n = hk.shape
    n_ctx = hist.n_ctx
    x = hk if hist.n_outer == 1 else torch.logsumexp(hk.view(b, o_n, hist.n_outer, n_ctx, k_n), dim=2)
    return _last_logsumexp(x, hist)


def _last_logsumexp(x: torch.Tensor, hist: PriorHistory) -> torch.Tensor:
    """Stage 2 alone: ``[B, O, n_ctx, K] -> [B, O, G, K]``, the ``last`` values of each group.

    Split out of :func:`_group_logsumexp` because the D3 matmul reduction (:func:`_emit_source_mm`)
    reduces ``outer`` INSIDE the GEMM and enters the grouping here with the same tensor stage 1
    would have handed over.
    """
    b, o_n, _, k_n = x.shape
    if hist.n_groups == 1:
        return torch.logsumexp(x, dim=2, keepdim=True)
    sel = torch.cat([x, x.new_full((b, o_n, 1, k_n), NEG_INF)], dim=2)
    sel = sel.index_select(2, hist.members.reshape(-1))
    return torch.logsumexp(sel.view(b, o_n, hist.n_groups, -1, k_n), dim=3)


def _emit_context(
    fwd: torch.Tensor, prior_term: torch.Tensor, hist: PriorHistory
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(hk [B, O, H, K], src [B, O, G, K], both [B, O, H])``: the emit source mass.

    ``hk[b, o, h, k] = fwd[b, o, h, f-allowed] + prior_weight * log P_psi(k | h)`` summed over the
    ALLOWED f (both, except that a non-SIL ``k == last(h)`` may only leave ``f = 0``), and ``src``
    is :func:`_group_logsumexp` of it. ``prior_term`` is already multiplied by ``prior_weight``.
    ``both = logaddexp(fwd_f0, fwd_f1)`` is returned because the blank transition needs exactly the
    same quantity in both passes; recomputing it costs one launch per frame.
    """
    f0 = fwd[..., 0]
    both = torch.logaddexp(f0, fwd[..., 1])
    hk = torch.where(hist.same_nonsil, f0.unsqueeze(-1), both.unsqueeze(-1)) + prior_term
    return hk, _group_logsumexp(hk, hist), both


# ---------------------------------------------------------------------------------------------------
# D3 -- both reductions of a frame are log-semiring MATRIX PRODUCTS, not elementwise expansions
# ---------------------------------------------------------------------------------------------------

# reports/estimate_prior_order_2026-09-15.full.md, D3 and section 1.6. A frame reduces over two
# axes: the source history ``h`` (weight ``log P_psi(k | h)``, a CONSTANT of the whole run) and the
# source band offset ``o`` (weight ``G[k, s, d]``, per frame but HISTORY-FREE). Written
# elementwise each reduction first materialises the axis it is about to sum away --
# ``hk [B, O, |h|, K]`` for the history and ``cand [B, O_src, G, O_dst, K]`` for the band -- so a
# frame moves ``|h| x K`` and ``G x K`` elements and the measured step time is ~linear in ``|h|``
# (bench 2026-09-15: 1.48 s at |h| = 41 vs 11.65 s at |h| = 369).
#
# Written as log-semiring matrix products neither reduced axis is ever materialised: the largest
# per-frame tensor is ``[B, O, n_ctx, K]`` (the bigram's own ``hk``) whatever ``|h|`` is, and the
# number of kernels a frame launches does not grow with ``|h|`` either. Both weight tables enter as
# a matrix:
#
# * history: ``src[b, o, g, k] = logsumexp_{h in g} (fwd[b, o, h] + prior_term[h, k])``. The CTC
#   repeat rule is the only thing that looks like it forbids a matrix, and it does not: the mask
#   ``same_nonsil`` is a constant, and it depends on ``(last(h), k)`` alone, so it commutes with the
#   reduction over ``outer`` and is BAKED INTO the constant operand -- ``P0`` (the prior with the
#   repeat diagonal at NEG_INF) against ``both``, ``P1`` (the diagonal alone, one column per
#   ``last``) against ``f = 0``. No log-subtraction anywhere, so the split is exact.
# * band: ``S[b, k, o_src, o_dst] = seg_pad[b, t + o_src, o_dst - o_src, k]`` is the SAME gather the
#   elementwise path applies to build ``cand``, but used as a matrix instead of being broadcast
#   against the ``G`` axis. The forward contracts ``o_src``, the backward ``o_dst``, and the
#   backward's group contraction ``logsumexp_g (src + bwd)`` is a third product against ``bwd``.
#
# Exactness: ``logmm`` is exact up to the fp32 tail -- a term more than ~87 nats below its row/column
# max underflows to zero instead of being added (the estimate's stated risk). Checked in the source
# against its ``brute_force_log_z`` (not ported) and against the elementwise path at fp32 and fp64 in
# its ``test_lattice.py``.
# The bigram path KEEPS the elementwise reduction by default (``reduction="auto"``), because a GEMM
# reassociates the sum and the 20 banked bit-identity digests of that path must not move.


@contextlib.contextmanager
def _exact_fp32_matmul():
    """Neither TF32 nor autocast may lower the precision of these GEMMs: 10 mantissa bits is ~5e-4
    on an ``exp``, 50x the oracle's tolerance, and the operands are already max-shifted into
    ``[0, 1]`` where a reduced mantissa buys nothing.

    ``_logmm`` accumulates in float64, which neither setting can reach (double is not an
    autocast-eligible dtype and TF32 answers no float64 GEMM), so this context is the second line
    of defence rather than the first. It is still taken around EVERY GEMM of the pass -- the
    forward loop, the three ``_logmm`` products of the backward loop AND the D4 recompute inside it
    (review_lattice_d3d4_2026-09-15, F2) -- because a recomputed frame is bit-identical to the frame
    it replaces only when both saw the same matmul precision: pinning one side of the pass and
    leaving the other on ambient global state would make the posteriors a function of the
    checkpoint stride.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _logmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``out[..., m, p] = logsumexp_n (a[..., m, n] + b[..., n, p])`` -- max-shift, exp, GEMM, log.

    Each operand is shifted by its own max along the CONTRACTED axis, so every exponent is <= 0 and
    the product can neither overflow nor exceed ``n``; an all-NEG_INF row shifts to zero and comes
    back out at NEG_INF through the clamp, exactly as ``torch.logsumexp`` returns NEG_INF + log(n)
    there. That ``NEG_INF`` is FINITE (``psi_align``) is what keeps ``inf - inf`` out of the shift.

    The exp / GEMM / log ACCUMULATION IS ALWAYS float64, whatever the caller's dtype, and only the
    result is cast back. The two shifts are per-operand, so they are not the max of the summand
    ``a[m,n] + b[n,p]``: an output entry whose every term sits more than ``-log(tiny)`` nats below
    ``a_max + b_max`` underflows to zero and is RAISED by the floor below to ``a_max + b_max +
    log(tiny)``. At fp32 that threshold is 87.3 nats, which a production-width lattice crosses on
    ~1 % of entries (up to +60 nats each), and since the forward and the backward group their sums
    differently the two inflate unequally: ``post_q`` row sums came out in [3e-9, 3.9e3] on the
    trigram path (debug_post_q_rowsum_2026-09-15). At float64 the same floor sits 708 nats down --
    below the resolution of any term the DP can still be influenced by -- and the row sums are 1 to
    1e-12. The cost is a float64 GEMM; the alternative, a per-output-entry shift, is exactly the
    reduction the GEMM exists to avoid.

    The product is floored at float64's smallest normal before the log, so no true ``-inf`` is ever
    created: a row whose every term underflowed comes out at ``a_max + b_max - 708`` rather than at
    ``-inf``, which is the same number to every downstream logsumexp and keeps ``0 * inf`` -- a NaN
    -- out of the autograd ORACLE that differentiates this forward on tiny shapes. An impossible
    state still lands at NEG_INF, because its own shift already is NEG_INF.
    """
    acc = torch.float64
    a_max = a.amax(dim=-1, keepdim=True)
    b_max = b.amax(dim=-2, keepdim=True)
    prod = torch.matmul((a - a_max).to(acc).exp(), (b - b_max).to(acc).exp())
    prod = prod.clamp(min=torch.finfo(acc).tiny)
    return (prod.log() + a_max.to(acc) + b_max.to(acc)).clamp(min=NEG_INF).to(a.dtype)


def _emit_matmul_tables(
    hist: PriorHistory,
    prior_term: torch.Tensor,
    raw_prior: Optional[torch.Tensor] = None,
    *,
    collect_stats: bool = False,
) -> Dict[str, torch.Tensor]:
    """The CONSTANT masked prior matrices of the D3 reduction, built ONCE per DP call.

    ``prior_term`` is ``(1/tau) * beta * log P_psi(k | h)`` (``[|h|, K]``, any leading 1-axes), read
    as ``[last, outer, K]`` because ``last`` is the batch axis of every one of these products.

    ============ ====================== ==================================================
    key          shape                  meaning
    ============ ====================== ==================================================
    ``p0``       ``[n_ctx, n_outer, K]``  the prior, NEG_INF on the repeat diagonal (``both``)
    ``p1``       ``[n_ctx, n_outer, 1]``  the diagonal column alone (``f = 0``)
    ``rep``      ``[n_ctx, 1, K]`` bool   which ``(last, k)`` the diagonal is
    ``rmat``     ``[n_ctx, K, n * n_outer]`` the backward's k-contractions, concatenated
    ============ ====================== ==================================================

    ``rmat`` stacks the constant matrices of every k-contraction the backward needs, because they
    all read the SAME left operand (the emit suffix): ``P`` (the f = 0 backward value, every k
    allowed), ``P0`` (the f = 1 value, the repeat forbidden) and, when ``collect_stats``, the two
    blocks of ``expected_prior``. The expectation of a NEGATIVE quantity is taken in the log domain
    by folding ``log(-log P_psi)`` into the matrix -- ``log P_psi <= 0`` for any log-probability
    table, and an exactly-zero row contributes exactly nothing, which is its own correct weight.
    """
    n_ctx, n_outer = hist.n_ctx, hist.n_outer
    k_n = prior_term.shape[-1]
    p = prior_term.reshape(n_outer, n_ctx, k_n).transpose(0, 1).contiguous().clamp(min=NEG_INF)
    rep = hist.same_nonsil.reshape(n_outer, n_ctx, k_n)[0].unsqueeze(1)  # outer-free by construction
    p0 = p.masked_fill(rep, NEG_INF)
    col = torch.arange(n_ctx, device=p.device).clamp(max=k_n - 1).view(n_ctx, 1, 1)
    p1 = p.gather(2, col.expand(n_ctx, n_outer, 1))  # the k = last(h) column; masked off elsewhere
    tab = {"p0": p0, "p1": p1, "rep": rep}
    if raw_prior is not None:
        blocks = [p.transpose(1, 2), p0.transpose(1, 2)]
        if collect_stats:
            neg_log = (-raw_prior.reshape(n_outer, n_ctx, k_n).transpose(0, 1)).clamp(min=0.0)
            n = (p + neg_log.log()).clamp(min=NEG_INF)
            blocks += [n.masked_fill(rep, NEG_INF).transpose(1, 2),
                       n.masked_fill(~rep, NEG_INF).transpose(1, 2)]
        tab["rmat"] = torch.cat(blocks, dim=2).contiguous()
        tab["n_blocks"] = len(blocks)
    return tab


def _emit_source_mm(
    fwd: torch.Tensor, tab: Dict[str, torch.Tensor], hist: PriorHistory
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """D3 history reduction: ``(src [B, O, G, K], both [B, O, H], a_both, a_f0)``, no ``[B,O,H,K]``.

    ``a_both`` / ``a_f0`` are the ``[n_ctx, B * O, n_outer]`` GEMM operands; the backward reuses them
    for ``expected_prior`` instead of permuting a second copy of the same numbers.
    """
    b, o_n, _, _ = fwd.shape
    n_ctx, n_outer = hist.n_ctx, hist.n_outer
    f0 = fwd[..., 0]
    both = torch.logaddexp(f0, fwd[..., 1])
    a_both = both.view(b, o_n, n_outer, n_ctx).permute(3, 0, 1, 2).reshape(n_ctx, b * o_n, n_outer)
    a_f0 = f0.view(b, o_n, n_outer, n_ctx).permute(3, 0, 1, 2).reshape(n_ctx, b * o_n, n_outer)
    x = torch.where(tab["rep"], _logmm(a_f0, tab["p1"]), _logmm(a_both, tab["p0"]))
    x = x.view(n_ctx, b, o_n, -1).permute(1, 2, 0, 3)  # [B, O, last, K]
    return _last_logsumexp(x, hist), both, a_both, a_f0


def _band_matrix(gwin: torch.Tensor, mm_gather: torch.Tensor) -> torch.Tensor:
    """``[B, O, D_pad, K] -> [B * K, O_src, O_dst]``: the frame's band/duration weights as a matrix.

    The same gather the elementwise path folds into ``cand`` (``d - 1 = o_dst - o_src``, a step
    ``<= 0`` reading the illegal ``d = 1`` column, which is NEG_INF because ``d_min >= 2``), but
    laid out so that ``o_src`` and ``o_dst`` are the two matrix axes and ``(b, k)`` is the batch.
    Destinations past the band are not columns here at all: the forward drops them and the backward
    reads NEG_INF there anyway (its table has no state outside the band).
    """
    b, o_n, _, k_n = gwin.shape
    return torch.gather(gwin.permute(0, 3, 1, 2), 3, mm_gather).reshape(b * k_n, o_n, o_n)


# ---------------------------------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------------------------------


def _use_matmul(reduction: str, hist: PriorHistory) -> bool:
    """``reduction`` in ``{"auto", "matmul", "elementwise"}`` -> is the D3 matrix path taken?

    ``"auto"`` (the default everywhere) keeps the ELEMENTWISE reduction at the bigram and takes the
    matrix path at every larger history. The bigram default is a numerical commitment, not a
    performance one: a GEMM reassociates the sum over ``h``, and the 20 banked bit-identity digests
    of the bigram arm (``test_lattice.py`` check 8) are the contract that that arm's floating point
    does not move. ``"matmul"`` takes the matrix path at the bigram too -- the two agree to the
    oracle's tolerance, and the test suite says which one every digest is taken under.
    """
    if reduction == "auto":
        return hist.n_outer > 1
    if reduction not in ("matmul", "elementwise"):
        raise ValueError(f"unknown reduction {reduction!r} (auto / matmul / elementwise)")
    return reduction == "matmul"


def _prior_term(prior_log_bi: torch.Tensor, scale: float, prior_weight: float) -> torch.Tensor:
    """``(1/tau) * beta * log P_psi(k | h)``, with the term SKIPPED (not scaled) at ``beta = 0``.

    ``prior_weight`` (beta) is a CONFIG CONSTANT -- a Python float on ``model.prior_weight``, never a
    tensor -- so the branch is taken on the host, once per DP call, and costs no sync.  At
    ``beta = 0`` (SAE_4A_attrib.md step 5: the odm / prior0 arms run the cycle with the lattice
    prior removed) the arcs lose the prior addend entirely: ``0 * log P_psi`` would be NaN on any
    entry of an UNSMOOTHED table that is ``-inf``, and one NaN arc poisons the whole forward.  The
    fitted tables this phase uses are finite (interpolated Witten-Bell: ``PhoneNgramPriorJob``
    ``TRPE0D5nF3bh`` and ``RtzbESkOedsT`` both have ``log_bi`` >= -21.7 and ``log_tri`` >= -35.5),
    so the guard changes no number for them -- it is what makes ``beta = 0`` safe for any table.
    At ``beta != 0`` the expression is the one that has always run, unchanged.

    The zero tensor is ADDED to the arcs rather than the addition being removed: adding an exact
    zero is bit-for-bit "no prior addend" in every dtype, and it keeps one code path through the
    elementwise and matmul reductions.  The unscaled ``expected_prior`` READOUT is deliberately
    untouched (``lattice.py`` reports the raw table, independent of beta), so it still requires the
    finite table it always did.
    """
    if float(prior_weight) == 0.0:
        return torch.zeros_like(prior_log_bi)
    return scale * float(prior_weight) * prior_log_bi


def _forward_ctx(
    log_q: torch.Tensor,
    log_q_init: Optional[torch.Tensor],
    seg_pad: torch.Tensor,
    prior_log_bi: torch.Tensor,
    feat_lens: torch.Tensor,
    cfg: LatticeConfig,
    hist: PriorHistory,
    *,
    temperature: float,
    anchor_weight: float,
    prior_weight: float,
    use_mm: bool,
    backward: bool = False,
    collect_stats: bool = False,
) -> Dict:
    """Everything a frame of the recursion needs that does NOT depend on ``t``, built once.

    The DP is launch-bound, so every constant is hoisted out of the loop: the ``[B, T]`` length
    masks, the band index tensors, the blank/repeat weight table and -- on the matmul path -- the
    constant masked prior matrices of D3. Both passes share this object, and so does the forward
    that D4 recomputes inside the backward, which is what makes a recomputed frame bit-identical to
    the frame it replaces.

    ``w_pair`` is ``[B, T, n_ctx, 2]``, NOT ``[B, T, |h|, 2]``: the blank weight is history-free and
    the repeat weight depends on ``last(h)`` alone, so the add broadcasts over ``outer`` instead of
    carrying a tiled copy (1.2 GiB per pass at |h| = 1681, B = 125, T = 704). At the bigram
    ``n_outer = 1`` and the tensor, the broadcast and every bit of the result are what they were.
    """
    device = log_q.device
    b, t_max, _ = log_q.shape
    o_n, h_n, k_n = cfg.n_offsets, hist.n_hist, cfg.n_phones
    scale = 1.0 / float(temperature)
    idx = _band_index(cfg, device)
    w_blank, w_sym = _arc_weights(log_q, log_q_init, temperature, anchor_weight, cfg)
    prior_term = _prior_term(prior_log_bi, scale, prior_weight).view(1, 1, h_n, k_n)
    steps = torch.arange(t_max, device=device)
    lens = feat_lens.to(device)
    cx = {
        "hist": hist, "b": b, "t_max": t_max, "o_n": o_n, "h_n": h_n, "k_n": k_n,
        "g_n": hist.n_groups, "n_outer": hist.n_outer, "n_ctx": hist.n_ctx, "w": cfg.band,
        "d_pad": cfg.d_pad, "seg_pad": seg_pad, "w_sym": w_sym, "prior_term": prior_term,
        "dst_flat": hist.dst.reshape(-1),
        "active_all": (steps.view(1, -1) < lens.view(-1, 1)).view(b, t_max, 1, 1, 1),
        "done_all": lens.view(-1, 1) == (steps + 1).view(1, -1),
        "end_frames": {int(v) for v in feat_lens.tolist()},
        # blank keeps h and clears f, repeat keeps both: one weight table lets the two share a
        # single shift and a single add per frame. The repeat re-emits ``last(h)``.
        "w_pair": torch.stack(
            [w_blank.unsqueeze(-1).expand(b, t_max, hist.n_ctx), w_sym[:, :, : hist.n_ctx]], dim=-1
        ),
        "shift_tail": log_q.new_full((b, cfg.recognizer_stride, h_n, 2), NEG_INF),
        "stride": cfg.recognizer_stride,
        "neg": log_q.new_full((), NEG_INF),
        "idx": idx,
        "mm": None,
    }
    if use_mm:
        cx["mm"] = _emit_matmul_tables(
            hist, prior_term, prior_log_bi if backward else None, collect_stats=collect_stats
        )
        cx["mm_gather"] = idx["mm_gather"].expand(b, k_n, o_n, o_n)
    else:
        cx["fwd_gather"] = idx["fwd_gather"].expand(b, o_n, hist.n_groups, o_n, k_n)
    return cx


def _forward_step(fwd: torch.Tensor, t: int, cx: Dict) -> torch.Tensor:
    """One frame of the forward recursion, ``fwd_t -> fwd_{t+1}``. The DP's only hot loop body.

    Vectorized over ``(B, band, h, f) x (k, d)``; the emit arc reduces the history and the band, the
    blank and the repeat share one shift and one add. ``cx["mm"]`` selects the D3 matrix products
    over the elementwise expansion -- same lattice, same semantics, different associativity.
    """
    b, o_n, h_n, k_n = cx["b"], cx["o_n"], cx["h_n"], cx["k_n"]
    hist, g_n = cx["hist"], cx["g_n"]
    gwin = cx["seg_pad"][:, t * cx["stride"] : t * cx["stride"] + o_n]
    w_sym_t = cx["w_sym"][:, t, :k_n].view(b, 1, 1, k_n)
    if cx["mm"] is None:
        _, src, both = _emit_context(fwd, cx["prior_term"], hist)
        # --- emit: (o, d) -> o' = o + d - 1, one gather + one logsumexp over the source offset.
        # No mask: the padded d axis turns every out-of-band step into a NEG_INF column.
        cand = torch.gather(src.unsqueeze(3) + gwin.unsqueeze(2), 3, cx["fwd_gather"])
        emit_g = torch.logsumexp(cand, dim=1) + w_sym_t  # [B, G, O_dst, K]
        emit_flat = emit_g.transpose(1, 2).reshape(b, o_n, g_n * k_n)
    else:
        src, both, _, _ = _emit_source_mm(fwd, cx["mm"], hist)
        # the band reduction as a product: [B*K, G, O_src] x [B*K, O_src, O_dst]
        u = src.permute(0, 3, 2, 1).reshape(b * k_n, g_n, o_n)
        emit_g = _logmm(u, _band_matrix(gwin, cx["mm_gather"])).view(b, k_n, g_n, o_n)
        emit_flat = (emit_g.permute(0, 3, 2, 1) + w_sym_t).reshape(b, o_n, g_n * k_n)
    # (g, k) -> h' = g * n_ctx + k: a placement, never an accumulation (the map is injective).
    # Every h' outside the image -- the histories ending in BOS -- stays NEG_INF, which is the
    # "BOS is never emitted" column of the bigram path.
    emit = fwd.new_full((b, o_n, h_n), NEG_INF)
    emit.index_copy_(2, cx["dst_flat"], emit_flat)
    # --- blank (from either f) and repeat (from f = 1 only); neither advances s
    prev = torch.stack([both, fwd[..., 1]], dim=-1)  # [B, O, H, 2] = (blank src, repeat src)
    nxt = (
        torch.cat([prev[:, cx["stride"]:], cx["shift_tail"]], dim=1).view(b, o_n, cx["n_outer"], cx["n_ctx"], 2)
        + cx["w_pair"][:, t].view(b, 1, 1, cx["n_ctx"], 2)
    ).view(b, o_n, h_n, 2)
    nxt = torch.cat(
        [nxt[..., :1], torch.logaddexp(nxt[..., 1:], emit.unsqueeze(-1))], dim=-1
    ).clamp(min=NEG_INF)
    return torch.where(cx["active_all"][:, t], nxt, cx["neg"])


def scaled_seg_pad(
    seg_table: torch.Tensor, unit_lens: torch.Tensor, cfg: LatticeConfig, temperature: float
) -> torch.Tensor:
    """``[B, K, D, S+1]`` -> ``[B, S+1+2W, D_pad, K]``: the layout the per-frame band slice reads.

    Slice ``[:, stride*t : stride*t + 2W + 1]`` is ``s = stride*t - W .. stride*t + W``; the d
    axis is padded to ``D_pad`` so an out-of-band gather cannot alias a legal duration
    (:func:`_band_index`). Illegal ``(k, d, s)`` are NEG_INF (:func:`_legality_mask`).

    EVERY term of the summand carries the ``1/tau`` of the objective, the reverse table and the
    prior included -- not just the recognizer's (module docstring).

    This is the single most expensive constant of a DP call (~0.28 GiB at B = 64, T = S = 500), so
    it is built ONCE per call in :func:`lattice_forward_backward` and handed to
    :func:`forward_log_z` (review_s0b_core_2026-09-15.md, CONCERN 4b).
    """
    s_max = seg_table.shape[-1] - 1
    seg = ((1.0 / float(temperature)) * seg_table).masked_fill(
        ~_legality_mask(cfg, unit_lens, s_max, seg_table.device), NEG_INF
    )
    b, k, d, s1 = seg.shape
    w = cfg.band
    out = seg.new_full((b, s1 + 2 * w, cfg.d_pad, k), NEG_INF)
    out[:, w : w + s1, :d] = seg.permute(0, 3, 2, 1)
    return out


def forward_log_z(
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
    table_out: Optional[torch.Tensor] = None,
    seg_pad: Optional[torch.Tensor] = None,
    history: Optional[PriorHistory] = None,
    reduction: str = "auto",
    checkpoint: int = 0,
    ctx: Optional[Dict] = None,
) -> torch.Tensor:
    """Exact banded log-semiring forward -> ``log_z [B]``.

    :param log_q: ``[B, T, n_symbols]`` recognizer log-probs in ``cfg``'s symbol order.
    :param seg_table: ``[B, K, D, S+1]`` from :func:`build_segment_table` (RAW; masked here).
    :param prior_log_bi: ``[|h|, K]`` ``log P_psi(k | h)`` over the rows of ``history``; at the
        default bigram history that is the ``[41, 40]`` ``log P_psi(k | h)``, h = 0..n_phones.
    :param history: the history axis h (:class:`PriorHistory`); ``None`` = the bigram.
    :param table_out: optional ``[B, T, O, H, 2]`` buffer that receives ``fwd_t`` for t = 0..T-1.
        Only the manual backward needs it, and it must be filled under ``no_grad``.
    :param seg_pad: optional prebuilt :func:`scaled_seg_pad` output, so a caller that needs it
        afterwards (the manual backward) does not pay for a second copy.
    :param reduction: which reduction over ``h`` and over the band offset (:func:`_use_matmul`).
    :param checkpoint: D4. ``0`` fills ``table_out`` at every frame (``[B, T, O, H, 2]``); ``S > 0``
        fills it only at ``t = 0, S, 2S, ...`` (``[B, ceil(T/S), O, H, 2]``), and the backward
        recomputes the frames in between. Nothing else about the pass changes.
    :param ctx: an already-built :func:`_forward_ctx` (the manual backward hands its own in, so the
        arc weights and the constant tables are built once per DP call, not twice).

    With ``table_out=None`` and grad-carrying inputs this is the AUTOGRAD ORACLE of the test suite;
    production always calls it under ``no_grad`` from :func:`lattice_forward_backward`.
    """
    device = log_q.device
    b, t_max, n_sym = log_q.shape
    assert n_sym == cfg.n_symbols, f"log_q has {n_sym} symbols, cfg wants {cfg.n_symbols}"
    s_max = seg_table.shape[-1] - 1
    assert seg_table.shape[1:3] == (cfg.n_phones, cfg.d_cap), seg_table.shape
    assert int(feat_lens.max()) <= t_max and int(unit_lens.max()) <= s_max
    if table_out is not None:
        assert not torch.is_grad_enabled(), "table_out is an in-place buffer; fill it under no_grad"

    hist = (history or bigram_history(cfg)).to(device)
    o_n, h_n, k_n, w = cfg.n_offsets, hist.n_hist, cfg.n_phones, cfg.band
    assert prior_log_bi.shape == (h_n, k_n), (
        f"the prior table is {tuple(prior_log_bi.shape)}, the {hist.name} history wants "
        f"({h_n}, {k_n})"
    )
    if seg_pad is None:
        seg_pad = scaled_seg_pad(seg_table, unit_lens, cfg, temperature)
    # Every constant of the loop is hoisted into the context: the DP is launch-bound, and a
    # per-frame host sync (``bool(x.any())``) or a per-frame allocation costs more than the
    # arithmetic. The length tests are [B, T] masks the loop only INDEXES, and the end-of-utterance
    # test is answered on the HOST from a set of lengths, so the loop never blocks on the device.
    cx = ctx or _forward_ctx(
        log_q, log_q_init, seg_pad, prior_log_bi, feat_lens, cfg, hist,
        temperature=temperature, anchor_weight=anchor_weight, prior_weight=prior_weight,
        use_mm=_use_matmul(reduction, hist),
    )
    ck = int(checkpoint)
    done_all, end_frames = cx["done_all"], cx["end_frames"]

    end_offsets = (unit_lens.to(device) - cfg.recognizer_stride * feat_lens.to(device) + w).long()
    batch_index = torch.arange(b, device=device)
    fwd = log_q.new_full((b, o_n, h_n, 2), NEG_INF)
    fwd[:, w, hist.start, 0] = 0.0  # start (t = 0, s = 0, h = (BOS, BOS), f = 0)
    log_z = log_q.new_full((b,), NEG_INF)

    with _exact_fp32_matmul() if cx["mm"] is not None else contextlib.nullcontext():
        for t in range(t_max):
            if table_out is not None and (ck <= 0 or t % ck == 0):
                table_out[:, t if ck <= 0 else t // ck] = fwd
            fwd = _forward_step(fwd, t, cx)
            # --- final read at t + 1 == T_b: s = S_b = T_b means offset W
            if (t + 1) in end_frames:
                log_z = torch.where(
                    done_all[:, t], torch.logsumexp(fwd[batch_index, end_offsets].reshape(b, -1), dim=1), log_z
                )
    return log_z.clamp(min=NEG_INF)


# ---------------------------------------------------------------------------------------------------
# manual backward
# ---------------------------------------------------------------------------------------------------


@torch.no_grad()
def lattice_forward_backward(
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
    collect_stats: bool = True,
    enforce_budget: bool = True,
    history: Optional[PriorHistory] = None,
    reduction: str = "auto",
    checkpoint: int = 0,
) -> LatticeOutput:
    """Forward + MANUAL log-semiring backward -> arc posteriors and the per-utterance ``log Z``.

    Batched over utterances, one Python loop over t and nothing else (user ruling 2026-09-15: make
    full use of the GH200); everything inside a frame is vectorized over
    ``(B, band, h, f) x (k, d)``. The ``[B, T, O, H, 2]`` forward table stays resident (~8.4 MB per
    utterance at T = 500, W = 25) and the per-frame transition tensors are transient -- which is the
    whole reason the backward is written by hand.

    ``collect_stats`` adds one ``[B, O, H, K]`` exp per frame for ``expected_prior`` (the h-resolved
    emit posterior). Switch it off to save ~25 % of the backward when the prior term is not logged.

    ``history`` (``None`` = the bigram) is the history axis h; ``prior_log_bi`` must be its
    ``[|h|, K]`` table. Only the forward table and the emit intermediates carry ``|h|``: the segment
    posteriors, the band and the duration axis are history-free.

    ``reduction`` (:func:`_use_matmul`) chooses how a frame reduces over ``h`` and over the band
    offset -- elementwise (the bigram default, bit-identical to the DP this module shipped with) or
    D3's log-semiring matrix products, which is what keeps the per-frame traffic at ``|h|`` instead
    of ``|h| x K``. ``checkpoint = S > 0`` is D4: the resident forward table holds every S-th frame
    and the backward recomputes the S - 1 frames in between with :func:`_forward_step`, which is the
    same function the forward ran, so a recomputed frame is bit-identical to the one it replaces.

    The batch must respect :func:`check_batch_budget`: resident memory is ``B * T_max``, not frames.
    ``enforce_budget=False`` downgrades that check to a warning and exists for :func:`bench`, which
    is what the caps are fitted on; a training step never sets it.
    """
    log_q = log_q.detach()
    seg_table = seg_table.detach()
    prior_log_bi = prior_log_bi.detach()
    if log_q_init is not None:
        log_q_init = log_q_init.detach()
    device = log_q.device
    b, t_max, _ = log_q.shape
    s_max = seg_table.shape[-1] - 1
    hist = (history or bigram_history(cfg)).to(device)
    o_n, h_n, k_n, d_cap, d_pad, w = (
        cfg.n_offsets, hist.n_hist, cfg.n_phones, cfg.d_cap, cfg.d_pad, cfg.band
    )
    g_n, n_outer, n_ctx = hist.n_groups, hist.n_outer, hist.n_ctx
    check_batch_budget(b, t_max, strict=enforce_budget)
    use_mm = _use_matmul(reduction, hist)
    ck = int(checkpoint)
    assert ck >= 0, "checkpoint is a frame stride S >= 0 (0 = keep the whole forward table)"

    # ONE padded segment table for both passes (~0.28 GiB at B = 64, T = S = 500); the forward used
    # to build its own copy (review_s0b_core_2026-09-15.md, CONCERN 4b). The frame context is built
    # once here and handed to the forward, so the arc weights, the band index and D3's constant
    # matrices exist once per DP call -- and the D4 recompute below runs the SAME context.
    seg_pad = scaled_seg_pad(seg_table, unit_lens, cfg, temperature)
    cx = _forward_ctx(
        log_q, log_q_init, seg_pad, prior_log_bi, feat_lens, cfg, hist,
        temperature=temperature, anchor_weight=anchor_weight, prior_weight=prior_weight,
        use_mm=use_mm, backward=True, collect_stats=collect_stats,
    )
    # D4: ``[B, T, O, |h|, 2]`` is 1.37 GiB at the bigram and 56 GiB at the full trigram (B = 125,
    # T = 704), so at ``ck > 0`` only every ck-th frame is resident and the backward recomputes the
    # chunk it is inside. One extra forward pass, no change to a single value.
    n_store = t_max if ck == 0 else (t_max + ck - 1) // ck
    table = log_q.new_full((b, n_store, o_n, h_n, 2), NEG_INF)
    log_z = forward_log_z(
        log_q, seg_table, prior_log_bi, feat_lens, unit_lens, cfg,
        temperature=temperature, anchor_weight=anchor_weight, prior_weight=prior_weight,
        log_q_init=log_q_init, table_out=table, seg_pad=seg_pad, history=hist,
        checkpoint=ck, ctx=cx,
    )
    chunk = None if ck == 0 else log_q.new_full((b, ck, o_n, h_n, 2), NEG_INF)
    chunk_id = -1
    z_zero = log_z <= NEG_INF / 2
    # A Z = 0 utterance has no arc with finite (fwd + bwd), so every mass below is exactly 0; the
    # substitute log Z only keeps (-1e30) - (-1e30) out of the exponent.
    log_z_safe = torch.where(z_zero, torch.zeros_like(log_z), log_z).view(b, 1, 1, 1)
    log_z_emit = log_z_safe.unsqueeze(-1)  # the emit tensors carry the extra group axis

    idx = cx["idx"]
    w_sym, prior_term, w_pair = cx["w_sym"], cx["prior_term"], cx["w_pair"]
    raw_prior = prior_log_bi.view(1, 1, h_n, k_n)  # the REPORTED prior term is unscaled
    dst_flat, active_all, neg = cx["dst_flat"], cx["active_all"], cx["neg"]
    done_all = cx["done_all"].view(b, t_max, 1, 1, 1)
    end_frames = cx["end_frames"]
    if use_mm:
        mm, mm_gather = cx["mm"], cx["mm_gather"]
        mm_dst = idx["mm_dst"].expand(b, k_n, o_n, d_pad)
        mm_oob = idx["mm_oob"]
        log_z_k = log_z_safe.view(b, 1, 1, 1)
        log_z_bo = log_z_safe.view(1, b, 1, 1)
    else:
        bwd_gather = idx["bwd_gather"].expand(b, o_n, d_pad, g_n * k_n)
        bwd_head = log_q.new_full((b, cfg.recognizer_stride - 1, g_n * k_n), NEG_INF)
        bwd_tail = log_q.new_full((b, d_pad - cfg.recognizer_stride, g_n * k_n), NEG_INF)
        o_ext = o_n + d_pad - 1
    shift_head = log_q.new_full((b, cfg.recognizer_stride, h_n, 2), NEG_INF)
    final_state = log_q.new_full((b, o_n, h_n, 2), NEG_INF)
    end_offsets = (unit_lens.to(device) - cfg.recognizer_stride * feat_lens.to(device) + w).long()
    final_state[torch.arange(b, device=device), end_offsets] = 0.0

    post_q = log_q.new_zeros((b, t_max, cfg.n_symbols))
    seg_post_pad = log_q.new_zeros((b, s_max + 1 + 2 * w, d_cap, k_n))
    expected_tokens = log_q.new_zeros(b)
    expected_reverse = log_q.new_zeros(b)
    expected_prior = log_q.new_zeros(b)
    bwd = log_q.new_full((b, o_n, h_n, 2), NEG_INF)

    # F2 (review_lattice_d3d4_2026-09-15): the backward's three GEMMs and the D4 recompute
    # below are pinned to the SAME matmul precision the forward loop ran under -- a recomputed
    # frame is bit-identical to the frame it replaces only if both saw it.
    with _exact_fp32_matmul() if use_mm else contextlib.nullcontext():
        for t in range(t_max - 1, -1, -1):
            # final condition injected exactly at the frame after this utterance's last one (host test)
            if (t + 1) in end_frames:
                bwd = torch.where(done_all[:, t], final_state, bwd)
            if ck == 0:
                fwd = table[:, t]
            else:  # D4: recompute this chunk's forward from its checkpoint, once per chunk
                if t // ck != chunk_id:
                    chunk_id = t // ck
                    f = table[:, chunk_id]
                    for j in range(chunk_id * ck, min((chunk_id + 1) * ck, t_max)):
                        chunk[:, j - chunk_id * ck] = f
                        f = _forward_step(f, j, cx)
                fwd = chunk[:, t - chunk_id * ck]
            gwin = seg_pad[:, t * cfg.recognizer_stride : t * cfg.recognizer_stride + o_n]
            if not use_mm:
                hk, src, both = _emit_context(fwd, prior_term, hist)
                # --- emit arcs: destination offset o + d - 1, flag f = 1, history h' = g * n_ctx + k
                # read through the same injective (g, k) map the forward scatters with. The offset axis
                # is NEG_INF-padded to O + D_pad - 1, so the gather needs neither a clamp nor a mask.
                bwd_ext = torch.cat([bwd_head, bwd[..., 1].index_select(2, dst_flat), bwd_tail], dim=1)
                bwd_at_dst = torch.gather(
                    bwd_ext.unsqueeze(2).expand(b, o_ext, d_pad, g_n * k_n), 1, bwd_gather
                ).view(b, o_n, d_pad, g_n, k_n)
                gb = gwin.unsqueeze(3) + bwd_at_dst + w_sym[:, t, :k_n].view(b, 1, 1, 1, k_n)
                # [B, O, D_pad, G, K]; the group axis is summed out at once for every history-free stat
                mass = torch.exp((src.unsqueeze(2) + gb - log_z_emit).clamp(max=0.0))
                mass_k = mass.sum(dim=3)  # [B, O, D_pad, K]
                seg_post_pad[:, t * cfg.recognizer_stride : t * cfg.recognizer_stride + o_n] += mass_k[:, :, :d_cap]
                post_q[:, t, cfg.n_symbols - k_n :] += mass_k.sum(dim=(1, 2))
                expected_tokens += mass_k.sum(dim=(1, 2, 3))
                expected_reverse += (mass_k * gwin).sum(dim=(1, 2, 3))
                suffix = torch.logsumexp(gb, dim=2)  # [B, O, G, K] emit suffix, h not yet resolved
                # resolve h: every h of a group shares its suffix (broadcast at the bigram, G = 1)
                suffix_h = suffix if g_n == 1 else suffix.index_select(2, hist.group)
                if collect_stats:
                    hk_mass = torch.exp((hk + suffix_h - log_z_safe).clamp(max=0.0))
                    expected_prior += (hk_mass * raw_prior).sum(dim=(1, 2, 3))
                # --- backward values
                z1 = suffix_h + prior_term  # [B, O, H, K]
                bwd_emit_f0 = torch.logsumexp(z1, dim=3)
                bwd_emit_f1 = torch.logsumexp(z1.masked_fill(hist.same_nonsil, NEG_INF), dim=3)
            else:
                # D3, the same three reductions as matrix products. The destination history (g, k) IS
                # the destination state, so ``bd`` is the backward table re-indexed by (o_dst, g, k) and
                # every product below contracts one axis of it: the band (o_dst) for the emit suffix,
                # the group (g) for the arc mass, and k for the two backward values.
                src, both, a_both, a_f0 = _emit_source_mm(fwd, mm, hist)
                bd = bwd[..., 1].index_select(2, dst_flat).view(b, o_n, g_n, k_n)  # [B, O_dst, G, K]
                bd_k = bd.permute(0, 3, 1, 2).reshape(b * k_n, o_n, g_n)  # [B*K, O_dst, G]
                sm = _band_matrix(gwin, mm_gather)  # [B*K, O_src, O_dst]
                suffix = _logmm(sm, bd_k).view(b, k_n, o_n, g_n).permute(0, 2, 3, 1) + w_sym[
                    :, t, :k_n
                ].view(b, 1, 1, k_n)  # [B, O, G, K] emit suffix, h not yet resolved
                # the arc mass needs sum_g exp(src + bwd), which is that logsumexp read on the (o, d)
                # diagonal of an (o_src, o_dst) product; destinations past the band are NEG_INF there
                # exactly as the backward table's own pad is on the elementwise path.
                lg = _logmm(src.permute(0, 3, 1, 2).reshape(b * k_n, o_n, g_n), bd_k.transpose(1, 2))
                lg = torch.gather(lg.view(b, k_n, o_n, o_n), 3, mm_dst).masked_fill(mm_oob, NEG_INF)
                gwin_k = gwin.permute(0, 3, 1, 2)  # [B, K, O, D_pad]
                mass_k = torch.exp(
                    (lg + gwin_k + w_sym[:, t, :k_n].view(b, k_n, 1, 1) - log_z_k).clamp(max=0.0)
                )
                seg_post_pad[:, t * cfg.recognizer_stride : t * cfg.recognizer_stride + o_n] += mass_k[:, :, :, :d_cap].permute(0, 2, 3, 1)
                post_q[:, t, cfg.n_symbols - k_n :] += mass_k.sum(dim=(2, 3))
                expected_tokens += mass_k.sum(dim=(1, 2, 3))
                expected_reverse += (mass_k * gwin_k).sum(dim=(1, 2, 3))
                # one product answers every k-contraction: the two backward values and, with
                # ``collect_stats``, the two blocks of expected_prior (log(-log P) folded into the
                # matrix, so a NEGATIVE expectation is still taken in the log domain)
                blk = _logmm(
                    suffix.index_select(2, hist.succ).permute(2, 0, 1, 3).reshape(n_ctx, b * o_n, k_n),
                    mm["rmat"],
                ).view(n_ctx, b, o_n, mm["n_blocks"], n_outer)
                if collect_stats:
                    expected_prior -= (
                        torch.exp(a_both.view(n_ctx, b, o_n, n_outer) + blk[:, :, :, 2] - log_z_bo)
                        + torch.exp(a_f0.view(n_ctx, b, o_n, n_outer) + blk[:, :, :, 3] - log_z_bo)
                    ).sum(dim=(0, 2, 3))
                bwd_emit_f0 = blk[:, :, :, 0].permute(1, 2, 3, 0).reshape(b, o_n, h_n)
                bwd_emit_f1 = blk[:, :, :, 1].permute(1, 2, 3, 0).reshape(b, o_n, h_n)
            # blank and repeat read the SAME shift of bwd (out[o] = bwd[o - 1]) and the same weight
            # table as the forward: one cat and one add for both arcs.
            back = (
                torch.cat([shift_head, bwd[:, :-cfg.recognizer_stride]], dim=1).view(b, o_n, n_outer, n_ctx, 2)
                + w_pair[:, t].view(b, 1, 1, n_ctx, 2)
            ).view(b, o_n, h_n, 2)
            from_blank, from_repeat = back[..., 0], back[..., 1]
            # --- blank / repeat posteriors (the repeat NEVER touches seg_post: it does not advance s)
            if cfg.topology == "ctc":
                post_q[:, t, 0] += torch.exp(
                    (both + from_blank - log_z_safe[..., 0]).clamp(max=0.0)
                ).sum(dim=(1, 2))
            rep = torch.exp((fwd[..., 1] + from_repeat - log_z_safe[..., 0]).clamp(max=0.0))
            # a repeat emits ``last(h)``: sum the histories that share one last phone (a no-op at the
            # bigram, where ``outer`` has a single value)
            rep_sym = rep if n_outer == 1 else rep.view(b, o_n, n_outer, hist.n_ctx).sum(dim=2)
            post_q[:, t, cfg.n_symbols - k_n :] += rep_sym[..., :k_n].sum(dim=1)
            nxt = torch.stack(
                [
                    torch.logaddexp(from_blank, bwd_emit_f0),
                    torch.logaddexp(torch.logaddexp(from_blank, from_repeat), bwd_emit_f1),
                ],
                dim=-1,
            ).clamp(min=NEG_INF)
            bwd = torch.where(active_all[:, t], nxt, neg)

    seg_post = seg_post_pad[:, w : w + s_max + 1].permute(0, 3, 2, 1).contiguous()
    expected_reverse = expected_reverse * float(temperature)  # undo the 1/tau carried by seg_pad
    return LatticeOutput(
        log_z=log_z,
        z_zero=z_zero,
        post_q=post_q,
        seg_post=seg_post,
        expected_tokens=expected_tokens,
        expected_reverse=expected_reverse,
        expected_prior=expected_prior,
    )


def lattice_loss(
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
    collect_stats: bool = True,
    enforce_budget: bool = True,
    history: Optional[PriorHistory] = None,
    reduction: str = "auto",
    checkpoint: int = 0,
) -> Tuple[torch.Tensor, LatticeOutput]:
    """``(loss [B] = -log Z_tau, the DP output)``, differentiable in ``log_q`` and ``seg_table``.

    The value is exactly ``-log Z_tau``; the gradient comes from the arc-posterior surrogate
    ``-(1/tau) * (sum post_q * log_q + sum seg_post * seg_table)``, which is ``dL/dtheta`` and
    ``dL/dphi`` (module docstring). Nothing back-propagates through the DP itself.

    A ``Z = 0`` utterance gets loss 0 and zero gradient here; the caller reads ``out.z_zero``, keeps
    it out of the mean and reports the fraction (never a -1e30 folded into the batch loss).
    """
    out = lattice_forward_backward(
        log_q, seg_table, prior_log_bi, feat_lens, unit_lens, cfg,
        temperature=temperature, anchor_weight=anchor_weight, prior_weight=prior_weight,
        log_q_init=log_q_init, collect_stats=collect_stats, enforce_budget=enforce_budget,
        history=history, reduction=reduction, checkpoint=checkpoint,
    )
    keep = (~out.z_zero).to(log_q.dtype)
    scale = 1.0 / float(temperature)
    surrogate = -scale * (
        (out.post_q * log_q).sum(dim=(1, 2)) + (out.seg_post * seg_table).sum(dim=(1, 2, 3))
    )
    value = torch.where(out.z_zero, torch.zeros_like(out.log_z), -out.log_z)
    loss = surrogate * keep + (value - surrogate * keep).detach()
    return loss, out


# ---------------------------------------------------------------------------------------------------
# benchmark (efficiency read before any large-scale training, user ruling 2026-09-15)
# ---------------------------------------------------------------------------------------------------


def bench(
    *,
    batch: int = 16,
    t_len: int = 500,
    device: str = "cuda",
    cfg: Optional[LatticeConfig] = None,
    temperature: float = 2.0,
    anchor_weight: float = 0.0,
    collect_stats: bool = True,
    warmup: int = 1,
    repeats: int = 3,
    history: str = "bigram",
    n_classes: Optional[int] = None,
    classes: Optional[Sequence[int]] = None,
    reduction: str = "auto",
    checkpoint: int = 0,
    stages: bool = False,
) -> Dict[str, float]:
    """Time forward + manual backward + the surrogate's autograd at the real S2 shape.

    Defaults are the reference shape of the brief (orchestrator 2026-09-15): B = 16 utterances,
    T = S = 500 frames (10 s at 50 Hz), W = 25, K = 40 phones, D = 25 / D_sil = 50, tau = 2. Reports
    ms per batch, utterance-frames per second and the peak allocation -- which is what the wall-time
    projection, the memory rung and the two caps above are read from. Sweep ``--batch`` to refit
    them; the budget check only warns here.
    """
    import time

    cfg = cfg or LatticeConfig()
    check_batch_budget(batch, t_len, strict=False)  # a bench MAY probe past the cap; it says so
    # The class map sets the SHAPE of the history here (a bench measures a shape, never a model).
    # ``classes`` (e.g. ``prior.manner8_class_ids()``) benches the real partition -- its class SIZES
    # matter, since the grouped reduction pads to the largest class; ``n_classes`` alone falls back
    # to an equal-sized round robin over the symbols with BOS in its own class, which is the layout
    # but not the size profile.
    if history == "class_trigram" and classes is None:
        assert n_classes, "class_trigram needs classes or n_classes (the history shape to bench)"
        classes = [i % int(n_classes) for i in range(cfg.n_phones)] + [int(n_classes)]
    hist = build_prior_history(cfg, history, classes)
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    log_q = torch.log_softmax(
        torch.randn(batch, t_len, cfg.n_symbols, generator=gen), dim=-1
    ).to(dev).requires_grad_(True)
    log_q_init = (
        torch.log_softmax(torch.randn(batch, t_len, cfg.n_symbols, generator=gen), dim=-1).to(dev)
        if anchor_weight else None
    )
    seg = (
        torch.randn(batch, cfg.n_phones, cfg.d_cap, t_len + 1, generator=gen) * 0.5 - 2.0
    ).to(dev).requires_grad_(True)
    prior = torch.log_softmax(
        torch.randn(hist.n_hist, cfg.n_phones, generator=gen), dim=-1
    ).to(dev)
    lens = torch.full((batch,), t_len, dtype=torch.long, device=dev)

    def one():
        loss, out = lattice_loss(
            log_q, seg, prior, lens, lens, cfg, temperature=temperature,
            anchor_weight=anchor_weight, log_q_init=log_q_init, collect_stats=collect_stats,
            enforce_budget=False, history=hist, reduction=reduction, checkpoint=checkpoint,
        )
        loss.sum().backward()
        return out

    def timed(fn) -> float:
        """ms per call, with a cuda.synchronize on both sides (never a host-side average)."""
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(repeats):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / repeats * 1e3

    stage_ms = {}
    if stages:
        # where the step's time goes: the forward alone, then + the manual backward, then + the
        # surrogate's autograd (``one()`` below). Each is its own synchronized section.
        dp_args = (log_q.detach(), seg.detach(), prior, lens, lens, cfg)
        dp_kw = dict(temperature=temperature, anchor_weight=anchor_weight, log_q_init=log_q_init,
                     history=hist, reduction=reduction)
        with torch.no_grad():
            stage_ms["ms_forward"] = timed(lambda: forward_log_z(*dp_args, **dp_kw))
        stage_ms["ms_fwd_bwd"] = timed(
            lambda: lattice_forward_backward(
                *dp_args, **dp_kw, collect_stats=collect_stats, enforce_budget=False,
                checkpoint=checkpoint,
            )
        )
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    for _ in range(warmup):
        one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / repeats * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
    res = {
        "n_hist": float(hist.n_hist),
        "checkpoint": float(checkpoint),
        "ms_per_batch": ms,
        "gib_per_utt_frame": peak / (batch * t_len),
        "utterances_per_sec": batch / (ms / 1e3),
        "frames_per_sec": batch * t_len / (ms / 1e3),
        "peak_gib": peak,
        "z_zero": float(out.z_zero.sum()),
        "expected_tokens_mean": float(out.expected_tokens.mean()),
        **stage_ms,
    }
    print(
        "lattice bench: B=%d T=S=%d W=%d tau=%.1f alpha=%.1f stats=%s history=%s |h|=%d "
        "reduction=%s(%s) ckpt=%d on %s -> %.1f ms/batch, %.1f utt/s, %.0f frames/s, "
        "peak %.2f GiB (%.2e GiB/utt-frame), E[tokens]=%.1f"
        % (batch, t_len, cfg.band, temperature, anchor_weight, collect_stats, hist.name,
           hist.n_hist, reduction, "matmul" if _use_matmul(reduction, hist) else "elementwise",
           checkpoint, device, res["ms_per_batch"], res["utterances_per_sec"],
           res["frames_per_sec"], res["peak_gib"], res["gib_per_utt_frame"],
           res["expected_tokens_mean"]),
        flush=True,
    )
    if stage_ms:
        print("  stages: forward %.1f ms | forward+manual backward %.1f ms | full step %.1f ms "
              "(backward %.1f, surrogate+autograd %.1f)"
              % (stage_ms["ms_forward"], stage_ms["ms_fwd_bwd"], res["ms_per_batch"],
                 stage_ms["ms_fwd_bwd"] - stage_ms["ms_forward"],
                 res["ms_per_batch"] - stage_ms["ms_fwd_bwd"]), flush=True)
    return res


# public aliases of helpers used by reverse_model/ (same objects)
arc_weights = _arc_weights
band_matrix = _band_matrix
forward_ctx = _forward_ctx
scaled_prior_term = _prior_term


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="time the SAE 4a lattice at the S2 shape")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--t-len", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--history", default="bigram", choices=["bigram", "class_trigram", "trigram"])
    ap.add_argument("--n-classes", type=int, default=None, help="C of a class_trigram history")
    ap.add_argument("--classes", default=None, choices=["manner8"], help="a NAMED class partition")
    ap.add_argument("--reduction", default="auto", choices=["auto", "matmul", "elementwise"])
    ap.add_argument("--checkpoint", type=int, default=0, help="D4 forward-table stride S (0 = off)")
    ap.add_argument("--stages", action="store_true", help="time the forward and the backward apart")
    a = ap.parse_args()
    cls = None
    if a.classes == "manner8":
        from .prior import manner8_class_ids

        cls = manner8_class_ids()
    bench(batch=a.batch, t_len=a.t_len, device=a.device, collect_stats=not a.no_stats,
          repeats=a.repeats, history=a.history, n_classes=a.n_classes, classes=cls,
          reduction=a.reduction, checkpoint=a.checkpoint, stages=a.stages)
