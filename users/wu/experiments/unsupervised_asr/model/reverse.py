"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/reverse.py.

Port note: ``NEG_INF`` (= -1.0e30, the value of ``sae/psi_align.NEG_INF``) is defined in this
module instead of imported from psi_align; every other line is the source's.

emc.reverse -- the segmental reverse model p_phi(z | y, eta) of SAE §4a (SAE_4A.md:50-56).

One state per token of the phone string y (39 ARPAbet + SIL; SIL may repeat), observations z a 50 Hz
stream of K = 500 unit ids, and an EXPLICIT categorical duration p(d | k) with d_min <= d <= D_k.
That is the §3a ``psi_align`` idea (one state per text symbol, exact log-space forward sum over
alignments, categorical emissions, d_min = 2) turned into the semi-Markov/HSMM form the method needs:
``psi_align``'s frame-synchronous self/advance/skip topology cannot express a duration DISTRIBUTION
or an emission that depends on the position INSIDE the segment, so the DP is written here rather than
imported; everything else (the fp32 log-space discipline, the finite ``NEG_INF``) is taken from it.

    log p_phi(z | y, eta) = log sum_{d_1..d_U : sum d_i = S}  prod_i  p(d_i | k_i)
                                                              prod_{r<d_i} nu_phi(k_i, d_i, r, eta)[z_.]

The emission nu_phi(k, d, r, eta) is a categorical over the K units conditioned on the token k, its
duration d (through a duration bucket and through the normalization of r), the within-segment
position r (through a relative-position bucket), and the frozen 16-d speaker vector eta. There is NO
left phonetic context in this build (SAE_4A.md:52-54), and -- structurally, not by convention -- no
recognizer posterior, hidden state or CTC path: the only inputs are (z, y, eta).

d_min = 2 is enforced STRUCTURALLY: durations below it are masked out of p(d | k) BEFORE its
log-softmax, so a segmentation containing a one-frame token has probability exactly zero. d_min is a
standing constraint of the campaign (SAE_4A.md "Standing constraints carried", SAE_3E1.md:161) and is
not a knob below 2.

Pure torch, no sisyphus -- ``test_reverse.py`` checks the forward sum against brute-force enumeration
over segmentations; the S1a read job lives in ``s1a_job.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# the same finite -inf as the source's ``sae/psi_align.NEG_INF`` (defined here, since psi_align is not
# ported): logsumexp of an all-masked row stays NaN-free
NEG_INF = -1.0e30

__all__ = [
    "ReverseConfig",
    "FitConfig",
    "SegmentalReverseModel",
    "duration_bucket",
    "position_bounds",
    "pack_batch",
    "feasible",
    "assert_finite_sufficient_stats",
    "fit",
    "length_buckets",
]


@dataclass
class ReverseConfig:
    """Shape and topology of p_phi. Every experimental constant carries its source."""

    n_types: int = 40  # 39 ARPAbet + SIL, the §1c inventory (setup report §3; dict.txt of bi2B89fES77z)
    n_units: int = 500  # K, the enc50 codebook (SAE_4A.md:43)
    sil_id: int = 39  # SIL is the last type (prior.PHONES order)
    d_min: int = 2  # standing constraint, never below 2 (SAE_4A.md "Standing constraints"; SAE_3E1.md:161)
    d_max: int = 25  # D = 25 frames = 0.5 s at 50 Hz, for phones (SAE_4A.md:53)
    d_max_sil: int = 50  # D_sil = 50 frames = 1.0 s (SAE_4A.md:53)
    eta_dim: int = 16  # PCA-16 speaker vector (SAE_4A.md:57)
    # --- capacity. Not an experimental constant: the S1a fit set is ~2.7e5 frames, so the binding
    # limit is the 4 h CPU refit budget of the S1a job, not expressiveness. psi_align's ~11 M is the
    # stated ceiling (SAE_4A.md:50-51), and this model is far below it by choice.
    d_model: int = 192
    d_ff: int = 512
    n_pos_buckets: int = 3  # onset / nucleus / coda of a segment, in normalized position r/d
    n_dur_buckets: int = 2  # short vs long segment
    dur_bucket_edge: int = 8  # d <= 8 frames (160 ms) is "short"

    def __post_init__(self):
        assert self.d_min >= 2, "d_min >= 2 is standing (SAE_4A.md); it is not a knob below 2"
        assert self.d_max >= self.d_min and self.d_max_sil >= self.d_min
        assert 0 <= self.sil_id < self.n_types
        assert self.n_pos_buckets >= 1 and self.n_dur_buckets >= 1

    @property
    def d_cap(self) -> int:
        """Largest duration any token may take (SIL's)."""
        return max(self.d_max, self.d_max_sil)

    def type_d_max(self, k: int) -> int:
        return self.d_max_sil if k == self.sil_id else self.d_max


@dataclass
class FitConfig:
    """Adam refit schedule. Explicit constants; the S1a job states the wall-time projection."""

    epochs: int = 8
    lr: float = 3.0e-3
    batch_size: int = 8  # utterances per step, bucketed by length so padding stays small
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    seed: int = 0
    log_every: int = 0  # 0 = once per epoch


# ---------------------------------------------------------------------------------------------------
# pure topology helpers (no torch state; the test's brute force uses exactly these)
# ---------------------------------------------------------------------------------------------------


def duration_bucket(d: int, cfg: ReverseConfig) -> int:
    """Which duration bucket a segment of ``d`` frames falls in."""
    return 0 if d <= cfg.dur_bucket_edge else 1


def position_bounds(d: int, n_pos: int) -> List[Tuple[int, int]]:
    """``[(start, end)]`` of the within-segment positions r (0-based, end-exclusive) per bucket.

    Bucket j covers ``r`` with ``ceil(j*d/n_pos) <= r < ceil((j+1)*d/n_pos)``, i.e. the segment is cut
    at fixed RELATIVE positions -- which is the dependence of the emission on the duration d beyond
    the duration bucket. Short segments leave trailing buckets empty (d = 2, n_pos = 3 -> the coda
    bucket is empty); an empty range contributes nothing.
    """
    return [(-((-j * d) // n_pos), -((-(j + 1) * d) // n_pos)) for j in range(n_pos)]


def pack_batch(
    items: Sequence[Tuple[Sequence[int], Sequence[int], Sequence[float]]],
    cfg: ReverseConfig,
    device=None,
) -> Dict[str, torch.Tensor]:
    """``[(z, y, eta)]`` -> padded tensors ``z [B, S]``, ``y [B, U]``, ``eta [B, E]``, lengths."""
    b = len(items)
    assert b > 0, "empty batch"
    s_max = max(len(z) for z, _, _ in items)
    u_max = max(len(y) for _, y, _ in items)
    z = torch.zeros(b, s_max, dtype=torch.long)
    y = torch.zeros(b, u_max, dtype=torch.long)
    eta = torch.zeros(b, cfg.eta_dim, dtype=torch.float32)
    s_lens = torch.zeros(b, dtype=torch.long)
    u_lens = torch.zeros(b, dtype=torch.long)
    for i, (zi, yi, ei) in enumerate(items):
        zi = torch.as_tensor(zi, dtype=torch.long)
        yi = torch.as_tensor(yi, dtype=torch.long)
        assert zi.numel() > 0 and yi.numel() > 0, "an empty utterance or transcript has no likelihood"
        assert int(zi.max()) < cfg.n_units and int(zi.min()) >= 0, "unit id outside the codebook"
        assert int(yi.max()) < cfg.n_types and int(yi.min()) >= 0, "phone id outside the inventory"
        z[i, : zi.numel()] = zi
        y[i, : yi.numel()] = yi
        eta[i] = torch.as_tensor(ei, dtype=torch.float32)
        s_lens[i] = zi.numel()
        u_lens[i] = yi.numel()
    out = {"z": z, "y": y, "eta": eta, "s_lens": s_lens, "u_lens": u_lens}
    return {k: v.to(device) if device is not None else v for k, v in out.items()}


def feasible(n_frames: int, y: Sequence[int], cfg: ReverseConfig) -> bool:
    """Is any segmentation of ``n_frames`` frames by the tokens ``y`` possible at all?

    ``U * d_min <= S <= sum_i D_{k_i}``. An infeasible pair has likelihood EXACTLY zero, which is a
    property of the topology, not a defect -- but it must be filtered before a paired read rather
    than averaged in as a -inf (memory: impossible scores poison means).
    """
    if len(y) == 0 or n_frames <= 0:
        return False
    lo = cfg.d_min * len(y)
    hi = sum(cfg.type_d_max(int(k)) for k in y)
    return lo <= n_frames <= hi


# ---------------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------------


class SegmentalReverseModel(nn.Module):
    """p_phi(z | y, eta) as an explicit-duration HSMM over the tokens of y.

    Parameters are (i) a duration logit table over ``d = 1..d_cap`` per token type, masked to
    ``[d_min, D_k]`` before its log-softmax, and (ii) an emission head mapping
    (type, duration bucket, position bucket, eta) to a categorical over the K units. The head is a
    one-hidden-layer MLP over summed embeddings: its whole input is the token identity and the
    speaker vector, so no phonetic context and no recognizer quantity can enter.
    """

    def __init__(self, cfg: ReverseConfig):
        super().__init__()
        self.cfg = cfg
        self.dur_logits = nn.Parameter(torch.zeros(cfg.n_types, cfg.d_cap))
        self.emb_type = nn.Embedding(cfg.n_types, cfg.d_model)
        self.emb_dur = nn.Embedding(cfg.n_dur_buckets, cfg.d_model)
        self.emb_pos = nn.Embedding(cfg.n_pos_buckets, cfg.d_model)
        self.eta_proj = nn.Linear(cfg.eta_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lin1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.lin2 = nn.Linear(cfg.d_ff, cfg.n_units)
        # the (type, dur bucket, pos bucket) grid, materialized once
        k = torch.arange(cfg.n_types).view(-1, 1, 1).expand(cfg.n_types, cfg.n_dur_buckets, cfg.n_pos_buckets)
        c = torch.arange(cfg.n_dur_buckets).view(1, -1, 1).expand_as(k)
        j = torch.arange(cfg.n_pos_buckets).view(1, 1, -1).expand_as(k)
        self.register_buffer("_grid_k", k.reshape(-1), persistent=False)
        self.register_buffer("_grid_c", c.reshape(-1), persistent=False)
        self.register_buffer("_grid_j", j.reshape(-1), persistent=False)
        self.register_buffer("_dur_mask", self._build_duration_mask(cfg), persistent=False)
        self.register_buffer("_dur_bucket", self._build_duration_bucket(cfg), persistent=False)
        self.reset_parameters()

    # -- construction -------------------------------------------------------------------------------

    @staticmethod
    def _build_duration_mask(cfg: ReverseConfig) -> torch.Tensor:
        """``[n_types, d_cap]`` bool: may token type k hold exactly d = index+1 frames?"""
        d = torch.arange(1, cfg.d_cap + 1).view(1, -1)
        d_max = torch.tensor([cfg.type_d_max(k) for k in range(cfg.n_types)]).view(-1, 1)
        return (d >= cfg.d_min) & (d <= d_max)

    @staticmethod
    def _build_duration_bucket(cfg: ReverseConfig) -> torch.Tensor:
        return torch.tensor([duration_bucket(d, cfg) for d in range(1, cfg.d_cap + 1)], dtype=torch.long)

    def reset_parameters(self, seed: Optional[int] = None) -> None:
        """Deterministic re-init, so every S1a condition refits from the IDENTICAL starting point."""
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
        for p in (self.emb_type.weight, self.emb_dur.weight, self.emb_pos.weight):
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
        for lin in (self.eta_proj, self.lin1, self.lin2):
            bound = 1.0 / math.sqrt(lin.in_features)
            with torch.no_grad():
                lin.weight.copy_((torch.rand(lin.weight.shape, generator=gen) * 2 - 1) * bound)
                lin.bias.zero_()
        with torch.no_grad():
            self.dur_logits.zero_()
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_table(self) -> Dict[str, int]:
        return {n: p.numel() for n, p in self.named_parameters()}

    # -- distributions ------------------------------------------------------------------------------

    def duration_log_probs(self) -> torch.Tensor:
        """``[n_types, d_cap]`` log p(d | k) for d = 1..d_cap; zero probability outside [d_min, D_k].

        The mask is applied BEFORE the log-softmax, so the distribution is normalized over the legal
        durations only and d = 1 is impossible by construction rather than by penalty.
        """
        masked = self.dur_logits.masked_fill(~self._dur_mask, NEG_INF)
        return torch.log_softmax(masked, dim=-1)

    def emission_log_probs(self, eta: torch.Tensor) -> torch.Tensor:
        """``[B, n_types, n_dur_buckets, n_pos_buckets, K]`` log nu_phi(k, bucket(d), bucket(r), eta)."""
        cfg = self.cfg
        b = eta.shape[0]
        x = (
            self.emb_type(self._grid_k) + self.emb_dur(self._grid_c) + self.emb_pos(self._grid_j)
        ).unsqueeze(0) + self.eta_proj(eta.float()).unsqueeze(1)
        h = F.gelu(self.lin1(self.norm(x)))
        logits = self.lin2(h).float()
        return torch.log_softmax(logits, dim=-1).view(
            b, cfg.n_types, cfg.n_dur_buckets, cfg.n_pos_buckets, cfg.n_units
        )

    # -- the DP -------------------------------------------------------------------------------------

    def segment_scores(self, z: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """``[B, n_types, d_cap, S+1]``: the emission sum of a segment of type k, duration d, start s.

        Entry ``[b, k, d-1, s]`` is ``sum_{r<d} log nu_phi(k, d, r, eta_b)[z_b[s+r]]``. Computed from
        per-bucket cumulative sums along time, so the cost is O(d_cap * n_pos) slices rather than
        O(d_cap^2). Entries with ``s + d > S`` are filled with 0 and are unreachable: a path can only
        read cell ``s`` of the forward table through segments that end at or before ``s``, and the
        likelihood is read at ``s = S_b``.
        """
        cfg = self.cfg
        emis = self.emission_log_probs(eta)  # [B, T, C, J, K]
        b, s_max = z.shape
        idx = z.view(b, 1, 1, 1, s_max).expand(b, cfg.n_types, cfg.n_dur_buckets, cfg.n_pos_buckets, s_max)
        f = torch.gather(emis, 4, idx)  # [B, T, C, J, S] log nu of the OBSERVED unit at each frame
        cum = torch.cat([f.new_zeros(f.shape[:-1] + (1,)), f.cumsum(dim=-1)], dim=-1)  # [B,T,C,J,S+1]
        out = f.new_zeros(b, cfg.n_types, cfg.d_cap, s_max + 1)
        for d in range(cfg.d_min, cfg.d_cap + 1):
            n_start = s_max - d + 1
            if n_start <= 0:
                continue
            c = duration_bucket(d, cfg)
            acc = f.new_zeros(b, cfg.n_types, n_start)
            for j, (lo, hi) in enumerate(position_bounds(d, cfg.n_pos_buckets)):
                if hi <= lo:
                    continue
                acc = acc + (cum[:, :, c, j, hi : hi + n_start] - cum[:, :, c, j, lo : lo + n_start])
            out[:, :, d - 1, :n_start] = acc
        return out

    def forward_logsum(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        eta: torch.Tensor,
        s_lens: torch.Tensor,
        u_lens: torch.Tensor,
        *,
        return_stats: bool = False,
    ):
        """Exact log-space forward sum over segmentations -> ``logp [B]``.

        ``alpha[i][b, s] = log p(z_b[0:s], the first i tokens exactly cover those s frames)``, and
        ``log p_phi(z | y, eta) = alpha[U_b][b, S_b]``. Everything is fp32 log space.
        """
        cfg = self.cfg
        b, s_max = z.shape
        u_max = y.shape[1]
        device = z.device
        seg = self.segment_scores(z, eta)  # [B, T, D, S+1]
        logdur = self.duration_log_probs()  # [T, D]

        # shift index: term[b, d, s] reads base[b, d, s-d]; pad d_cap cells of NEG_INF in front.
        s_ax = torch.arange(s_max + 1, device=device).view(1, -1)
        d_ax = torch.arange(cfg.d_cap, device=device).view(-1, 1) + 1
        shift_idx = (s_ax - d_ax + cfg.d_cap).clamp(min=0).view(1, cfg.d_cap, s_max + 1).expand(b, -1, -1)

        alpha = torch.full((b, s_max + 1), NEG_INF, device=device, dtype=torch.float32)
        alpha[:, 0] = 0.0
        alphas = [alpha]
        rows = torch.arange(b, device=device)
        for i in range(u_max):
            k_i = y[:, i]
            base = alpha.unsqueeze(1) + seg[rows, k_i]  # [B, D, S+1] alpha(start) + emission(start, d)
            padded = torch.cat(
                [base.new_full((b, cfg.d_cap, cfg.d_cap), NEG_INF), base], dim=2
            )
            shifted = torch.gather(padded, 2, shift_idx)  # [B, D, S+1] at s: alpha(s-d) + seg(s-d, d)
            alpha = torch.logsumexp(shifted + logdur[k_i].unsqueeze(-1), dim=1)
            alphas.append(alpha)

        table = torch.stack(alphas, dim=1)  # [B, U+1, S+1]
        logp = table[rows, u_lens, s_lens]
        if not return_stats:
            return logp
        stats = {
            "alpha_final": logp,
            "alpha_table": table,
            "segment_scores": seg,
            "duration_log_probs": logdur,
        }
        return logp, stats

    def log_likelihood(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        eta: torch.Tensor,
        s_lens: torch.Tensor,
        u_lens: torch.Tensor,
        *,
        return_stats: bool = False,
    ):
        """Per-utterance ``log p_phi(z | y, eta)`` (batched, padded). Alias of :meth:`forward_logsum`."""
        return self.forward_logsum(z, y, eta, s_lens, u_lens, return_stats=return_stats)

    # -- sampling -----------------------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        eta: torch.Tensor,
        u_lens: Optional[torch.Tensor] = None,
        *,
        generator: Optional[torch.Generator] = None,
        chunk: int = 16384,
        unit_draw: str = "sample",
    ) -> Dict[str, torch.Tensor]:
        """Draw ``z ~ p_phi(z | y, eta)`` -- the generative process :meth:`forward_logsum` sums over.

        Per token ``i``: one duration ``d_i ~ p(d | k_i)`` over the MASKED support, so
        ``d_min <= d_i <= D_{k_i}`` by construction and not by rejection (the same mask
        :meth:`duration_log_probs` applies before its log-softmax).  Then, per within-segment
        position ``r < d_i``, one unit ``z ~ nu_phi(k_i, d_i, r, eta)`` from the emission head's
        categorical for the cell ``(k_i, bucket(d_i), bucket(r))``.

        Its inputs are exactly the likelihood's: ``(y, eta)`` and nothing else, so a sampled stream
        carries no observation of any real utterance.  It is the back-translation direction of the
        same model (``bt_probe.py``); the duration histogram of many samples estimates ``p(d | k)``
        and the per-frame unit histogram of a cell estimates that cell's ``nu_phi``, which is what
        ``test_bt_probe`` asserts against these two tables.

        :param y: ``[B, U]`` token ids, padded (padding is never read; ``u_lens`` decides).
        :param eta: ``[B, E]`` frozen speaker vectors.
        :param u_lens: ``[B]`` token counts, or ``None`` for "every column is a token".
        :param generator: a ``torch.Generator`` ON THE MODEL'S DEVICE -- the only randomness here,
            so the draw is reproducible from its seed.
        :param chunk: frames per ``multinomial`` call, which bounds the ``[chunk, K]`` emission
            probability matrix that call materializes.  Not an experimental constant.
        :param unit_draw: how the per-frame UNIT is taken from its cell's categorical: ``"sample"``
            (the generative process above, the default and what every existing caller gets) or
            ``"argmax"``, the cell's most probable unit.  The DURATIONS are drawn either way.
            ``"argmax"`` is what ``emc.bt_aux`` uses inside the EMC loop (SAE_4A.md "S3b-BT-aux":
            "units taken by ARGMAX from the live phi ... with durations and the speaker / frame-pool
            draw SAMPLED" -- argmax the content, randomise the nuisance).
        :return: ``{"z": [B, S_max] long (zeros past ``s_lens``), "s_lens": [B],
            "durations": [B, U] (0 on padded tokens)}``.
        """
        assert unit_draw in ("sample", "argmax"), unit_draw
        cfg = self.cfg
        device = self.dur_logits.device
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        assert y.dim() == 2, y.shape
        b, u_max = y.shape
        eta = torch.as_tensor(eta, dtype=torch.float32, device=device)
        assert eta.shape == (b, cfg.eta_dim), (tuple(eta.shape), b, cfg.eta_dim)
        if u_lens is None:
            u_lens = torch.full((b,), u_max, dtype=torch.long, device=device)
        else:
            u_lens = torch.as_tensor(u_lens, dtype=torch.long, device=device)
        assert u_lens.shape == (b,), (tuple(u_lens.shape), b)
        assert int(u_lens.min()) >= 1, "an empty token string has no z to draw"
        assert int(u_lens.max()) <= u_max, (int(u_lens.max()), u_max)
        tok_mask = torch.arange(u_max, device=device).unsqueeze(0) < u_lens.unsqueeze(1)
        assert int(y[tok_mask].max()) < cfg.n_types and int(y[tok_mask].min()) >= 0, (
            "token id outside the inventory"
        )

        # -- one duration per token, from the legal support of its type -------------------------
        dur_p = self.duration_log_probs().exp()  # [n_types, d_cap]; exactly 0 outside [d_min, D_k]
        drawn = torch.multinomial(dur_p[y.reshape(-1)], 1, generator=generator).view(b, u_max) + 1
        durations = torch.where(tok_mask, drawn, torch.zeros_like(drawn))
        legal = self._dur_mask[y.reshape(-1), (drawn - 1).reshape(-1)].view(b, u_max)
        assert bool((legal | ~tok_mask).all()), "a duration draw left the [d_min, D_k] support"
        s_lens = durations.sum(dim=1)

        # -- the (utterance, type, duration bucket, position bucket) of every frame -------------
        k_flat = y[tok_mask]  # [M], tokens in row-major order, so each utterance is contiguous
        d_flat = durations[tok_mask]  # [M]
        b_flat = torch.arange(b, device=device).unsqueeze(1).expand(b, u_max)[tok_mask]
        total = int(d_flat.sum())
        k_frame = torch.repeat_interleave(k_flat, d_flat)
        d_frame = torch.repeat_interleave(d_flat, d_flat)
        b_frame = torch.repeat_interleave(b_flat, d_flat)
        seg_start = torch.cumsum(d_flat, dim=0) - d_flat
        r_frame = torch.arange(total, device=device) - torch.repeat_interleave(seg_start, d_flat)
        c_frame = self._dur_bucket[d_frame - 1]
        # The position bucket of r is the largest j with ceil(j*d/n_pos) <= r, i.e. floor(r*n_pos/d):
        # the closed form of :func:`position_bounds`, which ``test_bt_probe`` checks exhaustively.
        j_frame = (r_frame * cfg.n_pos_buckets) // d_frame

        emis = self.emission_log_probs(eta)  # [B, T, C, J, K]
        z_flat = torch.empty(total, dtype=torch.long, device=device)
        for lo in range(0, total, int(chunk)):
            hi = min(lo + int(chunk), total)
            probs = emis[b_frame[lo:hi], k_frame[lo:hi], c_frame[lo:hi], j_frame[lo:hi]].exp()
            if unit_draw == "argmax":
                z_flat[lo:hi] = probs.argmax(dim=-1)
            else:
                z_flat[lo:hi] = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        utt_start = torch.cumsum(s_lens, dim=0) - s_lens
        pos = torch.arange(total, device=device) - utt_start[b_frame]
        z = torch.zeros(b, int(s_lens.max()), dtype=torch.long, device=device)
        z[b_frame, pos] = z_flat
        return {"z": z, "s_lens": s_lens, "durations": durations}

    def forward(self, batch: Dict[str, torch.Tensor], **kw):
        return self.log_likelihood(
            batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"], **kw
        )


def assert_finite_sufficient_stats(stats: Dict[str, torch.Tensor]) -> None:
    """Finiteness is asserted on the SUFFICIENT STATISTICS, never inferred from the likelihood.

    A scaled forward recursion can produce a finite log-likelihood while a per-segment term or a
    forward cell is already NaN (memory: finite-likelihood-hides-nan-posterior), and a NaN passes
    every ``>`` tolerance check downstream. The floor value ``NEG_INF`` is legal (it is how an
    unreachable cell and an illegal duration are spelled); NaN and +inf are not.
    """
    for name in ("segment_scores", "duration_log_probs", "alpha_table", "alpha_final"):
        t = stats[name]
        if torch.isnan(t).any():
            raise FloatingPointError(f"{name} contains NaN")
        if torch.isposinf(t).any():
            raise FloatingPointError(f"{name} contains +inf")
        if torch.isneginf(t).any():
            raise FloatingPointError(f"{name} contains a true -inf (the floor must stay finite)")
    final = stats["alpha_final"]
    if bool((final <= NEG_INF / 2).any()):
        raise FloatingPointError(
            "an utterance has zero-probability under every segmentation (infeasible (S, U) pair); "
            "filter it before the read instead of averaging an impossible score"
        )


# ---------------------------------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------------------------------


def _length_buckets(items: Sequence[Tuple], batch_size: int) -> List[List[int]]:
    """Indices grouped into batches of similar (S, U), so padding stays small."""
    order = sorted(range(len(items)), key=lambda i: (len(items[i][0]), len(items[i][1])))
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


@torch.no_grad()
def evaluate(
    model: SegmentalReverseModel,
    items: Sequence[Tuple[Sequence[int], Sequence[int], Sequence[float]]],
    *,
    batch_size: int = 8,
    per_utterance: bool = False,
):
    """Held-out read: ``(sum log p_phi, total frames, [per-utterance rows])``.

    The per-frame quantity the gate reads is ``sum log p / sum frames`` over the held-out set; the
    per-utterance rows carry each utterance's own ``log_p`` and frame count so the paired read and
    its clustered bootstrap can be formed downstream.
    """
    model.eval()
    total_lp, total_frames = 0.0, 0
    rows: List[Dict[str, float]] = []
    for idx in _length_buckets(items, batch_size):
        batch = pack_batch([items[i] for i in idx], model.cfg)
        logp, stats = model.log_likelihood(
            batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"], return_stats=True
        )
        assert_finite_sufficient_stats(stats)
        for pos, i in enumerate(idx):
            lp = float(logp[pos])
            n = int(batch["s_lens"][pos])
            total_lp += lp
            total_frames += n
            if per_utterance:
                rows.append({"index": i, "log_p": lp, "frames": n, "log_p_per_frame": lp / n})
    if per_utterance:
        return total_lp, total_frames, rows
    return total_lp, total_frames


def fit(
    model: SegmentalReverseModel,
    items: Sequence[Tuple[Sequence[int], Sequence[int], Sequence[float]]],
    cfg: FitConfig,
    *,
    held: Optional[Sequence[Tuple[Sequence[int], Sequence[int], Sequence[float]]]] = None,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """Adam refit of phi on ``items``; returns the per-epoch history.

    The objective is the summed log-likelihood per FRAME (not per utterance), which is the quantity
    G4a.1 reads, so training and the gate optimize the same scale.
    """
    torch.manual_seed(int(cfg.seed))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    batches = _length_buckets(items, cfg.batch_size)
    rng = torch.Generator().manual_seed(int(cfg.seed))
    history: List[Dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(batches), generator=rng).tolist()
        ep_lp, ep_frames = 0.0, 0
        for step, bi in enumerate(order):
            idx = batches[bi]
            batch = pack_batch([items[i] for i in idx], model.cfg)
            logp, stats = model.log_likelihood(
                batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"],
                return_stats=True,
            )
            # The FULL sufficient-statistic check (every per-segment term and every forward cell)
            # runs once per epoch and on every held-out read; the per-step check is the cheap one on
            # the final forward cells. The cheap check is an addition to the full one, never a
            # substitute for it (memory: finite-likelihood-hides-nan-posterior).
            if step == 0:
                assert_finite_sufficient_stats(stats)
            elif not bool(torch.isfinite(logp).all()) or bool((logp <= NEG_INF / 2).any()):
                raise FloatingPointError("a training utterance scored non-finite or impossible")
            frames = int(batch["s_lens"].sum())
            loss = -logp.sum() / frames
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_lp += float(logp.sum().detach())
            ep_frames += frames
        row = {"epoch": epoch, "train_log_p_per_frame": ep_lp / max(ep_frames, 1)}
        if held:
            hlp, hframes = evaluate(model, held, batch_size=cfg.batch_size)
            row["held_log_p_per_frame"] = hlp / max(hframes, 1)
        history.append(row)
        if verbose:
            print(
                "epoch %d train %.5f%s"
                % (
                    epoch,
                    row["train_log_p_per_frame"],
                    "" if held is None else " held %.5f" % row["held_log_p_per_frame"],
                ),
                flush=True,
            )
    return history


# public aliases of helpers used by reverse_model/ (same objects)
length_buckets = _length_buckets
