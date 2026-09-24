"""Brute-force oracle of the blank-free lattice term (test plan 2026-09-24, T1.1).

INDEPENDENT of ``model/lattice.py``: nothing here imports the package. Every admissible latent --
a recognizer frame path ``a`` in ``{0..K-1}^T`` together with a segmentation of the ``S`` unit frames
into the tokens of its run collapse ``B(a)`` -- is listed explicitly, and its score is a float64 torch
expression, so ``logsumexp`` gives ``log Z`` and autograd gives exact gradients.

Definition implemented (plan T1.1)::

    runs of a:  tokens k_1..k_U, emit frames t_1 = 0 < t_2 < ... < t_U, t_{U+1} = T
    segmentation: 0 = s_0 < s_1 < ... < s_U = S,  d_u = s_u - s_{u-1} in [d_min, D(k_u)]
    admissible ("code" band): |s_u - stride * t| <= W for every t in [t_u + 1, t_{u+1}], every u
    term = (1/tau) [ sum_t log_q[t, a_t] + beta sum_u P[h_u, k_u] + sum_u seg[k_u, d_u - 1, s_{u-1}] ]
    h_u from (k_{u-2}, k_{u-1}) with BOS = K padding; bigram h = k_{u-1}; trigram h = k_{u-2} (K+1) + k_{u-1}
    no end-of-sentence term; log Z = logsumexp(terms), or -inf if there is no admissible latent.

``sil_split=True`` (off by default) adds the token readings in which a SIL run of frames is split
into several SIL tokens -- the CTC repeat rule that exempts SIL. It is not the definition; it exists
to show what a lattice built with that rule computes (T1.6).

Two other band readings are available for recording the band convention (plan T1.3, S2):
``"literal"`` (NOTE: ``|s_u - stride * t_u| <= W`` at the emitting frame only, no end constraint) and
``"next_frame"`` (``|s_u - stride * (t_u + 1)| <= W`` only, the lattice docstring's one-frame reading
without the repeat frames).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

BAND_MODES = ("code", "literal", "next_frame")


@dataclass(frozen=True)
class Topology:
    n_phones: int
    sil_id: int
    band: int
    d_min: int
    d_max: int
    d_max_sil: int
    stride: int = 3

    def d_max_of(self, k: int) -> int:
        return self.d_max_sil if k == self.sil_id else self.d_max

    @property
    def d_cap(self) -> int:
        return max(self.d_max, self.d_max_sil)

    @property
    def n_ctx(self) -> int:
        return self.n_phones + 1

    @property
    def bos(self) -> int:
        return self.n_phones

    def n_hist(self, history: str) -> int:
        return self.n_ctx if history == "bigram" else self.n_ctx * self.n_ctx


def run_collapse(path: Sequence[int]) -> Tuple[List[int], List[int]]:
    """``(tokens, emit_frames)``: one token per maximal run of equal symbols (SIL included)."""
    tokens, emits = [], []
    for t, c in enumerate(path):
        if t == 0 or c != path[t - 1]:
            tokens.append(int(c))
            emits.append(t)
    return tokens, emits


def token_readings(path: Sequence[int], sil_id: int, sil_split: bool) -> List[Tuple[List[int], List[int]]]:
    """Every ``(tokens, emit_frames)`` reading of a frame path.

    ``sil_split=False`` (the definition, NOTE §4.1): the run collapse only. ``sil_split=True`` is the
    CTC repeat rule that exempts SIL (lattice.py:355-357 under ``topology="ctc"``): inside a SIL run
    every frame after the first may either repeat the SIL token or open a NEW SIL token.
    """
    tokens, emits = run_collapse(path)
    if not sil_split:
        return [(tokens, emits)]
    optional = []  # frames that may open an extra SIL token
    for u, (k, t0) in enumerate(zip(tokens, emits)):
        if k == sil_id:
            t1 = emits[u + 1] if u + 1 < len(emits) else len(path)
            optional.extend(range(t0 + 1, t1))
    out = []
    for mask in itertools.product((False, True), repeat=len(optional)):
        extra = {t for t, m in zip(optional, mask) if m}
        em = sorted(set(emits) | extra)
        out.append(([int(path[t]) for t in em], em))
    return out


def history_rows(tokens: Sequence[int], topo: Topology, history: str) -> List[int]:
    """The prior-table row of every token: over TOKENS, never frames, BOS-padded."""
    rows = []
    for u in range(len(tokens)):
        h1 = tokens[u - 1] if u >= 1 else topo.bos
        h2 = tokens[u - 2] if u >= 2 else topo.bos
        if history == "bigram":
            rows.append(h1)
        elif history == "trigram":
            rows.append(h2 * topo.n_ctx + h1)
        else:
            raise ValueError(history)
    return rows


def _in_band(s: int, u: int, emits: Sequence[int], T: int, topo: Topology, band_mode: str) -> bool:
    w, st = topo.band, topo.stride
    if band_mode == "code":
        nxt = emits[u + 1] if u + 1 < len(emits) else T
        return all(abs(s - st * t) <= w for t in range(emits[u] + 1, nxt + 1))
    if band_mode == "literal":
        return abs(s - st * emits[u]) <= w
    if band_mode == "next_frame":
        return abs(s - st * (emits[u] + 1)) <= w
    raise ValueError(band_mode)


def segmentations(
    tokens: Sequence[int], emits: Sequence[int], T: int, S: int, topo: Topology, band_mode: str = "code"
) -> List[Tuple[int, ...]]:
    """Every duration tuple ``(d_1..d_U)`` summing to ``S`` that is admissible under ``band_mode``."""
    out: List[Tuple[int, ...]] = []
    n = len(tokens)

    def rec(u: int, s_prev: int, durs: List[int]):
        if u == n:
            if s_prev == S:
                out.append(tuple(durs))
            return
        for d in range(topo.d_min, topo.d_max_of(tokens[u]) + 1):
            s = s_prev + d
            if s + (n - u - 1) * topo.d_min > S:
                break
            if not _in_band(s, u, emits, T, topo, band_mode):
                continue
            durs.append(d)
            rec(u + 1, s, durs)
            durs.pop()

    rec(0, 0, [])
    return out


@dataclass
class Enumeration:
    """Index structures of every admissible latent of one utterance (value-free)."""

    T: int
    S: int
    s_cols: int  # columns of the seg table's s axis (S_max + 1 of the batch it came from)
    latents: List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]  # (path, tokens, durations)
    q_idx: torch.Tensor  # [N, T] flat index t * K + a_t
    seg_idx: torch.Tensor  # [N, U_max] flat index into [K, d_cap, s_cols], padded with the pad index
    p_idx: torch.Tensor  # [N, U_max] flat index into [|h|, K], padded with the pad index
    n_tokens: torch.Tensor  # [N] float64
    n_nonsil: torch.Tensor  # [N] float64
    seg_pad: int
    p_pad: int

    @property
    def n(self) -> int:
        return len(self.latents)


def enumerate_utterance(
    T: int,
    S: int,
    topo: Topology,
    history: str,
    *,
    s_cols: Optional[int] = None,
    band_mode: str = "code",
    paths: Optional[Iterable[Sequence[int]]] = None,
    sil_split: bool = False,
) -> Enumeration:
    """List every admissible (path, token reading, segmentation) of a ``T``-frame, ``S``-unit
    utterance. ``sil_split`` (default off = the definition): see :func:`token_readings`."""
    k_n, d_cap = topo.n_phones, topo.d_cap
    s_cols = S + 1 if s_cols is None else int(s_cols)
    seg_pad = k_n * d_cap * s_cols
    p_pad = topo.n_hist(history) * k_n
    latents, q_rows, seg_rows, p_rows, n_tok, n_ns = [], [], [], [], [], []
    path_iter = itertools.product(range(k_n), repeat=T) if paths is None else paths
    for path in path_iter:
        path = tuple(int(c) for c in path)
        assert len(path) == T
        for tokens, emits in token_readings(path, topo.sil_id, sil_split):
            segs = segmentations(tokens, emits, T, S, topo, band_mode)
            if not segs:
                continue
            rows = history_rows(tokens, topo, history)
            q_row = [t * k_n + c for t, c in enumerate(path)]
            p_row = [h * k_n + k for h, k in zip(rows, tokens)]
            for durs in segs:
                starts = [0]
                for d in durs[:-1]:
                    starts.append(starts[-1] + d)
                seg_row = [(k * d_cap + (d - 1)) * s_cols + s for k, d, s in zip(tokens, durs, starts)]
                latents.append((path, tuple(tokens), durs))
                q_rows.append(q_row)
                seg_rows.append(seg_row)
                p_rows.append(p_row)
                n_tok.append(len(tokens))
                n_ns.append(sum(1 for k in tokens if k != topo.sil_id))
    u_max = max([len(r) for r in seg_rows], default=1)
    pad = lambda rows, v: [r + [v] * (u_max - len(r)) for r in rows]  # noqa: E731
    long = dict(dtype=torch.long)
    f64 = dict(dtype=torch.float64)
    return Enumeration(
        T=T, S=S, s_cols=s_cols, latents=latents,
        q_idx=torch.tensor(q_rows, **long).view(-1, T),
        seg_idx=torch.tensor(pad(seg_rows, seg_pad), **long).view(-1, u_max),
        p_idx=torch.tensor(pad(p_rows, p_pad), **long).view(-1, u_max),
        n_tokens=torch.tensor(n_tok, **f64), n_nonsil=torch.tensor(n_ns, **f64),
        seg_pad=seg_pad, p_pad=p_pad,
    )


def _ext(x: torch.Tensor) -> torch.Tensor:
    """Flatten and append one exact zero (the pad slot)."""
    x = x.reshape(-1)
    return torch.cat([x, x.new_zeros(1)])


def pieces(enum: Enumeration, log_q: torch.Tensor, seg: torch.Tensor, prior: torch.Tensor):
    """``(acoustic [N], prior [N] raw, reverse [N] raw)`` per latent, float64, differentiable.

    ``log_q`` is ``[T, K]`` (this utterance's frames only), ``seg`` ``[K, d_cap, s_cols]``, ``prior``
    ``[|h|, K]``.
    """
    lq = log_q.to(torch.float64).reshape(-1)
    ac = lq[enum.q_idx].sum(dim=1)
    pr = _ext(prior.to(torch.float64))[enum.p_idx].sum(dim=1)
    rv = _ext(seg.to(torch.float64))[enum.seg_idx].sum(dim=1)
    return ac, pr, rv


def terms(enum: Enumeration, log_q, seg, prior, *, tau: float, beta: float) -> torch.Tensor:
    """``[N]`` the tempered score of every admissible latent."""
    ac, pr, rv = pieces(enum, log_q, seg, prior)
    total = ac + rv
    if float(beta) != 0.0:
        total = total + float(beta) * pr
    return total / float(tau)


def log_z(enum: Enumeration, log_q, seg, prior, *, tau: float, beta: float) -> torch.Tensor:
    """Scalar float64 ``log Z``; ``-inf`` when no latent is admissible."""
    if enum.n == 0:
        return torch.tensor(float("-inf"), dtype=torch.float64)
    return torch.logsumexp(terms(enum, log_q, seg, prior, tau=tau, beta=beta), dim=0)


def statistics(enum: Enumeration, log_q, seg, prior, *, tau: float, beta: float, n_symbols: int) -> dict:
    """Posterior expectations under ``pi = softmax(terms)`` (all detached, float64)."""
    k_n = n_symbols
    d_cap_s = enum.seg_pad  # = K * d_cap * s_cols
    with torch.no_grad():
        if enum.n == 0:
            return None
        t = terms(enum, log_q, seg, prior, tau=tau, beta=beta)
        pi = torch.softmax(t, dim=0)
        ac, pr, rv = pieces(enum, log_q, seg, prior)
        post_q = torch.zeros(enum.T * k_n, dtype=torch.float64)
        post_q.index_add_(0, enum.q_idx.reshape(-1), pi.repeat_interleave(enum.q_idx.shape[1]))
        seg_post = torch.zeros(d_cap_s + 1, dtype=torch.float64)
        seg_post.index_add_(0, enum.seg_idx.reshape(-1), pi.repeat_interleave(enum.seg_idx.shape[1]))
        return {
            "log_z": torch.logsumexp(t, dim=0),
            "pi": pi,
            "post_q": post_q.view(enum.T, k_n),
            "seg_post": seg_post[:-1],  # flat [K * d_cap * s_cols]
            "expected_tokens": (pi * enum.n_tokens).sum(),
            "expected_nonsil": (pi * enum.n_nonsil).sum(),
            "expected_reverse": (pi * rv).sum(),
            "expected_prior": (pi * pr).sum(),
        }


def count(T: int, S: int, topo: Topology, *, band_mode: str = "code", paths=None, sil_split=False) -> int:
    """Number of admissible latents (history does not matter for the count)."""
    return enumerate_utterance(T, S, topo, "bigram", band_mode=band_mode, paths=paths, sil_split=sil_split).n
