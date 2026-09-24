"""The max-plus decode of the blank-free lattice (test plan 2026-09-24, T2.9):
``reverse_model/genmarg_decode.py`` (``viterbi_blankfree``, ``segment_conditionals``) against the
brute-force enumerator ``tests/lattice_oracle.py``.

* ``viterbi_blankfree``'s ``log_w`` = the oracle's maximum tempered latent score (1e-10), and
  ``z_zero`` = "no admissible latent";
* the returned frame path, token string and segmentation form an admissible latent of the oracle
  whose score is that maximum;
* ``segment_conditionals`` re-adds the path weight from its per-token terms (1e-10), and every
  conditional's argmax is the Viterbi symbol.

Instance: the T1.1 inventory (K 3, SIL 2, d_min 2, D 3 / D_SIL 5, stride 3, S [12, 11, 10, 7]).  Two
history constructions: the blank-free one (run-collapse latents, ``sil_split=False``) and the one the
bed's model actually holds, built for the "ctc" topology (model/emc_model.py:367, pinned by T1.6),
under which a SIL run may split into several SIL tokens -- enumerated with ``sil_split=True``.
"""

from __future__ import annotations

import functools
from dataclasses import replace

import pytest
import torch

import lattice_oracle as O

from i6_experiments.users.wu.experiments.unsupervised_asr.model import lattice as L
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import genmarg_decode as GD

K, SIL = 3, 2
S_LENS, T_LENS = [12, 11, 10, 7], [4, 4, 4, 3]


def _cfg(band):
    return L.LatticeConfig(n_phones=K, sil_id=SIL, band=band, d_min=2, d_max=3, d_max_sil=5,
                           topology="blankfree", recognizer_stride=3)


@functools.lru_cache(maxsize=None)
def _inputs(history, seed):
    g = torch.Generator().manual_seed(seed)
    n_hist = (K + 1) if history == "bigram" else (K + 1) ** 2
    log_q = torch.log_softmax(2.0 * torch.randn(4, 4, K, generator=g, dtype=torch.float64), -1)
    seg = torch.randn(4, K, 5, 13, generator=g, dtype=torch.float64) - 2.0
    prior = torch.log_softmax(1.5 * torch.randn(n_hist, K, generator=g, dtype=torch.float64), -1)
    return log_q, seg, prior


CASES = [(w, h, tau, beta, ht, ck)
         for w in (1, 2, 40) for h in ("bigram", "trigram") for tau in (1.0, 2.0)
         for beta, ht, ck in ((1.0, "blankfree", 0), (0.5, "blankfree", 3), (1.0, "ctc", 2))]


def _id(c):
    return f"W{c[0]}-{c[1]}-tau{c[2]:g}-beta{c[3]:g}-{c[4]}-ck{c[5]}"


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_t2_9_viterbi_vs_brute_force(case):
    band, history, tau, beta, hist_topology, ck = case
    cfg = _cfg(band)
    hist_cfg = cfg if hist_topology == "blankfree" else replace(cfg, topology="ctc", recognizer_stride=1)
    hist = L.build_prior_history(hist_cfg, history)
    log_q, seg, prior = _inputs(history, 17 if history == "trigram" else 5)
    topo = O.Topology(K, SIL, band, 2, 3, 5, 3)
    path = GD.viterbi_blankfree(log_q, seg, prior, torch.tensor(T_LENS), torch.tensor(S_LENS), cfg,
                                temperature=tau, prior_weight=beta, history=hist, checkpoint=ck)
    cond = GD.segment_conditionals(path, log_q, seg, prior, cfg, temperature=tau, prior_weight=beta,
                                   history=hist)
    n_live = 0
    for b, (T, S) in enumerate(zip(T_LENS, S_LENS)):
        enum = O.enumerate_utterance(T, S, topo, history, s_cols=seg.shape[-1],
                                     sil_split=(hist_topology == "ctc"))
        if enum.n == 0:
            assert bool(path.z_zero[b]) and float(path.log_w[b]) <= L.NEG_INF / 2, b
            continue
        n_live += 1
        assert not bool(path.z_zero[b]), b
        terms = O.terms(enum, log_q[b, :T], seg[b], prior, tau=tau, beta=beta)
        best = float(terms.max())
        got = float(path.log_w[b])
        assert abs(got - best) <= 1e-10 * max(1.0, abs(best)), (b, got, best)
        # the returned latent: frame path, token string (the collapse of the emits), durations
        n_tok = int(path.n_tokens[b])
        tokens = tuple(int(k) for k in path.phones[b, :n_tok])
        frames = tuple(int(tokens[int(u)]) for u in path.frame_token[b, :T])
        emits = [t for t in range(T) if int(path.emit_phone[b, t]) >= 0]
        assert [int(path.emit_phone[b, t]) for t in emits] == list(tokens)
        durs = tuple(int(path.emit_duration[b, t]) for t in emits)
        starts = [int(path.emit_start[b, t]) for t in emits]
        assert starts == [sum(durs[:u]) for u in range(n_tok)] and sum(durs) == S
        assert emits[0] == 0 and (path.frame_token[b, T:] == -1).all()
        if hist_topology == "blankfree":
            assert list(tokens) == O.run_collapse(frames)[0]
        latent = (frames, tokens, durs)
        assert latent in enum.latents, (b, latent)
        own = float(terms[enum.latents.index(latent)])
        assert abs(own - best) <= 1e-10 * max(1.0, abs(best)), (b, own, best)
        # the per-token re-add of the path weight, and the conditionals' argmax
        assert abs(float(cond.path_log_w[b]) - got) <= 1e-10 * max(1.0, abs(got)), (b, float(cond.path_log_w[b]), got)
        assert cond.ids[b, :n_tok].tolist() == list(tokens) and bool(cond.valid[b, :n_tok].all())
        assert not bool(cond.valid[b, n_tok:].any())
    assert int(cond.mismatch) == 0
    assert n_live >= 1
