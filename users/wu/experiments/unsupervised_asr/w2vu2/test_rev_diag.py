"""Tests for ``rev_diag_worker`` -- the diagnostic readout of the arm's reverse model phi.

Run (from the setup dir):

    WS=$PWD; PYTHONPATH=$WS/recipe:$WS/tools/sisyphus:$WS/recipe/i6_models:$WS/recipe/returnn:\
$WS/recipe/2025-10-speech-llm/src \
    /e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python -m pytest <this file> -q

The checks:

1. the GOLD forced-alignment DP on a tiny hand-checkable case -- 3 phones, 8 frames, d_min 2, where
   the legal segmentations are exactly the six compositions (2,2,4) (2,3,3) (2,4,2) (3,2,3) (3,3,2)
   (4,2,2): its Viterbi score and segmentation equal the best of the six scored one by one with
   ``reverse.segmentation_log_prob``, and its forward equals their log-sum;
2. the derangement of (b) has NO fixed point, is a permutation, is a function of (seed, tag) alone
   and does not exist below two frames;
3. the lattice recursion of ``lattice_best_path`` is the production one: in ``sum`` mode it
   reproduces ``lattice.forward_log_z`` on tiny blank-free shapes at stride 3 and stride 1;
4. its ``max`` mode is the Viterbi of that same lattice: at a low temperature the forward log Z
   collapses onto the max, and the backtraced path re-scores arc by arc to exactly the max score
   and tiles the reverse frames.
"""

from __future__ import annotations

import numpy as np
import torch

from speech_llm.sae.emc.lattice import (
    LatticeConfig,
    bigram_history,
    build_segment_table,
    forward_log_z,
    scaled_seg_pad,
)
from speech_llm.sae.emc.reverse import (
    ReverseConfig,
    SegmentalReverseModel,
    enumerate_segmentations,
    segmentation_log_prob,
)

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.rev_diag_worker import (
    assert_tiling,
    frame_derangement,
    gold_forced_align,
    greedy_segments,
    lattice_best_path,
    predicted_units,
    rescore_path,
    top1_unit_table,
)

# 3 token types, d_min 2, D = 4: small enough that the segmentations of 8 frames are countable by
# hand and large enough to carry every bucket of the emission head.
TINY_REV = ReverseConfig(n_types=3, n_units=7, sil_id=2, d_min=2, d_max=4, d_max_sil=4,
                         d_model=8, d_ff=16)


def _tiny_phi(seed=0):
    torch.manual_seed(seed)
    phi = SegmentalReverseModel(TINY_REV)
    phi.reset_parameters(seed)
    with torch.no_grad():   # a flat duration table would make every segmentation tie
        phi.dur_logits.copy_(torch.randn(phi.dur_logits.shape, generator=torch.Generator().manual_seed(seed + 1)))
    phi.eval()
    return phi


def test_gold_forced_alignment_on_a_tiny_hand_checkable_case():
    phi = _tiny_phi()
    g = torch.Generator().manual_seed(3)
    z = torch.randint(0, TINY_REV.n_units, (1, 8), generator=g)
    y = torch.tensor([[0, 1, 2]])
    eta = torch.randn(1, TINY_REV.eta_dim, generator=g)
    s_lens, u_lens = torch.tensor([8]), torch.tensor([3])

    segmentations = list(enumerate_segmentations(8, y[0].tolist(), TINY_REV, d_min=TINY_REV.d_min))
    assert sorted(segmentations) == sorted([[2, 2, 4], [2, 3, 3], [2, 4, 2],
                                            [3, 2, 3], [3, 3, 2], [4, 2, 2]]), segmentations
    scores = [float(segmentation_log_prob(phi, z[0].tolist(), y[0].tolist(), eta[0].tolist(), d))
              for d in segmentations]
    want_max, want_sum = max(scores), float(torch.logsumexp(torch.tensor(scores), dim=0))
    want_durs = segmentations[int(np.argmax(scores))]

    with torch.no_grad():
        fwd, vit, paths = gold_forced_align(phi, z, y, eta, s_lens, u_lens)
    assert abs(float(fwd[0]) - want_sum) < 1e-4, (float(fwd[0]), want_sum)
    assert abs(float(vit[0]) - want_max) < 1e-4, (float(vit[0]), want_max)
    got_durs = [d for _s0, d, _k, _i in paths[0]]
    assert got_durs == want_durs, (got_durs, want_durs)
    assert [k for _s0, _d, k, _i in paths[0]] == [0, 1, 2]
    assert_tiling(paths[0], 8)
    with torch.no_grad():
        top1 = top1_unit_table(phi, eta).numpy()
    pred = predicted_units(paths[0], 8, top1[0], TINY_REV)
    assert pred.shape == (8,) and int(pred.min()) >= 0 and int(pred.max()) < TINY_REV.n_units
    print("PASS gold DP: viterbi %.6f (%s) forward %.6f over %d segmentations"
          % (want_max, want_durs, want_sum, len(segmentations)))


def test_the_derangement_has_no_fixed_point():
    for n in list(range(2, 40)) + [128, 517]:
        for tag in ("1272-128104-0000", "6930-75918-0003", "x"):
            perm = frame_derangement(tag, n, seed=0)
            assert sorted(perm.tolist()) == list(range(n)), "not a permutation"
            assert not bool((perm == np.arange(n)).any()), f"fixed point at n = {n}, tag {tag}"
            assert np.array_equal(perm, frame_derangement(tag, n, seed=0)), "not a function of (seed, tag)"
            assert not np.array_equal(perm, frame_derangement(tag, n, seed=1)) or n < 3
    assert frame_derangement("t", 1, seed=0) is None and frame_derangement("t", 0, seed=0) is None
    print("PASS derangement: no fixed point, deterministic in (seed, tag), none below two frames")


# --- the lattice ------------------------------------------------------------------------------------

SHAPES = [
    # (n_phones, sil_id, d_max, d_max_sil, band, stride, T, S, temperature)
    (2, 1, 3, 4, 6, 1, 5, 5, 1.0),
    (2, 1, 3, 4, 6, 3, 3, 8, 1.0),
    (3, 2, 4, 5, 5, 3, 4, 11, 1.0),
    (3, 2, 4, 5, 8, 3, 5, 13, 2.0),
]


def _lattice_inputs(cfg, t_len, s_len, seed):
    g = torch.Generator().manual_seed(seed)
    log_q = torch.log_softmax(torch.randn(1, t_len, cfg.n_symbols, generator=g), dim=-1).double()
    seg = (torch.randn(1, cfg.n_phones, cfg.d_cap, s_len + 1, generator=g) * 0.7 - 1.0).double()
    feat_lens = torch.tensor([t_len])
    unit_lens = torch.tensor([s_len])
    prior = torch.zeros(cfg.n_ctx, cfg.n_phones, dtype=torch.float64)
    return log_q, seg, prior, feat_lens, unit_lens


def test_sum_mode_reproduces_the_production_forward():
    for i, (n_ph, sil, dmax, dsil, band, stride, t_len, s_len, temp) in enumerate(SHAPES):
        cfg = LatticeConfig(n_phones=n_ph, sil_id=sil, band=band, d_min=2, d_max=dmax,
                            d_max_sil=dsil, topology="blankfree", recognizer_stride=stride)
        log_q, seg, prior, tl, sl = _lattice_inputs(cfg, t_len, s_len, seed=200 + i)
        want = float(forward_log_z(log_q, seg, prior, tl, sl, cfg, temperature=temp,
                                   prior_weight=0.0)[0])
        seg_pad = scaled_seg_pad(seg, sl, cfg, temp)
        got = float(lattice_best_path(log_q=log_q, seg_pad=seg_pad, feat_lens=tl, unit_lens=sl,
                                      cfg=cfg, hist=bigram_history(cfg), temperature=temp,
                                      mode="sum", paths=False)["score"][0])
        assert abs(got - want) < 1e-8, f"shape {i}: {got} vs {want}"
        print("PASS sum mode shape %d (T=%d S=%d stride=%d W=%d tau=%.1f): log Z = %.6f"
              % (i, t_len, s_len, stride, band, temp, want))


def test_max_mode_is_the_viterbi_of_the_same_lattice():
    for i, (n_ph, sil, dmax, dsil, band, stride, t_len, s_len, _temp) in enumerate(SHAPES):
        cfg = LatticeConfig(n_phones=n_ph, sil_id=sil, band=band, d_min=2, d_max=dmax,
                            d_max_sil=dsil, topology="blankfree", recognizer_stride=stride)
        log_q, seg, prior, tl, sl = _lattice_inputs(cfg, t_len, s_len, seed=300 + i)
        hist = bigram_history(cfg)
        # the max score at tau: the forward at a LOW temperature collapses ONTO it -- log Z - V is
        # the log-count of the near-optimal paths, and at tau = 0.002 every non-optimal path is
        # more than 100 nats down, so the two agree bit for bit in fp64 on these shapes.
        cold = 0.002
        vit = lattice_best_path(log_q=log_q, seg_pad=scaled_seg_pad(seg, sl, cfg, cold),
                                feat_lens=tl, unit_lens=sl, cfg=cfg, hist=hist, temperature=cold,
                                mode="max")
        want = float(forward_log_z(log_q, seg, prior, tl, sl, cfg, temperature=cold,
                                   prior_weight=0.0)[0])
        got = float(vit["score"][0])
        assert 0.0 <= want - got < 1e-6, f"shape {i}: viterbi {got} vs cold log Z {want}"
        # the backtrace is that path: it tiles the reverse frames and re-scores to the same number
        segs, reps = vit["paths"][0], vit["repeats"][0]
        assert segs is not None
        assert_tiling(segs, s_len)
        seg_pad = scaled_seg_pad(seg, sl, cfg, cold)
        ref = rescore_path(segs, reps, log_q[0].numpy(), seg_pad[0].numpy(), cfg, cold)
        assert abs(ref - got) < 1e-6, f"shape {i}: rescored {ref} vs viterbi {got}"
        assert len(segs) + len(reps) == t_len, (len(segs), len(reps), t_len)
        print("PASS max mode shape %d: V = %.6f, cold log Z = %.6f, %d emits + %d repeats"
              % (i, got, want, len(segs), len(reps)))


def test_greedy_segments_tile_the_unit_clock():
    for s_len, stride in ((11, 3), (12, 3), (13, 3), (5, 1)):
        t_len = -(-s_len // stride)
        labels = np.random.RandomState(0).randint(0, 3, size=t_len)
        segs = greedy_segments(labels, s_len, stride)
        assert_tiling(segs, s_len)
    print("PASS greedy segments tile [0, S) at stride 3 and stride 1")


def test_the_viterbi_path_of_the_real_reverse_model_is_consistent():
    """The same checks with the segment table of a real ``SegmentalReverseModel``, at stride 3."""
    cfg = LatticeConfig(n_phones=TINY_REV.n_types, sil_id=TINY_REV.sil_id, band=6, d_min=2,
                        d_max=TINY_REV.d_max, d_max_sil=TINY_REV.d_max_sil, topology="blankfree",
                        recognizer_stride=3)
    phi = _tiny_phi(seed=5)
    g = torch.Generator().manual_seed(7)
    s_len, t_len = 11, 4
    z = torch.randint(0, TINY_REV.n_units, (1, s_len), generator=g)
    eta = torch.randn(1, TINY_REV.eta_dim, generator=g)
    log_q = torch.log_softmax(torch.randn(1, t_len, cfg.n_phones, generator=g), dim=-1).double()
    with torch.no_grad():
        seg = build_segment_table(phi, z, eta).double()
    sl, tl = torch.tensor([s_len]), torch.tensor([t_len])
    seg_pad = scaled_seg_pad(seg, sl, cfg, 1.0)
    vit = lattice_best_path(log_q=log_q, seg_pad=seg_pad, feat_lens=tl, unit_lens=sl, cfg=cfg,
                            hist=bigram_history(cfg), temperature=1.0, mode="max")
    segs, reps = vit["paths"][0], vit["repeats"][0]
    assert_tiling(segs, s_len)
    ref = rescore_path(segs, reps, log_q[0].numpy(), seg_pad[0].numpy(), cfg, 1.0)
    assert abs(ref - float(vit["score"][0])) < 1e-6
    logz = float(forward_log_z(log_q, seg, torch.zeros(cfg.n_ctx, cfg.n_phones, dtype=torch.float64),
                               tl, sl, cfg, temperature=1.0, prior_weight=0.0)[0])
    assert float(vit["score"][0]) <= logz + 1e-9, (float(vit["score"][0]), logz)
    with torch.no_grad():
        top1 = top1_unit_table(phi, eta).numpy()
    pred = predicted_units(segs, s_len, top1[0], TINY_REV)
    assert pred.shape == (s_len,)
    print("PASS real-phi lattice: V = %.6f <= log Z = %.6f, %d emits + %d repeats, %d distinct preds"
          % (float(vit["score"][0]), logz, len(segs), len(reps), len(set(pred.tolist()))))


if __name__ == "__main__":
    test_gold_forced_alignment_on_a_tiny_hand_checkable_case()
    test_the_derangement_has_no_fixed_point()
    test_sum_mode_reproduces_the_production_forward()
    test_max_mode_is_the_viterbi_of_the_same_lattice()
    test_greedy_segments_tile_the_unit_clock()
    test_the_viterbi_path_of_the_real_reverse_model_is_consistent()
    print("ALL rev_diag TESTS PASSED")
