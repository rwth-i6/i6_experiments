"""Regression tests for the batched CTC prefix scoring (recognition/ctc_prefix.py).

Checks: the exact prefix / closure probabilities against a hand enumeration, that closure and token
increments are non-positive, that an impossible prefix is reported infeasible via min_frames (not
rewarded by sentinel arithmetic), and that clamping the closure with floor=0 removes the CTC term
(so floor=0 recovers plain beam). Run: python3 tests/test_ctc_prefix.py
"""

import math

import torch

from i6_experiments.users.sheremeta.recognition import ctc_prefix

B = 1
PAD, WORD = 98, 99


def _empty_state(r0, dtype_dev):
    dev = r0.device
    return (
        r0.unsqueeze(1),  # r  (B, 1, T, 2)
        torch.zeros(B, 1, device=dev),  # s
        torch.full((B, 1), -1, dtype=torch.long, device=dev),  # last
        torch.zeros(B, 1, dtype=torch.long, device=dev),  # length
        torch.zeros(B, 1, dtype=torch.long, device=dev),  # min_frames
    )


def _extend1(x, state, tok, blank):
    r, s, last, length, mf = state
    cand = torch.full((B, 1), tok, dtype=torch.long, device=x.device)
    incr, r, s, last, length, mf, _ = ctc_prefix.extend(
        x, r, s, last, length, mf, cand, blank_id=blank, pad_id=PAD, word_id=WORD
    )
    return incr, (r, s, last, length, mf)


def test_exact_prefix_and_closure():
    # vocab a=0, b=1, blank=2; two uniform 1/3 frames; hand enumeration gives P(prefix a)=4/9,
    # P(exactly a)=3/9, closure=log(3/4)
    V, T, blank = 3, 2, 2
    x = ctc_prefix.prepare_ctc_log_probs(torch.full((B, T, V), math.log(1 / 3)),
                                 torch.tensor([T]), blank_id=blank)
    r0 = ctc_prefix.initial_r(x, blank_id=blank)
    incr, st = _extend1(x, _empty_state(r0, x), 0, blank)
    s = st[1]
    closure = ctc_prefix.close_sequence(st[0], s, torch.tensor([T - 1]))
    assert abs(s.item() - math.log(4 / 9)) < 1e-5, s.item()
    assert abs(closure.item() - math.log(3 / 4)) < 1e-5, closure.item()
    # both are non-positive
    assert incr.item() <= 1e-9 and closure.item() <= 1e-9


def test_closure_floor0_zeroes_ctc():
    # finding 1: clamping closure with floor=0 must drop the whole CTC contribution, so floor=0 is
    # plain beam. token increments are already <= 0, so they clamp to 0 too.
    V, T, blank = 3, 2, 2
    x = ctc_prefix.prepare_ctc_log_probs(torch.full((B, T, V), math.log(1 / 3)),
                                 torch.tensor([T]), blank_id=blank)
    r0 = ctc_prefix.initial_r(x, blank_id=blank)
    incr, st = _extend1(x, _empty_state(r0, x), 0, blank)
    closure = ctc_prefix.close_sequence(st[0], st[1], torch.tensor([T - 1]))
    assert torch.clamp(incr, min=0.0).item() == 0.0
    assert torch.clamp(closure, min=0.0).item() == 0.0


def test_impossible_prefix_is_infeasible_and_not_rewarded():
    # finding 2: on 4 frames aaa needs a e a e a = 5 frames. min_frames must report it impossible,
    # and no extension may hand back a positive increment.
    V, T, blank = 2, 4, 1  # a=0, blank=1
    x = ctc_prefix.prepare_ctc_log_probs(torch.full((B, T, V), math.log(1 / 2)),
                                 torch.tensor([T]), blank_id=blank)
    r0 = ctc_prefix.initial_r(x, blank_id=blank)
    st = _empty_state(r0, x)
    min_frames_seen = []
    for step_i in range(1, 5):  # a, aa, aaa, aaaa
        incr, st = _extend1(x, st, 0, blank)
        mf = st[4].item()
        min_frames_seen.append(mf)
        assert incr.item() <= 1e-9, (step_i, incr.item())  # never positive
        assert mf == step_i + (step_i - 1), (step_i, mf)  # L + (L-1) repeats for all-same
    # aaa needs 5, aaaa needs 7, both exceed the 4 available frames -> flagged impossible
    assert min_frames_seen == [1, 3, 5, 7]
    assert min_frames_seen[2] > T and min_frames_seen[3] > T
    closure = ctc_prefix.close_sequence(st[0], st[1], torch.tensor([T - 1]))
    assert closure.item() <= 1e-9  # closure of an impossible prefix is not positive


def test_exact_matches_ctc_loss():
    # summed increments + closure of a feasible target must equal its exact CTC log-prob, so the
    # non-positive clamps did not perturb the exact (floor=-inf) scoring the first-pass result uses
    torch.manual_seed(0)
    V, T, blank = 3, 5, 2  # a=0, b=1, blank=2
    logp = torch.log_softmax(torch.randn(B, T, V), dim=-1)
    x = ctc_prefix.prepare_ctc_log_probs(logp, torch.tensor([T]), blank_id=blank)
    r0 = ctc_prefix.initial_r(x, blank_id=blank)
    target = [0, 1, 0]  # aba, feasible in 5 frames
    st = _empty_state(r0, x)
    total = 0.0
    for tok in target:
        incr, st = _extend1(x, st, tok, blank)
        total += incr.item()
    total += ctc_prefix.close_sequence(st[0], st[1], torch.tensor([T - 1])).item()
    ref = -torch.nn.functional.ctc_loss(
        logp.transpose(0, 1), torch.tensor([target]),
        torch.tensor([T]), torch.tensor([len(target)]),
        blank=blank, reduction="none", zero_infinity=True,
    ).item()
    assert abs(total - ref) < 1e-4, (total, ref)


def test_pad_word_are_no_ops():
    # pad/word are valid ctc-vocab indices (masked by is_real), not out-of-range specials
    V, T, blank, pad, word = 5, 3, 4, 2, 3  # a=0, b=1, pad=2, word=3, blank=4
    x = ctc_prefix.prepare_ctc_log_probs(torch.full((B, T, V), math.log(1 / 5)),
                                 torch.tensor([T]), blank_id=blank)
    r0 = ctc_prefix.initial_r(x, blank_id=blank)
    r, s, last, length, mf = _empty_state(r0, x)
    cand = torch.zeros(B, 1, dtype=torch.long)  # real label a
    _, r, s, last, length, mf, _pc = ctc_prefix.extend(x, r, s, last, length, mf, cand,
                                          blank_id=blank, pad_id=pad, word_id=word)
    mf_a, len_a = mf.item(), length.item()
    for tok in (pad, word):
        cand = torch.full((B, 1), tok, dtype=torch.long)
        incr, r2, s2, last2, len2, mf2, _pc2 = ctc_prefix.extend(x, r, s, last, length, mf, cand,
                                                   blank_id=blank, pad_id=pad, word_id=word)
        assert incr.item() == 0.0  # no score change
        assert mf2.item() == mf_a  # min_frames unchanged
        assert len2.item() == len_a  # length unchanged


if __name__ == "__main__":
    test_exact_prefix_and_closure()
    test_closure_floor0_zeroes_ctc()
    test_impossible_prefix_is_infeasible_and_not_rewarded()
    test_exact_matches_ctc_loss()
    test_pad_word_are_no_ops()
    print("ctc_prefix tests OK")
