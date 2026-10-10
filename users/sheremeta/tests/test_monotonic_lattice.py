import itertools
import unittest
from collections import Counter

import torch

from i6_experiments.users.sheremeta.transducer.monotonic_lattice import draw_paths, full_sum_loss

# (frames, labels) per sequence, one without labels and one with as many labels as frames
_CASES = [(5, 2), (3, 0), (4, 4), (6, 3)]


def _log_probs(cases, seed: int):
    """
    :return: blank and label log probs [B, T, U + 1] that leave mass for the other symbols, minus infinity outside a
        sequence and the label ones also where the prefix is complete, with the frame and label counts
    """
    torch.manual_seed(seed)
    max_frames, max_labels = max(t for t, _ in cases), max(u for _, u in cases)
    edges = torch.log_softmax(torch.randn(len(cases), max_frames, max_labels + 1, 3, dtype=torch.float64), dim=-1)
    frames = torch.arange(max_frames).view(1, -1, 1)
    prefixes = torch.arange(max_labels + 1).view(1, 1, -1)
    frame_lens = torch.tensor([t for t, _ in cases], dtype=torch.int32)
    label_lens = torch.tensor([u for _, u in cases], dtype=torch.int32)
    inside = (frames < frame_lens.view(-1, 1, 1)) & (prefixes <= label_lens.view(-1, 1, 1))
    blank_lp = edges[..., 0].masked_fill(~inside, float("-inf"))
    label_lp = edges[..., 1].masked_fill(~(inside & (prefixes < label_lens.view(-1, 1, 1))), float("-inf"))
    return blank_lp, label_lp, frame_lens, label_lens


def _path_scores(blank_lp: torch.Tensor, label_lp: torch.Tensor, num_frames: int, num_labels: int):
    """
    :return: every alignment of one sequence as the frames that emit a label, and their log probabilities
    """
    paths = list(itertools.combinations(range(num_frames), num_labels))
    scores = []
    for emits in paths:
        n, score = 0, blank_lp.new_zeros(())
        for t in range(num_frames):
            if t in emits:
                score = score + label_lp[t, n]
                n += 1
            else:
                score = score + blank_lp[t, n]
        scores.append(score)
    return paths, torch.stack(scores)


def test_full_sum_matches_the_sum_over_alignments():
    cases = _CASES + [(2, 3)]
    blank_lp, label_lp, frame_lens, label_lens = _log_probs(cases, seed=1)
    blank_lp.requires_grad_()
    label_lp.requires_grad_()
    want = torch.stack(
        [
            -_path_scores(blank_lp[b], label_lp[b], t, u)[1].logsumexp(0) if u <= t else blank_lp.new_zeros(())
            for b, (t, u) in enumerate(cases)
        ]
    )
    grad_out = torch.randn(len(cases), dtype=torch.float64)
    want_grads = torch.autograd.grad(want, (blank_lp, label_lp), grad_out)
    got = full_sum_loss(blank_lp, label_lp, frame_lens, label_lens)
    torch.testing.assert_close(got, want)
    for got_grad, want_grad in zip(torch.autograd.grad(got, (blank_lp, label_lp), grad_out), want_grads):
        torch.testing.assert_close(got_grad, want_grad)


def test_drawn_paths_follow_the_posterior():
    blank_lp, label_lp, frame_lens, label_lens = _log_probs(_CASES, seed=2)
    num_paths = 20_000
    torch.manual_seed(3)
    emits = draw_paths(blank_lp, label_lp, frame_lens, label_lens, num_paths=num_paths)
    assert emits.shape == (len(_CASES), num_paths, blank_lp.shape[1]), emits.shape
    for b, (t, u) in enumerate(_CASES):
        paths, scores = _path_scores(blank_lp[b], label_lp[b], t, u)
        posterior = (scores - scores.logsumexp(0)).exp()
        drawn = Counter(tuple(row.nonzero().flatten().tolist()) for row in emits[b])
        assert set(drawn) <= set(paths), set(drawn) - set(paths)
        got = torch.tensor([drawn[path] / num_paths for path in paths], dtype=torch.float64)
        torch.testing.assert_close(got, posterior, rtol=0, atol=0.02, msg=f"sequence {b}")


def test_kernel_draws_the_paths_of_the_reference():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    blank_lp, label_lp, frame_lens, label_lens = _log_probs(_CASES, seed=4)
    num_paths = 64
    uniform = torch.rand(len(_CASES), num_paths, blank_lp.shape[1], dtype=torch.float64)
    args = (blank_lp.float(), label_lp.float(), frame_lens, label_lens)
    want = draw_paths(*args, num_paths=num_paths, uniform=uniform.float())
    got = draw_paths(*(x.cuda() for x in args), num_paths=num_paths, uniform=uniform.float().cuda())
    torch.testing.assert_close(got.cpu(), want)
