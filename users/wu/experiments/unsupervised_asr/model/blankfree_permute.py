"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_permute.py.

emc.blankfree_permute -- the frame permutation of SAE_4A_attrib step 6, control E4.

E4 is the DESTROYED-STRUCTURE null of the step-6 pre-registration: the same arm, the same data, the
same schedule and the same weights, with the temporal order of every per-frame stream destroyed.  A
criterion that rises just as far on frame-permuted observations has not learned the temporal
structure it is credited with, so the arm's own rise is not evidence of content.

WHAT IS DESTROYED, AND WHAT IS NOT (pre-registered; implemented exactly here)

* ONE permutation per utterance, applied to EVERY per-frame stream of that utterance, so the
  frame-level pairs stay intact: after the permutation the feature frame and the reverse-model unit
  at position p are the pair that sat together at position ``pi(p)`` before it.  Only the temporal
  ORDER is destroyed.
* The permutation is drawn from the SEED and the UTTERANCE TAG alone
  (``RandomState(seed ^ crc32(tag))``), so it is the same permutation every time the utterance is
  seen -- across steps, sub-epochs and restarts -- and a different one for every other utterance.
* It runs INSIDE the valid length only: ``index[b, t] = t`` for ``t >= lens[b]``, so padding is
  untouched, the lengths are unchanged and every per-utterance marginal (the unigram counts of the
  unit stream, the feature moments, the frame count, eta, ``original_length``) is exactly what it
  was.

THE CLOCK (the question the step-6 brief asks this module to resolve) -- PER FRAME, s = 1.

In this bed the FEATURES and the UNITS are on ONE shared clock: both are the retained 50 Hz streams
``BlankfreeVadHdfJob`` writes, the blank-free train step asserts ``lengths == unit_lens`` frame for
frame, and no per-frame mask reaches the step (the rVAD mask was already applied upstream, and
every mask inside the step is derived from ``lens``, hence invariant under a within-length
permutation).  So the first case of the brief applies and the permutation is PER FRAME.

The recognizer's stride 3 does NOT make this the block case.  The stride is internal to the
recognizer: it takes the 50 Hz features and emits ``ceil(T / 3)`` posterior frames, which the
lattice pairs with the unit positions around ``3 t`` (``lattice.LatticeConfig.recognizer_stride``).
Because the features and the units take the SAME permutation on their SHARED clock, that pairing is
preserved exactly as it is without the permutation: the conv window centred on input frame ``3 t``
and the unit window centred on ``3 t`` still hold the same (feature, unit) pairs they would hold at
any other ordering of the stream.  Permuting blocks of 3 instead would additionally PRESERVE the
order inside each block (60 ms of intact local structure), i.e. destroy less than the control is
pre-registered to destroy.

Not used by any eval or decode step: the blank-free forward/decode path (``theta_checkpoint`` ->
``eval_jobs.posterior_dump`` -> the greedy PER) runs the recognizer alone, never reads
``permute_frames_seed`` and never calls this module.  RETURNN's cross-validation pass runs the
TRAIN step, so it sees the permutation too -- deliberately: the arm's dev column then reports the
same objective the arm is trained on, and reading it against an unpermuted one would be reading two
different quantities.  (The order-3 diagnostics the step-6 read is taken from, ``agg_ce_tri`` and
its floor, are marked only while ``BlankfreeAggLoss`` is in training mode, so the KL3 read is the
train column either way.)
"""

from __future__ import annotations

import zlib
from typing import Iterable, List, Sequence

import numpy as np
import torch

__all__ = ["utterance_permutation", "permutation_index", "apply_permutation", "permute_frames"]


def utterance_permutation(tag: str, length: int, *, seed: int) -> np.ndarray:
    """The ONE permutation of an utterance's valid frames, from ``seed`` and ``tag`` alone.

    ``numpy.random.RandomState(seed ^ crc32(tag))`` -- a fixed stream per (seed, tag) pair, drawn
    from no global RNG, so it neither depends on nor disturbs the training RNG, and the utterance
    carries the same permutation in every epoch and after any restart.
    """
    n = int(length)
    assert n >= 0, n
    state = np.random.RandomState((int(seed) ^ zlib.crc32(tag.encode("utf-8"))) & 0xFFFFFFFF)
    return state.permutation(n).astype(np.int64)


def permutation_index(
    tags: Sequence[str], lengths, t_max: int, *, seed: int
) -> torch.Tensor:
    """``[B, T_max]`` gather index: a permutation inside each valid length, identity on padding."""
    lens = [int(x) for x in (lengths.tolist() if torch.is_tensor(lengths) else list(lengths))]
    assert len(lens) == len(tags), (len(lens), len(tags))
    index = np.tile(np.arange(int(t_max), dtype=np.int64), (len(lens), 1))
    for b, (tag, n) in enumerate(zip(tags, lens)):
        assert 0 <= n <= int(t_max), (tag, n, t_max)
        index[b, :n] = utterance_permutation(tag, n, seed=seed)
    return torch.from_numpy(index)


def apply_permutation(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Reorder the TIME axis of one ``[B, T, ...]`` stream by ``index`` ``[B, T]``."""
    idx = index.to(device=x.device)
    assert tuple(x.shape[:2]) == tuple(idx.shape), (tuple(x.shape), tuple(idx.shape))
    while idx.dim() < x.dim():
        idx = idx.unsqueeze(-1)
    return torch.gather(x, 1, idx.expand(x.shape))


def permute_frames(
    streams: Iterable[torch.Tensor], *, tags: Sequence[str], lengths, seed: int
) -> List[torch.Tensor]:
    """Every per-frame stream of the batch under ONE per-utterance permutation (see the module).

    ``streams`` are ``[B, T, ...]`` tensors on the SAME clock and with the SAME valid lengths -- the
    features, the unit stream and any per-frame mask.  They all receive the same index, so the
    frame-level pairing survives and only the temporal order is destroyed.
    """
    streams = list(streams)
    assert streams, "permute_frames needs at least one per-frame stream"
    t_max = int(streams[0].shape[1])
    index = permutation_index(tags, lengths, t_max, seed=seed)
    return [apply_permutation(x, index) for x in streams]
