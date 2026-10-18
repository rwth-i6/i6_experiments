"""
How much say the emission model has, measured against the LM.

A guided run fails in a characteristic way when the acoustics cannot outvote the
other terms of the search: it converges confidently, uses every label, and is
wrong. Posterior entropy and the objective cannot tell that apart from success.
What can is a direct comparison, in nats after the AM scale, between what the
acoustics bring to a decision and what the LM charges for it:

``frame_gap``
    per frame, the cost of the second-best label minus that of the best

``segment_gap``
    the same over a true segment's summed cost - the acoustic evidence for a
    label-identity decision. Compare against the LM's pull between two
    plausible candidates (:class:`PhonemeLMCostJob` ``out_pull``).

``boundary_payback``
    for each pair of adjacent true segments A, B:
    ``min_k (S_A[k] + S_B[k]) - (min_k S_A[k] + min_k S_B[k])``, i.e. what the
    acoustics gain by putting a boundary where one belongs instead of covering
    both segments with one label. A free-segmentation search inserts that
    boundary only if this exceeds the price of one more phoneme
    (``lm_scale * LM cost`` plus the topology's transition penalty), so the
    median against the price - and ``payable_fraction``, the share of true
    boundaries that clear it - says whether the acoustics can hold the true
    segmentation at all.

High ``segment_gap`` with low ``boundary_payback`` is the signature worth
knowing: the labels are confident, but they do not separate *adjacent*
phonemes. That is what a run which merged segments early looks like - the
merged segmentation trains merged classes, and merged classes cannot pay for
the boundaries that would undo it.

The true segmentation enters only here, as an evaluation reference - the same
footing ``FrameErrorRateJob`` puts the GMM alignment on. Nothing on the
training path reads it.

Three jobs, because the expensive part does not depend on the epoch: the
densities are frozen in the configs this is for, so :class:`BalanceFramesJob`
computes the frame-to-density likelihoods once and every per-epoch
:class:`EmissionBalanceJob` only applies that epoch's mixture table.
"""

__all__ = [
    "BalanceFramesJob",
    "PhonemeLMCostJob",
    "EmissionBalanceJob",
    "emission_balance",
    "label_costs",
    "state_segment_costs",
    "load_balance_frames",
    "transition_boundary_penalty",
]

import json
import logging
import math
import pickle
from typing import Optional

import numpy as np

from sisyphus import Job, Task, tk

# Module level, not inside run(): Task.run chdirs into the job's work/ directory
# first, after which recipe-relative imports no longer resolve (see the note in
# statistics_jobs.py). This module is numpy-only, so importing it everywhere is
# cheap; the op itself is imported lazily, from an absolute path, in the one job
# that needs it.
from ..lib.guided_kmeans.backoff_fb_graph import ensure_importable, lexicon_symbols


def transition_boundary_penalty(
    num_states: int, loop_probability: float, transition_scale: float = 1.0
) -> float:
    """What the HMM topology charges for one extra phoneme, in nats.

    Over a fixed-length utterance of T frames split into n phonemes, a
    left-to-right topology with S states and loop probability p contributes

        T * log p  +  S * n * log((1 - p) / p)

    The first term is the same for every segmentation and cancels from the
    posterior; only the second depends on the path, so the topology acts on the
    segmentation purely as a per-phoneme penalty of ``S * log(p / (1 - p))``,
    scaled by ``transition_scale``. Note what this means for reading a loop
    probability: in this conditional setting it is a boundary price, not a
    duration target - ``p = 1/2`` charges nothing at all, at any S.

    :raises ValueError: for ``p`` outside (0, 1). At ``p = 0`` there is no loop
        and every frame is a boundary, so the question does not arise; the
        oracle-segmented configs, which use it, have no boundary decisions.
    """
    if not 0.0 < loop_probability < 1.0:
        raise ValueError(
            f"loop_probability must be in (0, 1) for a free segmentation, got "
            f"{loop_probability}; at 0 there are no boundary decisions to price")
    return num_states * transition_scale * math.log(loop_probability / (1.0 - loop_probability))


def label_costs(shifted: np.ndarray, offset: np.ndarray, mixtures: np.ndarray) -> np.ndarray:
    """``[F, L]`` label costs in nats, before any AM scale.

    :param shifted: ``exp(-(density_cost - offset))``, ``[F, C]``, as written by
        :class:`BalanceFramesJob`
    :param offset: the per-frame minimum density cost, ``[F]``
    :param mixtures: ``p(density | label)``, ``[L, C]``
    """
    mass = np.asarray(shifted, dtype=np.float64) @ np.asarray(mixtures, dtype=np.float64).T
    return -np.log(np.maximum(mass, 1e-300)) + np.asarray(offset, dtype=np.float64)[:, None]


def state_segment_costs(
    costs: np.ndarray, starts: np.ndarray, ends: np.ndarray, num_states: int
) -> np.ndarray:
    """``[N, V]``: each span's cost under each phoneme with untied states.

    With one emission per state, a phoneme's cost over a span is not a sum but
    the best left-to-right pass through its ``S`` states - each state one or
    more frames, entered in order, ending in the last - which is what the
    search pays for that span under that phoneme, transitions aside. A dynamic
    programme over the span, run for all spans at once and stepping through
    frame offsets, so the Python loop is as long as the longest span.

    A span shorter than ``S`` frames cannot visit every state; there the best
    pass through the states it can reach is taken, rather than an infinite
    cost that would make every median over spans meaningless. The search
    itself would have to stretch a neighbour over such a span, so this is
    the one place this measure is kinder than the model.

    :param costs: ``[F, V * S]`` label costs, phoneme-major
        (:class:`...lib.guided_kmeans.backoff_fb_graph.LabelLayout`)
    :param starts: ``[N]`` first frame of each span
    :param ends: ``[N]`` one past its last frame
    :param num_states: ``S``
    """
    F, L = costs.shape
    if L % num_states:
        raise ValueError(f"{L} label columns is not a multiple of {num_states} states")
    per_state = costs.reshape(F, L // num_states, num_states)
    lengths = np.asarray(ends) - np.asarray(starts)
    if (lengths < 1).any():
        raise ValueError("every span needs at least one frame")
    best = per_state[starts].copy()
    best[:, :, 1:] = np.inf  # a pass starts in the first state
    result = np.where(lengths[:, None] < num_states, best.min(axis=2), best[:, :, -1])
    for offset in range(1, int(lengths.max())):
        active = np.flatnonzero(lengths > offset)
        prev = best[active]
        # Stay, or step up one state from the left.
        stepped = np.concatenate(
            [np.full_like(prev[:, :, :1], np.inf), prev[:, :, :-1]], axis=2)
        best[active] = per_state[starts[active] + offset] + np.minimum(prev, stepped)
        last = active[lengths[active] == offset + 1]
        if last.size:
            short = lengths[last] < num_states
            result[last] = np.where(
                short[:, None], best[last].min(axis=2), best[last][:, :, -1])
    return result


def emission_balance(
    costs: np.ndarray,
    segment_starts: np.ndarray,
    adjacent: np.ndarray,
    am_scale: float,
    boundary_price: float,
    num_untied_states: int = 1,
) -> dict:
    """The balance statistics for one model, from its label costs.

    :param costs: ``[F, L]`` label costs, :func:`label_costs`
    :param segment_starts: first frame of every true segment, ascending, from 0;
        the segments must tile all F frames
    :param adjacent: ``[num_segments - 1]`` bool, True where segment i and i+1
        belong to the same sequence (a boundary between sequences is not one the
        search ever has to pay for)
    :param am_scale: the run's training AM scale; everything is reported after it
    :param boundary_price: nats a search charges for one extra phoneme
    :param num_untied_states: 1 when the columns are phonemes. ``S > 1`` when
        they are the untied states of a ``V * S`` layout: every quantity is
        then still per *phoneme* - a frame's phoneme cost is its best state's,
        a segment's is the best pass through the states
        (:func:`state_segment_costs`), and so is the cost of covering two
        segments with one phoneme - so the numbers read like a tied run's.
    """
    if segment_starts[0] != 0:
        raise ValueError("segment_starts must begin at frame 0")
    if len(adjacent) != len(segment_starts) - 1:
        raise ValueError(
            f"adjacent has {len(adjacent)} entries for {len(segment_starts)} segments")

    if num_untied_states > 1:
        S = num_untied_states
        ends = np.r_[segment_starts[1:], costs.shape[0]]
        frame_costs = costs.reshape(costs.shape[0], -1, S).min(axis=2)
        seg = state_segment_costs(costs, segment_starts, ends, S)
        # Two adjacent segments under one phoneme is one pass over their union,
        # not the sum of two passes - that would be two phonemes.
        merged = state_segment_costs(
            costs, segment_starts[:-1][adjacent], ends[1:][adjacent], S)
    else:
        frame_costs = costs
        seg = np.add.reduceat(costs, segment_starts, axis=0)
        merged = None

    frame_sorted = np.sort(frame_costs, axis=1)
    frame_gap = am_scale * np.median(frame_sorted[:, 1] - frame_sorted[:, 0])

    seg_sorted = np.sort(seg, axis=1)
    segment_gap = am_scale * np.median(seg_sorted[:, 1] - seg_sorted[:, 0])

    a, b = seg[:-1][adjacent], seg[1:][adjacent]
    together = (a + b) if merged is None else merged
    payback = am_scale * (together.min(axis=1) - (a.min(axis=1) + b.min(axis=1)))
    return {
        "frame_gap": float(frame_gap),
        "segment_gap": float(segment_gap),
        "boundary_payback": float(np.median(payback)),
        "boundary_payback_q25": float(np.percentile(payback, 25)),
        "boundary_payback_q75": float(np.percentile(payback, 75)),
        "payable_fraction": float(np.mean(payback > boundary_price)),
        "boundary_price": float(boundary_price),
        "num_frames": int(costs.shape[0]),
        "num_segments": int(len(segment_starts)),
        "num_boundaries": int(adjacent.sum()),
    }


def load_balance_frames(
    features_hdf: str, frame_labels: str, centroids: str, covs: str, num_seqs: int
) -> dict:
    """What :class:`BalanceFramesJob` stores: ``shifted``, ``offset``, ``starts``
    and ``adjacent`` for the first ``num_seqs`` sequences. See that job."""
    import h5py

    with h5py.File(features_hdf, "r") as f:
        lengths = np.asarray(f["seqLengths"][:num_seqs, 0], dtype=np.int64)
        tags = [t.decode() if isinstance(t, bytes) else str(t) for t in f["seqTags"][:num_seqs]]
        frames = np.asarray(f["inputs"][: int(lengths.sum())], dtype=np.float64)

    with open(frame_labels, "rb") as fp:
        labels = pickle.load(fp)
    starts, seq_index = [], []
    offset = 0
    for index, (tag, length) in enumerate(zip(tags, lengths)):
        lab = np.asarray(labels[tag])
        if len(lab) != length:
            raise ValueError(f"{tag}: {len(lab)} labels for {length} frames")
        first = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        starts.extend(offset + first)
        seq_index.extend([index] * len(first))
        offset += int(length)
    starts = np.asarray(starts, dtype=np.int64)
    seq_index = np.asarray(seq_index)

    means = np.load(centroids).astype(np.float64)
    cov_stack = np.load(covs, mmap_mode="r")
    # A sample of slices rather than all of them: the file is 1 GB, and the
    # failure this guards against (a per-density covariance) shows in any slice.
    for k in np.linspace(0, len(cov_stack) - 1, 8).astype(int):
        if not np.array_equal(np.asarray(cov_stack[k]), np.asarray(cov_stack[0])):
            raise ValueError(
                f"covariance {k} differs from covariance 0; this assumes one shared "
                f"covariance (frozen densities over DuplicateCovsJob)")
    chol = np.linalg.cholesky(np.asarray(cov_stack[0], dtype=np.float64))
    fw = np.linalg.solve(chol, frames.T).T
    mw = np.linalg.solve(chol, means.T).T
    cost = 0.5 * ((fw * fw).sum(1)[:, None] - 2.0 * fw @ mw.T + (mw * mw).sum(1)[None, :])
    offset_cost = cost.min(axis=1)
    return {
        # float32 halves the file; the label costs are rebuilt in float64 and the
        # gaps read here are four orders of magnitude above float32 resolution.
        "shifted": np.exp(-(cost - offset_cost[:, None])).astype(np.float32),
        "offset": offset_cost,
        "starts": starts,
        "adjacent": seq_index[1:] == seq_index[:-1],
    }


class BalanceFramesJob(Job):
    """Frame-to-density likelihoods and the true segmentation, once.

    Reads the first ``num_seqs`` sequences, whitens them with the shared
    covariance and stores ``exp(-(cost - min cost))`` per frame and density, so
    that every per-epoch :class:`EmissionBalanceJob` reduces to one small matrix
    product. Requires the densities to be frozen and to share one covariance -
    true of every weights-only config in this family (``update_densities=False``
    over ``DuplicateCovsJob`` covariances) and checked here rather than assumed,
    because a general per-density Mahalanobis would be 512x the work and is not
    what this job is for.

    :param features_hdf: the frames the run trains on
    :param frame_labels: ``{seq_tag: per-frame labels}`` pickle giving the true
        segmentation, e.g. ``SegmentedFeaturesFromAlignmentJob.out_labels``
    :param centroids: the frozen density means, ``[C, D]``
    :param covs: the density covariances, ``[C, D, D]``, all identical
    :param num_seqs: sequences to evaluate on. 60 gives ~27k frames and ~7k true
        boundaries, which pins the medians down well below the differences this
        is used to read, and keeps the stored matrix at ~55 MB.
    """

    def __init__(
        self,
        features_hdf: tk.Path,
        frame_labels: tk.Path,
        centroids: tk.Path,
        covs: tk.Path,
        num_seqs: int = 60,
    ):
        self.features_hdf = features_hdf
        self.frame_labels = frame_labels
        self.centroids = centroids
        self.covs = covs
        self.num_seqs = num_seqs

        self.out_frames = self.output_path("frames.npz")

        # Reads ~170 MB and does a [27k, 512] x [512, 512] whitening: a compute
        # node's I/O is several times the login host's on this filesystem, and it
        # only ever runs once per (features, densities) pair.
        self.rqmt = {"cpu": 4, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        data = load_balance_frames(
            self.features_hdf.get_path(), self.frame_labels.get_path(),
            self.centroids.get_path(), self.covs.get_path(), self.num_seqs)
        np.savez(self.out_frames.get_path(), **data)
        logging.info(
            "%d frames, %d true segments (%.2f frames each), %d boundaries inside "
            "sequences", len(data["offset"]), len(data["starts"]),
            len(data["offset"]) / len(data["starts"]), data["adjacent"].sum())


class PhonemeLMCostJob(Job):
    """What the phoneme LM charges, on its own typical histories.

    Samples sequences from the LM and reports, averaged over the visited
    histories:

    * ``out_cost``: the cost of one phoneme, ``-log P(a | h)`` for the sampled
      ``a`` - the price a search pays per phoneme at ``lm_scale = 1``, i.e. the
      LM's contribution to :func:`emission_balance`'s ``boundary_price``
    * ``out_pull``: the median ``|log P(a | h) - log P(b | h)|`` for two
      candidates drawn independently from ``P(. | h)`` - how hard the LM pushes
      between two labels it finds plausible, which is the number a label-identity
      decision's ``segment_gap`` has to beat

    Sampled rather than taken over arbitrary histories, because arbitrary
    histories are mostly ones the LM never sees and are far less predictable: on
    this 5-gram they give 2.52 nats per phoneme against 1.89 on typical ones.

    Runs once. Not a mini task: loading the ARPA goes through the op's package,
    which brings in its CUDA extension, and that is only built for the job
    container's torch.
    """

    def __init__(
        self,
        arpa_path: tk.Path,
        lexicon: tk.Path,
        backoff_fb_root: tk.Path,
        num_sequences: int = 200,
        length: int = 40,
        seed: int = 0,
    ):
        self.arpa_path = arpa_path
        self.lexicon = lexicon
        self.backoff_fb_root = backoff_fb_root
        self.num_sequences = num_sequences
        self.length = length
        self.seed = seed

        self.out_cost = self.output_var("phoneme_cost")
        self.out_pull = self.output_var("candidate_pull")
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"cpu": 1, "mem": 4, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        ensure_importable(self.backoff_fb_root.get_path())
        import backoff_fb

        symbols = lexicon_symbols(self.lexicon.get_path())
        lm = backoff_fb.load_arpa(self.arpa_path.get_path(), symbols)
        V = len(symbols)
        context = lm.order - 1
        rng = np.random.default_rng(self.seed)
        costs, entropies, pulls = [], [], []
        for _ in range(self.num_sequences):
            history = (lm.BOS,)
            for _ in range(self.length):
                logp = np.array([lm.logprob(a, history[-context:]) for a in range(V)])
                # Renormalised over the labels: the LM's remaining mass sits on
                # </s> and on the dropped <unk>, neither of which is a label.
                p = np.exp(logp)
                p /= p.sum()
                logp = np.log(p)
                a, b = (int(i) for i in rng.choice(V, size=2, p=p))
                costs.append(-logp[a])
                entropies.append(float(-(p * logp).sum()))
                pulls.append(abs(logp[a] - logp[b]))
                history = history + (a,)
        summary = {
            "phoneme_cost": float(np.mean(costs)),
            "entropy_rate": float(np.mean(entropies)),
            "candidate_pull": float(np.median(pulls)),
            "num_positions": len(costs),
            "lm_order": lm.order,
        }
        with open(self.out_summary.get_path(), "w") as fp:
            json.dump(summary, fp, indent=4)
        self.out_cost.set(summary["phoneme_cost"])
        self.out_pull.set(summary["candidate_pull"])
        logging.info(
            "one phoneme costs %.3f nats on typical histories (entropy rate %.3f); "
            "two plausible candidates differ by %.3f", summary["phoneme_cost"],
            summary["entropy_rate"], summary["candidate_pull"])


class EmissionBalanceJob(Job):
    """:func:`emission_balance` for one epoch's mixture table.

    A separate small job, like :class:`.statistics_jobs.MixtureDiagnosticsJob`:
    it keeps the diagnostic out of the expensive epoch job's hash, and can be
    pointed at runs that finished before it existed.

    :param frames: :class:`BalanceFramesJob` ``out_frames``
    :param mixtures: one epoch's ``mixtures.npy``,
        ``result.out_artifacts["mixtures"][epoch]``
    :param am_scale: the run's training AM scale
    :param lm_cost: :class:`PhonemeLMCostJob` ``out_cost``
    :param lm_scale: the run's training LM scale
    :param transition_penalty: the topology's own price per phoneme,
        :func:`transition_boundary_penalty`; 0 for a topology that charges
        nothing (S=2 at p=1/2) and for the oracle-segmented runs, where no
        boundary decision is ever made and the number answers "could this model
        pay for boundaries at this LM scale"
    :param num_untied_states: ``S`` for a model whose table rows are untied HMM
        states (``V * S`` rows); 1, the default and hash-excluded, for one row
        per phoneme. See :func:`emission_balance`.
    """

    __sis_hash_exclude__ = {"num_untied_states": 1}

    def __init__(
        self,
        frames: tk.Path,
        mixtures: tk.Path,
        am_scale: float,
        lm_cost: tk.Variable,
        lm_scale: float = 1.0,
        transition_penalty: float = 0.0,
        num_untied_states: int = 1,
    ):
        self.frames = frames
        self.num_untied_states = num_untied_states
        self.mixtures = mixtures
        self.am_scale = am_scale
        self.lm_cost = lm_cost
        self.lm_scale = lm_scale
        self.transition_penalty = transition_penalty

        self.out_scalars = self.output_var("scalars")
        self.out_frame_gap = self.output_var("frame_gap")
        self.out_segment_gap = self.output_var("segment_gap")
        self.out_payback = self.output_var("boundary_payback")
        self.out_payable = self.output_var("payable_fraction")
        self.out_price = self.output_var("boundary_price")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        data = np.load(self.frames.get_path())
        costs = label_costs(data["shifted"], data["offset"], np.load(self.mixtures.get_path()))
        price = self.lm_scale * float(self.lm_cost.get()) + self.transition_penalty
        stats = emission_balance(
            costs, data["starts"], data["adjacent"], self.am_scale, price,
            # getattr: a job pickled before untied states existed.
            num_untied_states=getattr(self, "num_untied_states", 1))
        self.out_scalars.set(stats)
        self.out_frame_gap.set(stats["frame_gap"])
        self.out_segment_gap.set(stats["segment_gap"])
        self.out_payback.set(stats["boundary_payback"])
        self.out_payable.set(stats["payable_fraction"])
        self.out_price.set(stats["boundary_price"])
        logging.info(
            "segment gap %.3f, boundary payback %.3f against a price of %.3f: "
            "%.1f%% of true boundaries payable", stats["segment_gap"],
            stats["boundary_payback"], price, 100 * stats["payable_fraction"])
