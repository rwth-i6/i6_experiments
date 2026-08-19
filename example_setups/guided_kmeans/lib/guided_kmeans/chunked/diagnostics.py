"""
Raw per-frame and per-segment records of one recognition pass, for offline
analysis of *why* a clustering run behaves the way it does.

The statistics counters in :mod:`..statistics` answer "what is the corpus
average"; this answers "which sequences are the outliers, and what do their
frames look like". It therefore dumps raw records rather than aggregates, and
leaves every histogram, threshold and correlation to a notebook - deciding
what counts as an outlier is exactly the part that should not be baked into a
job that takes half an hour to re-run.

What is deliberately *not* recorded is the features themselves: at 768
dimensions the encoder outputs of ls-100 are tens of gigabytes, against ~360 MB
for the five per-frame scalars below. Nothing is lost for the question at hand,
because ``scores[t, label[t]]`` already *is* the distance from frame ``t`` to
the centroid it was aligned with - squared Euclidean under
:class:`.models.EuclideanModel`, Mahalanobis under
:class:`.models.GaussianModel` - so the "how far away is the frame from the
cluster it landed in" question is answered by a lookup, with no recomputation
and no second pass over the data.

The two axes are joined by the sequence tag, which is what makes "are the
sequences with outlying recognition scores the same ones whose frames sit far
from their centroids" a two-line question on the loaded arrays.
"""

from __future__ import annotations

__all__ = [
    "FrameDiagnostics",
    "Diagnostics",
    "load_diagnostics",
    "META_NAME",
]

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .interfaces import Posteriors, as_hard_labels

#: Written next to the per-chunk ``.npz`` files by whoever ran the probe; see
#: :class:`...setup.diagnostics.ClusteringDiagnosticsJob`. Optional - a dump
#: without it loads fine, only without label names.
META_NAME = "meta.json"

_FRAME_FIELDS = (
    "frame_label",
    "frame_assigned_cost",
    "frame_best_label",
    "frame_best_cost",
    "frame_second_best_cost",
)
_SEGMENT_FIELDS = (
    "seg_label",
    "seg_start",
    "seg_end",
    "seg_am",
    "seg_lm",
    "seg_transition",
)
_SEQUENCE_FIELDS = (
    "seq_tag",
    "seq_num_frames",
    "seq_num_segments",
    "seq_frame_offset",
    "seq_segment_offset",
)


class FrameDiagnostics:
    """
    A :class:`.interfaces.Probe` that records, per frame, how the alignment
    relates to the model that scored it, and per traceback segment, the scores
    RASR attached to it.

    Per frame (``[F]``, in sequence order):

    ``frame_label``
        the cluster the frame was aligned to
    ``frame_assigned_cost``
        ``scores[t, frame_label[t]]`` - the distance to that cluster
    ``frame_best_label`` / ``frame_best_cost``
        the cluster the frame would have picked on its own, and its distance
    ``frame_second_best_cost``
        the runner-up distance, so per-frame confidence is available without
        keeping the full ``[T, K]`` matrix

    The gap ``frame_assigned_cost - frame_best_cost`` is the quantity the
    guiding actually contributes: how far the language and transition model
    pulled a frame away from its nearest centroid. It is zero exactly where the
    search agreed with plain nearest-centroid assignment.

    Per traceback segment (``[G]``) the label, its frame span and the three
    RASR scores are stored verbatim, with no differencing or normalization
    applied - see :meth:`Diagnostics.sequence_table` for why that matters.

    Scores are kept in float64 while frame costs are float32: the traceback
    scores appear to be accumulated along the sequence (both
    ``_traceback_to_score`` and ``ScoreStatisticsCounter`` read the totals off
    the *last* item), and recovering a per-segment contribution from cumulative
    values of order 1e5 by differencing loses about a digit too many in
    float32.
    """

    def __init__(self):
        self._seq_tag: List[str] = []
        self._seq_num_frames: List[int] = []
        self._seq_num_segments: List[int] = []
        self._frames: Dict[str, List[np.ndarray]] = {name: [] for name in _FRAME_FIELDS}
        self._segments: Dict[str, List[np.ndarray]] = {name: [] for name in _SEGMENT_FIELDS}

    def observe(
        self,
        *,
        seq_tag: str,
        features: np.ndarray,
        scores: np.ndarray,
        posteriors: Posteriors,
        traceback: List[Any],
    ) -> None:
        labels = np.asarray(as_hard_labels(posteriors), dtype=np.int64)
        scores = np.asarray(scores)
        if scores.ndim != 2:
            raise ValueError(f"{seq_tag}: expected a [T, K] score matrix, got {scores.shape}")
        num_frames, num_clusters = scores.shape
        if len(labels) != num_frames:
            raise ValueError(
                f"{seq_tag}: {len(labels)} labels for a [{num_frames}, {num_clusters}] "
                f"score matrix"
            )
        if num_frames and (labels.min() < 0 or labels.max() >= num_clusters):
            raise ValueError(
                f"{seq_tag}: label out of range for {num_clusters} clusters "
                f"(got {labels.min()}..{labels.max()})"
            )

        rows = np.arange(num_frames)
        best_label = scores.argmin(axis=1) if num_frames else np.zeros(0, dtype=np.int64)
        if num_clusters >= 2:
            second_best = np.partition(scores, 1, axis=1)[:, 1]
        else:
            # A single-cluster model has no runner-up; inf keeps any margin
            # computed downstream well defined rather than silently zero.
            second_best = np.full(num_frames, np.inf)

        self._frames["frame_label"].append(labels.astype(np.int32))
        self._frames["frame_assigned_cost"].append(scores[rows, labels].astype(np.float32))
        self._frames["frame_best_label"].append(np.asarray(best_label, dtype=np.int32))
        self._frames["frame_best_cost"].append(scores[rows, best_label].astype(np.float32))
        self._frames["frame_second_best_cost"].append(second_best.astype(np.float32))

        # The segment label is read off the frame axis rather than through a
        # lexicon lookup, so the two axes are guaranteed to live in the same
        # index space and this class stays free of any notion of what a label
        # means - the same reason transcription is a caller-supplied hook in
        # run_chunk. -1 marks a segment whose span falls outside the frames,
        # which should not happen and is worth seeing rather than crashing on.
        starts = np.array([int(item.start_time) for item in traceback], dtype=np.int32)
        ends = np.array([int(item.end_time) for item in traceback], dtype=np.int32)
        if num_frames:
            in_range = (starts >= 0) & (starts < num_frames)
            seg_labels = np.where(in_range, labels[np.clip(starts, 0, num_frames - 1)], -1)
        else:
            seg_labels = np.full(len(traceback), -1, dtype=np.int64)

        self._segments["seg_label"].append(np.asarray(seg_labels, dtype=np.int32))
        self._segments["seg_start"].append(starts)
        self._segments["seg_end"].append(ends)
        for name, attr in (
            ("seg_am", "am_score"),
            ("seg_lm", "lm_score"),
            ("seg_transition", "transition_score"),
        ):
            self._segments[name].append(
                np.array([float(getattr(item, attr, 0.0)) for item in traceback], dtype=np.float64)
            )

        self._seq_tag.append(seq_tag)
        self._seq_num_frames.append(num_frames)
        self._seq_num_segments.append(len(traceback))

    def arrays(self) -> Dict[str, np.ndarray]:
        """
        The recorded data as flat arrays plus the offsets that slice them back
        into sequences. Offsets are derived here rather than tracked
        incrementally so they cannot drift from the data they index.
        """
        num_frames = np.asarray(self._seq_num_frames, dtype=np.int64)
        num_segments = np.asarray(self._seq_num_segments, dtype=np.int64)
        out: Dict[str, np.ndarray] = {
            "seq_tag": np.asarray(self._seq_tag, dtype=np.str_),
            "seq_num_frames": num_frames.astype(np.int32),
            "seq_num_segments": num_segments.astype(np.int32),
            "seq_frame_offset": _offsets(num_frames),
            "seq_segment_offset": _offsets(num_segments),
        }
        for name, dtype in zip(_FRAME_FIELDS, (np.int32, np.float32, np.int32, np.float32, np.float32)):
            out[name] = _concat(self._frames[name], dtype)
        for name, dtype in zip(_SEGMENT_FIELDS, (np.int32, np.int32, np.int32, np.float64, np.float64, np.float64)):
            out[name] = _concat(self._segments[name], dtype)
        return out

    def save(self, path: str) -> None:
        """
        Write this chunk's records to ``path`` (an ``.npz``).

        Compressed because the label and span arrays are highly redundant and
        this runs once per chunk against half an hour of RASR search - the few
        seconds it costs are not measurable next to that.
        """
        with open(path, "wb") as fp:
            np.savez_compressed(fp, **self.arrays())


def _offsets(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out[:-1]


def _concat(parts: Sequence[np.ndarray], dtype) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


@dataclass
class Diagnostics:
    """
    One recognition pass' records, concatenated over chunks.

    The per-frame and per-segment arrays are flat; ``seq_frame_offset`` /
    ``seq_segment_offset`` together with the corresponding counts slice them
    back into sequences, and :meth:`frames_of` / :meth:`segments_of` do that
    for a single tag.
    """

    seq_tag: np.ndarray
    seq_num_frames: np.ndarray
    seq_num_segments: np.ndarray
    seq_frame_offset: np.ndarray
    seq_segment_offset: np.ndarray

    frame_label: np.ndarray
    frame_assigned_cost: np.ndarray
    frame_best_label: np.ndarray
    frame_best_cost: np.ndarray
    frame_second_best_cost: np.ndarray

    seg_label: np.ndarray
    seg_start: np.ndarray
    seg_end: np.ndarray
    seg_am: np.ndarray
    seg_lm: np.ndarray
    seg_transition: np.ndarray

    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_sequences(self) -> int:
        return len(self.seq_tag)

    @property
    def frame_margin(self) -> np.ndarray:
        """
        Per frame: how much worse the aligned cluster is than the nearest one.
        Zero where the search agreed with nearest-centroid assignment.
        """
        return self.frame_assigned_cost - self.frame_best_cost

    @property
    def labels(self) -> Optional[List[str]]:
        """Cluster index -> phoneme, if the dump carries a label inventory."""
        return self.meta.get("labels")

    def index_of(self, seq_tag: str) -> int:
        matches = np.flatnonzero(self.seq_tag == seq_tag)
        if len(matches) != 1:
            raise KeyError(f"{seq_tag!r} appears {len(matches)} times in this dump")
        return int(matches[0])

    def frames_of(self, seq_tag: str) -> Dict[str, np.ndarray]:
        """Every per-frame array restricted to one sequence."""
        i = self.index_of(seq_tag)
        start = int(self.seq_frame_offset[i])
        stop = start + int(self.seq_num_frames[i])
        arrays = {name: getattr(self, name)[start:stop] for name in _FRAME_FIELDS}
        arrays["frame_margin"] = self.frame_margin[start:stop]
        return arrays

    def segments_of(self, seq_tag: str) -> Dict[str, np.ndarray]:
        """Every per-segment array restricted to one sequence."""
        i = self.index_of(seq_tag)
        start = int(self.seq_segment_offset[i])
        stop = start + int(self.seq_num_segments[i])
        return {name: getattr(self, name)[start:stop] for name in _SEGMENT_FIELDS}

    def sequence_table(self) -> Dict[str, np.ndarray]:
        """
        Per-sequence aggregates, as a dict of equal-length arrays (feed it
        straight to ``pandas.DataFrame``).

        The recognition scores come in two forms because the traceback's score
        semantics are an assumption, not a documented contract: the rest of
        this code base reads a sequence total off the *last* traceback item
        (``_traceback_to_score``, ``ScoreStatisticsCounter``), which is only
        right if RASR accumulates them along the traceback. Both readings are
        provided - ``*_last`` and ``*_sum`` - so the dump settles the question
        instead of inheriting the assumption: if the scores are cumulative the
        two disagree and ``*_last`` is the total; if they are per-segment,
        ``*_sum`` is, and ``*_last`` is just the final phoneme.

        Everything is also given per frame (``*_per_frame``), which is the form
        to histogram. Unnormalized totals grow with utterance length, so their
        upper tail is a list of long utterances, not of badly scoring ones -
        the same reason ``ScoreStatisticsCounter`` keeps
        ``average_total_normed_score`` next to the raw average.
        """
        offsets, counts = self.seq_segment_offset, self.seq_num_segments.astype(np.int64)
        num_frames = self.seq_num_frames.astype(np.float64)
        # Guard the division rather than the data: a zero-frame sequence would
        # be a broken HDF entry, but it should show up as nan in the table, not
        # abort the analysis of the other 28k sequences.
        per_frame = np.divide(
            1.0, num_frames, out=np.full(len(num_frames), np.nan), where=num_frames > 0
        )
        # An empty traceback has no last item; nan keeps it in the table as the
        # anomaly it is instead of dropping the sequence. The index is clipped
        # rather than left negative so that a dump in which *every* sequence
        # came back empty - a whole pass the search produced nothing for, which
        # is a thing worth being able to look at - indexes nothing out of range.
        has_last = counts > 0
        num_segments_total = len(self.seg_am)
        last = np.clip(offsets + counts - 1, 0, max(num_segments_total - 1, 0))

        table: Dict[str, np.ndarray] = {
            "seq_tag": self.seq_tag,
            "num_frames": self.seq_num_frames,
            "num_segments": self.seq_num_segments,
        }
        totals_last, totals_sum = None, None
        for name, values in (
            ("am", self.seg_am),
            ("lm", self.seg_lm),
            ("transition", self.seg_transition),
        ):
            as_last = (
                np.where(has_last, values[last], np.nan)
                if num_segments_total
                else np.full(len(counts), np.nan)
            )
            as_sum = _group_sum(values, offsets, counts)
            table[f"{name}_last"] = as_last
            table[f"{name}_sum"] = as_sum
            totals_last = as_last if totals_last is None else totals_last + as_last
            totals_sum = as_sum if totals_sum is None else totals_sum + as_sum
        table["total_last"] = totals_last
        table["total_sum"] = totals_sum
        for name in ("am", "lm", "transition", "total"):
            for form in ("last", "sum"):
                table[f"{name}_{form}_per_frame"] = table[f"{name}_{form}"] * per_frame

        frame_offsets = self.seq_frame_offset
        frame_counts = self.seq_num_frames.astype(np.int64)
        for name, values in (
            ("assigned_cost", self.frame_assigned_cost),
            ("best_cost", self.frame_best_cost),
            ("margin", self.frame_margin),
        ):
            table[f"mean_{name}"] = (
                _group_sum(values.astype(np.float64), frame_offsets, frame_counts) * per_frame
            )
        table["max_assigned_cost"] = _group_max(
            self.frame_assigned_cost.astype(np.float64), frame_offsets, frame_counts
        )
        table["max_margin"] = _group_max(
            self.frame_margin.astype(np.float64), frame_offsets, frame_counts
        )
        # How often the search overrode nearest-centroid assignment, the
        # cheapest single number for "is the guiding doing anything here".
        table["frac_reassigned"] = (
            _group_sum(
                (self.frame_label != self.frame_best_label).astype(np.float64),
                frame_offsets,
                frame_counts,
            )
            * per_frame
        )
        return table


def _group_sum(values: np.ndarray, offsets: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Segment sums via a cumulative sum, which - unlike ``np.add.reduceat`` -
    gives 0 rather than a bogus element for an empty group.
    """
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    return cumulative[offsets + counts] - cumulative[offsets]


def _group_max(values: np.ndarray, offsets: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Segment maxima, nan for an empty group.

    ``reduceat`` reduces ``values[idx[i]:idx[i+1]]``, so dropping the empty
    groups from the index list is exactly right: their counts are 0, which
    makes the next non-empty group's offset coincide with the end of the
    previous one.
    """
    out = np.full(len(offsets), np.nan)
    nonempty = np.flatnonzero(counts > 0)
    if len(nonempty):
        out[nonempty] = np.maximum.reduceat(values, offsets[nonempty])
    return out


def load_diagnostics(path: str) -> Diagnostics:
    """
    Load a dump written by :class:`FrameDiagnostics`.

    ``path`` is either a single ``.npz`` or a directory of them - normally
    ``ClusteringDiagnosticsJob.out_diagnostics``, holding one file per chunk
    plus a ``meta.json``. Chunks are concatenated in filename order with their
    offsets rebased; since chunks partition the corpus by sequence, the result
    is independent of how many there were.
    """
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".npz"))
        if not files:
            raise FileNotFoundError(f"no .npz diagnostics files in {path}")
        chunks = [_load_npz(os.path.join(path, name)) for name in files]
        meta_path = os.path.join(path, META_NAME)
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as fp:
                meta = json.load(fp)
    else:
        chunks, meta = [_load_npz(path)], {}

    merged: Dict[str, np.ndarray] = {}
    for name in _FRAME_FIELDS + _SEGMENT_FIELDS:
        merged[name] = np.concatenate([chunk[name] for chunk in chunks])
    merged["seq_tag"] = np.concatenate([chunk["seq_tag"] for chunk in chunks])
    for name in ("seq_num_frames", "seq_num_segments"):
        merged[name] = np.concatenate([chunk[name] for chunk in chunks])
    # Rebase each chunk's offsets onto the concatenated arrays. Recomputing
    # from the counts would work too, but rebasing keeps the stored offsets
    # authoritative, so a chunk written by an older version with a different
    # packing still loads correctly.
    for name, size_key in (
        ("seq_frame_offset", "seq_num_frames"),
        ("seq_segment_offset", "seq_num_segments"),
    ):
        rebased, base = [], 0
        for chunk in chunks:
            rebased.append(chunk[name].astype(np.int64) + base)
            base += int(chunk[size_key].sum())
        merged[name] = np.concatenate(rebased) if rebased else np.zeros(0, dtype=np.int64)

    duplicates = len(merged["seq_tag"]) - len(np.unique(merged["seq_tag"]))
    if duplicates:
        raise ValueError(
            f"{duplicates} sequence tag(s) appear in more than one chunk of {path}; "
            f"the files do not come from a single partition of the corpus"
        )
    return Diagnostics(meta=meta, **merged)


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
