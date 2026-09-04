"""
Supervised vector-quantized baseline: a debugging rig, not part of the pipeline.

Answers one question the unsupervised runs cannot: *given* the right labels, how
well can a discrete model over a fixed codebook do? That number is the ceiling
every unsupervised run is trying to reach, and having it makes a failure
diagnosable - a run far below it is a training problem, a run at it with bad PER
is a search problem.

Measured before any of this was written, so the jobs below have a number to be
checked against. On 182k silence-free cv frames with an oracle table:

    codebook                C     frame accuracy    I(label; codeword)
    colleague's FAISS     512          67.2%          2.273 / 3.689
    ours, stage-1         512          65.6%          2.248 / 3.689
    ours, stage-1         256          62.7%          2.103 / 3.689
    ours, stage-1         128          56.7%          1.930 / 3.689

against 2.5% for chance and 6.9% for always guessing the majority label. Two
things follow. The codebooks are fine and ours is barely behind one trained
elsewhere, so the partition is not what is failing; and a trained
:class:`...chunked.models.GaussianMixtureModel` from the unsupervised runs
scores **1.7%** on the same measurement - below chance - so what is failing is
the weight estimation, not the codebook, the model class, or the search scale.

The two jobs here are deliberately monolithic and re-runnable rather than
composed into the graph: this is a diagnostic that wants changing often.
"""

from __future__ import annotations

__all__ = [
    "SegmentedFeaturesFromAlignmentJob",
    "SupervisedVQTableJob",
    "vq_table_diagnostics",
]

import json
import logging
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from sisyphus import Job, Task, tk

from ..lib.guided_kmeans.chunked.models import VectorQuantizedModel
from ..lib.guided_kmeans.util import ProgressLogger

#: Sequence tags in the feature HDFs carry a corpus prefix ("train-other-960/")
#: that the alignment pickles do not. Verified against the real files: stripping
#: it joins all 2786 cv sequences with zero length mismatches.
DEFAULT_CORPUS_PREFIX_SEGMENTS = 3


def _strip_corpus_prefix(tag: str, segments: int = DEFAULT_CORPUS_PREFIX_SEGMENTS) -> str:
    """``train-other-960/<rec>/<seg>`` -> ``<rec>/<seg>``; other shapes untouched."""
    parts = tag.split("/")
    return "/".join(parts[1:]) if len(parts) == segments else tag


def _load_alignment(
    alignment: Union[tk.Path, Sequence[tk.Path]], wanted: Optional[set] = None
) -> Dict[str, np.ndarray]:
    """
    Per-frame labels as ``{normalized tag: [T] int}``, from either storage form.

    Two forms exist in this setup and they are not interchangeable by shape
    alone, so the dispatch is on the file extension:

    ``.pkl``
        ``{seq_tag: array}`` directly, as ``constants.GMM_ALIGNMENT_CV`` is.
    ``.hdf`` (one or many shards)
        RETURNN's flat layout - ``inputs`` is every sequence's labels
        concatenated, cut by ``seqLengths``. The ``labels`` dataset in these
        files is a numeric placeholder ('0'..'39') and says nothing about the
        phoneme inventory; the indices are this setup's lexicon order with
        silence at 0, which was established by comparing 144 sequences that
        occur in both an HDF shard and the cv pickle - they agree on 100% of
        frames.

    ``wanted`` bounds memory: a 960h alignment is ~140M int32 frames, and a run
    over a 100h feature file needs a tenth of that. Given the feature file's
    tags, only those sequences are kept.
    """
    paths = list(alignment) if isinstance(alignment, (list, tuple)) else [alignment]
    loaded: Dict[str, np.ndarray] = {}
    for path in paths:
        name = path.get_path()
        if name.endswith(".pkl"):
            with open(name, "rb") as fp:
                raw = pickle.load(fp)
            items = raw.items()
        else:
            import h5py

            with h5py.File(name, "r") as fp:
                tags = [
                    t.decode() if isinstance(t, bytes) else str(t) for t in fp["seqTags"][:]
                ]
                lengths = np.asarray(fp["seqLengths"][:])
                lengths = lengths.reshape(len(lengths), -1)[:, 0].astype(np.int64)
                offsets = np.concatenate([[0], np.cumsum(lengths)])
                inputs = fp["inputs"]
                items = [
                    (tag, np.asarray(inputs[offsets[i] : offsets[i + 1]]))
                    for i, tag in enumerate(tags)
                    if wanted is None or _strip_corpus_prefix(tag) in wanted
                ]
        for tag, labels in items:
            key = _strip_corpus_prefix(tag)
            if wanted is not None and key not in wanted:
                continue
            if key in loaded:
                raise ValueError(
                    f"sequence {key!r} occurs twice across the alignment inputs. Tags are "
                    f"normalized by stripping the corpus prefix, so two corpora carrying "
                    f"the same recording would collide here - pass only the alignment "
                    f"that belongs to these features."
                )
            loaded[key] = np.asarray(labels).reshape(-1)
    return loaded


def _read_hdf_index(path: str):
    """``(tags, lengths, offsets)`` for a RETURNN feature HDF."""
    import h5py

    with h5py.File(path, "r") as fp:
        tags = [t.decode() if isinstance(t, bytes) else str(t) for t in fp["seqTags"][:]]
        lengths = np.asarray(fp["seqLengths"][:])
        lengths = lengths.reshape(len(lengths), -1)[:, 0].astype(np.int64)
    return tags, lengths, np.concatenate([[0], np.cumsum(lengths)])


class SegmentedFeaturesFromAlignmentJob(Job):
    """
    Pool features into the reference alignment's phoneme segments, optionally
    dropping silence.

    Reproduces exactly how this setup's existing ``segmented_features_*.hdf``
    files were built - verified rather than assumed: run-length encoding the GMM
    alignment gives 344,839 segments for ls-cv, which is precisely the frame
    count of the stored segmented file, and **mean** pooling reproduces its
    values to 1e-5 (float32 rounding) where max pooling is off by ~300. So a
    file produced here with ``exclude_labels=()`` is the existing file, and with
    the default it is that file minus silence.

    **Segment first, then drop.** Silence frames are not removed before the
    run-length encoding, because doing so would merge two separate realizations
    of a phoneme that happened to be separated by a pause (``AA SIL AA`` would
    collapse into one ``AA``). The alignment is encoded whole and silence
    *segments* are dropped afterwards.

    Emits the per-segment labels alongside the features, since the segmentation
    that defines them is computed here anyway - that pairing is what makes the
    supervised table a counting job.

    **The corpus prefix is not decoration.** ls-100h features are tagged
    ``train-clean-100/<rec>/<seg>`` while the 960h alignments covering them are
    tagged ``train-other-960/<rec>/<seg>`` - the same sequences under a
    different corpus name, with *zero* exact tag matches between the two. Both
    sides are normalized by dropping that first segment, which joins all 28,234
    ls-100h sequences with no length disagreement.

    :param features_hdf: unsegmented features, frame-aligned with ``alignment``
    :param alignment: per-frame label indices, as a ``.pkl`` of
        ``{seq_tag: [T] int}`` or as one or more RETURNN ``.hdf`` shards - see
        :func:`_load_alignment`
    :param exclude_labels: label indices whose segments are dropped. Defaults to
        silence, which is what a codebook trained on silence-free speech needs.
    :param pooling: ``"mean"`` (what the existing files use) or ``"max"``
    :param min_segment_frames: drop segments shorter than this before pooling
    """

    __sis_hash_exclude__ = {"rqmt": None}

    def __init__(
        self,
        features_hdf: tk.Path,
        alignment: Union[tk.Path, Sequence[tk.Path]],
        exclude_labels: Sequence[int] = (0,),
        pooling: str = "mean",
        min_segment_frames: int = 1,
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        if pooling not in ("mean", "max"):
            raise ValueError(f"pooling must be 'mean' or 'max', got {pooling!r}")
        self.features_hdf = features_hdf
        self.alignment = list(alignment) if isinstance(alignment, (list, tuple)) else alignment
        self.exclude_labels = tuple(sorted(set(int(x) for x in exclude_labels)))
        self.pooling = pooling
        self.min_segment_frames = min_segment_frames

        self.out_features = self.output_path("features.hdf")
        self.out_labels = self.output_path("labels.pkl")
        self.out_segments = self.output_path("segments.txt")
        self.out_statistics = self.output_path("statistics.json")

        # Streams sequence by sequence and appends to a resizable dataset, so
        # memory is one utterance rather than the whole corpus - the ls-100
        # output alone would be ~6.8 GB held as float32.
        self.rqmt = {"cpu": 2, "mem": 8, "time": 4}
        if rqmt:
            self.rqmt.update(rqmt)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import h5py

        tags, lengths, offsets = _read_hdf_index(self.features_hdf.get_path())
        # Loaded after the feature index so the tag set can bound it: a 960h
        # alignment is ~140M frames and a 100h feature file needs a tenth.
        alignment = _load_alignment(
            self.alignment, wanted={_strip_corpus_prefix(t) for t in tags}
        )
        excluded = set(self.exclude_labels)

        kept_tags, kept_lengths, labels_out = [], [], {}
        counts = {"sequences": 0, "skipped_no_alignment": 0, "frames_in": 0,
                  "segments_total": 0, "segments_kept": 0, "frames_pooled": 0}

        progress = ProgressLogger(max(len(tags), 1), bar_length=40, logging_step=256)
        progress.start()
        with h5py.File(self.features_hdf.get_path(), "r") as src, \
                h5py.File(self.out_features.get_path(), "w") as dst:
            dim = src["inputs"].shape[1]
            inputs = dst.create_dataset(
                "inputs", shape=(0, dim), maxshape=(None, dim), dtype=np.float32,
                chunks=(1024, dim),
            )
            for index, tag in enumerate(tags):
                labels = alignment.get(_strip_corpus_prefix(tag))
                if labels is None:
                    counts["skipped_no_alignment"] += 1
                    continue
                labels = np.asarray(labels).reshape(-1)
                if len(labels) != lengths[index]:
                    raise ValueError(
                        f"{tag}: alignment has {len(labels)} frames but the features have "
                        f"{lengths[index]}. The alignment has to be frame-synchronous with "
                        f"the unsegmented features; a segmented feature file is not the "
                        f"right input here."
                    )
                features = np.asarray(
                    src["inputs"][offsets[index]:offsets[index + 1]], dtype=np.float64
                )
                counts["sequences"] += 1
                counts["frames_in"] += len(labels)

                # Boundaries of every constant run, silence included - see the
                # class docstring on why the drop happens after this, not before.
                edges = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
                pooled, kept_labels = [], []
                for begin, end in zip(edges[:-1], edges[1:]):
                    counts["segments_total"] += 1
                    label = int(labels[begin])
                    if label in excluded or (end - begin) < self.min_segment_frames:
                        continue
                    block = features[begin:end]
                    pooled.append(block.mean(0) if self.pooling == "mean" else block.max(0))
                    kept_labels.append(label)

                if not pooled:
                    continue
                block = np.stack(pooled).astype(np.float32)
                inputs.resize(inputs.shape[0] + len(block), axis=0)
                inputs[-len(block):] = block
                kept_tags.append(tag)
                kept_lengths.append(len(block))
                labels_out[tag] = np.asarray(kept_labels, dtype=np.int64)
                counts["segments_kept"] += len(block)
                counts["frames_pooled"] += len(features)
                progress.progress(index)

            if not kept_tags:
                raise RuntimeError(
                    "no sequence survived; check that the alignment tags match the feature "
                    "tags (the corpus prefix is stripped) and that exclude_labels is not "
                    "removing everything"
                )
            dst["seqLengths"] = np.stack(
                [np.asarray(kept_lengths, dtype=np.int32), np.zeros(len(kept_tags), np.int32)],
                axis=1,
            )
            dst.create_dataset(
                "seqTags", data=[t.encode() for t in kept_tags],
                dtype=h5py.special_dtype(vlen=bytes),
            )

        with open(self.out_labels.get_path(), "wb") as fp:
            pickle.dump(labels_out, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.out_segments.get_path(), "w") as fp:
            fp.write("\n".join(kept_tags) + "\n")
        counts["excluded_labels"] = list(self.exclude_labels)
        counts["pooling"] = self.pooling
        counts["dropped_segment_fraction"] = (
            1 - counts["segments_kept"] / counts["segments_total"]
            if counts["segments_total"] else 0.0
        )
        with open(self.out_statistics.get_path(), "w") as fp:
            json.dump(counts, fp, indent=4)
        print(
            f"{counts['sequences']} sequences, {counts['frames_in']} frames -> "
            f"{counts['segments_kept']} segments "
            f"({100 * counts['dropped_segment_fraction']:.1f}% of segments dropped as "
            f"{self.exclude_labels}), {self.pooling} pooling",
            flush=True,
        )


def vq_table_diagnostics(
    counts: np.ndarray, codewords: np.ndarray, labels: np.ndarray, table: np.ndarray
) -> dict:
    """
    What a counted table is worth, from the counts and a held-out set.

    ``accuracy`` is ``argmax_l p(c | l) p(l)`` per observation - the model's own
    prediction with no language model, no transitions and no search, which is
    what makes it comparable across setups and an upper bound on what a search
    over the same scores can be expected to do.
    """
    prior = counts.sum(1) / max(counts.sum(), 1.0)
    used = counts.sum(1) > 0
    joint = table[:, codewords] * prior[:, np.newaxis]
    accuracy = float((joint.argmax(0) == labels).mean()) if len(labels) else float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        rows = table[used]
        entropy = -np.where(rows > 0, rows * np.log(np.where(rows > 0, rows, 1.0)), 0.0).sum(1)
        marginal = counts.sum(0) / max(counts.sum(), 1.0)
        ratio = np.where(rows > 0, rows / np.maximum(marginal[np.newaxis, :], 1e-300), 1.0)
        mi = float((prior[used][:, np.newaxis] * np.where(rows > 0, rows * np.log(ratio), 0.0)).sum())
    return {
        "accuracy": accuracy,
        "label_codeword_mi": mi,
        "label_codeword_mi_ceiling": float(np.log(max(int(used.sum()), 1))),
        "mean_label_entropy": float(entropy.mean()) if used.any() else 0.0,
        "majority_label_share": float(prior.max()),
        "chance_accuracy": 1.0 / max(int(used.sum()), 1),
        "labels_seen": int(used.sum()),
        "codewords_used": int((counts.sum(0) > 0).sum()),
        "num_codewords": int(counts.shape[1]),
        "zero_entry_fraction": float((counts[used] == 0).mean()) if used.any() else 0.0,
    }


class SupervisedVQTableJob(Job):
    """
    Count ``p(codeword | label)`` from a reference alignment - the whole
    "training" of a discrete model, one pass and no iteration.

    Writes a model *directory*, not just the table, so the result decodes
    through the ordinary path: ``load_forward_model`` reads the manifest, finds
    :class:`...chunked.models.VectorQuantizedModel` and hands the decode
    callback something with the right ``scores``. Nothing on the decode side
    knows this model was made differently from a trained one.

    :param features_hdf: segmented, silence-free features from
        :class:`SegmentedFeaturesFromAlignmentJob`
    :param labels: that job's ``out_labels``, one label per feature vector
    :param centroids: ``[C, D]`` codebook, quantized against with plain L2 -
        see :meth:`...chunked.models.VectorQuantizedModel.quantize` for why the
        metric matters more than where the quantization happens
    :param table_floor: added to every count before normalizing. **Not
        optional in practice**: a counted table is mostly zeros (measured 63.6%
        on a real 40x512 table), every zero is an ``+inf`` score, and a codeword
        no label admits leaves a frame with no viable label at all. It is also
        a scale knob - the label contrast a search sees was measured at 13.0
        nats at 1e-3 and 8.4 at 1e-1.
    :param heldout_fraction: sequences held out of the counting and scored with
        the resulting table. The honest number: the training accuracy of a
        counted table is optimistic by construction, and with no ls-100
        alignment available there is no other corpus to measure generalization
        on.
    """

    __sis_hash_exclude__ = {"rqmt": None}

    def __init__(
        self,
        features_hdf: tk.Path,
        labels: tk.Path,
        centroids: tk.Path,
        num_labels: int = 40,
        table_floor: float = 1e-2,
        heldout_fraction: float = 0.2,
        split_seed: int = 42,
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        if not 0.0 <= heldout_fraction < 1.0:
            raise ValueError(f"heldout_fraction must be in [0, 1), got {heldout_fraction}")
        if table_floor < 0:
            raise ValueError(f"table_floor must be >= 0, got {table_floor}")
        self.features_hdf = features_hdf
        self.labels = labels
        self.centroids = centroids
        self.num_labels = num_labels
        self.table_floor = table_floor
        self.heldout_fraction = heldout_fraction
        self.split_seed = split_seed

        self.out_model = self.output_path("model", directory=True)
        self.out_table = self.output_path("table.npy")
        self.out_counts = self.output_path("counts.npy")
        self.out_diagnostics = self.output_path("diagnostics.json")
        self.out_accuracy = self.output_var("accuracy")
        # The split, so a decode can be pointed at the half the table did not
        # see. Without these the only decodable set is the training set, and a
        # counted table's training PER means very little.
        self.out_heldout_segments = self.output_path("heldout_segments.txt")
        self.out_train_segments = self.output_path("train_segments.txt")

        self.rqmt = {"cpu": 4, "mem": 16, "time": 4}
        if rqmt:
            self.rqmt.update(rqmt)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import h5py
        from scipy.spatial.distance import cdist

        centroids = np.load(self.centroids.get_path()).astype(np.float64)
        if centroids.ndim != 2:
            raise ValueError(f"expected centroids [C, D], got {centroids.shape}")
        num_codewords = centroids.shape[0]
        with open(self.labels.get_path(), "rb") as fp:
            labels_by_tag = pickle.load(fp)
        tags, lengths, offsets = _read_hdf_index(self.features_hdf.get_path())

        rng = np.random.RandomState(self.split_seed)
        order = rng.permutation(len(tags))
        num_heldout = int(round(self.heldout_fraction * len(tags)))
        heldout = set(order[:num_heldout].tolist())

        counts = np.zeros((self.num_labels, num_codewords), dtype=np.float64)
        eval_codewords, eval_labels = [], []
        train_codewords, train_labels = [], []

        progress = ProgressLogger(max(len(tags), 1), bar_length=40, logging_step=256)
        progress.start()
        with h5py.File(self.features_hdf.get_path(), "r") as src:
            for index, tag in enumerate(tags):
                seq_labels = labels_by_tag.get(tag)
                if seq_labels is None:
                    raise KeyError(f"no labels for {tag!r}; labels and features disagree")
                features = np.asarray(
                    src["inputs"][offsets[index]:offsets[index + 1]], dtype=np.float64
                )
                if len(features) != len(seq_labels):
                    raise ValueError(
                        f"{tag}: {len(features)} vectors but {len(seq_labels)} labels"
                    )
                codewords = cdist(features, centroids, metric="sqeuclidean").argmin(axis=1)
                if index in heldout:
                    eval_codewords.append(codewords)
                    eval_labels.append(seq_labels)
                else:
                    np.add.at(counts, (seq_labels, codewords), 1.0)
                    train_codewords.append(codewords)
                    train_labels.append(seq_labels)
                progress.progress(index)

        if not counts.any():
            raise RuntimeError("nothing counted; every sequence landed in the held-out split")

        for path, wanted in (
            (self.out_heldout_segments, True),
            (self.out_train_segments, False),
        ):
            with open(path.get_path(), "w") as fp:
                chosen = [t for i, t in enumerate(tags) if (i in heldout) == wanted]
                fp.write("\n".join(chosen) + ("\n" if chosen else ""))

        # Every label needs a row that sums to 1 for the model to accept it, and
        # a label that never occurred has no evidence to build one from - it gets
        # the uniform row, which says exactly that.
        floored = counts + self.table_floor
        empty = counts.sum(1) == 0
        floored[empty] = 1.0
        table = floored / floored.sum(axis=1, keepdims=True)

        model = VectorQuantizedModel(centroids, table)
        model.save(self.out_model.get_path())
        np.save(self.out_table.get_path(), table)
        np.save(self.out_counts.get_path(), counts)

        diagnostics = {
            "table_floor": self.table_floor,
            "heldout_fraction": self.heldout_fraction,
            "num_sequences": len(tags),
            "num_heldout_sequences": len(heldout),
            "num_labels": int(self.num_labels),
            "labels_never_seen": int(empty.sum()),
            "train": vq_table_diagnostics(
                counts, np.concatenate(train_codewords), np.concatenate(train_labels), table
            ),
        }
        if eval_labels:
            diagnostics["heldout"] = vq_table_diagnostics(
                counts, np.concatenate(eval_codewords), np.concatenate(eval_labels), table
            )
        # The label contrast the search will see, which is what a distance_scale
        # has to be chosen against - and which the floor moves.
        with np.errstate(divide="ignore"):
            neg_log = -np.log(table)
        sample = np.concatenate(eval_codewords or train_codewords)[:20000]
        contrast = neg_log[:, sample]
        # With no floor the table has hard zeros, so the worst label for a
        # codeword costs +inf and the spread is inf - inf. Reported as the
        # unbounded contrast it is rather than as a nan, and measured over the
        # finite part so the number still says something.
        finite = np.where(np.isfinite(contrast), contrast, np.nan)
        with np.errstate(invalid="ignore"):
            spread = np.nanmax(finite, axis=0) - np.nanmin(finite, axis=0)
        spread = spread[np.isfinite(spread)]
        unbounded = bool((~np.isfinite(contrast)).any())
        diagnostics["label_contrast_nats"] = {
            "median": float(np.median(spread)) if spread.size else float("inf"),
            "p95": float(np.percentile(spread, 95)) if spread.size else float("inf"),
            # True when some label forbids a codeword outright, i.e. the real
            # contrast for those frames is infinite and the numbers above are
            # the finite part only.
            "unbounded_entries": unbounded,
        }
        with open(self.out_diagnostics.get_path(), "w") as fp:
            json.dump(diagnostics, fp, indent=4)
        best = diagnostics.get("heldout", diagnostics["train"])
        self.out_accuracy.set(best["accuracy"])
        logging.info(
            "table over %d codewords: train acc %.1f%%, held-out acc %.1f%% "
            "(chance %.1f%%, majority %.1f%%), I(l;c) %.3f nats, contrast %.1f nats",
            num_codewords,
            100 * diagnostics["train"]["accuracy"],
            100 * best["accuracy"],
            100 * best["chance_accuracy"],
            100 * best["majority_label_share"],
            best["label_codeword_mi"],
            diagnostics["label_contrast_nats"]["median"],
        )
