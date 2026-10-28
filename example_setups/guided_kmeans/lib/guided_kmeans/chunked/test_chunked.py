"""
Regression guard for the properties the chunked pipeline's design rests on.

No pytest setup exists for this code base; run it inside the pinned venv with
both sisyphus and the recipe root on the path::

    app-py -c "import sys
    sys.path[:0] = ['/u/mann/src/sisyphus', '<repo>/recipe']
    from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked \\
        .test_chunked import main
    sys.exit(main())"

The load-bearing claim is that every accumulator's ``merge`` is associative.
:func:`..setup.chunked_clustering.GuidedClusteringEpochJob.hash` excludes
``num_chunks`` on exactly that basis - if these checks ever fail, the number
of chunks is silently changing results across job re-runs that share a hash.
"""

import pickle
import sys

import numpy as np

from .accumulators import (
    FixedCovarianceAccumulator,
    VectorQuantizedAccumulator,
    GaussianAccumulator,
    MeanAccumulator,
    MixtureGaussianAccumulator,
    NullAccumulator,
    SoftGaussianAccumulator,
    keep_previous_where_dead,
)
from .flavors import (
    ClusteringFlavor,
    euclidean_flavor,
    unguided_flavor,
    gaussian_flavor,
    mixture_flavor,
    per_label_mixture_flavor,
)
from .diagnostics import FrameDiagnostics, load_diagnostics
from .interfaces import RecognitionResult
from .recognizers import ArgmaxRecognizer, RasrFBRecognizer
from .features import plan_chunks
from .models import (
    ArtifactModel,
    VectorQuantizedModel,
    EuclideanModel,
    GaussianMixtureModel,
    GaussianModel,
    PerLabelMixtureModel,
    load_forward_model,
    load_model,
    neg_log_matmul,
    read_manifest,
)
from .ema import (
    blend_parameters,
    effective_window,
    ema_statistics,
    load_state,
    save_state,
)
from .runner import merge_chunks, reduce_chunks, run_chunk, save_chunk
from .spec import Spec
from ..running_update import RunningAverageUpdater
from ..util import traceback_to_text

_FAILURES = []


class _DiagModel(ArtifactModel):
    """
    A model the package knows nothing about, used to check that adding one
    needs no changes to the job, the pipeline or the reduce step.
    """

    def __init__(self, centroids, variances, priors):
        self.centroids = np.asarray(centroids)
        self.variances = np.asarray(variances)
        self.priors = np.asarray(priors)

    @property
    def num_clusters(self) -> int:
        return self.centroids.shape[0]

    @property
    def dim(self) -> int:
        return self.centroids.shape[1]

    def scores(self, features):
        sq = (features[:, None, :] - self.centroids[None]) ** 2 / self.variances[None]
        return sq.sum(-1) - np.log(self.priors)[None]

    def artifacts(self):
        return {"centroids": self.centroids, "variances": self.variances, "priors": self.priors}

    @classmethod
    def from_artifacts(cls, arrays, meta):
        return cls(arrays["centroids"], arrays["variances"], arrays["priors"])


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    return False


def _check(name, condition, extra=""):
    if not condition:
        _FAILURES.append(name)
    print(f"  [{'ok ' if condition else 'FAIL'}] {name} {extra}")


class _StubItem:
    """Stand-in for a traceback item; only ``lemma`` is read by transcription."""

    def __init__(self, lemma):
        self.lemma = lemma


class _StubFBRecognizer:
    """
    Replays a fixed gamma matrix [T, K] per tag, so no RASR/librasr is needed
    for FB tests. Gammas are delivered as-is (already normalized), matching
    what RasrFBRecognizer._handle outputs after row-normalization.
    """

    def __init__(self, gammas_table):
        self.gammas_table = gammas_table  # {seq_tag: np.ndarray [T, K]}
        self._buffer = []
        self._cb = None

    def start(self, on_result):
        self._cb = on_result

    def submit(self, seq_tag, scores):
        self._buffer.append(seq_tag)

    def drain(self):
        for seq_tag in self._buffer:
            self._cb(RecognitionResult(
                seq_tag=seq_tag,
                posteriors=self.gammas_table[seq_tag],
                sequence_score=-1.0,
            ))
        self._buffer = []

    def shutdown(self):
        pass


class _StubTracebackItem:
    """A traceback item carrying spans and scores, for the diagnostics probe."""

    def __init__(self, lemma, start_time, end_time, am_score, lm_score, transition_score):
        self.lemma = lemma
        self.start_time = start_time
        self.end_time = end_time
        self.am_score = am_score
        self.lm_score = lm_score
        self.transition_score = transition_score


def _scored_traceback(labels, rng):
    """
    Run-length encode a label sequence into a traceback, with *accumulated*
    scores - the layout the rest of the code base assumes when it reads a
    sequence total off the last item.
    """
    items = []
    totals = np.zeros(3)
    start = 0
    for end in range(1, len(labels) + 1):
        if end < len(labels) and labels[end] == labels[end - 1]:
            continue
        totals = totals + np.abs(rng.randn(3))
        items.append(
            _StubTracebackItem(
                f"p{int(labels[start])}", start, end, totals[0], totals[1], totals[2]
            )
        )
        start = end
    return items


class _StubRecognizer:
    """Replays a fixed alignment per tag, so no RASR/librasr is needed."""

    def __init__(self, table, tracebacks=None):
        self.table = table
        self.tracebacks = tracebacks or {}
        self._buffer = []
        self._cb = None

    def start(self, on_result):
        self._cb = on_result

    def submit(self, seq_tag, scores):
        self._buffer.append(seq_tag)

    def drain(self):
        for seq_tag in self._buffer:
            self._cb(RecognitionResult(
                seq_tag=seq_tag,
                posteriors=self.table[seq_tag],
                traceback=self.tracebacks.get(seq_tag, []),
            ))
        self._buffer = []

    def shutdown(self):
        pass


class _ListSource:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class _PickledFeatures:
    """A feature source over a pickled ``{tag: [T, D]}``, chunked by sorted tag,
    so an epoch job's spec can name it."""

    def __init__(self, path, chunk=0, num_chunks=1):
        with open(path, "rb") as f:
            table = pickle.load(f)
        self.items = [(tag, table[tag]) for tag in sorted(table)][chunk::num_chunks]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


def main() -> int:
    import os
    import pickle
    import tempfile

    rng = np.random.RandomState(0)
    num_clusters, dim, num_seqs = 6, 9, 40
    feats = [rng.randn(rng.randint(20, 60), dim).astype(np.float32) for _ in range(num_seqs)]
    labels = [rng.randint(0, num_clusters, size=len(f)) for f in feats]
    previous = EuclideanModel(rng.randn(num_clusters, dim))
    lengths = [len(f) for f in feats]

    # Reference: the update the single-process callback performs.
    reference = RunningAverageUpdater((num_clusters, dim))
    for f, l in zip(feats, labels):
        onehot = np.eye(num_clusters)[l]
        reference.update(onehot.T @ f, onehot.sum(0))
    ref_centroids = reference.value.copy()
    ref_centroids[reference.counts == 0] = previous.centroids[reference.counts == 0]

    print("MeanAccumulator")
    acc = MeanAccumulator(num_clusters, dim)
    for f, l in zip(feats, labels):
        acc.observe(f, l)
    err = np.abs(acc.finalize(previous).centroids - ref_centroids).max()
    _check("matches RunningAverageUpdater", err < 1e-10, f"max err {err:.2e}")

    errs = []
    for chunks in (1, 3, 7, num_seqs):
        merged = None
        for part in plan_chunks(lengths, chunks):
            a = MeanAccumulator(num_clusters, dim)
            for i in part:
                a.observe(feats[i], labels[i])
            a = MeanAccumulator(num_clusters).load_state_dict(a.state_dict())
            merged = a if merged is None else merged.merge(a)
        errs.append(np.abs(merged.finalize(previous).centroids - ref_centroids).max())
    _check("merge associative over chunk counts", max(errs) < 1e-10, f"max err {max(errs):.2e}")

    sparse = MeanAccumulator(num_clusters, dim)
    sparse.observe(feats[0], np.zeros(len(feats[0]), dtype=int))
    _check(
        "empty clusters keep the previous centroid",
        np.allclose(sparse.finalize(previous).centroids[1:], previous.centroids[1:]),
    )

    soft = MeanAccumulator(num_clusters, dim)
    for f, l in zip(feats, labels):
        soft.observe(f, (l[:, None], np.ones((len(l), 1))))
    _check(
        "one-hot soft posteriors == hard labels",
        np.allclose(soft.finalize(previous).centroids, ref_centroids, atol=1e-10),
    )

    print("GaussianAccumulator")
    prev_gauss = GaussianModel(rng.randn(num_clusters, dim), np.stack([np.eye(dim)] * num_clusters))
    g_ref = GaussianAccumulator(num_clusters)
    for f, l in zip(feats, labels):
        g_ref.observe(f, l)
    m_ref = g_ref.finalize(prev_gauss)

    g_errs = []
    for chunks in (2, 5, 13):
        merged = None
        for part in plan_chunks(lengths, chunks):
            a = GaussianAccumulator(num_clusters)
            for i in part:
                a.observe(feats[i], labels[i])
            a = GaussianAccumulator(num_clusters).load_state_dict(a.state_dict())
            merged = a if merged is None else merged.merge(a)
        model = merged.finalize(prev_gauss)
        g_errs.append(
            max(
                np.abs(model.centroids - m_ref.centroids).max(),
                np.abs(model.covs - m_ref.covs).max(),
            )
        )
    _check("merge associative over chunk counts", max(g_errs) < 1e-9, f"max err {max(g_errs):.2e}")

    print("runner")
    tags = [f"seq{i}" for i in range(num_seqs)]
    table = dict(zip(tags, labels))
    # A silence item in front of every traceback, which transcription has to
    # drop: the references these hypotheses get scored against have no silence.
    tracebacks = {
        tag: [_StubItem("[SILENCE]")] + [_StubItem(f"p{int(label)}") for label in seq_labels]
        for tag, seq_labels in table.items()
    }
    expected_hyps = {
        tag: " ".join(f"p{int(label)}" for label in seq_labels) for tag, seq_labels in table.items()
    }
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for chunk_idx, part in enumerate(plan_chunks(lengths, 4)):
            result = run_chunk(
                features=_ListSource([(tags[i], feats[i]) for i in part]),
                model=previous,
                recognizer=_StubRecognizer(table, tracebacks),
                accumulator=MeanAccumulator(num_clusters, dim),
                counter=None,
                transcribe=traceback_to_text,
                verbosity=0,
            )
            path = os.path.join(tmp, f"chunk.{chunk_idx}.pkl")
            save_chunk(result, path)
            paths.append(path)
        model, _stats, totals, hypotheses = reduce_chunks(
            chunk_paths=paths,
            accumulator_factory=lambda: MeanAccumulator(num_clusters),
            previous_model=previous,
        )
        err = np.abs(model.centroids - ref_centroids).max()
        _check("run_chunk + reduce_chunks == reference", err < 1e-10, f"max err {err:.2e}")
        _check(
            "every sequence recognized exactly once",
            totals["num_seqs"] == num_seqs and totals["num_recognized"] == num_seqs,
            str(totals),
        )
        _check(
            "hypotheses survive chunking, without silence",
            hypotheses == expected_hyps,
            f"{len(hypotheses)} of {num_seqs} sequences",
        )

    print("SoftGaussianAccumulator — min_mass")
    # Soft posteriors have no floor, so a cluster can end an epoch holding a
    # trace of mass, pass a `> 0` test, and be re-estimated from evidence worth
    # far less than one frame. min_mass restores the hard-alignment rule.
    prev_soft = GaussianModel(rng.randn(num_clusters, dim), np.stack([np.eye(dim)] * num_clusters))
    soft_feats = rng.randn(200, dim) * 3 + 10
    trace = np.zeros((200, num_clusters))
    trace[:, 0] = 1.0          # cluster 0 carries everything
    trace[:, 1] = 1e-12        # cluster 1 gets a trace, ~2e-10 frames in total

    default_acc = SoftGaussianAccumulator(num_clusters, dim)
    default_acc.observe(soft_feats, trace)
    _check(
        "default min_mass=1.0 keeps a trace-mass cluster at its previous model",
        np.array_equal(default_acc.finalize(prev_soft).centroids[1], prev_soft.centroids[1]),
    )

    permissive = SoftGaussianAccumulator(num_clusters, dim, min_mass=0.0)
    permissive.observe(soft_feats, trace)
    _check(
        "min_mass=0.0 reproduces the old `> 0` behaviour",
        not np.array_equal(permissive.finalize(prev_soft).centroids[1], prev_soft.centroids[1]),
    )
    _check(
        "min_mass=0.0 still treats zero mass as dead",
        np.array_equal(permissive.finalize(prev_soft).centroids[2], prev_soft.centroids[2]),
    )
    _check(
        "a genuinely populated cluster is still re-estimated",
        not np.array_equal(default_acc.finalize(prev_soft).centroids[0], prev_soft.centroids[0]),
    )

    print("MeanAccumulator — dense soft (FB) posteriors")
    # Soft gammas: each row is a proper probability distribution over clusters.
    gammas = [rng.dirichlet(np.ones(num_clusters), size=len(f)) for f in feats]

    # Reference: the update update_centroids_soft + RunningAverageUpdater performs.
    ref_sums = np.zeros((num_clusters, dim), dtype=np.float64)
    ref_counts = np.zeros(num_clusters, dtype=np.float64)
    for f, g in zip(feats, gammas):
        ref_sums += g.T @ f.astype(np.float64)
        ref_counts += g.sum(0)
    ref_centroids_soft = np.divide(
        ref_sums, ref_counts[:, np.newaxis],
        out=np.zeros_like(ref_sums),
        where=ref_counts[:, np.newaxis] > 0,
    )
    ref_centroids_soft[ref_counts == 0] = previous.centroids[ref_counts == 0]

    soft_acc = MeanAccumulator(num_clusters, dim)
    for f, g in zip(feats, gammas):
        soft_acc.observe(f, g)
    err = np.abs(soft_acc.finalize(previous).centroids - ref_centroids_soft).max()
    _check("soft observe matches update_centroids_soft reference", err < 1e-10, f"max err {err:.2e}")

    soft_errs = []
    for chunks in (1, 3, 7, num_seqs):
        merged = None
        for part in plan_chunks(lengths, chunks):
            a = MeanAccumulator(num_clusters, dim)
            for i in part:
                a.observe(feats[i], gammas[i])
            a = MeanAccumulator(num_clusters).load_state_dict(a.state_dict())
            merged = a if merged is None else merged.merge(a)
        soft_errs.append(np.abs(merged.finalize(previous).centroids - ref_centroids_soft).max())
    _check("soft merge associative over chunk counts", max(soft_errs) < 1e-10, f"max err {max(soft_errs):.2e}")

    # Verify that the 2-D branch rejects shape mismatches.
    bad_cols = MeanAccumulator(num_clusters, dim)
    try:
        bad_cols.observe(feats[0], np.ones((len(feats[0]), num_clusters + 1)))
        _check("wrong gamma column count raises ValueError", False)
    except ValueError:
        _check("wrong gamma column count raises ValueError", True)

    print("RasrFBRecognizer._handle — gamma normalization and slicing")
    # Instantiate without starting the pool; _handle does not touch the executor.
    fb_rec = RasrFBRecognizer("/nonexistent.config", num_clusters=num_clusters)
    delivered = []
    fb_rec._on_result = delivered.append

    T = 15
    raw_gammas = rng.exponential(1.0, size=(T, num_clusters + 3)).astype(np.float64)
    seq_tag_fb = "test_fb_seq"
    fb_rec._handle(seq_tag_fb, raw_gammas, log_likelihood=-42.0)
    assert len(delivered) == 1
    out_tag, out_gammas = delivered[0].seq_tag, delivered[0].posteriors
    _check("_handle: seq_tag passed through", out_tag == seq_tag_fb)
    _check("_handle: extra RASR label columns stripped", out_gammas.shape == (T, num_clusters))
    _check("_handle: rows sum to 1 after normalization",
           np.allclose(out_gammas.sum(axis=1), 1.0, atol=1e-12))
    # The log-likelihood rides in its own field; run_chunk dispatches on that
    # rather than on the type of whatever sits in the traceback slot, and a
    # pathless search leaves the traceback empty.
    _check("_handle: log-likelihood reported as sequence_score",
           delivered[0].sequence_score == -42.0, repr(delivered[0].sequence_score))
    _check("_handle: no discrete path reported for FB", delivered[0].traceback == [])

    # A sequence the worker produced nothing for carries no FB metadata; the
    # empty gammas then trip run_chunk's frame-count check, as intended.
    empty_delivered = []
    fb_rec._on_result = empty_delivered.append
    fb_rec._handle("empty_seq", np.zeros((0, num_clusters + 3)), log_likelihood=float("nan"))
    _check("_handle: empty gammas carry no score and no path",
           len(empty_delivered) == 1 and empty_delivered[0].traceback == []
           and empty_delivered[0].sequence_score is None
           and empty_delivered[0].posteriors.shape[0] == 0)
    fb_rec._on_result = delivered.append

    # All-zero row (degenerate sequence) must produce a zero row, not nan.
    zero_row = np.zeros((T, num_clusters + 3), dtype=np.float64)
    delivered_zero = []
    fb_rec._on_result = lambda result: delivered_zero.append(result.posteriors)
    fb_rec._handle("zero_seq", zero_row, float("nan"))
    _check("_handle: all-zero row produces zero row (not nan)",
           delivered_zero and not np.isnan(delivered_zero[0]).any()
           and np.all(delivered_zero[0] == 0.0))

    print("FB runner — run_chunk + reduce_chunks")
    tags_fb = [f"seq{i}" for i in range(num_seqs)]
    gammas_table = dict(zip(tags_fb, gammas))

    with tempfile.TemporaryDirectory() as tmp:
        paths_fb = []
        for chunk_idx, part in enumerate(plan_chunks(lengths, 4)):
            result = run_chunk(
                features=_ListSource([(tags_fb[i], feats[i]) for i in part]),
                model=previous,
                recognizer=_StubFBRecognizer({tags_fb[i]: gammas[i] for i in part}),
                accumulator=MeanAccumulator(num_clusters, dim),
                counter=None,
                verbosity=0,
            )
            path = os.path.join(tmp, f"chunk_fb.{chunk_idx}.pkl")
            save_chunk(result, path)
            paths_fb.append(path)
        model_fb, _stats_fb, totals_fb, _ = reduce_chunks(
            chunk_paths=paths_fb,
            accumulator_factory=lambda: MeanAccumulator(num_clusters),
            previous_model=previous,
        )
        err_fb = np.abs(model_fb.centroids - ref_centroids_soft).max()
        _check("FB run_chunk + reduce_chunks == reference", err_fb < 1e-10, f"max err {err_fb:.2e}")
        _check(
            "FB: every sequence recognized exactly once",
            totals_fb["num_seqs"] == num_seqs and totals_fb["num_recognized"] == num_seqs,
            str(totals_fb),
        )

    print("diagnostics")
    # One sequence gets an empty traceback: a search that emitted nothing is a
    # real case (run_chunk keeps it as an empty hypothesis) and the dump has to
    # carry it as the anomaly it is rather than dropping or crashing on it.
    scored_tracebacks = {tag: _scored_traceback(table[tag], rng) for tag in tags}
    silent_tag = tags[3]
    scored_tracebacks[silent_tag] = []

    with tempfile.TemporaryDirectory() as tmp:
        for chunk_idx, part in enumerate(plan_chunks(lengths, 4)):
            probe = FrameDiagnostics()
            run_chunk(
                features=_ListSource([(tags[i], feats[i]) for i in part]),
                model=previous,
                recognizer=_StubRecognizer(table, scored_tracebacks),
                accumulator=NullAccumulator(),
                probe=probe,
                verbosity=0,
            )
            probe.save(os.path.join(tmp, f"diagnostics.4.{chunk_idx:04d}.npz"))
        diag = load_diagnostics(tmp)

        _check(
            "every sequence survives chunking, exactly once",
            diag.num_sequences == num_seqs and sorted(diag.seq_tag) == sorted(tags),
            f"{diag.num_sequences} of {num_seqs}",
        )

        # The load-bearing claim of the whole dump: the cost recorded for a
        # frame is the distance to the centroid it was aligned with, and the
        # best cost is the distance to the one it would have picked alone.
        cost_err, best_err, label_err = 0.0, 0.0, 0
        for tag, seq_feats, seq_labels in zip(tags, feats, labels):
            scored = previous.scores(seq_feats)
            got = diag.frames_of(tag)
            rows = np.arange(len(seq_labels))
            cost_err = max(cost_err, np.abs(got["frame_assigned_cost"] - scored[rows, seq_labels]).max())
            best_err = max(best_err, np.abs(got["frame_best_cost"] - scored.min(axis=1)).max())
            label_err += int((got["frame_label"] != seq_labels).sum())
        _check(
            "assigned cost is the distance to the aligned centroid",
            cost_err < 1e-4 and label_err == 0,
            f"max err {cost_err:.2e}, {label_err} label mismatches",
        )
        _check("best cost is the distance to the nearest centroid", best_err < 1e-4,
               f"max err {best_err:.2e}")

        margin = diag.frame_margin
        agreed = diag.frame_label == diag.frame_best_label
        _check(
            "margin is non-negative and zero exactly where the search agreed",
            margin.min() >= 0 and np.all(margin[agreed] == 0) and np.all(margin[~agreed] > 0),
        )

        # Segment labels are read off the frame axis, so the two axes have to
        # agree by construction - the property that lets the dump carry no
        # lexicon of its own.
        seg_ok = all(
            np.array_equal(
                diag.segments_of(tag)["seg_label"],
                np.array([int(item.lemma[1:]) for item in scored_tracebacks[tag]]),
            )
            for tag in tags
        )
        _check("segment labels live in the same index space as frame labels", seg_ok)

        table_out = diag.sequence_table()
        rows = {tag: i for i, tag in enumerate(table_out["seq_tag"])}
        last_err, sum_err = 0.0, 0.0
        for tag in tags:
            i = rows[tag]
            items = scored_tracebacks[tag]
            if not items:
                continue
            last_err = max(last_err, abs(table_out["am_last"][i] - items[-1].am_score))
            sum_err = max(sum_err, abs(table_out["am_sum"][i] - sum(it.am_score for it in items)))
        _check(
            "both readings of the traceback scores are recorded",
            last_err < 1e-9 and sum_err < 1e-9,
            f"last {last_err:.2e}, sum {sum_err:.2e}",
        )
        _check(
            "per-frame normalization divides by the sequence's frame count",
            np.allclose(
                table_out["total_last_per_frame"] * table_out["num_frames"],
                table_out["total_last"],
                equal_nan=True,
            ),
        )
        i = rows[silent_tag]
        _check(
            "an empty traceback stays in the table, as nan",
            table_out["num_segments"][i] == 0
            and np.isnan(table_out["am_last"][i])
            and table_out["am_sum"][i] == 0.0
            and not np.isnan(table_out["mean_assigned_cost"][i]),
        )

    # The probe must be incapable of changing what the pass computes; that is
    # what allows it to be attached to a pass whose model is used downstream.
    with_probe = MeanAccumulator(num_clusters, dim)
    without_probe = MeanAccumulator(num_clusters, dim)
    for accumulator, attached in ((with_probe, FrameDiagnostics()), (without_probe, None)):
        run_chunk(
            features=_ListSource(list(zip(tags, feats))),
            model=previous,
            recognizer=_StubRecognizer(table, scored_tracebacks),
            accumulator=accumulator,
            probe=attached,
            verbosity=0,
        )
    _check(
        "attaching a probe does not change the result",
        np.array_equal(
            with_probe.finalize(previous).centroids, without_probe.finalize(previous).centroids
        ),
    )

    print("model artifacts")
    with tempfile.TemporaryDirectory() as tmp:
        # A model class the rest of the package has never heard of must work
        # end to end: the job declares only the model directory, so persistence
        # and parameter carry-over come purely from artifacts() + the manifest.
        diag = _DiagModel(
            rng.randn(num_clusters, dim),
            np.abs(rng.randn(num_clusters, dim)) + 1.0,
            np.ones(num_clusters) / num_clusters,
        )
        directory = os.path.join(tmp, "diag")
        diag.save(directory)
        _check(
            "unknown model persists its own artifact set",
            read_manifest(directory)["artifacts"] == ["centroids", "priors", "variances"],
        )
        restored = _DiagModel.load(directory)
        _check(
            "unknown model round-trips",
            all(np.array_equal(v, restored.artifacts()[k]) for k, v in diag.artifacts().items()),
        )

        for model in (EuclideanModel(rng.randn(num_clusters, dim)), prev_gauss):
            path = os.path.join(tmp, type(model).__name__)
            model.save(path)
            _check(
                f"{type(model).__name__}: load_model dispatches via the manifest",
                type(load_model(path)) is type(model),
            )

    dead = np.zeros(num_clusters, bool)
    dead[[1, 4]] = True
    # Operates on the artifact mapping, not a model: a model built from
    # pre-fallback arrays may not survive construction (GaussianModel inverts
    # its covariances, and a dead cluster's is all zeros).
    carried = keep_previous_where_dead(
        {
            "centroids": np.zeros((num_clusters, dim)),
            "variances": np.zeros((num_clusters, dim)),
            "priors": np.zeros(num_clusters),
        },
        diag,
        dead,
    )
    _check(
        "dead-cluster carry-over is generic over artifacts",
        np.array_equal(carried["variances"][dead], diag.variances[dead])
        and np.array_equal(carried["priors"][dead], diag.priors[dead])
        and np.array_equal(carried["priors"][~dead], np.zeros(num_clusters - 2)),
    )
    _check(
        "a model rebuilt from the carried artifacts is intact",
        np.array_equal(
            _DiagModel.from_artifacts(carried, {}).variances[dead], diag.variances[dead]
        ),
    )

    print("bind_model")
    # Every accumulator takes the epoch's model before the first observe, and
    # the ones that do not need it must be unaffected by getting it - that is
    # what keeps the hook from becoming another thing to special-case.
    class _BindRecorder:
        def __init__(self):
            self.bound = []
            self.observed = 0

        def bind_model(self, model):
            self.bound.append(model)

        def observe(self, features, posteriors):
            if not self.bound:
                raise AssertionError("observe() before bind_model()")
            self.observed += 1

        def merge(self, other):
            return self

        def finalize(self, previous):
            return previous

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return self

    recorder = _BindRecorder()
    run_chunk(
        features=_ListSource([(tags[i], feats[i]) for i in range(3)]),
        model=previous,
        recognizer=_StubRecognizer({tags[i]: table[tags[i]] for i in range(3)}),
        accumulator=recorder,
        counter=None,
        verbosity=0,
    )
    _check(
        "run_chunk binds the model exactly once, before observing",
        recorder.bound == [previous] and recorder.observed == 3,
        f"{len(recorder.bound)} bind(s), {recorder.observed} observe(s)",
    )

    bound_mean = MeanAccumulator(num_clusters, dim)
    unbound_mean = MeanAccumulator(num_clusters, dim)
    bound_mean.bind_model(previous)
    for f, l in zip(feats, labels):
        bound_mean.observe(f, l)
        unbound_mean.observe(f, l)
    _check(
        "bind_model is a no-op for accumulators that ignore it",
        np.array_equal(
            bound_mean.finalize(previous).centroids, unbound_mean.finalize(previous).centroids
        ),
    )

    print("GaussianMixtureModel — scoring")
    num_labels, num_densities, mix_dim = 4, 6, 3
    mix_rng = np.random.RandomState(7)
    mix_means = mix_rng.randn(num_densities, mix_dim) * 2
    _a = mix_rng.randn(num_densities, mix_dim, mix_dim)
    mix_covs = np.einsum("cij,ckj->cik", _a, _a) + np.eye(mix_dim)[None] * mix_dim
    mix_weights = mix_rng.dirichlet(np.ones(num_densities), size=num_labels)
    gmm = GaussianMixtureModel(mix_means, mix_covs, mix_weights)
    mix_feats = mix_rng.randn(37, mix_dim)

    _check(
        "num_clusters is the score width (labels), densities counted separately",
        gmm.num_clusters == num_labels
        and gmm.num_labels == num_labels
        and gmm.num_densities == num_densities
        and gmm.scores(mix_feats).shape == (len(mix_feats), num_labels),
    )

    # At this dimension exp() is safe, so the definition can be checked directly.
    naive = -np.log(np.exp(-gmm.scores_gaussian(mix_feats)) @ mix_weights.T)
    _check(
        "scores == -log sum_c w_lc p(x|c)",
        np.allclose(gmm.scores(mix_feats), naive, atol=1e-9),
        f"max err {np.abs(gmm.scores(mix_feats) - naive).max():.2e}",
    )

    small = mix_rng.rand(5, 4) + 0.1
    small_b = mix_rng.rand(4, 3) + 0.1
    _check(
        "neg_log_matmul == -log((e**-A) @ (e**-B))",
        np.allclose(
            neg_log_matmul(small, small_b), -np.log(np.exp(-small) @ np.exp(-small_b))
        ),
    )

    # The reason any of this is in the log domain: at the real feature
    # dimension the intermediate probabilities are not representable.
    big_dim, big_densities = 512, 8
    big_means = mix_rng.randn(big_densities, big_dim) * 1.3
    big_covs = np.stack([np.eye(big_dim) * 90.0] * big_densities)
    big_weights = mix_rng.dirichlet(np.ones(big_densities), size=num_labels)
    big_gmm = GaussianMixtureModel(big_means, big_covs, big_weights)
    big_feats = mix_rng.randn(11, big_dim) * 9.5
    big_gauss = big_gmm.scores_gaussian(big_feats)
    big_scores = big_gmm.scores(big_feats)
    _check(
        "D=512 scores stay finite where exp(-cost) underflows to zero",
        np.isfinite(big_scores).all() and np.all(np.exp(-big_gauss) == 0.0),
        f"costs ~{big_gauss.mean():.0f}, exp(-cost) max {np.exp(-big_gauss).max():.1e}",
    )
    _check(
        "a label's score is never better than its best density",
        np.all(big_scores >= big_gauss.min(axis=1, keepdims=True) - 1e-6),
    )

    print("GaussianMixtureModel — responsibilities (E-step)")
    label_gammas = mix_rng.dirichlet(np.ones(num_labels), size=len(mix_feats))
    density_gammas, joint_counts = gmm.responsibilities(mix_feats, label_gammas)

    # Brute force, straight from the definition: gamma_tl * p(c | l, x_t).
    p_c = np.exp(-gmm.scores_gaussian(mix_feats))
    p_l = np.exp(-gmm.scores(mix_feats))
    ref_joint = np.zeros((len(mix_feats), num_labels, num_densities))
    for t in range(len(mix_feats)):
        for l in range(num_labels):
            ref_joint[t, l] = label_gammas[t, l] * mix_weights[l] * p_c[t] / p_l[t, l]
    _check(
        "density posteriors match the brute-force definition",
        np.allclose(density_gammas, ref_joint.sum(axis=1), atol=1e-10),
        f"max err {np.abs(density_gammas - ref_joint.sum(1)).max():.2e}",
    )
    _check(
        "mixture-weight counts match the brute-force definition",
        np.allclose(joint_counts, ref_joint.sum(axis=0), atol=1e-10),
        f"max err {np.abs(joint_counts - ref_joint.sum(0)).max():.2e}",
    )
    # Both are marginals of one joint, so they have to agree with each other
    # and conserve the mass the recognizer handed over. These hold for any
    # parameters, which makes them the cheap check on a real corpus too.
    _check(
        "mass is conserved: sum_c gamma_tc == sum_l gamma_tl",
        np.allclose(density_gammas.sum(axis=1), label_gammas.sum(axis=1)),
    )
    _check(
        "the two marginals agree: counts.sum(0) == density totals",
        np.allclose(joint_counts.sum(axis=0), density_gammas.sum(axis=0)),
    )
    blocked = GaussianMixtureModel(mix_means, mix_covs, mix_weights)
    blocked.RESPONSIBILITY_BLOCK_FRAMES = 1  # one frame per block
    _check(
        "frame blocking does not change the result",
        np.allclose(blocked.responsibilities(mix_feats, label_gammas)[0], density_gammas)
        and np.allclose(blocked.responsibilities(mix_feats, label_gammas)[1], joint_counts),
    )
    big_dense, big_counts = big_gmm.responsibilities(
        big_feats, mix_rng.dirichlet(np.ones(num_labels), size=len(big_feats))
    )
    _check(
        "the E-step survives D=512, where the naive ratio is 0/0",
        np.isfinite(big_dense).all() and np.isfinite(big_counts).all()
        and np.allclose(big_counts.sum(0), big_dense.sum(0)),
    )

    print("SoftGaussianAccumulator — starved covariances")
    # A cluster can pass min_mass and still have too few frames to support a
    # [D, D] covariance: rank <= frame count. The next model's constructor
    # inverts it, so this has to be caught where it is produced.
    starve_dim, starve_k = 8, 4
    prev_starve = GaussianModel(
        rng.randn(starve_k, starve_dim), np.stack([np.eye(starve_dim)] * starve_k)
    )
    starve = SoftGaussianAccumulator(starve_k, starve_dim)
    # cluster 0: plenty of frames. cluster 1: three frames, so rank <= 3 < 8.
    plenty = rng.randn(200, starve_dim)
    g0 = np.zeros((200, starve_k)); g0[:, 0] = 1.0
    starve.observe(plenty, g0)
    few = rng.randn(3, starve_dim)
    g1 = np.zeros((3, starve_k)); g1[:, 1] = 1.0
    starve.observe(few, g1)

    starved_model = starve.finalize(prev_starve)
    _check(
        "a rank-deficient covariance keeps the previous parameters instead of crashing",
        np.array_equal(starved_model.covs[1], prev_starve.covs[1])
        and np.array_equal(starved_model.centroids[1], prev_starve.centroids[1]),
    )
    _check(
        "a well-fed cluster in the same epoch is still re-estimated",
        not np.array_equal(starved_model.covs[0], prev_starve.covs[0]),
    )
    _check(
        "the resulting model is invertible, which is the point",
        np.isfinite(starved_model.scores(rng.randn(5, starve_dim))).all(),
    )
    # Without the guard this is exactly the LinAlgError seen in the wild.
    raw = np.cov(few.T, bias=True)
    _check(
        "and the covariance it rejected really was singular",
        _raises(lambda: np.linalg.inv(raw), np.linalg.LinAlgError)
        or np.linalg.eigvalsh(raw).min() <= np.linalg.eigvalsh(raw).max() * starve_dim * 2.3e-16,
    )

    print("MixtureGaussianAccumulator")
    mix_seqs = [mix_rng.randn(mix_rng.randint(15, 40), mix_dim) for _ in range(12)]
    mix_gammas = [mix_rng.dirichlet(np.ones(num_labels), size=len(f)) for f in mix_seqs]
    mix_lengths = [len(f) for f in mix_seqs]

    unbound = MixtureGaussianAccumulator(num_labels)
    try:
        unbound.observe(mix_seqs[0], mix_gammas[0])
        raised = False
    except RuntimeError:
        raised = True
    _check("observe() before bind_model() is refused", raised)

    def _mix_acc(**kwargs):
        acc = MixtureGaussianAccumulator(num_labels, **kwargs)
        acc.bind_model(gmm)
        return acc

    ref_mix = _mix_acc()
    for f, g in zip(mix_seqs, mix_gammas):
        ref_mix.observe(f, g)
    ref_model = ref_mix.finalize(gmm)
    _check(
        "finalize produces a usable model (weights normalized, shapes right)",
        ref_model.mixtures.shape == (num_labels, num_densities)
        and ref_model.centroids.shape == (num_densities, mix_dim)
        and np.allclose(ref_model.mixtures.sum(-1), 1.0),
    )

    mix_errs = []
    for chunks in (2, 3, 5):
        merged = None
        for part in plan_chunks(mix_lengths, chunks):
            a = _mix_acc()
            for i in part:
                a.observe(mix_seqs[i], mix_gammas[i])
            # Through the state dict, exactly as the reduce step does it, and
            # through pickle, exactly as save_chunk/load_chunk do.
            state = pickle.loads(pickle.dumps(a.state_dict()))
            a = MixtureGaussianAccumulator(num_labels).load_state_dict(state)
            merged = a if merged is None else merged.merge(a)
        model = merged.finalize(gmm)
        mix_errs.append(
            max(
                np.abs(model.centroids - ref_model.centroids).max(),
                np.abs(model.covs - ref_model.covs).max(),
                np.abs(model.mixtures - ref_model.mixtures).max(),
            )
        )
    _check(
        "merge associative over chunk counts (via pickled state)",
        max(mix_errs) < 1e-9,
        f"max err {max(mix_errs):.2e}",
    )

    shuffled = None
    order = list(range(len(mix_seqs)))
    mix_rng.shuffle(order)
    for i in order:
        a = _mix_acc()
        a.observe(mix_seqs[i], mix_gammas[i])
        shuffled = a if shuffled is None else shuffled.merge(a)
    shuffled_model = shuffled.finalize(gmm)
    _check(
        "merge order does not matter",
        np.abs(shuffled_model.mixtures - ref_model.mixtures).max() < 1e-9
        and np.abs(shuffled_model.centroids - ref_model.centroids).max() < 1e-9,
    )

    # Means-only never allocates the second moment, so merging two chunks of one
    # used to add None to None. Only this mode reaches that line: a weights-only
    # run never observes into the Gaussian accumulator, so its merge returns
    # early. Single-chunk runs finalize fine, which is why it surfaced in reduce.
    means_only_ref = _mix_acc(update_covariances=False)
    for f, g in zip(mix_seqs, mix_gammas):
        means_only_ref.observe(f, g)
    means_only_model = means_only_ref.finalize(gmm)
    means_only_merged = None
    for part in plan_chunks(mix_lengths, 3):
        a = _mix_acc(update_covariances=False)
        for i in part:
            a.observe(mix_seqs[i], mix_gammas[i])
        state = pickle.loads(pickle.dumps(a.state_dict()))
        a = MixtureGaussianAccumulator(num_labels).load_state_dict(state)
        means_only_merged = a if means_only_merged is None else means_only_merged.merge(a)
    means_only_chunked = means_only_merged.finalize(gmm)
    _check(
        "means-only merges over chunks, moving means and keeping covariances",
        np.abs(means_only_chunked.centroids - means_only_model.centroids).max() < 1e-9
        and np.abs(means_only_chunked.centroids - ref_model.centroids).max() < 1e-9
        and np.allclose(means_only_chunked.covs, gmm.covs),
    )

    full_chunk = _mix_acc()
    full_chunk.observe(mix_seqs[0], mix_gammas[0])
    means_only_chunk = _mix_acc(update_covariances=False)
    means_only_chunk.observe(mix_seqs[1], mix_gammas[1])
    try:
        full_chunk.merge(means_only_chunk)
        raised = False
    except ValueError:
        raised = True
    _check("merging chunks that disagree on update_covariances is refused", raised)

    try:
        MixtureGaussianAccumulator(
            num_labels, pool_covariances=True, update_covariances=False
        )
        raised = False
    except ValueError:
        raised = True
    _check("pool_covariances without update_covariances is refused", raised)

    # One density per label with weight 1 is the degenerate mixture, and the
    # E-step then has nothing to do: the accumulator must reduce exactly to the
    # single-density one it delegates to. This is the tie between the new path
    # and the one already in use.
    eye_weights = np.eye(num_labels)
    eye_means = mix_rng.randn(num_labels, mix_dim)
    _b = mix_rng.randn(num_labels, mix_dim, mix_dim)
    eye_covs = np.einsum("cij,ckj->cik", _b, _b) + np.eye(mix_dim)[None] * mix_dim
    eye_gmm = GaussianMixtureModel(eye_means, eye_covs, eye_weights)
    eye_prev_gauss = GaussianModel(eye_means, eye_covs)

    degenerate = MixtureGaussianAccumulator(num_labels)
    degenerate.bind_model(eye_gmm)
    plain = SoftGaussianAccumulator(num_labels, mix_dim)
    for f, g in zip(mix_seqs, mix_gammas):
        degenerate.observe(f, g)
        plain.observe(f, g)
    deg_model = degenerate.finalize(eye_gmm)
    plain_model = plain.finalize(eye_prev_gauss)
    _check(
        "one-hot mixtures reproduce SoftGaussianAccumulator exactly",
        np.allclose(deg_model.centroids, plain_model.centroids, atol=1e-12)
        and np.allclose(deg_model.covs, plain_model.covs, atol=1e-12)
        and np.allclose(deg_model.mixtures, eye_weights),
        f"max err {np.abs(deg_model.centroids - plain_model.centroids).max():.2e}",
    )

    # A label nobody aligned to keeps its previous weights: an unnormalized row
    # would break the model's own invariant, a zeroed one would leave the label
    # with no emission probability at all.
    starved = _mix_acc()
    one_label = np.zeros((len(mix_seqs[0]), num_labels))
    one_label[:, 0] = 1.0
    starved.observe(mix_seqs[0], one_label)
    starved_model = starved.finalize(gmm)
    _check(
        "a label with no mass keeps its previous mixture weights",
        np.array_equal(starved_model.mixtures[1:], gmm.mixtures[1:])
        and not np.array_equal(starved_model.mixtures[0], gmm.mixtures[0]),
    )
    _check(
        "a density with no mass keeps its previous mean",
        np.allclose(starved_model.mixtures.sum(-1), 1.0),
    )

    floored = _mix_acc(mixture_floor=1e-3)
    floored.observe(mix_seqs[0], one_label)
    floored_model = floored.finalize(gmm)
    _check(
        "mixture_floor keeps every density reachable, still normalized",
        (floored_model.mixtures[0] > 0).all()
        and np.allclose(floored_model.mixtures.sum(-1), 1.0),
    )
    _check(
        "mixture_floor=0 (the default) is textbook EM and can zero a weight",
        (ref_mix.finalize(gmm).mixtures >= 0).all(),
    )

    with tempfile.TemporaryDirectory() as tmp:
        directory = os.path.join(tmp, "gmm")
        ref_model.save(directory)
        _check(
            "mixture model persists all three artifacts",
            read_manifest(directory)["artifacts"] == ["centroids", "covs", "mixtures"],
        )
        loaded = load_model(directory)
        _check(
            "mixture model round-trips through load_model",
            type(loaded) is GaussianMixtureModel
            and np.array_equal(loaded.mixtures, ref_model.mixtures)
            and np.allclose(loaded.scores(mix_feats), ref_model.scores(mix_feats)),
        )

        class _Mismatched(GaussianMixtureModel):
            ARTIFACT_NAMES = ("centroids", "covs")

        try:
            _Mismatched(mix_means, mix_covs, mix_weights).save(os.path.join(tmp, "bad"))
            caught = False
        except ValueError:
            caught = True
        _check("save() rejects an ARTIFACT_NAMES that disagrees with artifacts()", caught)

        # Frozen parameters are hardlinked to the file they were loaded from
        # instead of copied into every epoch - see ArtifactModel.save.
        src_dir = os.path.join(tmp, "src")
        ref_model.save(src_dir)
        srcs = {n: os.path.join(src_dir, f"{n}.npy") for n in ("centroids", "covs", "mixtures")}
        moved = ref_model.mixtures + 1.0
        moved /= moved.sum(-1, keepdims=True)
        frozen = GaussianMixtureModel(centroids=srcs["centroids"], covs=srcs["covs"], mixtures=moved)
        linked_dir = os.path.join(tmp, "linked")
        frozen.save(linked_dir, sources=srcs)

        def _ino(path):
            return os.stat(path).st_ino

        _check(
            "save() hardlinks artifacts unchanged from their source",
            _ino(os.path.join(linked_dir, "covs.npy")) == _ino(srcs["covs"])
            and _ino(os.path.join(linked_dir, "centroids.npy")) == _ino(srcs["centroids"]),
        )
        _check(
            "save() writes an artifact that changed instead of linking it",
            _ino(os.path.join(linked_dir, "mixtures.npy")) != _ino(srcs["mixtures"])
            and np.array_equal(np.load(os.path.join(linked_dir, "mixtures.npy")), moved),
        )
        _check(
            "the manifest records what was linked, and only then",
            sorted(read_manifest(linked_dir).get("linked", {})) == ["centroids", "covs"]
            and "linked" not in read_manifest(src_dir),
        )
        reloaded = load_model(linked_dir)
        _check(
            "a model with linked artifacts loads back identical",
            np.array_equal(reloaded.covs, frozen.covs)
            and np.array_equal(reloaded.mixtures, frozen.mixtures),
        )

        # The hazard the atomic write is for: np.save(path) truncates the inode
        # in place, so overwriting a linked name with different values would
        # have rewritten the source, and every other epoch sharing it.
        covs_before = np.load(srcs["covs"]).copy()
        GaussianMixtureModel(
            centroids=ref_model.centroids, covs=ref_model.covs * 2.0, mixtures=moved
        ).save(linked_dir)
        _check(
            "rewriting a directory of links leaves their sources untouched",
            np.array_equal(np.load(srcs["covs"]), covs_before)
            and np.array_equal(np.load(os.path.join(linked_dir, "covs.npy")), ref_model.covs * 2.0),
        )

        real_link = os.link

        def _refuse_link(*args, **kwargs):
            raise OSError(18, "Invalid cross-device link")

        os.link = _refuse_link
        try:
            fallback_dir = os.path.join(tmp, "fallback")
            frozen.save(fallback_dir, sources=srcs)
        finally:
            os.link = real_link
        _check(
            "a refused link falls back to writing the array",
            _ino(os.path.join(fallback_dir, "covs.npy")) != _ino(srcs["covs"])
            and np.array_equal(np.load(os.path.join(fallback_dir, "covs.npy")), frozen.covs)
            and "linked" not in read_manifest(fallback_dir),
        )
        _check(
            "no temporary files are left behind",
            not [f for d in (linked_dir, fallback_dir) for f in os.listdir(d) if f.endswith(".tmp")],
        )

    print("PerLabelMixtureModel")
    per_n = 3
    pl_means = mix_rng.randn(num_labels * per_n, mix_dim) * 2
    _c = mix_rng.randn(num_labels * per_n, mix_dim, mix_dim)
    pl_covs = np.einsum("cij,ckj->cik", _c, _c) + np.eye(mix_dim)[None] * mix_dim
    pl_weights = mix_rng.dirichlet(np.ones(per_n), size=num_labels)      # [L, n]
    pl = PerLabelMixtureModel(pl_means, pl_covs, pl_weights)

    _check(
        "counts multiply out: L labels x n densities each",
        pl.num_labels == num_labels
        and pl.densities_per_label == per_n
        and pl.num_densities == num_labels * per_n
        and pl.num_clusters == num_labels,
    )
    try:
        PerLabelMixtureModel(pl_means[:-1], pl_covs[:-1], pl_weights)
        caught = False
    except ValueError:
        caught = True
    _check("a density count that does not factor is refused", caught)

    # The same model as a shared codebook whose weights are block diagonal.
    # These have to agree exactly - which is what makes the per-label class an
    # optimization rather than a different algorithm.
    block = np.zeros((num_labels, num_labels * per_n))
    for l in range(num_labels):
        block[l, l * per_n : (l + 1) * per_n] = pl_weights[l]
    as_shared = GaussianMixtureModel(pl_means, pl_covs, block)
    pl_scores = pl.scores(mix_feats)
    _check(
        "scores match the equivalent block-diagonal shared codebook",
        np.allclose(pl_scores, as_shared.scores(mix_feats), atol=1e-10),
        f"max err {np.abs(pl_scores - as_shared.scores(mix_feats)).max():.2e}",
    )
    pl_dense, pl_counts = pl.responsibilities(mix_feats, label_gammas)
    sh_dense, sh_counts = as_shared.responsibilities(mix_feats, label_gammas)
    _check(
        "density posteriors match the block-diagonal shared codebook",
        np.allclose(pl_dense, sh_dense, atol=1e-12),
        f"max err {np.abs(pl_dense - sh_dense).max():.2e}",
    )
    # The shared form spends [L, C] on weights that are structurally zero; the
    # per-label form keeps only the [L, n] block that can be nonzero.
    folded = np.stack(
        [sh_counts[l, l * per_n : (l + 1) * per_n] for l in range(num_labels)]
    )
    _check(
        "mixture counts match, once the structural zeros are folded away",
        np.allclose(pl_counts, folded, atol=1e-12)
        and np.allclose(sh_counts.sum() - folded.sum(), 0.0),
    )
    _check(
        "mass is conserved for the per-label layout too",
        np.allclose(pl_dense.sum(axis=1), label_gammas.sum(axis=1))
        and np.allclose(pl_counts.sum(axis=0), pl_dense.sum(axis=0).reshape(num_labels, per_n).sum(0)),
    )

    # Brute force, straight from the definition, restricted to the label's own
    # densities.
    pl_pc = np.exp(-pl.scores_gaussian(mix_feats))
    ref_pl = np.zeros((len(mix_feats), num_labels, per_n))
    for t in range(len(mix_feats)):
        for l in range(num_labels):
            own = pl_pc[t, l * per_n : (l + 1) * per_n]
            ref_pl[t, l] = label_gammas[t, l] * pl_weights[l] * own / (pl_weights[l] @ own)
    _check(
        "per-label E-step matches the brute-force definition",
        np.allclose(pl_dense.reshape(len(mix_feats), num_labels, per_n), ref_pl, atol=1e-12),
        f"max err {np.abs(pl_dense.reshape(len(mix_feats), num_labels, per_n) - ref_pl).max():.2e}",
    )

    print("MixtureGaussianAccumulator — per-label layout")
    # The accumulator is told nothing about the layout; it takes the shape of
    # the statistic from the model it is bound to.
    def _pl_acc():
        acc = MixtureGaussianAccumulator(num_labels)
        acc.bind_model(pl)
        return acc

    pl_ref = _pl_acc()
    for f, g in zip(mix_seqs, mix_gammas):
        pl_ref.observe(f, g)
    _check(
        "the mixture statistic is shaped like the model's own weights",
        pl_ref.weighted_c.shape == (num_labels, per_n),
        str(pl_ref.weighted_c.shape),
    )
    pl_ref_model = pl_ref.finalize(pl)
    _check(
        "finalize returns a per-label model with normalized weights",
        type(pl_ref_model) is PerLabelMixtureModel
        and pl_ref_model.mixtures.shape == (num_labels, per_n)
        and np.allclose(pl_ref_model.mixtures.sum(-1), 1.0),
    )

    pl_errs = []
    for chunks in (2, 4):
        merged = None
        for part in plan_chunks(mix_lengths, chunks):
            a = _pl_acc()
            for i in part:
                a.observe(mix_seqs[i], mix_gammas[i])
            state = pickle.loads(pickle.dumps(a.state_dict()))
            a = MixtureGaussianAccumulator(num_labels).load_state_dict(state)
            merged = a if merged is None else merged.merge(a)
        model = merged.finalize(pl)
        pl_errs.append(
            max(
                np.abs(model.centroids - pl_ref_model.centroids).max(),
                np.abs(model.mixtures - pl_ref_model.mixtures).max(),
            )
        )
    _check(
        "merge associative for the per-label layout, through pickled state",
        max(pl_errs) < 1e-9,
        f"max err {max(pl_errs):.2e}",
    )

    # Training one epoch either way has to land on the same model, up to the
    # structural zeros - the strongest statement that the two are one algorithm.
    sh_acc = MixtureGaussianAccumulator(num_labels)
    sh_acc.bind_model(as_shared)
    for f, g in zip(mix_seqs, mix_gammas):
        sh_acc.observe(f, g)
    sh_model = sh_acc.finalize(as_shared)
    folded_w = np.stack(
        [sh_model.mixtures[l, l * per_n : (l + 1) * per_n] for l in range(num_labels)]
    )
    _check(
        "one epoch of per-label == one epoch of block-diagonal shared",
        np.allclose(pl_ref_model.centroids, sh_model.centroids, atol=1e-9)
        and np.allclose(pl_ref_model.covs, sh_model.covs, atol=1e-9)
        and np.allclose(pl_ref_model.mixtures, folded_w, atol=1e-9),
        f"max weight err {np.abs(pl_ref_model.mixtures - folded_w).max():.2e}",
    )
    _check(
        "EM keeps the block structure: the structural zeros stay zero",
        np.array_equal(sh_model.mixtures == 0, block == 0),
    )

    with tempfile.TemporaryDirectory() as tmp:
        directory = os.path.join(tmp, "pl")
        pl_ref_model.save(directory)
        reloaded = load_model(directory)
        _check(
            "per-label model round-trips and load_forward_model adapts it",
            type(reloaded) is PerLabelMixtureModel
            and np.allclose(
                load_forward_model(directory).forward(mix_feats),
                pl_ref_model.scores(mix_feats),
            ),
        )

    print("MixtureGaussianAccumulator — pooled covariances")
    # One covariance per label, shared by its densities. Accumulated at group level
    # rather than pooled afterwards, so the equivalence is the thing to pin down.
    def _pooled_acc(pool=True):
        acc = MixtureGaussianAccumulator(num_labels, pool_covariances=pool)
        acc.bind_model(pl)
        return acc

    pooled_acc, unpooled_acc = _pooled_acc(True), _pooled_acc(False)
    for f, g in zip(mix_seqs, mix_gammas):
        pooled_acc.observe(f, g)
        unpooled_acc.observe(f, g)
    pooled_model = pooled_acc.finalize(pl)
    unpooled_model = unpooled_acc.finalize(pl)

    _check(
        "every density of a label shares one covariance",
        all(
            np.array_equal(pooled_model.covs[l * per_n], pooled_model.covs[l * per_n + k])
            for l in range(num_labels)
            for k in range(per_n)
        )
        and not np.allclose(pooled_model.covs[0], pooled_model.covs[per_n]),
    )
    _check(
        "pooling leaves the means and the mixture weights untouched",
        np.allclose(pooled_model.centroids, unpooled_model.centroids)
        and np.allclose(pooled_model.mixtures, unpooled_model.mixtures)
        and not np.allclose(pooled_model.centroids[0], pooled_model.centroids[1]),
    )
    # Accumulating the second moment per label must equal averaging the per-density
    # covariances by their counts - that identity is what lets the state shrink.
    counts = unpooled_acc.gaussian_accumulator.counts
    # Only over labels whose every density was re-estimated: a starved density in
    # the unpooled model carries the *previous* covariance, which is exactly the
    # case pooling is meant to avoid and cannot be a reference for it.
    live = (counts > 0) & (counts >= 1.0)
    full = [l for l in range(num_labels) if live[l * per_n : (l + 1) * per_n].all()]
    reference = np.stack(
        [
            sum(counts[l * per_n + k] * unpooled_model.covs[l * per_n + k] for k in range(per_n))
            / sum(counts[l * per_n + k] for k in range(per_n))
            for l in full
        ]
    )
    got = np.stack([pooled_model.covs[l * per_n] for l in full])
    _check(
        "pooled == count-weighted mean of the per-density covariances",
        len(full) > 0 and np.allclose(got, reference, atol=1e-9),
        f"{len(full)}/{num_labels} labels fully alive, max err "
        f"{np.abs(got - reference).max():.2e}",
    )
    starved = [l for l in range(num_labels) if not live[l * per_n : (l + 1) * per_n].all()]
    _check(
        "a label whose densities are individually starved still gets a covariance",
        all(
            not np.array_equal(pooled_model.covs[l * per_n], pl.covs[l * per_n])
            for l in starved
            if counts[l * per_n : (l + 1) * per_n].sum() >= 1.0
        ),
        f"{len(starved)} label(s) with a starved density",
    )
    _check(
        "the second moment is stored per label, not per density",
        pooled_acc.gaussian_accumulator.weighted_sq.shape[0] == num_labels
        and unpooled_acc.gaussian_accumulator.weighted_sq.shape[0] == num_labels * per_n,
        f"{pooled_acc.gaussian_accumulator.weighted_sq.shape[0]} vs "
        f"{unpooled_acc.gaussian_accumulator.weighted_sq.shape[0]}",
    )

    pool_errs = []
    for chunks in (2, 4):
        merged = None
        for part in plan_chunks(mix_lengths, chunks):
            a = _pooled_acc(True)
            for i in part:
                a.observe(mix_seqs[i], mix_gammas[i])
            state = pickle.loads(pickle.dumps(a.state_dict()))
            a = MixtureGaussianAccumulator(num_labels).load_state_dict(state)
            merged = a if merged is None else merged.merge(a)
        model = merged.finalize(pl)
        pool_errs.append(np.abs(model.covs - pooled_model.covs).max())
    _check(
        "pooled merge associative, and the pooling survives the state dict",
        max(pool_errs) < 1e-9,
        f"max err {max(pool_errs):.2e}",
    )
    _check(
        "a shared codebook cannot pool - it has no per-label grouping",
        _raises(lambda: _pooled_acc.__wrapped__ if False else
                MixtureGaussianAccumulator(num_labels, pool_covariances=True).bind_model(gmm),
                TypeError),
    )
    _check(
        "mixing pooled and unpooled statistics is refused",
        _raises(lambda: _pooled_acc(True).merge(_pooled_acc(False)), ValueError),
    )

    print("MixtureGaussianAccumulator — run_chunk + reduce_chunks")
    mix_tags = [f"mseq{i}" for i in range(len(mix_seqs))]
    with tempfile.TemporaryDirectory() as tmp:
        paths_mix = []
        for chunk_idx, part in enumerate(plan_chunks(mix_lengths, 3)):
            result = run_chunk(
                features=_ListSource([(mix_tags[i], mix_seqs[i]) for i in part]),
                model=gmm,
                recognizer=_StubFBRecognizer({mix_tags[i]: mix_gammas[i] for i in part}),
                accumulator=MixtureGaussianAccumulator(num_labels),
                counter=None,
                verbosity=0,
            )
            path = os.path.join(tmp, f"chunk_mix.{chunk_idx}.pkl")
            save_chunk(result, path)
            paths_mix.append(path)
        model_mix, _s, totals_mix, _h = reduce_chunks(
            chunk_paths=paths_mix,
            accumulator_factory=lambda: MixtureGaussianAccumulator(num_labels),
            previous_model=gmm,
        )
        _check(
            "end to end == the single-process reference",
            np.abs(model_mix.mixtures - ref_model.mixtures).max() < 1e-9
            and np.abs(model_mix.centroids - ref_model.centroids).max() < 1e-9
            and np.abs(model_mix.covs - ref_model.covs).max() < 1e-9,
            f"max weight err {np.abs(model_mix.mixtures - ref_model.mixtures).max():.2e}",
        )
        _check(
            "the reduce step's zero-argument factory still works",
            totals_mix["num_seqs"] == len(mix_seqs),
            str(totals_mix),
        )

    print("continuation")
    # Continuing a run must reuse the jobs an uninterrupted run would create.
    # That holds only while every epoch's model spec has the same shape, so
    # this guards against a first-epoch special case creeping back in.
    from sisyphus import tk

    # Imported here rather than at module level: this package is the algorithm
    # layer and must not depend on the sisyphus wiring, but the property under
    # test lives in the pipeline that wires it up.
    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        chunked_clustering,
    )

    common = dict(
        features_hdf=tk.Path("/f.hdf"),
        recognition_config=tk.Path("/r.config"),
        lexicon=tk.Path("/l.gz"),
        num_clusters=40,
    )
    ten = chunked_clustering(num_epochs=10, initial_centroids=tk.Path("/init/c.npy"), **common)
    head = chunked_clustering(num_epochs=5, initial_centroids=tk.Path("/init/c.npy"), **common)
    tail = chunked_clustering(num_epochs=5, initial_centroids=head.out_centroids[5], **common)
    ids_ten = [j._sis_id() for j in ten.jobs]
    ids_split = [j._sis_id() for j in head.jobs + tail.jobs]
    diverged = next((i for i, (a, b) in enumerate(zip(ids_ten, ids_split), 1) if a != b), None)
    _check("5+5 continued == 10 in one call", ids_ten == ids_split,
           "identical" if diverged is None else f"diverges at epoch {diverged}")

    print("hashes")
    # Recorded from the graph these arguments produced before hypothesis output
    # and guided scoring existed. Everything added since has to stay out of the
    # hash - the models are unchanged, and re-running a whole sweep to get a
    # by-product of a search that already ran would defeat the purpose. A
    # deliberate change to what an epoch computes updates these literals; an
    # unexpected failure here means something non-computational leaked in
    # (check GuidedClusteringEpochJob.hash and __sis_hash_exclude__).
    gauss = chunked_clustering(
        num_epochs=1,
        initial_centroids=tk.Path("/init/c.npy"),
        initial_covs=tk.Path("/init/cov.npy"),
        **common,
    )
    for label, jobs, expected in (
        ("euclidean", ten.jobs[:2], ["hgUvNvDgy6ob", "3ZlRk7zFC5hg"]),
        ("gaussian", gauss.jobs, ["57zSk7rGJDVJ"]),
    ):
        got = [j._sis_id().rsplit(".", 1)[-1] for j in jobs]
        _check(f"{label} epoch job hashes unchanged", got == expected, str(got))

    # The scoring jobs hang off the epoch jobs, so asking for them must not
    # change the epoch jobs themselves.
    scored = chunked_clustering(
        num_epochs=2,
        initial_centroids=tk.Path("/init/c.npy"),
        score_reference=tk.Path("/ref.txt"),
        **common,
    )
    _check(
        "score_reference does not touch the epoch jobs",
        [j._sis_id() for j in scored.jobs] == ids_ten[:2],
    )
    _check(
        "guided scores are keyed by the model they measure",
        sorted(scored.out_guided_scores) == [0, 1]
        and sorted(scored.out_hypotheses) == [0, 1]
        and scored.guided_score_row(2) == {},
    )

    # How a task splits its CPUs between the parent and the workers is retuned
    # mid-run, so it must never move a job - and it is what the request is sized
    # from.
    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        parent_thread_budget,
    )

    threaded = chunked_clustering(
        num_epochs=2,
        initial_centroids=tk.Path("/init/c.npy"),
        num_workers=6,
        parent_threads=4,
        **common,
    )
    _check(
        "the thread split does not touch the epoch jobs",
        [j._sis_id() for j in threaded.jobs] == ids_ten[:2],
    )
    # A job whose hash already exists in the graph is that same object, built
    # with the first caller's arguments - so the request is checked on jobs no
    # earlier call has created.
    fresh = chunked_clustering(
        num_epochs=1,
        initial_centroids=tk.Path("/init/threads.npy"),
        num_workers=6,
        parent_threads=4,
        **common,
    )
    _check(
        "a task requests num_workers + parent_threads CPUs",
        [(j.rqmt["cpu"], j.parent_threads) for j in fresh.jobs] == [(10, 4)]
        and all(j.rqmt["cpu"] == 8 + 1 for j in ten.jobs),
    )
    import torch
    from threadpoolctl import threadpool_info

    def _blas_threads():
        return {i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"}

    torch_before, blas_before = torch.get_num_threads(), _blas_threads()
    with parent_thread_budget(3, num_workers=7):
        torch_inside, blas_inside = torch.get_num_threads(), _blas_threads()
    _check(
        "parent_thread_budget sizes torch and BLAS for the block, then restores them",
        torch_inside == 3 and blas_inside == {3}
        and torch.get_num_threads() == torch_before and _blas_threads() == blas_before,
        f"inside torch={torch_inside} blas={blas_inside}",
    )

    print("flavors")
    # The flag form and the flavor form have to be the same experiment, or the
    # refactor silently orphans every job the configs have already computed.
    flag_gauss = chunked_clustering(
        num_epochs=2,
        initial_centroids=tk.Path("/init/c.npy"),
        initial_covs=tk.Path("/init/cov.npy"),
        **common,
    )
    di_gauss = chunked_clustering(
        num_epochs=2,
        flavor=gaussian_flavor(
            centroids=tk.Path("/init/c.npy"),
            covs=tk.Path("/init/cov.npy"),
            recognition_config=common["recognition_config"],
            lexicon=common["lexicon"],
            num_clusters=common["num_clusters"],
        ),
        **common,
    )
    _check(
        "an explicit flavor reproduces the flag form's jobs",
        [j._sis_id() for j in di_gauss.jobs] == [j._sis_id() for j in flag_gauss.jobs],
    )
    di_euclid = chunked_clustering(
        num_epochs=2,
        flavor=euclidean_flavor(
            centroids=tk.Path("/init/c.npy"),
            recognition_config=common["recognition_config"],
            lexicon=common["lexicon"],
            num_clusters=common["num_clusters"],
        ),
        **common,
    )
    _check(
        "... for the euclidean flavor too",
        [j._sis_id() for j in di_euclid.jobs] == ids_ten[:2],
    )

    try:
        chunked_clustering(
            num_epochs=1,
            initial_centroids=tk.Path("/init/c.npy"),
            flavor=euclidean_flavor(
                centroids=tk.Path("/init/c.npy"),
                recognition_config=common["recognition_config"],
                lexicon=common["lexicon"],
                num_clusters=common["num_clusters"],
            ),
            **common,
        )
        caught = False
    except TypeError:
        caught = True
    _check("a flavor and initial_* together are refused", caught)

    # A model the pipeline has to work with without naming it: the artifact set
    # comes off the class, so the epoch-to-epoch spec is derived, not written.
    mix_kwargs = dict(
        centroids=tk.Path("/init/c.npy"),
        covs=tk.Path("/init/cov.npy"),
        mixtures=tk.Path("/init/mix.npy"),
        recognition_config=common["recognition_config"],
        lexicon=common["lexicon"],
        num_clusters=common["num_clusters"],
    )
    mix_flavor = mixture_flavor(**mix_kwargs)
    _check(
        "artifact names come from the model class, not from the pipeline",
        mix_flavor.artifact_names == ("centroids", "covs", "mixtures"),
    )
    nxt = mix_flavor.next_model(
        {n: tk.Path(f"/e1/{n}.npy") for n in mix_flavor.artifact_names}
    )
    _check(
        "next_model keeps the initial spec's shape",
        nxt.cls is mix_flavor.model.cls
        and sorted(nxt.kwargs) == sorted(mix_flavor.model.kwargs),
    )

    mix_ten = chunked_clustering(num_epochs=10, flavor=mixture_flavor(**mix_kwargs), **common)
    mix_head = chunked_clustering(num_epochs=5, flavor=mixture_flavor(**mix_kwargs), **common)
    mix_tail = chunked_clustering(
        num_epochs=5,
        flavor=mixture_flavor(
            **{
                **mix_kwargs,
                "centroids": mix_head.out_artifacts["centroids"][5],
                "covs": mix_head.out_artifacts["covs"][5],
                "mixtures": mix_head.out_artifacts["mixtures"][5],
            }
        ),
        **common,
    )
    _check(
        "5+5 continued == 10, for a three-artifact model",
        [j._sis_id() for j in mix_ten.jobs]
        == [j._sis_id() for j in mix_head.jobs + mix_tail.jobs],
    )
    _check(
        "out_artifacts exposes every artifact, epoch 0 included",
        sorted(mix_ten.out_artifacts) == ["centroids", "covs", "mixtures"]
        and sorted(mix_ten.out_artifacts["mixtures"]) == list(range(11))
        and mix_ten.out_covs is mix_ten.out_artifacts["covs"]
        and mix_ten.out_centroids is mix_ten.out_artifacts["centroids"],
    )
    _check(
        "a model without covs reports out_covs as None",
        di_euclid.out_covs is None and sorted(di_euclid.out_artifacts) == ["centroids"],
    )

    pl_kwargs = {**mix_kwargs, "centroids": tk.Path("/init/split.npy")}
    pl_flavor = per_label_mixture_flavor(**pl_kwargs)
    _check(
        "the per-label flavor is the same wiring with a different model class",
        pl_flavor.model.cls is not mix_flavor.model.cls
        and pl_flavor.artifact_names == mix_flavor.artifact_names
        and pl_flavor.accumulator == mix_flavor.accumulator
        and pl_flavor.recognizer == mix_flavor.recognizer,
    )
    pl_ten = chunked_clustering(num_epochs=4, flavor=per_label_mixture_flavor(**pl_kwargs), **common)
    _check(
        "swapping the mixture layout changes the jobs",
        [j._sis_id() for j in pl_ten.jobs] != [j._sis_id() for j in mix_ten.jobs[:4]],
    )

    # Epoch 0 has to be reachable as a model directory, or the one model whose
    # quality is known in advance is the one that cannot be decoded.
    _check(
        "epoch 0 is exposed as a model directory like every other epoch",
        0 in mix_ten.out_models
        and sorted(mix_ten.out_models) == list(range(11))
        and mix_ten.out_models[0].get_path().endswith("/model"),
    )
    _check(
        "the initial model directory is built by MaterializeModelJob",
        type(mix_ten.out_models[0].creator).__name__ == "MaterializeModelJob",
        type(mix_ten.out_models[0].creator).__name__,
    )
    _check(
        "continuing a run does not change what epoch 0 materializes",
        mix_ten.out_models[0].creator._sis_id() == mix_head.out_models[0].creator._sis_id(),
    )
    # Runs differing only in a scheduling knob share one initial model.
    other_chunks = chunked_clustering(
        num_epochs=2, flavor=mixture_flavor(**mix_kwargs), num_chunks=7, **common
    )
    _check(
        "runs sharing an initialization share the job that materializes it",
        other_chunks.out_models[0].creator._sis_id()
        == mix_ten.out_models[0].creator._sis_id(),
    )

    with tempfile.TemporaryDirectory() as tmp:
        # The job's actual work, on a model small enough to build here: loose
        # artifact files in, a directory load_model() can dispatch on out.
        from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
            MaterializeModelJob,
        )

        paths = {}
        for name, array in (
            ("centroids", pl_means),
            ("covs", pl_covs),
            ("mixtures", pl_weights),
        ):
            paths[name] = tk.Path(os.path.join(tmp, f"{name}.npy"))
            np.save(paths[name].get_path(), array)
        job = MaterializeModelJob(Spec(PerLabelMixtureModel, paths))
        directory = os.path.join(tmp, "materialized")
        job.out_model = tk.Path(directory)
        job.run()
        restored = load_model(directory)
        _check(
            "MaterializeModelJob writes a directory load_model can dispatch on",
            type(restored) is PerLabelMixtureModel
            and np.allclose(restored.scores(mix_feats), pl.scores(mix_feats)),
        )
        _check(
            "it is not a mini-task - building a covariance model inverts every cov",
            bool(job.rqmt) and [t.rqmt() for t in job.tasks()] == [job.rqmt],
        )

    try:
        ClusteringFlavor(
            model=Spec(GaussianModel, {"centroids": tk.Path("/c.npy")}),
            accumulator=mix_flavor.accumulator,
            recognizer=mix_flavor.recognizer,
        )
        caught = False
    except ValueError:
        caught = True
    _check("a flavor missing a starting artifact is refused at construction", caught)

    from sisyphus.hash import sis_hash_helper

    print("unguided clustering — ArgmaxRecognizer + FixedCovarianceAccumulator")
    # The unguided pass has no search, so the properties that matter are the
    # ones the search used to provide: that assignment is a pure per-frame
    # argmin, and that the covariance it scores under does not drift.
    u_rng = np.random.RandomState(20260827)
    u_dim, u_k = 6, 4
    u_covs = np.stack([np.eye(u_dim) * (i + 1.0) for i in range(u_k)])
    u_prev = GaussianModel(u_rng.randn(u_k, u_dim), u_covs)
    u_feats = u_rng.randn(200, u_dim)
    u_labels = u_rng.randint(0, u_k, size=200)

    delivered = []
    recog = ArgmaxRecognizer(num_clusters=u_k)
    recog.start(delivered.append)
    u_scores = u_rng.rand(11, u_k)
    recog.submit("seq0", u_scores)
    recog.submit("seq1", u_scores * 17.0)
    # While still started: submitting after shutdown() is a different failure
    # (the callback is gone), and would mask the width check.
    _check(
        "a score matrix of the wrong width is refused, not silently assigned",
        _raises(lambda: recog.submit("s", u_rng.rand(4, u_k + 1)), ValueError),
    )
    recog.drain()
    recog.shutdown()
    _check(
        "argmax assignment is the per-frame argmin of the score matrix",
        np.array_equal(delivered[0].posteriors, u_scores.argmin(axis=1)),
    )
    _check(
        "it reports no traceback - there is no discrete path to report",
        delivered[0].traceback == [] and delivered[0].sequence_score is None,
    )
    # Why the flavor exposes no distance_scale: it provably cannot matter here.
    _check(
        "assignment is invariant under a positive rescaling of the scores",
        np.array_equal(delivered[0].posteriors, delivered[1].posteriors),
    )

    def _fixed_acc(groups):
        built = []
        for g in groups:
            a = FixedCovarianceAccumulator(num_clusters=u_k)
            a.bind_model(u_prev)
            a.observe(u_feats[g], u_labels[g])
            built.append(a)
        merged = built[0]
        for a in built[1:]:
            merged = merged.merge(a)
        return merged

    whole = _fixed_acc([np.arange(200)])
    split = _fixed_acc([np.arange(0, 70), np.arange(70, 130), np.arange(130, 200)])
    m_whole, m_split = whole.finalize(u_prev), split.finalize(u_prev)
    _check(
        "FixedCovarianceAccumulator merge is associative",
        np.allclose(m_whole.centroids, m_split.centroids, atol=1e-12),
    )
    _check(
        "the covariance is carried over untouched, not re-estimated",
        np.array_equal(m_whole.covs, u_covs) and np.array_equal(m_split.covs, u_covs),
    )
    _check(
        "the means are the per-cluster frame means",
        all(
            np.allclose(m_whole.centroids[k], u_feats[u_labels == k].mean(0), atol=1e-12)
            for k in range(u_k)
            if (u_labels == k).any()
        ),
    )
    _check(
        "it finalizes to the model class it was given, so the spec shape survives",
        type(m_whole) is GaussianModel,
    )

    starved = FixedCovarianceAccumulator(num_clusters=u_k)
    starved.bind_model(u_prev)
    keep = u_labels != 2
    starved.observe(u_feats[keep], u_labels[keep])
    m_starved = starved.finalize(u_prev)
    _check(
        "a cluster that got no frames keeps the previous mean",
        np.allclose(m_starved.centroids[2], u_prev.centroids[2])
        and not np.allclose(m_starved.centroids[0], u_prev.centroids[0]),
    )
    # The whole reason this exists rather than SoftGaussianAccumulator: the
    # second moment is what makes a large cluster count unaffordable at D=512.
    _check(
        "the state is O(K x D) - no second moment is held",
        starved.sums.nbytes == u_k * u_dim * 8
        and not hasattr(starved, "weighted_sq"),
    )
    _check(
        "a model with no covariance to carry over is refused with a pointer",
        _raises(
            lambda: _fixed_acc([np.arange(20)]).finalize(EuclideanModel(u_prev.centroids)),
            TypeError,
        ),
    )

    u_flavor = unguided_flavor(
        centroids=tk.Path("/init/c.npy"), covs=tk.Path("/init/cov.npy"), num_clusters=u_k
    )
    _check(
        "unguided_flavor carries a covariance and records no traceback statistics",
        u_flavor.artifact_names == ("centroids", "covs") and u_flavor.statistics is None,
    )
    u_next = u_flavor.next_model(
        {"centroids": tk.Path("/e1/c.npy"), "covs": tk.Path("/e1/cov.npy")}
    )
    _check(
        "its next-epoch spec has the same shape, so continuation still reuses jobs",
        u_next.cls is u_flavor.model.cls
        and set(u_next.kwargs) == set(u_flavor.model.kwargs),
    )
    _check(
        "without covs it is plain squared-Euclidean k-means",
        unguided_flavor(centroids=tk.Path("/init/c.npy"), num_clusters=u_k).model.cls
        is EuclideanModel,
    )

    class _UnguidedSource:
        def __init__(self, num_seqs):
            self.num_seqs = num_seqs
            self.rng = np.random.RandomState(7)

        def __iter__(self):
            for i in range(self.num_seqs):
                yield f"seq{i}", self.rng.randn(self.rng.randint(5, 20), u_dim)

        def __len__(self):
            return self.num_seqs

    u_result = run_chunk(
        features=_UnguidedSource(9),
        model=u_prev,
        recognizer=ArgmaxRecognizer(num_clusters=u_k),
        accumulator=FixedCovarianceAccumulator(num_clusters=u_k),
        verbosity=0,
    )
    _check(
        "an unguided chunk runs end to end with no lexicon and no RASR",
        u_result.num_seqs == 9 and u_result.num_recognized == 9,
    )

    print("MixtureGaussianAccumulator — frozen densities")
    # Stage 3 of the codebook experiment: the partition is already decided, and
    # the only thing an epoch learns is p(density | label).
    f_rng = np.random.RandomState(4242)
    f_l, f_c, f_dim = 3, 5, 6
    f_feats = f_rng.randn(200, f_dim)
    f_prev = GaussianMixtureModel(
        f_rng.randn(f_c, f_dim),
        np.stack([np.eye(f_dim) * 2.0 for _ in range(f_c)]),
        f_rng.dirichlet(np.ones(f_c), size=f_l),
    )
    f_gammas = f_rng.rand(200, f_l)
    f_gammas /= f_gammas.sum(1, keepdims=True)

    def _mix_acc(groups, **kwargs):
        built = []
        for g in groups:
            a = MixtureGaussianAccumulator(num_clusters=f_l, **kwargs)
            a.bind_model(f_prev)
            a.observe(f_feats[g], f_gammas[g])
            built.append(a)
        merged = built[0]
        for a in built[1:]:
            merged = merged.merge(a)
        return merged

    frozen_whole = _mix_acc([np.arange(200)], update_densities=False)
    frozen_split = _mix_acc([np.arange(0, 90), np.arange(90, 200)], update_densities=False)
    f1, f2 = frozen_whole.finalize(f_prev), frozen_split.finalize(f_prev)
    _check(
        "frozen-density merge is still associative",
        np.allclose(f1.mixtures, f2.mixtures, atol=1e-12),
    )
    _check(
        "the codebook comes through untouched",
        np.array_equal(f1.centroids, f_prev.centroids)
        and np.array_equal(f1.covs, f_prev.covs),
    )
    # The memory claim in the docstring, asserted rather than asserted-in-prose:
    # a frozen codebook must not accumulate the O(D^2) statistic at all.
    _check(
        "no density statistics are accumulated, not merely discarded at finalize",
        frozen_whole.gaussian_accumulator.weighted_sq is None
        and frozen_whole.gaussian_accumulator.weighted_sums is None,
    )
    live = _mix_acc([np.arange(200)], update_densities=True)
    live_model = live.finalize(f_prev)
    _check(
        "freezing changes the densities only - the weights are the same E-step",
        np.allclose(live_model.mixtures, f1.mixtures, atol=1e-12)
        and not np.allclose(live_model.centroids, f_prev.centroids),
    )
    reloaded = MixtureGaussianAccumulator(
        num_clusters=f_l, update_densities=False
    ).load_state_dict(frozen_whole.state_dict())
    _check(
        "the flag survives the reduce step's state_dict round-trip",
        reloaded.update_densities is False
        and np.allclose(reloaded.finalize(f_prev).mixtures, f1.mixtures),
    )
    _check(
        "pooling covariances that are never re-estimated is refused",
        _raises(
            lambda: MixtureGaussianAccumulator(
                num_clusters=f_l, pool_covariances=True, update_densities=False
            ),
            ValueError,
        ),
    )

    # Hash discipline: the flag is absent from the spec at its default, which is
    # what kept every existing mixture run's jobs valid when it was added.
    _mix_common = dict(
        recognition_config=tk.Path("/x/r.config"),
        lexicon=tk.Path("/x/lex.xml"),
        num_clusters=f_l,
        centroids=tk.Path("/x/c.npy"),
        covs=tk.Path("/x/v.npy"),
        mixtures=tk.Path("/x/m.npy"),
    )
    _default = mixture_flavor(**_mix_common)
    _explicit = mixture_flavor(**_mix_common, update_densities=True)
    _frozen = mixture_flavor(**_mix_common, update_densities=False)
    _check(
        "update_densities=True leaves the accumulator spec, and so the hash, alone",
        "update_densities" not in _default.accumulator.kwargs
        and sis_hash_helper(_default.accumulator.hashed())
        == sis_hash_helper(_explicit.accumulator.hashed()),
    )
    _check(
        "freezing does change it - it is a different experiment",
        sis_hash_helper(_default.accumulator.hashed())
        != sis_hash_helper(_frozen.accumulator.hashed()),
    )

    print("MixtureGaussianAccumulator — over-relaxation")
    _relax = dict(update_densities=False, mixture_floor=1e-2)
    plain = _mix_acc([np.arange(200)], **_relax).finalize(f_prev)
    unit_acc = _mix_acc([np.arange(200)], over_relaxation=1.0, **_relax)
    _check(
        "over_relaxation=1.0 is the plain M-step, bit for bit, and reports no step",
        np.array_equal(unit_acc.finalize(f_prev).mixtures, plain.mixtures)
        and unit_acc.step_statistics is None,
    )
    relax_acc = _mix_acc([np.arange(200)], over_relaxation=2.0, **_relax)
    relaxed = relax_acc.finalize(f_prev)
    doubled = f_prev.mixtures ** -1.0 * plain.mixtures ** 2.0
    doubled /= doubled.sum(-1, keepdims=True)
    _check(
        "eta=2 is the EM step taken twice in log space: w_em**2 / w_prev, normalized",
        np.allclose(relaxed.mixtures, doubled, atol=1e-12)
        and relax_acc.step_statistics["mixture_clipped_fraction"] == 0.0,
        f"max err {np.abs(relaxed.mixtures - doubled).max():.2e}",
    )
    _check(
        "the over-relaxed step is still associative over chunks",
        np.allclose(
            _mix_acc([np.arange(0, 90), np.arange(90, 200)], over_relaxation=2.0, **_relax)
            .finalize(f_prev).mixtures,
            relaxed.mixtures, atol=1e-12,
        ),
    )
    _stats = relax_acc.step_statistics
    _check(
        "it reports the step EM proposed and the larger one it took",
        _stats["mixture_over_relaxation"] == 2.0
        and 0.0 < _stats["mixture_kl_em_step"] < _stats["mixture_kl_applied_step"],
        f"{_stats}",
    )
    # A large eta drives the weights EM is shrinking far below anything the
    # plain M-step could produce; they must stop at its floor instead.
    wild_acc = _mix_acc([np.arange(200)], over_relaxation=25.0, **_relax)
    wild = wild_acc.finalize(f_prev).mixtures
    _lower = 1e-2 / (wild_acc.weighted_c.sum(-1) + f_c * 1e-2)
    _check(
        "an extreme step stays normalized, finite and above the plain M-step's floor",
        np.isfinite(wild).all()
        and np.allclose(wild.sum(-1), 1.0)
        and (wild >= _lower[:, None] * (1 - 1e-12)).all()
        and wild_acc.step_statistics["mixture_clipped_fraction"] > 0.0,
    )
    dead_gammas = f_gammas.copy()
    dead_gammas[:, 1] = 0.0
    dead_gammas /= dead_gammas.sum(1, keepdims=True)
    dead_acc = MixtureGaussianAccumulator(num_clusters=f_l, over_relaxation=2.0, **_relax)
    dead_acc.bind_model(f_prev)
    dead_acc.observe(f_feats, dead_gammas)
    _check(
        "a label with no mass keeps its previous weights under over-relaxation too",
        np.array_equal(dead_acc.finalize(f_prev).mixtures[1], f_prev.mixtures[1]),
    )
    _check(
        "eta <= 0, a zero floor, or live densities are refused",
        _raises(lambda: MixtureGaussianAccumulator(
            num_clusters=f_l, over_relaxation=0.0, **_relax), ValueError)
        and _raises(lambda: MixtureGaussianAccumulator(
            num_clusters=f_l, over_relaxation=2.0, update_densities=False), ValueError)
        and _raises(lambda: MixtureGaussianAccumulator(
            num_clusters=f_l, over_relaxation=2.0, mixture_floor=1e-2), ValueError),
    )
    _frozen_relax = mixture_flavor(
        **_mix_common, update_densities=False, mixture_floor=1e-2, over_relaxation=2.0)
    _frozen_floor = mixture_flavor(**_mix_common, update_densities=False, mixture_floor=1e-2)
    _frozen_unit = mixture_flavor(
        **_mix_common, update_densities=False, mixture_floor=1e-2, over_relaxation=1.0)
    _check(
        "over_relaxation=1.0 leaves the accumulator spec, and so the hash, alone",
        "over_relaxation" not in _frozen_unit.accumulator.kwargs
        and sis_hash_helper(_frozen_unit.accumulator.hashed())
        == sis_hash_helper(_frozen_floor.accumulator.hashed()),
    )
    _check(
        "any other eta is a different experiment",
        _frozen_relax.accumulator.kwargs["over_relaxation"] == 2.0
        and sis_hash_helper(_frozen_relax.accumulator.hashed())
        != sis_hash_helper(_frozen_floor.accumulator.hashed()),
    )


    print("initialization jobs")
    from i6_experiments.example_setups.guided_kmeans.setup.array_job import ArrayJob
    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        DuplicateCovsJob,
        IdentityCovsJob,
        RandomMixturesJob,
        RepeatCovsJob,
        SplitCentroidsJob,
        UniformMixturesJob,
    )

    _check(
        "every output name has a matching out_ attribute and .npy file",
        all(
            all(
                getattr(j, f"out_{n}").get_path().endswith(f"{n}.npy") for n in j.OUTPUTS
            )
            for j in (
                UniformMixturesJob(4, 3),
                IdentityCovsJob(4, feature_dim=2),
                SplitCentroidsJob(tk.Path("/c.npy"), 2),
            )
        ),
    )
    # compute() is the part with logic in it; run() is the base class's.
    _check(
        "uniform mixtures are normalized",
        np.allclose(UniformMixturesJob(5, 4).compute().sum(-1), 1.0),
    )
    rnd_mix = RandomMixturesJob(5, 4, seed=1).compute()
    _check(
        "random mixtures are normalized, positive and differ per label",
        np.allclose(rnd_mix.sum(-1), 1.0)
        and (rnd_mix > 0).all()
        and not np.allclose(rnd_mix, rnd_mix[:1]),
    )
    _check(
        "random mixtures are reproducible from the seed",
        np.array_equal(rnd_mix, RandomMixturesJob(5, 4, seed=1).compute())
        and not np.array_equal(rnd_mix, RandomMixturesJob(5, 4, seed=2).compute()),
    )

    with tempfile.TemporaryDirectory() as tmp:
        one_cov = np.array([[4.0, 1.0], [1.0, 9.0]])
        cov_path = tk.Path(os.path.join(tmp, "cov.npy"))
        np.save(cov_path.get_path(), one_cov)
        dup = DuplicateCovsJob(cov_path, 6).compute()
        _check(
            "DuplicateCovsJob tiles one matrix",
            dup.shape == (6, 2, 2) and all(np.array_equal(c, one_cov) for c in dup),
        )
        wrapped = tk.Path(os.path.join(tmp, "cov1.npy"))
        np.save(wrapped.get_path(), one_cov[None])
        _check(
            "... and accepts the [1, D, D] spelling of the same thing",
            np.array_equal(DuplicateCovsJob(wrapped, 6).compute(), dup),
        )
        stack = tk.Path(os.path.join(tmp, "covs.npy"))
        np.save(stack.get_path(), np.stack([one_cov * (i + 1) for i in range(3)]))
        try:
            DuplicateCovsJob(stack, 6).compute()
            caught = False
        except ValueError:
            caught = True
        _check("DuplicateCovsJob refuses a stack and names the job that takes one", caught)

        rep = RepeatCovsJob(stack, 2).compute()
        _check(
            "RepeatCovsJob keeps each matrix's copies adjacent",
            rep.shape == (6, 2, 2)
            and np.array_equal(rep[0], rep[1])
            and np.array_equal(rep[2], rep[3])
            and not np.array_equal(rep[1], rep[2]),
        )
        try:
            RepeatCovsJob(cov_path, 2).compute()
            caught = False
        except ValueError:
            caught = True
        _check("a 2-D input where a stack was meant is refused, not broadcast", caught)

        # --- covariance-driven splitting -----------------------------------
        seed_c = np.array([[0.0, 0.0], [10.0, 10.0], [-5.0, 2.0]])
        c_path = tk.Path(os.path.join(tmp, "seed.npy"))
        np.save(c_path.get_path(), seed_c)
        # Deliberately anisotropic: all the variance on axis 0, so the
        # principal direction is known exactly.
        aniso = np.stack([np.diag([25.0, 0.01])] * 3)
        a_path = tk.Path(os.path.join(tmp, "aniso.npy"))
        np.save(a_path.get_path(), aniso)

        n = 3
        by_cov = SplitCentroidsJob(c_path, n, perturbation=0.2, covs=a_path).compute()
        _check(
            "cov split keeps the layout: L * n copies, label l's adjacent at l * n",
            by_cov.shape == (len(seed_c) * n, 2),
        )
        disp = by_cov - np.repeat(seed_c, n, axis=0)
        _check(
            "copies are displaced along the principal axis only",
            np.abs(disp[:, 1]).max() < 1e-9 and np.abs(disp[:, 0]).max() > 0,
            f"off-axis {np.abs(disp[:, 1]).max():.1e}, on-axis {np.abs(disp[:, 0]).max():.3f}",
        )
        _check(
            "displacement is +/- perturbation * principal sigma",
            np.allclose(sorted(disp[:3, 0]), [-0.2 * 5.0, 0.0, 0.2 * 5.0]),
            str(sorted(np.round(disp[:3, 0], 6))),
        )
        _check(
            "the split is symmetric about the centroid it came from",
            np.allclose(by_cov.reshape(len(seed_c), n, 2).mean(axis=1), seed_c),
        )
        _check(
            "cov splitting is deterministic - the seed is unused",
            np.array_equal(
                by_cov, SplitCentroidsJob(c_path, n, perturbation=0.2, seed=7, covs=a_path).compute()
            ),
        )
        shared_cov = tk.Path(os.path.join(tmp, "shared.npy"))
        np.save(shared_cov.get_path(), np.diag([25.0, 0.01]))
        _check(
            "a single [D, D] covariance is shared across all centroids",
            np.array_equal(
                SplitCentroidsJob(c_path, n, perturbation=0.2, covs=shared_cov).compute(), by_cov
            ),
        )
        short_stack = tk.Path(os.path.join(tmp, "short.npy"))
        np.save(short_stack.get_path(), aniso[:2])   # 2 covariances, 3 centroids
        _check(
            "a covariance count that does not match the centroids is refused",
            _raises(lambda: SplitCentroidsJob(c_path, n, covs=short_stack).compute(), ValueError),
        )
        wrong_dim = tk.Path(os.path.join(tmp, "wrongdim.npy"))
        np.save(wrong_dim.get_path(), np.stack([np.eye(5)] * 3))
        _check(
            "a covariance whose dimension does not match the centroids is refused",
            _raises(lambda: SplitCentroidsJob(c_path, n, covs=wrong_dim).compute(), ValueError),
        )

        jitter = SplitCentroidsJob(c_path, n).compute()
        _check(
            "without covs the copies are jittered, all of them distinct",
            jitter.shape == by_cov.shape
            and all(
                not np.array_equal(jitter[l * n + k], seed_c[l])
                for l in range(len(seed_c))
                for k in range(n)
            ),
        )
        _check(
            "a single density per label is a no-op split",
            np.allclose(SplitCentroidsJob(c_path, 1, covs=a_path).compute(), seed_c),
        )

        # The two halves of a per-label init have to line up density for density.
        split_covs = RepeatCovsJob(a_path, n).compute()
        _check(
            "SplitCentroidsJob and RepeatCovsJob agree on the density layout",
            len(split_covs) == len(by_cov)
            and PerLabelMixtureModel(
                by_cov, split_covs, UniformMixturesJob(len(seed_c), n).compute()
            ).num_densities == len(seed_c) * n,
        )

    class _MislabelledJob(ArrayJob):
        """compute() returning a name OUTPUTS does not declare."""

        OUTPUTS = ("centroids",)

        def compute(self):
            return {"covs": np.zeros((2, 2))}

    class _TooManyOutputsJob(ArrayJob):
        """Two outputs but a bare array from compute()."""

        OUTPUTS = ("centroids", "covs")

        def compute(self):
            return np.zeros((2, 2))

    # GlobalCovarianceJob reads a real HDF, so it gets a real one.
    import h5py
    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        GlobalCovarianceJob,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cov_dim, cov_seqs = 12, 40
        seq_lengths = mix_rng.randint(20, 60, size=cov_seqs)
        _m = mix_rng.randn(cov_dim, cov_dim)
        frames = mix_rng.multivariate_normal(
            mix_rng.randn(cov_dim) * 5, _m @ _m.T + np.eye(cov_dim), size=int(seq_lengths.sum())
        )
        hdf_path = os.path.join(tmp, "feat.hdf")
        with h5py.File(hdf_path, "w") as fp:
            fp["inputs"] = frames.astype(np.float32)
            fp["seqLengths"] = seq_lengths
            fp["seqTags"] = np.array([f"seq{i:03d}".encode() for i in range(cov_seqs)])
        stored = frames.astype(np.float32).astype(np.float64)

        gcov = GlobalCovarianceJob(tk.Path(hdf_path))
        out = gcov.compute()
        _check(
            "global covariance and mean match numpy over the same frames",
            np.allclose(out["mean"], stored.mean(axis=0), atol=1e-10)
            and np.allclose(out["cov"], np.cov(stored.T, bias=True), atol=1e-9),
            f"max err {np.abs(out['cov'] - np.cov(stored.T, bias=True)).max():.2e}",
        )
        _check(
            "the covariance comes out symmetric, not just nearly so",
            np.array_equal(out["cov"], out["cov"].T),
        )
        _check(
            "it declares real requirements rather than running as a mini-task",
            bool(gcov.rqmt) and [t.rqmt() for t in gcov.tasks()] == [gcov.rqmt],
        )

        # Segment filtering has to select the same frames the clustering would.
        seg_path = os.path.join(tmp, "seg.txt")
        keep = list(range(0, cov_seqs, 3))
        with open(seg_path, "w") as fp:
            fp.write("\n".join(f"seq{i:03d}" for i in keep) + "\n")
        offsets = np.concatenate([[0], np.cumsum(seq_lengths)])
        rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in keep])
        filtered = GlobalCovarianceJob(tk.Path(hdf_path), segments=tk.Path(seg_path)).compute()
        _check(
            "segment filtering selects exactly the segments' frames",
            np.allclose(filtered["cov"], np.cov(stored[rows].T, bias=True), atol=1e-9),
            f"max err {np.abs(filtered['cov'] - np.cov(stored[rows].T, bias=True)).max():.2e}",
        )

        # out_cov is a single [D, D], which is what DuplicateCovsJob takes -
        # the composition the config relies on for a self-contained init.
        gcov_path = tk.Path(os.path.join(tmp, "gcov.npy"))
        np.save(gcov_path.get_path(), out["cov"])
        _check(
            "out_cov feeds DuplicateCovsJob directly",
            DuplicateCovsJob(gcov_path, 5).compute().shape == (5, cov_dim, cov_dim),
        )


        # --- ClusterCovarianceJob: covariances over a partition it does not move
        import json

        from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
            ClusterCovarianceJob,
        )

        cc_centroids = stored[mix_rng.choice(len(stored), 4, replace=False)]
        cc_cpath = os.path.join(tmp, "cc_centroids.npy")
        np.save(cc_cpath, cc_centroids)
        cc_shared = np.cov(stored.T, bias=True)
        cc_vpath = os.path.join(tmp, "cc_covs.npy")
        np.save(cc_vpath, np.repeat(cc_shared[np.newaxis], 4, axis=0))

        def _cluster_covs(**kwargs):
            job = ClusterCovarianceJob(tk.Path(hdf_path), tk.Path(cc_cpath), **kwargs)
            out = tempfile.mkdtemp(dir=tmp)
            job.out_covs = tk.Path(os.path.join(out, "covs.npy"))
            job.out_counts = tk.Path(os.path.join(out, "counts.npy"))
            job.out_diagnostics = tk.Path(os.path.join(out, "diagnostics.json"))
            job.run()
            with open(job.out_diagnostics.get_path()) as fp:
                return (
                    np.load(job.out_covs.get_path()),
                    np.load(job.out_counts.get_path()),
                    json.load(fp),
                )

        # The partition is an input, so it has to be exactly the one an unguided
        # epoch with the same centroids and covariance would have produced.
        cc_labels = (
            GaussianModel(cc_centroids, np.repeat(cc_shared[np.newaxis], 4, axis=0), device="cpu")
            .scores(stored)
            .argmin(axis=1)
        )
        full_covs, full_counts, full_diag = _cluster_covs(
            assignment_covs=tk.Path(cc_vpath), structure="full", min_frames=1
        )
        _check(
            "the partition is reproduced, not recomputed differently",
            np.array_equal(full_counts, np.bincount(cc_labels, minlength=4)),
        )
        _check(
            "structure='full' matches numpy's covariance over each cluster's frames",
            all(
                np.allclose(full_covs[k], np.cov(stored[cc_labels == k].T, bias=True), atol=1e-8)
                for k in range(4)
                if full_counts[k] > cov_dim
            ),
        )

        diag_covs, _, diag_diag = _cluster_covs(
            assignment_covs=tk.Path(cc_vpath), structure="diagonal", min_frames=1
        )
        cc_chol = np.linalg.cholesky(cc_shared)
        _check(
            "structure='diagonal' is exactly diagonal in the whitened space",
            all(
                np.abs(
                    (lambda b: b - np.diag(np.diag(b)))(
                        np.linalg.solve(cc_chol, np.linalg.solve(cc_chol, diag_covs[k]).T).T
                    )
                ).max()
                < 1e-8 * max(1.0, np.abs(diag_covs[k]).max())
                for k in range(4)
            ),
        )
        # The whole reason for the structure: D free parameters instead of
        # D(D+1)/2, which is a factor of 256 at D=512.
        _check(
            "it declares the parameter saving it buys",
            diag_diag["free_parameters_per_cluster"] == cov_dim
            and full_diag["free_parameters_per_cluster"] == cov_dim * (cov_dim + 1) // 2,
            f"{full_diag['free_parameters_per_cluster']} -> "
            f"{diag_diag['free_parameters_per_cluster']}",
        )

        shared_covs, _, _ = _cluster_covs(
            assignment_covs=tk.Path(cc_vpath), structure="shared", min_frames=1
        )
        _check(
            "structure='shared' gives every cluster one pooled covariance",
            np.allclose(shared_covs, shared_covs[0][np.newaxis]),
        )

        # The knob the full-covariance negative result is swept along: it has to
        # actually move the conditioning, or the sweep says nothing.
        _, _, no_ridge = _cluster_covs(
            assignment_covs=tk.Path(cc_vpath), structure="full", min_frames=1, ridge=0.0
        )
        _, _, with_ridge = _cluster_covs(
            assignment_covs=tk.Path(cc_vpath), structure="full", min_frames=1, ridge=1e-2
        )
        _check(
            "a ridge lifts the smallest eigenvalue and lowers the condition number",
            with_ridge["condition_number"]["max"] < no_ridge["condition_number"]["max"]
            and with_ridge["smallest_eigenvalue"]["min"]
            > no_ridge["smallest_eigenvalue"]["min"],
            f"condition {no_ridge['condition_number']['max']:.3g} -> "
            f"{with_ridge['condition_number']['max']:.3g}",
        )

        starved_covs, _, starved_diag = _cluster_covs(
            assignment_covs=tk.Path(cc_vpath), structure="full", min_frames=10 ** 9
        )
        _check(
            "a cluster below min_frames takes the pooled covariance, not a rank-poor one",
            starved_diag["num_starved"] == 4
            and np.allclose(starved_covs, starved_covs[0][np.newaxis]),
        )

        cc_varying = os.path.join(tmp, "cc_varying.npy")
        np.save(cc_varying, np.stack([cc_shared * (1.0 + 0.1 * i) for i in range(4)]))
        _check(
            "structure='diagonal' refuses a per-cluster transform",
            _raises(
                lambda: _cluster_covs(
                    assignment_covs=tk.Path(cc_varying), structure="diagonal", min_frames=1
                ),
                ValueError,
            ),
        )
        _check(
            "an unknown structure is refused at construction",
            _raises(
                lambda: ClusterCovarianceJob(
                    tk.Path(hdf_path), tk.Path(cc_cpath), structure="tied"
                ),
                ValueError,
            ),
        )

        # Collinear dimensions make a singular covariance; it has to fail here
        # rather than inside np.linalg.inv two jobs later.
        flat_path = os.path.join(tmp, "flat.hdf")
        collinear = mix_rng.randn(30, cov_dim)
        collinear[:, 3] = collinear[:, 2] * 2.0
        with h5py.File(flat_path, "w") as fp:
            fp["inputs"] = collinear.astype(np.float32)
            fp["seqLengths"] = np.array([30])
            fp["seqTags"] = np.array([b"only"])
        _check(
            "a singular covariance is caught where it is produced",
            _raises(lambda: GlobalCovarianceJob(tk.Path(flat_path)).compute(), ValueError),
        )

    _check(
        "ArrayJob rejects a compute() that disagrees with OUTPUTS",
        _raises(lambda: _MislabelledJob().run(), ValueError)
        and _raises(lambda: _TooManyOutputsJob().run(), TypeError),
    )

    print("VectorQuantizedModel")
    vq_C = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    vq_T = np.array([[0.5, 0.5, 0.0], [0.0, 0.2, 0.8]])
    vq = VectorQuantizedModel(vq_C, vq_T)
    vq_X = np.array([[0.1, 0.0], [9.0, 0.0], [0.0, 9.0]])
    _check(
        "quantization is plain L2 argmin - the metric a k-means codebook was built under",
        np.array_equal(vq.quantize(vq_X), [0, 1, 2]),
    )
    vq_S = vq.scores(vq_X)
    _check(
        "scoring is a table gather: -log p(q | l), one column per frame",
        vq_S.shape == (3, 2)
        and np.isclose(vq_S[0, 0], -np.log(0.5))
        and np.isclose(vq_S[2, 1], -np.log(0.8)),
    )
    _check(
        "a forbidden (label, codeword) pair is +inf, not a small number",
        np.isinf(vq_S[0, 1]) and np.isinf(vq_S[2, 0]),
    )
    _check(
        "the counts are label x codeword, and the two axes are told apart",
        vq.num_clusters == 2 and vq.num_labels == 2 and vq.num_codewords == 3 and vq.dim == 2,
    )
    # A codeword no label admits leaves a frame with no viable label at all -
    # an unsearchable lattice rather than a bad one, so it is reported.
    _orphan = VectorQuantizedModel(vq_C, np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]))
    _check(
        "a codeword no label can emit is reported where it is produced",
        _raises(lambda: _orphan.scores(np.array([[0.0, 9.0]])), FloatingPointError),
    )
    _check(
        "an unnormalized table and a wrong-width table are both refused",
        _raises(lambda: VectorQuantizedModel(vq_C, np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.6]])), ValueError)
        and _raises(lambda: VectorQuantizedModel(vq_C, np.array([[1.0, 0.0], [0.0, 1.0]])), ValueError)
        and _raises(lambda: VectorQuantizedModel(vq_C, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])), ValueError),
    )
    with tempfile.TemporaryDirectory() as vq_dir:
        vq.save(vq_dir)
        vq_back = load_model(vq_dir)
        _check(
            "it round-trips through the manifest, so a decode needs no VQ-specific code",
            type(vq_back) is VectorQuantizedModel
            and np.allclose(vq_back.scores(vq_X), vq_S, equal_nan=True)
            and np.allclose(load_forward_model(vq_dir).forward(vq_X), vq_S, equal_nan=True),
        )

    print("supervised VQ baseline jobs")
    import h5py
    from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
        SegmentedFeaturesFromAlignmentJob,
        SupervisedVQTableJob,
    )

    with tempfile.TemporaryDirectory() as vq_tmp:
        # Two utterances. The first is the case that decides whether silence is
        # dropped before or after segmentation: label 1, silence, label 1 again.
        # Dropping first would merge the two 1s into a single segment and lose a
        # realization; segmenting first keeps them apart.
        vq_align = {
            "rec/a": np.array([1, 1, 0, 0, 1, 2, 2]),
            "rec/b": np.array([2, 3, 3, 0]),
        }
        vq_feats = np.arange(11 * 2, dtype=np.float64).reshape(11, 2)
        vq_hdf = os.path.join(vq_tmp, "raw.hdf")
        with h5py.File(vq_hdf, "w") as fp:
            fp["inputs"] = vq_feats.astype(np.float32)
            fp["seqLengths"] = np.array([[7, 0], [4, 0]], dtype=np.int32)
            fp.create_dataset(
                "seqTags", data=[b"corpus/rec/a", b"corpus/rec/b"],
                dtype=h5py.special_dtype(vlen=bytes),
            )
        vq_ali = os.path.join(vq_tmp, "ali.pkl")
        with open(vq_ali, "wb") as fp:
            pickle.dump(vq_align, fp)

        def _run_features(alignment_override=None, **kwargs):
            job = SegmentedFeaturesFromAlignmentJob(
                tk.Path(vq_hdf), alignment_override or tk.Path(vq_ali), **kwargs
            )
            out = tempfile.mkdtemp(dir=vq_tmp)
            job.out_features = tk.Path(os.path.join(out, "f.hdf"))
            job.out_labels = tk.Path(os.path.join(out, "l.pkl"))
            job.out_segments = tk.Path(os.path.join(out, "s.txt"))
            job.out_statistics = tk.Path(os.path.join(out, "st.json"))
            job.run()
            with open(job.out_labels.get_path(), "rb") as fp:
                labels = pickle.load(fp)
            with h5py.File(job.out_features.get_path(), "r") as fp:
                feats = np.asarray(fp["inputs"][:], dtype=np.float64)
                lens = fp["seqLengths"][:][:, 0].tolist()
            return job, labels, feats, lens

        job_all, labels_all, feats_all, lens_all = _run_features(exclude_labels=())
        _check(
            "with nothing excluded the segmentation is the alignment's own runs",
            lens_all == [4, 3] and labels_all["corpus/rec/a"].tolist() == [1, 0, 1, 2],
        )
        job_sil, labels_sil, feats_sil, lens_sil = _run_features(exclude_labels=(0,))
        _check(
            "silence is dropped after segmenting, so a phoneme split by a pause stays two segments",
            labels_sil["corpus/rec/a"].tolist() == [1, 1, 2] and lens_sil == [3, 2],
            f"got {labels_sil['corpus/rec/a'].tolist()}",
        )
        _check(
            "mean pooling averages each run - the pooling the existing segmented files use",
            np.allclose(feats_sil[0], vq_feats[0:2].mean(0))
            and np.allclose(feats_sil[1], vq_feats[4:5].mean(0)),
        )
        _, _, feats_max, _ = _run_features(exclude_labels=(0,), pooling="max")
        _check(
            "max pooling is available and is a different answer",
            np.allclose(feats_max[0], vq_feats[0:2].max(0))
            and not np.allclose(feats_max[0], feats_sil[0]),
        )
        # A segmented feature file passed where an unsegmented one belongs is the
        # easy mistake here, and it shows up as a length disagreement.
        vq_bad_ali = os.path.join(vq_tmp, "bad_ali.pkl")
        with open(vq_bad_ali, "wb") as fp:
            pickle.dump({"rec/a": np.array([1, 1]), "rec/b": np.array([2, 3, 3, 0])}, fp)

        def _run_bad():
            job = SegmentedFeaturesFromAlignmentJob(tk.Path(vq_hdf), tk.Path(vq_bad_ali))
            out = tempfile.mkdtemp(dir=vq_tmp)
            job.out_features = tk.Path(os.path.join(out, "f.hdf"))
            job.out_labels = tk.Path(os.path.join(out, "l.pkl"))
            job.out_segments = tk.Path(os.path.join(out, "s.txt"))
            job.out_statistics = tk.Path(os.path.join(out, "st.json"))
            job.run()

        _check(
            "an alignment that is not frame-synchronous with the features is refused",
            _raises(_run_bad, ValueError),
        )


        # Unsegmented: the alignment removes silence frames but collapses nothing.
        _, labels_fr, feats_fr, lens_fr = _run_features(exclude_labels=(0,), pooling="none")
        _keep_a = np.array([0, 1, 4, 5, 6])          # rec/a minus its two silence frames
        _keep_b = np.array([7, 8, 9])                # rec/b minus its trailing silence frame
        _check(
            "pooling='none' writes exactly the non-silence frames, unchanged",
            lens_fr == [5, 3]
            and np.allclose(feats_fr, vq_feats[np.concatenate([_keep_a, _keep_b])]),
        )
        _check(
            "its labels are one per frame - the silence-free frame alignment",
            labels_fr["corpus/rec/a"].tolist() == [1, 1, 1, 2, 2]
            and labels_fr["corpus/rec/b"].tolist() == [2, 3, 3],
        )
        # The documented consequence of dropping frames: AA SIL AA arrives as one
        # uninterrupted run, which a label-collapsing search reads as one phoneme.
        _check(
            "a phoneme split by a removed pause becomes one contiguous run of frames",
            labels_fr["corpus/rec/a"][:3].tolist() == [1, 1, 1],
        )
        _check(
            "an unknown pooling mode is refused",
            _raises(lambda: SegmentedFeaturesFromAlignmentJob(
                tk.Path(vq_hdf), tk.Path(vq_ali), pooling="median"), ValueError),
        )

        # --- alignments arrive in two storage forms and must agree
        from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
            _load_alignment,
            _strip_corpus_prefix,
        )

        _check(
            "the corpus prefix is stripped, so train-clean-100 and train-other-960 join",
            _strip_corpus_prefix("train-clean-100/a-b-c/a-b-c") == "a-b-c/a-b-c"
            and _strip_corpus_prefix("train-other-960/a-b-c/a-b-c") == "a-b-c/a-b-c"
            and _strip_corpus_prefix("a-b-c/a-b-c") == "a-b-c/a-b-c",
        )
        # The same alignment written both ways has to load identically.
        vq_ali_hdf = os.path.join(vq_tmp, "ali.hdf")
        _flat = np.concatenate([vq_align["rec/a"], vq_align["rec/b"]]).astype(np.int32)
        with h5py.File(vq_ali_hdf, "w") as fp:
            fp["inputs"] = _flat
            fp["seqLengths"] = np.array([[7], [4]], dtype=np.int32)
            fp.create_dataset(
                "seqTags",
                data=[b"train-other-960/rec/a", b"train-other-960/rec/b"],
                dtype=h5py.special_dtype(vlen=bytes),
            )
        _from_pkl = _load_alignment(tk.Path(vq_ali))
        _from_hdf = _load_alignment(tk.Path(vq_ali_hdf))
        _check(
            "a pickle and an HDF shard load to the same thing, prefix aside",
            set(_from_pkl) == set(_from_hdf)
            and all(np.array_equal(_from_pkl[k], _from_hdf[k]) for k in _from_pkl),
        )
        _check(
            "wanted= bounds the load - a 960h alignment is ~140M frames",
            set(_load_alignment(tk.Path(vq_ali_hdf), wanted={"rec/a"})) == {"rec/a"}
            and _load_alignment(tk.Path(vq_ali_hdf), wanted={"nope"}) == {},
        )
        # Stripping the prefix could collide if two corpora carried one recording.
        vq_dup = os.path.join(vq_tmp, "dup.hdf")
        with h5py.File(vq_dup, "w") as fp:
            fp["inputs"] = vq_align["rec/a"].astype(np.int32)
            fp["seqLengths"] = np.array([[7]], dtype=np.int32)
            fp.create_dataset(
                "seqTags", data=[b"train-clean-100/rec/a"],
                dtype=h5py.special_dtype(vlen=bytes),
            )
        _check(
            "the same sequence arriving under two corpus names is refused, not silently kept",
            _raises(
                lambda: _load_alignment([tk.Path(vq_ali_hdf), tk.Path(vq_dup)]), ValueError
            ),
        )
        # And the job takes either form for the same result.
        _, labels_from_hdf, feats_from_hdf, _ = _run_features(
            exclude_labels=(0,), alignment_override=tk.Path(vq_ali_hdf)
        )
        _check(
            "the features job produces the same output from either alignment form",
            np.allclose(feats_from_hdf, feats_sil)
            and labels_from_hdf["corpus/rec/a"].tolist()
            == labels_sil["corpus/rec/a"].tolist(),
        )

        # Counting: the table is exactly the normalized co-occurrence.
        vq_cent = os.path.join(vq_tmp, "cent.npy")
        np.save(vq_cent, np.array([[0.0, 1.0], [8.0, 9.0], [18.0, 19.0]]))
        table_job = SupervisedVQTableJob(
            job_sil.out_features, job_sil.out_labels, tk.Path(vq_cent),
            num_labels=4, table_floor=0.0, heldout_fraction=0.0,
        )
        out = tempfile.mkdtemp(dir=vq_tmp)
        table_job.out_model = tk.Path(os.path.join(out, "model"))
        os.makedirs(table_job.out_model.get_path(), exist_ok=True)
        table_job.out_table = tk.Path(os.path.join(out, "t.npy"))
        table_job.out_counts = tk.Path(os.path.join(out, "c.npy"))
        table_job.out_diagnostics = tk.Path(os.path.join(out, "d.json"))
        table_job.out_accuracy = tk.Variable(os.path.join(out, "acc"))
        table_job.out_heldout_segments = tk.Path(os.path.join(out, "ho.txt"))
        table_job.out_train_segments = tk.Path(os.path.join(out, "tr.txt"))
        table_job.run()
        vq_counts = np.load(table_job.out_counts.get_path())
        vq_table = np.load(table_job.out_table.get_path())
        _check(
            "counts are the label x codeword co-occurrence over all observations",
            vq_counts.sum() == sum(lens_sil),
        )
        _check(
            "rows normalize to a distribution, and a label never seen gets a uniform row",
            np.allclose(vq_table.sum(1), 1.0) and np.allclose(vq_table[0], 1 / vq_counts.shape[1]),
        )
        _check(
            "the model directory it writes loads back as a VectorQuantizedModel",
            type(load_model(table_job.out_model.get_path())) is VectorQuantizedModel,
        )
        with open(table_job.out_train_segments.get_path()) as fp:
            train_tags = [line for line in fp.read().split("\n") if line]
        _check(
            "at heldout_fraction=0 everything is training data and the split files say so",
            len(train_tags) == 2 and os.path.getsize(table_job.out_heldout_segments.get_path()) == 0,
        )
        # Without a floor a counted table is mostly zeros, every zero is +inf,
        # and a codeword no label admits makes a frame unsearchable.
        floored_job = SupervisedVQTableJob(
            job_sil.out_features, job_sil.out_labels, tk.Path(vq_cent),
            num_labels=4, table_floor=1e-2, heldout_fraction=0.0,
        )
        out = tempfile.mkdtemp(dir=vq_tmp)
        floored_job.out_model = tk.Path(os.path.join(out, "model"))
        os.makedirs(floored_job.out_model.get_path(), exist_ok=True)
        floored_job.out_table = tk.Path(os.path.join(out, "t.npy"))
        floored_job.out_counts = tk.Path(os.path.join(out, "c.npy"))
        floored_job.out_diagnostics = tk.Path(os.path.join(out, "d.json"))
        floored_job.out_accuracy = tk.Variable(os.path.join(out, "acc"))
        floored_job.out_heldout_segments = tk.Path(os.path.join(out, "ho.txt"))
        floored_job.out_train_segments = tk.Path(os.path.join(out, "tr.txt"))
        floored_job.run()
        floored = np.load(floored_job.out_table.get_path())
        _check(
            "a floor puts mass on every codeword, so no score comes out +inf",
            (floored > 0).all()
            and (np.load(table_job.out_table.get_path()) == 0).any()
            and np.isfinite(
                load_model(floored_job.out_model.get_path()).scores(feats_sil)
            ).all(),
        )

    print("VectorQuantizedAccumulator")
    va_C = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    va_T = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    va_model = VectorQuantizedModel(va_C, va_T)
    va_rng = np.random.RandomState(3)
    va_X = np.concatenate([
        va_rng.randn(40, 2) * 0.3,
        va_rng.randn(40, 2) * 0.3 + [10.0, 0.0],
        va_rng.randn(40, 2) * 0.3 + [0.0, 10.0],
    ])
    va_labels = va_rng.randint(0, 2, size=len(va_X))

    def _vq_acc(groups, **kwargs):
        built = []
        for g in groups:
            acc = VectorQuantizedAccumulator(num_clusters=2, **kwargs)
            acc.bind_model(va_model)
            acc.observe(va_X[g], va_labels[g])
            built.append(acc)
        merged = built[0]
        for acc in built[1:]:
            merged = merged.merge(acc)
        return merged

    va_whole = _vq_acc([np.arange(len(va_X))])
    va_split = _vq_acc([np.arange(0, 37), np.arange(37, 81), np.arange(81, len(va_X))])
    _check(
        "merge is associative - the partition cannot change the counts",
        np.array_equal(va_whole.counts, va_split.counts),
    )
    # The counts are the co-occurrence, which is checkable directly.
    va_q = va_model.quantize(va_X)
    va_expected = np.zeros((2, 3))
    np.add.at(va_expected, (va_labels, va_q), 1.0)
    _check(
        "counts are exactly N_lc = #{t : label_t = l, q(x_t) = c}",
        np.array_equal(va_whole.counts, va_expected),
    )
    _check(
        "the state is [L, C] and carries no feature-dimension arithmetic",
        va_whole.counts.shape == (2, 3),
    )
    va_next = va_whole.finalize(va_model)
    _check(
        "finalize normalizes each row and leaves the codebook untouched",
        type(va_next) is VectorQuantizedModel
        and np.allclose(va_next.table.sum(1), 1.0)
        and np.array_equal(va_next.centroids, va_C),
    )
    _check(
        "with no floor the table is the plain relative frequency",
        np.allclose(va_next.table, va_expected / va_expected.sum(1, keepdims=True)),
    )
    va_floored = _vq_acc([np.arange(len(va_X))], table_floor=1.0).finalize(va_model)
    _check(
        "a floor smooths every entry, so no score comes out +inf",
        (va_floored.table > 0).all()
        and np.isfinite(va_floored.scores(va_X)).all()
        and np.allclose(va_floored.table.sum(1), 1.0),
    )
    # Soft posteriors are the same statistic with non-integer weights, which is
    # what makes a forward-backward epoch work without any change here.
    va_soft = VectorQuantizedAccumulator(num_clusters=2)
    va_soft.bind_model(va_model)
    va_gammas = np.zeros((len(va_X), 2))
    va_gammas[np.arange(len(va_X)), va_labels] = 1.0
    va_soft.observe(va_X, va_gammas)
    _check(
        "dense posteriors give the same counts as hard labels when they are one-hot",
        np.array_equal(va_soft.counts, va_whole.counts),
    )
    va_reload = VectorQuantizedAccumulator(num_clusters=2).load_state_dict(va_whole.state_dict())
    _check(
        "state survives the reduce step's round-trip",
        np.array_equal(va_reload.counts, va_whole.counts)
        and np.allclose(va_reload.finalize(va_model).table, va_next.table),
    )
    _check(
        "a label nothing aligned to keeps its previous row rather than being zeroed",
        np.allclose(
            _vq_acc([np.flatnonzero(va_labels == 0)]).finalize(va_model).table[1], va_T[1]
        ),
    )
    _check(
        "a model that cannot quantize is refused with a pointer",
        _raises(lambda: VectorQuantizedAccumulator(num_clusters=2).bind_model(
            EuclideanModel(va_C)), TypeError),
    )

    print("NormalTableJob")
    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        NormalTableJob,
    )

    nt = NormalTableJob(4, 64, sigma=0.1, seed=7).compute()
    _check(
        "rows are normalized distributions",
        nt.shape == (4, 64) and np.allclose(nt.sum(1), 1.0) and (nt > 0).all(),
    )
    _check(
        "it is reproducible from the seed and the seed changes it",
        np.array_equal(nt, NormalTableJob(4, 64, sigma=0.1, seed=7).compute())
        and not np.array_equal(nt, NormalTableJob(4, 64, sigma=0.1, seed=8).compute()),
    )
    # sigma is what survives renormalization, so it has to move the spread and
    # therefore how far the run starts from the degenerate uniform table.
    _spreads = [
        NormalTableJob(8, 256, sigma=s, seed=1).compute().std(1).mean() for s in (0.05, 0.5)
    ]
    _check(
        "sigma controls the spread - the only symmetry break a frozen codebook has",
        _spreads[0] < _spreads[1],
        f"std {_spreads[0]:.2e} -> {_spreads[1]:.2e}",
    )
    _wide = NormalTableJob(8, 256, sigma=0.05, seed=1).compute()
    _check(
        "a small sigma really is near-uniform, which is the failure mode to avoid",
        abs(_wide.max() - _wide.min()) < 0.5 * _wide.mean(),
    )

    print("MixtureGaussianAccumulator — means-only updates")
    # The mode the tau sweep needs: densities move to where the labels put them,
    # but their shape stays an input. Both reasons are load-bearing - a full
    # covariance at D=512 gets 0.05 vectors per free parameter here, and
    # re-estimating it would pull the softness scale back to the data's own
    # within one epoch, so tau would not survive as a hyperparameter.
    mo_rng = np.random.RandomState(90210)
    mo_l, mo_c, mo_d = 3, 5, 6
    mo_prev = GaussianMixtureModel(
        mo_rng.randn(mo_c, mo_d),
        np.stack([np.eye(mo_d) * 2.0 for _ in range(mo_c)]),
        mo_rng.dirichlet(np.ones(mo_c), size=mo_l),
    )
    mo_x = mo_rng.randn(150, mo_d)
    mo_g = mo_rng.rand(150, mo_l)
    mo_g /= mo_g.sum(1, keepdims=True)

    def _mo_acc(**kwargs):
        acc = MixtureGaussianAccumulator(num_clusters=mo_l, **kwargs)
        acc.bind_model(mo_prev)
        acc.observe(mo_x, mo_g)
        return acc

    mo_both = _mo_acc()
    mo_means = _mo_acc(update_covariances=False)
    mo_frozen = _mo_acc(update_densities=False)
    m_both, m_means, m_frozen = (a.finalize(mo_prev) for a in (mo_both, mo_means, mo_frozen))
    _check(
        "means-only moves the centroids and leaves the covariances alone",
        not np.allclose(m_means.centroids, mo_prev.centroids)
        and np.array_equal(m_means.covs, mo_prev.covs),
    )
    _check(
        "its means are identical to the full update - only the shapes differ",
        np.allclose(m_means.centroids, m_both.centroids),
    )
    _check(
        "the two neighbouring modes still behave: both moves, frozen moves nothing",
        not np.allclose(m_both.covs, mo_prev.covs)
        and np.array_equal(m_frozen.centroids, mo_prev.centroids)
        and np.array_equal(m_frozen.covs, mo_prev.covs),
    )
    # The point of the mode is that the O(D^2) statistic is never formed, not
    # that it is discarded at finalize.
    _check(
        "no second moment is allocated at all in means-only mode",
        mo_means.gaussian_accumulator.weighted_sq is None
        and mo_both.gaussian_accumulator.weighted_sq is not None,
    )
    _reloaded = MixtureGaussianAccumulator(
        num_clusters=mo_l, update_covariances=False
    ).load_state_dict(mo_means.state_dict())
    _check(
        "the flag survives the reduce step's state round-trip",
        _reloaded.update_covariances is False
        and np.allclose(_reloaded.finalize(mo_prev).centroids, m_means.centroids),
    )
    _check(
        "estimating covariances about means this run is not estimating is refused",
        _raises(
            lambda: MixtureGaussianAccumulator(
                num_clusters=mo_l, update_densities=False, update_covariances=True
            ),
            ValueError,
        ),
    )
    _mo_common = dict(
        recognition_config=tk.Path("/x/r.config"), lexicon=tk.Path("/x/lex.xml"),
        num_clusters=mo_l, centroids=tk.Path("/x/c.npy"),
        covs=tk.Path("/x/v.npy"), mixtures=tk.Path("/x/m.npy"),
    )
    _check(
        "update_covariances=None is absent from the spec, so existing hashes are untouched",
        "update_covariances" not in mixture_flavor(**_mo_common).accumulator.kwargs
        and sis_hash_helper(mixture_flavor(**_mo_common).accumulator.hashed())
        == sis_hash_helper(
            mixture_flavor(**_mo_common, update_covariances=None).accumulator.hashed()
        ),
    )
    _check(
        "means-only hashes differently - it is a different experiment",
        sis_hash_helper(mixture_flavor(**_mo_common).accumulator.hashed())
        != sis_hash_helper(
            mixture_flavor(**_mo_common, update_covariances=False).accumulator.hashed()
        ),
    )

    print("ScaleCovsJob")
    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        ScaleCovsJob,
    )

    with tempfile.TemporaryDirectory() as sc_tmp:
        sc_path = os.path.join(sc_tmp, "covs.npy")
        sc_covs = np.stack([np.eye(4) * (i + 1.0) for i in range(3)])
        np.save(sc_path, sc_covs)
        _check(
            "scaling multiplies every covariance, which is the tau of the mixture",
            np.allclose(ScaleCovsJob(tk.Path(sc_path), 4.0).compute(), sc_covs * 4.0),
        )
        _check(
            "a factor of 1 is the identity, so tau=1 is the untouched covariance",
            np.array_equal(ScaleCovsJob(tk.Path(sc_path), 1.0).compute(), sc_covs),
        )
        _check(
            "a non-positive factor is refused - it is not a covariance any more",
            _raises(lambda: ScaleCovsJob(tk.Path(sc_path), 0.0).compute(), ValueError)
            and _raises(lambda: ScaleCovsJob(tk.Path(sc_path), -1.0).compute(), ValueError),
        )
        # tau is a scale on the *distance*, so it divides the Mahalanobis term:
        # this is what makes the density assignment softer rather than just
        # relabelling the scores.
        sc_model_1 = GaussianMixtureModel(
            mo_prev.centroids, mo_prev.covs, mo_prev.mixtures
        )
        sc_model_t = GaussianMixtureModel(
            mo_prev.centroids, mo_prev.covs * 8.0, mo_prev.mixtures
        )
        sc_d1 = sc_model_1.scores_gaussian(mo_x)
        sc_dt = sc_model_t.scores_gaussian(mo_x)
        gap1 = np.sort(sc_d1, 1)[:, 1] - np.sort(sc_d1, 1)[:, 0]
        gapt = np.sort(sc_dt, 1)[:, 1] - np.sort(sc_dt, 1)[:, 0]
        _check(
            "raising tau shrinks the density gap, which is what softens assignment",
            np.median(gapt) < np.median(gap1),
            f"median gap {np.median(gap1):.2f} -> {np.median(gapt):.2f} nats at tau=8",
        )

    print("epoch statistics — Viterbi and forward-backward counter sets")
    # The bug this guards: EpochStatisticsJob computes the prior distance
    # eagerly, so a counter set it did not recognize failed the whole job -
    # taking the scalar columns down with it - rather than costing it one
    # column. FBStatisticsCounter is exactly such a set.
    from i6_experiments.example_setups.guided_kmeans.setup.statistics_jobs import (
        epoch_phoneme_frequencies,
        phoneme_prior_distance,
    )

    _viterbi_stats = {"relative_phoneme_frequencies": {"AA": 0.25, "AE": 0.75}}
    _fb_stats = {
        "soft_cluster_frequencies": [0.1, 0.2, 0.7],
        "mean_log_likelihood_per_frame": -12.0,
    }
    _phonemes = ["[SILENCE]", "AA", "AE"]

    _check(
        "a Viterbi run's named distribution is read straight through",
        epoch_phoneme_frequencies(_viterbi_stats) == {"AA": 0.25, "AE": 0.75}
        and epoch_phoneme_frequencies(_viterbi_stats, _phonemes) == {"AA": 0.25, "AE": 0.75},
    )
    _check(
        "an FB run's cluster masses are named from the lexicon, in cluster order",
        epoch_phoneme_frequencies(_fb_stats, _phonemes)
        == {"[SILENCE]": 0.1, "AA": 0.2, "AE": 0.7},
    )
    # The regression itself: this used to raise AssertionError out of the job.
    _check(
        "without a lexicon an FB run costs the distribution, not the job",
        epoch_phoneme_frequencies(_fb_stats) is None,
    )
    _check(
        "a statistics dict with no distribution at all is still None, not a crash",
        epoch_phoneme_frequencies({"mean_log_likelihood_per_frame": -1.0}) is None,
    )
    # Silently zipping a short inventory would mis-name every cluster, which is
    # worse than reporting nothing.
    _check(
        "an inventory that does not match the cluster axis is refused",
        _raises(lambda: epoch_phoneme_frequencies(_fb_stats, ["AA", "AE"]), ValueError),
    )
    _reference = {"AA": 0.5, "AE": 0.5}
    _check(
        "the prior distance is computed from either counter set",
        abs(phoneme_prior_distance(_viterbi_stats, _reference, epoch=0) - 0.5) < 1e-12
        and abs(
            phoneme_prior_distance(_fb_stats, _reference, epoch=0, phonemes=_phonemes)
            - abs(0.2 / 0.9 - 0.5) * 2
        )
        < 1e-12,
    )

    print("ema")
    # The batched update: B re-estimations per pass over the corpus, each damped
    # against what came before. The two modes are different algorithms (see
    # lib.guided_kmeans.chunked.ema), so they are checked against their own
    # closed forms rather than against each other.
    em_C = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    em_T = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    em_model = VectorQuantizedModel(em_C, em_T)

    def _acc(counts):
        """A VQ accumulator holding exactly these counts."""
        return VectorQuantizedAccumulator(num_clusters=2).load_state_dict(
            {"num_clusters": 2, "num_codewords": 3, "counts": np.asarray(counts, float),
             "table_floor": 0.0}
        )

    em_N = [
        np.array([[6.0, 2.0, 0.0], [1.0, 4.0, 3.0]]),
        np.array([[1.0, 5.0, 2.0], [0.0, 1.0, 9.0]]),
        np.array([[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]),
    ]
    em_a = 0.75

    _check(
        "scale multiplies the sufficient statistic",
        np.allclose(_acc(em_N[0]).scale(0.25).counts, 0.25 * em_N[0]),
    )
    _check(
        "a negative decay is refused rather than flipping the statistic's sign",
        _raises(lambda: _acc(em_N[0]).scale(-0.5), ValueError),
    )
    _check(
        "the first step has nothing to damp against and is taken whole",
        np.allclose(ema_statistics(None, _acc(em_N[0]), em_a).counts, em_N[0]),
    )

    # S_t = a * S_{t-1} + (1 - a) * N_t, written out.
    em_expected = em_N[0]
    for n in em_N[1:]:
        em_expected = em_a * em_expected + (1.0 - em_a) * n
    em_running = None
    for n in em_N:
        em_running = ema_statistics(em_running, _acc(n), em_a)
    _check(
        "statistics mode is exactly S <- a*S + (1-a)*N",
        np.allclose(em_running.counts, em_expected),
    )
    _check(
        "alpha=0 keeps only the batch, i.e. no damping at all",
        np.allclose(ema_statistics(_acc(em_N[0]), _acc(em_N[1]), 0.0).counts, em_N[1]),
    )
    _check(
        "alpha=1 would never update and is refused",
        _raises(lambda: ema_statistics(_acc(em_N[0]), _acc(em_N[1]), 1.0), ValueError),
    )

    # The load-bearing property for GuidedClusteringEpochJob.hash: damping is
    # applied to the *merged* statistic, so how a batch was partitioned across
    # chunks still cannot change the result, and num_chunks stays unhashed.
    em_half = _acc(em_N[1] * 0.5)
    em_chunked = em_half.merge(_acc(em_N[1] * 0.5))
    _check(
        "chunking a batch commutes with damping it",
        np.allclose(
            ema_statistics(_acc(em_N[0]), em_chunked, em_a).counts,
            ema_statistics(_acc(em_N[0]), _acc(em_N[1]), em_a).counts,
        ),
    )
    # Statistics-level damping keeps the statistic a count, so a label's row
    # moves in proportion to the evidence that label actually got - the
    # difference from parameter-level damping, and the reason for having both.
    em_thin = np.array([[100.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    em_stat_table = ema_statistics(_acc(em_N[0]), _acc(em_thin), 0.5).finalize(em_model).table
    em_param_table = blend_parameters(
        _acc(em_N[0]).finalize(em_model), _acc(em_thin).finalize(em_model), 0.5
    ).table
    _check(
        "a barely-seen label moves less under statistics damping than under parameter damping",
        abs(em_stat_table[1, 2] - em_N[0][1, 2] / em_N[0][1].sum())
        < abs(em_param_table[1, 2] - em_N[0][1, 2] / em_N[0][1].sum()),
    )

    em_prev = _acc(em_N[0]).finalize(em_model)
    em_new = _acc(em_N[1]).finalize(em_model)
    em_blend = blend_parameters(em_prev, em_new, em_a)
    _check(
        "parameters mode is exactly a*theta_prev + (1-a)*theta_new",
        np.allclose(em_blend.table, em_a * em_prev.table + (1.0 - em_a) * em_new.table)
        and np.allclose(em_blend.centroids, em_C),
    )
    _check(
        "a blend of two normalized tables is a valid model",
        type(em_blend) is VectorQuantizedModel and np.allclose(em_blend.table.sum(1), 1.0),
    )
    _check(
        "blending is refused across model classes",
        _raises(lambda: blend_parameters(em_prev, EuclideanModel(em_C), 0.5), TypeError),
    )
    _check(
        "the window is batch_size / (1 - alpha), the two knobs' joint meaning",
        np.isclose(effective_window(2000, 0.8), 10000.0)
        and np.isclose(effective_window(500, 0.95), 10000.0),
    )

    import os as _os
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as em_dir:
        em_path = _os.path.join(em_dir, "state.pkl")
        save_state(em_running, em_path)
        em_back = VectorQuantizedAccumulator(num_clusters=2).load_state_dict(
            load_state(em_path)
        )
        _check(
            "the running statistic survives the trip between two jobs",
            np.allclose(em_back.counts, em_running.counts),
        )

    print("batched clustering")
    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        EMAConfig,
        GuidedClusteringEpochJob,
    )
    from .flavors import vq_flavor

    _check(
        "an out-of-range alpha or an unknown mode is caught at graph time",
        _raises(lambda: EMAConfig(alpha=1.0), ValueError)
        and _raises(lambda: EMAConfig(alpha=0.5, mode="counts"), ValueError),
    )

    def _batched(**kwargs):
        params = dict(
            num_epochs=2,
            batches_per_epoch=3,
            segments=tk.Path("/segments.txt"),
            flavor=vq_flavor(
                centroids=tk.Path("/cb.npy"),
                table=tk.Path("/t.npy"),
                recognition_config=common["recognition_config"],
                lexicon=common["lexicon"],
                num_clusters=40,
            ),
            **common,
        )
        params.update(kwargs)
        return chunked_clustering(**params)

    em_stats_run = _batched(ema=EMAConfig(alpha=0.8, mode="statistics"))
    em_param_run = _batched(ema=EMAConfig(alpha=0.8, mode="parameters"))

    _relaxed_flavor = mixture_flavor(
        centroids=tk.Path("/cb.npy"), covs=tk.Path("/v.npy"), mixtures=tk.Path("/m.npy"),
        recognition_config=common["recognition_config"], lexicon=common["lexicon"],
        num_clusters=40, update_densities=False, mixture_floor=1e-2, over_relaxation=2.0,
    )
    _check(
        "over-relaxation with EMA damping is refused, in either mode and the old route",
        all(
            _raises(lambda e=e: _batched(flavor=_relaxed_flavor, ema=e), NotImplementedError)
            for e in (EMAConfig(alpha=0.5, mode="statistics"),
                      EMAConfig(alpha=0.5, mode="parameters"))
        )
        and _raises(
            lambda: _batched(flavor=_relaxed_flavor, batches_per_epoch=1, ema_alpha=0.5),
            NotImplementedError,
        ),
    )
    _check(
        "undamped online steps take it",
        len(_batched(flavor=_relaxed_flavor, ema=EMAConfig(alpha=0.0)).jobs) == 6,
    )
    _check(
        "an epoch becomes B jobs",
        len(em_stats_run.jobs) == 6 and len(em_stats_run.out_batch_models) == 6,
    )
    _check(
        "every key that meant an epoch still means an epoch",
        sorted(em_stats_run.out_models) == [0, 1, 2]
        and sorted(em_stats_run.out_epoch_statistics) == [1, 2]
        and sorted(em_stats_run.out_hypotheses) == [0, 1],
    )
    _check(
        "a model numbered e is what epoch e ended with, a recognition what it started with",
        em_stats_run.out_models[1] == em_stats_run.jobs[2].out_model
        and em_stats_run.out_epoch_statistics[1] == em_stats_run.jobs[0].out_statistics,
    )
    _check(
        "the batches partition the corpus and repeat every epoch",
        [j.features.kwargs["segments"] for j in em_stats_run.jobs[:3]]
        == [j.features.kwargs["segments"] for j in em_stats_run.jobs[3:]]
        and len({j.features.kwargs["segments"].get_path() for j in em_stats_run.jobs}) == 3,
    )
    _check(
        "statistics mode chains a running statistic, parameters mode carries nothing",
        em_stats_run.jobs[0].prior_state is None
        and em_stats_run.jobs[1].prior_state == em_stats_run.jobs[0].out_accumulator_state
        and all(j.prior_state is None for j in em_param_run.jobs)
        and all(j.out_accumulator_state is None for j in em_param_run.jobs),
    )
    _check(
        "the two modes are different jobs",
        [j._sis_id() for j in em_stats_run.jobs] != [j._sis_id() for j in em_param_run.jobs],
    )
    _check(
        "a prior state without the mode that consumes it is refused",
        _raises(
            lambda: GuidedClusteringEpochJob(
                features=Spec(EuclideanModel, {}),
                model=Spec(EuclideanModel, {"centroids": tk.Path("/c.npy")}),
                recognizer=Spec(ArgmaxRecognizer, {}),
                accumulator=Spec(MeanAccumulator, {}),
                num_clusters=4,
                ema=EMAConfig(alpha=0.5, mode="parameters"),
                prior_state=tk.Path("/state.pkl"),
            ),
            TypeError,
        ),
    )
    # The job's own reduce path, not just the functions under it: both branches
    # write a model that a cluster job would otherwise only prove wrong after
    # its recognition has already run.
    def _epoch_job(mode, prior_state=None):
        return GuidedClusteringEpochJob(
            features=Spec(EuclideanModel, {}),
            model=Spec(VectorQuantizedModel, {"centroids": tk.Path("/cb.npy"),
                                              "table": tk.Path("/t.npy")}),
            recognizer=Spec(ArgmaxRecognizer, {}),
            accumulator=Spec(VectorQuantizedAccumulator, {}),
            num_clusters=2,
            ema=EMAConfig(alpha=em_a, mode=mode),
            prior_state=prior_state,
        )

    with _tempfile.TemporaryDirectory() as em_dir:
        em_prior_path = _os.path.join(em_dir, "prior.pkl")
        save_state(_acc(em_N[0]), em_prior_path)

        em_job = _epoch_job("statistics", tk.Path(em_prior_path))
        em_job.out_accumulator_state = tk.Path(_os.path.join(em_dir, "out.pkl"))
        em_totals = {"num_seqs": 100}
        em_job_model = em_job._update(_acc(em_N[1]), em_model, em_totals)
        _check(
            "the job's statistics branch is the recursion the tests above pin down",
            np.allclose(
                em_job_model.table,
                ema_statistics(_acc(em_N[0]), _acc(em_N[1]), em_a).finalize(em_model).table,
            ),
        )
        _check(
            "it hands the damped statistic on rather than the raw batch",
            np.allclose(
                load_state(em_job.out_accumulator_state.get_path())["counts"],
                em_a * em_N[0] + (1.0 - em_a) * em_N[1],
            ),
        )
        _check(
            "the run records the window it actually averaged over",
            em_totals["ema_mode"] == "statistics"
            and em_totals["ema_alpha"] == em_a
            and np.isclose(em_totals["ema_effective_window_seqs"], 100 / (1 - em_a)),
        )

    em_pjob = _epoch_job("parameters")
    em_ptotals = {"num_seqs": 100}
    _check(
        "the job's parameters branch damps after the M-step, against what it recognized with",
        np.allclose(
            em_pjob._update(_acc(em_N[1]), em_model, em_ptotals).table,
            blend_parameters(em_model, _acc(em_N[1]).finalize(em_model), em_a).table,
        )
        and em_pjob.out_accumulator_state is None,
    )
    em_plain = GuidedClusteringEpochJob(
        features=Spec(EuclideanModel, {}),
        model=Spec(VectorQuantizedModel, {"centroids": tk.Path("/cb.npy"),
                                          "table": tk.Path("/t.npy")}),
        recognizer=Spec(ArgmaxRecognizer, {}),
        accumulator=Spec(VectorQuantizedAccumulator, {}),
        num_clusters=2,
    )
    em_plain_totals = {"num_seqs": 100}
    _check(
        "without an ema the job is the plain M-step and records no ema fields",
        np.allclose(
            em_plain._update(_acc(em_N[1]), em_model, em_plain_totals).table,
            _acc(em_N[1]).finalize(em_model).table,
        )
        and em_plain_totals == {"num_seqs": 100},
    )

    _check(
        "batching without a segment list to split is refused",
        _raises(lambda: chunked_clustering(
            num_epochs=1, batches_per_epoch=2,
            initial_centroids=tk.Path("/init/c.npy"), **common), TypeError),
    )
    _check(
        "the two batched-EMA routes cannot be combined",
        _raises(lambda: _batched(ema_alpha=0.8), TypeError),
    )

    # The same continuation property as an unbatched run, which for statistics
    # mode holds only if the running statistic is carried over too - the model
    # alone does not determine the next step there.
    em_whole = _batched(num_epochs=2, ema=EMAConfig(alpha=0.8, mode="statistics"))
    em_head = _batched(num_epochs=1, ema=EMAConfig(alpha=0.8, mode="statistics"))
    em_tail = _batched(
        num_epochs=1,
        ema=EMAConfig(alpha=0.8, mode="statistics"),
        flavor=vq_flavor(
            centroids=em_head.out_artifacts["centroids"][1],
            table=em_head.out_artifacts["table"][1],
            recognition_config=common["recognition_config"],
            lexicon=common["lexicon"],
            num_clusters=40,
        ),
        initial_accumulator_state=em_head.out_accumulator_states[1],
    )
    _check(
        "1+1 continued == 2 in one call, statistics carried",
        [j._sis_id() for j in em_whole.jobs]
        == [j._sis_id() for j in em_head.jobs + em_tail.jobs],
    )
    _check(
        "continuing without the statistic is refused rather than silently restarting it",
        _raises(
            lambda: _batched(num_epochs=1, initial_accumulator_state=tk.Path("/s.pkl")),
            TypeError,
        ),
    )

    print("untied states")
    from itertools import product as _product

    from ..backoff_fb_graph import LabelLayout, build_topology
    from ..statistics import FBStatisticsCounter
    from ..util import posteriors_to_text
    from .runner import run_chunk as _run_chunk
    from ....setup.emission_balance import emission_balance, state_segment_costs

    tied, untied = LabelLayout(3, 2), LabelLayout(3, 2, tie_states=False)
    _check(
        "layout: tied S=2 emits the phoneme from every state",
        tied.num_labels == 3 and tied.emission_table().tolist() == [[0, 0], [1, 1], [2, 2]],
    )
    _check(
        "layout: untied is phoneme-major, V*S labels",
        untied.num_labels == 6
        and untied.emission_table().tolist() == [[0, 1], [2, 3], [4, 5]]
        and untied.phoneme_of().tolist() == [0, 0, 1, 1, 2, 2]
        and untied.state_of().tolist() == [0, 1, 0, 1, 0, 1],
    )
    _check("layout: untying one state is the tied layout",
           not LabelLayout(3, 1, tie_states=False).untied
           and LabelLayout(3, 1, tie_states=False).num_labels == 3)
    # The reference package, which the op's checkout carries; the same root
    # tools.BACKOFF_FB points at.
    fb_root = "/work/asr4/mann/tools/backoff_fb"
    topo = build_topology(["<SIL>", "A", "B"], loop_probability=0.5, num_states=2,
                          tie_states=False, backoff_fb_root=fb_root)
    _check("build_topology: untied emissions",
           topo.emis.tolist() == [[0, 1], [2, 3], [4, 5]] and topo.F == 6)
    topo = build_topology(["<SIL>", "A", "B"], loop_probability=0.5, num_states=2,
                          backoff_fb_root=fb_root)
    _check("build_topology: tied by default", topo.emis.tolist() == [[0, 0], [1, 1], [2, 2]])

    def onehot(labels, width):
        out = np.zeros((len(labels), width))
        out[np.arange(len(labels)), labels] = 1.0
        return out

    # a0 a1 | a0 a1 | b0 b1 b1 : two A's back to back, then B.
    seq = onehot([2, 3, 2, 3, 4, 5, 5], 6)
    _check("argmax segments: untied sees a repeated phoneme through the state reset",
           untied.argmax_segments(seq).tolist() == [1, 1, 2])
    _check("argmax segments: tied merges the same repeat",
           tied.argmax_segments(onehot([1, 1, 1, 1, 2, 2, 2], 3)).tolist() == [1, 2])
    _check("argmax segments: an untied state step is not a boundary",
           untied.argmax_segments(onehot([2, 2, 3, 3], 6)).tolist() == [1])
    _check("argmax segments: dead rows yield nothing",
           untied.argmax_segments(np.zeros((4, 6))).size == 0
           and tied.argmax_segments(np.zeros((0, 3))).size == 0)
    names = ["[SILENCE]", "AA", "B"]
    _check("posteriors_to_text: silence dropped, repeats kept untied",
           posteriors_to_text(onehot([0, 1, 2, 3, 2, 3, 0, 4, 5], 6), names, untied)
           == "AA AA B")
    _check("posteriors_to_text: tied",
           posteriors_to_text(onehot([1, 1, 0, 2], 3), names, tied) == "AA B")

    # The counter on untied posteriors whose states split each phoneme's mass
    # must describe the phonemes exactly as a tied counter describes the
    # folded posteriors - except argmax runs, which untied can split further.
    rng = np.random.default_rng(7)
    phon = rng.dirichlet(np.ones(3) * 0.3, size=(5, 40))
    split = rng.uniform(0.2, 0.8, size=(5, 40, 3, 1))
    states = np.concatenate([phon[..., None] * split, phon[..., None] * (1 - split)], axis=3)
    states = states.reshape(5, 40, 6)
    c_tied, c_untied = FBStatisticsCounter(3), FBStatisticsCounter(6, num_states=2, tie_states=False)
    for p_seq, s_seq in zip(phon, states):
        c_tied.read_gammas(p_seq, -1.0)
        c_untied.read_gammas(s_seq, -1.0)
    ft, fu = c_tied.finalize(), c_untied.finalize()
    same = all(
        np.allclose(ft[k], fu[k])
        for k in ("soft_cluster_frequencies", "mean_posterior_entropy",
                  "relative_loop_frequency", "average_segment_duration",
                  "mean_log_likelihood_per_frame")
    )
    _check("FB counter: untied phoneme statistics equal tied on folded posteriors", same)
    _check("FB counter: per-state keys only when untied",
           "soft_state_frequencies" in fu and len(fu["soft_state_frequencies"]) == 6
           and "soft_state_frequencies" not in ft
           and np.isclose(sum(fu["soft_state_frequencies"]), 1.0))
    expected_runs = sum(len(untied.argmax_segments(x)) for x in states)
    _check("FB counter: untied argmax runs follow the layout's rule",
           np.isclose(fu["average_segment_duration_argmax"], 200 / expected_runs))
    merged = pickle.loads(pickle.dumps(c_untied)).merge(pickle.loads(pickle.dumps(c_untied)))
    _check("FB counter: untied counters pickle and merge",
           np.allclose(merged.finalize()["soft_cluster_frequencies"], fu["soft_cluster_frequencies"]))
    try:
        FBStatisticsCounter(3).merge(FBStatisticsCounter(6, num_states=2, tie_states=False))
        mixed = False
    except ValueError:
        mixed = True
    _check("FB counter: refuses to merge different layouts", mixed)
    legacy = FBStatisticsCounter(3)
    for name in ("num_states", "tie_states", "state_entropy_sum"):
        del legacy.__dict__[name]
    revived = pickle.loads(pickle.dumps(legacy))
    _check("FB counter: a pickle from before layouts reads as tied",
           revived.layout == LabelLayout(3) and revived.merge(FBStatisticsCounter(3)) is revived)

    # Per-label argmax runs: silence (label 0) and speech durations apart.
    by_label = FBStatisticsCounter(3)
    by_label.read_gammas(onehot([0, 0, 0, 1, 1, 2, 0, 0], 3), -1.0)
    fl = by_label.finalize()
    _check("FB counter: argmax runs and frames per label",
           fl["argmax_runs_per_label"] == [2, 1, 1] and fl["argmax_frames_per_label"] == [5, 2, 1])
    _check("FB counter: silence and speech argmax durations, silence share",
           np.isclose(fl["silence_duration_argmax"], 2.5)
           and np.isclose(fl["speech_duration_argmax"], 1.5)
           and np.isclose(fl["silence_share_argmax"], 5 / 8)
           and np.isclose(fl["silence_share_soft"], 5 / 8)
           and np.isclose(fu["silence_share_soft"], fu["soft_cluster_frequencies"][0]))
    _check("FB counter: per-label runs add up to the argmax runs (tied)",
           sum(ft["argmax_runs_per_label"]) == round(200 / ft["average_segment_duration_argmax"])
           and sum(ft["argmax_frames_per_label"]) == 200)
    _check("FB counter: untied per-label runs follow argmax_segments",
           fu["argmax_runs_per_label"] == np.bincount(
               np.concatenate([untied.argmax_segments(x) for x in states]), minlength=3).tolist()
           and sum(fu["argmax_frames_per_label"]) == 200)
    dead = FBStatisticsCounter(3)
    dead.read_gammas(np.zeros((4, 3)), -1.0)
    dead.read_gammas(onehot([1, 1, 2], 3), -1.0)
    fd = dead.finalize()
    _check("FB counter: frames without posterior mass are not counted as silence",
           fd["argmax_runs_per_label"] == [0, 1, 1] and fd["argmax_frames_per_label"] == [0, 2, 1]
           and fd["silence_duration_argmax"] == 0.0)
    twice = pickle.loads(pickle.dumps(by_label)).merge(pickle.loads(pickle.dumps(by_label)))
    _check("FB counter: per-label counts pickle and merge",
           twice.finalize()["argmax_runs_per_label"] == [4, 2, 2]
           and np.isclose(twice.finalize()["silence_duration_argmax"], 2.5))
    old = pickle.loads(pickle.dumps(by_label))
    for name in ("argmax_label_runs", "argmax_label_frames"):
        del old.__dict__[name]
    old = pickle.loads(pickle.dumps(old))
    _check("FB counter: a pickle from before per-label runs has no per-label keys",
           "silence_duration_argmax" not in old.finalize()
           and "average_segment_duration_argmax" in old.finalize())
    mixed_merge = old.merge(pickle.loads(pickle.dumps(by_label))).finalize()
    _check("FB counter: an old pickle merges with a new one, per-label keys from the new",
           mixed_merge["argmax_runs_per_label"] == [2, 1, 1])

    # run_chunk: posterior hypotheses only for results that carry a sequence
    # score, and only when asked for.
    class _FakeFB:
        def __init__(self, posteriors, score):
            self.posteriors, self.score = posteriors, score

        def start(self, on_result):
            self.on_result = on_result

        def submit(self, tag, scores):
            self.on_result(RecognitionResult(seq_tag=tag, posteriors=self.posteriors,
                                             sequence_score=self.score))

        def drain(self):
            pass

        def shutdown(self):
            pass

    one_seq = [("s1", np.zeros((9, 2)))]

    def chunk_hyps(recognizer, **kw):
        return _run_chunk(
            features=_ListSource(one_seq), model=EuclideanModel(np.zeros((6, 2))),
            recognizer=recognizer, accumulator=NullAccumulator(),
            transcribe=lambda tb: "<traceback>", verbosity=0, **kw,
        ).hypotheses

    post = onehot([0, 1, 2, 3, 2, 3, 0, 4, 5], 6)
    transcriber = lambda g: posteriors_to_text(g, names, untied)
    _check("run_chunk: forward-backward results transcribed from posteriors",
           chunk_hyps(_FakeFB(post, -3.0), transcribe_posteriors=transcriber) == {"s1": "AA AA B"})
    _check("run_chunk: without a posterior transcriber nothing changes",
           chunk_hyps(_FakeFB(post, -3.0)) == {"s1": "<traceback>"})
    _check("run_chunk: a result without a sequence score keeps the traceback path",
           chunk_hyps(_FakeFB(post, None), transcribe_posteriors=transcriber)
           == {"s1": "<traceback>"})

    # Emission balance. Brute force first: the best left-to-right pass.
    def brute(c, S):
        F = len(c)
        best = np.full(c.shape[1] // S, np.inf)
        for cut in _product(range(1, F), repeat=S - 1):
            if list(cut) != sorted(set(cut)):
                continue
            bounds = (0, *cut, F)
            total = sum(c[bounds[j]:bounds[j + 1]].reshape(-1, c.shape[1] // S, S)[:, :, j].sum(0)
                        for j in range(S))
            best = np.minimum(best, total)
        return best

    costs = rng.normal(size=(30, 8))
    starts, ends = np.array([0, 3, 7, 12, 20]), np.array([3, 7, 12, 20, 30])
    got = state_segment_costs(costs, starts, ends, 2)
    want = np.stack([brute(costs[b:e], 2) for b, e in zip(starts, ends)])
    _check("balance: state segment costs equal brute force", np.allclose(got, want))
    got3 = state_segment_costs(costs[:, :6], starts, ends, 3)
    want3 = np.stack([brute(costs[b:e, :6], 3) for b, e in zip(starts, ends)])
    _check("balance: ... at S=3", np.allclose(got3, want3))
    one = state_segment_costs(costs, np.array([5]), np.array([6]), 2)
    _check("balance: a one-frame span takes its first state",
           np.allclose(one[0], costs[5].reshape(4, 2)[:, 0]))
    # Untied columns that duplicate the tied ones: every pass costs the sum,
    # so the balance must equal the tied one.
    tied_costs = rng.normal(size=(60, 5))
    seg_starts = np.array([0, 1, 4, 9, 15, 22, 30, 37, 45, 52])
    adjacent = np.array([True, True, False, True, True, True, False, True, True])
    a = emission_balance(tied_costs, seg_starts, adjacent, 0.35, 1.0)
    b = emission_balance(np.repeat(tied_costs, 2, axis=1), seg_starts, adjacent, 0.35, 1.0,
                         num_untied_states=2)
    _check("balance: duplicated untied columns give the tied balance",
           all(np.isclose(a[k], b[k]) for k in a))

    print("transition scores")
    import tempfile as _tempfile_tr

    from backoff_fb_ref import random_lm as _random_lm

    from .flavors import BackoffFBSearch
    from .interfaces import TransitionModel
    from .recognizers import BackoffFBRecognizer, OracleSegmentFBRecognizer
    from .transitions import EXIT, FORWARD, LOOP, OracleBoundaryTransitions, boundary_transitions

    ninf = -np.inf
    tr_labels = np.array([1, 1, 2, 2, 2, 1, 0])
    hard, inside = [ninf, ninf, 0.0], [0.0, 0.0, ninf]
    _check("boundary_transitions: exit forced where the label changes and on the last frame, "
           "forbidden elsewhere",
           np.array_equal(boundary_transitions(tr_labels),
                          np.array([inside, hard, inside, inside, hard, hard, hard])))
    soft = boundary_transitions(tr_labels, boundary_error=0.1)
    _check("boundary_transitions: a boundary error is a normalized boundary probability",
           np.allclose(np.exp(soft[:, LOOP]) + np.exp(soft[:, EXIT]), 1.0)
           and np.array_equal(soft[:, LOOP], soft[:, FORWARD])
           and np.allclose(np.exp(soft[[1, 4, 5, 6], EXIT]), 0.9)
           and np.allclose(np.exp(soft[[0, 2, 3], EXIT]), 0.1))
    _check("boundary_transitions: an error of 0.5 or more is refused",
           _raises(lambda: boundary_transitions(tr_labels, 0.5), ValueError)
           and _raises(lambda: boundary_transitions(tr_labels, -0.1), ValueError))

    from .transitions import boundary_probability, jitter_boundaries, segment_ends

    # Long segments, so that clipping never binds and the shifts are the draw.
    long_ends = segment_ends(np.repeat(np.arange(400) % 3, 40))
    shifts = []
    for s in range(5):
        moved = jitter_boundaries(long_ends, 2.0, np.random.default_rng(s))
        shifts.append(np.flatnonzero(moved) - np.flatnonzero(long_ends))
    shifts = np.concatenate(shifts)
    _check("jitter: shifts are round(2 * N(0, 1)) when nothing clips",
           abs(shifts.mean()) < 0.1 and abs(shifts.std() - 2.0) < 0.1
           # P(round(2z) = 0) = P(|z| < 1/4) = 0.197
           and abs(np.mean(shifts == 0) - 0.197) < 0.03,
           f"mean {shifts.mean():.3f} std {shifts.std():.3f} "
           f"P(0) {np.mean(shifts == 0):.3f}")
    crowded = segment_ends(np.array([0, 1, 2, 2, 0, 1, 1, 1, 2, 0]))
    ok = True
    for s in range(200):
        moved = jitter_boundaries(crowded, 3.0, np.random.default_rng(s))
        ok &= moved.sum() == crowded.sum() and bool(moved[-1])
    _check("jitter: one-frame segments survive any draw - same count, last frame kept", ok)
    _check("jitter: 0 moves nothing",
           np.array_equal(jitter_boundaries(crowded, 0.0, np.random.default_rng(0)), crowded))
    p, not_p = boundary_probability(long_ends, boundary_error=0.02, blur=2.0)
    b = np.flatnonzero(long_ends)[3]
    _check("blur: p peaks at 1 - eps on the boundary, is symmetric, and falls to eps",
           np.isclose(p[b], 0.98) and np.allclose(p[b - 3:b], p[b + 3:b:-1])
           and np.isclose(p[b + 20], 0.02, atol=1e-6) and np.allclose(p + not_p, 1.0))
    _check("blur: a vanishing width is the unblurred model",
           np.allclose(boundary_transitions(tr_labels, 0.1, blur=1e-3),
                       boundary_transitions(tr_labels, 0.1)))
    _check("blur needs a boundary error, or the boundary frame stays hard",
           _raises(lambda: boundary_transitions(tr_labels, 0.0, blur=1.0), ValueError))

    zt = build_topology(["<SIL>", "A", "B"], loop_probability=0.0, num_states=2,
                        zero_topology=True, backoff_fb_root=fb_root)
    _check("zero topology: every weight 0 except the last state's forward arc",
           (zt.tau_loop == 0).all() and (zt.tau_exit == 0).all()
           and (zt.tau_fwd[:, 0] == 0).all() and np.isneginf(zt.tau_fwd[:, 1]).all())
    _check("zero topology: loop parameters it would ignore are refused",
           _raises(lambda: build_topology(["A"], loop_probability=0.5, zero_topology=True,
                                          backoff_fb_root=fb_root), ValueError))

    # The two routes to the oracle on a real (tiny) LM, through the CPU
    # reference the recognizer falls back on without a GPU. Segment lists are
    # (label, frames); utt2 is one-frame segments only, utt4 has none shorter
    # than 2 frames.
    tr_segs = {
        "utt0": [(1, 3), (2, 1), (0, 2), (1, 4)],
        "utt1": [(2, 5)],
        "utt2": [(1, 1), (2, 1), (1, 1)],
        "utt3": [(0, 2), (1, 1), (2, 6), (1, 2), (2, 3)],
        "utt4": [(2, 2), (1, 3), (2, 2), (0, 4)],
    }
    tr_frames = {t: np.repeat([a for a, _ in s], [n for _, n in s]) for t, s in tr_segs.items()}
    tr_rng = np.random.RandomState(11)
    tr_feats = {t: tr_rng.randn(len(lab), 4) for t, lab in tr_frames.items()}

    def _collect(rec, items):
        out = {}
        rec.start(lambda r: out.__setitem__(r.seq_tag, r))
        for tag, scores, theta in items:
            if theta is None:
                rec.submit(tag, scores)
            else:
                rec.submit(tag, scores, transitions=theta)
        rec.drain()
        rec.shutdown()
        return out

    with _tempfile_tr.TemporaryDirectory() as tmp:
        arpa = os.path.join(tmp, "lm.arpa")
        _random_lm(3, 3, density=0.7, seed=3).to_arpa(arpa, ["<SIL>", "A", "B"])
        lex = os.path.join(tmp, "lexicon.xml")
        with open(lex, "w") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n<lexicon>\n'
                    '<lemma><orth>[SILENCE]</orth><synt><tok>&lt;SIL&gt;</tok></synt></lemma>\n'
                    '<lemma><orth>A</orth></lemma>\n<lemma><orth>B</orth></lemma>\n'
                    '<lemma special="sentence-begin"><orth>[b]</orth><synt><tok>&lt;s&gt;</tok></synt></lemma>\n'
                    '<lemma special="sentence-end"><orth>[e]</orth><synt><tok>&lt;/s&gt;</tok></synt></lemma>\n'
                    '</lexicon>\n')
        labels_pkl = os.path.join(tmp, "labels.pkl")
        with open(labels_pkl, "wb") as f:
            pickle.dump(tr_frames, f)
        op = dict(backoff_fb_root=fb_root, arpa_path=arpa, lexicon_path=lex,
                  distance_scale=0.3, device="cpu", batch_size=2)
        oracle = OracleBoundaryTransitions(labels_pkl)
        _check("OracleBoundaryTransitions is a TransitionModel, and fixed",
               isinstance(oracle, TransitionModel) and not oracle.trainable)
        _check("OracleBoundaryTransitions: unknown tags and pooled features are refused",
               _raises(lambda: oracle.transition_scores("nope", tr_feats["utt0"]), KeyError)
               and _raises(lambda: oracle.transition_scores("utt0", tr_feats["utt0"][:3]), ValueError))

        tied_model = EuclideanModel(tr_rng.randn(3, 4))
        scores = {t: tied_model.scores(x) for t, x in tr_feats.items()}
        seg_route = _collect(OracleSegmentFBRecognizer(labels_pkl, num_clusters=3, **op),
                             [(t, scores[t], None) for t in tr_segs])
        frame_route = _collect(
            BackoffFBRecognizer(num_clusters=3, zero_topology=True, **op),
            [(t, scores[t], oracle.transition_scores(t, tr_feats[t])) for t in tr_segs])
        _check("oracle transitions at S=1 reproduce the oracle-segment recognizer: Gamma",
               all(np.abs(seg_route[t].posteriors - frame_route[t].posteriors).max() < 1e-10
                   for t in tr_segs),
               str(max(np.abs(seg_route[t].posteriors - frame_route[t].posteriors).max()
                       for t in tr_segs)))
        _check("... and log Z, which the zero topology leaves unshifted",
               all(abs(seg_route[t].sequence_score - frame_route[t].sequence_score) < 1e-9
                   for t in tr_segs))
        _check("... and a frame-level result has one row per frame",
               all(len(frame_route[t].posteriors) == len(tr_frames[t]) for t in tr_segs))

        stationary = dict(num_clusters=3, loop_probability=0.5, **op)
        none_route = _collect(BackoffFBRecognizer(**stationary),
                              [(t, scores[t], None) for t in tr_segs])
        zero_route = _collect(BackoffFBRecognizer(**stationary),
                              [(t, scores[t], np.zeros((len(scores[t]), 3))) for t in tr_segs])
        _check("theta = 0 is bitwise the stationary model",
               all(np.array_equal(none_route[t].posteriors, zero_route[t].posteriors)
                   and none_route[t].sequence_score == zero_route[t].sequence_score
                   for t in tr_segs))

        untied_model = EuclideanModel(tr_rng.randn(6, 4))
        untied_scores = {t: untied_model.scores(x) for t, x in tr_feats.items()}
        untied = dict(num_clusters=6, num_states=2, tie_states=False, zero_topology=True, **op)
        hard2 = _collect(BackoffFBRecognizer(**untied),
                         [(t, untied_scores[t], oracle.transition_scores(t, tr_feats[t]))
                          for t in tr_segs])
        short = {t for t, s in tr_segs.items() if min(n for _, n in s) < 2}
        _check("untied S=2, hard oracle: exactly the sequences with a one-frame segment die",
               {t for t, r in hard2.items() if not np.isfinite(r.sequence_score)} == short
               and all(not hard2[t].posteriors.any() for t in short)
               and all(np.allclose(hard2[t].posteriors.sum(1), 1.0) for t in set(tr_segs) - short))
        soft_oracle = OracleBoundaryTransitions(labels_pkl, boundary_error=0.05)
        soft2 = _collect(BackoffFBRecognizer(**untied),
                         [(t, untied_scores[t], soft_oracle.transition_scores(t, tr_feats[t]))
                          for t in tr_segs])
        _check("untied S=2 with a boundary error: every sequence has a path",
               all(np.isfinite(r.sequence_score) and np.allclose(r.posteriors.sum(1), 1.0)
                   for r in soft2.values()))

        # The sweep in vq_gmm_tau_boundary_noise rests on its two endpoints being
        # existing runs: hard boundaries on a loop-0.75 prior give the oracle's
        # posteriors (the prior is a constant per segment), and as the error
        # rate approaches 0.5 the scores go flat and leave the free segmentation.
        prior = dict(num_clusters=3, loop_probability=0.75, **op)
        hard_on_prior = _collect(BackoffFBRecognizer(**prior),
                                 [(t, scores[t], oracle.transition_scores(t, tr_feats[t]))
                                  for t in tr_segs])
        _check("degraded oracle: hard boundaries on a duration prior give the oracle's Gamma",
               all(np.abs(hard_on_prior[t].posteriors - seg_route[t].posteriors).max() < 1e-10
                   for t in tr_segs))
        flat = OracleBoundaryTransitions(labels_pkl, boundary_error=0.5 - 1e-12)
        flat_on_prior = _collect(BackoffFBRecognizer(**prior),
                                 [(t, scores[t], flat.transition_scores(t, tr_feats[t]))
                                  for t in tr_segs])
        free = _collect(BackoffFBRecognizer(**prior), [(t, scores[t], None) for t in tr_segs])
        _check("degraded oracle: an error rate of 1/2 leaves the free segmentation's Gamma",
               all(np.abs(flat_on_prior[t].posteriors - free[t].posteriors).max() < 1e-9
                   for t in tr_segs))

        # Jitter, end to end: the jittered hard boundaries at S=1 must equal
        # the segment-level recognizer run on labels with those boundaries.
        jit = OracleBoundaryTransitions(labels_pkl, jitter=1.5, seed=3)
        jit_frames = {t: np.cumsum(np.r_[False, jit.boundaries(t, len(tr_frames[t]))[:-1]])
                      for t in tr_segs}
        jit_pkl = os.path.join(tmp, "jittered.pkl")
        with open(jit_pkl, "wb") as f:
            pickle.dump(jit_frames, f)
        jit_seg = _collect(OracleSegmentFBRecognizer(jit_pkl, num_clusters=3, **op),
                           [(t, scores[t], None) for t in tr_segs])
        jit_frame = _collect(BackoffFBRecognizer(num_clusters=3, zero_topology=True, **op),
                             [(t, scores[t], jit.transition_scores(t, tr_feats[t]))
                              for t in tr_segs])
        _check("jitter: per-frame route equals the segment route on the jittered boundaries",
               all(np.abs(jit_seg[t].posteriors - jit_frame[t].posteriors).max() < 1e-10
                   for t in tr_segs)
               and any(not np.array_equal(jit_frames[t] != np.r_[jit_frames[t][1:], -1],
                                          tr_frames[t] != np.r_[tr_frames[t][1:], -1])
                       for t in tr_segs))

        # The change-point detector (findings §6.11): its scores against the
        # plain loop that produced the measurements, its boundaries, and the
        # same segment-route equivalence as the jitter above.
        from .transitions import ChangePointTransitions, change_point_boundaries, change_point_scores

        def _cp_loop(x, w, dist):
            T, s = len(x), np.full(len(x), -np.inf)
            for t in range(1, T):
                a, b = max(0, t - w), min(T, t + w)
                left, right = x[a:t].mean(0), x[t:b].mean(0)
                s[t] = (np.linalg.norm(left - right) if dist == "euclidean" else
                        1 - left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
            return s

        cp_x = tr_rng.randn(37, 5)
        _check("change points: scores equal the plain loop (windows 1-3, both distances)",
               all(np.allclose(change_point_scores(cp_x, w, d)[1:], _cp_loop(cp_x, w, d)[1:],
                               rtol=1e-12, atol=1e-12)
                   and change_point_scores(cp_x, w, d)[0] == ninf
                   for w in (1, 2, 3) for d in ("euclidean", "cosine")))
        cp_s = change_point_scores(cp_x)
        cp_b = change_point_boundaries(cp_s, 0.25)
        cp_peaks = [t for t in range(1, 36) if cp_s[t] > cp_s[t - 1] and cp_s[t] >= cp_s[t + 1]]
        _check("change points: the round(rate T) highest local maxima, or all of them",
               set(cp_b) <= set(cp_peaks)
               and len(cp_b) == min(round(0.25 * 37), len(cp_peaks))
               and (len(cp_b) == len(cp_peaks)
                    or cp_s[cp_b].min() >= max(cp_s[t] for t in cp_peaks if t not in cp_b))
               and np.all(np.diff(cp_b) >= 2)
               and len(change_point_boundaries(cp_s, 0.9)) == len(cp_peaks))
        cp_model = ChangePointTransitions()
        cp_ends = cp_model.boundaries(cp_x)
        _check("change points: a fixed TransitionModel whose ends are the detected starts minus one",
               isinstance(cp_model, TransitionModel) and not cp_model.trainable
               and np.array_equal(np.flatnonzero(cp_ends[:-1]) + 1, cp_b) and cp_ends[-1]
               and np.array_equal(cp_model.transition_scores("any", cp_x),
                                  boundary_transitions(None, 0.1, ends=cp_ends)))
        _check("change points: bad settings are refused when the model is built",
               all(_raises(lambda kw=kw: ChangePointTransitions(**kw), ValueError) for kw in (
                   {"window": 0}, {"distance": "manhattan"}, {"rate": 0.0}, {"rate": 1.0},
                   {"boundary_error": 0.5}, {"blur": 1.0, "boundary_error": 0.0})))
        hard_cp = ChangePointTransitions(boundary_error=0.0)
        cp_frames = {t: np.cumsum(np.r_[False, hard_cp.boundaries(tr_feats[t])[:-1]])
                     for t in tr_segs}
        cp_pkl = os.path.join(tmp, "change_points.pkl")
        with open(cp_pkl, "wb") as f:
            pickle.dump(cp_frames, f)
        cp_seg = _collect(OracleSegmentFBRecognizer(cp_pkl, num_clusters=3, **op),
                          [(t, scores[t], None) for t in tr_segs])
        cp_frame = _collect(BackoffFBRecognizer(num_clusters=3, zero_topology=True, **op),
                            [(t, scores[t], hard_cp.transition_scores(t, tr_feats[t]))
                             for t in tr_segs])
        _check("change points: hard detector boundaries equal the segment route on them",
               all(np.abs(cp_seg[t].posteriors - cp_frame[t].posteriors).max() < 1e-10
                   for t in tr_segs))
        soft_cp = _collect(BackoffFBRecognizer(**untied),
                           [(t, untied_scores[t], cp_model.transition_scores(t, tr_feats[t]))
                            for t in tr_segs])
        cp_prior = _collect(BackoffFBRecognizer(**prior),
                            [(t, scores[t], cp_model.transition_scores(t, tr_feats[t]))
                             for t in tr_segs])
        _check("change points: at S=2 and on a duration prior every sequence has a path",
               all(np.isfinite(r.sequence_score) and np.allclose(r.posteriors.sum(1), 1.0)
                   for r in list(soft_cp.values()) + list(cp_prior.values())))

        _check("a batch mixing sequences with and without transition scores is refused",
               _raises(lambda: _collect(BackoffFBRecognizer(**stationary), [
                   ("utt0", scores["utt0"], None),
                   ("utt1", scores["utt1"], np.zeros((len(scores["utt1"]), 3)))]), ValueError))
        _check("transition scores of the wrong shape are refused",
               _raises(lambda: _collect(BackoffFBRecognizer(**stationary), [
                   ("utt0", scores["utt0"], np.zeros((len(scores["utt0"]), 2)))]), ValueError))
        _check("transition posteriors raise until the op computes them (T3)",
               _raises(lambda: BackoffFBRecognizer(transition_posteriors=True, num_clusters=3, **op),
                       NotImplementedError))
        _check("the oracle-segment recognizer refuses a free loop",
               _raises(lambda: OracleSegmentFBRecognizer(labels_pkl, num_clusters=3,
                                                         zero_topology=True, **op), ValueError))

        # The same comparison one level up, through run_chunk: the transition
        # model is called on the features and its scores reach the search.
        def _chunk(recognizer, transitions=None, transition_accumulator=None):
            return run_chunk(
                features=_ListSource([(t, tr_feats[t]) for t in tr_segs]),
                model=tied_model, recognizer=recognizer,
                accumulator=MeanAccumulator(3, 4), counter=None,
                transitions=transitions, transition_accumulator=transition_accumulator,
                verbosity=0)

        by_segments = _chunk(OracleSegmentFBRecognizer(labels_pkl, num_clusters=3, **op))
        by_frames = _chunk(BackoffFBRecognizer(num_clusters=3, zero_topology=True, **op),
                           transitions=oracle)
        _check("run_chunk: the two oracle routes accumulate the same statistics",
               by_frames.num_recognized == len(tr_segs)
               and all(np.allclose(by_segments.accumulator_state[k],
                                   by_frames.accumulator_state[k], atol=1e-10)
                       for k in ("counts", "sums"))
               and by_frames.accumulator_state["counts"].sum() > 0)
        _check("run_chunk: a recognizer without transition input is refused",
               _raises(lambda: _chunk(_StubFBRecognizer({}), transitions=oracle), ValueError))
        _check("run_chunk: a transition accumulator raises until T3",
               _raises(lambda: _chunk(BackoffFBRecognizer(num_clusters=3, **op),
                                      transition_accumulator=object()), NotImplementedError))

    oracle_spec = Spec(OracleBoundaryTransitions, {"frame_labels": tk.Path("/labels.pkl")})
    fb_common = dict(backoff_fb_root=tk.Path("/fb"), arpa_path=tk.Path("/lm.arpa"),
                     lexicon=tk.Path("/fb_lex.gz"))
    flavor_common = dict(centroids=tk.Path("/init/c.npy"), recognition_config=tk.Path("/r.config"),
                         lexicon=tk.Path("/l.gz"), num_clusters=40, use_forward_backward=True)
    plain_flavor = euclidean_flavor(**flavor_common, backoff_fb=BackoffFBSearch(**fb_common))
    oracle_flavor = euclidean_flavor(**flavor_common, backoff_fb=BackoffFBSearch(
        **fb_common, transitions=oracle_spec, zero_topology=True))
    _check("flavor: without transitions nothing new enters the recognizer spec",
           plain_flavor.transitions is None and "zero_topology" not in plain_flavor.recognizer.kwargs)
    _check("flavor: the transition model rides on the flavor, the zero topology on the search",
           oracle_flavor.transitions is oracle_spec
           and oracle_flavor.recognizer.kwargs.get("zero_topology") is True)
    for what, kwargs in [
        ("a zero topology without transitions", dict(zero_topology=True)),
        ("transitions next to frame_labels", dict(transitions=oracle_spec, frame_labels=tk.Path("/x"))),
        ("a zero topology with a loop probability",
         dict(transitions=oracle_spec, zero_topology=True, loop_probability=0.5)),
    ]:
        _check(f"flavor: {what} is refused",
               _raises(lambda: euclidean_flavor(**flavor_common, backoff_fb=BackoffFBSearch(
                   **fb_common, **kwargs)), ValueError))

    class _TrainableTransitions:
        trainable = True

    _check("flavor: a trainable transition model raises until it can be re-estimated",
           _raises(lambda: euclidean_flavor(**flavor_common, backoff_fb=BackoffFBSearch(
               **fb_common, transitions=Spec(_TrainableTransitions, {}))), NotImplementedError))
    _check("flavor: a transition accumulator raises",
           _raises(lambda: ClusteringFlavor(
               model=plain_flavor.model, accumulator=plain_flavor.accumulator,
               recognizer=plain_flavor.recognizer,
               transition_accumulator=Spec(MeanAccumulator, {})), NotImplementedError))

    tr_common = dict(features_hdf=tk.Path("/f.hdf"), recognition_config=tk.Path("/r.config"),
                     lexicon=tk.Path("/l.gz"), num_clusters=40, use_forward_backward=True)
    plain_run = chunked_clustering(num_epochs=2, flavor=plain_flavor, **tr_common)
    oracle_run = chunked_clustering(num_epochs=2, flavor=oracle_flavor, **tr_common)
    _check("chunked_clustering: the transition model enters the epoch job hash",
           all(a._sis_id() != b._sis_id() for a, b in zip(plain_run.jobs, oracle_run.jobs)))
    _check("chunked_clustering: every epoch carries the same transition model, its "
           "alignment an input",
           all(j.transitions == oracle_spec
               and "/labels.pkl" in {p.get_path() for p in j._sis_inputs}
               for j in oracle_run.jobs)
           and all(j.transitions is None for j in plain_run.jobs))
    _check("chunked_clustering: the single-task EMA route refuses a transition model",
           _raises(lambda: chunked_clustering(num_epochs=1, flavor=oracle_flavor, ema_alpha=0.5,
                                              **tr_common), NotImplementedError))

    print("stationary topology")
    _stationary_topology_checks(fb_root, fb_common, flavor_common, tr_common, plain_flavor,
                                plain_run)

    print("arpa decompression")
    import gzip as _gzip
    import tempfile as _tempfile

    from ..backoff_fb_graph import _plain_arpa

    text = b"\\data\\\nngram 1=2\n\n\\1-grams:\n-1.0\tA\n-1.0\tB\n\n\\end\\\n"
    with _tempfile.TemporaryDirectory() as tmp:
        gz = os.path.join(tmp, "lm.arpa.gz")
        with _gzip.open(gz, "wb") as f:
            f.write(text)
        with _plain_arpa(gz) as plain:
            ok = open(plain, "rb").read() == text and not plain.endswith(".gz")
        _check("arpa: a .gz is handed over decompressed and removed afterwards",
               ok and not os.path.exists(plain))
        with _plain_arpa(os.path.join(tmp, "x.arpa")) as plain:
            _check("arpa: a plain path passes through", plain == os.path.join(tmp, "x.arpa"))
        # A temp directory that cannot take the file (here: does not exist, as
        # good as full) falls back to the working directory.
        work = os.path.join(tmp, "work")
        os.mkdir(work)
        saved_tempdir, saved_cwd = _tempfile.tempdir, os.getcwd()
        _tempfile.tempdir = os.path.join(tmp, "no-such-dir")
        os.chdir(work)
        try:
            with _plain_arpa(gz) as plain:
                fell_back = (os.path.dirname(plain) == work
                             and open(plain, "rb").read() == text)
            _check("arpa: an unusable temp dir falls back to the working directory",
                   fell_back and not os.listdir(work))
        finally:
            _tempfile.tempdir = saved_tempdir
            os.chdir(saved_cwd)

    _label_covariance_checks()

    print("spec")
    from sisyphus.hash import sis_hash_helper
    from sisyphus.tools import extract_paths

    base = Spec(EuclideanModel, {"centroids": tk.Path("/x/c.npy")}, {"num_workers": 8})
    other_workers = Spec(EuclideanModel, {"centroids": tk.Path("/x/c.npy")}, {"num_workers": 32})
    other_path = Spec(EuclideanModel, {"centroids": tk.Path("/y/c.npy")})
    _check(
        "unhashed kwargs excluded from the hash",
        sis_hash_helper(base.hashed()) == sis_hash_helper(other_workers.hashed()),
    )
    _check(
        "hashed kwargs still affect the hash",
        sis_hash_helper(base.hashed()) != sis_hash_helper(other_path.hashed()),
    )
    _check(
        "sisyphus finds paths nested in a spec",
        [p.get_path() for p in extract_paths(base)] == ["/x/c.npy"],
    )

    print()
    if _FAILURES:
        print(f"FAILED: {', '.join(_FAILURES)}")
        return 1
    print("all checks passed")
    return 0


def _stationary_topology_checks(fb_root, fb_common, flavor_common, tr_common, plain_flavor,
                                plain_run):
    """The loop table, the transition counts and the EM update of the loop
    probabilities (the op's SPEC.md 15.6 and 15.7), on the op's CPU reference."""
    import json
    import os
    import tempfile

    from sisyphus import tk

    from backoff_fb_ref import random_lm

    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        EMAConfig,
        GuidedClusteringEpochJob,
        chunked_clustering,
    )
    from ..backoff_fb_graph import build_topology
    from .flavors import BackoffFBSearch
    from .recognizers import BackoffFBRecognizer
    from .runner import ChunkResult, load_chunk
    from .topology import TYINGS, StationaryTopology, TopologyAccumulator, tying_groups
    from .transitions import OracleBoundaryTransitions

    symbols = ["<SIL>", "A", "B"]
    groups, names = tying_groups("global", symbols, 2)
    _check("tying: global is one group", groups.tolist() == [[0, 0]] * 3 and names == ["all"])
    groups, names = tying_groups("speech_silence", symbols, 2)
    _check("tying: speech_silence gives silence a group of its own",
           groups.tolist() == [[1, 1], [0, 0], [0, 0]] and names == ["speech", "silence"])
    groups, names = tying_groups("phoneme", symbols, 2)
    _check("tying: phoneme ties a phoneme's states",
           groups.tolist() == [[0, 0], [1, 1], [2, 2]] and names == symbols)
    groups, names = tying_groups("state", symbols, 2)
    _check("tying: state is one group per pair",
           groups.tolist() == [[0, 1], [2, 3], [4, 5]] and names[3] == "A.1")
    _check("tying: an unknown tying, and speech_silence without silence, are refused",
           _raises(lambda: tying_groups("label", symbols, 2), ValueError)
           and _raises(lambda: tying_groups("speech_silence", ["A", "B"], 2), ValueError))

    # The table route and the scalar route build the same weights, bitwise.
    table = StationaryTopology.from_scalars(symbols, 2, 0.6, 0.3)
    _check("table: from_scalars fills the silence row and the speech rows",
           table.loop_probabilities.tolist() == [[0.3, 0.3], [0.6, 0.6], [0.6, 0.6]])
    topo_common = dict(transition_scale=0.35, num_states=2, backoff_fb_root=fb_root)
    by_scalars = build_topology(symbols, loop_probability=0.6, silence_loop_probability=0.3,
                                **topo_common)
    by_table = build_topology(symbols, loop_table=table.loop_probabilities, **topo_common)
    _check("build_topology: a table filled from the scalars is bitwise the scalars' topology",
           all(np.array_equal(getattr(by_scalars, k), getattr(by_table, k))
               for k in ("tau_loop", "tau_fwd", "tau_exit", "emis")))
    offset = np.array([0.0, -0.5, 0.25])
    by_offset = build_topology(symbols, loop_table=table.loop_probabilities, exit_offset=offset,
                               **topo_common)
    _check("build_topology: the exit offset is added after the scale, to the exit alone",
           np.array_equal(by_offset.tau_exit, by_table.tau_exit + offset)
           and np.array_equal(by_offset.tau_loop, by_table.tau_loop)
           and np.array_equal(by_offset.tau_fwd, by_table.tau_fwd))
    per_state = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    by_state = build_topology(symbols, loop_table=per_state, **topo_common)
    _check("build_topology: a per-state table sets each state's loop, and the last state's exit",
           np.allclose(by_state.tau_loop, 0.35 * np.log(per_state))
           and np.allclose(by_state.tau_fwd[:, 0], 0.35 * np.log1p(-per_state[:, 0]))
           and np.allclose(by_state.tau_exit, 0.35 * np.log1p(-per_state[:, 1])))
    for what, kwargs in [
        ("a table next to the scalars", dict(loop_table=per_state, loop_probability=0.5)),
        ("a table of the wrong shape", dict(loop_table=per_state[:2])),
        ("a probability of 1", dict(loop_table=np.ones((3, 2)))),
        ("a table on the zero topology", dict(loop_table=per_state, zero_topology=True,
                                              transition_scale=1.0)),
        ("an infinite exit offset", dict(loop_table=per_state, exit_offset=-np.inf)),
        ("neither scalars nor a table", dict()),
    ]:
        kw = {**topo_common, **kwargs}
        _check(f"build_topology: {what} is refused",
               _raises(lambda: build_topology(symbols, **kw), (ValueError, TypeError)))

    with tempfile.TemporaryDirectory() as tmp:
        saved = os.path.join(tmp, "topology")
        StationaryTopology(per_state, symbols).save(saved)
        back = StationaryTopology.load(saved)
        _check("StationaryTopology: save and load round-trip",
               np.array_equal(back.loop_probabilities, per_state) and back.symbols == tuple(symbols)
               and sorted(os.listdir(saved)) == ["loop_probabilities.npy", "topology.json"])
    _check("StationaryTopology: values outside [0, 1), NaN and a short symbol list are refused",
           _raises(lambda: StationaryTopology(np.full((3, 2), 1.0), symbols), ValueError)
           and _raises(lambda: StationaryTopology(np.full((3, 2), np.nan), symbols), ValueError)
           and _raises(lambda: StationaryTopology(per_state, symbols[:2]), ValueError))

    # The accumulator on synthetic counts.
    rng = np.random.default_rng(5)
    synthetic = [rng.uniform(0.0, 5.0, size=(3, 2, 2)) for _ in range(12)]
    whole = TopologyAccumulator("phoneme")
    for c in synthetic:
        whole.observe(c)
    parts = [TopologyAccumulator("phoneme") for _ in range(3)]
    for i, c in enumerate(synthetic):
        parts[i % 3].observe(c)
    merged = TopologyAccumulator("phoneme").load_state_dict(parts[0].state_dict())
    merged.merge(parts[1]).merge(parts[2])
    _check("TopologyAccumulator: merge is associative over chunkings",
           np.allclose(merged.counts, whole.counts, rtol=1e-14, atol=0) and merged.num_seqs == 12)
    _check("TopologyAccumulator: an empty accumulator merges as the identity",
           np.array_equal(TopologyAccumulator().merge(whole).counts, whole.counts)
           and np.array_equal(TopologyAccumulator().load_state_dict(whole.state_dict())
                              .merge(TopologyAccumulator()).counts, whole.counts))
    _check("TopologyAccumulator: scale",
           np.allclose(TopologyAccumulator().load_state_dict(whole.state_dict()).scale(0.5).counts,
                       0.5 * whole.counts))

    def fresh(tying="phoneme", **kw):
        return TopologyAccumulator(tying, **kw).load_state_dict(whole.state_dict())

    start = StationaryTopology.from_scalars(symbols, 2, 0.5)
    stay, total = whole.counts[..., 0], whole.counts.sum(-1)
    expected = {
        "global": np.full((3, 2), stay.sum() / total.sum()),
        "speech_silence": np.array([[stay[0].sum() / total[0].sum()] * 2]
                                   + [[stay[1:].sum() / total[1:].sum()] * 2] * 2),
        "phoneme": np.repeat((stay.sum(1) / total.sum(1))[:, None], 2, axis=1),
        "state": stay / total,
    }
    _check("TopologyAccumulator: each tying's update is its stays over its frames",
           all(np.allclose(fresh(t).finalize(start).loop_probabilities, expected[t], rtol=1e-13)
               for t in TYINGS))
    floor = float(total[1, 0]) + 1e-9
    floored = fresh("state", min_frames=floor).finalize(start)
    _check("TopologyAccumulator: a group below the floor keeps its previous p",
           np.allclose(floored.loop_probabilities, np.where(total >= floor, stay / total, 0.5))
           and floored.loop_probabilities[1, 0] == 0.5)
    p_global = stay.sum() / total.sum()
    _check("TopologyAccumulator: a Beta prior adds m pseudo-frames at its mean ...",
           np.allclose(fresh(prior_weight=10.0, prior_mean=0.75).finalize(start).loop_probabilities,
                       np.repeat(((stay.sum(1) + 7.5) / (total.sum(1) + 10.0))[:, None], 2, axis=1)))
    _check("... or at the epoch's global estimate",
           np.allclose(fresh(prior_weight=10.0, prior_mean="global").finalize(start).loop_probabilities,
                       np.repeat(((stay.sum(1) + 10.0 * p_global) / (total.sum(1) + 10.0))[:, None],
                                 2, axis=1)))
    no_b = whole.counts.copy()
    no_b[2] = 0.0
    empty = TopologyAccumulator("phoneme", prior_weight=10.0, prior_mean=0.75)
    empty.observe(no_b)
    unprimed = TopologyAccumulator("phoneme")
    unprimed.observe(no_b)
    _check("... which an empty group takes, where plain EM keeps the previous value",
           np.allclose(empty.finalize(start).loop_probabilities[2], 0.75)
           and np.array_equal(unprimed.finalize(start).loop_probabilities[2], [0.5, 0.5]))
    forbid = StationaryTopology(np.array([[0.0, 0.0], [0.5, 0.5], [0.5, 0.5]]), symbols)
    blocked = whole.counts.copy()
    blocked[0, :, 0] = 0.0  # a forbidden loop is never taken
    acc = TopologyAccumulator("global")
    acc.observe(blocked)
    got = acc.finalize(forbid).loop_probabilities
    _check("TopologyAccumulator: a forbidden loop stays forbidden, and its frames stay out of the group",
           (got[0] == 0.0).all() and np.allclose(got[1:], blocked[1:, :, 0].sum() / blocked[1:].sum()))
    _check("TopologyAccumulator: bad settings are refused",
           all(_raises(lambda kw=kw: TopologyAccumulator(**kw), ValueError) for kw in [
               dict(tying="label"), dict(min_frames=-1.0), dict(prior_weight=1.0),
               dict(prior_weight=1.0, prior_mean=1.0), dict(prior_mean=0.5),
               dict(prior_weight=1.0, prior_mean=0.5, min_frames=10.0)]))
    _check("TopologyAccumulator: counts of the wrong shape, and an update from nothing, are refused",
           _raises(lambda: fresh().observe(np.zeros((2, 2, 2))), ValueError)
           and _raises(lambda: TopologyAccumulator().observe(np.zeros((3, 2))), ValueError)
           and _raises(lambda: TopologyAccumulator().observe(-np.ones((3, 2, 2))), ValueError)
           and _raises(lambda: TopologyAccumulator().finalize(start), RuntimeError))
    acc = fresh()
    updated = acc.finalize(start)
    info = acc.summary(start, updated, 0.35)
    _check("summary: the global estimate, its lambda, the mean duration and every group",
           np.isclose(info["topology_p_global"], p_global)
           and np.isclose(info["topology_lambda_global"], 0.35 * np.log(p_global / (1 - p_global)))
           and np.isclose(info["topology_mean_duration"],
                          whole.counts.sum() / whole.counts[:, -1, 1].sum())
           and set(info["topology_groups"]) == set(symbols)
           and np.isclose(info["topology_groups"]["A"]["p"], updated.loop_probabilities[1, 0])
           and info["topology_groups"]["A"]["p_previous"] == 0.5
           and info["topology_groups"]["A"]["updated"]
           and json.loads(json.dumps(info))["topology_tying"] == "phoneme")

    # The recognizer and the epoch on a real (tiny) LM, through the op's CPU
    # reference.
    with tempfile.TemporaryDirectory() as tmp:
        arpa = os.path.join(tmp, "lm.arpa")
        random_lm(3, 3, density=0.7, seed=3).to_arpa(arpa, symbols)
        lex = os.path.join(tmp, "lexicon.xml")
        with open(lex, "w") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n<lexicon>\n'
                    '<lemma><orth>[SILENCE]</orth><synt><tok>&lt;SIL&gt;</tok></synt></lemma>\n'
                    '<lemma><orth>A</orth></lemma>\n<lemma><orth>B</orth></lemma>\n'
                    '<lemma special="sentence-begin"><orth>[b]</orth><synt><tok>&lt;s&gt;</tok></synt></lemma>\n'
                    '<lemma special="sentence-end"><orth>[e]</orth><synt><tok>&lt;/s&gt;</tok></synt></lemma>\n'
                    '</lexicon>\n')
        op = dict(backoff_fb_root=fb_root, arpa_path=arpa, lexicon_path=lex, num_clusters=3,
                  distance_scale=0.3, device="cpu", batch_size=2, num_states=2)
        st_rng = np.random.RandomState(13)
        feats = {f"utt{i}": st_rng.randn(n, 4) for i, n in enumerate([6, 9, 4, 12, 7])}
        model = EuclideanModel(st_rng.randn(3, 4))
        scores = {t: model.scores(x) for t, x in feats.items()}
        base = dict(loop_probability=0.6, silence_loop_probability=0.3, transition_scale=0.35)

        def collect(**kw):
            out = {}
            rec = BackoffFBRecognizer(**op, **kw)
            rec.start(lambda r: out.__setitem__(r.seq_tag, r))
            for tag in feats:
                rec.submit(tag, scores[tag])
            rec.drain()
            rec.shutdown()
            return out

        def total_logz(**kw):
            return sum(r.sequence_score for r in collect(**kw).values())

        off = collect(**base)
        on = collect(**base, transition_counts=True)
        _check("counts: asking for them changes no posterior and no log Z, bitwise",
               all(np.array_equal(off[t].posteriors, on[t].posteriors)
                   and off[t].sequence_score == on[t].sequence_score for t in feats))
        _check("counts: [V, S, 2] per sequence when asked for, None otherwise",
               all(on[t].transition_counts.shape == (3, 2, 2)
                   and off[t].transition_counts is None for t in feats))
        occ = max(abs(on[t].transition_counts.sum() - len(feats[t])) for t in feats)
        lab = max(np.abs(on[t].transition_counts.sum(axis=(1, 2)) - on[t].posteriors.sum(0)).max()
                  for t in feats)
        flow = max(np.abs(on[t].transition_counts[:, 0, 1] - on[t].transition_counts[:, 1, 1]).max()
                   for t in feats)
        _check("counts: occupancy (to T, and per label to its Gamma) and flow (one move per state)",
               occ < 1e-9 and lab < 1e-9 and flow < 1e-9, f"{occ:.1e} {lab:.1e} {flow:.1e}")

        tabled = collect(**base, loop_table=table.loop_probabilities, transition_counts=True)
        _check("loop_table filled from the scalars: posteriors, log Z and counts bitwise the scalars'",
               all(np.array_equal(on[t].posteriors, tabled[t].posteriors)
                   and on[t].sequence_score == tabled[t].sequence_score
                   and np.array_equal(on[t].transition_counts, tabled[t].transition_counts)
                   for t in feats))
        rec = BackoffFBRecognizer(**op, **base)
        _check("recognizer: topology() is the table its scalars fill, in the lexicon's order",
               np.array_equal(rec.topology().loop_probabilities, table.loop_probabilities)
               and rec.symbols == symbols)
        _check("recognizer: a table of the wrong shape and an unknown offset symbol are refused",
               _raises(lambda: collect(**base, loop_table=per_state[:2]), ValueError)
               and _raises(lambda: collect(**base, exit_offset={"C": 1.0}), ValueError)
               and _raises(lambda: BackoffFBRecognizer(**op, zero_topology=True, exit_offset=1.0),
                           ValueError))

        # The counts are d log Z / d tau (the op's SPEC.md 15.2). With tau_loop
        # = s ln p and tau_fwd = tau_exit = s ln(1 - p), the chain rule gives
        # d log Z / d p = s * sum (stay / p - move / (1 - p)) over the pairs p
        # sets. Central differences through the table, and through the offset.
        c = sum(on[t].transition_counts for t in feats)
        s, eps = 0.35, 1e-5
        plus, minus = table.loop_probabilities.copy(), table.loop_probabilities.copy()
        plus[1] += eps
        minus[1] -= eps
        fd = (total_logz(**base, loop_table=plus) - total_logz(**base, loop_table=minus)) / (2 * eps)
        p = table.loop_probabilities[1, 0]
        analytic = s * (c[1, :, 0].sum() / p - c[1, :, 1].sum() / (1 - p))
        _check("counts: d log Z / d p_A by central differences through the loop table",
               abs(fd - analytic) < 1e-6 * max(1.0, abs(analytic)), f"{fd:.9f} vs {analytic:.9f}")
        fd_exit = (total_logz(**base, exit_offset={"A": eps})
                   - total_logz(**base, exit_offset={"A": -eps})) / (2 * eps)
        _check("counts: d log Z / d exit_offset[A] is A's exit count",
               abs(fd_exit - c[1, -1, 1]) < 1e-6 * max(1.0, c[1, -1, 1]),
               f"{fd_exit:.9f} vs {c[1, -1, 1]:.9f}")
        flat_offset = collect(**base, exit_offset=-0.25)
        dict_offset = collect(**base, exit_offset={sym: -0.25 for sym in symbols})
        _check("recognizer: a float exit offset is the same offset for every symbol",
               all(flat_offset[t].sequence_score == dict_offset[t].sequence_score for t in feats)
               and any(flat_offset[t].sequence_score != off[t].sequence_score for t in feats))

        worst, ok = np.inf, True
        for s in (1.0, 0.35):
            loops = dict(loop_probability=0.6, silence_loop_probability=0.3, transition_scale=s)
            first = collect(**loops, transition_counts=True)
            before = sum(r.sequence_score for r in first.values())
            for tying in TYINGS:
                acc = TopologyAccumulator(tying)
                for r in first.values():
                    acc.observe(r.transition_counts)
                new = acc.finalize(BackoffFBRecognizer(**op, **loops).topology())
                gain = total_logz(**loops, loop_table=new.loop_probabilities) - before
                worst = min(worst, gain)
                ok &= gain >= -1e-9
        _check("EM: one update never lowers the summed log Z, every tying, s = 1 and 0.35",
               ok, f"smallest gain {worst:.3e}")

        def chunk(recognizer, topology_accumulator, tags):
            return run_chunk(features=_ListSource([(t, feats[t]) for t in tags]), model=model,
                             recognizer=recognizer, accumulator=MeanAccumulator(3, 4),
                             topology_accumulator=topology_accumulator, verbosity=0)

        tags = sorted(feats)
        paths = []
        for i, part in enumerate((tags[:2], tags[2:])):
            result = chunk(BackoffFBRecognizer(**op, **base, transition_counts=True),
                           TopologyAccumulator("phoneme"), part)
            paths.append(os.path.join(tmp, f"chunk{i}.pkl"))
            save_chunk(result, paths[-1])
        merged = merge_chunks(chunk_paths=paths, accumulator_factory=lambda: MeanAccumulator(3, 4),
                              topology_accumulator_factory=lambda: TopologyAccumulator("phoneme"))
        _check("run_chunk + merge_chunks: the chunks' counts add up to the sequences' counts",
               np.allclose(merged.topology_accumulator.counts, c, rtol=1e-13, atol=0)
               and merged.topology_accumulator.num_seqs == len(feats))
        _check("run_chunk: a recognizer that does not count is refused",
               _raises(lambda: chunk(BackoffFBRecognizer(**op, **base), TopologyAccumulator(), tags),
                       ValueError))
        plain_chunk = chunk(BackoffFBRecognizer(**op, **base), None, tags)
        plain_path = os.path.join(tmp, "plain.pkl")
        save_chunk(plain_chunk, plain_path)
        _check("run_chunk: without a topology accumulator nothing is counted, and a merge that "
               "expects counts refuses the chunk",
               plain_chunk.topology_state is None
               and merge_chunks(chunk_paths=[plain_path],
                                accumulator_factory=lambda: MeanAccumulator(3, 4)
                                ).topology_accumulator is None
               and _raises(lambda: merge_chunks(
                   chunk_paths=[plain_path], accumulator_factory=lambda: MeanAccumulator(3, 4),
                   topology_accumulator_factory=TopologyAccumulator), RuntimeError))
        legacy = ChunkResult(accumulator_state={}, counter=None, num_seqs=0, num_frames=0,
                             num_recognized=0)
        del legacy.__dict__["topology_state"]
        legacy_path = os.path.join(tmp, "legacy.pkl")
        save_chunk(legacy, legacy_path)
        _check("ChunkResult: a pickle from before topology_state reads it as None",
               load_chunk(legacy_path).topology_state is None)

        # Two epochs through the epoch job's own tasks, with outputs redirected
        # into the temporary directory.
        import pickle as _pickle

        feats_pkl = os.path.join(tmp, "features.pkl")
        with open(feats_pkl, "wb") as f:
            _pickle.dump(feats, f)
        centroids = os.path.join(tmp, "centroids.npy")
        np.save(centroids, model.centroids)
        topology_spec = Spec(TopologyAccumulator, {"tying": "phoneme"})

        def epoch(name, topology):
            job = GuidedClusteringEpochJob(
                features=Spec(_PickledFeatures, {"path": feats_pkl}),
                model=Spec(EuclideanModel, {"centroids": centroids}),
                recognizer=Spec(BackoffFBRecognizer, {**op, **base}),
                accumulator=Spec(MeanAccumulator, {}),
                num_clusters=3, num_chunks=2, verbosity=0,
                topology_accumulator=topology_spec, topology=topology,
            )
            out = os.path.join(tmp, name)
            for attr, sub in (("out_model", "model"), ("out_statistics", "statistics.json"),
                              ("out_hypotheses", "hyp.txt.gz"), ("out_topology", "topology")):
                setattr(job, attr, tk.Path(os.path.join(out, sub)))
            work = os.path.join(out, "work")
            os.makedirs(work)
            cwd = os.getcwd()
            os.chdir(work)
            try:
                job.recognize(0)
                job.recognize(1)
                job.reduce()
            finally:
                os.chdir(cwd)
            with open(os.path.join(out, "statistics.json")) as f:
                stats = json.load(f)
            return StationaryTopology.load(os.path.join(out, "topology")), stats

        topo1, stats1 = epoch("epoch1", None)
        want1 = TopologyAccumulator("phoneme").load_state_dict(
            {"counts": c, "num_seqs": len(feats)}).finalize(table)
        _check("epoch job: reduce writes the M-step of the chunks' counts to out_topology",
               np.allclose(topo1.loop_probabilities, want1.loop_probabilities, rtol=1e-12)
               and stats1["topology_tying"] == "phoneme"
               and np.isclose(stats1["topology_frames"], sum(len(x) for x in feats.values()))
               and stats1["topology_groups"]["A"]["p_previous"] == 0.6)
        topo2, stats2 = epoch("epoch2", tk.Path(os.path.join(tmp, "epoch1", "topology")))
        second = collect(**base, loop_table=topo1.loop_probabilities, transition_counts=True)
        want2 = TopologyAccumulator("phoneme").load_state_dict(
            {"counts": sum(r.transition_counts for r in second.values()),
             "num_seqs": len(feats)}).finalize(topo1)
        _check("epoch job: the next epoch recognizes with that table and trains on from it",
               np.allclose(topo2.loop_probabilities, want2.loop_probabilities, rtol=1e-12)
               and all(np.isclose(stats2["topology_groups"][sym]["p_previous"],
                                  topo1.loop_probabilities[a, 0]) for a, sym in enumerate(symbols)))
        _check("epoch job: a topology input without an accumulator is refused",
               _raises(lambda: GuidedClusteringEpochJob(
                   features=Spec(_PickledFeatures, {"path": feats_pkl}),
                   model=Spec(EuclideanModel, {"centroids": centroids}),
                   recognizer=Spec(BackoffFBRecognizer, {**op, **base}),
                   accumulator=Spec(MeanAccumulator, {}), num_clusters=3,
                   topology=tk.Path(os.path.join(tmp, "epoch1", "topology"))), TypeError))

    # The graph: flavors and chunked_clustering.
    trained = BackoffFBSearch(**fb_common, loop_probability=0.5, num_states=2,
                              train_topology="global", topology_min_frames=1000.0)
    held = BackoffFBSearch(**fb_common, loop_probability=0.5, num_states=2)
    trained_flavor = euclidean_flavor(**flavor_common, backoff_fb=trained)
    held_flavor = euclidean_flavor(**flavor_common, backoff_fb=held)
    _check("flavor: train_topology adds a topology accumulator and leaves the recognizer spec alone",
           trained_flavor.topology_accumulator == Spec(TopologyAccumulator, {
               "tying": "global", "min_frames": 1000.0, "prior_weight": 0.0, "prior_mean": None})
           and held_flavor.topology_accumulator is None
           and plain_flavor.topology_accumulator is None
           and trained_flavor.recognizer == held_flavor.recognizer)
    offset_flavor = euclidean_flavor(**flavor_common,
                                     backoff_fb=BackoffFBSearch(**fb_common, exit_offset=-0.5))
    _check("flavor: an exit offset enters the recognizer spec only when set",
           offset_flavor.recognizer.kwargs.get("exit_offset") == -0.5
           and "exit_offset" not in plain_flavor.recognizer.kwargs)
    oracle = Spec(OracleBoundaryTransitions, {"frame_labels": tk.Path("/labels.pkl")})
    for what, kwargs in [
        ("next to frame_labels", dict(frame_labels=tk.Path("/x"))),
        ("next to per-frame transition scores", dict(transitions=oracle)),
        ("on the zero topology", dict(transitions=oracle, zero_topology=True, loop_probability=0.0)),
        ("from a loop probability of 0", dict(loop_probability=0.0)),
        ("with an unknown tying", dict(train_topology="label")),
        ("with a prior and a floor", dict(topology_prior_weight=10.0, topology_prior_mean=0.5,
                                         topology_min_frames=1.0)),
    ]:
        kw = {"loop_probability": 0.5, "train_topology": "global", **kwargs}
        _check(f"flavor: train_topology {what} is refused",
               _raises(lambda: euclidean_flavor(**flavor_common, backoff_fb=BackoffFBSearch(
                   **fb_common, **kw)), ValueError))
    _check("flavor: topology settings without train_topology are refused",
           _raises(lambda: euclidean_flavor(**flavor_common, backoff_fb=BackoffFBSearch(
               **fb_common, topology_min_frames=10.0)), ValueError))

    run3 = chunked_clustering(num_epochs=3, flavor=trained_flavor, **tr_common)
    held3 = chunked_clustering(num_epochs=3, flavor=held_flavor, **tr_common)
    _check("chunked_clustering: every epoch trains, each from the table the one before wrote",
           run3.jobs[0].topology is None
           and all(b.topology == a.out_topology for a, b in zip(run3.jobs, run3.jobs[1:]))
           and all(j.topology_accumulator == trained_flavor.topology_accumulator for j in run3.jobs)
           and run3.out_topologies == {e + 1: j.out_topology for e, j in enumerate(run3.jobs)})
    _check("chunked_clustering: training enters the epoch job hash, holding the topology fixed "
           "adds nothing",
           all(a._sis_id() != b._sis_id() for a, b in zip(run3.jobs, held3.jobs))
           and held3.out_topologies is None and all(j.out_topology is None for j in held3.jobs)
           and plain_run.out_topologies is None)
    head = chunked_clustering(num_epochs=2, flavor=trained_flavor, **tr_common)
    tail = chunked_clustering(
        num_epochs=1, initial_topology=head.out_topologies[2], **tr_common,
        flavor=euclidean_flavor(**{**flavor_common, "centroids": head.out_centroids[2]},
                                backoff_fb=trained))
    _check("chunked_clustering: 2 + 1 continued with initial_topology == 3 in one call",
           [j._sis_id() for j in head.jobs + tail.jobs] == [j._sis_id() for j in run3.jobs]
           and tail.out_topologies[0] == head.out_topologies[2])
    segments = tk.Path("/segments.txt")
    _check("chunked_clustering: a topology-training run refuses an EMA and batched updates",
           _raises(lambda: chunked_clustering(num_epochs=1, flavor=trained_flavor,
                                              ema=EMAConfig(0.5), batches_per_epoch=2,
                                              segments=segments, **tr_common), NotImplementedError)
           and _raises(lambda: chunked_clustering(num_epochs=1, flavor=trained_flavor,
                                                  batches_per_epoch=2, segments=segments,
                                                  **tr_common), NotImplementedError))
    _check("chunked_clustering: an initial topology needs a run that trains one",
           _raises(lambda: chunked_clustering(num_epochs=1, flavor=held_flavor,
                                              initial_topology=tk.Path("/t"), **tr_common),
                   TypeError))


def _label_covariance_checks():
    """One covariance per label over a shared codebook,
    ``p(x | l) = sum_k w_lk N(x | mu_k, Sigma_l)``: the model, its ECM update,
    the semi-tied transform, and the wiring."""
    import os
    import tempfile

    from sisyphus import tk

    from .accumulators import LabelCovarianceAccumulator, semi_tied_transform
    from .flavors import label_covariance_flavor
    from .models import LabelCovarianceMixtureModel

    rng = np.random.RandomState(7)
    L, K, D = 3, 8, 6

    def random_spd(scale):
        a = rng.randn(D, D)
        return scale * (a @ a.T / D + 0.5 * np.eye(D))

    def brute_costs(x, means, covs):
        """float64 ``0.5 (x - mu_k)^T Sigma_l^-1 (x - mu_k) + 0.5 log det Sigma_l``."""
        out = np.empty((len(x), len(covs), len(means)))
        for l, cov in enumerate(covs):
            diff = x[:, None, :] - means[None]
            out[:, l] = 0.5 * np.einsum("tkd,de,tke->tk", diff, np.linalg.inv(cov), diff)
            out[:, l] += 0.5 * np.linalg.slogdet(cov)[1]
        return out

    print("LabelCovarianceMixtureModel")
    means = rng.randn(K, D) * 2.0
    covs = np.stack([random_spd(s) for s in (0.5, 1.0, 2.0)])
    weights = rng.dirichlet(np.ones(K), size=L)
    model = LabelCovarianceMixtureModel(means, covs, weights, device="cpu")
    x = rng.randn(50, D) * 2.0

    ref = brute_costs(x, means, covs)
    err = np.abs(model.density_costs(x) - ref).max()
    _check("density costs == the float64 brute force", err < 1e-3, f"max err {err:.2e} (float32)")
    ref_scores = -np.log((weights[np.newaxis] * np.exp(-ref)).sum(axis=2))
    err = np.abs(model.scores(x) - ref_scores).max()
    _check("scores == -log sum_k w_lk N(x | mu_k, Sigma_l)", err < 1e-3, f"max err {err:.2e}")

    shared = random_spd(1.0)
    tied = LabelCovarianceMixtureModel(means, np.repeat(shared[np.newaxis], L, 0), weights, device="cpu")
    gmm = GaussianMixtureModel(means, np.repeat(shared[np.newaxis], K, 0), weights, device="cpu")
    gam = rng.dirichlet(np.ones(L), size=len(x))
    (dg_t, jc_t), (dg_g, jc_g) = tied.responsibilities(x, gam), gmm.responsibilities(x, gam)
    _check(
        "equal covariances: the shared-codebook GaussianMixtureModel exactly",
        np.abs(tied.scores(x) - gmm.scores(x)).max() < 1e-3
        and np.abs(dg_t - dg_g).max() < 1e-5 and np.abs(jc_t - jc_g).max() < 1e-5,
        f"score err {np.abs(tied.scores(x) - gmm.scores(x)).max():.2e}",
    )
    joint = model.joint_posteriors(x, gam)
    _check("the joint posterior marginalizes to the label posterior",
           np.abs(joint.sum(axis=2) - gam).max() < 1e-12)
    # The recognizers hand scores to librasr as a raw buffer, which it reads
    # row-major. A column-major [T, L] passes every numpy comparison above and
    # still scrambles frames and labels in the search - which is exactly how the
    # first label-covariance runs failed. So: every model class, by layout.
    pl_means = rng.randn(L * 2, D)
    layouts = {
        "LabelCovarianceMixtureModel": model.scores(x),
        "GaussianMixtureModel": gmm.scores(x),
        "PerLabelMixtureModel": PerLabelMixtureModel(
            pl_means, np.repeat(shared[np.newaxis], L * 2, 0),
            np.full((L, 2), 0.5), device="cpu").scores(x),
        "GaussianModel": GaussianModel(means, np.repeat(shared[np.newaxis], K, 0),
                                       device="cpu").scores(x),
        "EuclideanModel": EuclideanModel(means).scores(x),
        "VectorQuantizedModel": VectorQuantizedModel(means, weights).scores(x),
    }
    not_row_major = [name for name, s in layouts.items() if not s.flags["C_CONTIGUOUS"]]
    _check("every model's scores are row-major, as librasr reads them",
           not not_row_major, f"not row-major: {not_row_major}" if not_row_major else "")
    _check("... and so are the label-covariance model's density costs",
           model.density_costs(x).flags["C_CONTIGUOUS"])
    _check("covs must be one per label",
           _raises(lambda: LabelCovarianceMixtureModel(means, covs[:2], weights), ValueError))
    _check("a non-positive-definite covariance fails at construction",
           _raises(lambda: LabelCovarianceMixtureModel(means, covs - 10 * np.eye(D), weights),
                   np.linalg.LinAlgError))
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        loaded = load_model(tmp)
        _check("save/load round trip, the class picked out of the manifest",
               type(loaded) is LabelCovarianceMixtureModel
               and np.abs(loaded.scores(x) - model.scores(x)).max() < 1e-9)
        _check("decodable through load_forward_model",
               np.abs(load_forward_model(tmp).forward(x) - model.scores(x)).max() < 1e-9)

    print("LabelCovarianceAccumulator")
    # Data from the model family itself: codeword from the label's weights,
    # then that codeword's mean under the label's own covariance.
    true_covs = np.stack([random_spd(s) for s in (0.3, 1.0, 3.0)])
    seqs, gams = [], []
    for _ in range(30):
        T = rng.randint(20, 60)
        labels = rng.randint(L, size=T)
        codewords = [rng.choice(K, p=weights[l]) for l in labels]
        seqs.append(np.stack([rng.multivariate_normal(means[k], true_covs[l])
                              for l, k in zip(labels, codewords)]))
        g = np.full((T, L), 0.05)
        g[np.arange(T), labels] = 0.9
        gams.append(g / g.sum(axis=1, keepdims=True))
    start = LabelCovarianceMixtureModel(
        means + 0.3 * rng.randn(K, D), np.repeat(2.0 * np.eye(D)[np.newaxis], L, 0),
        np.full((L, K), 1.0 / K), device="cpu",
    )

    def accumulate(model_, idx=range(30), gs=None, **cfg):
        acc = LabelCovarianceAccumulator(L, **cfg)
        acc.bind_model(model_)
        for i in idx:
            acc.observe(seqs[i], (gs or gams)[i])
        return acc

    def brute_stats(model_):
        n, F, S = np.zeros((L, K)), np.zeros((L, K, D)), np.zeros((L, D, D))
        for xs, g in zip(seqs, gams):
            j = model_.joint_posteriors(xs, g)
            n += j.sum(axis=0)
            F += np.einsum("tlk,td->lkd", j, xs)
            S += np.einsum("tl,td,te->lde", g, xs, xs)
        return n, F, S

    full_cfg = dict(update_means=True, covariance="full", covariance_relevance=5.0)
    whole = accumulate(start, **full_cfg)
    n, F, S = brute_stats(start)
    _check("statistics == the einsums over the joint posterior",
           np.abs(whole.weighted_c - n).max() < 1e-9 and np.abs(whole.first - F).max() < 1e-8
           and np.abs(whole.second - S).max() < 1e-7)
    merged = LabelCovarianceAccumulator(L, **full_cfg)
    for c in range(3):
        part = accumulate(start, range(c, 30, 3), **full_cfg)
        restored = LabelCovarianceAccumulator(L, **full_cfg)
        merged.merge(restored.load_state_dict(pickle.loads(pickle.dumps(part.state_dict()))))
    a, b = whole.finalize(start), merged.finalize(start)
    _check("merge is associative, through the state dict",
           all(np.abs(np.asarray(a.artifacts()[k]) - np.asarray(b.artifacts()[k])).max() < 1e-9
               for k in a.ARTIFACT_NAMES))
    _check("a state from another configuration is refused",
           _raises(lambda: LabelCovarianceAccumulator(L).load_state_dict(whole.state_dict()),
                   ValueError))

    m_only = accumulate(start, update_means=True).finalize(start)
    _check("equal covariances: the mean step is the weighted average",
           np.abs(m_only.centroids - F.sum(0) / n.sum(0)[:, np.newaxis]).max() < 1e-9)
    _check("a frozen parameter comes through as the previous model's own array",
           m_only.covs is start.covs and accumulate(start, covariance="full").finalize(start)
           .centroids is start.centroids)
    unequal = LabelCovarianceMixtureModel(start.centroids, true_covs, start.mixtures, device="cpu")
    n_u, F_u, _ = brute_stats(unequal)
    mu = accumulate(unequal, update_means=True).finalize(unequal).centroids
    gradient = np.einsum("lde,lke->kd", np.linalg.inv(true_covs), F_u - n_u[:, :, np.newaxis] * mu)
    _check("unequal covariances: the mean step zeroes the precision-weighted gradient",
           np.abs(gradient).max() < 1e-8, f"max {np.abs(gradient).max():.1e}")

    def scatter_about(mu_):
        out = np.zeros((L, D, D))
        for xs, g in zip(seqs, gams):
            diff = xs[:, None, :] - mu_[np.newaxis]
            out += np.einsum("tlk,tkd,tke->lde", start.joint_posteriors(xs, g), diff, diff)
        return out

    full = accumulate(start, update_means=True, covariance="full").finalize(start)
    W_ref = scatter_about(full.centroids) / n.sum(1)[:, np.newaxis, np.newaxis]
    _check("full, r=0: the scatter about the *new* means (ECM)",
           np.abs(full.covs - W_ref).max() < 1e-8)
    frozen_scatter = scatter_about(start.centroids)
    pooled = frozen_scatter.sum(0) / n.sum()
    tied_fit = accumulate(start, covariance="full", covariance_relevance=float("inf")).finalize(start)
    _check("r=inf: every label gets the mass-weighted pool",
           np.abs(tied_fit.covs - pooled[np.newaxis]).max() < 1e-9)
    r = 40.0
    smoothed = accumulate(start, covariance="full", covariance_relevance=r).finalize(start)
    expected = (frozen_scatter + r * pooled) / (n.sum(1) + r)[:, np.newaxis, np.newaxis]
    _check("relevance r: (n_l W_l + r W) / (n_l + r)", np.abs(smoothed.covs - expected).max() < 1e-9)

    # The auxiliary function every M-step maximizes, under the start model's
    # E-step. The start covariances are isotropic, so every mode - semi-tied and
    # diagonal included, which restart their transform from the pooled
    # eigenbasis - can represent them, and none may end lower.
    def auxiliary(theta):
        total = 0.0
        with np.errstate(divide="ignore"):
            log_w = np.log(theta.mixtures)
        for xs, g in zip(seqs, gams):
            total += (start.joint_posteriors(xs, g)
                      * (log_w[np.newaxis] - brute_costs(xs, theta.centroids, theta.covs))).sum()
        return total

    q_start = auxiliary(start)
    gains = {}
    for name, cfg in [
        ("weights", {}),
        ("weights + means", dict(update_means=True)),
        ("full", dict(covariance="full")),
        ("full + means", dict(update_means=True, covariance="full")),
        ("tied + means", dict(update_means=True, covariance="full", covariance_relevance=float("inf"))),
        ("semi-tied + means", dict(update_means=True, covariance="semi_tied", semi_tied_iterations=10)),
        ("diagonal", dict(covariance="diagonal")),
    ]:
        gains[name] = auxiliary(accumulate(start, **cfg).finalize(start)) - q_start
        _check(f"ECM does not lower the auxiliary function: {name}", gains[name] >= -1e-6,
               f"{gains[name]:+.1f}")
    _check("label covariances fit data generated with them better than one tied covariance",
           gains["full + means"] > gains["tied + means"],
           f"{gains['full + means']:.1f} vs {gains['tied + means']:.1f}")

    print("semi_tied_transform")
    H = rng.randn(D, D)
    diagonals = rng.uniform(0.2, 3.0, size=(L, D))
    W = np.einsum("di,li,ei->lde", H, diagonals, H)
    counts = np.array([100.0, 300.0, 50.0])
    # Linear convergence: 2000 rounds to get within 1e-10 here (see the function).
    A, history = semi_tied_transform(W, counts, np.eye(D), 2000)
    from .accumulators import common_diagonalizer
    recovered = common_diagonalizer(W)
    _check("common_diagonalizer recovers a transform for semi-tied covariances, none otherwise",
           recovered is not None
           and abs(semi_tied_transform(W, counts, recovered, 0)[1][0] - (-0.5 * (counts @ np.linalg.slogdet(W)[1]) / counts.sum())) < 1e-8
           and common_diagonalizer(np.stack([random_spd(1.0) for _ in range(L)])) is None)
    bound = -0.5 * (counts @ np.linalg.slogdet(W)[1]) / counts.sum()
    _check("the objective never decreases", (np.diff(history) >= -1e-12).all())
    _check("jointly diagonalizable scatters: reaches the bound -1/2 sum (n_l/N) log det W_l",
           abs(history[-1] - bound) < 1e-6 and history[0] < bound,
           f"gap {bound - history[-1]:.1e} (from {bound - history[0]:.2f})")
    variances = np.einsum("id,lde,ie->li", A, W, A)
    rebuilt = np.einsum("di,li,ei->lde", np.linalg.inv(A), variances, np.linalg.inv(A))
    _check("... and A^-1 diag(A W_l A^T) A^-T rebuilds every W_l",
           np.abs(rebuilt - W).max() < 1e-4 * np.abs(W).max())
    zero = accumulate(start, covariance="semi_tied", semi_tied_iterations=0).finalize(start)
    diagonal = accumulate(start, covariance="diagonal").finalize(start)
    _check("semi-tied with no iterations is the diagonal mode",
           np.abs(zero.covs - diagonal.covs).max() < 1e-12)

    print("LabelCovarianceAccumulator - dead labels, weights only")
    starved = [g * np.array([1.0, 1.0, 0.0]) / g[:, :2].sum(axis=1, keepdims=True) for g in gams]
    dead = accumulate(start, gs=starved, update_means=True, covariance="full",
                      covariance_relevance=1.0).finalize(start)
    _check("a label with no mass keeps its weights and its covariance",
           np.array_equal(dead.mixtures[2], start.mixtures[2])
           and np.array_equal(dead.covs[2], start.covs[2]))
    gmm_start = GaussianMixtureModel(
        start.centroids, np.repeat(start.covs[:1], K, 0), start.mixtures, device="cpu"
    )
    mix_acc = MixtureGaussianAccumulator(L, update_densities=False, mixture_floor=1e-2)
    mix_acc.bind_model(gmm_start)
    for xs, g in zip(seqs, gams):
        mix_acc.observe(xs, g)
    weights_only = accumulate(start, mixture_floor=1e-2)
    _check("weights only allocates no moments",
           weights_only.first is None and weights_only.second is None)
    err = np.abs(weights_only.finalize(start).mixtures - mix_acc.finalize(gmm_start).mixtures).max()
    _check("weights only, equal covariances: == MixtureGaussianAccumulator(update_densities=False)",
           err < 1e-6, f"max err {err:.1e}")

    print("label_covariance_flavor")
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for c in range(3):
            idx = range(c, 30, 3)
            result = run_chunk(
                features=_ListSource([(f"s{i}", seqs[i]) for i in idx]),
                model=start,
                recognizer=_StubFBRecognizer({f"s{i}": gams[i] for i in idx}),
                accumulator=LabelCovarianceAccumulator(L, **full_cfg),
                counter=None,
                verbosity=0,
            )
            paths.append(os.path.join(tmp, f"chunk.{c}.pkl"))
            save_chunk(result, paths[-1])
        reduced, _s, totals, _h = reduce_chunks(
            chunk_paths=paths,
            accumulator_factory=lambda: LabelCovarianceAccumulator(L, **full_cfg),
            previous_model=start,
        )
    _check("run_chunk + reduce_chunks == one accumulator over everything",
           np.abs(reduced.covs - a.covs).max() < 1e-9
           and np.abs(reduced.centroids - a.centroids).max() < 1e-9
           and totals["num_seqs"] == 30)

    from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
        chunked_clustering,
    )
    common = dict(
        features_hdf=tk.Path("/f.hdf"),
        recognition_config=tk.Path("/r.config"),
        lexicon=tk.Path("/l.gz"),
        num_clusters=40,
    )
    flavor_kwargs = dict(
        centroids=tk.Path("/init/c.npy"),
        covs=tk.Path("/init/label_covs.npy"),
        mixtures=tk.Path("/init/mix.npy"),
        recognition_config=common["recognition_config"],
        lexicon=common["lexicon"],
        num_clusters=common["num_clusters"],
        update_means=True,
        covariance="full",
        covariance_relevance=2048.0,
    )
    flavor = label_covariance_flavor(**flavor_kwargs)
    _check("the flavor names the model and accumulator, and the same artifacts",
           flavor.model.cls is LabelCovarianceMixtureModel
           and flavor.accumulator.cls is LabelCovarianceAccumulator
           and flavor.artifact_names == ("centroids", "covs", "mixtures"))
    _check("an unknown covariance mode fails while the graph is built",
           _raises(lambda: label_covariance_flavor(**{**flavor_kwargs, "covariance": "spherical"}),
                   ValueError))
    _check("semi_tied_iterations enters the spec only where it is read",
           "semi_tied_iterations" not in flavor.accumulator.kwargs
           and label_covariance_flavor(**{**flavor_kwargs, "covariance": "semi_tied"})
           .accumulator.kwargs["semi_tied_iterations"] == 20)
    ten = chunked_clustering(num_epochs=10, flavor=label_covariance_flavor(**flavor_kwargs), **common)
    head = chunked_clustering(num_epochs=5, flavor=label_covariance_flavor(**flavor_kwargs), **common)
    tail = chunked_clustering(
        num_epochs=5,
        flavor=label_covariance_flavor(**{
            **flavor_kwargs,
            **{name: head.out_artifacts[name][5] for name in ("centroids", "covs", "mixtures")},
        }),
        **common,
    )
    _check("5+5 continued == 10",
           [j._sis_id() for j in ten.jobs] == [j._sis_id() for j in head.jobs + tail.jobs])
    tied_flavor = label_covariance_flavor(**{**flavor_kwargs, "covariance_relevance": float("inf")})
    _check("r=inf hashes, and differently from a finite r",
           chunked_clustering(num_epochs=1, flavor=tied_flavor, **common).jobs[-1]._sis_id()
           != chunked_clustering(num_epochs=1, flavor=flavor, **common).jobs[-1]._sis_id())


if __name__ == "__main__":
    sys.exit(main())
