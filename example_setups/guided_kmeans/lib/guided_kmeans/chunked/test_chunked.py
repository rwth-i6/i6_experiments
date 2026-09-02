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

import sys

import numpy as np

from .accumulators import (
    FixedCovarianceAccumulator,
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
    EuclideanModel,
    GaussianMixtureModel,
    GaussianModel,
    PerLabelMixtureModel,
    load_forward_model,
    load_model,
    neg_log_matmul,
    read_manifest,
)
from .runner import reduce_chunks, run_chunk, save_chunk
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


if __name__ == "__main__":
    sys.exit(main())
