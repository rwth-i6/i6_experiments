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

from .accumulators import GaussianAccumulator, MeanAccumulator, keep_previous_where_dead
from .recognizers import RasrFBRecognizer
from .features import plan_chunks
from .models import ArtifactModel, EuclideanModel, GaussianModel, load_model, read_manifest
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
            self._cb(seq_tag, self.gammas_table[seq_tag], [])
        self._buffer = []

    def shutdown(self):
        pass


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
            self._cb(seq_tag, self.table[seq_tag], self.tracebacks.get(seq_tag, []))
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
    fb_rec._on_result = lambda seq_tag, posteriors, tb: delivered.append((seq_tag, posteriors, tb))

    T = 15
    raw_gammas = rng.exponential(1.0, size=(T, num_clusters + 3)).astype(np.float64)
    seq_tag_fb = "test_fb_seq"
    fb_rec._handle(seq_tag_fb, raw_gammas, log_likelihood=-42.0)
    assert len(delivered) == 1
    out_tag, out_gammas, out_tb = delivered[0]
    _check("_handle: seq_tag passed through", out_tag == seq_tag_fb)
    _check("_handle: extra RASR label columns stripped", out_gammas.shape == (T, num_clusters))
    _check("_handle: rows sum to 1 after normalization",
           np.allclose(out_gammas.sum(axis=1), 1.0, atol=1e-12))
    _check("_handle: empty traceback passed through", out_tb == [])

    # All-zero row (degenerate sequence) must produce a zero row, not nan.
    zero_row = np.zeros((T, num_clusters + 3), dtype=np.float64)
    delivered_zero = []
    fb_rec._on_result = lambda seq_tag, posteriors, tb: delivered_zero.append(posteriors)
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
    carried = keep_previous_where_dead(
        _DiagModel(
            np.zeros((num_clusters, dim)), np.zeros((num_clusters, dim)), np.zeros(num_clusters)
        ),
        diag,
        dead,
    )
    _check(
        "dead-cluster carry-over is generic over artifacts",
        np.array_equal(carried.variances[dead], diag.variances[dead])
        and np.array_equal(carried.priors[dead], diag.priors[dead])
        and np.array_equal(carried.priors[~dead], np.zeros(num_clusters - 2)),
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
