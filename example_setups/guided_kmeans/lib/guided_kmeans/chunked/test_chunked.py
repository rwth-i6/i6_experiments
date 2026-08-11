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
