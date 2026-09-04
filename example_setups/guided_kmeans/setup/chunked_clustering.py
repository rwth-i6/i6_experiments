"""
Sisyphus wiring for chunked guided k-means.

One job per epoch. Each job runs ``num_chunks`` independent ``recognize``
tasks (the expensive RASR search, spread over the cluster) followed by a
single ``reduce`` mini-task that merges their sufficient statistics into the
next model. Epochs are chained by model directory, so every epoch's model is a
first-class job output that downstream decoding can consume as soon as that
epoch finishes - rather than after the whole run, as with the monolithic
forward job.

The existing ``clustering()`` pipeline in :mod:`.clustering_config` is
untouched and keeps working; migrating a config means swapping the call.
"""

from __future__ import annotations

__all__ = [
    "ClusteringFlavor",
    "GuidedClusteringEpochJob",
    "IdentityCovsJob",
    "MaterializeModelJob",
    "MergeEpochStatisticsJob",
    "RandomCentroidsJob",
    "DuplicateCovsJob",
    "GlobalCovarianceJob",
    "ClusterCovarianceJob",
    "RandomMixturesJob",
    "NormalTableJob",
    "RepeatCovsJob",
    "SelectCovJob",
    "SplitCentroidsJob",
    "UniformMixturesJob",
    "ChunkedClusteringExpResult",
    "chunked_clustering",
    "euclidean_flavor",
    "gaussian_flavor",
    "mixture_flavor",
    "per_label_mixture_flavor",
    "unguided_flavor",
    "vq_flavor",
    "prepare_worker_sys_path",
]

import gzip
import json
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import i6_experiments
from typing import Any, Dict, List, Optional, Sequence, Union

from sisyphus import Job, Task, tk

import numpy as np

from .array_job import ArrayJob
from .score import JiwerScoringJob, ScoreResult
from ..lib.guided_kmeans.util import DEFAULT_EXCLUDED_LEMMATA, ProgressLogger, traceback_to_text
from ..lib.guided_kmeans.chunked import (
    ClusteringFlavor,
    EuclideanModel,
    GaussianModel,
    HDFFeatureSource,
    per_label_mixture_flavor,
    Spec,
    default_stats_hooks,
    euclidean_flavor,
    gaussian_flavor,
    mixture_flavor,
    unguided_flavor,
    vq_flavor,
    reduce_chunks,
    run_chunk,
    save_chunk,
)

_CHUNK_FILE = "chunk.{num_chunks}.{index}.pkl"


def prepare_worker_sys_path(rasr_path: Optional[tk.Path] = None) -> None:
    """
    Make ``i6_experiments`` and ``librasr`` importable in the recognition
    worker processes.

    ParallelSegmentRecognizer's pool uses the "spawn" start method (see that
    module - "fork" is unsafe with a live CUDA context). A spawned child is a
    fresh interpreter that receives the parent's ``sys.path`` via
    multiprocessing's preparation data, but *not* the parent's
    ``sys.meta_path``. Sisyphus resolves recipe modules through a custom
    meta-path finder (loader.RecipeFinder) rather than through sys.path, so
    without this the child cannot import the module holding ``_init_worker``
    and dies with ModuleNotFoundError: i6_experiments before it ever runs a
    search.

    The old pipeline got this for free: its RETURNN config prolog does the same
    absolute ``sys.path.insert`` (see clustering_config.get_base_config).

    Module-level rather than a method because every job that drives a RASR
    worker pool needs it, not just the epoch job.
    """
    recipe_root = str(Path(i6_experiments.__file__).parent.parent)
    for path in (recipe_root, rasr_path.get_path() if rasr_path else None):
        if path and path not in sys.path:
            sys.path.insert(0, path)


class GuidedClusteringEpochJob(Job):
    """
    One guided k-means epoch: recognize the corpus with the current model,
    accumulate sufficient statistics, produce the next model.

    :param features: spec for a FeatureSource; built per task with
        ``chunk``/``num_chunks`` injected
    :param model: spec for the ScoreModel this epoch recognizes with
    :param recognizer: spec for the Recognizer
    :param accumulator: spec for the Accumulator; built with ``num_clusters``
        injected
    :param lexicon: lexicon defining the label inventory, used for the default
        statistics hooks
    :param num_chunks: how many parallel tasks the epoch is split into.
        UNHASHED - the accumulators' merge is associative, so the partition
        cannot change the result; it is purely a scheduling knob.
    :param statistics: optional spec for a statistics counter. UNHASHED, and
        deliberately not exposed by :func:`chunked_clustering` - which
        diagnostics get recorded must not change a job's identity.
    :param exclude_lemmata: lemmata dropped when writing ``out_hypotheses``;
        silence by default, because the references those hypotheses get scored
        against do not contain it. Hash-excluded at its default value, so
        adding this parameter left every existing job's hash untouched - but
        that only holds for the default *tuple*: passing the equivalent list
        changes the hash.
    """

    __sis_hash_exclude__ = {"exclude_lemmata": DEFAULT_EXCLUDED_LEMMATA}

    def __init__(
        self,
        *,
        features: Spec,
        model: Spec,
        recognizer: Spec,
        accumulator: Spec,
        num_clusters: int,
        lexicon: Optional[tk.Path] = None,
        rasr_path: Optional[tk.Path] = None,
        num_chunks: int = 30,
        statistics: Optional[Spec] = None,
        exclude_lemmata: Sequence[str] = DEFAULT_EXCLUDED_LEMMATA,
        verbosity: int = 1,
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        self.features = features
        self.model = model
        self.recognizer = recognizer
        self.accumulator = accumulator
        self.num_clusters = num_clusters
        self.lexicon = lexicon
        self.rasr_path = rasr_path
        self.num_chunks = num_chunks
        self.statistics = statistics
        self.exclude_lemmata = tuple(exclude_lemmata)
        self.verbosity = verbosity

        self.rqmt = {"cpu": 9, "mem": 16, "time": 4}
        if rqmt:
            self.rqmt.update(rqmt)

        # The whole model is one directory output; which files land in it is
        # the model class's business (see chunked.models: artifacts + manifest),
        # so a model with a different parameter set needs no change here.
        # Reach individual artifacts with self.artifact("centroids"), following
        # the i6_core convention of `some_dir_path.join_right("file")`.
        self.out_model = self.output_path("model", directory=True)
        self.out_statistics = self.output_path("statistics.json")
        # The epoch's recognition of the whole corpus, in the format
        # ClusteringDecodeCallback writes, so a JiwerScoringJob can score it
        # against a TaggedCorpusToTxtJob reference exactly like a decode's.
        # Written unconditionally: the search that produces it dominates epoch
        # wall time and has already run, so a run that turns out to want PER
        # should not have to repeat it. Gzipped - ~17 MB of text per epoch for
        # ls-100 compresses to a few MB, and nothing reads it hot.
        #
        # NB these hypotheses come from recognizing with the model this epoch
        # *started* from, i.e. from `model`, not from the model it produces.
        self.out_hypotheses = self.output_path("hyp.txt.gz")

    @classmethod
    def hash(cls, kwargs):
        """
        Exclude everything that only affects *how* the epoch is executed, not
        what it computes. ``num_chunks`` belongs in this list because merging
        is associative; if that ever stops holding, it has to move back into
        the hash. Specs additionally drop their own unhashed arguments (worker
        counts, timeouts).
        """
        unhashed = {"num_chunks", "statistics", "verbosity", "rqmt"}
        return super().hash(
            {
                k: (v.hashed() if isinstance(v, Spec) else v)
                for k, v in kwargs.items()
                if k not in unhashed
            }
        )

    def artifact(self, name: str) -> tk.Path:
        """
        Path to one of the produced model's artifacts, e.g. ``"centroids"``.

        Which artifacts exist is decided by the model class, not by this job,
        so these are derived from the model directory rather than declared as
        separate outputs - the same shape as ``MakeJob``'s output repository or
        ``returnn_root.join_right("rnn.py")`` in i6_core. Dependency tracking
        is unaffected: the creator is still this job.
        """
        return self.out_model.join_right(f"{name}.npy")

    def tasks(self):
        yield Task(
            "recognize",
            resume="recognize",
            args=range(self.num_chunks),
            rqmt=self.rqmt,
            parallel=self.num_chunks,
        )
        yield Task("reduce", mini_task=True)

    def _chunk_path(self, index: int) -> str:
        # Relative to the task's cwd, which sisyphus sets to the job's work/
        # directory (Task.run -> execute_in_dir(JOB_WORK_DIR)). That directory
        # is removed when the job finishes (JOB_AUTO_CLEANUP = True,
        # JOB_CLEANUP_KEEP_WORK = False), so per-chunk statistics - up to
        # ~84 MB each for the full-covariance model - never accumulate across
        # epochs. num_chunks is part of the filename because it is unhashed:
        # re-running with a different value must not silently mix results from
        # two different partitions of the corpus.
        return _CHUNK_FILE.format(num_chunks=self.num_chunks, index=index)

    def _prepare_worker_sys_path(self):
        prepare_worker_sys_path(self.rasr_path)

    def _build_accumulator(self):
        return self.accumulator.build(num_clusters=self.num_clusters)

    def _build_statistics(self):
        if self.statistics is not None:
            return self.statistics.build()
        if self.lexicon is None:
            return None
        from ..lib.guided_kmeans.chunked.recognizers import PhonemeIdxMap

        return default_stats_hooks(PhonemeIdxMap(self.lexicon.get_path()))

    def recognize(self, index: int):
        # A recognition failure raises RecognizerAborted out of run_chunk, so
        # sisyphus' own Task.run error handling records the task as failed.
        # Nothing to do here beyond letting it propagate.
        self._prepare_worker_sys_path()

        model = self.model.build()
        result = run_chunk(
            features=self.features.build(chunk=index, num_chunks=self.num_chunks),
            model=model,
            recognizer=self.recognizer.build(),
            accumulator=self._build_accumulator(),
            counter=self._build_statistics(),
            transcribe=partial(traceback_to_text, exclude_lemmata=self.exclude_lemmata),
            verbosity=self.verbosity,
        )
        save_chunk(result, self._chunk_path(index))

    def reduce(self):
        chunk_paths = [self._chunk_path(i) for i in range(self.num_chunks)]
        missing = [p for p in chunk_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} chunk result(s) missing, e.g. {missing[:3]}. "
                f"If num_chunks was changed after some chunks ran, delete "
                f"{os.getcwd()} (the job's work/ directory) and rerun."
            )

        model, statistics, totals, hypotheses = reduce_chunks(
            chunk_paths=chunk_paths,
            accumulator_factory=self._build_accumulator,
            previous_model=self.model.build(),
        )

        # save() writes the manifest and verifies every artifact it lists was
        # actually written, so there is nothing model-specific to check here.
        model.save(self.out_model.get_path())

        # Sorted by tag, so the file is a function of the epoch alone: chunk
        # order would otherwise leak num_chunks, an unhashed knob, into the
        # content of a hashed output.
        with gzip.open(self.out_hypotheses.get_path(), "wt") as fp:
            for seq_tag in sorted(hypotheses):
                fp.write(f"{seq_tag}\t{hypotheses[seq_tag]}\n")

        statistics = dict(statistics)
        statistics.update(totals)
        with open(self.out_statistics.get_path(), "w") as fp:
            json.dump(statistics, fp, indent=4, default=str)

        print(f"Epoch done: {totals}")


class MergeEpochStatisticsJob(Job):
    """
    Combine the per-epoch statistics files into the single
    ``{epoch: {...}}`` document the configs already register as an output,
    matching what ``EpochwiseStatisticsLogger`` wrote for the old pipeline.
    """

    def __init__(self, statistics: Dict[int, tk.Path]):
        self.statistics = statistics
        self.out_statistics = self.output_path("epoch_statistics.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        merged = {}
        for epoch, path in sorted(self.statistics.items()):
            with open(path.get_path()) as fp:
                merged[str(epoch)] = json.load(fp)
        with open(self.out_statistics.get_path(), "w") as fp:
            json.dump(merged, fp, indent=4)


class MaterializeModelJob(Job):
    """Write a model built from loose artifact files out as a model directory.

    Epoch 0 - whatever a run was initialized with - is the one model in a run
    that exists only as separate ``.npy`` files, because it was never produced
    by an epoch job. Everything that consumes a *model* rather than a pair of
    arrays therefore could not see it: notably decoding, since
    ``DecodeConfig(model_dir=...)`` is the only way to score with a model whose
    parameters are not exactly ``(centroids, covs)``.

    That gap matters more than it sounds. The initialization is the one point
    in a run whose quality is known in advance - for a cheating init it should
    already recognize well - so being unable to decode it removes the obvious
    way to tell a broken update from a broken model.

    Takes the model :class:`Spec` rather than the artifacts, so it works for
    any model class without being told which: the spec already knows how to
    build one, and ``save`` already knows how to write one.
    """

    def __init__(self, model: Spec, rqmt: Optional[Dict[str, Any]] = None):
        self.model = model
        self.out_model = self.output_path("model", directory=True)
        # Not a mini-task: constructing a covariance model inverts every
        # covariance eagerly, which at 120 densities and D=512 is a few hundred
        # megabytes and more than the local engine's default allowance.
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}
        if rqmt:
            self.rqmt.update(rqmt)

    @classmethod
    def hash(cls, kwargs):
        return super().hash({k: v for k, v in kwargs.items() if k != "rqmt"})

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        self.model.build().save(self.out_model.get_path())


class RandomCentroidsJob(ArrayJob):
    """Sample K random frames from a feature HDF as initial centroids.

    Reads only K rows from ``inputs`` (total-frames x feature-dim), so memory
    usage is O(K x feature-dim) regardless of corpus size.
    """

    OUTPUTS = ("centroids",)

    def __init__(self, features_hdf: tk.Path, num_clusters: int, seed: int = 42):
        self.features_hdf = features_hdf
        self.num_clusters = num_clusters
        self.seed = seed
        super().__init__()

    def compute(self):
        import h5py

        rng = np.random.RandomState(self.seed)
        with h5py.File(self.features_hdf.get_path(), "r") as f:
            total_frames = f["inputs"].shape[0]
            indices = np.sort(rng.choice(total_frames, size=self.num_clusters, replace=False))
            return f["inputs"][indices]


class IdentityCovsJob(ArrayJob):
    """Write K identity covariance matrices as the initial state for covariance models.

    With identity covariances the Mahalanobis distance degenerates to Euclidean,
    so the first epoch of a random-init covariance run behaves like plain k-means.
    Subsequent epochs learn cluster-specific covariances from the data.

    For a less arbitrary start, :class:`DuplicateCovsJob` tiles a real
    covariance - the corpus-wide one, say - across the same K slots.
    """

    OUTPUTS = ("covs",)

    def __init__(self, num_clusters: int, feature_dim: int = 512):
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        super().__init__()

    def compute(self):
        return np.stack([np.eye(self.feature_dim) for _ in range(self.num_clusters)])


class GlobalCovarianceJob(ArrayJob):
    """The covariance of every frame in the corpus, in one pass.

    ``out_cov`` is ``[D, D]`` and feeds :class:`DuplicateCovsJob` directly, so a
    run can start every density from the shape of the data rather than from the
    identity without any covariance having to exist beforehand - which is what
    makes a mixture config self-contained. ``out_mean`` comes along for free and
    is worth a look: the encoder's output is not centred, and how far the mean
    is from zero says something about what the distances are measuring.

    Reads through :class:`...lib.guided_kmeans.chunked.HDFFeatureSource` with a
    single chunk rather than over the raw ``inputs`` rows. That costs
    per-sequence reads instead of big block reads, and buys exactness: segment
    filtering and subsampling are applied the same way the clustering applies
    them, so the covariance describes the frames the model will actually see.
    Pass the same ``segments``/``subsampling``/``pooling_function`` the run
    uses, or the two disagree silently.

    One pass, O(D^2) memory. Accumulated about a shift - the first sequence's
    mean - rather than as raw second moments. The per-cluster accumulators can
    use raw moments because they sum over a cluster's frames at a scale where
    the cancellation is harmless (see SoftGaussianAccumulator); here the sum
    runs over every frame in the corpus, tens of millions of them, and shifting
    costs one subtraction to remove the question entirely.
    """

    __sis_hash_exclude__ = {"rqmt": None}

    OUTPUTS = ("cov", "mean")

    def __init__(
        self,
        features_hdf: Union[tk.Path, Sequence[tk.Path]],
        segments: Optional[tk.Path] = None,
        subsampling: Optional[int] = None,
        pooling_function: str = "maxpool_time_np",
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        self.features_hdf = features_hdf
        self.segments = segments
        self.subsampling = subsampling
        self.pooling_function = pooling_function
        super().__init__()

        # Dominated by reading the features: ~29 GB for ls-100 over a ~100 MB/s
        # filesystem, plus a [T, D] x [D, T] product per sequence. Memory is
        # one sequence plus the two D x D accumulators, i.e. megabytes.
        self.rqmt = {"cpu": 4, "mem": 8, "time": 4}
        if rqmt:
            self.rqmt.update(rqmt)

    def compute(self):
        files = (
            list(self.features_hdf)
            if isinstance(self.features_hdf, (list, tuple))
            else [self.features_hdf]
        )
        source = HDFFeatureSource(
            files=[f.get_path() for f in files],
            segments=self.segments.get_path() if self.segments else None,
            subsampling=self.subsampling,
            pooling_function=self.pooling_function,
        )

        num_frames = 0
        shift = None
        total = None            # sum of (x - shift)
        outer = None            # sum of (x - shift)(x - shift)^T
        progress = ProgressLogger(max(len(source), 1), bar_length=40, logging_step=256)
        progress.start()
        for seq_idx, (_seq_tag, features) in enumerate(source):
            features = np.asarray(features, dtype=np.float64)
            if shift is None:
                dim = features.shape[1]
                # Any constant works; the corpus mean is unknown at this point
                # and the first sequence's is close enough to keep the shifted
                # values small.
                shift = features.mean(axis=0)
                total = np.zeros(dim, dtype=np.float64)
                outer = np.zeros((dim, dim), dtype=np.float64)
            elif features.shape[1] != len(shift):
                raise ValueError(
                    f"feature dim changed: {len(shift)} -> {features.shape[1]}"
                )
            centered = features - shift
            num_frames += len(features)
            total += centered.sum(axis=0)
            outer += centered.T @ centered
            progress.progress(seq_idx)

        if not num_frames:
            raise RuntimeError("no frames read; check the segment list and the HDF files")

        offset = total / num_frames                     # mean - shift
        mean = shift + offset
        # sum (x - shift)(x - shift)^T = sum (x - mean)(x - mean)^T + n * offset offset^T
        cov = (outer - num_frames * np.outer(offset, offset)) / num_frames
        cov = (cov + cov.T) / 2

        # Checked here rather than left to blow up inside np.linalg.inv two
        # jobs downstream, where nothing points back at the cause. The test is
        # numerical rank, not strict positivity: a direction the data does not
        # span comes out as a tiny *positive* eigenvalue rather than a negative
        # one, and inverting that is what produces the garbage. Same threshold
        # np.linalg.matrix_rank uses.
        eigenvalues = np.linalg.eigvalsh(cov)
        threshold = eigenvalues.max() * len(eigenvalues) * np.finfo(np.float64).eps
        if eigenvalues.min() <= threshold:
            raise ValueError(
                f"covariance is numerically singular over {num_frames} frames: "
                f"{int((eigenvalues <= threshold).sum())} of {len(eigenvalues)} eigenvalues "
                f"are at or below {threshold:.3g} (smallest {eigenvalues.min():.3g}, "
                f"condition {eigenvalues.max() / max(eigenvalues.min(), np.finfo(np.float64).tiny):.3g}). "
                f"Expected with fewer frames than dimensions; otherwise some feature "
                f"dimensions are linearly dependent, and every model inverting this "
                f"covariance would produce nonsense"
            )
        print(
            f"{num_frames} frames, dim {len(mean)}: |mean| {np.linalg.norm(mean):.3f}, "
            f"eigenvalues {eigenvalues.min():.4g}..{eigenvalues.max():.4g}, "
            f"condition {eigenvalues.max() / eigenvalues.min():.3g}"
        )
        return {"cov": cov, "mean": mean}


class ClusterCovarianceJob(Job):
    """Per-cluster covariances over a fixed partition - the plan's "stage 2".

    Takes a codebook that some earlier run converged on, assigns every frame in
    the corpus to its nearest centroid, and estimates one covariance per
    cluster from the frames that landed there. Nothing is re-partitioned: the
    centroids are an input and come out unchanged, so this measures the shape
    of a partition rather than looking for one.

    A separate job rather than an epoch flavor because it is not an epoch. It
    runs once between a partition-finding stage and whatever consumes the
    covariances, its output plugs straight into ``mixture_flavor(covs=...)`` as
    a ``[K, D, D]`` array like any other, and re-estimating with a different
    ``structure`` or ``ridge`` re-runs this alone rather than a clustering run.

    **The point of ``structure``.** A full ``[D, D]`` covariance has D(D+1)/2
    free parameters - 131k at D=512 - and ls-100 supplies roughly 18M frames
    total, so at K=512 each cluster gets ~35k frames: a quarter of a parameter's
    worth of evidence each, and the estimate is singular by construction rather
    than by bad luck. Frames within a segment are strongly correlated too, so
    the effective count is lower still. The three settings span what can be done
    about that:

    ``"full"``
        Every parameter free. Included to be *shown* failing rather than
        asserted to fail: pair it with a ``ridge`` sweep and the point at which
        the ridge needed for invertibility is large enough to swamp the
        estimate is the empirical statement that there was nothing to estimate.
        ``out_diagnostics`` carries the condition numbers that make that
        concrete.
    ``"diagonal"``
        Diagonal in the space ``assignment_covs`` whitens, which is the
        "diagonal covariance plus a shared linear transform" arrangement -
        semi-tied covariances with the transform taken from the corpus
        covariance rather than re-estimated. D free parameters per cluster
        instead of D(D+1)/2, a factor of ~256 at D=512, so the evidence per
        parameter goes from a quarter of a frame to ~70. The result is still
        written as a dense ``[K, D, D]`` stack, because a covariance of the form
        ``L diag(v) L^T`` *is* a full covariance - the constraint is on how it
        was estimated, not on what it can be used by, and nothing downstream
        needs to know.
    ``"shared"``
        One covariance for every cluster, pooled over the corpus. The control:
        if per-cluster estimation is worth anything, it has to beat this.

    :param features_hdf: the feature file(s) the partition was found on
    :param centroids: ``[K, D]`` codebook, e.g. an unguided run's
        ``out_centroids[N]``
    :param assignment_covs: ``[K, D, D]`` covariances defining the metric
        frames are assigned under - normally the same fixed covariance the
        partition was found with, so that this job reproduces exactly that
        partition. Omit for squared-Euclidean assignment. Also supplies the
        whitening transform for ``structure="diagonal"``, which is why it is
        one argument rather than two: the space clusters are measured in should
        be the space they were found in.
    :param ridge: added to each covariance's diagonal as a fraction of its own
        mean variance, i.e. ``Sigma += ridge * trace(Sigma)/D * I``. Relative
        rather than absolute so one value means the same thing across feature
        scalings and across clusters of very different spread. 0.0 leaves the
        estimate untouched, which is what the ``"full"`` arm wants at the
        bottom of its sweep.
    :param min_frames: a cluster with fewer frames than this keeps the pooled
        corpus covariance instead of its own. Defaults to the feature
        dimension, below which a sample covariance is singular by rank
        regardless of ridge; set it higher to decide by evidence rather than by
        rank.
    """

    __sis_hash_exclude__ = {"rqmt": None}

    def __init__(
        self,
        features_hdf: Union[tk.Path, Sequence[tk.Path]],
        centroids: tk.Path,
        assignment_covs: Optional[tk.Path] = None,
        structure: str = "full",
        ridge: float = 0.0,
        min_frames: Optional[int] = None,
        segments: Optional[tk.Path] = None,
        subsampling: Optional[int] = None,
        pooling_function: str = "maxpool_time_np",
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        if structure not in ("full", "diagonal", "shared"):
            raise ValueError(
                f"structure must be 'full', 'diagonal' or 'shared', got {structure!r}"
            )
        if ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        self.features_hdf = features_hdf
        self.centroids = centroids
        self.assignment_covs = assignment_covs
        self.structure = structure
        self.ridge = ridge
        self.min_frames = min_frames
        self.segments = segments
        self.subsampling = subsampling
        self.pooling_function = pooling_function

        self.out_covs = self.output_path("covs.npy")
        self.out_counts = self.output_path("counts.npy")
        self.out_diagnostics = self.output_path("diagnostics.json")

        # Dominated by reading the features (~29 GB for ls-100), as for
        # GlobalCovarianceJob. Memory is the difference: a full second moment is
        # K x D x D float64, which at K=512, D=512 is 1.07 GB, and the finished
        # stack is another. A diagonal run holds K x D and needs almost nothing.
        self.rqmt = {"cpu": 4, "mem": 8 if structure == "diagonal" else 32, "time": 6}
        if rqmt:
            self.rqmt.update(rqmt)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        centroids = np.load(self.centroids.get_path())
        if centroids.ndim != 2:
            raise ValueError(f"expected centroids [K, D], got {centroids.shape}")
        num_clusters, dim = centroids.shape

        assignment_covs = (
            np.load(self.assignment_covs.get_path()) if self.assignment_covs else None
        )
        if assignment_covs is not None and assignment_covs.shape != (num_clusters, dim, dim):
            raise ValueError(
                f"expected assignment covariances [{num_clusters}, {dim}, {dim}], "
                f"got {assignment_covs.shape}"
            )

        # The same scoring the unguided epoch used, so the partition this job
        # measures is the partition that run produced rather than a near miss.
        model = (
            GaussianModel(centroids, assignment_covs, device="cpu")
            if assignment_covs is not None
            else EuclideanModel(centroids)
        )

        # For "diagonal": the whitening the shared covariance induces. Taken
        # from cluster 0 because the assignment covariances are one matrix
        # duplicated - the transform is shared by definition, and a per-cluster
        # transform would defeat the parameter saving this exists for.
        transform = None
        if self.structure == "diagonal":
            shared = assignment_covs[0] if assignment_covs is not None else np.eye(dim)
            if assignment_covs is not None and not np.allclose(
                assignment_covs, shared[np.newaxis], atol=0, rtol=1e-10
            ):
                raise ValueError(
                    "structure='diagonal' needs one shared assignment covariance to take "
                    "the transform from, but the covariances differ between clusters; "
                    "a per-cluster transform is a full covariance by another name"
                )
            # Sigma_shared = L L^T, y = L^-1 x. Cholesky rather than an
            # eigendecomposition: it is the transform the Mahalanobis distance
            # already factors through, so "diagonal in this space" means
            # diagonal in the coordinates the assignment metric uses.
            transform = np.linalg.cholesky(shared)

        source = HDFFeatureSource(
            files=[
                f.get_path()
                for f in (
                    list(self.features_hdf)
                    if isinstance(self.features_hdf, (list, tuple))
                    else [self.features_hdf]
                )
            ],
            segments=self.segments.get_path() if self.segments else None,
            subsampling=self.subsampling,
            pooling_function=self.pooling_function,
        )

        counts = np.zeros(num_clusters, dtype=np.float64)
        # Accumulated about each cluster's own centroid, which is already the
        # right order of magnitude for its mean - so the second moment needs no
        # shift trick and the subtraction below cancels almost nothing.
        first = np.zeros((num_clusters, dim), dtype=np.float64)
        if self.structure == "diagonal":
            second = np.zeros((num_clusters, dim), dtype=np.float64)
        else:
            second = np.zeros((num_clusters, dim, dim), dtype=np.float64)
        pooled = np.zeros((dim, dim), dtype=np.float64)

        progress = ProgressLogger(max(len(source), 1), bar_length=40, logging_step=256)
        progress.start()
        num_frames = 0
        for seq_idx, (_seq_tag, features) in enumerate(source):
            features = np.asarray(features, dtype=np.float64)
            if features.shape[1] != dim:
                raise ValueError(f"feature dim {features.shape[1]} != centroid dim {dim}")
            labels = model.scores(features).argmin(axis=1)
            num_frames += len(features)
            for k in np.unique(labels):
                block = features[labels == k] - centroids[k]
                counts[k] += len(block)
                first[k] += block.sum(axis=0)
                if self.structure == "diagonal":
                    # Whitened coordinates: y - L^-1 c = L^-1 (x - c).
                    whitened = np.linalg.solve(transform, block.T).T
                    second[k] += (whitened ** 2).sum(axis=0)
                else:
                    second[k] += block.T @ block
                    pooled += block.T @ block
            progress.progress(seq_idx)

        if not num_frames:
            raise RuntimeError("no frames read; check the segment list and the HDF files")

        min_frames = self.min_frames if self.min_frames is not None else dim
        alive = counts >= max(min_frames, 1)
        safe = np.where(counts > 0, counts, 1.0)[:, np.newaxis]
        offset = first / safe                       # mean - centroid

        if self.structure == "diagonal":
            # Var along each whitened axis, then back: Sigma = L diag(v) L^T.
            whitened_offset = np.linalg.solve(transform, offset.T).T
            variances = second / safe - whitened_offset ** 2
            variances = np.maximum(variances, 0.0)
            covs = np.einsum("ij,kj,lj->kil", transform, variances, transform)
            # Pooled fallback in the same family, so a starved cluster does not
            # silently get a differently-shaped covariance from its neighbours.
            pooled_cov = (
                transform
                @ np.diag((second[alive].sum(0) / max(counts[alive].sum(), 1.0)))
                @ transform.T
                if alive.any()
                else np.eye(dim)
            )
        else:
            covs = second / safe[:, :, np.newaxis] - offset[:, :, np.newaxis] * offset[:, np.newaxis, :]
            pooled_cov = pooled / num_frames

        covs = (covs + covs.transpose(0, 2, 1)) / 2
        pooled_cov = (pooled_cov + pooled_cov.T) / 2

        if self.structure == "shared":
            covs = np.repeat(pooled_cov[np.newaxis], num_clusters, axis=0)
            alive = np.ones(num_clusters, dtype=bool)

        # Starved clusters take the pooled covariance rather than their own.
        # Not a repair of a bad estimate but a refusal to make one: below D
        # frames the sample covariance is singular by rank, and no ridge that
        # leaves it meaningful will fix that.
        starved = ~alive
        if starved.any():
            covs[starved] = pooled_cov

        if self.ridge:
            # Relative to each covariance's own mean variance, so one ridge
            # value means the same thing for a tight cluster and a diffuse one.
            scale = np.trace(covs, axis1=1, axis2=2) / dim
            covs = covs + (self.ridge * scale)[:, None, None] * np.eye(dim)[None]

        eigenvalues = np.linalg.eigvalsh(covs)
        smallest = eigenvalues[:, 0]
        largest = eigenvalues[:, -1]
        condition = largest / np.maximum(smallest, np.finfo(np.float64).tiny)
        threshold = largest * dim * np.finfo(np.float64).eps
        singular = smallest <= threshold
        _, logdet = np.linalg.slogdet(covs)

        diagnostics = {
            "structure": self.structure,
            "ridge": self.ridge,
            "num_clusters": int(num_clusters),
            "dim": int(dim),
            "num_frames": int(num_frames),
            # The headline number for the negative-result arm: how much evidence
            # each free parameter of a full covariance actually got.
            "free_parameters_per_cluster": int(dim * (dim + 1) // 2)
            if self.structure != "diagonal"
            else int(dim),
            "frames_per_free_parameter": float(
                num_frames
                / num_clusters
                / (dim * (dim + 1) / 2 if self.structure != "diagonal" else dim)
            ),
            "num_singular": int(singular.sum()),
            "num_starved": int(starved.sum()),
            "min_frames_threshold": int(min_frames),
            "counts": {
                "min": float(counts.min()),
                "median": float(np.median(counts)),
                "max": float(counts.max()),
                "empty_clusters": int((counts == 0).sum()),
            },
            "condition_number": {
                "min": float(condition.min()),
                "median": float(np.median(condition)),
                "p95": float(np.percentile(condition, 95)),
                "max": float(condition.max()),
            },
            "log_determinant": {
                "min": float(logdet.min()),
                "median": float(np.median(logdet)),
                "max": float(logdet.max()),
            },
            "smallest_eigenvalue": {
                "min": float(smallest.min()),
                "median": float(np.median(smallest)),
            },
        }
        print(
            f"{num_frames} frames over {num_clusters} clusters, structure={self.structure}, "
            f"ridge={self.ridge}: {diagnostics['frames_per_free_parameter']:.2f} frames per "
            f"free parameter, {int(singular.sum())} singular, {int(starved.sum())} starved, "
            f"median condition {np.median(condition):.3g}",
            flush=True,
        )
        if singular.any() and not self.ridge:
            print(
                f"WARNING: {int(singular.sum())} of {num_clusters} covariances are "
                f"numerically singular. Every model inverting them produces nonsense - "
                f"GaussianModelNumpy uses np.linalg.inv and casts to float32, so this "
                f"surfaces as meaningless scores rather than as an exception. Raise "
                f"ridge, raise min_frames, or use structure='diagonal'.",
                flush=True,
            )

        np.save(self.out_covs.get_path(), covs)
        np.save(self.out_counts.get_path(), counts)
        with open(self.out_diagnostics.get_path(), "w") as fp:
            json.dump(diagnostics, fp, indent=4)


class SelectCovJob(ArrayJob):
    """Pick one matrix out of a stack of covariances.

    ``[K, D, D] -> [D, D]``. Exists because the covariances lying around this
    setup are mostly stacks - ``constants.SHARED_COVS`` is the corpus-wide
    covariance already duplicated 40 times - while the jobs that build an
    initial state from *one* covariance want the single matrix. Composes:
    ``SelectCovJob(SHARED_COVS) -> DuplicateCovsJob(..., 128)`` re-tiles that
    same matrix to whatever density count a run needs.
    """

    OUTPUTS = ("cov",)

    def __init__(self, covs: tk.Path, index: int = 0):
        self.covs = covs
        self.index = index
        super().__init__()

    def compute(self):
        covs = self.load(self.covs, ndim=3, name="covariances")
        if not -len(covs) <= self.index < len(covs):
            raise IndexError(f"index {self.index} out of range for {len(covs)} covariances")
        return covs[self.index]


class DuplicateCovsJob(ArrayJob):
    """Tile one covariance matrix across ``num_densities`` slots.

    ``[D, D] -> [num_densities, D, D]``. The point of starting every density
    from the same real covariance - the corpus-wide one, say - rather than from
    the identity is that the first epoch already scores in a whitened space,
    so the initial partition reflects the shape of the data instead of the
    arbitrary scaling of the encoder's dimensions. Each density then specializes
    from that common starting point.

    This is what ``constants.SHARED_COVS`` was built by hand to be. Accepts a
    ``[1, D, D]`` input too, since that is how a single matrix often ends up
    saved.
    """

    OUTPUTS = ("covs",)

    def __init__(self, cov: tk.Path, num_densities: int):
        self.cov = cov
        self.num_densities = num_densities
        super().__init__()

        # Not a mini-task: a covariance stack at D=512 is 2 MB per matrix, so a few
        # hundred densities is most of a gigabyte and well past the local engine's
        # allowance. Set after super().__init__() rather than taken as a constructor
        # argument, so it stays out of the hash.
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def compute(self):
        cov = self.load(self.cov, ndim=(2, 3), name="covariance")
        if cov.ndim == 3:
            if len(cov) != 1:
                raise ValueError(
                    f"expected a single covariance to duplicate, got a stack of "
                    f"{len(cov)}; use RepeatCovsJob to expand a stack in place"
                )
            cov = cov[0]
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"expected a square covariance, got {cov.shape}")
        return np.broadcast_to(cov, (self.num_densities, *cov.shape)).copy()


class RepeatCovsJob(ArrayJob):
    """Repeat each covariance in a stack ``repeats`` times, in place.

    ``[K, D, D] -> [K * repeats, D, D]``, with each input matrix's copies
    adjacent - the layout :class:`SplitCentroidsJob` produces and
    :class:`...lib.guided_kmeans.chunked.PerLabelMixtureModel` reads, so the two
    line up density for density when a label's centroid is split into several.
    """

    OUTPUTS = ("covs",)

    def __init__(self, covs: tk.Path, repeats: int):
        self.covs = covs
        self.repeats = repeats
        super().__init__()

        # Not a mini-task: a covariance stack at D=512 is 2 MB per matrix, so a few
        # hundred densities is most of a gigabyte and well past the local engine's
        # allowance. Set after super().__init__() rather than taken as a constructor
        # argument, so it stays out of the hash.
        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def compute(self):
        covs = self.load(self.covs, ndim=3, name="covariances")
        # repeat(), not tile(): copies of one matrix have to be adjacent.
        return np.repeat(covs, self.repeats, axis=0)


class UniformMixturesJob(ArrayJob):
    """Uniform mixture weights, the starting point for either mixture layout.

    ``[num_labels, num_densities]`` filled with ``1 / num_densities``. What
    ``num_densities`` means depends on the layout the weights are for: pass the
    codebook size for :class:`...lib.guided_kmeans.chunked.GaussianMixtureModel`
    (every label starts able to use every density equally), or the per-label
    density count for
    :class:`...lib.guided_kmeans.chunked.PerLabelMixtureModel`.

    Uniform on purpose: the weights are the one parameter that carries no
    information at initialization - the densities do - so a uniform start lets
    the first E-step assign weight from the data rather than from a guess. It
    also means no weight starts at zero, which matters because zero is
    absorbing under the default ``mixture_floor=0``.

    Right for per-label densities, **wrong for a shared codebook**: there,
    identical weights make every label score identically, so the first
    recognition pass sees no acoustic difference between labels at all. Use
    :class:`RandomMixturesJob` for that layout.
    """

    OUTPUTS = ("mixtures",)

    def __init__(self, num_labels: int, num_densities: int):
        self.num_labels = num_labels
        self.num_densities = num_densities
        super().__init__()

    def compute(self):
        return np.full((self.num_labels, self.num_densities), 1.0 / self.num_densities)


class RandomMixturesJob(ArrayJob):
    """Random mixture weights, one Dirichlet draw per label.

    The initializer to use for a *shared* codebook, where uniform weights are
    not a neutral starting point but a degenerate one: if every label weights
    the codebook identically then every label scores identically, and the first
    recognition pass has no acoustic information to go on at all - it is driven
    entirely by the language model and the transition costs. Drawing each
    label's weights separately breaks that symmetry before the first pass.

    Per-label densities do not have this problem, because the densities
    themselves already differ per label; :class:`UniformMixturesJob` is the
    right neutral start there.

    :param concentration: Dirichlet parameter. 1.0 draws uniformly from the
        simplex; below it the draws are sparser, so labels commit harder to a
        few densities from the outset.
    """

    OUTPUTS = ("mixtures",)

    def __init__(
        self,
        num_labels: int,
        num_densities: int,
        concentration: float = 1.0,
        seed: int = 42,
    ):
        self.num_labels = num_labels
        self.num_densities = num_densities
        self.concentration = concentration
        self.seed = seed
        super().__init__()

    def compute(self):
        rng = np.random.RandomState(self.seed)
        mixtures = rng.dirichlet(
            np.full(self.num_densities, self.concentration), size=self.num_labels
        )
        # A draw can underflow to exactly zero, and zero is absorbing unless
        # mixture_floor is set - which would silently shrink the codebook
        # before the first epoch. Renormalize away from it.
        mixtures = np.maximum(mixtures, np.finfo(np.float64).tiny)
        mixtures /= mixtures.sum(axis=-1, keepdims=True)
        return mixtures


class NormalTableJob(ArrayJob):
    """Initial ``p(codeword | label)``: a normal draw per entry, rows normalized.

    ``table[l, c] = max(1 + sigma * z, eps)`` with ``z ~ N(0, 1)``, then each row
    divided by its sum. Written around a mean of 1 rather than ``1/C`` because
    normalization removes the mean anyway - what survives it is the *relative*
    spread, so ``sigma`` is the coefficient of variation and means the same
    thing at any codebook size.

    This is the only place an unsupervised run's symmetry can be broken. With a
    frozen codebook the table is the entire model, so a table that starts
    (near-)uniform gives every label the same score column, leaves the first
    search with nothing acoustic to separate labels by, and hands the counting
    step an alignment produced by the language model alone. ``sigma`` is
    therefore a real hyperparameter and not a nuisance: it decides how far from
    that degenerate point a run begins.

    For calibration against :class:`RandomMixturesJob`, whose Dirichlet draws do
    the same job for the mixture models: a Dirichlet(alpha) row over C
    categories has coefficient of variation ``sqrt((C-1)/(C*alpha+1))``, so at
    C=512 alpha=1.0 corresponds to ``sigma`` near 1.0, alpha=0.1 to about 3.1.
    Small ``sigma`` is the *weak* break, not the safe one.

    :param sigma: relative standard deviation before normalizing
    :param clip: floor applied to a draw before normalizing. Normal draws go
        negative once ``sigma`` approaches 1 (16% of entries at sigma=1), and a
        negative probability is not a small problem - the fraction clipped is
        reported so a sigma large enough to distort the distribution is visible
        rather than silent.
    """

    OUTPUTS = ("table",)

    def __init__(
        self,
        num_labels: int,
        num_codewords: int,
        sigma: float = 0.1,
        seed: int = 42,
        clip: float = 1e-6,
    ):
        self.num_labels = num_labels
        self.num_codewords = num_codewords
        self.sigma = sigma
        self.seed = seed
        self.clip = clip
        super().__init__()

    def compute(self):
        rng = np.random.RandomState(self.seed)
        draws = 1.0 + self.sigma * rng.randn(self.num_labels, self.num_codewords)
        clipped = int((draws < self.clip).sum())
        if clipped:
            print(
                f"WARNING: {clipped} of {draws.size} entries "
                f"({100 * clipped / draws.size:.1f}%) were negative at sigma={self.sigma} "
                f"and were clipped to {self.clip}. The draw is no longer normal at this "
                f"width; treat sigma as a shape knob rather than a standard deviation.",
                flush=True,
            )
        draws = np.maximum(draws, self.clip)
        table = draws / draws.sum(axis=1, keepdims=True)
        print(
            f"table [{self.num_labels}, {self.num_codewords}] at sigma={self.sigma}: "
            f"row min {table.min():.3e}, row max {table.max():.3e}, "
            f"relative spread {table.std() / table.mean():.3f}",
            flush=True,
        )
        return table


class SplitCentroidsJob(ArrayJob):
    """Split each centroid into ``num_densities`` displaced copies.

    Takes ``[L, D]`` and produces ``[L * num_densities, D]`` in the layout
    :class:`...lib.guided_kmeans.chunked.PerLabelMixtureModel` expects - label
    ``l``'s densities contiguous at ``l * num_densities`` - so a converged
    single-Gaussian run can seed a per-label mixture run. Pair it with
    :class:`RepeatCovsJob` on the same centroids' covariances to get a matching
    ``[L * num_densities, D, D]``.

    The copies must not be identical. Densities that start equal receive equal
    responsibility for every frame and stay equal forever, so EM would never
    break the tie and the run would be a slower single-density run.

    How they are displaced depends on whether ``covs`` is given:

    ``covs=None``
        Isotropic jitter, scaled by the per-dimension spread of the centroid
        set. Needs no covariance, but knows nothing about the shape of the
        cluster it is splitting, so some copies land in directions the data
        never goes.

    ``covs`` given
        Copies are placed along the cluster's principal axis, at offsets spread
        symmetrically over ``+/- perturbation * sqrt(largest eigenvalue)``. This
        is the classic mixture-splitting move: the principal axis is where the
        cluster is widest, so it is where a second density has something to
        explain, and the offset is in the units of that spread rather than of
        the encoder's arbitrary scaling. Deterministic - no seed is used.

    :param covs: ``[L, D, D]``, one covariance per input centroid, or ``[D, D]``
        to use one shared covariance for every split. Hash-excluded at its
        default, so adding it left the random-jitter form's hash untouched.
    :param perturbation: displacement scale. Against ``covs`` it multiplies the
        principal standard deviation, where ~0.2 is the conventional value;
        without it, the per-dimension spread of the centroid set.
    """

    __sis_hash_exclude__ = {"covs": None}

    OUTPUTS = ("centroids",)

    def __init__(
        self,
        centroids: tk.Path,
        num_densities: int,
        perturbation: float = 0.05,
        seed: int = 42,
        covs: Optional[tk.Path] = None,
    ):
        self.centroids = centroids
        self.num_densities = num_densities
        self.perturbation = perturbation
        self.seed = seed
        self.covs = covs
        super().__init__()

    def _offsets(self) -> np.ndarray:
        """``[num_densities]`` multiples of the displacement, symmetric about 0.

        ``linspace`` rather than a +/- pair so this generalizes past a binary
        split; for ``num_densities == 1`` it is a single zero, i.e. a no-op.
        """
        if self.num_densities == 1:
            return np.zeros(1)
        return np.linspace(-1.0, 1.0, self.num_densities)

    def _principal_axes(self, centroids: np.ndarray) -> np.ndarray:
        """``[L, D]``: each cluster's widest direction, scaled by its spread."""
        covs = self.load(self.covs, ndim=(2, 3), name="covariances")
        if covs.ndim == 2:
            covs = np.broadcast_to(covs, (len(centroids), *covs.shape))
        elif len(covs) != len(centroids):
            raise ValueError(
                f"got {len(covs)} covariances for {len(centroids)} centroids; pass one "
                f"per centroid or a single [D, D] to share"
            )
        if covs.shape[1:] != (centroids.shape[1], centroids.shape[1]):
            raise ValueError(
                f"covariances {covs.shape} do not match centroids {centroids.shape}"
            )
        # eigh, not eig: covariances are symmetric, and eigh returns real
        # eigenvalues in ascending order, so the last column is the principal
        # axis. Eigenvectors are unit length, so scaling by sqrt(eigenvalue)
        # makes the offset one standard deviation along that axis.
        eigenvalues, eigenvectors = np.linalg.eigh(covs)
        principal = eigenvectors[:, :, -1]                       # [L, D]
        spread = np.sqrt(np.maximum(eigenvalues[:, -1], 0.0))    # [L]
        return principal * spread[:, np.newaxis]

    def compute(self):
        centroids = self.load(self.centroids, ndim=2, name="centroids")
        # repeat() rather than tile(): label l's copies have to be adjacent,
        # because the model reads density l * n + k as label l's k-th.
        split = np.repeat(centroids, self.num_densities, axis=0)
        offsets = np.tile(self._offsets(), len(centroids))[:, np.newaxis]

        if self.covs is not None:
            direction = np.repeat(self._principal_axes(centroids), self.num_densities, axis=0)
        else:
            rng = np.random.RandomState(self.seed)
            direction = centroids.std(axis=0, keepdims=True) * rng.randn(*split.shape)
            # The random form displaces every copy independently rather than
            # spreading them along one axis, so the offsets would only rescale
            # an already-random direction - and would put one copy exactly on
            # the centroid for odd counts.
            offsets = 1.0

        return split + self.perturbation * offsets * direction


@dataclass
class ChunkedClusteringExpResult:
    """
    ``out_models[epoch]`` is the authoritative result: a model directory that
    can be loaded without knowing the model class. ``out_models[0]`` is the
    starting point, assembled from the caller's loose files by
    :class:`MaterializeModelJob` so that every epoch including the initial one
    can be decoded the same way. ``out_centroids`` and
    ``out_covs`` are conveniences pointing inside those directories, kept
    because the existing decode configs take individual ``tk.Path``s
    (``DecodeConfig(centroids=..., covs=...)``); a model with other parameters
    is reached with ``result.jobs[epoch - 1].artifact("name")``.

    ``out_hypotheses`` and ``out_guided_scores`` are keyed like the models they
    describe, not like the job that produced them: epoch ``e``'s job recognizes
    with model ``e - 1``, so its hypotheses and PER are filed under ``e - 1``,
    next to ``out_centroids[e - 1]``. The last model therefore has no entry -
    no epoch ever recognized with it. Getting its number needs one decode of
    its own (or one more clustering epoch whose model you ignore).
    """

    jobs: List[GuidedClusteringEpochJob]
    out_centroids: Dict[int, tk.Path]
    out_statistics: tk.Path
    out_models: Dict[int, tk.Path]
    out_covs: Optional[Dict[int, tk.Path]] = None
    out_hypotheses: Optional[Dict[int, tk.Path]] = None
    out_guided_scores: Optional[Dict[int, ScoreResult]] = None
    #: ``{epoch: statistics.json}``, one file per epoch job, alongside the merged
    #: ``out_statistics``. A report on a finished run can read the merged file; one
    #: on a run still going has to read these, because the merge cannot happen until
    #: the last epoch does - see ``latex_report.clustering_statistics_per_epoch``.
    out_epoch_statistics: Optional[Dict[int, tk.Path]] = None
    #: ``{artifact_name: {epoch: path}}`` for whatever this run's model class
    #: declares, epoch 0 being the caller's starting point. ``out_centroids``
    #: and ``out_covs`` are the two named views of it that predate it and that
    #: the decode configs take directly; a model with other parameters (mixture
    #: weights, say) is reached through here without adding a field per model.
    out_artifacts: Optional[Dict[str, Dict[int, tk.Path]]] = None

    def guided_score_row(self, epoch: int) -> Dict[str, Any]:
        """
        ``values=`` dict for ``LatexTableReport.add_row``, filling the
        ``guided_per``/``guided_del``/``guided_ins``/``guided_sub`` columns::

            latex_report.add_row(
                result=res, epoch=epoch, statistics=statistics,
                values=exp_result.guided_score_row(epoch),
            )

        Empty for an epoch that has no guided score - the last one, or any
        epoch if the run was built without ``score_reference`` - so it can be
        passed unconditionally and those cells simply stay blank.
        """
        score = (self.out_guided_scores or {}).get(epoch)
        if score is None:
            return {}
        return {
            "guided_per": score.wer,
            "guided_del": score.deletions,
            "guided_ins": score.insertions,
            "guided_sub": score.substitutions,
        }


def _flavor_from_flags(
    *,
    initial_centroids: Optional[tk.Path],
    initial_covs: Optional[tk.Path],
    initial_mixtures: Optional[tk.Path],
    recognition_config: tk.Path,
    lexicon: tk.Path,
    num_clusters: int,
    distance_scale: float,
    use_forward_backward: bool,
    num_workers: int,
    task_timeout: Optional[float],
) -> ClusteringFlavor:
    """
    The pre-flavor keyword form, translated.

    Kept so that every existing config keeps building exactly the jobs it built
    before - the specs these factories produce are the ones this function used
    to construct inline, which the recorded hashes in ``test_chunked`` pin
    down. New combinations should be expressed as a flavor rather than as
    another initial_* argument here.
    """
    if initial_centroids is None:
        raise TypeError("chunked_clustering() needs either initial_centroids or a flavor")
    if initial_mixtures is not None and initial_covs is None:
        raise TypeError("initial_mixtures needs initial_covs: a mixture is a mixture of Gaussians")

    common = dict(
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=num_clusters,
        distance_scale=distance_scale,
        use_forward_backward=use_forward_backward,
        num_workers=num_workers,
        task_timeout=task_timeout,
    )
    if initial_mixtures is not None:
        return mixture_flavor(
            centroids=initial_centroids, covs=initial_covs, mixtures=initial_mixtures, **common
        )
    if initial_covs is not None:
        return gaussian_flavor(centroids=initial_centroids, covs=initial_covs, **common)
    return euclidean_flavor(centroids=initial_centroids, **common)


def chunked_clustering(
    *,
    num_epochs: int,
    features_hdf: Union[tk.Path, Sequence[tk.Path]],
    recognition_config: Optional[tk.Path] = None,
    lexicon: Optional[tk.Path] = None,
    num_clusters: int,
    initial_centroids: Optional[tk.Path] = None,
    initial_covs: Optional[tk.Path] = None,
    initial_mixtures: Optional[tk.Path] = None,
    flavor: Optional[ClusteringFlavor] = None,
    segments: Optional[tk.Path] = None,
    subsampling: Optional[int] = None,
    pooling_function: str = "maxpool_time_np",
    distance_scale: float = 1.0,
    use_forward_backward: bool = False,
    score_reference: Optional[tk.Path] = None,
    rasr_path: Optional[tk.Path] = None,
    num_chunks: int = 30,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
    rqmt: Optional[Dict[str, Any]] = None,
    alias_prefix: str = "guided_kmeans/chunked",
) -> ChunkedClusteringExpResult:
    """
    Chain ``num_epochs`` clustering epochs.

    Starts from an existing set of centroids: unlike the single-process
    pipeline, initialization is not folded into the run as a phase-0 pass over
    the corpus. Use a separate initialization job (or the cheating centroids
    the configs already reference) and pass its output here - which also makes
    the starting point a reusable, independently inspectable artifact.

    **Continuing a run is free.** Extending an experiment by calling this again
    with the previous result's last centroids::

        first = chunked_clustering(num_epochs=5, initial_centroids=init, ...)
        more  = chunked_clustering(num_epochs=5,
                                   initial_centroids=first.out_centroids[5], ...)

    produces exactly the jobs a single ``num_epochs=10`` call would, so nothing
    is recomputed and the two spellings of the same experiment share results.
    This holds because every epoch's model spec has the same shape - same model
    class, same artifact names - whether its inputs are the caller's files or
    the previous epoch's model directory. Keep it that way: introducing a
    different spec form for the first epoch silently breaks continuation, which
    is why ``test_chunked`` asserts the 5+5 == 10 property.

    :param recognition_config: RASR config for the guiding search. Optional
        only because an unguided flavor has no search to configure - every
        RASR-guided flavor needs one, and builds its recognizer spec around it.
    :param lexicon: the label inventory. Optional on the same grounds: it is
        what makes the cluster axis mean *phonemes*, so a run whose clusters
        are not labels (:func:`...lib.guided_kmeans.chunked.unguided_flavor`)
        has none, and the epoch job then records no traceback-driven
        statistics. Leave ``score_reference`` unset for such a run too - there
        is nothing to score its hypotheses against.
    :param initial_covs: pass to run with full-covariance Gaussian scoring;
        omit for plain squared-Euclidean k-means. For continuation pass the
        previous result's ``out_covs[N]`` alongside its ``out_centroids[N]``.
    :param score_reference: tagged phoneme reference for the clustering corpus
        (``TaggedCorpusToTxtJob(phoneme_corpus(setup_corpus(...))).out_txt``).
        Given one, every epoch's own recognition of the corpus is scored, which
        is where ``out_guided_scores`` comes from. Note what that PER measures:
        the *guiding* search, over the training corpus, with this run's
        recognition config - not a decoding of held-out data with the decoding
        config, so it is a convergence diagnostic rather than a number to
        compare against the decode tables. For a run whose recognition config
        has ``cheating=True`` it is degenerate by construction (the search is
        constrained to the reference), so leave it unset there.

        Scoring is a separate job rather than part of the epoch: the reference
        stays out of the expensive job's inputs and hash, and re-scoring - with
        different lemma exclusions, or with a metric not invented yet - never
        re-runs a RASR search.
    """
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    files = list(features_hdf) if isinstance(features_hdf, (list, tuple)) else [features_hdf]

    features_spec = Spec(
        HDFFeatureSource,
        {
            "files": files,
            "segments": segments,
            "subsampling": subsampling,
            "pooling_function": pooling_function,
        },
    )
    # Which model, which search and which updating routine go together is one
    # decision, not three independent flags, so it arrives as one object. The
    # legacy keyword form builds the same object here, which is why callers
    # that never heard of a flavor keep producing the jobs they always did.
    if flavor is None:
        flavor = _flavor_from_flags(
            initial_centroids=initial_centroids,
            initial_covs=initial_covs,
            initial_mixtures=initial_mixtures,
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            num_workers=num_workers,
            task_timeout=task_timeout,
        )
    elif any(a is not None for a in (initial_centroids, initial_covs, initial_mixtures)):
        raise TypeError(
            "pass either a flavor or the initial_* artifacts, not both: a flavor "
            "already carries its own starting point"
        )

    # The model class and its artifact names are fixed for the whole run, and
    # both come from the flavor rather than from a branch here - which is what
    # lets a new model be a new factory in lib.guided_kmeans.chunked.flavors
    # instead of another case in this function.
    artifact_names = flavor.artifact_names
    recognizer_spec = flavor.recognizer
    accumulator_spec = flavor.accumulator
    statistics_spec = flavor.statistics
    model_spec = flavor.model

    job_rqmt = {"cpu": num_workers + 1, "mem": 16, "time": 168}
    if rqmt:
        job_rqmt.update(rqmt)

    jobs: List[GuidedClusteringEpochJob] = []
    out_models: Dict[int, tk.Path] = {}
    statistics: Dict[int, tk.Path] = {}
    # Epoch 0 is the starting point the flavor carries, exposed in the same
    # per-epoch shape as the produced models so decode loops can index it.
    out_artifacts: Dict[str, Dict[int, tk.Path]] = {
        name: {0: flavor.model.kwargs[name]} for name in artifact_names
    }
    # Epoch 0 as a model directory as well as loose files, so a consumer that
    # takes whole models - decoding a parameter set that is not (centroids,
    # covs) - can reach the starting point too. Costs nothing unless something
    # asks for it: sisyphus only runs jobs a registered output depends on.
    initial_model_job = MaterializeModelJob(flavor.model)
    initial_model_job.add_alias(f"{alias_prefix}/epoch_000")
    out_models[0] = initial_model_job.out_model
    out_hypotheses: Dict[int, tk.Path] = {}
    out_guided_scores: Dict[int, ScoreResult] = {}

    for epoch in range(1, num_epochs + 1):
        job = GuidedClusteringEpochJob(
            features=features_spec,
            model=model_spec,
            recognizer=recognizer_spec,
            accumulator=accumulator_spec,
            num_clusters=num_clusters,
            lexicon=lexicon,
            rasr_path=rasr_path,
            num_chunks=num_chunks,
            statistics=statistics_spec,
            rqmt=job_rqmt,
        )
        job.add_alias(f"{alias_prefix}/epoch_{epoch:03d}")
        jobs.append(job)

        out_models[epoch] = job.out_model
        statistics[epoch] = job.out_statistics
        for name in artifact_names:
            out_artifacts[name][epoch] = job.artifact(name)

        # Filed under the model that produced them: this job recognized with
        # the model of epoch-1, so its hypotheses describe out_centroids[epoch-1].
        out_hypotheses[epoch - 1] = job.out_hypotheses
        if score_reference is not None:
            score_job = JiwerScoringJob(
                score_reference,
                job.out_hypotheses,
                # Full corpus: the per-sentence visualization would be hundreds
                # of megabytes per epoch, and the counts no longer come from it.
                write_alignment=False,
            )
            score_job.add_alias(f"{alias_prefix}/guided_score/epoch_{epoch - 1:03d}")
            out_guided_scores[epoch - 1] = ScoreResult.from_job(score_job)

        # Structurally identical to the initial spec above - same class, same
        # artifact names, only the paths now point into this epoch's model
        # directory. That sameness is what keeps a continued run hash-identical
        # to an uninterrupted one, and building it through the flavor is what
        # stops a model from being able to break it.
        model_spec = flavor.next_model({name: job.artifact(name) for name in artifact_names})

    merge_job = MergeEpochStatisticsJob(statistics)

    return ChunkedClusteringExpResult(
        jobs=jobs,
        out_centroids=out_artifacts["centroids"],
        out_covs=out_artifacts.get("covs"),
        out_models=out_models,
        out_statistics=merge_job.out_statistics,
        out_hypotheses=out_hypotheses,
        out_guided_scores=out_guided_scores or None,
        out_epoch_statistics=statistics,
        out_artifacts=out_artifacts,
    )
