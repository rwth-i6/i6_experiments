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
    "GuidedClusteringEpochJob",
    "IdentityCovsJob",
    "MergeEpochStatisticsJob",
    "RandomCentroidsJob",
    "ChunkedClusteringExpResult",
    "chunked_clustering",
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

from .score import JiwerScoringJob, ScoreResult
from ..lib.guided_kmeans.util import DEFAULT_EXCLUDED_LEMMATA, traceback_to_text
from ..lib.guided_kmeans.chunked import (
    EuclideanModel,
    GaussianAccumulator,
    GaussianModel,
    HDFFeatureSource,
    MeanAccumulator,
    RasrFBRecognizer,
    RasrViterbiRecognizer,
    SoftGaussianAccumulator,
    Spec,
    default_stats_hooks,
    reduce_chunks,
    run_chunk,
    save_chunk,
)
from ..lib.guided_kmeans.statistics import FBStatisticsCounter

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
        result = run_chunk(
            features=self.features.build(chunk=index, num_chunks=self.num_chunks),
            model=self.model.build(),
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


class RandomCentroidsJob(Job):
    """Sample K random frames from a feature HDF as initial centroids.

    Reads only K rows from ``inputs`` (total-frames × feature-dim), so memory
    usage is O(K × feature-dim) regardless of corpus size.
    """

    def __init__(self, features_hdf: tk.Path, num_clusters: int, seed: int = 42):
        self.features_hdf = features_hdf
        self.num_clusters = num_clusters
        self.seed = seed
        self.out_centroids = self.output_path("centroids.npy")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import h5py
        import numpy as np

        rng = np.random.RandomState(self.seed)
        with h5py.File(self.features_hdf.get_path(), "r") as f:
            total_frames = f["inputs"].shape[0]
            indices = np.sort(rng.choice(total_frames, size=self.num_clusters, replace=False))
            centroids = f["inputs"][indices]
        np.save(self.out_centroids.get_path(), centroids)


class IdentityCovsJob(Job):
    """Write K identity covariance matrices as the initial state for covariance models.

    With identity covariances the Mahalanobis distance degenerates to Euclidean,
    so the first epoch of a random-init covariance run behaves like plain k-means.
    Subsequent epochs learn cluster-specific covariances from the data.
    """

    def __init__(self, num_clusters: int, feature_dim: int = 512):
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.out_covs = self.output_path("covs.npy")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import numpy as np

        covs = np.stack([np.eye(self.feature_dim) for _ in range(self.num_clusters)])
        np.save(self.out_covs.get_path(), covs)


@dataclass
class ChunkedClusteringExpResult:
    """
    ``out_models[epoch]`` is the authoritative result: a model directory that
    can be loaded without knowing the model class. ``out_centroids`` and
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


def chunked_clustering(
    *,
    num_epochs: int,
    features_hdf: Union[tk.Path, Sequence[tk.Path]],
    recognition_config: tk.Path,
    lexicon: tk.Path,
    num_clusters: int,
    initial_centroids: tk.Path,
    initial_covs: Optional[tk.Path] = None,
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
    # num_workers/task_timeout are per-task scheduling knobs, like num_chunks:
    # they change how fast a chunk runs, never its result.
    if use_forward_backward:
        recognizer_spec = Spec(
            RasrFBRecognizer,
            {
                "recognition_config": recognition_config,
                "num_clusters": num_clusters,
                "distance_scale": distance_scale,
            },
            {"num_workers": num_workers, "task_timeout": task_timeout},
        )
    else:
        recognizer_spec = Spec(
            RasrViterbiRecognizer,
            {
                "recognition_config": recognition_config,
                "lexicon_path": lexicon,
                "distance_scale": distance_scale,
            },
            {"num_workers": num_workers, "task_timeout": task_timeout},
        )
    # Model class and its artifact names are chosen once, here, and then used
    # to build every epoch's model spec identically. Keeping the *form* of that
    # spec the same for the first epoch and all later ones is what makes
    # continuing a run reuse the jobs an uninterrupted run would have created:
    # picking up from `result.out_centroids[N]` reproduces exactly the spec
    # epoch N+1 had, because that path *is* epoch N's centroids artifact.
    if initial_covs is not None:
        model_cls, artifact_names = GaussianModel, ("centroids", "covs")
        initial_artifacts = {"centroids": initial_centroids, "covs": initial_covs}
        accumulator_spec = Spec(
            SoftGaussianAccumulator if use_forward_backward else GaussianAccumulator, {}
        )
    else:
        model_cls, artifact_names = EuclideanModel, ("centroids",)
        initial_artifacts = {"centroids": initial_centroids}
        accumulator_spec = Spec(MeanAccumulator, {})

    model_spec = Spec(model_cls, initial_artifacts)

    job_rqmt = {"cpu": num_workers + 1, "mem": 16, "time": 168}
    if rqmt:
        job_rqmt.update(rqmt)

    jobs: List[GuidedClusteringEpochJob] = []
    out_models: Dict[int, tk.Path] = {}
    statistics: Dict[int, tk.Path] = {}
    # Epoch 0 is the starting point the caller supplied, exposed in the same
    # per-epoch shape as the produced models so decode loops can index it.
    out_centroids: Dict[int, tk.Path] = {0: initial_centroids}
    out_covs: Dict[int, tk.Path] = {}
    if initial_covs is not None:
        out_covs[0] = initial_covs
    out_hypotheses: Dict[int, tk.Path] = {}
    out_guided_scores: Dict[int, ScoreResult] = {}

    # FBStatisticsCounter replaces the Viterbi traceback counters for FB epochs.
    # Unhashed: changing which diagnostics are recorded must not alter job identity.
    statistics_spec = (
        Spec(FBStatisticsCounter, {"num_clusters": num_clusters})
        if use_forward_backward
        else None
    )

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
        out_centroids[epoch] = job.artifact("centroids")
        if initial_covs is not None:
            out_covs[epoch] = job.artifact("covs")

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
        # to an uninterrupted one.
        model_spec = Spec(model_cls, {name: job.artifact(name) for name in artifact_names})

    merge_job = MergeEpochStatisticsJob(statistics)

    return ChunkedClusteringExpResult(
        jobs=jobs,
        out_centroids=out_centroids,
        out_covs=out_covs or None,
        out_models=out_models,
        out_statistics=merge_job.out_statistics,
        out_hypotheses=out_hypotheses,
        out_guided_scores=out_guided_scores or None,
    )
