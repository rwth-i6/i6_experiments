"""
Sisyphus wiring for a diagnostics pass: recognize the corpus with a given
model and record raw per-frame / per-segment data about what happened, without
updating anything.

Why this is a job of its own rather than a hook on
:class:`.chunked_clustering.GuidedClusteringEpochJob`, whose search this
duplicates:

* An epoch job's ``statistics`` parameter is unhashed on purpose - which
  diagnostics get recorded must not change what a job *is*. A raw dump written
  from an unhashed knob cannot be a declared output: two runs, one with the
  probe and one without, share a job directory, and the one without would leave
  a declared output missing. Hashing the probe instead would re-run all ten
  epochs of a sweep every time a field is added to the dump, which is exactly
  the cost this indirection avoids.
* A diagnostics pass wants a *different corpus* from the epoch it inspects: a
  few hundred segments is enough for a distribution and turns half an hour of
  cluster time into a couple of minutes. Pass ``segments`` for that.
* It can be pointed at any epoch's model, including epochs of runs that have
  already finished - no re-running of anything that produced a result.

The pass itself is the same code as a clustering epoch (``run_chunk`` with the
same feature source, model and recognizer specs), so what it observes is what
the guiding search actually does, not an approximation of it. The only
difference is that the accumulator is a
:class:`...lib.guided_kmeans.chunked.accumulators.NullAccumulator` and a
:class:`...lib.guided_kmeans.chunked.diagnostics.FrameDiagnostics` probe is
attached.
"""

from __future__ import annotations

__all__ = ["ClusteringDiagnosticsJob", "clustering_diagnostics"]

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from sisyphus import Job, Task, tk

from .chunked_clustering import prepare_worker_sys_path
from ..lib.guided_kmeans.chunked import (
    EuclideanModel,
    GaussianModel,
    HDFFeatureSource,
    NullAccumulator,
    RasrViterbiRecognizer,
    Spec,
    resolve,
    run_chunk,
)
from ..lib.guided_kmeans.chunked.diagnostics import META_NAME, FrameDiagnostics

_CHUNK_FILE = "diagnostics.{num_chunks}.{index:04d}.npz"


class ClusteringDiagnosticsJob(Job):
    """
    One recognition pass over the corpus, recorded frame by frame.

    :param features: spec for a FeatureSource; built per task with
        ``chunk``/``num_chunks`` injected
    :param model: spec for the ScoreModel to recognize with. Note which model
        an epoch's numbers belong to: epoch ``e``'s job recognizes with the
        model of epoch ``e - 1``, so to reproduce what epoch ``e`` saw, pass
        ``result.out_models[e - 1]``'s artifacts.
    :param recognizer: spec for the Recognizer
    :param lexicon: lexicon defining the label inventory; only used to record
        cluster index -> phoneme in the dump's metadata, so the analysis can
        put names on the label axis
    :param num_chunks: how many parallel tasks the pass is split into.
        UNHASHED, as for the epoch job: chunks partition the corpus by
        sequence, so the union of what they record cannot depend on the
        partition.
    """

    def __init__(
        self,
        *,
        features: Spec,
        model: Spec,
        recognizer: Spec,
        lexicon: Optional[tk.Path] = None,
        rasr_path: Optional[tk.Path] = None,
        num_chunks: int = 30,
        verbosity: int = 1,
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        self.features = features
        self.model = model
        self.recognizer = recognizer
        self.lexicon = lexicon
        self.rasr_path = rasr_path
        self.num_chunks = num_chunks
        self.verbosity = verbosity

        self.rqmt = {"cpu": 9, "mem": 16, "time": 4}
        if rqmt:
            self.rqmt.update(rqmt)

        # A directory of per-chunk .npz files plus a meta.json, following the
        # same convention as the epoch job's model output: what exactly lands
        # inside is the probe's business, and consumers reach in with
        # join_right (or just hand the directory to load_diagnostics, which is
        # what the analysis does).
        self.out_diagnostics = self.output_path("diagnostics", directory=True)

    @classmethod
    def hash(cls, kwargs):
        unhashed = {"num_chunks", "verbosity", "rqmt"}
        return super().hash(
            {
                k: (v.hashed() if isinstance(v, Spec) else v)
                for k, v in kwargs.items()
                if k not in unhashed
            }
        )

    def tasks(self):
        yield Task(
            "recognize",
            resume="recognize",
            args=range(self.num_chunks),
            rqmt=self.rqmt,
            parallel=self.num_chunks,
        )
        yield Task("collect", mini_task=True)

    def _chunk_file(self, index: int) -> str:
        # num_chunks in the name for the same reason as the epoch job's chunk
        # pickles: it is unhashed, so a re-run with a different value must not
        # silently mix records from two different partitions of the corpus.
        # collect() additionally prunes anything left over from an older one.
        return _CHUNK_FILE.format(num_chunks=self.num_chunks, index=index)

    def recognize(self, index: int):
        prepare_worker_sys_path(self.rasr_path)
        probe = FrameDiagnostics()
        run_chunk(
            features=self.features.build(chunk=index, num_chunks=self.num_chunks),
            model=self.model.build(),
            recognizer=self.recognizer.build(),
            accumulator=NullAccumulator(),
            counter=None,
            probe=probe,
            verbosity=self.verbosity,
        )
        # Into the task's cwd (the job's work/ directory), not straight into
        # the output: collect() renames them across, which is free - same
        # filesystem - and keeps a half-finished pass from looking like a
        # complete dump.
        probe.save(self._chunk_file(index))

    def collect(self):
        import numpy as np

        target = self.out_diagnostics.get_path()
        os.makedirs(target, exist_ok=True)

        expected = [self._chunk_file(i) for i in range(self.num_chunks)]
        missing = [name for name in expected if not os.path.exists(name)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} chunk file(s) missing, e.g. {missing[:3]}. "
                f"If num_chunks was changed after some chunks ran, delete "
                f"{os.getcwd()} (the job's work/ directory) and rerun."
            )

        num_sequences, num_frames = 0, 0
        for name in expected:
            with np.load(name, allow_pickle=False) as data:
                num_sequences += len(data["seq_tag"])
                num_frames += int(data["seq_num_frames"].sum())
            os.replace(name, os.path.join(target, name))

        # A partition left behind by an interrupted run with a different
        # num_chunks would otherwise be loaded alongside this one, double
        # counting every sequence it covers.
        keep = set(expected) | {META_NAME}
        for name in os.listdir(target):
            if name not in keep:
                os.remove(os.path.join(target, name))

        meta: Dict[str, Any] = {
            "num_chunks": self.num_chunks,
            "num_sequences": num_sequences,
            "num_frames": num_frames,
            # The scores in the dump are the model's own, before the recognizer
            # scales them; recording the scale is what lets the analysis
            # reconstruct what the search was actually comparing.
            "distance_scale": resolve(self.recognizer.kwargs.get("distance_scale", 1.0)),
            "model": {
                name: resolve(value) for name, value in sorted(self.model.kwargs.items())
            },
        }
        if self.lexicon is not None:
            from ..lib.guided_kmeans.chunked.recognizers import PhonemeIdxMap

            inverse = PhonemeIdxMap(self.lexicon.get_path()).inverse()
            meta["labels"] = [inverse[i] for i in range(len(inverse))]

        with open(os.path.join(target, META_NAME), "w") as fp:
            json.dump(meta, fp, indent=2, default=str)

        print(f"Diagnostics pass done: {num_sequences} sequences, {num_frames} frames")


def clustering_diagnostics(
    *,
    features_hdf: Union[tk.Path, Sequence[tk.Path]],
    recognition_config: tk.Path,
    lexicon: tk.Path,
    centroids: tk.Path,
    covs: Optional[tk.Path] = None,
    segments: Optional[tk.Path] = None,
    subsampling: Optional[int] = None,
    pooling_function: str = "maxpool_time_np",
    distance_scale: float = 1.0,
    rasr_path: Optional[tk.Path] = None,
    num_chunks: int = 30,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
    rqmt: Optional[Dict[str, Any]] = None,
    alias: Optional[str] = None,
) -> ClusteringDiagnosticsJob:
    """
    Record a diagnostics pass for one model of a clustering run.

    The parameters mirror :func:`.chunked_clustering.chunked_clustering` name
    for name, so inspecting a run means repeating its call with the model of
    interest substituted::

        diag = clustering_diagnostics(
            features_hdf=..., recognition_config=recognition_config,
            lexicon=lexicon, distance_scale=..., subsampling=..., rasr_path=...,
            centroids=exp_result.out_centroids[0],
            covs=exp_result.out_covs[0],
            segments=a_few_hundred_segments,   # optional, and much cheaper
            alias=f"guided_kmeans/{exp_dir}/diagnostics/{exp_name}_epoch-0",
        )
        tk.register_output(f"guided_kmeans/{exp_dir}/diagnostics/epoch-0",
                           diag.out_diagnostics)

    Anything that differs from the run being inspected - a different
    ``recognition_config``, a different ``distance_scale`` - makes the dump
    describe a different search, so keep them in sync unless the difference is
    the point of the experiment.

    :param centroids: the model to recognize with, e.g.
        ``exp_result.out_centroids[epoch]``
    :param covs: pass alongside ``centroids`` to score with full covariances,
        exactly as ``initial_covs`` selects the model class for a run
    :param segments: restrict to a subset of the corpus. Recognition dominates
        the cost (~17 s of CPU per sequence), so a few hundred segments answer
        a distribution question in minutes rather than half an hour.
    """
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
    recognizer_spec = Spec(
        RasrViterbiRecognizer,
        {
            "recognition_config": recognition_config,
            "lexicon_path": lexicon,
            "distance_scale": distance_scale,
        },
        {"num_workers": num_workers, "task_timeout": task_timeout},
    )
    if covs is not None:
        model_spec = Spec(GaussianModel, {"centroids": centroids, "covs": covs})
    else:
        model_spec = Spec(EuclideanModel, {"centroids": centroids})

    job_rqmt = {"cpu": num_workers + 1, "mem": 16, "time": 4}
    if rqmt:
        job_rqmt.update(rqmt)

    job = ClusteringDiagnosticsJob(
        features=features_spec,
        model=model_spec,
        recognizer=recognizer_spec,
        lexicon=lexicon,
        rasr_path=rasr_path,
        num_chunks=num_chunks,
        rqmt=job_rqmt,
    )
    if alias:
        job.add_alias(alias)
    return job
