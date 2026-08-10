"""
Chunked guided k-means: one clustering epoch split across independent tasks.

The single-process pipeline (``..clustering.GuidedKMeansClusteringCallback``
driven by a ``ReturnnForwardJobV2``) is unchanged and still available; this is
an alternative execution model for the same algorithm, built so that RASR
recognition - which dominates epoch wall time - can be spread over the cluster
instead of one node's worker pool, and so that a killed run resumes at chunk
granularity.

Injection points, all supplied as :class:`.spec.Spec` objects:

* :mod:`.features`     - where encoder outputs come from
* :mod:`.models`       - how features are scored against clusters
* :mod:`.recognizers`  - how scores become per-frame labels
* :mod:`.accumulators` - how labelled frames update the model
"""

from .spec import Spec, resolve
from .interfaces import (
    Accumulator,
    FeatureSource,
    Posteriors,
    Recognizer,
    ScoreModel,
    as_hard_labels,
    as_responsibilities,
)
from .features import HDFFeatureSource, plan_chunks, read_segment_file
from .models import (
    ArtifactModel,
    EuclideanModel,
    GaussianModel,
    load_model,
    read_manifest,
)
from .recognizers import PhonemeIdxMap, RasrViterbiRecognizer, SerialRasrRecognizer
from .accumulators import GaussianAccumulator, MeanAccumulator, keep_previous_where_dead
from .runner import ChunkResult, load_chunk, reduce_chunks, run_chunk, save_chunk
from .stats import default_stats_hooks, merge_counters

__all__ = [
    "Spec",
    "resolve",
    "Accumulator",
    "FeatureSource",
    "Posteriors",
    "Recognizer",
    "ScoreModel",
    "as_hard_labels",
    "as_responsibilities",
    "HDFFeatureSource",
    "plan_chunks",
    "read_segment_file",
    "ArtifactModel",
    "EuclideanModel",
    "GaussianModel",
    "load_model",
    "read_manifest",
    "PhonemeIdxMap",
    "RasrViterbiRecognizer",
    "SerialRasrRecognizer",
    "GaussianAccumulator",
    "MeanAccumulator",
    "keep_previous_where_dead",
    "ChunkResult",
    "load_chunk",
    "reduce_chunks",
    "run_chunk",
    "save_chunk",
    "default_stats_hooks",
    "merge_counters",
]
