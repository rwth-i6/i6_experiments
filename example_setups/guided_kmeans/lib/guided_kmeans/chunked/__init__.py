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
* :mod:`.diagnostics`  - what a pass records about itself, changing nothing
"""

from .spec import Spec, resolve
from .interfaces import (
    Accumulator,
    FeatureSource,
    Posteriors,
    Probe,
    RecognitionResult,
    Recognizer,
    ScoreModel,
    as_dense_responsibilities,
    as_hard_labels,
    as_responsibilities,
)
from .features import HDFFeatureSource, cached_file, plan_chunks, read_segment_file
from .models import (
    ArtifactModel,
    EuclideanModel,
    GaussianMixtureModel,
    GaussianModel,
    MixtureModelBase,
    PerLabelMixtureModel,
    load_forward_model,
    load_model,
    neg_log_matmul,
    read_manifest,
)
from .recognizers import (
    ArgmaxRecognizer,
    PhonemeIdxMap,
    RasrFBRecognizer,
    RasrViterbiRecognizer,
    SerialRasrRecognizer,
)
from .accumulators import (
    FixedCovarianceAccumulator,
    GaussianAccumulator,
    MeanAccumulator,
    MixtureGaussianAccumulator,
    NullAccumulator,
    SoftGaussianAccumulator,
    alive_mask,
    if_alive_else,
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
from .runner import ChunkResult, load_chunk, reduce_chunks, run_chunk, save_chunk
from .stats import default_stats_hooks, fb_stats_hooks, merge_counters
from .diagnostics import Diagnostics, FrameDiagnostics, load_diagnostics

__all__ = [
    "Spec",
    "resolve",
    "Accumulator",
    "FeatureSource",
    "Posteriors",
    "Probe",
    "RecognitionResult",
    "Recognizer",
    "ScoreModel",
    "as_dense_responsibilities",
    "as_hard_labels",
    "as_responsibilities",
    "HDFFeatureSource",
    "cached_file",
    "plan_chunks",
    "read_segment_file",
    "ArtifactModel",
    "EuclideanModel",
    "GaussianMixtureModel",
    "GaussianModel",
    "MixtureModelBase",
    "PerLabelMixtureModel",
    "load_forward_model",
    "load_model",
    "neg_log_matmul",
    "read_manifest",
    "ArgmaxRecognizer",
    "PhonemeIdxMap",
    "RasrFBRecognizer",
    "RasrViterbiRecognizer",
    "SerialRasrRecognizer",
    "FixedCovarianceAccumulator",
    "GaussianAccumulator",
    "MeanAccumulator",
    "MixtureGaussianAccumulator",
    "SoftGaussianAccumulator",
    "NullAccumulator",
    "alive_mask",
    "if_alive_else",
    "keep_previous_where_dead",
    "ClusteringFlavor",
    "euclidean_flavor",
    "unguided_flavor",
    "gaussian_flavor",
    "mixture_flavor",
    "per_label_mixture_flavor",
    "ChunkResult",
    "load_chunk",
    "reduce_chunks",
    "run_chunk",
    "save_chunk",
    "default_stats_hooks",
    "fb_stats_hooks",
    "merge_counters",
    "Diagnostics",
    "FrameDiagnostics",
    "load_diagnostics",
]
