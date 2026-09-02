"""
One consistent choice of model, updating routine and search, as a single
object.

Every combination of these has to agree on more than the pipeline can check:
a covariance model needs an accumulator that estimates covariances, a
forward-backward search hands back dense gammas that a Welford accumulator
cannot weight, a mixture model needs the accumulator that runs its E-step. The
pipeline used to encode those agreements as a cross product of boolean flags
(``initial_covs is not None`` x ``use_forward_backward``), which meant a new
model was a new branch inside :func:`...setup.chunked_clustering.chunked_clustering`
and a new flag in every caller's signature.

Here instead each combination is one factory returning a
:class:`ClusteringFlavor`. Adding a model is adding a factory: the pipeline
never names a model class, an accumulator class or a recognizer class, and the
existing flag-based entry point keeps working by calling these itself.

Deliberately in the algorithm layer rather than next to the jobs: a flavor is
:class:`.spec.Spec` objects, which are plain descriptions with no sisyphus
machinery in them, and keeping them here means the combinations can be built
and inspected in a test without constructing a job graph.
"""

from __future__ import annotations

__all__ = [
    "ClusteringFlavor",
    "euclidean_flavor",
    "unguided_flavor",
    "gaussian_flavor",
    "mixture_flavor",
    "per_label_mixture_flavor",
]

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..statistics import FBStatisticsCounter
from .accumulators import (
    FixedCovarianceAccumulator,
    GaussianAccumulator,
    MeanAccumulator,
    MixtureGaussianAccumulator,
    SoftGaussianAccumulator,
)
from .models import (
    EuclideanModel,
    GaussianMixtureModel,
    GaussianModel,
    PerLabelMixtureModel,
)
from .recognizers import ArgmaxRecognizer, RasrFBRecognizer, RasrViterbiRecognizer
from .spec import Spec


@dataclass(frozen=True)
class ClusteringFlavor:
    """
    What an epoch is made of: the model it starts from, the search that guides
    it, the routine that re-estimates it, and what it records about itself.

    :param model: spec for the *initial* model. Later epochs' specs are derived
        from it by the pipeline, which swaps the artifact paths for the
        previous epoch's outputs and changes nothing else - that sameness of
        shape is what makes continuing a run reuse jobs, so a flavor must not
        special-case the first epoch.
    :param accumulator: spec for the Accumulator; the epoch job injects
        ``num_clusters``, and anything else an accumulator needs it either
        takes from the spec or from the model handed to it by ``bind_model``.
    :param recognizer: spec for the Recognizer
    :param statistics: optional spec for a statistics counter. Unhashed by the
        job - which diagnostics get recorded must not change its identity.
    """

    model: Spec
    accumulator: Spec
    recognizer: Spec
    statistics: Optional[Spec] = None

    def __post_init__(self):
        names = self.artifact_names
        if not names:
            raise ValueError(
                f"{self.model.cls.__name__} declares no ARTIFACT_NAMES, so the "
                f"pipeline cannot derive the next epoch's model spec from it"
            )
        missing = set(names) - set(self.model.kwargs)
        if missing:
            raise ValueError(
                f"initial model spec for {self.model.cls.__name__} is missing "
                f"{sorted(missing)}; every artifact needs a starting value, or "
                f"epoch 1's spec will not have the shape later epochs' specs do"
            )

    @property
    def artifact_names(self) -> Tuple[str, ...]:
        """
        The model's artifact set, taken from the class rather than tracked
        alongside it - see :attr:`.models.ArtifactModel.ARTIFACT_NAMES`.
        """
        return tuple(getattr(self.model.cls, "ARTIFACT_NAMES", ()))

    def next_model(self, artifacts: Dict[str, Any]) -> Spec:
        """
        The same spec pointing at a later epoch's artifacts. Structurally
        identical to :attr:`model` by construction, which is the property run
        continuation rests on.
        """
        missing = set(self.artifact_names) - set(artifacts)
        if missing:
            raise ValueError(f"missing artifacts for the next epoch: {sorted(missing)}")
        return Spec(self.model.cls, {name: artifacts[name] for name in self.artifact_names})


def _recognizer_spec(
    *,
    recognition_config,
    lexicon,
    num_clusters: int,
    distance_scale: float,
    use_forward_backward: bool,
    num_workers: int,
    task_timeout: Optional[float],
) -> Spec:
    # num_workers/task_timeout are per-task scheduling knobs, like num_chunks:
    # they change how fast a chunk runs, never its result.
    unhashed = {"num_workers": num_workers, "task_timeout": task_timeout}
    if use_forward_backward:
        return Spec(
            RasrFBRecognizer,
            {
                "recognition_config": recognition_config,
                "num_clusters": num_clusters,
                "distance_scale": distance_scale,
            },
            unhashed,
        )
    return Spec(
        RasrViterbiRecognizer,
        {
            "recognition_config": recognition_config,
            "lexicon_path": lexicon,
            "distance_scale": distance_scale,
        },
        unhashed,
    )


def _statistics_spec(num_clusters: int, use_forward_backward: bool) -> Optional[Spec]:
    # FBStatisticsCounter replaces the Viterbi traceback counters for FB epochs;
    # a Viterbi epoch gets its counters built from the lexicon by the job.
    if not use_forward_backward:
        return None
    return Spec(FBStatisticsCounter, {"num_clusters": num_clusters})


def euclidean_flavor(
    *,
    centroids,
    recognition_config,
    lexicon,
    num_clusters: int,
    distance_scale: float = 1.0,
    use_forward_backward: bool = False,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
) -> ClusteringFlavor:
    """Plain k-means: squared Euclidean scoring, centroids from frame means."""
    return ClusteringFlavor(
        model=Spec(EuclideanModel, {"centroids": centroids}),
        accumulator=Spec(MeanAccumulator, {}),
        recognizer=_recognizer_spec(
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            num_workers=num_workers,
            task_timeout=task_timeout,
        ),
        statistics=_statistics_spec(num_clusters, use_forward_backward),
    )


def unguided_flavor(
    *,
    centroids,
    covs=None,
    num_clusters: int,
    min_mass: float = 0.0,
) -> ClusteringFlavor:
    """
    Plain k-means: no search, no lexicon, no language model.

    Every frame takes its own nearest cluster
    (:class:`.recognizers.ArgmaxRecognizer`) and the means move to match. This
    is the unguided partition-finding pass - it discovers where the data
    concentrates, with nothing said about what the clusters mean, and produces
    a codebook for a later guided stage to attach labels to.

    ``num_clusters`` here is a free choice rather than the size of a label
    inventory, which is the substantive difference from every other flavor in
    this module: with no lexicon there is nothing requiring the cluster count
    to be 40, and sweeping it is the point.

    :param covs: the covariance to score under, ``[K, D, D]``. **Held fixed** -
        it is what the metric is, not something this pass estimates. Give it
        one shared covariance duplicated across the clusters
        (``DuplicateCovsJob(GlobalCovarianceJob(features).out_cov, K)``) and
        the run is k-means in the globally whitened space, which is what you
        want for encoder features: without it, squared Euclidean distance
        partitions along whichever dimensions the encoder happened to give the
        largest scale, and the partition describes that scaling as much as the
        data. Omit it for genuinely plain squared-Euclidean k-means.

        Per-cluster covariances are deliberately *not* estimated here even
        though the model could carry them; see
        :class:`.accumulators.FixedCovarianceAccumulator` for why separating
        the partition from the covariance is the point rather than an omission.
    :param min_mass: mass a cluster needs before its mean moves. The default
        keeps any cluster that got at least one frame; a cluster that got none
        keeps its previous mean and stays a reachable candidate rather than
        collapsing to the origin.
    """
    if covs is None:
        return ClusteringFlavor(
            model=Spec(EuclideanModel, {"centroids": centroids}),
            accumulator=Spec(MeanAccumulator, {}),
            recognizer=Spec(ArgmaxRecognizer, {"num_clusters": num_clusters}),
        )
    return ClusteringFlavor(
        model=Spec(GaussianModel, {"centroids": centroids, "covs": covs}),
        accumulator=Spec(FixedCovarianceAccumulator, {"min_mass": min_mass}),
        recognizer=Spec(ArgmaxRecognizer, {"num_clusters": num_clusters}),
    )


def gaussian_flavor(
    *,
    centroids,
    covs,
    recognition_config,
    lexicon,
    num_clusters: int,
    distance_scale: float = 1.0,
    use_forward_backward: bool = False,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
) -> ClusteringFlavor:
    """
    One full-covariance Gaussian per label.

    The accumulator follows the search: a Viterbi alignment gives hard labels,
    which Welford's method handles directly and better-conditioned; a
    forward-backward pass gives dense gammas, which it cannot weight, so those
    go to the raw-moment accumulator instead.
    """
    return ClusteringFlavor(
        model=Spec(GaussianModel, {"centroids": centroids, "covs": covs}),
        accumulator=Spec(
            SoftGaussianAccumulator if use_forward_backward else GaussianAccumulator, {}
        ),
        recognizer=_recognizer_spec(
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            num_workers=num_workers,
            task_timeout=task_timeout,
        ),
        statistics=_statistics_spec(num_clusters, use_forward_backward),
    )


def _mixture_flavor(
    model_cls,
    *,
    centroids,
    covs,
    mixtures,
    recognition_config,
    lexicon,
    num_clusters: int,
    distance_scale: float,
    use_forward_backward: bool,
    min_mass: float,
    mixture_floor: float,
    pool_covariances: bool,
    update_densities: bool,
    num_workers: int,
    task_timeout: Optional[float],
) -> ClusteringFlavor:
    # Both mixture layouts wire up identically - same artifacts, same
    # accumulator, same search - and differ only in the model class, because
    # the layout-specific arithmetic all lives behind the model's
    # responsibilities(). That is the payoff of putting the E-step there.
    accumulator_args = {
        "min_mass": min_mass,
        "mixture_floor": mixture_floor,
    }
    # Both are set only when they differ from the accumulator's default, so
    # adding either left every existing run's job hash untouched. Keep that up:
    # an argument that is always present changes the spec, and the spec is
    # hashed.
    if pool_covariances:
        accumulator_args["pool_covariances"] = True
    if not update_densities:
        accumulator_args["update_densities"] = False
    return ClusteringFlavor(
        model=Spec(model_cls, {"centroids": centroids, "covs": covs, "mixtures": mixtures}),
        accumulator=Spec(MixtureGaussianAccumulator, accumulator_args),
        recognizer=_recognizer_spec(
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            num_workers=num_workers,
            task_timeout=task_timeout,
        ),
        statistics=_statistics_spec(num_clusters, use_forward_backward),
    )


def mixture_flavor(
    *,
    centroids,
    covs,
    mixtures,
    recognition_config,
    lexicon,
    num_clusters: int,
    distance_scale: float = 1.0,
    use_forward_backward: bool = False,
    min_mass: float = 1.0,
    mixture_floor: float = 0.0,
    pool_covariances: bool = False,
    update_densities: bool = True,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
) -> ClusteringFlavor:
    """
    A shared codebook of densities, with per-label weights over all of them
    (:class:`.models.GaussianMixtureModel`).

    Works with either search: the density-level E-step takes label posteriors
    in whatever form the recognizer reports them, and a Viterbi alignment is
    just the one-hot case. The density count is not a parameter - it follows
    from the shape of ``mixtures``, and the accumulator reads it off the model
    it gets bound to.

    :param update_densities: False trains the mixture weights alone against a
        frozen codebook, which is what pairs this with a codebook found by an
        earlier :func:`unguided_flavor` run: the partition is decided there,
        and all this stage learns is which densities each label draws on. It
        also drops the only O(D^2) statistic in the pipeline - see
        :class:`.accumulators.MixtureGaussianAccumulator`.

        Note the interaction with ``mixture_floor``: with the densities frozen,
        the weights are the only parameters left, so a density a label loses
        cannot be compensated for by the remaining ones moving. A zero weight
        is absorbing at the default floor of 0.0, so a frozen-codebook run
        wants a small nonzero floor unless the shrinking is intended.
    """
    return _mixture_flavor(
        GaussianMixtureModel,
        centroids=centroids,
        covs=covs,
        mixtures=mixtures,
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=num_clusters,
        distance_scale=distance_scale,
        use_forward_backward=use_forward_backward,
        min_mass=min_mass,
        mixture_floor=mixture_floor,
        pool_covariances=pool_covariances,
        update_densities=update_densities,
        num_workers=num_workers,
        task_timeout=task_timeout,
    )


def per_label_mixture_flavor(
    *,
    centroids,
    covs,
    mixtures,
    recognition_config,
    lexicon,
    num_clusters: int,
    distance_scale: float = 1.0,
    use_forward_backward: bool = False,
    min_mass: float = 1.0,
    mixture_floor: float = 0.0,
    pool_covariances: bool = False,
    update_densities: bool = True,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
) -> ClusteringFlavor:
    """
    ``n`` densities per label, owned by that label
    (:class:`.models.PerLabelMixtureModel`).

    ``mixtures`` is ``[L, n]`` and ``centroids`` is ``[L * n, D]`` with label
    ``l``'s densities contiguous at ``l * n``. ``pool_covariances=True`` ties the
    covariance across a label's densities, which is what makes a large ``n``
    affordable - see :class:`.accumulators.MixtureGaussianAccumulator`. Use
    :class:`...setup.chunked_clustering.SplitCentroidsJob` and
    :class:`...setup.chunked_clustering.UniformMixturesJob` to build a starting
    point in that layout from an existing set of per-label centroids.
    """
    return _mixture_flavor(
        PerLabelMixtureModel,
        centroids=centroids,
        covs=covs,
        mixtures=mixtures,
        recognition_config=recognition_config,
        lexicon=lexicon,
        num_clusters=num_clusters,
        distance_scale=distance_scale,
        use_forward_backward=use_forward_backward,
        min_mass=min_mass,
        mixture_floor=mixture_floor,
        pool_covariances=pool_covariances,
        update_densities=update_densities,
        num_workers=num_workers,
        task_timeout=task_timeout,
    )
