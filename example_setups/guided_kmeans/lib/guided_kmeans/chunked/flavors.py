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
    "BackoffFBSearch",
    "ClusteringFlavor",
    "euclidean_flavor",
    "unguided_flavor",
    "gaussian_flavor",
    "mixture_flavor",
    "per_label_mixture_flavor",
    "vq_flavor",
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
    VectorQuantizedAccumulator,
)
from .models import (
    EuclideanModel,
    GaussianMixtureModel,
    GaussianModel,
    PerLabelMixtureModel,
    VectorQuantizedModel,
)
from .recognizers import (
    ArgmaxRecognizer,
    BackoffFBRecognizer,
    OracleSegmentFBRecognizer,
    RasrFBRecognizer,
    RasrViterbiRecognizer,
)
from .spec import Spec


@dataclass(frozen=True)
class BackoffFBSearch:
    """Run the training search on the GPU with the ``backoff_fb`` CUDA op
    instead of the RASR worker pool.

    Passed to a flavor in place of nothing; when present it replaces the
    recognizer and the RASR config is then only used by the decode side. The
    search parameters are named separately rather than read back out of the
    ``RasrConfig`` because they have to reach the graph compiler as numbers,
    and because stating them here makes the two paths comparable by
    inspection.

    ``apply_sentence_end=False`` drops ``ln P(</s> | g)`` from the final
    weight, which is what makes the op agree with RASR's forward-backward; True
    is the correct model. Changing it changes the job hash, as it must.

    Giving ``frame_labels`` switches to
    :class:`.recognizers.OracleSegmentFBRecognizer`: the features are then
    frames and the segmentation is taken from that alignment instead of being
    searched for.

    ``backoff_fb_root`` is where the op is checked out: pass
    ``tools.BACKOFF_FB``. It is hashed, so a different checkout is a different
    tool, the same bargain the RASR paths make.

    ``lexicon`` must be the *forward-backward* lexicon
    (:func:`...setup.librasr_recognition.create_fb_lexicon`), which is what
    fixes the score column order.

    :param batch_size, max_frames, checkpoint_interval: throughput and memory
        knobs, unhashed. ``checkpoint_interval`` is provably result-neutral -
        the op's K1 test asserts bitwise identity against storing every frame.
    """

    backoff_fb_root: Any
    arpa_path: Any
    lexicon: Any
    lm_scale: float = 1.0
    emission_scale: float = 1.0
    loop_probability: float = 0.0
    silence_loop_probability: Optional[float] = None
    transition_scale: float = 1.0
    #: HMM states per phoneme, uniform across the inventory. 1 makes a
    #: phoneme's duration geometric with mean ``1 / (1 - loop_probability)``;
    #: 2 makes it negative-binomial with mean ``2 / (1 - loop_probability)``,
    #: which has an interior mode where the geometric always peaks at one
    #: frame. See :func:`..backoff_fb_graph.build_topology`.
    num_states: int = 1
    #: Whether a phoneme's states share one emission label. False gives every
    #: state its own row in the model's table, so the run's ``num_clusters``
    #: must be ``V * num_states`` (phoneme-major, see
    #: :class:`..backoff_fb_graph.LabelLayout`). Only meaningful with
    #: ``num_states > 1``.
    tie_states: bool = True
    apply_sentence_end: bool = True
    #: Per-frame reference labels (``out_labels`` of
    #: ``SegmentedFeaturesFromAlignmentJob(pooling="none")``). Given one, the
    #: features are frames and the boundaries between them come from this
    #: alignment rather than from the search -- see
    #: :class:`.recognizers.OracleSegmentFBRecognizer`. ``loop_probability``
    #: must then stay 0.
    frame_labels: Any = None
    #: Per-frame transition scores: a :class:`.spec.Spec` for a
    #: :class:`.interfaces.TransitionModel`, e.g.
    #: ``Spec(OracleBoundaryTransitions, {"frame_labels": ...})``. The features
    #: are then frames, and the scores are added to the static transition
    #: weights. Only for a fixed model so far; see :class:`ClusteringFlavor`.
    transitions: Any = None
    #: Every static transition weight 0, so that ``transitions`` is the whole
    #: transition model. The loop parameters must then stay at their defaults.
    #: See :func:`..backoff_fb_graph.build_topology`.
    zero_topology: bool = False
    batch_size: int = 16
    max_frames: int = 0
    checkpoint_interval: int = -1
    storage_dtype: str = "float32"


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
    :param transitions: optional spec for a
        :class:`.interfaces.TransitionModel` whose per-frame scores the
        recognizer adds to its static transition weights. Held fixed for the
        whole run: a trainable one is refused until it can be re-estimated.
    :param transition_accumulator: spec for the
        :class:`.interfaces.TransitionAccumulator` that would re-estimate a
        trainable transition model. Not implemented: its input, the transition
        posteriors, comes from the op's milestone T3, and the epoch job does not
        yet merge, save or carry transition statistics from one epoch to the
        next.
    """

    model: Spec
    accumulator: Spec
    recognizer: Spec
    statistics: Optional[Spec] = None
    transitions: Optional[Spec] = None
    transition_accumulator: Optional[Spec] = None

    def __post_init__(self):
        if self.transition_accumulator is not None:
            raise NotImplementedError(
                "re-estimating a transition model is not implemented: it needs "
                "the transition posteriors from backoff_fb milestone T3 (SPEC.md "
                "14.8), and the epoch job does not yet merge, save or carry "
                "transition statistics")
        if self.transitions is not None and getattr(self.transitions.cls, "trainable", False):
            raise NotImplementedError(
                f"{self.transitions.cls.__name__} is trainable, and a trainable "
                f"transition model cannot be re-estimated yet; it would run "
                f"every epoch with its initial parameters")
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
    backoff_fb: Optional[BackoffFBSearch] = None,
) -> Spec:
    # num_workers/task_timeout are per-task scheduling knobs, like num_chunks:
    # they change how fast a chunk runs, never its result.
    unhashed = {"num_workers": num_workers, "task_timeout": task_timeout}
    if backoff_fb is not None:
        if not use_forward_backward:
            raise ValueError(
                "BackoffFBSearch is a forward-backward search; pass "
                "use_forward_backward=True alongside it")
        cls = BackoffFBRecognizer
        extra = {}
        if backoff_fb.frame_labels is not None:
            cls = OracleSegmentFBRecognizer
            extra["frame_labels"] = backoff_fb.frame_labels
        if backoff_fb.num_states != 1:
            # Only when it differs from the default: adding the key
            # unconditionally would rehash every S=1 epoch job that already
            # exists (the finished vq_gmm_tau_cuda arms, the in-flight oracle
            # sweep) over a parameter whose value did not change. Same bargain
            # a job's __sis_hash_exclude__ strikes.
            extra["num_states"] = backoff_fb.num_states
        if not backoff_fb.tie_states:
            # Same bargain: hashed only when untied, so every tied job keeps
            # its identity. Untying a single state is the tied model under a
            # second hash, so it is refused rather than allowed to duplicate.
            if backoff_fb.num_states == 1:
                raise ValueError(
                    "tie_states=False needs num_states > 1: with one state per "
                    "phoneme there is nothing to untie")
            if backoff_fb.frame_labels is not None:
                raise ValueError(
                    "untied states need a frame-level search; the oracle-segment "
                    "recognizer sums each segment into one observation")
            extra["tie_states"] = False
        if backoff_fb.transitions is not None and backoff_fb.frame_labels is not None:
            raise ValueError(
                "transitions and frame_labels are two routes to the same oracle: "
                "frame_labels sums each segment into one observation, while "
                "transition scores are per frame. Pass one of them")
        if backoff_fb.zero_topology:
            # Same bargain as num_states: hashed only when set.
            if backoff_fb.transitions is None:
                raise ValueError(
                    "zero_topology=True without transitions leaves every "
                    "segmentation equally likely, which is loop_probability=0.5 "
                    "with a shifted log Z; say that instead, or pass the "
                    "transition model the zero topology is meant for")
            if backoff_fb.frame_labels is not None:
                raise ValueError(
                    "zero_topology makes the loop free, which the oracle-segment "
                    "recognizer (frame_labels) must not have")
            if (backoff_fb.loop_probability != 0.0
                    or backoff_fb.silence_loop_probability is not None
                    or backoff_fb.transition_scale != 1.0):
                raise ValueError(
                    "zero_topology=True sets every static transition weight to "
                    "0; leave loop_probability, silence_loop_probability and "
                    "transition_scale at their defaults")
            extra["zero_topology"] = True
        return Spec(
            cls,
            {
                **extra,
                "backoff_fb_root": backoff_fb.backoff_fb_root,
                "arpa_path": backoff_fb.arpa_path,
                "lexicon_path": backoff_fb.lexicon,
                "num_clusters": num_clusters,
                "distance_scale": distance_scale,
                "emission_scale": backoff_fb.emission_scale,
                "lm_scale": backoff_fb.lm_scale,
                "loop_probability": backoff_fb.loop_probability,
                "silence_loop_probability": backoff_fb.silence_loop_probability,
                "transition_scale": backoff_fb.transition_scale,
                "apply_sentence_end": backoff_fb.apply_sentence_end,
            },
            {
                "batch_size": backoff_fb.batch_size,
                "max_frames": backoff_fb.max_frames,
                "checkpoint_interval": backoff_fb.checkpoint_interval,
                "storage_dtype": backoff_fb.storage_dtype,
            },
        )
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


def _statistics_spec(
    num_clusters: int,
    use_forward_backward: bool,
    backoff_fb: Optional[BackoffFBSearch] = None,
) -> Optional[Spec]:
    # FBStatisticsCounter replaces the Viterbi traceback counters for FB epochs;
    # a Viterbi epoch gets its counters built from the lexicon by the job.
    if not use_forward_backward:
        return None
    kwargs = {"num_clusters": num_clusters}
    if backoff_fb is not None and not backoff_fb.tie_states:
        # Untied posteriors have one column per state; the counter folds them
        # back into phonemes. Added only then, so a tied spec stays as it was.
        kwargs.update(num_states=backoff_fb.num_states, tie_states=False)
    return Spec(FBStatisticsCounter, kwargs)


def _transitions_spec(backoff_fb: Optional[BackoffFBSearch]) -> Optional[Spec]:
    # Only the backoff_fb search takes transition scores, so a RASR flavor
    # cannot get one by construction.
    return backoff_fb.transitions if backoff_fb is not None else None


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
    backoff_fb: Optional[BackoffFBSearch] = None,
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
            backoff_fb=backoff_fb,
        ),
        statistics=_statistics_spec(num_clusters, use_forward_backward, backoff_fb),
        transitions=_transitions_spec(backoff_fb),
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
    backoff_fb: Optional[BackoffFBSearch] = None,
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
            backoff_fb=backoff_fb,
        ),
        statistics=_statistics_spec(num_clusters, use_forward_backward, backoff_fb),
        transitions=_transitions_spec(backoff_fb),
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
    update_covariances,
    num_workers: int,
    task_timeout: Optional[float],
    backoff_fb: Optional[BackoffFBSearch] = None,
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
    if update_covariances is not None:
        accumulator_args["update_covariances"] = update_covariances
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
            backoff_fb=backoff_fb,
        ),
        statistics=_statistics_spec(num_clusters, use_forward_backward, backoff_fb),
        transitions=_transitions_spec(backoff_fb),
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
    update_covariances: Optional[bool] = None,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
    backoff_fb: Optional[BackoffFBSearch] = None,
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
        update_covariances=update_covariances,
        num_workers=num_workers,
        task_timeout=task_timeout,
        backoff_fb=backoff_fb,
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
    update_covariances: Optional[bool] = None,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
    backoff_fb: Optional[BackoffFBSearch] = None,
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
        update_covariances=update_covariances,
        num_workers=num_workers,
        task_timeout=task_timeout,
        backoff_fb=backoff_fb,
    )


def vq_flavor(
    *,
    centroids,
    table,
    recognition_config,
    lexicon,
    num_clusters: int,
    distance_scale: float = 1.0,
    use_forward_backward: bool = False,
    table_floor: float = 0.0,
    min_mass: float = 0.0,
    num_workers: int = 8,
    task_timeout: Optional[float] = 1800.0,
    backoff_fb: Optional[BackoffFBSearch] = None,
) -> ClusteringFlavor:
    """
    A discrete-density HMM over a frozen codebook
    (:class:`.models.VectorQuantizedModel`).

    The whole model is ``p(codeword | label)``: an epoch quantizes each frame,
    counts which label the search assigned it to, and normalizes. There are no
    covariances and no feature-dimension arithmetic in the update at all, so
    this is the cheapest flavor here by a wide margin - the per-chunk statistic
    is ``[L, C]``, 160 kB at 40 labels and 512 codewords.

    Takes either search: a Viterbi alignment is the one-hot case of the dense
    posteriors forward-backward reports, and the counting is unchanged by
    which arrives.

    :param centroids: the codebook, held fixed for the whole run. Quantization
        is plain L2, so a codebook trained with a k-means library partitions
        here exactly as it did there.
    :param table: initial ``p(codeword | label)``. This is the only thing a run
        learns, so it is also the only place a run's symmetry can be broken -
        see :class:`...setup.chunked_clustering.NormalTableJob`.
    :param table_floor: smoothing applied at every re-estimation. Worth
        setting: a zero entry is an ``+inf`` score and is absorbing under a
        frozen codebook, and a codeword no label admits leaves a frame with no
        viable label at all.
    """
    return ClusteringFlavor(
        model=Spec(VectorQuantizedModel, {"centroids": centroids, "table": table}),
        accumulator=Spec(
            VectorQuantizedAccumulator,
            {"table_floor": table_floor, "min_mass": min_mass},
        ),
        recognizer=_recognizer_spec(
            recognition_config=recognition_config,
            lexicon=lexicon,
            num_clusters=num_clusters,
            distance_scale=distance_scale,
            use_forward_backward=use_forward_backward,
            num_workers=num_workers,
            task_timeout=task_timeout,
            backoff_fb=backoff_fb,
        ),
        statistics=_statistics_spec(num_clusters, use_forward_backward, backoff_fb),
        transitions=_transitions_spec(backoff_fb),
    )
