"""
Score models: encoder features in, per-frame per-cluster costs out.

A model is defined by its *artifacts* - a name -> array mapping covering
everything needed to reconstruct it. Persistence is then generic: each
artifact is written as ``<name>.npy`` next to a ``model.json`` manifest
listing them. Adding a model with a different parameter set (diagonal
variances, priors, ...) requires no changes to the job, the pipeline, or the
reduce step - which is why the epoch job declares only the model *directory*
as its output and lets consumers reach inside it with ``join_right``.

Artifacts are assumed to be indexed by cluster on their first axis, which is
what lets :mod:`.accumulators` apply the "keep the previous value for clusters
that got no data" rule without knowing any model's parameter set. A model
carrying a genuinely global parameter would need that rule extended; the
assertion in the accumulators points at the spot.

Constructors take each artifact either as an array or as a path to a ``.npy``.
The path form is what the pipeline's model :class:`.Spec` uses - ``Spec.build``
resolves a ``tk.Path`` to a string - so a model spec reads the same whether it
points at loose input files or at ``previous_job.artifact("centroids")``. That
uniformity is what makes continuing a run produce the same job hashes as
running it in one go.

The two implementations delegate scoring to the exact expressions the
single-process callback uses, so a chunked run stays numerically comparable to
the pipeline it replaces. They are adapters, not reimplementations - resist
the urge to "improve" the arithmetic, or the equivalence tests stop meaning
anything.
"""

from __future__ import annotations

__all__ = [
    "ArtifactModel",
    "EuclideanModel",
    "GaussianModel",
    "GaussianMixtureModel",
    "MixtureModelBase",
    "PerLabelMixtureModel",
    "neg_log_matmul",
    "load_model",
    "read_manifest",
    "MODEL_CLASSES",
    "MANIFEST_NAME",
]

import json
import os
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

from ..model import GaussianModelNumpy

MANIFEST_NAME = "model.json"

#: Populated automatically by every :class:`ArtifactModel` subclass, so
#: :func:`load_model` can reconstruct a model directory written by a class it
#: was never told about - the module defining it only has to be imported.
MODEL_CLASSES: Dict[str, type] = {}


def neg_log_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    ``-log((e**-A) @ (e**-B))`` without ever forming ``e**-A``.

    **Both arguments are costs**, i.e. negative log probabilities - including
    ``B``. Passing raw probabilities there is a units error that produces a
    plausible-looking but meaningless matrix, so the mixture weights are kept
    as ``log_mixtures`` and negated on the way in.

    Doing this naively is not an option at this feature dimension: a
    Mahalanobis cost at D=512 runs to ~1400, ``exp(-1400)`` is exactly 0.0 in
    float64 (the limit is ~709), and every score downstream comes out ``inf``.
    Each operand is therefore shifted by its own best (smallest) cost along the
    contracted axis, which makes the largest term in every sum exactly 1; the
    shifts are constants of that sum and are added back afterwards, so the
    result is exact rather than approximate.

    ``+inf`` entries in ``B`` are meaningful and safe - that is how a mixture
    weight of exactly zero arrives, and it contributes nothing to the sum.
    Each row of ``A`` and each column of ``B`` must hold at least one finite
    value, which for mixture weights means every label needs a nonzero weight
    somewhere.
    """
    mA = A.min(axis=1, keepdims=True)
    mB = B.min(axis=0, keepdims=True)
    if not (np.isfinite(mA).all() and np.isfinite(mB).all()):
        raise ValueError(
            "neg_log_matmul: every row of A and column of B needs one finite entry "
            "(a label whose mixture weights are all zero has no emission probability)"
        )

    pA = np.exp(mA - A)
    pB = np.exp(mB - B)

    res_shift = pA @ pB
    # The shift guarantees a largest term of 1 in each factor, but the argmins
    # of the two operands need not meet in the same column, so a row/column
    # pair with disjoint support can still sum to zero. Report it as the
    # modelling problem it is instead of emitting inf scores into the search.
    if not (res_shift > 0).all():
        raise FloatingPointError(
            "neg_log_matmul underflowed to zero for "
            f"{int((res_shift <= 0).sum())} of {res_shift.size} entries: some label's "
            "densities are all astronomically unlikely for some frame"
        )
    return -np.log(res_shift) + mA + mB


class ArtifactModel:
    """
    Base class providing manifest-driven save/load.

    Subclasses implement :meth:`artifacts` and :meth:`from_artifacts`; nothing
    else needs to know what parameters a given model has.

    They also declare :attr:`ARTIFACT_NAMES`, the artifact set as a *class*
    property - available without an instance, which is what lets the pipeline
    build every epoch's model spec (``{name: job.artifact(name)}``) from the
    class alone instead of carrying a hand-maintained tuple alongside it. That
    tuple going out of sync with :meth:`artifacts` used to be enough to break
    run continuation silently; :meth:`save` now checks the two agree.
    """

    #: Names of the arrays this model is reconstructed from. Must match the
    #: keys of :meth:`artifacts` and the constructor's parameter names.
    ARTIFACT_NAMES: ClassVar[Tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        MODEL_CLASSES[cls.__name__] = cls

    def artifacts(self) -> Dict[str, np.ndarray]:
        """Everything needed to reconstruct this model, as name -> array."""
        raise NotImplementedError

    @classmethod
    def from_artifacts(cls, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> "ArtifactModel":
        raise NotImplementedError

    def meta(self) -> Dict[str, Any]:
        """Small JSON-serializable extras recorded in the manifest."""
        return {}

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        arrays = self.artifacts()
        if self.ARTIFACT_NAMES and set(arrays) != set(self.ARTIFACT_NAMES):
            raise ValueError(
                f"{type(self).__name__}.artifacts() returned {sorted(arrays)} but "
                f"ARTIFACT_NAMES declares {sorted(self.ARTIFACT_NAMES)}; the pipeline "
                f"builds each epoch's model spec from the declaration, so the two "
                f"disagreeing would break run continuation"
            )
        for name, array in arrays.items():
            # np.save happily turns a None or a ragged list into a 0-d object
            # array, which then needs allow_pickle to read back and fails far
            # away from the model that produced it. Reject it here instead.
            array = np.asarray(array)
            if array.dtype == object:
                raise TypeError(
                    f"{type(self).__name__} artifact {name!r} is not a numeric array "
                    f"(got {type(arrays[name]).__name__})"
                )
            np.save(os.path.join(directory, f"{name}.npy"), array)
        manifest = {
            "class": type(self).__name__,
            "artifacts": sorted(arrays),
            "num_clusters": int(self.num_clusters),
            "dim": int(self.dim),
            "meta": self.meta(),
        }
        with open(os.path.join(directory, MANIFEST_NAME), "w") as fp:
            json.dump(manifest, fp, indent=2)

    @classmethod
    def load(cls, directory: str) -> "ArtifactModel":
        manifest = read_manifest(directory)
        arrays = {
            name: np.load(os.path.join(directory, f"{name}.npy"))
            for name in manifest["artifacts"]
        }
        return cls.from_artifacts(arrays, manifest.get("meta", {}))


def read_manifest(directory: str) -> Dict[str, Any]:
    with open(os.path.join(directory, MANIFEST_NAME)) as fp:
        return json.load(fp)


class EuclideanModel(ArtifactModel):
    """Squared Euclidean distance to each centroid (plain k-means)."""

    ARTIFACT_NAMES = ("centroids",)

    def __init__(self, centroids: Union[np.ndarray, str]):
        self.centroids = _as_array(centroids)
        if self.centroids.ndim != 2:
            raise ValueError(f"expected centroids [K, D], got {self.centroids.shape}")

    @property
    def num_clusters(self) -> int:
        return self.centroids.shape[0]

    @property
    def dim(self) -> int:
        return self.centroids.shape[1]

    def scores(self, features: np.ndarray) -> np.ndarray:
        # identical to GuidedKMeansClusteringCallback.compute_squared_distances
        return cdist(features, self.centroids, metric="sqeuclidean")

    def artifacts(self) -> Dict[str, np.ndarray]:
        return {"centroids": self.centroids}

    @classmethod
    def from_artifacts(cls, arrays, meta) -> "EuclideanModel":
        return cls(centroids=arrays["centroids"])


class GaussianModel(ArtifactModel):
    """
    Mahalanobis distance under a per-cluster full covariance.

    Wraps :class:`GaussianModelNumpy`, which does the heavy lifting on the GPU
    when one is visible.
    """

    ARTIFACT_NAMES = ("centroids", "covs")

    def __init__(
        self,
        centroids: Union[np.ndarray, str],
        covs: Union[np.ndarray, str],
        device: Optional[str] = None,
    ):
        self.centroids = _as_array(centroids)
        self.covs = _as_array(covs)
        if self.centroids.ndim != 2:
            raise ValueError(f"expected centroids [K, D], got {self.centroids.shape}")
        if self.covs.shape[0] != self.centroids.shape[0]:
            raise ValueError(
                f"expected {self.centroids.shape[0]} covariances, got {self.covs.shape[0]}"
            )
        self.device = device
        self._impl = GaussianModelNumpy(self.centroids, self.covs, device=device)

    @property
    def num_clusters(self) -> int:
        return self.centroids.shape[0]

    @property
    def dim(self) -> int:
        return self.centroids.shape[1]

    def scores(self, features: np.ndarray) -> np.ndarray:
        return self._impl.forward(features)

    def artifacts(self) -> Dict[str, np.ndarray]:
        return {"centroids": self.centroids, "covs": self.covs}

    @classmethod
    def from_artifacts(cls, arrays, meta) -> "GaussianModel":
        return cls(centroids=arrays["centroids"], covs=arrays["covs"])


class MixtureModelBase(ArtifactModel):
    """
    Shared machinery for the models that back a label with several Gaussian
    densities. Three artifacts, whatever the layout::

        centroids [C, D]    density means
        covs      [C, D, D] density covariances
        mixtures  [L, ...]  p(density | label), rows summing to 1

    Two counts matter and they are not the same number: ``num_densities`` (C)
    is how many parameter sets exist, ``num_labels`` (L) is how wide the score
    matrix is. ``num_clusters`` - the name the :class:`.interfaces.ScoreModel`
    protocol and the recognizers use - is the score width, so it aliases
    ``num_labels`` here while it means C on the single-density models. Anything
    indexing "per cluster" has to say which it means.

    What subclasses decide is how the ``mixtures`` rows map onto the density
    axis, which is the whole difference between a shared codebook and per-label
    densities; everything below - validation, persistence, the log-domain
    convention - is common. Scoring and the E-step run in the negative-log
    domain throughout; see :func:`neg_log_matmul` for why exponentiating is not
    an option at D=512.
    """

    ARTIFACT_NAMES = ("centroids", "covs", "mixtures")

    def __init__(
        self,
        centroids: Union[np.ndarray, str],
        covs: Union[np.ndarray, str],
        mixtures: Union[np.ndarray, str],
        device: Optional[str] = None,
    ):
        self.centroids = _as_array(centroids)
        self.covs = _as_array(covs)
        self.mixtures = _as_array(mixtures)
        if self.centroids.ndim != 2:
            raise ValueError(f"expected centroids [C, D], got {self.centroids.shape}")
        num_densities, dim = self.centroids.shape
        if self.covs.shape != (num_densities, dim, dim):
            raise ValueError(
                f"expected covs [{num_densities}, {dim}, {dim}], got {self.covs.shape}"
            )
        if self.mixtures.ndim != 2:
            raise ValueError(f"expected 2-D mixtures, got {self.mixtures.shape}")
        self._validate_layout()
        if (self.mixtures < 0).any():
            raise ValueError("mixture weights must be non-negative")
        if not np.allclose(self.mixtures.sum(axis=-1), 1.0):
            raise ValueError(
                "mixture weights are not normalized per label; row sums range over "
                f"[{self.mixtures.sum(-1).min():.6g}, {self.mixtures.sum(-1).max():.6g}]"
            )
        if not (self.mixtures > 0).any(axis=-1).all():
            raise ValueError(
                "every label needs at least one density with nonzero weight; labels "
                f"{np.flatnonzero(~(self.mixtures > 0).any(-1)).tolist()} have none"
            )
        self.device = device
        # A weight of exactly zero is legitimate (a density this label has no
        # use for) and gives -inf here, which the log-domain arithmetic handles
        # as the "contributes nothing" it is - hence errstate rather than a
        # floor. It is also why a zero weight is permanent under EM: see
        # MixtureGaussianAccumulator's mixture_floor.
        with np.errstate(divide="ignore"):
            self.log_mixtures = np.log(self.mixtures)
        self.gaussian_model = GaussianModel(self.centroids, self.covs, device)

    def _validate_layout(self) -> None:
        """Check ``mixtures``' second axis against the density count."""
        raise NotImplementedError

    @property
    def num_clusters(self) -> int:
        """Score width, i.e. the label count - see the class docstring."""
        return self.mixtures.shape[0]

    @property
    def num_labels(self) -> int:
        return self.mixtures.shape[0]

    @property
    def num_densities(self) -> int:
        return self.centroids.shape[0]

    @property
    def dim(self) -> int:
        return self.centroids.shape[1]

    def density_groups(self) -> Optional[np.ndarray]:
        """
        ``[C]`` group index per density, or None if the densities cannot be grouped.

        Used to pool covariances: densities sharing a group share one covariance.
        Only a layout in which each density belongs to exactly one label has such a
        grouping - a shared codebook does not, since a density there is common
        property and there is no label to pool it under.
        """
        return None

    def scores_gaussian(self, features: np.ndarray) -> np.ndarray:
        """``[T, D] -> [T, C]``, the per-density costs."""
        return self.gaussian_model.scores(features)

    def scores_from_gaussian(self, scores_gaussian: np.ndarray) -> np.ndarray:
        """
        ``[T, C] -> [T, L]``: marginalize the densities out of already-computed
        density costs. Split out of :meth:`scores` so a caller needing both -
        the E-step does - pays for the Mahalanobis pass once.
        """
        raise NotImplementedError

    def scores(self, features: np.ndarray) -> np.ndarray:
        return self.scores_from_gaussian(self.scores_gaussian(features))

    def _check_posteriors(self, features: np.ndarray, label_gammas: np.ndarray) -> np.ndarray:
        label_gammas = np.asarray(label_gammas, dtype=np.float64)
        if label_gammas.ndim != 2 or label_gammas.shape[1] != self.num_labels:
            raise ValueError(
                f"expected label posteriors [T, {self.num_labels}], "
                f"got {label_gammas.shape}"
            )
        if len(label_gammas) != len(features):
            raise ValueError(
                f"frame count mismatch: {len(features)} features vs "
                f"{len(label_gammas)} posteriors"
            )
        return label_gammas

    def responsibilities(
        self,
        features: np.ndarray,
        label_gammas: np.ndarray,
        *,
        scores_gaussian: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The mixture E-step: split posteriors over labels into posteriors over
        the densities behind them.

        Lives on the model, not on the accumulator, because it is entirely a
        statement about this parameterization - the accumulator only has to
        know it is being handed density-indexed weights. An accumulator pair
        (densities + mixture weights) therefore works for any model offering
        this method, which is what lets a new mixture layout be a new model
        class and nothing else.

        :param label_gammas: ``[T, L]``, what the recognizer reported
        :param scores_gaussian: the ``[T, C]`` density costs if the caller
            already has them; recomputed if not
        :return: ``(density_gammas [T, C], joint_counts [L, ...])`` where
            ``density_gammas[t, c] = sum_l gamma[t, l] p(c | l, x_t)`` and
            ``joint_counts`` is the same joint summed over frames instead,
            shaped like ``mixtures``. Both are marginals of one joint, so
            ``density_gammas.sum(1)`` equals ``label_gammas.sum(1)`` - the
            invariant the tests assert for every layout.
        """
        raise NotImplementedError

    def artifacts(self) -> Dict[str, np.ndarray]:
        return {"centroids": self.centroids, "covs": self.covs, "mixtures": self.mixtures}

    def meta(self) -> Dict[str, Any]:
        return {"num_labels": int(self.num_labels), "num_densities": int(self.num_densities)}

    @classmethod
    def from_artifacts(cls, arrays, meta) -> "MixtureModelBase":
        return cls(
            centroids=arrays["centroids"], covs=arrays["covs"], mixtures=arrays["mixtures"]
        )


class GaussianMixtureModel(MixtureModelBase):
    """
    A shared codebook: every label may draw on every density.

    ``mixtures`` is ``[L, C]``, so the C densities are common property and a
    label is defined purely by how it weights them - the semi-continuous /
    tied-mixture arrangement. Densities are shared, so a density is re-estimated
    from every label that uses it, which is what makes this the parameter-frugal
    option; the price is that scoring a label marginalizes over all C densities
    and the E-step forms a ``[T, L, C]`` joint.

    For the opposite trade - each label owning its own densities, L times
    cheaper to score and to update - see :class:`PerLabelMixtureModel`.
    """

    #: Frames per block in :meth:`responsibilities`, which forms a
    #: ``[frames, L, C]`` intermediate. Purely a memory knob - the result is
    #: identical for any value - and not a speed tax either: 256 measured no
    #: slower than doing a whole sequence at once, and 2.7x faster for a long
    #: one, since the intermediate stops fitting in cache long before it stops
    #: fitting in memory.
    RESPONSIBILITY_BLOCK_FRAMES: ClassVar[int] = 256

    def _validate_layout(self) -> None:
        if self.mixtures.shape[1] != self.num_densities:
            raise ValueError(
                f"shape {self.mixtures.shape} of mixtures does not match with "
                f"covariances {self.covs.shape}; a shared codebook needs one weight "
                f"per label per density"
            )

    def scores_from_gaussian(self, scores_gaussian: np.ndarray) -> np.ndarray:
        return neg_log_matmul(scores_gaussian, -self.log_mixtures.T)

    def responsibilities(self, features, label_gammas, *, scores_gaussian=None):
        """See :meth:`MixtureModelBase.responsibilities`."""
        label_gammas = self._check_posteriors(features, label_gammas)
        if scores_gaussian is None:
            scores_gaussian = self.scores_gaussian(features)
        scores_label = self.scores_from_gaussian(scores_gaussian)   # [T, L]

        density_gammas = np.zeros((len(features), self.num_densities), dtype=np.float64)
        joint_counts = np.zeros((self.num_labels, self.num_densities), dtype=np.float64)

        # Blocks contribute independently, so the block size changes nothing
        # about the result - it only bounds the intermediate below.
        for begin in range(0, len(features), self.RESPONSIBILITY_BLOCK_FRAMES):
            stop = begin + self.RESPONSIBILITY_BLOCK_FRAMES
            # log p(c | l, x) = log w_lc - cost_c(x) + cost_l(x), which is <= 0
            # for every entry, so the exponential cannot overflow.
            log_posterior = (
                self.log_mixtures[np.newaxis, :, :]
                - scores_gaussian[begin:stop, np.newaxis, :]
                + scores_label[begin:stop, :, np.newaxis]
            )                                                       # [frames, L, C]
            joint = label_gammas[begin:stop, :, np.newaxis] * np.exp(log_posterior)
            density_gammas[begin:stop] = joint.sum(axis=1)
            joint_counts += joint.sum(axis=0)

        return density_gammas, joint_counts


class PerLabelMixtureModel(MixtureModelBase):
    """
    ``n`` densities per label, owned by that label alone.

    ``mixtures`` is ``[L, n]`` and the density axis is the two flattened
    together: label ``l``'s densities occupy ``centroids[l * n : (l + 1) * n]``,
    so ``C == L * n``. This is the conventional continuous-density arrangement,
    and the one to reach for when labels are acoustically unrelated enough that
    sharing densities between them is a constraint rather than a saving.

    The disjointness pays twice over. Scoring a label marginalizes over its own
    ``n`` densities rather than all ``C``, and in the E-step a density belongs
    to exactly one label, so the joint is ``[T, L, n]`` - the same size as the
    score matrix. No blocking, and no ``[T, L, C]`` tensor to bound.

    The same model is expressible as a :class:`GaussianMixtureModel` whose
    ``[L, C]`` weights are block diagonal, and EM preserves that structure
    exactly, because a zero weight is an absorbing state (see
    ``mixture_floor``). That equivalence is asserted in the tests and is the
    reason this class is an optimization rather than a new algorithm - but it
    is a large one, and it makes the intent explicit instead of leaving it
    implicit in the initializer's sparsity pattern.
    """

    def _validate_layout(self) -> None:
        expected = self.num_labels * self.mixtures.shape[1]
        if expected != self.num_densities:
            raise ValueError(
                f"{self.num_labels} labels x {self.mixtures.shape[1]} densities each "
                f"= {expected}, but got {self.num_densities} centroids; per-label "
                f"densities are not shared, so the counts have to multiply out"
            )

    @property
    def densities_per_label(self) -> int:
        return self.mixtures.shape[1]

    def density_groups(self) -> np.ndarray:
        """Each density's own label: ``[0]*n + [1]*n + ...``, matching ``l*n + k``."""
        return np.repeat(np.arange(self.num_labels), self.densities_per_label)

    def _per_label(self, scores_gaussian: np.ndarray) -> np.ndarray:
        """``[T, C] -> [T, L, n]``, undoing the flattening of the density axis."""
        return scores_gaussian.reshape(
            len(scores_gaussian), self.num_labels, self.densities_per_label
        )

    def scores_from_gaussian(self, scores_gaussian: np.ndarray) -> np.ndarray:
        per_label = self._per_label(scores_gaussian)                # [T, L, n]
        # Same shift as neg_log_matmul, but the contraction is per label rather
        # than a matmul: subtract each label's best density cost so the largest
        # term of the sum is exactly 1, then add it back.
        best = per_label.min(axis=2, keepdims=True)                 # [T, L, 1]
        total = (self.mixtures[np.newaxis] * np.exp(best - per_label)).sum(axis=2)
        if not (total > 0).all():
            raise FloatingPointError(
                f"label emission probability underflowed to zero for "
                f"{int((total <= 0).sum())} of {total.size} entries: some label's "
                "densities are all astronomically unlikely for some frame"
            )
        return best[:, :, 0] - np.log(total)

    def responsibilities(self, features, label_gammas, *, scores_gaussian=None):
        """See :meth:`MixtureModelBase.responsibilities`."""
        label_gammas = self._check_posteriors(features, label_gammas)
        if scores_gaussian is None:
            scores_gaussian = self.scores_gaussian(features)
        per_label = self._per_label(scores_gaussian)                # [T, L, n]
        scores_label = self.scores_from_gaussian(scores_gaussian)   # [T, L]

        # log p(k | l, x) = log w_lk - cost_lk(x) + cost_l(x), <= 0 as ever.
        # Only label l's own densities appear, so this is [T, L, n] - the size
        # of the score matrix - and needs no blocking.
        log_posterior = (
            self.log_mixtures[np.newaxis, :, :] - per_label + scores_label[:, :, np.newaxis]
        )
        joint = label_gammas[:, :, np.newaxis] * np.exp(log_posterior)

        # A density belongs to one label, so there is nothing to sum over l:
        # flattening [T, L, n] is already the [T, C] density posterior.
        return joint.reshape(len(features), self.num_densities), joint.sum(axis=0)


def _as_array(value: Union[np.ndarray, str]) -> np.ndarray:
    """An artifact given either directly or as a path to its ``.npy``."""
    return np.load(value) if isinstance(value, str) else np.asarray(value)


def load_forward_model(model_dir: str):
    """
    Any model directory as the object the decode callback scores with.

    :class:`...decode.ClusteringDecodeCallback` wants something with a
    ``forward`` method; the models here call that ``scores``. Rather than
    teaching the decode path about each model class - which is what a
    ``mixtures=`` argument next to ``covs=`` would amount to, and would need
    doing again for the next parameter set - it is handed a directory and
    :func:`load_model` picks the class out of the manifest. A model that can be
    trained is then decodable with no decode-side change at all.
    """
    model = load_model(model_dir)

    class _ScoresAsForward:
        def __init__(self, inner):
            self.model = inner

        def forward(self, features):
            return self.model.scores(features)

    return _ScoresAsForward(model)


def load_model(directory: str) -> ArtifactModel:
    """
    Load whichever model class wrote ``directory``, per its manifest.

    Not used by the pipeline - which names the model class explicitly, so that
    every epoch's spec has the same shape - but the convenient way to pick a
    model directory up for analysis without knowing what wrote it.
    """
    manifest = read_manifest(directory)
    name = manifest["class"]
    try:
        cls = MODEL_CLASSES[name]
    except KeyError:
        raise ValueError(
            f"unknown model class {name!r} in {directory}; known classes are "
            f"{sorted(MODEL_CLASSES)}. Subclasses of ArtifactModel register "
            f"themselves on definition, so the module defining {name!r} probably "
            f"has not been imported in this process."
        ) from None
    return cls.load(directory)


