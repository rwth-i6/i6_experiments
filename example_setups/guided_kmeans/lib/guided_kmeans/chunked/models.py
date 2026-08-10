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
    "load_model",
    "read_manifest",
    "MODEL_CLASSES",
    "MANIFEST_NAME",
]

import json
import os
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
from scipy.spatial.distance import cdist

from ..model import GaussianModelNumpy

MANIFEST_NAME = "model.json"

#: Populated automatically by every :class:`ArtifactModel` subclass, so
#: :func:`load_model` can reconstruct a model directory written by a class it
#: was never told about - the module defining it only has to be imported.
MODEL_CLASSES: Dict[str, type] = {}


class ArtifactModel:
    """
    Base class providing manifest-driven save/load.

    Subclasses implement :meth:`artifacts` and :meth:`from_artifacts`; nothing
    else needs to know what parameters a given model has.
    """

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


def _as_array(value: Union[np.ndarray, str]) -> np.ndarray:
    """An artifact given either directly or as a path to its ``.npy``."""
    return np.load(value) if isinstance(value, str) else np.asarray(value)


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


