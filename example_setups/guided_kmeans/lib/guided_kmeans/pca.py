from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .update import UpdaterBase_
from .model import GaussianModelNumpy

class NotEnoughSamplesError(RuntimeError):
    pass

class SequenceBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size

        self.buffer = []
        self.seq_count = 0
    
    def maybe_get_buffer(self, seq) -> Optional[np.ndarray]:
        if self.seq_count < self.buffer_size:
            self.buffer.append(seq)
            self.seq_count += 1
            return None
        
        result = np.concatenate(self.buffer)

        # reset buffer and count
        self.buffer = []
        self.seq_count = 0

        return result
    
    def flush(self) -> np.ndarray:
        result = np.concatenate(self.buffer)

        # reset buffer and count
        self.buffer = []
        self.seq_count = 0

        return result


@dataclass(frozen=True, slots=True)
class PCAResult:
    n_samples: int
    mean: np.ndarray  # (d,)
    components: np.ndarray  # (k, d) rows are principal axes
    explained_variance: np.ndarray  # (k,)
    explained_variance_ratio: np.ndarray  # (k,)
    singular_values: np.ndarray  # (k,)
    covariance: Optional[np.ndarray] = None  # (d, d) if requested


class StreamingPCA(SequenceBuffer):
    """
    Streaming PCA via online mean + covariance accumulation (Welford).
    
    Memory: O(d^2) for covariance accumulator.
    Time:   O(d^2) per sample update (outer products).
    
    Usage:
        spca = StreamingPCA(n_components=10, dtype=np.float64)
        for x in data_stream:
            spca.add_sample(x)
        result = spca.finalize()
        
    result contains:
        - mean: (d,)
        - components: (k, d) rows are principal axes (unit vectors)
        - explained_variance: (k,) eigenvalues of covariance
        - explained_variance_ratio: (k,)
        - singular_values: (k,) like sklearn (sqrt(eigval*(n-1)))
        - n_samples: int
    """

    def __init__(self, n_components=None, dtype=np.float64, ddof=1, store_covariance=False, buffer_size=128):
        """
        n_components: int or None. If None, keep all.
        dtype: numeric dtype used internally.
        ddof: 1 for sample covariance (default), 0 for population covariance.
        """
        super().__init__(buffer_size=buffer_size)
        self.n_components = n_components
        self.dtype = dtype
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 (population) or 1 (sample)")
        self.ddof = ddof
        self.store_covariance = store_covariance

        self.n_samples = 0
        self.mean_ = None          # (d,)
        self.M2_ = None            # (d, d) sum of squares matrix for covariance

        self._finalized = False
    
    @property
    def has_data(self):
        return self.buffer or self.n_samples > 0

    def add_sample(self, x):
        """
        Add one sample vector x of shape (d,).
        """
        if self._finalized:
            raise RuntimeError("Cannot add samples after finalize()")

        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 1:
            raise ValueError("x must be a 1D array of shape (d,)")

        if self.n_samples == 0:
            d = x.shape[0]
            self.mean_ = np.zeros(d, dtype=self.dtype)
            self.M2_ = np.zeros((d, d), dtype=self.dtype)

        if x.shape[0] != self.mean_.shape[0]:
            raise ValueError(f"Dimension mismatch: got {x.shape[0]}, expected {self.mean_.shape[0]}")

        self.n_samples += 1
        n = self.n_samples

        # Welford update for vector mean + covariance accumulator
        delta = x - self.mean_
        self.mean_ += delta / n
        delta2 = x - self.mean_   # uses updated mean
        self.M2_ += np.outer(delta, delta2)

    def add_batch(self, X) -> None:
        """
        Add a batch X of shape (m, d) with a *vectorized* update.
        """
        if self._finalized:
            raise RuntimeError("Cannot add data after finalize()")

        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2:
            raise ValueError("X must be 2D (m, d)")

        m, d = X.shape
        if m == 0:
            return
        
        # Initialize state on first batch
        if self.n_samples == 0:
            self.mean_ = np.zeros(d, dtype=self.dtype)
            self.M2_ = np.zeros((d, d), dtype=self.dtype)

        assert self.mean_ is not None and self.M2_ is not None

        if d != self.mean_.shape[0]:
            raise ValueError(f"Dimension mismatch: got {d}, expected {self.mean_.shape[0]}")

        # Batch stats
        batch_mean = X.mean(axis=0)             # (d,)
        Xc = X - batch_mean                     # (m, d)
        batch_M2 = Xc.T @ Xc                    # (d, d) scatter (sum of outer products)

        # Merge (parallel variance / covariance merge)
        n_a = self.n_samples
        n_b = m
        n = n_a + n_b

        delta = batch_mean - self.mean_         # (d,)
        self.mean_ = self.mean_ + delta * (n_b / n)

        # M2_total = M2_a + M2_b + (n_a*n_b/n) * delta*delta^T
        self.M2_ = self.M2_ + batch_M2 + (n_a * n_b / n) * np.outer(delta, delta)

        self.n_samples = n
    
    def process_sequence(self, sequence) -> None:
        buffer = self.maybe_get_buffer(sequence)
        if buffer is not None:
            self.add_batch(buffer)

    def flush_buffer(self) -> None:
        """
        Fold any buffered sequences into the accumulator. Idempotent.

        add_batch()/finalize() already do this implicitly; merge() and
        state_dict() need it explicitly, since they read the accumulator
        fields directly and would otherwise silently drop buffered data.
        """
        if self.buffer:
            self.add_batch(self.flush())

    def merge(self, other: "StreamingPCA") -> "StreamingPCA":
        """
        Combine another instance's accumulated state into this one, in place.

        This is the same pairwise (Chan et al.) merge that add_batch() already
        performs against a freshly summarized batch - only here the second
        operand's (n, mean, M2) come from another accumulator rather than from
        raw samples. It is therefore associative, so accumulating over disjoint
        chunks and merging gives the same answer as one sequential pass (up to
        float summation order), which is what makes chunked clustering exact.

        Keeping the Welford/M2 form rather than raw second moments matters at
        the feature dimensions used here (D=512): forming a covariance as
        Sum(xx^T) - n*mu*mu^T loses precision to cancellation, this does not.
        """
        if self._finalized:
            raise RuntimeError("Cannot merge into a finalized StreamingPCA")
        if other._finalized:
            raise RuntimeError("Cannot merge from a finalized StreamingPCA")
        if self.ddof != other.ddof:
            raise ValueError(f"ddof mismatch: {self.ddof} vs {other.ddof}")

        self.flush_buffer()
        # Do not mutate `other`: summarize its buffer without folding it in.
        other_n, other_mean, other_M2 = other._accumulator_state()

        if other_n == 0:
            return self

        if self.n_samples == 0:
            self.mean_ = other_mean.copy()
            self.M2_ = other_M2.copy()
            self.n_samples = other_n
            return self

        assert self.mean_ is not None and self.M2_ is not None
        if self.mean_.shape != other_mean.shape:
            raise ValueError(
                f"Dimension mismatch: got {other_mean.shape}, expected {self.mean_.shape}"
            )

        n_a, n_b = self.n_samples, other_n
        n = n_a + n_b

        delta = other_mean - self.mean_
        self.mean_ = self.mean_ + delta * (n_b / n)
        self.M2_ = self.M2_ + other_M2 + (n_a * n_b / n) * np.outer(delta, delta)
        self.n_samples = n
        return self

    def _accumulator_state(self):
        """
        (n_samples, mean, M2) including anything still sitting in the buffer,
        without mutating self.
        """
        n, mean, M2 = self.n_samples, self.mean_, self.M2_
        if not self.buffer:
            if n == 0:
                return 0, None, None
            return n, mean, M2

        X = np.concatenate(self.buffer)
        m, d = X.shape
        batch_mean = X.mean(axis=0)
        Xc = X - batch_mean
        batch_M2 = Xc.T @ Xc

        if n == 0:
            return m, batch_mean, batch_M2

        assert mean is not None and M2 is not None
        total = n + m
        delta = batch_mean - mean
        return (
            total,
            mean + delta * (m / total),
            M2 + batch_M2 + (n * m / total) * np.outer(delta, delta),
        )

    def state_dict(self) -> dict:
        """Picklable/npz-able accumulator state. Folds in the buffer first."""
        self.flush_buffer()
        return {
            "n_samples": self.n_samples,
            "mean": self.mean_,
            "M2": self.M2_,
            "ddof": self.ddof,
        }

    def load_state_dict(self, state: dict) -> "StreamingPCA":
        """Restore state produced by :func:`state_dict`."""
        if int(state["ddof"]) != self.ddof:
            raise ValueError(f"ddof mismatch: {state['ddof']} vs {self.ddof}")
        self.n_samples = int(state["n_samples"])
        self.mean_ = None if state["mean"] is None else np.asarray(state["mean"], dtype=self.dtype)
        self.M2_ = None if state["M2"] is None else np.asarray(state["M2"], dtype=self.dtype)
        self.buffer = []
        self.seq_count = 0
        self._finalized = False
        return self

    def finalize(self):
        """
        Compute PCA from the accumulated covariance.
        Returns a dict with PCA results.
        """
        if self._finalized:
            raise RuntimeError("finalize() already called")

        # process last sequences in buffer
        if self.buffer:
            self.add_batch(self.flush())

        if self.n_samples - self.ddof <= 0:
            raise NotEnoughSamplesError(
                f"Not enough samples to compute covariance with ddof={self.ddof}. "
                f"Need n_samples > {self.ddof}."
            )

        assert self.mean_ is not None and self.M2_ is not None

        denom = (self.n_samples - self.ddof)
        cov = self.M2_ / denom

        # Symmetric eigen-decomposition; eigh returns ascending eigenvalues
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]  # columns are eigenvectors

        # Choose k
        d = eigvecs.shape[0]
        k = d if self.n_components is None else int(self.n_components)
        if not (1 <= k <= d):
            raise ValueError(f"n_components must be in [1, {d}] or None")

        eigvals_k = eigvals[:k]
        eigvecs_k = eigvecs[:, :k]

        # Components as rows (sklearn-style): (k, d)
        components = eigvecs_k.T

        total_var = eigvals.sum()
        explained_variance_ratio = eigvals_k / total_var if total_var > 0 else np.zeros_like(eigvals_k)

        # sklearn-like singular values: sqrt(eigval * (n_samples - 1))
        # Note: sklearn uses n_samples - 1 for centered data SVD relationship.
        singular_values = np.sqrt(np.maximum(eigvals_k, 0) * max(self.n_samples - 1, 1))

        self._finalized = True

        return PCAResult(
            n_samples=self.n_samples,
            mean=self.mean_.copy(),
            components=components,
            explained_variance=eigvals_k,
            explained_variance_ratio=explained_variance_ratio,
            singular_values=singular_values,
            covariance=cov if self.store_covariance else None,
        )

    def transform(self, x, result):
        """
        Project a sample x onto the PCA components from finalize() result.
        x: (d,)
        returns: (k,)
        """
        x = np.asarray(x, dtype=self.dtype)
        mean = result["mean"]
        comps = result["components"]  # (k, d)
        return comps @ (x - mean)

    def inverse_transform(self, z, result):
        """
        Map a projected vector z back to original space approximation.
        z: (k,)
        returns: (d,)
        """
        z = np.asarray(z, dtype=self.dtype)
        mean = result["mean"]
        comps = result["components"]  # (k, d)
        return mean + comps.T @ z


class PCAUpdater(UpdaterBase_):
    def __init__(self, num_clusters):
        self.pcas = [
            StreamingPCA(
                n_components=None,
                ddof=0,
                store_covariance=True
            )
            for _ in range(num_clusters)
        ]

    def update(self, features, idxs):
        for idx in np.unique(idxs):
            mask = idxs == idx
            self.pcas[idx].add_batch(features[mask])

    def merge(self, other: "PCAUpdater") -> "PCAUpdater":
        """
        Combine another updater's per-cluster accumulators into this one,
        in place. Associative, since StreamingPCA.merge() is.
        """
        if len(self.pcas) != len(other.pcas):
            raise ValueError(
                f"cluster count mismatch: {len(self.pcas)} vs {len(other.pcas)}"
            )
        for own, incoming in zip(self.pcas, other.pcas):
            own.merge(incoming)
        return self

    def state_dict(self) -> dict:
        """
        Per-cluster accumulator state as dense arrays, for the npz round-trip
        between a chunk task and the reduce step.

        Clusters that saw no data have no allocated mean/M2 yet; they are
        stored as zeros and identified by n_samples == 0 on load, which is
        also what get_model() treats as "keep the previous model".
        """
        n_samples = np.array([pca.state_dict()["n_samples"] for pca in self.pcas], dtype=np.int64)
        dim = next(
            (pca.mean_.shape[0] for pca in self.pcas if pca.mean_ is not None),
            None,
        )
        if dim is None:
            return {"n_samples": n_samples, "mean": None, "M2": None}

        means = np.zeros((len(self.pcas), dim), dtype=np.float64)
        m2s = np.zeros((len(self.pcas), dim, dim), dtype=np.float64)
        for idx, pca in enumerate(self.pcas):
            if pca.mean_ is not None:
                means[idx] = pca.mean_
                m2s[idx] = pca.M2_
        return {"n_samples": n_samples, "mean": means, "M2": m2s}

    def load_state_dict(self, state: dict) -> "PCAUpdater":
        """Restore state produced by :func:`state_dict`."""
        n_samples = np.asarray(state["n_samples"])
        if len(n_samples) != len(self.pcas):
            raise ValueError(
                f"cluster count mismatch: {len(n_samples)} vs {len(self.pcas)}"
            )
        for idx, pca in enumerate(self.pcas):
            n = int(n_samples[idx])
            pca.load_state_dict({
                "n_samples": n,
                "mean": None if n == 0 or state["mean"] is None else state["mean"][idx],
                "M2": None if n == 0 or state["M2"] is None else state["M2"][idx],
                "ddof": pca.ddof,
            })
        return self

    def get_model(self, old_model: GaussianModelNumpy) -> GaussianModelNumpy:
        pca_means = []
        pca_covs = []
        for idx, pca in enumerate(self.pcas):
            try:
                pca_res = pca.finalize()
                mean = pca_res.mean
                cov = pca_res.covariance
            except NotEnoughSamplesError:
                mean = old_model.means[idx,:]
                cov = old_model.covs[idx,:]
            pca_means.append(mean)
            pca_covs.append(cov)
        centroids = np.stack(pca_means, axis=0)
        covs = np.stack(pca_covs, axis=0)
        return GaussianModelNumpy(centroids, covs)
