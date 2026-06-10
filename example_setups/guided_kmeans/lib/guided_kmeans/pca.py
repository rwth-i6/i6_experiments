from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

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
        return np.concatenate(self.buffer)


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

    def __init__(self, n_components=None, dtype=np.float64, ddof=1, store_covariance=False):
        """
        n_components: int or None. If None, keep all.
        dtype: numeric dtype used internally.
        ddof: 1 for sample covariance (default), 0 for population covariance.
        """
        super().__init__(buffer_size=128)
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

    def finalize(self):
        """
        Compute PCA from the accumulated covariance.
        Returns a dict with PCA results.
        """
        if self._finalized:
            raise RuntimeError("finalize() already called")

        if self.n_samples == 0:
            raise RuntimeError("No samples were added")

        if self.n_samples - self.ddof <= 0:
            raise RuntimeError(
                f"Not enough samples to compute covariance with ddof={self.ddof}. "
                f"Need n_samples > {self.ddof}."
            )
        
        # process last sequences in buffer
        if self.buffer:
            self.add_batch(self.flush())

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
