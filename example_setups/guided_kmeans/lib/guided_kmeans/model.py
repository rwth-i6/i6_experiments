import numpy as np
import torch

class GaussianModelNumpy:
    def __init__(
        self,
        centroids,
        covariance_matrices=None,
        device: str | None = None,
    ):
        # This runs once per model load inside the RETURNN forward job's main
        # process, which already holds the encoder's CUDA context - so by
        # default put the (otherwise CPU-numpy-bound) Mahalanobis computation
        # on the GPU too, where it's cheap relative to the per-utterance RASR
        # search happening in parallel on the CPU workers.
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.means = torch.as_tensor(centroids, dtype=torch.float32, device=self.device)

        if covariance_matrices is not None:
            # np.linalg.solve() would otherwise LU-factorize these same N
            # (D, D) covariance matrices from scratch on every forward()
            # call. They never change, so invert once here (on CPU - a single
            # ~1s cost, not worth shipping to the GPU) and turn the per-call
            # solve into a single batched matmul instead.
            inv_covs = np.linalg.inv(covariance_matrices)
            # float32 because these GPUs (GTX 1080 Ti / RTX 2080 Ti) have
            # only a small fraction of their fp32 throughput available for
            # fp64; the result is cast back to float64 below for interface
            # consistency with the cdist-based path, but the actual
            # computation loses some precision. That's fine for a
            # nearest-centroid distance score, not an iterative computation
            # where error would accumulate.
            self.inv_covs = torch.as_tensor(inv_covs, dtype=torch.float32, device=self.device)
        else:
            self.inv_covs = None

    @classmethod
    def load(cls, centroids_path, covs_path=None):
        centroids = np.load(centroids_path)
        covs = np.load(covs_path) if covs_path is not None else None

        return cls(centroids, covs)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the PDF of N multivariate normal distributions at M points.

        X: (M, D) array of points
        means: (N, D) array of centroids
        covs: (N, D, D) array of covariance matrices

        Returns: (M, N) array of PDF values
        """
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        # 1. Compute differences: (M, N, D)
        diff = x[:, None, :] - self.means[None, :, :]

        # 2. Solve the linear system: cov^{-1} * diff
        # Permute diff to (N, D, M) to align with covs (N, D, D)
        diff_T = diff.permute(1, 2, 0)

        # Equivalent to solving cov @ y = diff_T for y, but using the
        # precomputed inverse turns this into a single batched matmul
        # instead of re-factorizing `covs` on every call.
        if self.inv_covs is not None:
            y = self.inv_covs @ diff_T  # Shape: (N, D, M)
        else:
            y = diff_T

        # Permute y back to match diff: (M, N, D)
        y_aligned = y.permute(2, 0, 1)

        # 3. Compute Mahalanobis distance squared: diff^T * cov^{-1} * diff
        mahal = (diff * y_aligned).sum(dim=-1)  # Shape: (M, N)

        # 4. Compute the normalization constant
        # Use slogdet for numerical stability instead of det
        # sign, logdet = np.linalg.slogdet(self.covs)
        # log_norm = -0.5 * (D * np.log(2 * np.pi) + logdet) # Shape: (N,)

        # 5. Compute final log-PDF and exponentiate
        # log_pdf = log_norm[None, :] - 0.5 * mahal

        # Downstream (ProcessPoolExecutor IPC to CPU search workers, and the
        # cdist-based code path this feeds alongside) expects a plain float64
        # numpy array, not a CUDA tensor.
        return (0.5 * mahal).to("cpu", dtype=torch.float64).numpy()

def load_gaussian_model(centroids_path: str, covs_path: str | None=None) -> GaussianModelNumpy:
    centroids = np.load(centroids_path)
    covs = np.load(covs_path) if covs_path is not None else None

    return GaussianModelNumpy(centroids, covs)
