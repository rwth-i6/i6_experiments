from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

__all__ = [
    "IncrementalPCA",
    "PCAProjectionQuantizer",
]


class IncrementalPCA(nn.Module):
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that leverages PyTorch for GPU acceleration.
    The code is taken from https://github.com/dnhkng/PCAonGPU/blob/main/gpu_pca/pca_module.py.

    This class provides methods to fit the model on data incrementally in batches, and to transform new data
    based on the principal components learned during the fitting process.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        *,
        whiten: bool = False,
        copy: bool = True,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        :param n_components: Number of components to keep. If `None`, it's set to the minimum of the number of samples
                             and features. Defaults to None.
        :param whiten: When True, the `components_` vectors are divided to ensure uncorrelated outputs with unit
                       component-wise variances. Defaults to False.
        :param copy: If False, input data will be overwritten. Defaults to True.
        :param batch_size: The number of samples to use for each batch. If `None`, it's inferred from the data and set to
                           `5 * n_features`. Defaults to None.
        :param device: if set to None, then it use the x.device from the first input data x
        """
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.device = device

        # Initialize attributes to avoid errors during the first call to partial_fit
        self.mean = None  # Will be initialized properly in partial_fit based on data dimensions
        self.var = None  # Will be initialized properly in partial_fit based on data dimensions
        self.n_samples_seen = 0

    def _validate_data(self, x: torch.Tensor, dtype=torch.float32, copy: bool = True):
        """
        Validates and converts the input data `x` to the appropriate tensor format.

        This method ensures that the input data is in the form of a PyTorch tensor and resides on the correct device (CPU or GPU).
        It also provides an option to create a copy of the tensor, which is useful when the input data should not be overwritten.
        """

        x = torch.tensor(x, dtype=dtype).to(self.device)

        if copy:
            x = x.clone()
        return x

    @staticmethod
    def _incremental_mean_and_var(
        x: torch.Tensor,
        last_mean: Optional[torch.tensor],
        last_variance: Optional[torch.tensor],
        last_sample_count: int,
    ):
        """
        Computes the incremental mean and variance for the data `x`.

        :param X: The batch input data tensor with shape (n_samples, n_features).
        :param last_mean: The previous mean tensor with shape (n_features,).
        :param last_variance: The previous variance tensor with shape (n_features,).
        :param last_sample_count: The count tensor of samples processed before the current batch.
        :return: Tuple[torch.Tensor, torch.Tensor, int]: Updated mean, variance tensors, and total sample count.
        """
        if x.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        # If last_mean or last_variance is None, initialize them with zeros
        if last_mean is None:
            last_mean = torch.zeros(x.shape[1], device=x.device)
        if last_variance is None:
            last_variance = torch.zeros(x.shape[1], device=x.device)

        new_sample_count = x.shape[0]
        new_mean = torch.mean(x, dim=0)
        new_sum_square = torch.sum((x - new_mean) ** 2, dim=0)

        updated_sample_count = last_sample_count + new_sample_count

        updated_mean = (last_sample_count * last_mean + new_sample_count * new_mean) / updated_sample_count
        updated_variance = (
            last_variance * (last_sample_count + new_sample_count * last_mean**2)
            + new_sum_square
            + new_sample_count * new_mean**2
        ) / updated_sample_count - updated_mean**2

        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def _svd_flip(u: torch.Tensor, v: torch.Tensor, u_based_decision: bool = True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for deterministic output.
        This method ensures that the output remains consistent across different runs.

        :param u: Left singular vectors tensor.
        :param v: Right singular vectors tensor.
        :param u_based_decision: If True, uses the left singular vectors to determine the sign flipping. Defaults to True.
        :return: Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v

    def fit(self, x: torch.Tensor, check_input: bool = True):
        """
        Fits the model with data `X` using minibatches of size `batch_size`.

        :param X: The input data tensor with shape (n_samples, n_features).
        :returnn: IncrementalPCAGPU: The fitted IPCA model.
        """
        if self.device is None:
            self.device = x.device

        if check_input:
            x = self._validate_data(x)
        n_samples, n_features = x.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for start in range(0, n_samples, self.batch_size_):
            end = min(start + self.batch_size_, n_samples)
            x_batch = x[start:end]
            self.partial_fit(x_batch, check_input=False)

        return self

    def partial_fit(self, x: torch.Tensor, check_input: bool = True):
        """
        Incrementally fits the model with batch data `X`.

        :param X: The batch input data tensor with shape (n_samples, n_features).
        :param check_input: If True, validates the input. Defaults to True.
        :return: IncrementalPCAGPU: The updated IPCA model after processing the batch.
        """
        if self.device is None:
            self.device = x.device

        first_pass = not hasattr(self, "components_")

        if check_input:
            x = self._validate_data(x)
        n_samples, n_features = x.shape

        if first_pass:
            self.components_ = None
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            x, self.mean, self.var, torch.tensor([self.n_samples_seen], device=x.device)
        )

        # Whitening
        if self.n_samples_seen == 0:
            x -= col_mean
        else:
            col_batch_mean = torch.mean(x, dim=0)
            x -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                torch.tensor((self.n_samples_seen / n_total_samples.item()) * n_samples, device=x.device)
            )
            mean_correction = mean_correction_factor * (self.mean - col_batch_mean)

            if self.singular_values_ is not None and self.components_ is not None:
                x = torch.vstack(
                    (
                        self.singular_values_.view((-1, 1)) * self.components_,
                        x,
                        mean_correction,
                    )
                )

        u, s, v_t = torch.linalg.svd(x, full_matrices=False)
        u, v_t = self._svd_flip(u, v_t, u_based_decision=False)
        explained_variance = s**2 / (n_total_samples.item() - 1)
        explained_variance_ratio = s**2 / torch.sum(col_var * n_total_samples.item())

        self.n_samples_seen = n_total_samples.item()
        self.components_ = v_t[: self.n_components]
        self.singular_values_ = s[: self.n_components]
        self.mean = col_mean
        self.var = col_var
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]
        if self.n_components not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components :].mean().item()
        else:
            self.noise_variance_ = 0.0
        return self


class PCAProjectionQuantizer(nn.Module):
    """
    Replace the fixed random linear projection in RandomProjectionQuantizer with PCA components
    """

    def __init__(self, input_dim: int, codebook_dim: int, codebook_num_vars: int, use_pca_projection: bool = True):
        """
        :param input_dim: number of feature dimension of input
        :param codebook_dim: number of dimension for vocab in the codebook
        :param codebook_num_vars: vocab size of the codebook
        :param use_pca_projection: whether to use pca projection, if set to False, random
        """
        super().__init__()

        self.input_dim = input_dim

        self.pca = IncrementalPCA(n_components=codebook_dim)
        self.register_buffer("pca_components", torch.empty((codebook_dim, input_dim)))

        # projection matrix use Xavier initialization
        p_init = torch.empty((input_dim, codebook_dim))
        self.register_buffer("p", nn.init.xavier_uniform_(p_init))

        # normalize random matrix for codebook
        self.register_buffer("cb", F.normalize(torch.randn(codebook_num_vars, codebook_dim)))
        self.use_pca_projection = use_pca_projection

    def forward(
        self,
        x: torch.Tensor,
        sequence_mask: torch.Tensor,
        pca_update_steps: Optional[int] = None,
        global_train_step: Optional[int] = None,
    ) -> torch.tensor:
        """
        :param x: the input tensor
        :param sequence_mask: the sequence padding mask
        :param pca_update_steps: the maximal number of training steps that we update the PCA
        :param global_train_step: the current training step
        """
        if global_train_step is not None and self.use_pca_projection:
            if (pca_update_steps is None) or (pca_update_steps is not None and global_train_step < pca_update_steps):
                self.pca.partial_fit(x[sequence_mask])
                self.pca_components = self.pca.components_

        if self.use_pca_projection and pca_update_steps > 0:
            x = x @ self.pca_components.T
            x = F.normalize(x, dim=-1)
        else:
            x = F.normalize(x @ self.p, dim=-1)

        targets = vector_norm((self.cb.unsqueeze(1) - x.unsqueeze(1)), dim=-1).argmin(dim=1)

        return targets
