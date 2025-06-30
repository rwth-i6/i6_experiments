"""
Batch norm variations, e.g. Batch Renormalization

https://github.com/rwth-i6/returnn/issues/1539
"""

from __future__ import annotations
from typing import Optional, Union, Any, Callable
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


class BatchRenorm(rf.Module):
    """
    Batch Renormalization. https://arxiv.org/abs/1702.03275

    We calculate statistics over all axes except the given in_dim.
    I.e. all other axes are reduced for the statistics.

    To compensate the normalization, there are learnable parameters gamma and beta
    (optional, used when option `affine` is True, which is the default).

    The usual behavior depends on whether this is used in training or evaluation,
    although this often configurable in other frameworks.
    The usual behavior, in training::

        # Using statistics from current batch.
        mean_cur_batch, variance_cur_batch = moments(source, reduce_dims)
        y = (x - mean_cur_batch) / sqrt(variance_cur_batch + epsilon)
        y = gamma * y + beta

        # Updating running statistics for later use.
        mean = (1 - momentum) * mean + momentum * mean_cur_batch
        variance = (1 - momentum) * variance + momentum * variance_cur_batch

    The usual behavior, not in training (i.e. in evaluation)::

        # Using collected statistics. Not using statistics from current batch.
        y = (x - mean) / sqrt(variance + epsilon)
        y = gamma * y + beta

    """

    def __init__(
        self,
        in_dim: Dim,
        *,
        affine: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-3,
        use_mask: Optional[bool] = None,
        unbiased: bool = False,
        r_max: Union[float, Callable[[BatchRenorm], Union[float, Tensor]], Any] = 1.0,
        d_max: Union[float, Callable[[BatchRenorm], Union[float, Tensor]], Any] = 0.0,
    ):
        """
        :param in_dim: the feature dimension of the input
        :param affine: whether to use learnable parameters gamma and beta
        :param momentum: momentum for the running mean and variance
        :param eps: epsilon for the variance
        :param use_mask: whether to use a mask for dynamic spatial dims.
            This must be specified if the input has dynamic spatial dims.
            True would use the correct masking then. However, that is inconsistent to all other frameworks.
            False would be consistent to all other frameworks.
        :param unbiased: if True, uses unbiased variance calculation
            via `Bessel correction <https://en.wikipedia.org/wiki/Bessel%27s_correction>`__
        :param r_max: clip how much we should use the running variance instead of the current batch variance
            during training.
            Value should be >= 1.0.
            r_max=1.0 means always use the current batch variance, i.e. like standard batch norm.
            r_max=inf means always use the running variance.
            r_max can also be scheduled via a callable, e.g. using rf.get_run_ctx().step inside.
            The original paper suggests to keep r_max=1.0 for the first 5k steps,
            then linearly increase to reach r_max=3.0 at 40k steps.
            You can use ``rf.build_dict(rf.PiecewiseLinearStepwiseScheduler, points={5_000: 1.0, 40_000: 3.0})``.
        :param d_max: clip how much we should use the running mean instead of the current batch mean during training.
            Value should be >= 0.0.
            d_max=0.0 means always use the current batch mean, i.e. like standard batch norm.
            d_max=inf means always use the running mean.
            d_max can also be scheduled via a callable, e.g. using rf.get_run_ctx().step inside.
            The original paper suggests to keep d_max=0.0 for the first 5k steps,
            then linearly increase to reach d_max=5.0 at 25k steps.
            You can use ``rf.build_dict(rf.PiecewiseLinearStepwiseScheduler, points={5_000: 0.0, 25_000: 5.0})``.
        """
        super().__init__()
        assert isinstance(in_dim, Dim)
        self.in_dim = in_dim
        self.affine = affine
        self.momentum = momentum
        self.eps = eps
        self.use_mask = use_mask
        self.unbiased = unbiased
        if isinstance(r_max, dict):
            r_max = rf.build_from_dict(r_max)
        if isinstance(d_max, dict):
            d_max = rf.build_from_dict(d_max)
        self.r_max = r_max
        self.d_max = d_max
        self.running_mean = rf.Parameter([in_dim], auxiliary=True)
        self.running_mean.initial = 0.0
        self.running_variance = rf.Parameter([in_dim], auxiliary=True)
        self.running_variance.initial = 1.0
        self.gamma = None  # type: Optional[rf.Parameter]
        self.beta = None  # type: Optional[rf.Parameter]
        if self.affine:
            self.gamma = rf.Parameter([in_dim])
            self.gamma.initial = 1.0
            self.beta = rf.Parameter([in_dim])
            self.beta.initial = 0.0

    def __call__(self, source: Tensor) -> Tensor:
        assert self.in_dim in source.dims

        if any(d.need_masking() for d in source.dims if d != self.in_dim):
            if self.use_mask is None:
                raise ValueError(
                    f"{self}: use_mask must be specified if the input {source} has any dynamic spatial dims"
                )
            use_mask = self.use_mask
        else:
            use_mask = False  # not needed. False because this potentially enables an efficient fused op.

        train_flag = rf.get_run_ctx().train_flag
        d_max = self.d_max(self) if callable(self.d_max) else self.d_max
        r_max = self.r_max(self) if callable(self.r_max) else self.r_max

        mean_cur_batch, variance_cur_batch = rf.cond(
            train_flag,
            # Only conditionally calculate the moments when needed.
            lambda: rf.moments(
                source,
                axis=[d for d in source.dims if d != self.in_dim],
                use_mask=use_mask,
                correction=1 if self.unbiased else 0,
            ),
            # Return some dummy values. They are not used.
            lambda: (self.running_mean, self.running_variance),
        )

        def _update_running_stats():
            self.running_mean.assign_add((mean_cur_batch - self.running_mean) * self.momentum)
            self.running_variance.assign_add((variance_cur_batch - self.running_variance) * self.momentum)

        rf.cond(train_flag, _update_running_stats, lambda: None)

        def _train_mean_std_dev():
            inv_std_dev_ = rf.rsqrt(variance_cur_batch + self.eps)
            if isinstance(r_max, Tensor) or r_max > 1:
                inv_std_dev_ *= rf.clip_by_value(
                    rf.rsqrt(self.running_variance + self.eps)
                    * rf.sqrt(rf.stop_gradient(variance_cur_batch) + self.eps),
                    1 / r_max,
                    r_max,
                )
            mean_ = mean_cur_batch
            if isinstance(d_max, Tensor) or d_max > 0:
                limit = d_max * rf.reciprocal(rf.stop_gradient(inv_std_dev_))
                mean_ += rf.clip_by_value(self.running_mean - rf.stop_gradient(mean_cur_batch), -limit, limit)
            return mean_, inv_std_dev_

        mean, inv_std_dev = rf.cond(
            train_flag, _train_mean_std_dev, lambda: (self.running_mean, rf.rsqrt(self.running_variance + self.eps))
        )

        m = inv_std_dev
        if self.gamma is not None:
            m *= self.gamma
        bn = (source - mean) * m
        if self.beta is not None:
            bn += self.beta
        return bn
