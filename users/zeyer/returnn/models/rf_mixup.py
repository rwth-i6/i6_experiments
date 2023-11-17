"""
Mixup with RF
"""

from __future__ import annotations
from dataclasses import dataclass
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


@dataclass
class MixupOpts:
    """
    Arguments:
        buffer_size: number of frames.
        apply_prob: probability to apply mixup at all
        max_num_mix: maximum number of mixups (random int in [1, max_num_mix])
        lambda_min: minimum lambda value
        lambda_max: maximum lambda value
    """

    buffer_size: int = 1_000_000
    apply_prob: float = 1.0
    max_num_mix: int = 4
    lambda_min: float = 0.1
    lambda_max: float = 0.4


class Mixup(rf.Module):
    """
    Mixup
    """

    def __init__(self, *, feature_dim: Dim, opts: MixupOpts):
        super().__init__()
        self.feature_dim = feature_dim
        self.opts = opts
        self.buffer_size_dim = Dim(opts.buffer_size, name="buffer_size")
        self.buffer = rf.Parameter([self.buffer_size_dim, feature_dim], auxiliary=True)
        self.buffer_pos = rf.Parameter(
            [], dtype="int32", sparse_dim=self.buffer_size_dim, initial=0, auxiliary=True, device="cpu"
        )
        self.buffer_filled = rf.Parameter([], dtype="bool", initial=False, auxiliary=True, device="cpu")

    def __call__(self, src: Tensor, *, spatial_dim: Dim) -> Tensor:
        if not rf.get_run_ctx().train_flag:
            return src

        assert spatial_dim in src.dims and self.feature_dim in src.dims

        # Apply mixup before we add the new data to the buffer.
        src_ = self._maybe_apply_mixup(src, spatial_dim=spatial_dim)

        self._append_to_buffer(src, spatial_dim=spatial_dim)

        return src_

    def _append_to_buffer(self, src: Tensor, *, spatial_dim: Dim):
        batch_dims = src.remaining_dims((spatial_dim, self.feature_dim))
        opts = self.opts

        # Fill buffer with new data:
        src_flat, src_flat_dim = rf.pack_padded(src, dims=batch_dims + [spatial_dim])
        new_pos = rf.minimum(self.buffer_pos + src_flat_dim.get_size_tensor(), opts.buffer_size)
        part_fill_len = new_pos - self.buffer_pos
        src_flat_part, src_flat_part_dim = rf.slice(src_flat, axis=src_flat_dim, end=part_fill_len)
        self.buffer.assign_key(
            axis=self.buffer_size_dim,
            key=slice(self.buffer_pos, new_pos),
            key_dim=src_flat_part_dim,
            value=src_flat_part,
        )
        if (self.buffer_pos + src_flat_dim.get_size_tensor() >= opts.buffer_size).raw_tensor:
            self.buffer_filled.assign(True)
            part_fill_len_ = rf.minimum(src_flat_dim.get_size_tensor() - part_fill_len, opts.buffer_size)
            src_flat_part, src_flat_part_dim = rf.slice(
                src_flat, axis=src_flat_dim, start=part_fill_len, end=part_fill_len + part_fill_len_
            )
            self.buffer.assign_key(
                axis=self.buffer_size_dim, key=slice(0, part_fill_len_), key_dim=src_flat_part_dim, value=src_flat_part
            )
            new_pos = part_fill_len_
        self.buffer_pos.assign(new_pos)

    def _maybe_apply_mixup(self, src: Tensor, *, spatial_dim: Dim) -> Tensor:
        if (rf.random_uniform((), device="cpu") >= opts.apply_prob).raw_tensor:
            return src

        batch_dims = src.remaining_dims((spatial_dim, self.feature_dim))
        opts = self.opts

        buffer_filled_size = rf.where(self.buffer_filled, opts.buffer_size, self.buffer_pos)
        if (buffer_filled_size < spatial_dim.get_dim_value_tensor()).raw_tensor:
            return src

        # Apply Mixup. Collect all data we are going to add for each sequence.
        num_mixup = rf.random_uniform(
            batch_dims, minval=1, maxval=opts.max_num_mix + 1, dtype="int32", device="cpu"
        )  # [B]
        num_mixup_dim = Dim(num_mixup, name="num_mixup")

        buffer_start = rf.random_uniform(
            batch_dims + [num_mixup_dim],
            maxval=buffer_filled_size - spatial_dim.get_dim_value_tensor() + 1,
            dtype="int32",
            sparse_dim=self.buffer_size_dim,
        )  # [B, N]
        n_mask = rf.sequence_mask(num_mixup_dim)  # [B, N]
        buffer_start_flat, num_mixup_flat_dim = rf.masked_select(
            buffer_start, mask=n_mask, dims=batch_dims + [num_mixup_dim]
        )  # [B_N']

        idx = rf.range_over_dim(spatial_dim)  # [T]
        idx = rf.combine_bc(idx, "+", buffer_start_flat)  # [B_N', T]

        mixup_values = rf.gather(self.buffer, indices=idx)  # [B_N', T, F]

        # Scale the mixup values.
        lambda_ = rf.random_uniform(
            batch_dims + [num_mixup_dim], minval=opts.lambda_min, maxval=opts.lambda_max, dtype=src.dtype
        )
        mixup_scales = rf.random_uniform(batch_dims + [num_mixup_dim], minval=0.001, maxval=1.0, dtype=src.dtype)
        mixup_scales *= lambda_ / rf.reduce_sum(mixup_scales, axis=num_mixup_dim)  # [B,N]
        mixup_scales_flat, _ = rf.masked_select(
            mixup_scales, mask=n_mask, dims=batch_dims + [num_mixup_dim], out_dim=num_mixup_flat_dim
        )  # [B_N']
        mixup_values *= mixup_scales_flat  # [B_N', T, F]

        mixup_value = rf.masked_scatter(
            mixup_values, mask=n_mask, dims=batch_dims + [num_mixup_dim], in_dim=num_mixup_flat_dim
        )  # [B,T,F]

        src = src + rf.stop_gradient(mixup_value)
        return src


def test_mixup():
    import numpy as np

    rf.select_backend_torch()
    rf.init_train_step_run_ctx(train_flag=True, step=0)

    batch_dim = Dim(2, name="batch")
    time_dim = Dim(rf.convert_to_tensor(np.array([7, 8], dtype="int32"), dims=[batch_dim]), name="time")
    feature_dim = Dim(5, name="feature")
    data = rf.random_uniform([batch_dim, time_dim, feature_dim])
    print("data:", data, data.raw_tensor)

    mixup = Mixup(feature_dim=feature_dim, opts=MixupOpts(buffer_size=100))

    x = mixup(data, spatial_dim=time_dim)
    print("x:", x, x.raw_tensor)
    print("buffer:", mixup.buffer, mixup.buffer.raw_tensor)

    batch_dim = Dim(3, name="batch")
    time_dim = Dim(rf.convert_to_tensor(np.array([3, 4, 2], dtype="int32"), dims=[batch_dim]), name="time")
    data = rf.ones([batch_dim, time_dim, feature_dim])
    x = mixup(data, spatial_dim=time_dim)
    print("x':", x, x.raw_tensor)
    print("buffer':", mixup.buffer, mixup.buffer.raw_tensor)
