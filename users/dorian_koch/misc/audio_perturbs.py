from typing import Optional
from i6_experiments.users.zeyer.nn_rf.mixup import MixupOpts
from returnn.frontend import Tensor, Dim

import returnn.frontend as rf
from returnn.frontend.run_ctx import batch_dim
import numpy as np


def make_whitenoise_spectrogram(x: Tensor, *, spatial_dim: Dim, feature_dim: Dim) -> Tensor:
    """
    Create a white noise spectrogram for the given input tensor `x`.
    :param x: spectrogram like
    """
    from returnn.util import math as util_math

    batch_dims = x.remaining_dims((spatial_dim, feature_dim))
    assert len(batch_dims) == 1
    batch_dim = batch_dims[0]

    # TODO check these params
    sampling_rate = 16000
    window_len = 0.025
    window_num_frames = int(window_len * sampling_rate)
    frame_step = int(0.010 * sampling_rate)
    n_fft = util_math.next_power_of_two(window_num_frames)
    assert feature_dim.get_dim_value() == 80, "Expected feature_dim to be 80 for MelSpectrogram"
    #  ⌈(T - n_fft + 1) / frame_step⌉
    T = spatial_dim.get_dim_value_tensor() * frame_step + n_fft - 1
    white_spatial_dim = Dim(T, name="whitenoise_spatial_dim")
    white_signal = rf.random_normal([batch_dim, white_spatial_dim])
    source, white_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
        white_signal,
        in_spatial_dim=white_spatial_dim,
        out_dim=feature_dim,
        sampling_rate=sampling_rate,
    )
    # print(white_spatial_dim.get_dim_value())
    # print(spatial_dim.get_dim_value())
    assert white_spatial_dim.get_dim_value() >= spatial_dim.get_dim_value(), (
        f"{white_spatial_dim.get_dim_value_tensor().raw_tensor.cpu().numpy()} < {spatial_dim.get_dim_value_tensor().raw_tensor.cpu().numpy()}"
    )
    source = rf.replace_dim_v2(
        source, in_dim=white_spatial_dim, out_dim=spatial_dim, allow_expand=False, allow_shrink=True
    )
    return source


def save_spectogram(srctensor: Tensor, spectogram_path: str, *, feature_dim: Dim, spatial_dim: Dim):
    spectogram = (
        rf.gather(srctensor, indices=0, axis=batch_dim)
        .copy_compatible_to_dims([feature_dim, spatial_dim])
        .raw_tensor.cpu()
        .numpy()
    )
    # make image
    import matplotlib.pyplot as plt

    spectogram = np.flip(spectogram, axis=0)  # flip the frequency
    # spectogram = np.log(spectogram + 1e-6)  # log
    spectogram = np.clip(spectogram, -5, 5)  # clip
    spectogram = (spectogram + 5) / 10  # normalize
    # save to disk
    # spectogram_path =
    plt.imsave(spectogram_path, spectogram, cmap="viridis")
    print(f"Saved spectrogram to {spectogram_path}")


def generalized_specaugment(
    x: Tensor,
    *,
    spatial_dim: Dim,
    feature_dim: Optional[Dim] = None,
    only_on_train: bool = True,
    max_consecutive_spatial_dims: int = 20,
    max_consecutive_feature_dims: Optional[int] = None,
    num_spatial_mask_factor: int = 100,
) -> Tensor:
    """
    Adapted from rf.audio.specaugment
    """
    if feature_dim is None:
        assert x.feature_dim
        feature_dim = x.feature_dim
    if max_consecutive_feature_dims is None:
        max_consecutive_feature_dims = feature_dim.dimension // 5

    def _mask_branch():
        x_masked = x
        spatial_len = spatial_dim.get_dim_value_tensor()

        # make white noise spectograms
        wn = make_whitenoise_spectrogram(x_masked, spatial_dim=spatial_dim, feature_dim=feature_dim)
        # apply normalizations to make it more similar to actual input TODO is this needed?
        wn = rf.normalize(wn, axis=[batch_dim, spatial_dim])
        x_mean, x_variance = rf.moments(x_masked, axis=[batch_dim, spatial_dim], correction=1)
        wn = wn * rf.sqrt(x_variance + 1e-6) + x_mean

        """if model.feature_batch_norm:
            wn = model.feature_batch_norm(wn)
        if model.feature_norm:
            wn = rf.normalize(wn, axis=spatial_dim)
        if model.feature_stats:
            wn = (wn - model.feature_stats.mean) / model.feature_stats.std_dev"""

        # time mask
        if max_consecutive_spatial_dims > 0 and num_spatial_mask_factor > 0:
            x_masked = rf.audio.random_mask(
                x_masked,
                mask_axis=spatial_dim,
                broadcast_axis=feature_dim,
                min_num=rf.minimum(spatial_len, 0),
                max_num=rf.minimum(rf.maximum(spatial_len // num_spatial_mask_factor, 2) * 4, spatial_len),
                max_dims=max_consecutive_spatial_dims,
                mask_value=wn,
            )
        # feature mask
        if max_consecutive_feature_dims > 0:
            x_masked = rf.audio.random_mask(
                x_masked,
                mask_axis=feature_dim,
                broadcast_axis=spatial_dim,
                min_num=0,  # how many masks to apply
                max_num=5,
                max_dims=max_consecutive_feature_dims,  # how many consecutive dims to mask
                mask_value=wn,
            )
        return x_masked

    return rf.cond(
        rf.get_run_ctx().is_train_flag_enabled(func=generalized_specaugment) | (not only_on_train),
        _mask_branch,
        lambda: x,
    )


class MixupWithBugsFixed(rf.Module):
    """
    see i6_experiments/users/zeyer/nn_rf/mixup.py

    Changes:
    - use appropriate trainflag
    - fix scatter logic at the end
    - (1-lambda)src + lambda * mixup instead of src + lambda * mixup
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
        if not rf.get_run_ctx().is_train_flag_enabled(func=MixupWithBugsFixed.__call__):
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
        src_flat, src_flat_dim = rf.pack_padded(src, dims=[*batch_dims, spatial_dim])
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
        batch_dims = src.remaining_dims((spatial_dim, self.feature_dim))
        opts = self.opts

        if (rf.random_uniform((), device="cpu") >= opts.apply_prob).raw_tensor:
            return src

        buffer_filled_size = rf.where(self.buffer_filled, opts.buffer_size, self.buffer_pos)
        if (buffer_filled_size < spatial_dim.get_dim_value_tensor()).raw_tensor.item():
            return src

        # Apply Mixup. Collect all data we are going to add for each sequence.
        num_mixup = rf.random_uniform(
            batch_dims, minval=1, maxval=opts.max_num_mix + 1, dtype="int32", device="cpu"
        )  # [B]
        num_mixup_dim = Dim(num_mixup, name="num_mixup")

        buffer_start = rf.random_uniform(
            [*batch_dims, num_mixup_dim],
            maxval=buffer_filled_size - spatial_dim.get_dim_value_tensor() + 1,
            dtype="int32",
            sparse_dim=self.buffer_size_dim,
        )  # [B, N]
        n_mask = rf.sequence_mask(num_mixup_dim)  # [B, N]
        buffer_start_flat, num_mixup_flat_dim = rf.masked_select(
            buffer_start, mask=n_mask, dims=[*batch_dims, num_mixup_dim]
        )  # [B_N']

        idx = rf.range_over_dim(spatial_dim)  # [T]
        idx = rf.combine_bc(idx, "+", buffer_start_flat)  # [B_N', T]

        mixup_values = rf.gather(self.buffer, indices=idx, axis=self.buffer_size_dim)  # [B_N', T, F]

        # Scale the mixup values.
        lambda_ = rf.random_uniform(batch_dims, minval=opts.lambda_min, maxval=opts.lambda_max, dtype=src.dtype)
        mixup_scales = rf.random_uniform([*batch_dims, num_mixup_dim], minval=0.001, maxval=1.0, dtype=src.dtype)
        mixup_scales /= rf.reduce_sum(mixup_scales, axis=num_mixup_dim)  # [B,N]
        mixup_scales_flat, _ = rf.masked_select(
            mixup_scales, mask=n_mask, dims=[*batch_dims, num_mixup_dim], out_dim=num_mixup_flat_dim
        )  # [B_N']
        mixup_values *= mixup_scales_flat  # [B_N', T, F]

        mixup_value = rf.masked_scatter(
            mixup_values, mask=n_mask, dims=[*batch_dims, num_mixup_dim], in_dim=num_mixup_flat_dim
        )
        mixup_value = rf.reduce_sum(mixup_value, axis=num_mixup_dim)
        # idx_b = rf.range_over_merged_dims(batch_dims)  # [B] -> B
        # idx_b, _ = rf.masked_select(
        #    idx_b, mask=n_mask, dims=batch_dims + [num_mixup_dim], out_dim=num_mixup_flat_dim
        # )  # [B_N'] -> B

        # mixup_value = rf.scatter(
        #    mixup_values, indices=idx_b, indices_dim=num_mixup_flat_dim, out_dim=batch_dims
        # )  # [B,T,F]

        src = (1 - lambda_) * src + lambda_ * rf.stop_gradient(mixup_value)
        return src
