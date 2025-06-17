from typing import Optional
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
    if not feature_dim:
        assert x.feature_dim
        feature_dim = x.feature_dim
    if not max_consecutive_feature_dims:
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
                min_num=rf.minimum(spatial_len, 2),
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
                min_num=2,
                max_num=5,
                max_dims=max_consecutive_feature_dims,
                mask_value=wn,
            )
        return x_masked

    return rf.cond(
        rf.get_run_ctx().is_train_flag_enabled(func=gensa_impl) | (not only_on_train), _mask_branch, lambda: x
    )
