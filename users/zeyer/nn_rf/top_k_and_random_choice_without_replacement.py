"""
Top-K and random choice without replacement
"""

from typing import Union, Optional, Sequence, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def top_k_and_random_choice_without_replacement(
    log_probs: Tensor,
    *,
    axis: Union[Dim, Sequence[Dim]],
    k: Union[int, Tensor, Dim],
    min_noise_scale: float = 0.0,
    max_noise_scale: float = 1.0,
    max_noise_point: Optional[int] = None,
    top_p: Optional[Union[float, Tensor]] = None,
    top_p_one_more: bool = True,
) -> Tuple[Tensor, Union[Tensor, Sequence[Tensor]], Dim]:
    """
    :param log_probs: shape [ProbsDims..., Axis]
    :param axis: axis to sort and sample from (over classes)
    :param k: number of samples to draw
    :param min_noise_scale: minimum noise scale. with min_noise_scale=1, this reduces to pure random sampling.
    :param max_noise_scale: maximum noise scale. with max_noise_scale=0, this reduces to top-k.
    :param max_noise_point: index in the range of the axis where the Gumbel noise scaling starts to reach its maximum.
        If not given, uses ``k``.
    :param top_p: if given, use top-p sampling
    :param top_p_one_more: if True, keep one more token above the threshold
    :return: selected_log_probs, indices, out_dim.
        selected_log_probs shape [ProbsDims...].
        indices shape [ProbsDims..., OutDim]. indices -> axis.
        out_dim is k.
        The interface is compatible to :func:`top_k`.
    """
    import math

    # Similar as in :func:`top_p_mask`.
    # https://github.com/ari-holtzman/degen/blob/master/gen.py
    if isinstance(axis, Dim):
        assert axis.is_static()
        log_probs_, axis_ = log_probs, axis
    else:
        assert isinstance(axis, (tuple, list)) and all(isinstance(d, Dim) and d.is_static() for d in axis)
        log_probs_, axis_ = rf.merge_dims(log_probs, dims=axis)
    sorted_log_probs, sorted_indices, sorted_dim = rf.sort(log_probs_, axis=axis_, descending=True)
    # sorted_indices: {probs_dims..., sorted_dim} -> axis_
    out_dim = k if isinstance(k, Dim) else Dim(k, name=f"top_k_and_random_samples")
    if top_p is not None:
        cum_probs = rf.cumsum(rf.exp(sorted_log_probs), spatial_dim=sorted_dim)
        mask = cum_probs <= top_p  # {probs_dims..., sorted_dim}
        if top_p_one_more:
            # keep also the first token above the threshold
            mask = rf.shift_right(mask, axis=sorted_dim, pad_value=True)
        # Make sure enough tokens are included for top_k and sampling.
        mask = mask | (rf.range_over_dim(sorted_dim, device=mask.device) < out_dim.get_size_tensor(device=mask.device))
        sorted_log_probs = rf.where(mask, sorted_log_probs, float("-inf"))
    gumble_noise = -rf.log(-rf.log(rf.random_uniform(sorted_log_probs.dims)))  # {probs_dims..., sorted_dim}
    # Make sure the noise values are in the range [-inf, 0].
    gumble_noise = rf.log_softmax(gumble_noise, axis=sorted_dim)  # {probs_dims..., sorted_dim}
    indices = rf.range_over_dim(sorted_dim, dtype=gumble_noise.dtype, device=gumble_noise.device)  # {sorted_dim}
    if max_noise_point is not None:
        if max_noise_point == 0:
            assert min_noise_scale == max_noise_scale  # doesn't make sense otherwise
        linspace = indices * (1.0 / max(max_noise_point, 1))
    else:
        linspace = indices / rf.cast(rf.maximum(out_dim.get_size_tensor(device=indices.device), 1), dtype=indices.dtype)
    gumble_noise_scale = rf.clip_by_value(linspace, 0.0, 1.0)
    gumble_noise_scale = rf.sin(gumble_noise_scale * (math.pi / 2))  # make smoother
    gumble_noise_scale = gumble_noise_scale * (max_noise_scale - min_noise_scale) + min_noise_scale  # maybe scale
    noisy_log_probs = sorted_log_probs + gumble_noise * gumble_noise_scale
    _, indices, _ = rf.top_k(noisy_log_probs, k_dim=out_dim, axis=sorted_dim)
    # indices: {probs_dims..., out_dim} -> sorted_dim
    indices = rf.gather(sorted_indices, indices=indices)  # {probs_dims..., out_dim} -> axis_
    if isinstance(axis, Dim):
        log_probs = rf.gather(log_probs, indices=indices)  # {probs_dims..., out_dim}
        return log_probs, indices, out_dim
    else:
        assert isinstance(axis, (tuple, list)) and all(isinstance(d, Dim) for d in axis)
        # Exactly like in top_k.
        indices_out = []
        for i, a in reversed(list(enumerate(axis))):
            a: Dim
            assert a.is_static()
            indices_out_ = indices % a.dimension
            indices = indices // a.dimension
            indices_out_.sparse_dim = a
            indices_out.insert(0, indices_out_)
        for indices in indices_out:
            log_probs = rf.gather(log_probs, indices=indices)
        return log_probs, indices_out, out_dim
