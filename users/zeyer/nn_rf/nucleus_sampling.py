"""
Nucleus sampling / top-p sampling implementation

See also :func:`top_k_and_random_choice_without_replacement`
"""

from typing import Union, Optional, Sequence, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def nucleus_sampling_beam_search(
    prev_accumulated_log_probs: Tensor,
    log_probs: Tensor,
    *,
    axis: Union[Dim, Sequence[Dim]],
    k: Union[int, Tensor, Dim],
    top_p_mask_on_accumulated_log_probs: bool = False,
    top_p_mask_on_combined_axes: bool = False,
    top_p: Union[float, Tensor],
    gumble_noise_scale: float = 1.0,
) -> Tuple[Tensor, Union[Tensor, Sequence[Tensor]]]:
    """
    Nucleus sampling / top-p sampling implementation for stochastic beam search.
    The sampling is always from the combined prob distrib over the given axes.

    :param prev_accumulated_log_probs: shape [ProbsDims..., CommonAxis...].
        CommonAxis... could be [InBeam].
    :param log_probs: shape [ProbsDims..., CommonAxis..., RemainingAxis...].
        CommonAxis... could be [InBeam], RemainingAxis... could be [Vocab].
        This is usually the current log softmax output (not yet accumulated).
    :param axis: axis / axes to sort and sample from (over classes). If multiple, e.g. like [InBeam,Vocab].
        CommonAxis... + RemainingAxis.... CommonAxis... could be [InBeam], RemainingAxis... could be [Vocab].
    :param k: number of (total) samples to draw. (Sometimes we call this OutBeam.)
    :param top_p_mask_on_accumulated_log_probs: if True, apply top-p mask on prev_accumulated_log_probs + log_probs,
        otherwise only on log_probs.
    :param top_p_mask_on_combined_axes: if True and if multiple axes given,
        combine the axes first, apply top-p mask, then split again.
        (This is only about the top-p masking, not the sampling. The sampling is always from the combined prob distrib.)
    :param top_p: for top-p masking.
    :param gumble_noise_scale: for sampling. 1.0 means normal sampling (stochastic beam search),
        0.0 means pure topk (no sampling), i.e. 0.0 is standard (non-stochastic) beam-search.
    :return: selected_log_probs, indices.
        selected_log_probs shape [ProbsDims...], selected from prev_accumulated_log_probs + log_probs.
        indices shape [ProbsDims...]. indices -> axis.
    """
    if isinstance(axis, Dim):
        assert axis.is_static()
        axes = [axis]
    else:
        assert (
            isinstance(axis, (tuple, list))
            and all(isinstance(d, Dim) and d.is_static() for d in axis)
            and len(axis) > 0
        )
        axes = list(axis)
    assert set(axes).issubset(log_probs.dims)
    remaining_axes = log_probs.remaining_dims(prev_accumulated_log_probs.dims)
    assert set(remaining_axes).issubset(axes)
    assert len(remaining_axes) == 1
    remaining_axis = remaining_axes[0]
    if len(axes) == 1:
        assert axes[0] == remaining_axis

    if top_p_mask_on_accumulated_log_probs:
        # combine now (otherwise, combine later)
        log_probs = prev_accumulated_log_probs + log_probs

    if top_p_mask_on_combined_axes and len(axes) > 1:
        log_probs_, axis_ = rf.merge_dims(log_probs, dims=axes)
        log_probs_ = rf.where(
            rf.top_p_mask(rf.softmax(log_probs_, axis=axis_), p=top_p, axis=axis_), log_probs_, float("-inf")
        )
        log_probs = rf.split_dims(log_probs_, axis=axis_, dims=axes)
    else:
        log_probs = rf.where(
            rf.top_p_mask(rf.softmax(log_probs, axis=remaining_axis), p=top_p, axis=remaining_axis),
            log_probs,
            float("-inf"),
        )

    if not top_p_mask_on_accumulated_log_probs:
        log_probs = prev_accumulated_log_probs + log_probs

    if len(axes) == 1:
        log_probs_, axis_ = log_probs, axes[0]
    else:
        log_probs_, axis_ = rf.merge_dims(log_probs, dims=axes)

    # log_probs_ (again after masking) to have proper prob distrib and for better combination with Gumble noise.
    # The sampling is from the combined prob distrib over the given axes.
    log_probs_ = rf.log_softmax(log_probs_, axis=axis_)
    gumble_noise = -rf.log(-rf.log(rf.random_uniform(log_probs_.dims)))  # {probs_dims..., sorted_dim}
    # Make sure the noise values are in the range [-inf, 0].
    gumble_noise = rf.log_softmax(gumble_noise, axis=axis_)  # {probs_dims..., sorted_dim}
    noisy_log_probs = log_probs_ + gumble_noise * gumble_noise_scale
    out_dim = k if isinstance(k, Dim) else Dim(k, name="top_k_and_random_samples")
    _, indices, _ = rf.top_k(noisy_log_probs, k_dim=out_dim, axis=axis_)

    if len(axes) == 1:
        log_probs = rf.gather(log_probs, indices=indices)  # {probs_dims...}
        return log_probs, indices
    else:
        # Exactly like in top_k.
        indices_out = []
        for i, a in reversed(list(enumerate(axes))):
            a: Dim
            assert a.is_static()
            indices_out_ = indices % a.dimension
            indices = indices // a.dimension
            indices_out_.sparse_dim = a
            indices_out.insert(0, indices_out_)
        for indices in indices_out:
            log_probs = rf.gather(log_probs, indices=indices)
        return log_probs, indices_out


def nucleus_sampling(
    log_probs: Tensor,
    *,
    axis: Union[Dim, Sequence[Dim]],
    gumble_noise_scale: float = 1.0,
    top_p: Optional[Union[float, Tensor]],
    top_p_one_more: bool = True,
) -> Tuple[Tensor, Union[Tensor, Sequence[Tensor]]]:
    """
    :param log_probs: shape [ProbsDims..., Axis]
    :param axis: axis to sort and sample from (over classes)
    :param gumble_noise_scale:
    :param top_p: if given, use top-p sampling
    :param top_p_one_more: if True, keep one more token above the threshold
    :return: selected_log_probs, indices.
        selected_log_probs shape [ProbsDims...].
        indices shape [ProbsDims...]. indices -> axis.
    """

    # Similar as in :func:`top_p_mask`.
    # https://github.com/ari-holtzman/degen/blob/master/gen.py
    if isinstance(axis, Dim):
        assert axis.is_static()
        log_probs_, axis_ = log_probs, axis
    else:
        assert isinstance(axis, (tuple, list)) and all(isinstance(d, Dim) and d.is_static() for d in axis)
        log_probs_, axis_ = rf.merge_dims(log_probs, dims=axis)

    if top_p is not None:
        sorted_log_probs, sorted_indices, sorted_dim = rf.sort(log_probs_, axis=axis_, descending=True)
        # sorted_indices: {probs_dims..., sorted_dim} -> axis_
        # renorm in case it was not normalized (e.g. during search)
        probs = rf.softmax(sorted_log_probs, axis=sorted_dim)
        cum_probs = rf.cumsum(probs, spatial_dim=sorted_dim)
        mask = cum_probs <= top_p  # {probs_dims..., sorted_dim}
        if top_p_one_more:
            # keep also the first token above the threshold
            mask = rf.shift_right(mask, axis=sorted_dim, pad_value=True)
        else:
            # Make sure enough tokens are included for top_k and sampling.
            mask = mask | (rf.range_over_dim(sorted_dim, device=mask.device) < 1)
        sorted_log_probs = rf.where(mask, sorted_log_probs, float("-inf"))
    else:  # top_p is None, no masking needed, no sorting needed
        sorted_log_probs = log_probs_
        sorted_dim = axis_
        sorted_indices = None

    if gumble_noise_scale == 1:  # default. do normal sampling
        # renorm (again after masking) to have proper prob distrib, and also such that we have no underflow
        probs = rf.softmax(sorted_log_probs, axis=sorted_dim)
        indices = rf.random_choice_with_replacement(
            sorted_log_probs.remaining_dims(sorted_dim), probs=probs, axis=sorted_dim
        )
    elif gumble_noise_scale:
        # renorm (again after masking) to have proper prob distrib and for better combination with gumble noise
        sorted_log_probs = rf.log_softmax(sorted_log_probs, axis=sorted_dim)
        gumble_noise = -rf.log(-rf.log(rf.random_uniform(sorted_log_probs.dims)))  # {probs_dims..., sorted_dim}
        # Make sure the noise values are in the range [-inf, 0].
        gumble_noise = rf.log_softmax(gumble_noise, axis=sorted_dim)  # {probs_dims..., sorted_dim}
        noisy_log_probs = sorted_log_probs + gumble_noise * gumble_noise_scale
        indices = rf.reduce_argmax(noisy_log_probs, axis=sorted_dim)
    else:  # gumble_noise_scale == 0
        indices = rf.reduce_argmax(sorted_log_probs, axis=sorted_dim)

    if sorted_indices is not None:
        # indices: {probs_dims...} -> sorted_dim
        indices = rf.gather(sorted_indices, indices=indices)  # {probs_dims...} -> axis_

    if isinstance(axis, Dim):
        log_probs = rf.gather(log_probs, indices=indices)  # {probs_dims...}
        return log_probs, indices
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
        return log_probs, indices_out
