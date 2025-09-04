"""
Nucleus sampling / top-p sampling implementation

See also :func:`top_k_and_random_choice_without_replacement`
"""

from typing import Union, Optional, Sequence, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


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
