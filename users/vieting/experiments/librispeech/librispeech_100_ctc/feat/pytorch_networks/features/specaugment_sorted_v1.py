import numpy as np
import torch
from typing import Optional


def _mask(
    tensor: torch.Tensor,
    batch_axis: int,
    axis: int,
    pos: torch.Tensor,
    max_len: int,
    sorted_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    :param tensor: e.g. [B, ..., A, ...] but arbitrary axis order
    :param batch_axis: index of the batch axis
    :param axis: which axis A to mask
    :param pos: at which positions along axis to start the mask (size [B])
    :param max_len: mask length drawn uniformly from [0, max_len]
    :param sorted_indices: apply masks based on specified sorting
    """
    batch_dim_size = tensor.shape[batch_axis]
    mask_dim_size = tensor.shape[axis]
    mask_len = torch.randint(low=1, high=max_len + 1, size=(batch_dim_size,), dtype=torch.int32, device=tensor.device)
    end_pos = torch.min(pos + mask_len, torch.tensor([mask_dim_size] * batch_dim_size, device=tensor.device))
    idxs = torch.arange(0, mask_dim_size, device=tensor.device).unsqueeze(0)  # [1,dim]
    pos_bc = pos.unsqueeze(1)  # [B,1]
    end_pos_bc = end_pos.unsqueeze(1)  # [B,1]
    mask = torch.logical_and(torch.greater_equal(idxs, pos_bc), torch.less(idxs, end_pos_bc))  # [B,dim]
    if batch_axis > axis:
        mask = mask.transpose(0, 1)  # [dim,B]
    mask = torch.reshape(mask, shape=[tensor.shape[i] if i in (batch_axis, axis) else 1 for i in range(tensor.ndim)])
    if sorted_indices is not None:
        axes_template = [1 for _ in range(tensor.ndim)]
        axes_template[batch_axis] = batch_dim_size
        inverse_permutation = torch.argsort(sorted_indices).repeat(axes_template)
        inverse_permutation.transpose(axis, -1)
        mask = torch.gather(mask, axis, inverse_permutation)
    tensor = torch.where(mask, 0.0, tensor)
    return tensor


def _random_mask(tensor: torch.Tensor, batch_axis: int, axis: int, min_num: int, max_num: int, max_len: int, **kwargs):
    """
    Mask tensor along axis using N in [min_num, max_num] masks of length [0, max_len]

    :param tensor: e.g. [B, ..., A, ...] but arbitrary axis order
    :param batch_axis: index of the batch axis
    :param axis: which axis to mask
    :param min_num: minimum number of masks
    :param max_num: maximum number of masks
    :param max_amount: mask length drawn uniformly from [0, max_amount]
    """

    batch_dim_size = tensor.shape[batch_axis]
    if max_num < min_num:
        max_num = min_num
    num_masks = torch.randint(min_num, max_num + 1, size=(batch_dim_size,), device="cpu")  # [B]

    max_num_masks = num_masks.max().item()

    z = torch.rand((batch_dim_size, tensor.shape[axis]), device=tensor.device)  # [B,dim]
    _, indices = torch.topk(z, max_num_masks, dim=1)

    # Make num_masks broadcastable to shape of tensor for torch.where.
    num_masks = torch.reshape(num_masks, [1] * batch_axis + [batch_dim_size] + [1] * (tensor.dim() - batch_axis - 1))

    num_masks = num_masks.to(device=tensor.device)

    for i in range(max_num_masks):
        tensor = torch.where(i < num_masks, _mask(tensor, batch_axis, axis, indices[:, i], max_len, **kwargs), tensor)

    return tensor


def specaugment_v1(
    audio_features: torch.Tensor,
    *,
    time_min_num_masks: int,
    time_max_num_masks: int,
    time_mask_max_size: int,
    freq_min_num_masks: int,
    freq_max_num_masks: int,
    freq_mask_max_size: int,
    sorted_indices: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Specaugment from legacy rossenbach/zeineldeen/zeyer attention setups e.g.,
    https://github.com/rwth-i6/i6_experiments/blob/main/users/zeineldeen/data_aug/specaugment/specaug_tf2.py
    but without any step-based scheduling and without dependence on length.
    See `specaugment_v1_by_length` for a variant which is more close to the original.

    Fills masks with zeros.

    Basically just a convenience wrapper around _random_mask.

    See also: https://arxiv.org/abs/1904.08779

    :param audio_features: e.g. log-mel features as [B, T, F]
    :param time_min_num_masks: minimum number of masks along T
    :param time_max_num_masks: maximum number of masks along T
    :param time_mask_max_size: maximum size of masks along T
    :param freq_min_num_masks: minimum number of masks along F
    :param freq_max_num_masks: maximum number of masks along F
    :param freq_mask_max_size: maximum size of masks along F
    :param sorted_indices: apply masks based on specified sorting
    :return: masked audio features
    """
    assert len(audio_features.shape) == 3
    assert time_min_num_masks <= time_max_num_masks
    assert freq_min_num_masks <= freq_max_num_masks
    masked_audio_features = _random_mask(
        audio_features, 0, 1, time_min_num_masks, time_max_num_masks, time_mask_max_size,
    )  # time masking
    masked_audio_features = _random_mask(
        masked_audio_features, 0, 2, freq_min_num_masks, freq_max_num_masks, freq_mask_max_size,
        sorted_indices=sorted_indices,
    )  # freq masking
    return masked_audio_features


def specaugment_v1_by_length(
    audio_features: torch.Tensor,
    *,
    time_min_num_masks: int,
    time_max_mask_per_n_frames: int,
    time_mask_max_size: int,
    freq_min_num_masks: int,
    freq_max_num_masks: int,
    freq_mask_max_size: int,
    **kwargs,
):
    """
    Convenience wrapper around specaugment_v1 with time-length adaptive number of masks.

    :param audio_features: e.g. log-mel features as [B, T, F]
    :param time_max_mask_per_n_frames: used for the maximum number time masks,
        max_num_masks = T / max_mask_per_n_frames for each batch.
        They are still drawn depending on the full batch length, so shorter sequences
        might get more masks than that by chance, or none at all when all masks
        fall into the padding space.
    :param time_min_num_masks: minimum number of masks along T
    :param time_mask_max_size: maximum size of masks along T
    :param freq_min_num_masks: minimum number of masks along F
    :param freq_max_num_masks: maximum number of masks along F
    :param freq_mask_max_size: maximum size of masks along F
    :return: masked audio features
    """
    return specaugment_v1(
        audio_features,
        time_min_num_masks=time_min_num_masks,
        time_max_num_masks=np.maximum(audio_features.size(1) // time_max_mask_per_n_frames, time_min_num_masks),
        time_mask_max_size=time_mask_max_size,
        freq_min_num_masks=freq_min_num_masks,
        freq_max_num_masks=freq_max_num_masks,
        freq_mask_max_size=freq_mask_max_size,
        **kwargs,
    )
