from dataclasses import dataclass
from sisyphus.tools import functools

import torch
import torchaudio

from i6_models.config import ModelConfiguration


def _mask(tensor: torch.Tensor, batch_axis: int, axis: int, pos: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    :param tensor: e.g. [B, ..., A, ...] but arbitrary axis order
    :param batch_axis: index of the batch axis
    :param axis: which axis A to mask
    :param pos: at which positions along axis to start the mask (size [B])
    :param max_len: mask length drawn uniformly from [0, max_len]
    """
    batch_dim = tensor.shape[batch_axis]
    dim = tensor.shape[axis]
    amount = torch.randint(low=1, high=max_len + 1, size=(batch_dim,), dtype=torch.int32, device=tensor.device)
    pos2 = torch.min(pos + amount, torch.tensor([dim] * batch_dim, device=tensor.device))
    idxs = torch.arange(0, dim, device=tensor.device).unsqueeze(0)  # [1,dim]
    pos_bc = pos.unsqueeze(1)  # [B,1]
    pos2_bc = pos2.unsqueeze(1)  # [B,1]
    cond = torch.logical_and(torch.greater_equal(idxs, pos_bc), torch.less(idxs, pos2_bc))  # [B,dim]
    if batch_axis > axis:
        cond = cond.transpose(0, 1)  # [dim,B]
    cond = torch.reshape(cond, shape=[tensor.shape[i] if i in (batch_axis, axis) else 1 for i in range(tensor.ndim)])
    tensor = torch.where(cond, 0.0, tensor)
    return tensor


def _random_mask(tensor: torch.Tensor, batch_axis: int, axis: int, min_num: int, max_num: int, max_len: int):
    """
    Mask tensor along axis using N in [min_num, max_num] masks of length [0, max_len]
    :param tensor: e.g. [B, ..., A, ...] but arbitrary axis order
    :param batch_axis: index of the batch axis
    :param axis: which axis to mask
    :param min_num: minimum number of masks
    :param max_num: maximum number of masks
    :param max_amount: mask length drawn uniformly from [0, max_amount]
    """

    batch_dim = tensor.shape[batch_axis]
    num_masks = torch.randint(min_num, max_num, size=(batch_dim,), device="cpu")  # [B]

    max_num_masks = num_masks.max().item()

    z = torch.rand((batch_dim, tensor.shape[axis]), device=tensor.device)  # [B,dim]
    _, indices = torch.topk(z, int(max_num_masks), dim=1)

    # Make num_masks broadcastable to shape of tensor for torch.where.
    num_masks = torch.reshape(num_masks, [1] * batch_axis + [batch_dim] + [1] * (tensor.dim() - batch_axis - 1))

    num_masks = num_masks.to(device=tensor.device)

    for i in range(int(max_num_masks)):
        tensor = torch.where(i < num_masks, _mask(tensor, batch_axis, axis, indices[:, i], max_len), tensor)

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
    :return: masked audio features
    """
    assert len(audio_features.shape) == 3
    assert time_min_num_masks <= time_max_num_masks
    assert freq_min_num_masks <= freq_max_num_masks
    masked_audio_features = _random_mask(
        audio_features, 0, 1, time_min_num_masks, time_max_num_masks, time_mask_max_size
    )  # time masking
    masked_audio_features = _random_mask(
        masked_audio_features, 0, 2, freq_min_num_masks, freq_max_num_masks, freq_mask_max_size
    )  # freq masking
    return masked_audio_features


@dataclass
class SpecaugmentConfigV1(ModelConfiguration):
    time_min_num_masks: int
    time_max_num_masks: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


class SpecaugmentModuleV1(torch.nn.Module):
    def __init__(self, step: int, cfg: SpecaugmentConfigV1):
        super().__init__()
        self.specaug_func = functools.partial(
            specaugment_v1,
            time_min_num_masks=cfg.time_min_num_masks,
            time_max_num_masks=cfg.time_max_num_masks,
            time_mask_max_size=cfg.time_mask_max_size,
            freq_min_num_masks=cfg.freq_min_num_masks,
            freq_max_num_masks=cfg.freq_max_num_masks,
            freq_mask_max_size=cfg.freq_mask_max_size,
        )

    def forward(self, audio_features: torch.Tensor):
        """
        :param audio_features: input tensor of shape [B, T, F]
        :return: torch.Tensor of shape [B, T, F]
        """
        if not self.training:
            return audio_features

        return self.specaug_func(audio_features)
