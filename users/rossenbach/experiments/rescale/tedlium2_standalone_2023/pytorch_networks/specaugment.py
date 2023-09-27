import torch

def _mask(tensor, batch_axis, axis, pos, max_amount):
    batch_dim = tensor.shape[batch_axis]
    dim = tensor.shape[axis]
    amount = torch.randint(low=1, high=max_amount + 1, size=(batch_dim,), dtype=torch.int32).to(device=tensor.device)
    pos2 = torch.min(pos + amount, torch.tensor([dim] * batch_dim).to(device=tensor.device))
    idxs = torch.arange(0, dim).to(device=tensor.device).unsqueeze(0)  # [1,dim]
    pos_bc = pos.unsqueeze(1)  # [B,1]
    pos2_bc = pos2.unsqueeze(1)  # [B,1]
    cond = torch.logical_and(torch.greater_equal(idxs, pos_bc), torch.less(idxs, pos2_bc))  # [B,dim]
    if batch_axis > axis:
        cond = cond.transpose(0, 1)  # [dim,B]
    cond = torch.reshape(
        cond, shape=[tensor.shape[i] if i in (batch_axis, axis) else 1 for i in range(len(tensor.shape))]
    )
    tensor = torch.where(cond, 0.0, tensor)
    return tensor


def _random_mask(tensor, batch_axis, axis, min_num, max_num, max_dims):
    batch_dim = tensor.shape[batch_axis]
    if min_num >= max_num:
        num_masks = torch.ones((batch_dim,), dtype=torch.int64) * min_num
    else:
        num_masks = torch.randint(min_num, max_num, size=(batch_dim,))  # [B]
    max_num_masks = num_masks.max().item()
    z = -torch.log(-torch.log(torch.rand((batch_dim, tensor.shape[axis])).to(device=tensor.device)))  # [B,dim]
    _, indices = torch.topk(z, max_num_masks, dim=1)
    
    # Make num_masks broadcastable to shape of tensor for torch.where.
    for i in range(tensor.dim() - 1):
        if i < batch_axis:
            num_masks = num_masks.unsqueeze(0)
        else:
            num_masks = num_masks.unsqueeze(-1)

    num_masks = num_masks.to(device=tensor.device)

    for i in range(max_num_masks):
        tensor = torch.where(
            i < num_masks,
            _mask(tensor, batch_axis, axis, indices[:, i], max_dims),
            tensor
        )

    return tensor


def returnn_specaugment(tensor: torch.Tensor, time_num_masks, time_mask_max_size, freq_num_masks, freq_mask_max_size):
    """
    Returnn like specaugment from legacy rossenbach/zeineldeen attention setups (usually called specaugment_v2 or so)

    :param tensor:
    :param time_num_masks:
    :param time_mask_max_size:
    :param freq_num_masks:
    :param freq_mask_max_size:
    :return:
    """
    assert len(tensor.shape) == 3
    tensor = _random_mask(tensor, 0, 1, 2, time_num_masks, time_mask_max_size)  # time masking
    tensor = _random_mask(tensor, 0, 2, 2, freq_num_masks, freq_mask_max_size)  # freq masking
    return tensor


def returnn_specaugment_by_length(audio_features, repeat_per_n_frames, max_dim_time, num_repeat_feat, max_dim_feat):
    """
    like returnn_specaugment, but with length adaptive num of time masks

    :param audio_features:
    :param repeat_per_n_frames:
    :param max_dim_time:
    :param num_repeat_feat:
    :param max_dim_feat:
    :return:
    """
    return returnn_specaugment(
        audio_features,
        time_num_masks=audio_features.size(1) // repeat_per_n_frames,
        time_mask_max_size=max_dim_time,
        freq_num_masks=num_repeat_feat,
        freq_mask_max_size=max_dim_feat)

