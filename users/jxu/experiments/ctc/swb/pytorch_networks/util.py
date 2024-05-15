import torch

def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to an equivalent boolean mask

    :param lengths: [B]
    :return: B x T, where 1 means within sequence and 0 means outside sequence
    """
    max_length = torch.max(lengths)
    index_range = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype)
    sequence_mask = torch.less(index_range[None, :], lengths[:, None])

    return sequence_mask