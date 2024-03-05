import torch


def lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 1 means within sequence and 0 means outside sequence
    """
    batch_size = lengths.shape[0]
    max_length = torch.max(lengths)
    sequence_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return sequence_mask
