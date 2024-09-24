"""
Common functions to add BOS/EOS tokens
"""
import torch

def add_bos(seqs: torch.Tensor, bos_index: int = 0):
    """
    :param seqs: Sequences in (B, T), should be long
    :param bos_index: index used as BOS
    :returns: The sequences with BOS added
    """
    batch_size = seqs.shape[0]
    bos_seqs = torch.cat(
        [torch.full((batch_size, 1), bos_index, device=seqs.device), seqs],
        dim=1,
    ).long()
    return bos_seqs

def add_eos(seqs: torch.Tensor, eos_index: int = 0):
    """
    :param seqs: Sequences in (B, T), should be long
    :param eos_index: index used as EOS
    :returns: The sequences with EOS added
    """
    batch_size = seqs.shape[0]
    seqs_eos = torch.cat(
        [seqs, torch.full((batch_size, 1), eos_index, device=seqs.device)],
        dim=1,
    ).long()
    return seqs_eos
