import torch

def get_seq_mask(seq_lens, max_seq_len, device):
    """
    Convert seq_lens to a sequence mask.
    1 is in sequence, 0 is not in sequence.

    :param seq_lens: Sequence lengths (B,)
    :param max_seq_len: Maximum sequence length
    :return: Sequence mask in float32 (B, max_seq_len)
    """
    assert (seq_lens <= max_seq_len).all(), "One of the sequence length is larger than max seq len"
    batch_size = seq_lens.shape[0]
    seq_lens = seq_lens.to(device)
    seq_mask = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1) < seq_lens.unsqueeze(-1).expand(-1, max_seq_len)
    seq_mask = seq_mask.float().to(device)
    return seq_mask
