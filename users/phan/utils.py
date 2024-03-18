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


def init_lstm(lstm: torch.nn.LSTM, n_lstm_layers: int, init_func, init_args):
    """
    Init a torch lstm layer.

    :param lstm: A torch.nn.LSTM instance
    :param n_lstm_layers: Number of layer of the LSTM
    :param init_func: A function from torch.nn.init
    :param init_args: Args passed to the function besides tensor
    """
    for k in range(n_lstm_layers):
        init_func(getattr(lstm, f"weight_ih_l{k}"), **init_args)
        init_func(getattr(lstm, f"weight_hh_l{k}"), **init_args)
        init_func(getattr(lstm, f"bias_ih_l{k}"), **init_args)
        init_func(getattr(lstm, f"bias_hh_l{k}"), **init_args)
        weight_hr_lk = getattr(lstm, f"weight_hr_l{k}", None)
        if weight_hr_lk is not None:
            init_func(weight_hr_lk, **init_args)


def init_linear(linear: torch.nn.Linear, init_func, init_args):
    """
    Init a torch linear layer.
    """
    init_func(linear.weight, **init_args)
    init_func(linear.bias, **init_args)
