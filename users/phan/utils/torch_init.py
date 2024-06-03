import torch
import numpy as np

def init_lstm(lstm: torch.nn.LSTM, n_lstm_layers: int, init_func, init_args, bidirectional: bool = False):
    """
    Init a torch lstm layer.

    :param lstm: A torch.nn.LSTM instance
    :param n_lstm_layers: Number of layer of the LSTM
    :param init_func: A function from torch.nn.init
    :param init_args: Args passed to the function besides tensor
    :param bidirectional: If the LSTM is bidirectional or not
    """
    for k in range(n_lstm_layers):
        init_func(getattr(lstm, f"weight_ih_l{k}"), **init_args)
        init_func(getattr(lstm, f"weight_hh_l{k}"), **init_args)
        init_func(getattr(lstm, f"bias_ih_l{k}"), **init_args)
        init_func(getattr(lstm, f"bias_hh_l{k}"), **init_args)
        if bidirectional:
            init_func(getattr(lstm, f"weight_ih_l{k}_reverse"), **init_args)
            init_func(getattr(lstm, f"weight_hh_l{k}_reverse"), **init_args)
            init_func(getattr(lstm, f"bias_ih_l{k}_reverse"), **init_args)
            init_func(getattr(lstm, f"bias_hh_l{k}_reverse"), **init_args)
        weight_hr_lk = getattr(lstm, f"weight_hr_l{k}", None)
        if weight_hr_lk is not None:
            init_func(weight_hr_lk, **init_args)
        weight_hr_lk_reverse = getattr(lstm, f"weight_hr_l{k}_reverse", None)
        if weight_hr_lk_reverse is not None:
            init_func(weight_hr_lk_reverse, **init_args)


def init_linear(linear: torch.nn.Linear, init_func, init_args, bias=True):
    """
    Init a torch linear layer.
    """
    init_func(linear.weight, **init_args)
    if bias:
        init_func(linear.bias, **init_args)
