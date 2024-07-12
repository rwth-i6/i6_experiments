import torch
import torch.nn as nn

from returnn.tensor import Tensor

def mask_eos_label(input: Tensor, eos_idx: int=0, blank_idx: int=10025, dim: int=-1, log_inf: float=-20000, add_to_blank: bool=False):
    """
    take rf tensor as input and return a rf tensor, however, the operation is done with torch tensors
    if add the eos prob to blank, then the log-prob has to be computed
    """

    output = input.raw_tensor
    assert dim==-1

    if add_to_blank:
        output = nn.functional.log_softmax(output, dim=dim)
    new_output = output.clone()
    if add_to_blank:
        new_output[:,:, blank_idx] = torch.logaddexp(output[:,:,eos_idx], output[:,:, blank_idx])
    new_output[:,:,eos_idx] = torch.ones_like(new_output[:,:,eos_idx]) * log_inf
    input.raw_tensor = new_output
    return input




