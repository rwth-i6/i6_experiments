import torch


def map_tensor_to_minus1_plus1_interval(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(tensor):
        return tensor

    dtype = tensor.dtype
    info = torch.iinfo(dtype)
    min_val = info.min
    max_val = info.max

    return 2.0 * (tensor.float() - min_val) / (max_val - min_val) - 1.0
