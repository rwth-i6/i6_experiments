from torch import nn


def get_model_params(model: nn.Module) -> int:
    params = 0
    for param in model.parameters():
        params += param.numel()
    return params
