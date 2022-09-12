from returnn_common import nn
from returnn_common.asr.specaugment import specaugment_eval_func


class SpecAugment(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, input_data: nn.Tensor) -> nn.Tensor:
        out = nn.eval(source=input_data, eval=specaugment_eval_func)
        return out
