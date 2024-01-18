"""
Inspired from recent Twitter discussion:
https://twitter.com/francoisfleuret/status/1747150719969321294
https://twitter.com/ducha_aiki/status/1747176379009896838

Where the following paper were mentioned:

- Layer-sequential unit-variance (LSUV), All you need is a good init, https://arxiv.org/abs/1511.06422, ICLR 2016
    - code: https://github.com/ducha-aiki/LSUV-pytorch
- Data-dependent Initializations of Convolutional Neural Networks, https://arxiv.org/abs/1511.06856, ICLR 2016
- Self-Normalizing Neural Networks, https://arxiv.org/abs/1706.02515, NeurIPS 2017
- Steering Deep Feature Learning with Backward Aligned Feature Updates, https://arxiv.org/abs/2311.18718

"""


from returnn.tensor import Dim
from returnn.torch.frontend.bridge import rf_module_to_pt_module
from .aed import from_scratch_model_def, Model


def test():
    import torch

    dev = torch.device("cuda")
    model: Model = from_scratch_model_def(epoch=1, in_dim=Dim(1, name="in"), target_dim=Dim(1000, name="targets"))
    pt_model = rf_module_to_pt_module(model)
    pt_model.to(dev)

    # TODO ...
