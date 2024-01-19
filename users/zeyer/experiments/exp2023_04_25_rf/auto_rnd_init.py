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

from __future__ import annotations
from typing import TYPE_CHECKING
import time

from .aed import from_scratch_model_def, Model, from_scratch_training

if TYPE_CHECKING:
    import torch
    from returnn.tensor import Dim


def test():
    import torch
    import returnn.__main__
    from returnn.util import better_exchook
    from returnn.util import basic as util
    from returnn.datasets.util.vocabulary import Vocabulary
    import returnn.frontend as rf
    from returnn.tensor import Dim
    from i6_experiments.users.zeyer.audio.torch.random_speech_like import generate_dummy_train_input_kwargs

    better_exchook.install()
    returnn.__main__.init_backend_engine(
        config_opts={"behavior_version": 20, "backend": "torch", "use_lovely_tensors": True}
    )

    target_dim = Dim(1000, name="targets")
    target_dim.vocab = Vocabulary.create_vocab_from_labels([str(i) for i in range(target_dim.dimension)], eos_label=0)
    model: Model = from_scratch_model_def(epoch=1, in_dim=Dim(1, name="in"), target_dim=target_dim)
    print("Num model parameters:", util.human_size(sum(p.num_elements() for p in model.parameters())))

    rf.set_default_device("cuda")
    model.to(device=rf.get_default_device())
    pt_dev = torch.device(rf.get_default_device())
    print(f"GPU memory usage (allocated model): {util.human_bytes_size(torch.cuda.memory_allocated(pt_dev))}")

    train_input_kwargs = generate_dummy_train_input_kwargs(dev=rf.get_default_device(), target_dim=target_dim)

    # TODO how to setup hooks?

    start_time = time.time()
    rf.init_train_step_run_ctx(train_flag=False)
    with torch.no_grad():
        from_scratch_training(model=model, **train_input_kwargs)
    print("One train forward step, duration:", util.hms_fraction(time.time() - start_time), "sec")
    print(f"GPU peak memory allocated: {util.human_bytes_size(torch.cuda.max_memory_allocated(pt_dev))}")

    for name, loss in rf.get_run_ctx().losses.items():
        print(f"Loss {name}: {loss.get_mean_loss().raw_tensor.item()}")
