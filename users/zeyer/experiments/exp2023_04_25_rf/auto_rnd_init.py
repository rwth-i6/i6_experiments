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


import time

from .aed import from_scratch_model_def, Model, from_scratch_training


def test():
    import torch
    from returnn.util import better_exchook
    from returnn.util import basic as util
    from returnn.config import get_global_config
    from returnn.datasets.util.vocabulary import Vocabulary
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    better_exchook.install()

    config = get_global_config(auto_create=True)
    config.update({"behavior_version": 20})

    dev = torch.device("cuda")
    target_dim = Dim(1000, name="targets")
    target_dim.vocab = Vocabulary.create_vocab_from_labels([str(i) for i in range(target_dim.dimension)], eos_label=0)
    model: Model = from_scratch_model_def(epoch=1, in_dim=Dim(1, name="in"), target_dim=target_dim)

    pt_model = rf_module_to_pt_module(model)
    pt_model.to(dev)
    pt_model.eval()

    from i6_experiments.users.zeyer.audio.torch.random_speech_like import generate_random_speech_like_audio

    batch_size = 10
    sample_rate = 16_000
    duration = 5.0
    num_frames = int(duration * sample_rate)
    audio_raw = generate_random_speech_like_audio(batch_size, num_frames, samples_per_sec=sample_rate)
    audio_raw = audio_raw.to(dev)
    audio_lens_raw = torch.tensor([num_frames] * batch_size, dtype=torch.int32)
    audio_spatial_dim = Dim(Tensor("time", [batch_dim], dtype="int32", raw_tensor=audio_lens_raw))
    audio = Tensor("audio", [batch_dim, audio_spatial_dim], dtype="float32", raw_tensor=audio_raw)

    targets_len = int(duration * 3)
    targets_lens_raw = torch.tensor([targets_len] * batch_size, dtype=torch.int32)
    targets_spatial_dim = Dim(Tensor("targets_len", [batch_dim], dtype="int32", raw_tensor=targets_lens_raw))
    targets_raw = torch.randint(0, model.target_dim.dimension, size=(batch_size, targets_len), dtype=torch.int32)
    targets = Tensor(
        "targets", [batch_dim, targets_spatial_dim], dtype="int32", sparse_dim=model.target_dim, raw_tensor=targets_raw
    )

    start_time = time.time()
    rf.init_train_step_run_ctx(train_flag=False)
    from_scratch_training(
        model=model,
        data=audio,
        data_spatial_dim=audio_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
    )
    print("One train forward step, duration:", util.hms_fraction(time.time() - start_time), "sec")

    for name, loss in rf.get_run_ctx().losses.items():
        print(f"Loss {name}: {loss.get_mean_loss().raw_tensor.item()}")
