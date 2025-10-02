import torch
from torch import nn
from typing import Tuple
import numpy as np

from ..trainers.train_handler import TrainStepMode, Dict, TrainMode
from ..streamable_module import StreamableModule
from ..common import Mode


class CTCTrainStepMode(TrainStepMode):
    def __init__(self):
        super().__init__()

    def step(self, model: StreamableModule, data: dict, mode: Mode, scale: float) -> Tuple[Dict, int]:
        raw_audio = data["raw_audio"]  # [B, T', F]
        raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

        labels = data["labels"]  # [B, N] (sparse)
        labels_len = data["labels:size1"]  # [B, N]

        num_phonemes = torch.sum(labels_len)

        model.set_mode_cascaded(mode)
        logits, audio_features_len = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
        )
        model.unset_mode_cascaded()
        logprobs = torch.log_softmax(logits, dim=2)  # in older setup done inside of model

        transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, #vocab + 1]

        ctc_loss = nn.functional.ctc_loss(
            transposed_logprobs,
            labels,
            input_lengths=audio_features_len,
            target_lengths=labels_len,
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )

        mode_str = mode.name.lower()[:3]
        loss_dict = {
            "ctc.%s" % mode_str: {
                "loss": ctc_loss,
                "scale": scale
            }
        }

        return loss_dict, num_phonemes
    


def prior_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: StreamableModule, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    if model.cfg.train_mode == TrainMode.OFFLINE:
        model.set_mode_cascaded(Mode.OFFLINE)
    elif model.cfg.train_mode == TrainMode.STREAMING:
        model.set_mode_cascaded(Mode.STREAMING)
    elif model.cfg.train_mode == TrainMode.SWITCHING:
        model.set_mode_cascaded(Mode.STREAMING if run_ctx.global_step % 2 == 0 else Mode.OFFLINE)
    else:
        raise NotImplementedError

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    model.unset_mode_cascaded()

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))


