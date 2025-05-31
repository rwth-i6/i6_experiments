import torch
from torch import nn
from typing import Tuple

from i6_native_ops import warp_rnnt
from .train_handler import TrainStepMode, LossEntry, List, Dict
from ..rnnt.base_streamable_rnnt import StreamableRNNT
from ..common import Mode



class RNNTTrainStepMode(TrainStepMode):
    def __init__(self):
        super().__init__()

    def step(self, model: StreamableRNNT, data: dict, mode: Mode, scale: float) -> Tuple[Dict, int]:

        raw_audio = data["raw_audio"]  # [B, T', F]
        raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

        labels = data["labels"]  # [B, N] (sparse)
        labels_len = data["labels:size1"]  # [B, N]

        prepended_targets = labels.new_empty([labels.size(0), labels.size(1) + 1])
        prepended_targets[:, 1:] = labels
        prepended_targets[:, 0] = model.cfg.label_target_size  # blank is last index
        prepended_target_lengths = labels_len + 1

        num_phonemes = torch.sum(labels_len)
    
        model.set_mode_cascaded(mode)
        logits, audio_features_len, ctc_logprobs = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
            labels=prepended_targets, labels_len=prepended_target_lengths,
        )
        
        logprobs = torch.log_softmax(logits, dim=-1)
        rnnt_loss = warp_rnnt.rnnt_loss(
            log_probs=logprobs,
            frames_lengths=audio_features_len.to(dtype=torch.int32),
            labels=labels,
            labels_lengths=labels_len.to(dtype=torch.int32),
            blank=model.cfg.label_target_size,
            fastemit_lambda=0.0,
            reduction="sum",
            gather=True,
        )

        ctc_loss = None
        if ctc_logprobs is not None:
            transposed_logprobs = torch.permute(ctc_logprobs, (1, 0, 2))  # CTC needs [T, B, #vocab + 1]

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
            "rnnt.%s" % mode_str: {
                "loss": rnnt_loss,
                "scale": scale
            },
            "ctc.%s" % mode_str: {
                "loss": ctc_loss,
                "scale": model.cfg.ctc_output_loss * scale
            }
        }
        
        return loss_dict, num_phonemes
