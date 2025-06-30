import torch
from torch import nn
from typing import Tuple

from i6_native_ops import monotonic_rnnt
from .train_handler import TrainStepMode, LossEntry, List, Dict
from ..streamable_module import StreamableModule
from ..common import Mode



class MRNNTTrainStepMode(TrainStepMode):
    def __init__(self):
        super().__init__()

    def step(self, model: StreamableModule, data: dict, mode: Mode, scale: float) -> Tuple[Dict, int]:

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

        has_mismatch = False
        for b in range(audio_features_len.size(0)):
            # check if T <= S, otherwise mismatch
            if labels_len[b] > audio_features_len[b]:
                assert False, "T <= S"
                # print(
                #     data["seq_tag"][b], "has", labels_len[b], "targets but only", audio_features_len[b], "encoder states"
                # )
                # has_mismatch = True

        if not has_mismatch:
            rnnt_loss = monotonic_rnnt.monotonic_rnnt_loss(
                acts=logits.to(dtype=torch.float32),
                labels=labels,
                input_lengths=audio_features_len.to(dtype=torch.int32),
                label_lengths=labels_len.to(dtype=torch.int32),
                blank_label=model.cfg.label_target_size,
            ).sum()
        else:
            rnnt_loss = torch.zeros([], dtype=torch.float32, device=logits.device)

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
            "mrnnt.%s" % mode_str: {
                "loss": rnnt_loss,
                "scale": scale
            },
            "ctc.%s" % mode_str: {
                "loss": ctc_loss,
                "scale": model.cfg.ctc_output_loss * scale
            }
        }
        
        return loss_dict, num_phonemes