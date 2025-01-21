import torch
from torch import nn
from typing import Callable, List, Optional

from .unified_lookahead_transducer import (
    Model as UnifiedLAHTransducer,
    Mode
)

from returnn.torch.context import get_run_ctx



class Model(UnifiedLAHTransducer):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__(model_config_dict, **kwargs)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    from i6_native_ops import warp_rnnt

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    prepended_targets = labels.new_empty([labels.size(0), labels.size(1) + 1])
    prepended_targets[:, 1:] = labels
    prepended_targets[:, 0] = model.cfg.label_target_size  # blank is last index
    prepended_target_lengths = labels_len + 1

    #
    #
    #
    encoder_mode = Mode.STREAMING if run_ctx.global_step % 2 == 0 else Mode.OFFLINE

    model.mode = encoder_mode
    logits, audio_features_len, ctc_logprobs = model(
        raw_audio=raw_audio, raw_audio_len=raw_audio_len,
        labels=prepended_targets, labels_len=prepended_target_lengths,
    )

    logprobs = torch.log_softmax(logits, dim=-1)
    fastemit_lambda = model.cfg.fastemit_lambda

    rnnt_loss = warp_rnnt.rnnt_loss(
        log_probs=logprobs,
        frames_lengths=audio_features_len.to(dtype=torch.int32),
        labels=labels,
        labels_lengths=labels_len.to(dtype=torch.int32),
        blank=model.cfg.label_target_size,
        fastemit_lambda=fastemit_lambda if fastemit_lambda is not None else 0.0,
        reduction="sum",
        gather=True,
    )

    num_phonemes = torch.sum(labels_len)

    scale = model.cfg.online_model_scale if encoder_mode == Mode.STREAMING else (1 - model.cfg.online_model_scale)
    mode_str = encoder_mode.name.lower()[:3]

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
        run_ctx.mark_as_loss(name="ctc.%s" % mode_str, loss=ctc_loss, inv_norm_factor=num_phonemes,
                                scale=model.cfg.ctc_output_loss * scale)

    run_ctx.mark_as_loss(name="rnnt.%s" % mode_str, loss=rnnt_loss, inv_norm_factor=num_phonemes,
                            scale=scale)
