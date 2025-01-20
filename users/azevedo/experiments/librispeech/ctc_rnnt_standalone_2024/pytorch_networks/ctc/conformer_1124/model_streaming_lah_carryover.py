import numpy as np
import torch
from torch import nn
from typing import Callable, List, Optional
from enum import Enum
from dataclasses import dataclass

from i6_models.primitives.specaugment import specaugment_v1_by_length

from returnn.torch.context import get_run_ctx

from .model_lah_carryover_cfg import ModelConfig

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerBlockV1Config

# frontend imports
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from ...rnnt.streaming_conformer.generic_frontend_v2 import GenericFrontendV2

# feature extract and conformer module imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config

from ...rnnt.conformer_0924.conf_lah_carryover_v2 import ConformerEncoderCOV2

from ...rnnt.auxil.functional import num_samples_to_frames, mask_tensor, Mode, TrainingStrategy



class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        # net_args are passed as a dict to returnn and here the config is retransformed into its dataclass
        self.cfg = ModelConfig.from_dict(model_config_dict)

        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size),
                ),
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderCOV2(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        self.mode: Mode = None
        self.lookahead_size = self.cfg.lookahead_size
        self.carry_over_size = self.cfg.carry_over_size

    def extract_features(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        mask = mask_tensor(conformer_in, audio_features_len)

        return conformer_in, mask

    def prep_streaming_input(self, conformer_in, mask):
        batch_size = conformer_in.size(0)

        chunk_size_frames = num_samples_to_frames(
            n_fft=self.feature_extraction.n_fft, 
            hop_length=self.feature_extraction.hop_length,
            center=self.feature_extraction.center,
            num_samples=int(self.cfg.chunk_size)
        )

        # pad conformer time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -conformer_in.size(1) % chunk_size_frames
        # (B, T, ...) -> (B, T+time_dim_pad, ...) = (B, T', ...)
        conformer_in = torch.nn.functional.pad(conformer_in, (0, 0, 0, time_dim_pad),
                                               "constant", 0)
        mask = torch.nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

        # separate chunks to signal the conformer that we are chunking input
        conformer_in = conformer_in.view(batch_size, -1, chunk_size_frames,
                                         conformer_in.size(-1))  # (B, (T'/C), C, F) = (B, N, C, F)
        mask = mask.view(batch_size, -1, chunk_size_frames)  # (B, N, C)

        return conformer_in, mask

    def merge_chunks(self, conformer_out, out_mask):
        batch_size = conformer_out.size(0)

        conformer_out = conformer_out.view(batch_size, -1, conformer_out.size(-1))  # (B, C'*N, F')
        out_mask = out_mask.view(batch_size, -1)  # (B, C'*N)

        return conformer_out, out_mask
    
    def set(self, streaming: bool) -> None:
        self.mode = Mode.STREAMING if streaming else Mode.OFFLINE

    def forward(
        self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, N]
        :param labels_len: length of N as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        assert self.mode is not None

        conformer_in, mask = self.extract_features(raw_audio, raw_audio_len)

        if self.mode == Mode.STREAMING:
            conformer_in, mask = self.prep_streaming_input(conformer_in, mask)

        conformer_out, out_mask = self.conformer(conformer_in, mask, 
                                                 lookahead_size=self.lookahead_size,
                                                 carry_over_size=self.carry_over_size)

        if self.mode == Mode.STREAMING:
            conformer_out, out_mask = self.merge_chunks(conformer_out, out_mask)

        # final linear layer
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    num_phonemes = torch.sum(labels_len)

    def _train_step_mode(encoder_mode: Mode, scale: float):
        model.mode = encoder_mode

        logprobs, audio_features_len = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
        )

        mode_str = encoder_mode.name.lower()[:3]
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
        
        return {
            "ctc.%s" % mode_str: {
                "loss": ctc_loss,
                "scale": scale
            }
        }

    def _train_step_unified():
        str_loss = _train_step_mode(Mode.STREAMING, scale=model.cfg.online_model_scale)
        off_loss = _train_step_mode(Mode.OFFLINE, scale=1-model.cfg.online_model_scale)
        return {**str_loss, **off_loss}

    def _train_step_switching():
        encoder_mode = Mode.STREAMING if run_ctx.global_step % 2 == 0 else Mode.OFFLINE
        switch_loss = _train_step_mode(encoder_mode, scale=1)
        return switch_loss

    if model.cfg.training_strategy == TrainingStrategy.UNIFIED:
        loss_dict = _train_step_unified()
    elif model.cfg.training_strategy == TrainingStrategy.SWITCHING:
        loss_dict = _train_step_switching()
    else:
        loss_dict = _train_step_mode(Mode.STREAMING, scale=1)

    for loss_key in loss_dict:
        run_ctx.mark_as_loss(
            name=loss_key, 
            loss=loss_dict[loss_key]["loss"], 
            inv_norm_factor=num_phonemes,
            scale=loss_dict[loss_key]["scale"]
        )


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


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    model.mode = Mode.STREAMING if run_ctx.global_step % 2 == 0 else Mode.OFFLINE

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))