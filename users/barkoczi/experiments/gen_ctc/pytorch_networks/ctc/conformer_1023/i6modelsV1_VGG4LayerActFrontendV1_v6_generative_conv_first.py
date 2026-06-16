import numpy as np
import torch
from torch import nn
from typing import Tuple

from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.config import ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerConvolutionV1Config,
    ConformerMHSAV1,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.primitives.specaugment import specaugment_v1_by_length
from returnn.torch.context import get_run_ctx

from i6_experiments.users.barkoczi.experiments.gen_ctc.loss.fixed_ctc_loss import torch_ctc_fixed_grad
from i6_experiments.users.barkoczi.experiments.gen_ctc.loss.generative_loss import generative_nce

from .i6modelsV1_VGG4LayerActFrontendV1_v6_generative_cfg import ModelConfig


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    seq_len = seq_len.to(device=tensor.device)
    positions = torch.arange(tensor.shape[1], device=tensor.device)
    return positions[None, :] < seq_len[:, None]


class ConformerBlockV1ConvFirst(nn.Module):
    def __init__(self, cfg: ConformerBlockV1Config):
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        x = 0.5 * self.ff1(x) + x
        x = self.conv(x) + x
        x = self.mhsa(x, sequence_mask) + x
        x = 0.5 * self.ff2(x) + x
        return self.final_layer_norm(x)


class ConformerEncoderV1ConvFirst(nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__()
        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList(
            [ConformerBlockV1ConvFirst(cfg.block_cfg) for _ in range(cfg.num_layers)]
        )

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)
        for module in self.module_list:
            x = module(x, sequence_mask)
        return x, sequence_mask


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        conformer_size = self.cfg.conformer_size
        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=self.cfg.frontend_config),
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
        self.conformer = ConformerEncoderV1ConvFirst(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.cfg.label_target_size + 1)
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)
            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )

        mask = mask_tensor(audio_features, audio_features_len)
        conformer_out, out_mask = self.conformer(audio_features, mask)
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)
        log_probs = torch.nn.functional.logsigmoid(logits)
        return log_probs, torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to("cpu")
    labels = data["labels"]
    labels_len = data["labels:size1"]

    with torch.enable_grad():
        log_probs, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
        log_probs_for_ctc = log_probs.float()
        ctc_loss = torch_ctc_fixed_grad(
            torch.permute(log_probs_for_ctc, (1, 0, 2)),
            labels,
            input_lengths=audio_features_len.cpu(),
            target_lengths=labels_len.cpu(),
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )
        soft_target = -torch.autograd.grad(ctc_loss, log_probs_for_ctc, retain_graph=True)[0].detach()
        loss = generative_nce(
            log_probs,
            soft_target.to(log_probs.dtype),
            sampling_type=model.cfg.sampling_type,
            seq_len=audio_features_len,
            sampling_ratio=model.cfg.sampling_ratio,
            share_samples=model.cfg.share_samples,
            ratio_corrector=model.cfg.ratio_corrector,
        )
        inv_norm_factor = torch.clamp(audio_features_len.sum().to(dtype=torch.float32), min=1.0)
        run_ctx.mark_as_loss(name="ctc_generative_nce", loss=loss.sum(), inv_norm_factor=inv_norm_factor)


def prior_init_hook(run_ctx, **kwargs):
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space:", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"]
    log_probs, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    probs = torch.exp(log_probs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))
