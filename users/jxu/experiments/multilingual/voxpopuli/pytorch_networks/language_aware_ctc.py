"""
Like v2, but with i6_models specaugment (v3)
and now controllable start time for when specaugment is applied (v4)
and with the proper feature extraction from i6-models
"""

import numpy as np
from librosa import filters
import torch
from torch import nn
from typing import Tuple, List
from dataclasses import dataclass

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v2 import ConformerEncoderV2Config
from i6_models.assemblies.conformer.conformer_v2 import ConformerBlockV2Config, ConformerEncoderV2
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV2Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config

# from returnn.torch.context import get_run_ctx
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
from returnn.tensor import batch_dim

from .i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
    SpecaugConfig,
    VGG4LayerActFrontendV1Config_mod,
    LogMelFeatureExtractionV1Config,
)


@dataclass
class ModelConfig:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
    specauc_start_epoch: int
    label_target_size: int
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    conv_kernel_size: int
    final_dropout: float
    module_list: List[str]
    module_scales: List[float]
    lid_sc_layer: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        return ModelConfig(**d)


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


class LogMelFeatureExtractionV1OnnxExportable(nn.Module):
    """
    Librosa-compatible log-mel feature extraction using log10. Does not use torchaudio.

    Using it wrapped with torch.no_grad() is recommended if no gradient is needed
    """

    def __init__(self, cfg: LogMelFeatureExtractionV1Config):
        super().__init__()
        self.center = cfg.center
        self.hop_length = int(cfg.hop_size * cfg.sample_rate)
        self.min_amp = cfg.min_amp
        self.n_fft = cfg.n_fft
        self.win_length = int(cfg.win_size * cfg.sample_rate)

        self.register_buffer(
            "mel_basis",
            torch.tensor(
                filters.mel(
                    sr=cfg.sample_rate,
                    n_fft=cfg.n_fft,
                    n_mels=cfg.num_filters,
                    fmin=cfg.f_min,
                    fmax=cfg.f_max,
                )
            ),
        )
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return features as [B,T,F] and length in frames [B]
        """

        power_spectrum = (
            torch.sum(
                torch.stft(
                    raw_audio.float(),
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    center=self.center,
                    pad_mode="constant",
                    return_complex=False,
                ) ** 2,
                dim=-1,
            )
        )
        if len(power_spectrum.size()) == 2:
            # For some reason torch.stft removes the batch axis for batch sizes of 1, so we need to add it again
            power_spectrum = torch.unsqueeze(power_spectrum, 0)
        melspec = torch.einsum("...ft,mf->...mt", power_spectrum, self.mel_basis)
        log_melspec = torch.log10(torch.clamp(melspec, min=self.min_amp))
        feature_data = torch.transpose(log_melspec, 1, 2)
        if self.center:
            length = (length // self.hop_length) + 1
        else:
            length = ((length - self.n_fft) // self.hop_length) + 1

        return feature_data, length.int()


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size

        # self.run_ctx = kwargs.get("run_ctx", None)
        conformer_config = ConformerEncoderV2Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV2Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV2Config(
                    input_dim=conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                    dropout_broadcast_axes=None,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size)
                ),
                modules=self.cfg.module_list,
                scales=self.cfg.module_scales,
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1OnnxExportable(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderV2(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
        self.lid_final_linear = nn.Linear(conformer_size, 17)  # + CTC blank
        self.lid_sc_linear = nn.Linear(17, conformer_size)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch
        self.lid_sc_layer = self.cfg.lid_sc_layer

    def forward(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: torch.Tensor,
            global_train_step: int
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]#
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels + blank]
        """

        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            if self.training and global_train_step >= self.specaug_start_epoch * 1000:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,  # TODO: make configurable
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        # conformer_out, out_mask = self.conformer(conformer_in, mask)

        return_layers = [len(self.conformer.module_list) - 1]

        if isinstance(self.conformer.frontend, nn.Identity):
            x = conformer_in
            out_mask = mask
        else:
            x, out_mask = self.conformer.frontend(conformer_in, mask)  # [B, T, F']

        outputs = []

        lid_out_probs = None

        for i in range(max(return_layers) + 1):
            x = self.conformer.module_list[i](x, out_mask)  # [B, T, F']
            if i + 1 == self.lid_sc_layer:
                lid_out = x.clone()
                lid_out_probs = self.lid_final_linear(lid_out)
                lid_out_probs = torch.log_softmax(lid_out_probs, dim=2)
                sc_lid_features = self.lid_sc_linear(torch.exp(lid_out_probs))
                x = x + sc_lid_features.detach()
                print("x", x)
                print("sc_lid_features", sc_lid_features)
            if i in return_layers:
                outputs.append(x)

        conformer_out = self.final_dropout(outputs[0])
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, lid_out_probs, torch.sum(out_mask, dim=1)


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, global_train_step, **kwargs):
    raw_audio = extern_data["data"].raw_tensor
    raw_audio = raw_audio.squeeze(-1)
    raw_audio_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    labels = extern_data["targets"].raw_tensor.long()
    labels_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor
    labels_len_rf = extern_data["targets"].dims[1].dyn_size_ext

    language = extern_data["language"].raw_tensor.long()
    language_len = extern_data["language"].dims[1].dyn_size_ext.raw_tensor
    language_len_rf = extern_data["language"].dims[1].dyn_size_ext

    logprobs, lid_out_probs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        global_train_step=global_train_step
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        labels,
        input_lengths=audio_features_len,
        target_lengths=labels_len,
        blank=model.cfg.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    rf.get_run_ctx().mark_as_loss(name="ctc", loss=ctc_loss,
                                  custom_inv_norm_factor=rf.reduce_sum(labels_len_rf, axis=batch_dim))

    if lid_out_probs is not None:
        transposed_lid_out_probs = torch.permute(lid_out_probs, (1, 0, 2))  # CTC needs [T, B, F]
        lid_ctc_loss = nn.functional.ctc_loss(
            transposed_lid_out_probs,
            language,
            input_lengths=audio_features_len,
            target_lengths=language_len,
            blank=16,
            reduction="sum",
            zero_infinity=True,
        )
        rf.get_run_ctx().mark_as_loss(name="lid_ctc_loss", loss=lid_ctc_loss,
                                      custom_inv_norm_factor=rf.reduce_sum(language_len_rf, axis=batch_dim), scale=0.3)


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
    with open("prior.txt", 'w') as f:
        np.savetxt(f, log_average_probs, delimiter=' ')
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["data"]  # [B, T', F]
    raw_audio_len = data["data:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

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


def get_model_config(vocab_size_without_blank: int, network_args: dict) -> ModelConfig:
    specauc_start_epoch = 1 if "specauc_start_epoch" not in network_args else network_args["specauc_start_epoch"]
    lid_sc_layer = network_args["lid_sc_layer"]

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=True,
    )

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(1, 2),
        pool1_stride=None,
        pool1_padding=None,
        pool2_kernel_size=(1, 2),
        pool2_stride=(4, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )
    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=specauc_start_epoch,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        lid_sc_layer=lid_sc_layer
    )

    return model_config
