"""
Like v2, but with i6_models specaugment (v3)
and now controllable start time for when specaugment is applied (v4)
and with the proper feature extraction from i6-models
"""

import contextlib
import numpy as np
import torch
from torch import nn
from typing import Tuple

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosEncoderV1Config
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosBlockV1Config, ConformerRelPosEncoderV1
from i6_models.config import ModuleFactoryV1, ModelConfiguration

from i6_models.parts.conformer.convolution import ConformerConvolutionV2Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
from i6_models.parts.dropout import BroadcastDropout
from i6_models.parts.frontend.common import mask_pool, calculate_output_dim, get_same_padding
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from returnn.torch.context import get_run_ctx

from .i6models_relposV1_VGGNLayerActFrontendV1_feat_v2_cfg import (
    ModelConfig,
    LinearConfig,
    SpecaugConfig,
    SpecaugStftConfig,
    SpecaugStftV2Config,
    SpecaugStftV3Config,
    SpecaugStftV4Config,
    SpecaugMultiplierLinearConfig,
    VGGNLayerActFrontendV1Config,
    VGGNLayerActFrontendV2Config,
)
from .i6models_relposV1_VGG4LayerActFrontendV1_v1 import (
    mask_tensor, train_step, prior_init_hook, prior_finish_hook, prior_step
)
from ..features.scf import (
    SupervisedConvolutionalFeatureExtractionV1,
    SupervisedConvolutionalFeatureExtractionV2,
)
from ..features.stft import StftFeatureExtractionV1, StftFeatureExtractionV2
from ..features.conv import ConvFeatureExtractionV1, ConvFeatureExtractionV2
from ..features.wav2vec import Wav2vecFeatureExtractionV1


class Identity(nn.Module):
    def __init__(self, model_cfg: ModelConfiguration):
        super().__init__()
        self.cfg = model_cfg

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor, sequence_mask


class Linear(nn.Module):
    def __init__(self, model_cfg: LinearConfig):
        super().__init__()
        self.linear = nn.Linear(
            in_features=model_cfg.in_features,
            out_features=model_cfg.out_features,
            bias=True,
        )

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.linear(tensor), sequence_mask


class Log1p(nn.Module):
    def __init__(self):
        super(Log1p, self).__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.log1p(tensor)


class VGGNLayerActFrontendV1(nn.Module):
    """
    Convolutional Front-End

    The frond-end utilizes convolutional and pooling layers, as well as activation functions
    to transform a feature vector, typically Log-Mel or Gammatone for audio, into an intermediate
    representation.

    Structure of the front-end:
      - Conv
      - Conv
      - Activation
      - Pool
      - Conv
      - Conv
      - Activation
      - Pool

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: VGGNLayerActFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        self.cfg = model_cfg

        self.layers = nn.ModuleList()
        in_channels = 1
        for conv, activation_str, pooling in zip(self.cfg.convs, self.cfg.activations, self.cfg.poolings):
            conv_dim, conv_kernel_size, conv_stride = conv
            if activation_str in [None, ""]:
                activation = nn.Identity()
            elif activation_str == "ReLU":
                activation = nn.ReLU()
            elif activation_str.startswith("ReLU_Dropout"):
                dropout = float(activation_str[len("ReLU_Dropout"):])
                activation = nn.Sequential(nn.ReLU(), nn.Dropout(p=dropout))
            elif activation_str.startswith("ReLU_Log1p"):
                activation = nn.Sequential(nn.ReLU(), Log1p())
            else:
                assert False, f"Unsupported activation {activation_str}"
            self.layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_dim,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=get_same_padding(conv_kernel_size),
                ),
                activation,
                nn.MaxPool2d(
                    kernel_size=pooling[0],
                    stride=pooling[1],
                    padding=pooling[2] if pooling[2] is not None else (0, 0),
                ) if pooling is not None else nn.Identity(),
            ))
            in_channels = conv_dim
        self.linear = nn.Linear(
            in_features=self._calculate_dim(),
            out_features=model_cfg.out_features,
            bias=True,
        )

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' depending on stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: the sequence mask for the tensor
        :return: torch.Tensor of shape [B,T',F'] and the shape of the sequence mask
        """
        assert tensor.shape[-1] == self.cfg.in_features, f"shape {tensor.shape} vs in features {self.cfg.in_features}"
        # and add a dim
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        for layer in self.layers:
            tensor = layer(tensor)

            # mask for conv and pooling if not None
            conv, _, pool = layer
            sequence_mask = mask_pool(
                seq_mask=sequence_mask,
                kernel_size=conv.kernel_size[0],
                stride=conv.stride[0],
                padding=conv.padding[0],
            )
            if not isinstance(pool, nn.Identity):
                sequence_mask = mask_pool(
                    seq_mask=sequence_mask,
                    kernel_size=pool.kernel_size[0],
                    stride=pool.stride[0],
                    padding=pool.padding[0],
                )

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T',C,F']
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T',C*F']

        tensor = self.linear(tensor)

        return tensor, sequence_mask

    def _calculate_dim(self) -> int:
        out_dim = self.cfg.in_features
        for layer in self.layers:
            conv, _, pool = layer
            out_dim = calculate_output_dim(
                in_dim=out_dim,
                filter_size=conv.kernel_size[1],
                stride=conv.stride[1],
                padding=conv.padding[1],
            )
            if not isinstance(pool, nn.Identity):
                out_dim = calculate_output_dim(
                    in_dim=out_dim,
                    filter_size=pool.kernel_size[1],
                    stride=pool.stride[1],
                    padding=pool.padding[1],
                )
        out_dim *= self.layers[-1][0].out_channels
        return out_dim


class VGGNLayerActFrontendV2(nn.Module):
    def __init__(self, model_cfg: VGGNLayerActFrontendV2Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        self.cfg = model_cfg

        self.layers = nn.ModuleList()
        in_channels = model_cfg.in_channels
        for conv, activation_str, pooling in zip(self.cfg.convs, self.cfg.activations, self.cfg.poolings):
            conv_dim, conv_kernel_size, conv_stride = conv
            if activation_str in [None, ""]:
                activation = nn.Identity()
            elif activation_str == "ReLU":
                activation = nn.ReLU()
            elif activation_str.startswith("ReLU_Dropout"):
                dropout = float(activation_str[len("ReLU_Dropout"):])
                activation = nn.Sequential(nn.ReLU(), nn.Dropout(p=dropout))
            elif activation_str.startswith("ReLU_Log1p"):
                activation = nn.Sequential(nn.ReLU(), Log1p)
            else:
                assert False, f"Unsupported activation {activation_str}"
            self.layers.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_dim,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=get_same_padding(conv_kernel_size),
                ),
                activation,
                nn.MaxPool2d(
                    kernel_size=pooling[0],
                    stride=pooling[1],
                    padding=pooling[2] if pooling[2] is not None else (0, 0),
                ) if pooling is not None else nn.Identity(),
            ))
            in_channels = conv_dim
        if model_cfg.project_out:
            self.linear = nn.Linear(
                in_features=self._calculate_dim(),
                out_features=model_cfg.out_features,
                bias=True,
            )

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' depending on stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: the sequence mask for the tensor
        :return: torch.Tensor of shape [B,T',F'] and shape of the sequence mask
        """
        tensor = tensor.unflatten(-1, (self.cfg.in_channels, self.cfg.in_features)).transpose(1, 2)  # [B,C,T,F]
        assert tensor.shape[-1] == self.cfg.in_features, f"shape {tensor.shape} vs in features {self.cfg.in_features}"
        assert tensor.shape[1] == self.cfg.in_channels, f"shape {tensor.shape} vs in channels {self.cfg.in_channels}"

        for layer in self.layers:
            tensor = layer(tensor)

            # mask for conv and pooling if not None
            conv, _, pool = layer
            sequence_mask = mask_pool(
                seq_mask=sequence_mask,
                kernel_size=conv.kernel_size[0],
                stride=conv.stride[0],
                padding=conv.padding[0],
            )
            if not isinstance(pool, nn.Identity):
                sequence_mask = mask_pool(
                    seq_mask=sequence_mask,
                    kernel_size=pool.kernel_size[0],
                    stride=pool.stride[0],
                    padding=pool.padding[0],
                )

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T',C,F']
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T',C*F']

        if self.cfg.project_out:
            tensor = self.linear(tensor)

        return tensor, sequence_mask

    def _calculate_dim(self) -> int:
        out_dim = self.cfg.in_features
        for layer in self.layers:
            conv, _, pool = layer
            out_dim = calculate_output_dim(
                in_dim=out_dim,
                filter_size=conv.kernel_size[1],
                stride=conv.stride[1],
                padding=conv.padding[1],
            )
            if not isinstance(pool, nn.Identity):
                out_dim = calculate_output_dim(
                    in_dim=out_dim,
                    filter_size=pool.kernel_size[1],
                    stride=pool.stride[1],
                    padding=pool.padding[1],
                )
        out_dim *= self.layers[-1][0].out_channels
        return out_dim


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        frontend_config_class = globals()[self.cfg.frontend_config_class[:-len("Config")]]
        conformer_size = self.cfg.conformer_size
        conformer_config = ConformerRelPosEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=frontend_config_class, cfg=frontend_config),
            block_cfg=ConformerRelPosBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                    input_dim=conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
                ),
                mhsa_cfg=ConformerMHSARelPosV1Config(
                    input_dim=conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    with_bias=self.cfg.mhsa_with_bias,
                    dropout=self.cfg.mhsa_dropout,
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
                    learnable_pos_emb=self.cfg.pos_emb_config.learnable_pos_emb,
                    rel_pos_clip=self.cfg.pos_emb_config.rel_pos_clip,
                    with_linear_pos=self.cfg.pos_emb_config.with_linear_pos,
                    with_pos_bias=self.cfg.pos_emb_config.with_pos_bias,
                    separate_pos_emb_per_head=self.cfg.pos_emb_config.separate_pos_emb_per_head,
                    pos_emb_dropout=self.cfg.pos_emb_config.pos_emb_dropout,
                ),
                conv_cfg=ConformerConvolutionV2Config(
                    channels=conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size),
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
                ),
                modules=self.cfg.module_list,
                scales=self.cfg.module_scales,
            ),
        )

        feature_extractor = ModuleFactoryV1(
            module_class=globals()[self.cfg.feature_extraction_config.module_class],
            cfg=self.cfg.feature_extraction_config,
        )
        self.feature_extraction = feature_extractor()
        self.conformer = ConformerRelPosEncoderV1(cfg=conformer_config)
        self.num_output_linears = 1 if self.cfg.aux_ctc_loss_layers is None else len(self.cfg.aux_ctc_loss_layers)
        self.output_linears = nn.ModuleList([
            nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
            for _ in range(self.num_output_linears)
        ])
        self.output_dropout = BroadcastDropout(p=self.cfg.final_dropout, dropout_broadcast_axes=self.cfg.dropout_broadcast_axes)
        
        self.return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]
        self.scales = self.cfg.aux_ctc_loss_scales or [1.0]
        
        self.specaug_start_epoch = self.cfg.specaug_start_epoch

        # No particular weight init!

    def forward(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels + blank]
        """
        run_ctx = get_run_ctx()
        
        audio_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad() if not self.training else contextlib.ExitStack():
            if isinstance(self.cfg.specaug_config, SpecaugStftConfig):
                if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                    if isinstance(self.cfg.specaug_config, SpecaugStftV2Config):
                        audio_features_masked = torch.stft(
                            audio_features,
                            self.cfg.specaug_config.fft_size,
                            self.cfg.specaug_config.window_shift,
                            self.cfg.specaug_config.window_size,
                            return_complex=True,
                        )
                        multiplier = 1.
                        if isinstance(self.cfg.specaug_config.multiplier, SpecaugMultiplierLinearConfig):
                            if run_ctx.epoch <= self.cfg.specaug_config.multiplier.start_epoch:
                                multiplier = self.cfg.specaug_config.multiplier.start_factor
                            elif run_ctx.epoch >= self.cfg.specaug_config.multiplier.end_epoch:
                                multiplier = self.cfg.specaug_config.multiplier.end_factor
                            else:
                                progress = (run_ctx.epoch - self.cfg.specaug_config.multiplier.start_epoch) / ((
                                    self.cfg.specaug_config.multiplier.end_epoch -
                                    self.cfg.specaug_config.multiplier.start_epoch)
                                )
                                multiplier = self.cfg.specaug_config.multiplier.start_factor + progress * ((
                                    self.cfg.specaug_config.multiplier.end_factor -
                                    self.cfg.specaug_config.multiplier.start_factor)
                                )
                        audio_features_masked = specaugment_v1_by_length(
                            audio_features_masked,
                            time_min_num_masks=self.cfg.specaug_config.min_num_time,
                            time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                            time_mask_max_size=int(self.cfg.specaug_config.max_dim_time * multiplier),
                            freq_min_num_masks=self.cfg.specaug_config.min_num_feat,
                            freq_mask_max_size=int(self.cfg.specaug_config.max_dim_feat * multiplier),
                            freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                        )
                        audio_features_masked = torch.istft(
                            audio_features_masked,
                            self.cfg.specaug_config.fft_size,
                            self.cfg.specaug_config.window_shift,
                            self.cfg.specaug_config.window_size,
                        )
                    elif isinstance(self.cfg.specaug_config, SpecaugStftV3Config):
                        audio_features_masked = torch.stft(
                            audio_features,
                            self.cfg.specaug_config.fft_size,
                            self.cfg.specaug_config.window_shift,
                            self.cfg.specaug_config.window_size,
                            return_complex=True,
                        )
                        if not hasattr(self, "_mel_fbank"):
                            import torchaudio
                            self._mel_fbank = torchaudio.functional.melscale_fbanks(
                                self.cfg.specaug_config.fft_size // 2 + 1,
                                0,
                                8000,
                                self.cfg.specaug_config.num_mels,
                                16000,
                            ).to(audio_features_masked)
                        audio_features_masked = torch.matmul(
                            audio_features_masked.transpose(1, 2), self._mel_fbank
                        ).transpose(1, 2)
                        audio_features_masked = specaugment_v1_by_length(
                            audio_features_masked,
                            time_min_num_masks=2,
                            time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                            time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                            freq_min_num_masks=2,
                            freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                            freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                        )
                        audio_features_masked = torch.matmul(
                            audio_features_masked.transpose(1, 2), torch.linalg.pinv(self._mel_fbank)
                        ).transpose(1, 2)
                        audio_features_masked = torch.istft(
                            audio_features_masked,
                            self.cfg.specaug_config.fft_size,
                            self.cfg.specaug_config.window_shift,
                            self.cfg.specaug_config.window_size,
                        )
                    elif isinstance(self.cfg.specaug_config, SpecaugStftV4Config):
                        device = audio_features.device

                        # signal to STFT-domain
                        if self.cfg.specaug_config.window == "hann":
                            window = torch.hann_window(self.cfg.specaug_config.window_size).to(device)
                        else:
                            raise NotImplementedError
                        audio_features_masked = torch.stft(
                            audio_features,
                            self.cfg.specaug_config.fft_size,
                            self.cfg.specaug_config.window_shift,
                            self.cfg.specaug_config.window_size,
                            window=window,
                            return_complex=True,
                            onesided=True,
                        )

                        # sample mask in Mel-domain
                        if not hasattr(self, "_mel_fbank"):
                            import torchaudio
                            self._mel_fbank = torchaudio.functional.melscale_fbanks(
                                self.cfg.specaug_config.fft_size // 2 + 1,
                                0,
                                8000,
                                self.cfg.specaug_config.num_mels,
                                16000,
                            ).to(device)
                            # fill up triangle boundaries of the first and last channel to ensure the sum over all
                            # triangles is always 1
                            self._mel_fbank[:torch.argmax(self._mel_fbank[:, 0]), 0] = 1.0
                            self._mel_fbank[torch.argmax(self._mel_fbank[:, -1]) + 1:, -1] = 1.0

                        mel_mask_shape = list(audio_features_masked.shape)
                        mel_mask_shape[-2] = self.cfg.specaug_config.num_mels
                        mel_mask = specaugment_v1_by_length(
                            torch.ones(mel_mask_shape).to(device),
                            time_min_num_masks=2,
                            time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                            time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                            freq_min_num_masks=2,
                            freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                            freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                        )

                        # convert mask to STFT-domain
                        stft_mask = torch.einsum("...mt,fm->...ft", mel_mask, self._mel_fbank) >= 1.0

                        # apply mask: set masked regions to mean magnitude with random phase
                        phase = torch.rand(audio_features_masked.shape).to(device) * 2 * torch.pi
                        j = torch.complex(torch.tensor(0.0), torch.tensor(1.0)).to(device)
                        audio_features_masked = torch.where(
                            stft_mask,
                            audio_features_masked,
                            audio_features_masked.abs().mean() * torch.exp(j * phase),
                        )

                        audio_features_masked = torch.istft(
                            audio_features_masked,
                            self.cfg.specaug_config.fft_size,
                            self.cfg.specaug_config.window_shift,
                            self.cfg.specaug_config.window_size,
                            window=window,
                            onesided=True,
                        )
                    else:
                        audio_features_masked = torch.stft(
                            audio_features,
                            self.cfg.specaug_config.fft_size,
                            self.cfg.specaug_config.window_shift,
                            self.cfg.specaug_config.window_size,
                            return_complex=True,
                        )
                        audio_features_masked = specaugment_v1_by_length(
                            audio_features_masked,
                            time_min_num_masks=2,  # TODO: make configurable
                            time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                            time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                            freq_min_num_masks=2,
                            freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                            freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                        )
                        audio_features_masked = torch.istft(
                            audio_features_masked,
                            self.cfg.specaug_config.fft_size,
                            self.cfg.specaug_config.window_shift,
                            self.cfg.specaug_config.window_size,
                        )
                    # fill with original audio to original length
                    audio_features = torch.concat(
                        [audio_features_masked, audio_features[:, audio_features_masked.shape[1]:]],
                        axis=1,
                    )

                audio_features, audio_features_len = self.feature_extraction(audio_features, raw_audio_len)
            else:
                assert isinstance(self.cfg.specaug_config, SpecaugConfig)

                audio_features, audio_features_len = self.feature_extraction(audio_features, raw_audio_len)

                if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                    audio_features = specaugment_v1_by_length(
                        audio_features,
                        time_min_num_masks=2,  # TODO: make configurable
                        time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                        time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                        freq_min_num_masks=2,
                        freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                        freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                    )

        conformer_in = audio_features
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        conformer_out_layers, out_mask = self.conformer(conformer_in, mask, return_layers=self.return_layers)
        log_probs_list = []
        for i, (out_layer, scale) in enumerate(zip(conformer_out_layers, self.scales)):
            if scale == 0.0:
                continue
            conformer_out = self.output_dropout(out_layer)
            logits = self.output_linears[i](conformer_out)
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs_list.append(log_probs)

        return log_probs_list, torch.sum(out_mask, dim=1)
