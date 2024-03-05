"""
Trying to make the aligner more AppTek-Like

Extended weight init code
"""

from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
import multiprocessing
from librosa import filters
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union
import math

from torchaudio.functional import mask_along_axis

from i6_models.parts.blstm import BlstmEncoderV1, BlstmEncoderV1Config


from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerEncoderV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config

from ..i6modelsV1_VGG4LayerActFrontendV1_v4_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig


from returnn.torch.context import get_run_ctx

from .shared.configs import DbMelFeatureExtractionConfig
from .shared.feature_extraction import DbMelFeatureExtraction
from .shared.spec_augment import apply_spec_aug
from .shared.mask import mask_tensor

from .shared import modules
from .shared import commons
from .shared import attentions
from .monotonic_align import maximum_path

from .shared.forward import forward_init_hook, forward_step, forward_finish_hook, prior_init_hook, prior_finish_hook, prior_step
from .shared.train import train_step

from IPython import embed

class Model(nn.Module):
    """
    Flow-based ASR model based on GlowTTS Structure using a pre-trained flow-based decoder
    trained to generate spectrograms from given statistics coming from an encoder

    Model was pretrained using the architecture in
    users/rilling/experiments/librispeech/librispeech_glowtts/pytorch_networks/glowTTS.py
    """

    def __init__(
        self,
        n_vocab: int,
        hidden_channels: int = 192,
        out_channels: int = 80,
        n_blocks_dec: int = 12,
        kernel_size_dec: int = 5,
        dilation_rate: int = 1,
        n_block_layers: int = 4,
        p_dropout: float = 0.1,
        p_dropout_flow: float = 0.05,
        gin_channels: int = 0,
        n_split: int = 4,
        n_sqz: int = 2,
        sigmoid_scale: bool = False,
        window_size: int = 4,
        block_length: int = None,
        hidden_channels_dec: int = None,
        label_target_size: int = None,
        spec_augment: bool = False,
        layer_norm: bool = False,
        batch_norm: bool = False,
        conformer_model_config: ModelConfig = None,
        **kwargs,
    ):
        """_summary_

        Args:
            n_vocab (int): vocabulary size
            hidden_channels (int): Number of hidden channels in encoder
            out_channels (int): Number of channels in the output
            n_blocks_dec (int, optional): Number of coupling blocks in the decoder. Defaults to 12.
            kernel_size_dec (int, optional): Kernel size in the decoder. Defaults to 5.
            dilation_rate (int, optional): Dilation rate for CNNs of coupling blocks in decoder. Defaults to 5.
            n_block_layers (int, optional): Number of layers in the CNN of the coupling blocks in decoder. Defaults to 4.
            p_dropout_dec (_type_, optional): Dropout probability in the decoder. Defaults to 0..
            n_speakers (int, optional): Number of speakers. Defaults to 0.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            window_size (int, optional): Window size  in Multi-Head Self-Attention for encoder. Defaults to None.
            block_length (_type_, optional): Block length for optional block masking in Multi-Head Attention for encoder. Defaults to None.
            hidden_channels_dec (_type_, optional): Number of hidden channels in decodder. Defaults to hidden_channels.
            final_hidden_channels: Number of hidden channels in the final network
            final_n_layers: Number of layers in the final network
            label_target_size: Target size of target vocabulary, target size for final network
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_blocks_dec = n_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.n_block_layers = n_block_layers
        self.p_dropout = p_dropout
        self.p_dropout_flow = p_dropout_flow
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.window_size = window_size
        self.block_length = block_length
        self.hidden_channels_dec = hidden_channels_dec
        self.spec_augment = spec_augment
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.net_kwargs = {
            "repeat_per_num_frames": 100,
            "max_dim_feat": 8,
            "num_repeat_feat": 5,
            "max_dim_time": 20,
        }

        fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

        if label_target_size is None:
            if n_vocab is None:
                run_ctx = get_run_ctx()
                dataset = run_ctx.engine.train_dataset or run_ctx.engine.forward_dataset
                self.label_target_size = len(dataset.datasets["zip_dataset"].targets.labels)
            else:
                self.label_target_size = n_vocab
        else:
            self.label_target_size = label_target_size

        self.decoder = modules.Flow(
            out_channels,
            hidden_channels_dec or hidden_channels,
            kernel_size_dec,
            dilation_rate,
            n_blocks_dec,
            n_block_layers,
            p_dropout=p_dropout_flow,
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=gin_channels,
        )
        if conformer_model_config is None:
            specaug_config = SpecaugConfig(
                repeat_per_n_frames=25,
                max_dim_time=20,
                max_dim_feat=16,
                num_repeat_feat=5,
            )
            frontend_config = VGG4LayerActFrontendV1Config(
                in_features=80,
                conv1_channels=32,
                conv2_channels=64,
                conv3_channels=64,
                conv4_channels=32,
                conv_kernel_size=(3, 3),
                conv_padding=None,
                pool1_kernel_size=(2, 1),
                pool1_stride=(2, 1),
                pool1_padding=None,
                pool2_kernel_size=(2, 1),
                pool2_stride=(2, 1),
                pool2_padding=None,
                out_features=384,
                activation=nn.ReLU(),
            )
            conformer_model_config = ModelConfig(
                frontend_config=frontend_config,
                specaug_config=specaug_config,
                label_target_size=self.n_vocab,
                conformer_size=384,
                num_layers=12,
                num_heads=4,
                ff_dim=1536,
                att_weights_dropout=0.2,
                conv_dropout=0.2,
                ff_dropout=0.2,
                mhsa_dropout=0.2,
                conv_kernel_size=31,
                final_dropout=0.2,
                specauc_start_epoch=1
            )
    
        self.cfg = conformer_model_config
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
                    channels=conformer_size, kernel_size=self.cfg.conv_kernel_size, dropout=self.cfg.conv_dropout, activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size)
                ),
            ),
        )

        if self.layer_norm:
            print("Using Layer Norm after Flow...")

        if self.batch_norm:
            print("Using Batch Norm after Flow...")
            self.bn = nn.BatchNorm1d(self.out_channels)
        else:
            self.bn = None

        self.conformer = ConformerEncoderV1(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch


    def forward(self, raw_audio, raw_audio_len):
        with torch.no_grad():
            squeezed_audio = torch.squeeze(raw_audio)
            log_mel_features, log_mel_features_len = self.feature_extraction(squeezed_audio, raw_audio_len)  # [B, T, F]

            audio_max_length = log_mel_features.size(1)
            spec_augment_in = log_mel_features
            if self.training and self.spec_augment:
                audio_features_masked_2 = apply_spec_aug(
                    spec_augment_in,
                    num_repeat_time=torch.max(log_mel_features_len).detach().cpu().numpy()
                    // self.net_kwargs["repeat_per_num_frames"],
                    max_dim_time=self.net_kwargs["max_dim_time"],
                    num_repeat_feat=self.net_kwargs["num_repeat_feat"],
                    max_dim_feat=self.net_kwargs["max_dim_feat"],
                )
            else:
                audio_features_masked_2 = spec_augment_in

            flow_in = audio_features_masked_2.transpose(1,2) # [B, F, T]
            flow_in, flow_in_length, flow_in_max_length = self.preprocess(flow_in, log_mel_features_len, audio_max_length)
            mask = torch.unsqueeze(commons.sequence_mask(log_mel_features_len, flow_in.size(2)), 1).to(flow_in.dtype)
            flow_out, _ = self.decoder(flow_in, mask, reverse=False) # [B, F, T]
            conformer_in = flow_out.transpose(1,2)
            mask = mask_tensor(conformer_in, flow_in_length)
        
        if self.layer_norm:
            conformer_in = torch.nn.functional.layer_norm(conformer_in, (conformer_in.size(-1),))
        elif self.bn is not None:
            conformer_in = self.bn(conformer_in.transpose(1,2)).transpose(1,2)
        
        conformer_out, out_mask = self.conformer(conformer_in, mask)
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)

        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()

