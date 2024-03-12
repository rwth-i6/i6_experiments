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
import os
import soundfile

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

from returnn.torch.context import get_run_ctx

from .shared.i6modelsV1_VGG4LayerActFrontendV1_v4_cfg import (
    SpecaugConfig,
    VGG4LayerActFrontendV1Config_mod,
    ModelConfig,
    FlowDecoderConfig,
    TextEncoderConfig,
)

from .shared.configs import DbMelFeatureExtractionConfig
from .shared.feature_extraction import DbMelFeatureExtraction
from .shared.spec_augment import apply_spec_aug
from .shared.mask import mask_tensor

from .shared import modules
from .shared import commons
from .shared import attentions
from .monotonic_align import maximum_path

from .shared.forward import search_init_hook, search_finish_hook

from IPython import embed


class XVector(nn.Module):
    def __init__(self, input_dim=40, num_classes=8, **kwargs):
        super(XVector, self).__init__()
        self.tdnn1 = modules.TDNN(
            input_dim=input_dim, output_dim=512, context_size=5, dilation=1, dropout_p=0.5, batch_norm=True
        )
        self.tdnn2 = modules.TDNN(
            input_dim=512, output_dim=512, context_size=3, dilation=2, dropout_p=0.5, batch_norm=True
        )
        self.tdnn3 = modules.TDNN(
            input_dim=512, output_dim=512, context_size=2, dilation=3, dropout_p=0.5, batch_norm=True
        )
        self.tdnn4 = modules.TDNN(
            input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5, batch_norm=True
        )
        self.tdnn5 = modules.TDNN(
            input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5, batch_norm=True
        )
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        # self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

    def forward(self, x, x_lengths):
        # with torch.no_grad():
        #     squeezed_audio = torch.squeeze(raw_audio)
        #     x, x_lengths = self.feature_extraction(squeezed_audio, raw_audio_lengths)  # [B, T, F]

        # x = x.transpose(1, 2)
        tdnn1_out = self.tdnn1(x)
        # return tdnn1_out
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out, 2)
        std = torch.std(tdnn5_out, 2)
        stat_pooling = torch.cat((mean, std), 1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        output = self.output(x_vec)
        predictions = self.softmax(output)
        return output, predictions, x_vec


class DurationPredictor(nn.Module):
    """
    Duration Predictor module, trained using calculated durations coming from monotonic alignment search
    """

    def __init__(self, in_channels, filter_channels, filter_size, p_dropout):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.filter_size = filter_size
        self.p_dropout = p_dropout

        self.convs = nn.Sequential(
            modules.Conv1DBlock(
                in_size=self.in_channels,
                out_size=self.filter_channels,
                filter_size=self.filter_size,
                p_dropout=p_dropout,
            ),
            modules.Conv1DBlock(
                in_size=self.filter_channels,
                out_size=self.filter_channels,
                filter_size=self.filter_size,
                p_dropout=p_dropout,
            ),
        )
        self.proj = nn.Conv1d(in_channels=self.filter_channels, out_channels=1, kernel_size=1)

    def forward(self, x, x_mask):
        x_with_mask = (x, x_mask)
        (x, x_mask) = self.convs(x_with_mask)
        x = self.proj(x * x_mask)
        return x


class FlowDecoder(nn.Module):
    def __init__(self, cfg: FlowDecoderConfig, in_channels, gin_channels):
        """Flow-based decoder model

        Args:
            in_channels (int): Number of incoming channels
            hidden_channels (int): Number of hidden channels
            kernel_size (int): Kernel Size for convolutions in coupling blocks
            dilation_rate (float): Dilation Rate to define dilation in convolutions of coupling block
            n_blocks (int): Number of coupling blocks
            n_layers (int): Number of layers in CNN of the coupling blocks
            p_dropout (float, optional): Dropout probability for CNN in coupling blocks. Defaults to 0..
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
        """
        super().__init__()
        self.cfg = cfg

        self.flows = nn.ModuleList()

        for _ in range(self.cfg.n_blocks):
            self.flows.append(modules.ActNorm(channels=in_channels * self.cfg.n_sqz))
            self.flows.append(modules.InvConvNear(channels=in_channels * self.cfg.n_sqz, n_split=self.cfg.n_split))
            self.flows.append(
                attentions.CouplingBlock(
                    in_channels * self.cfg.n_sqz,
                    self.cfg.hidden_channels,
                    kernel_size=self.cfg.kernel_size,
                    dilation_rate=self.cfg.dilation_rate,
                    n_layers=self.cfg.n_layers,
                    gin_channels=gin_channels,
                    p_dropout=self.cfg.p_dropout,
                    sigmoid_scale=self.cfg.sigmoid_scale,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if g is not None:
            g = g.unsqueeze(-1)

        if self.cfg.n_sqz > 1:
            x, x_mask = commons.channel_squeeze(x, x_mask, self.cfg.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.cfg.n_sqz > 1:
            x, x_mask = commons.channel_unsqueeze(x, x_mask, self.cfg.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

class TextEncoder(nn.Module):
    """
    Text Encoder model
    """

    def __init__(self, cfg: TextEncoderConfig, out_channels, gin_channels):
        """Text Encoder Model based on Multi-Head Self-Attention combined with FF-CCNs

        Args:
            n_vocab (int): Size of vocabulary for embeddings
            out_channels (int): Number of output channels
            hidden_channels (int): Number of hidden channels
            filter_channels (int): Number of filter channels
            filter_channels_dp (int): Number of filter channels for duration predictor
            n_heads (int): Number of heads in encoder's Multi-Head Attention
            n_layers (int): Number of layers consisting of Multi-Head Attention and CNNs in encoder
            kernel_size (int): Kernel Size for CNNs in encoder layers
            p_dropout (float): Dropout probability for both encoder and duration predictor
            window_size (int, optional): Window size  in Multi-Head Self-Attention for encoder. Defaults to None.
            block_length (_type_, optional): Block length for optional block masking in Multi-Head Attention for encoder. Defaults to None.
            mean_only (bool, optional): Boolean to only project text encodings to mean values instead of mean and std. Defaults to False.
            prenet (bool, optional): Boolean to add ConvReluNorm prenet before encoder . Defaults to False.
            gin_channels (int, optional): Number of channels for speaker condition. Defaults to 0.
        """
        super().__init__()
        self.cfg = cfg

        self.emb = nn.Embedding(self.cfg.n_vocab, self.cfg.hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, self.cfg.hidden_channels**-0.5)

        if self.cfg.prenet:
            self.pre = modules.ConvReluNorm(
                self.cfg.hidden_channels,
                self.cfg.hidden_channels,
                self.cfg.hidden_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        self.encoder = attentions.Encoder(
            self.cfg.hidden_channels,
            self.cfg.filter_channels,
            self.cfg.n_heads,
            self.cfg.n_layers,
            self.cfg.kernel_size,
            self.cfg.p_dropout,
            window_size=self.cfg.window_size,
            block_length=self.cfg.block_length,
        )

        self.proj_m = nn.Conv1d(self.cfg.hidden_channels, out_channels, 1)
        if not self.cfg.mean_only:
            self.proj_s = nn.Conv1d(self.cfg.hidden_channels, out_channels, 1)
        self.proj_w = DurationPredictor(
            self.cfg.hidden_channels + gin_channels,
            self.cfg.filter_channels_dp,
            self.cfg.kernel_size,
            self.cfg.p_dropout,
        )

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.cfg.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.cfg.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)

        if g is not None:
            g_exp = g.unsqueeze(-1).expand(-1, -1, x.size(-1))
            # print(f"Dimension of input in Text Encoder: x.shape: {x.shape}; g: {g.shape}, g_exp: {g_exp.shape}")
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        x_m = self.proj_m(x) * x_mask
        if not self.cfg.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        # print(f"Dimension of input in Text Encoder before DP: {x_dp.shape}")

        logw = self.proj_w(x_dp, x_mask)
        return x_m, x_logs, logw, x_mask

class Model(nn.Module):
    """
    Flow-based ASR model based on GlowTTS Structure using a pre-trained flow-based decoder
    trained to generate spectrograms from given statistics coming from an encoder

    Model was pretrained using the architecture in
    users/rilling/experiments/librispeech/librispeech_glowtts/pytorch_networks/glowTTS.py
    """

    def __init__(
        self,
        model_config: dict,
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

        self.net_kwargs = {
            "repeat_per_num_frames": 100,
            "max_dim_feat": 8,
            "num_repeat_feat": 5,
            "max_dim_time": 20,
        }

        fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

        # if label_target_size is None:
        #     if n_vocab is None:
        #         run_ctx = get_run_ctx()
        #         dataset = run_ctx.engine.train_dataset or run_ctx.engine.forward_dataset
        #         self.label_target_size = len(dataset.datasets["zip_dataset"].targets.labels)
        #     else:
        #         self.label_target_size = n_vocab
        # else:
        #     self.label_target_size = label_target_size

        self.cfg = ModelConfig.from_dict(model_config)
        frontend_config = self.cfg.frontend_config
        text_encoder_config = self.cfg.text_encoder_config
        decoder_config = self.cfg.decoder_config

        if self.cfg.n_speakers > 1:
            self.x_vector = XVector(self.cfg.out_channels, self.cfg.n_speakers)

        self.encoder = TextEncoder(
            text_encoder_config, out_channels=self.cfg.out_channels, gin_channels=self.cfg.gin_channels
        )

        self.decoder = FlowDecoder(
            decoder_config, in_channels=self.cfg.out_channels, gin_channels=self.cfg.gin_channels
        )

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

        # if self.layer_norm:
        #     print("Using Layer Norm after Flow...")

        # if self.batch_norm:
        #     print("Using Batch Norm after Flow...")
        #     self.bn = nn.BatchNorm1d(self.out_channels)
        # else:
        #     self.bn = None

        self.conformer = ConformerEncoderV1(cfg=conformer_config)
        self.conformer_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.asr_output = nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank

        self.specaug_start_epoch = self.cfg.specauc_start_epoch

    def forward(
        self, x=None, x_lengths=None, raw_audio=None, raw_audio_lengths=None, g=None, gen=False, recognition=False, noise_scale=1.0, length_scale=1.0
    ):
        if not gen:
            with torch.no_grad():
                squeezed_audio = torch.squeeze(raw_audio)
                y, y_lengths = self.feature_extraction(squeezed_audio, raw_audio_lengths)  # [B, T, F]
                y = y.transpose(1, 2)  # [B, F, T]
                _, _, g = self.x_vector(y, y_lengths)
        else:
            y, y_lengths = (None, None)

        if not recognition:
            x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)  # mean, std logs, duration logs, mask

        if gen:  # durations from dp only used during generation
            w = torch.exp(logw) * x_mask * length_scale  # durations
            w_ceil = torch.ceil(w)  # durations ceiled
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
        else:
            y_max_length = y.size(2)

        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(torch.int32)

        if not recognition:
            attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        if gen:
            attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2)
            z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2)
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
            y, logdet = self.decoder(z, z_mask, g=g, reverse=True)

            return (y, z_m, z_logs, logdet, z_mask, y_lengths), (x_m, x_logs, x_mask), (attn, logw, logw_)
        else:
            z, logdet = self.decoder(y, z_mask, g=g, reverse=False)

            # spec_augment_in = z.transpose(1, 2)  # [B, T, F]

            # if self.training and self.cfg.specaug_config is not None:
            #     audio_features_masked_2 = apply_spec_aug(
            #         spec_augment_in,
            #         num_repeat_time=torch.max(y_lengths).detach().cpu().numpy()
            #         // self.cfg.specaug_config.repeat_per_n_frames,
            #         max_dim_time=self.cfg.specaug_config.max_dim_time,
            #         num_repeat_feat=self.cfg.specaug_config.num_repeat_feat,
            #         max_dim_feat=self.cfg.specaug_config.max_dim_feat,
            #     )
            # else:
            #     audio_features_masked_2 = spec_augment_in

            if recognition:
                conformer_in = z.transpose(1,2)
                mask = mask_tensor(conformer_in, y_lengths)

                conformer_out, out_mask = self.conformer(conformer_in, mask)
                conformer_out = self.conformer_dropout(conformer_out)
                logits = self.asr_output(conformer_out)
                log_probs = torch.log_softmax(logits, dim=2)

                return log_probs, torch.sum(out_mask, dim=1)
            else:
                with torch.no_grad():
                    x_s_sq_r = torch.exp(-2 * x_logs)
                    logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)  # [b, t, 1]
                    logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z**2))  # [b, t, d] x [b, d, t'] = [b, t, t']
                    logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
                    logp4 = torch.sum(-0.5 * (x_m**2) * x_s_sq_r, [1]).unsqueeze(-1)  # [b, t, 1]
                    logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

                    attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
                    # embed()

                z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(
                    1, 2
                )  # [b, t', t], [b, t, d] -> [b, d, t']
                z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(
                    1, 2
                )  # [b, t', t], [b, t, d] -> [b, d, t']

                logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

                # Sample from encoder ouput for ASR during training to only affect updates in the encoder and leave flow principle untouched.
                conformer_in = ((z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask).transpose(1,2)
                mask = mask_tensor(conformer_in, y_lengths)

                conformer_out, out_mask = self.conformer(conformer_in, mask)
                conformer_out = self.conformer_dropout(conformer_out)
                logits = self.asr_output(conformer_out)
                log_probs = torch.log_softmax(logits, dim=2)

                return (
                    (z, z_m, z_logs, logdet, z_mask),
                    (x_m, x_logs, x_mask),
                    y_lengths,
                    (attn, logw, logw_),
                    (log_probs, torch.sum(out_mask, dim=1)),
                )

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.cfg.decoder_config.n_sqz) * self.cfg.decoder_config.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.cfg.decoder_config.n_sqz) * self.cfg.decoder_config.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()


def train_step(*, model: Model, data, run_ctx, **kwargs):
    tags = data["seq_tag"]
    audio_features = data["audio_features"]  # [B, T, F]
    # audio_features = audio_features.transpose(1, 2) # [B, F, T] necessary because glowTTS expects the channels to be in the 2nd dimension
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    phonemes_eow = data["phonemes_eow"][indices, :]  # [B, T]
    phonemes_eow_len = data["phonemes_eow:size1"][indices]
    # speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)
    tags = list(np.array(tags)[indices.detach().cpu().numpy()])

    (
        (z, z_m, z_logs, logdet, z_mask),
        (x_m, x_logs, x_mask),
        y_lengths,
        (attn, logw, logw_),
        (logprobs, ctc_input_length),
    ) = model(phonemes, phonemes_len, audio_features, audio_features_len)
    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    l_dp = commons.duration_loss(logw, logw_, phonemes_len)

    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))
    l_ctc = nn.functional.ctc_loss(
        transposed_logprobs,
        phonemes_eow,
        input_lengths=ctc_input_length,
        target_lengths=phonemes_eow_len,
        blank=model.cfg.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )

    run_ctx.mark_as_loss(name="mle", loss=l_mle)
    run_ctx.mark_as_loss(name="dp", loss=l_dp)

    num_phonemes = torch.sum(phonemes_eow_len)
    if "ctc_scale" in kwargs:
        ctc_scale = kwargs["ctc_scale"]
    else:
        ctc_scale = 1
    run_ctx.mark_as_loss(name="ctc", loss=l_ctc, inv_norm_factor=num_phonemes, scale=ctc_scale)


def forward_init_hook(run_ctx, **kwargs):
    import json
    import utils
    from utils import AttrDict
    from inference import load_checkpoint
    from generator import UnivNet as Generator
    import numpy as np

    with open("/u/lukas.rilling/experiments/glow_tts_asr_v2/config_univ.json") as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(run_ctx.device)

    state_dict_g = load_checkpoint(
        "/work/asr3/rossenbach/rilling/vocoder/univnet/glow_finetuning/g_01080000", run_ctx.device
    )
    generator.load_state_dict(state_dict_g["generator"])

    run_ctx.generator = generator
    run_ctx.speaker_x_vectors = torch.load(
        "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/forward/ReturnnForwardJob.U6UwGhE7ENbp/output/output_pooled.hdf"
    )


def forward_finish_hook(run_ctx, **kwargs):
    pass


MAX_WAV_VALUE = 32768.0


def forward_step(*, model: Model, data, run_ctx, **kwargs):
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)
    audio_features = data["audio_features"]

    tags = data["seq_tag"]

    speaker_x_vector = run_ctx.speaker_x_vectors[speaker_labels.detach().cpu().numpy(), :].squeeze(1)

    (log_mels, z_m, z_logs, logdet, z_mask, y_lengths), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(
        phonemes,
        phonemes_len,
        g=speaker_x_vector.to(run_ctx.device),
        gen=True,
        noise_scale=kwargs["noise_scale"],
        length_scale=kwargs["length_scale"],
    )

    noise = torch.randn([1, 64, log_mels.shape[-1]]).to(device=log_mels.device)
    audios = run_ctx.generator.forward(noise, log_mels)
    audios = audios * MAX_WAV_VALUE
    audios = audios.cpu().numpy().astype("int16")

    if not os.path.exists("/var/tmp/lukas.rilling/"):
        os.makedirs("/var/tmp/lukas.rilling/")
    if not os.path.exists("/var/tmp/lukas.rilling/out"):
        os.makedirs("/var/tmp/lukas.rilling/out/", exist_ok=True)
    for audio, tag in zip(audios, tags):
        soundfile.write(f"/var/tmp/lukas.rilling/out/" + tag.replace("/", "_") + ".wav", audio[0], 16000)

def search_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    from torchaudio.models.decoder import ctc_decoder
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")
    import subprocess
    if kwargs["arpa_lm"] is not None:
        lm = subprocess.check_output(["cf", kwargs["arpa_lm"]]).decode().strip()
    else:
        lm = None
    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=kwargs["returnn_vocab"], unknown_label=None)
    labels = vocab.labels

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=kwargs["lexicon"],
        lm=lm,
        lm_weight=kwargs["lm_weight"],
        tokens=labels + ["[blank]", "[SILENCE]", "[UNK]"],
        # "[SILENCE]" and "[UNK]" are not actually part of the vocab,
        # but the decoder is happy as long they are defined in the token list
        # even if they do not exist as label index in the softmax output,
        blank_token="[blank]",
        sil_token="[SILENCE]",
        unk_word="[unknown]",
        nbest=1,
        beam_size=kwargs["beam_size"],
        beam_size_token=kwargs.get("beam_size_token", None),
        beam_threshold=kwargs["beam_threshold"],
        sil_score=kwargs.get("sil_score", 0.0),
        word_score=kwargs.get("word_score", 0.0),
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = kwargs.get("blank_log_penalty", None)

    if kwargs.get("prior_file", None):
        run_ctx.prior = np.loadtxt(kwargs["prior_file"], dtype="float32")
        run_ctx.prior_scale = kwargs["prior_scale"]
    else:
        run_ctx.prior = None

def search_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

def search_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_lengths=raw_audio_len,
        recognition=True
    )

    tags = data["seq_tag"]

    logprobs_cpu = logprobs.cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
