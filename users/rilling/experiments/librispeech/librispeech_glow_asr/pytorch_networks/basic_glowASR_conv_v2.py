"""
Trying to make the aligner more AppTek-Like

Extended weight init code
"""

from dataclasses import dataclass
import torch
import numpy
from torch import nn
import multiprocessing
from librosa import filters
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union
import math

from torchaudio.functional import mask_along_axis

# from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
# from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
# from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
# from i6_models.parts.conformer.norm import LayerNormNC
# from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
# from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerEncoderV1
# from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config
# from i6_models.config import ModuleFactoryV1, ModelConfiguration
# from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
# from i6_models.parts.frontend.common import mask_pool

from returnn.torch.context import get_run_ctx

from .shared.configs import DbMelFeatureExtractionConfig
from .shared.feature_extraction import DbMelFeatureExtraction

from . import modules
from . import commons
from . import attentions
from .monotonic_align import maximum_path


def apply_spec_aug(input, num_repeat_time, max_dim_time, num_repeat_feat, max_dim_feat):
    """
    :param Tensor input: the input audio features (B,T,F)
    :param int num_repeat_time: number of repetitions to apply time mask
    :param int max_dim_time: number of columns to be masked on time dimension will be uniformly sampled from [0, mask_param]
    :param int num_repeat_feat: number of repetitions to apply feature mask
    :param int max_dim_feat: number of columns to be masked on feature dimension will be uniformly sampled from [0, mask_param]
    """
    for _ in range(num_repeat_time):
        input = mask_along_axis(input, mask_param=max_dim_time, mask_value=0.0, axis=1)

    for _ in range(num_repeat_feat):
        input = mask_along_axis(input, mask_param=max_dim_feat, mask_value=0.0, axis=2)
    return input


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    r = torch.arange(tensor.shape[2], device=get_run_ctx().device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


class FinalLinear(nn.Module):
    def __init__(self, in_size, hidden_size, target_size, n_layers=1, p_dropout=0.01):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()

        self.layers.append(modules.LinearBlock(self.in_size, self.hidden_size, self.p_dropout))

        for i in range(1, n_layers - 1):
            self.layers.append(modules.LinearBlock(self.hidden_size, self.hidden_size, self.p_dropout))

        self.layers.append(modules.LinearBlock(self.hidden_size, self.target_size, self.p_dropout))

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n_layers):
            x = self.layers[i](x)

        return x


class FinalConvolutional(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, target_channels, kernel_size=3, padding=1, n_layers=1, p_dropout=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.target_channels = target_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()

        self.layers.append(
            modules.Conv1DBlock(self.in_channels, self.hidden_channels, self.kernel_size, p_dropout=p_dropout, norm="batch")
        )

        for i in range(1, n_layers - 1):
            self.layers.append(
                modules.Conv1DBlock(self.hidden_channels, self.hidden_channels, self.kernel_size, p_dropout=p_dropout, norm="batch")
            )

        self.layers.append(
            modules.Conv1DBlock(self.in_channels, self.hidden_channels, self.kernel_size, p_dropout=p_dropout, norm="batch")
        )

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)

        return x


class Flow(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout=0.0,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
    ):
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

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        for b in range(n_blocks):
            self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
            self.flows.append(
                attentions.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = commons.channel_squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = commons.channel_unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()


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
        p_dropout_dec: float = 0.05,
        gin_channels: int = 0,
        n_split: int = 4,
        n_sqz: int = 2,
        sigmoid_scale: bool = False,
        window_size: int = 4,
        block_length: int = None,
        hidden_channels_dec: int = None,
        final_hidden_channels=80,
        final_n_layers=1,
        label_target_size=None,
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
        self.p_dropout_dec = p_dropout_dec
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.window_size = window_size
        self.block_length = block_length
        self.hidden_channels_dec = hidden_channels_dec

        self.net_kwargs = {
            "repeat_per_num_frames": 100,
            "max_dim_feat": 8,
            "num_repeat_feat": 5,
            "max_dim_time": 20,
        }

        fe_config = DbMelFeatureExtractionConfig.from_dict(kwargs["fe_config"])
        self.feature_extraction = DbMelFeatureExtraction(config=fe_config)

        if label_target_size is None:
            run_ctx = get_run_ctx()
            dataset = run_ctx.engine.train_dataset or run_ctx.engine.forward_dataset
            self.label_target_size = len(dataset.datasets["zip_dataset"].targets.labels)
        else:
            self.label_target_size = label_target_size

        self.decoder = Flow(
            out_channels,
            hidden_channels_dec or hidden_channels,
            kernel_size_dec,
            dilation_rate,
            n_blocks_dec,
            n_block_layers,
            p_dropout=p_dropout_dec,
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=gin_channels,
        )

        self.final = nn.Sequential(
            FinalConvolutional(out_channels, final_hidden_channels, final_hidden_channels, n_layers=final_n_layers),
            FinalLinear(
                final_hidden_channels, final_hidden_channels, self.label_target_size + 1, n_layers=final_n_layers
            ),
        )

    def forward(self, raw_audio, raw_audio_len):
        with torch.no_grad():
            squeezed_audio = torch.squeeze(raw_audio)
            log_mel_features, log_mel_features_len = self.feature_extraction(squeezed_audio, raw_audio_len)  # [B, T, F]
            log_mel_features = log_mel_features.transpose(1, 2)  # [B, F, T]

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= 10:
                audio_features_masked_2 = apply_spec_aug(
                    log_mel_features,
                    num_repeat_time=torch.max(log_mel_features_len).detach().cpu().numpy()
                    // self.net_kwargs["repeat_per_num_frames"],
                    max_dim_time=self.net_kwargs["max_dim_time"],
                    num_repeat_feat=self.net_kwargs["num_repeat_feat"],
                    max_dim_feat=self.net_kwargs["max_dim_feat"],
                )
            else:
                audio_features_masked_2 = log_mel_features

        flow_in = audio_features_masked_2
        audio_max_length = log_mel_features.size(2)

        flow_in, flow_in_length, flow_in_max_length = self.preprocess(flow_in, log_mel_features_len, audio_max_length)

        # mask = mask_tensor(flow_in, log_mel_features_len)
        mask = torch.unsqueeze(commons.sequence_mask(log_mel_features_len, flow_in.size(2)), 1).to(flow_in.dtype)

        with torch.no_grad():
            flow_out, _ = self.decoder(flow_in, mask, reverse=False)

        # flow_out = flow_out.transpose(1, 2)
        # mask = mask.transpose(1, 2)

        flow_out = flow_out * mask
        logits = self.final(flow_out)

        log_probs = torch.log_softmax(logits, dim=2)
        return log_probs, flow_in_length

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    phon_labels = data["phon_labels"]  # [B, N] (sparse)
    phon_labels_len = data["phon_labels:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (2, 0, 1))  # CTC needs [T, B, F] and from glow is [B, F, T] coming
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        phon_labels,
        input_lengths=audio_features_len,
        target_lengths=phon_labels_len,
        blank=model.label_target_size,
        reduction="sum",
    )
    num_phonemes = torch.sum(phon_labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    from torchaudio.models.decoder import ctc_decoder

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")
    labels = run_ctx.engine.forward_dataset.datasets["zip_dataset"].targets.labels
    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=None,
        tokens=labels + ["[blank]", "[SILENCE]"],
        blank_token="[blank]",
        sil_token="[SILENCE]",
        unk_word="[UNKNOWN]",
        nbest=1,
        beam_size=1,
    )
    run_ctx.labels = labels


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()


def search_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    tags = data["seq_tag"]

    hypothesis = run_ctx.ctc_decoder(logprobs.cpu(), audio_features_len.cpu())
    for hyp, tag in zip(hypothesis, tags):
        tokens = hyp[0].tokens.numpy()
        sequence = " ".join([run_ctx.labels[i] for i in tokens[1:-2]])
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
