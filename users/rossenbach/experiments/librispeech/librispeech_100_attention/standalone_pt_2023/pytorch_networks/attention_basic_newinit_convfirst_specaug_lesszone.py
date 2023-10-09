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

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config, ConformerConvolutionV1
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config, ConformerPositionwiseFeedForwardV1
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config, ConformerMHSAV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.parts.frontend.common import mask_pool

from returnn.torch.context import get_run_ctx

##################
##### Attention imports
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.assemblies.conformer import (
    ConformerEncoderV1,
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
)
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import (
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.decoder.attention import (
    AttentionLSTMDecoderV1,
    AttentionLSTMDecoderV1Config,
    AdditiveAttentionConfig,
)
from i6_models.primitives.feature_extraction import (
    LogMelFeatureExtractionV1,
    LogMelFeatureExtractionV1Config,
)


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
    r = torch.arange(tensor.shape[1], device=get_run_ctx().device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


@dataclass
class TwoLayer1DFrontendConfig(ModelConfiguration):
    """
    Attributes:
        in_features: number of input features to module
        conv1_channels: number of channels for first conv layer
        conv2_channels: number of channels for second conv layer
    """

    in_features: int
    conv1_channels: int
    conv2_channels: int
    conv1_kernel_size: int
    conv1_stride: int
    conv2_kernel_size: int
    conv2_stride: int


    def check_valid(self):
        pass

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()

class TwoLayer1DFrontend(nn.Module):
    """
    Convolutional Front-End
    """

    def __init__(self, model_cfg: TwoLayer1DFrontendConfig):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        self.cfg = model_cfg

        self.conv1 = nn.Conv1d(
            in_channels=model_cfg.in_features,
            out_channels=model_cfg.conv1_channels,
            kernel_size=model_cfg.conv1_kernel_size,
            stride=model_cfg.conv1_stride
        )
        self.conv2 = nn.Conv1d(
            in_channels=model_cfg.conv1_channels,
            out_channels=model_cfg.conv2_channels,
            kernel_size=model_cfg.conv2_kernel_size,
            stride=model_cfg.conv2_stride
        )

        self.bn1 = nn.BatchNorm1d(num_features=model_cfg.conv1_channels)
        self.bn2 = nn.BatchNorm1d(num_features=model_cfg.conv1_channels)


    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' or T'' depending on stride of the layers

        stride is only allowed for the pool1 and pool2 operation.
        other ops do not have stride configurable -> no update of mask sequence required but added anyway

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: the sequence mask for the tensor
        :return: torch.Tensor of shape [B,T",F'] and the shape of the sequence mask
        """
        tensor = tensor.permute(0, 2, 1)  # [B,T,F] -> [B,C,T]

        tensor = self.conv1(tensor)
        tensor = self.bn1(tensor)
        sequence_mask = mask_pool(
            seq_mask=sequence_mask,
            kernel_size=self.conv1.kernel_size[0],
            stride=self.conv1.stride[0],
            padding=self.conv1.padding[0],
        )

        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        sequence_mask = mask_pool(
            sequence_mask,
            kernel_size=self.conv2.kernel_size[0],
            stride=self.conv2.stride[0],
            padding=self.conv2.padding[0],
        )

        tensor = tensor.permute(0, 2, 1) # [B,C,T] -> [B, T, hidden]

        return tensor, sequence_mask

    def _calculate_dim(self) -> int:
        return self.conv2.out_channels


class ConformerBlockV1ConvFirst(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = 0.5 * self.ff1(x) + x  #  [B, T, F]
        x = self.conv(x) + x  #  [B, T, F]
        x = self.mhsa(x, sequence_mask) + x  #  [B, T, F]
        x = 0.5 * self.ff2(x) + x  #  [B, T, F]
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


@dataclass
class ConformerEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockV1Config


class ConformerEncoderV1ConvFirstNewInit(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication,
    but with convolution first and a different init.

    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV1ConvFirst(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        return x, sequence_mask

class ConformerAEDModelConfig:
    def __init__(
        self,
        feat_extraction_cfg: LogMelFeatureExtractionV1Config,
        encoder_cfg: ConformerEncoderV1Config,
        decoder_cfg: AttentionLSTMDecoderV1Config,
    ):
        self.feat_extraction_cfg = feat_extraction_cfg
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg


class ConformerAEDModel(nn.Module):
    """
    Conformer Encoder-Decoder Attention model for ASR
    """

    def __init__(self, cfg: ConformerAEDModelConfig):
        super().__init__()

        self.feat_extraction = LogMelFeatureExtractionV1(cfg=cfg.feat_extraction_cfg)
        # TODO: add specaugment
        self.encoder = ConformerEncoderV1(cfg=cfg.encoder_cfg)
        self.vocab_size = cfg.decoder_cfg.vocab_size
        self.ctc_linear = nn.Linear(in_features=256, out_features=self.vocab_size + 1)  # +1 for CTC
        self.decoder = AttentionLSTMDecoderV1(cfg=cfg.decoder_cfg)

        self.apply(self._weight_init)
        
    @staticmethod
    def _weight_init(module: torch.nn.Module):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            print("apply xavier weight init for %s" % str(module))
            nn.init.xavier_uniform_(module.weight)


    def forward(
        self,
        raw_audio_features: torch.Tensor,
        raw_audio_features_lens: torch.Tensor,
        bpe_labels: torch.Tensor,
    ):
        """
        :param raw_audio_features: audio raw samples of shape [B,T,F=1]
        :param raw_audio_features_lens: audio sequence length of shape [B]
        :param bpe_labels: bpe targets of shape [B,N]
        :return:
        """

        audio_features = raw_audio_features.squeeze(-1)  # [B,T]
        with torch.no_grad():
            audio_features, audio_features_lens = self.feat_extraction(
                audio_features, raw_audio_features_lens
            )  # [B,T,F]
            if self.training:
                audio_features = apply_spec_aug(
                    input=audio_features,
                    num_repeat_time=audio_features.size()[1] // 100,
                    max_dim_time=20,
                    num_repeat_feat=5,
                    max_dim_feat=8
                )

        time_arange = torch.arange(audio_features.size(1), device=audio_features.device)  # [0, ..., T-1]
        time_mask = torch.less(time_arange[None, :], audio_features_lens[:, None])  # [B,T]

        encoder_outputs, encoder_seq_mask = self.encoder(audio_features, time_mask)
        decoder_logits, state = self.decoder(
            encoder_outputs, bpe_labels, audio_features_lens
        )

        encoder_seq_len = torch.sum(encoder_seq_mask, dim=1)  # [B]
        ctc_logits = self.ctc_linear(encoder_outputs)  # [B, T, #vocab + 1]
        ctc_logprobs = nn.functional.log_softmax(ctc_logits, dim=2)

        return decoder_logits, state, encoder_outputs, encoder_seq_len, ctc_logprobs


class Model(torch.nn.Module):
    def __init__(self, **net_kwargs):
        super().__init__()
        feat_extraction_cfg = LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.05,
            hop_size=0.0125,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=True,
        )
        frontend_config = TwoLayer1DFrontendConfig(
            in_features=80,
            conv1_channels=256,
            conv1_kernel_size=5,
            conv1_stride=2,
            conv2_channels=256,
            conv2_stride=2,
            conv2_kernel_size=5,
        )
        conformer_encoder_cfg = ConformerEncoderV1Config(
            num_layers=8,
            frontend=ModuleFactoryV1(module_class=TwoLayer1DFrontend, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=256, hidden_dim=1024, dropout=0.1, activation=nn.SiLU()
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=256, num_att_heads=4, att_weights_dropout=0.1, dropout=0.1
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=256,
                    kernel_size=31,
                    dropout=0.1,
                    activation=nn.SiLU(),
                    norm=nn.BatchNorm1d(num_features=256, affine=False),
                ),
            ),
        )
        decoder_attention_cfg = AdditiveAttentionConfig(
            attention_dim=256, att_weights_dropout=0.1
        )
        run_ctx = get_run_ctx()
        dataset = run_ctx.engine.train_dataset or run_ctx.engine.forward_dataset
        self.num_labels = len(dataset.datasets["zip_dataset"].targets.labels)
        decoder_cfg = AttentionLSTMDecoderV1Config(
            encoder_dim=256,
            vocab_size=self.num_labels,
            target_embed_dim=128,
            target_embed_dropout=0.1,
            lstm_hidden_size=256,
            zoneout_drop_h=0.05,
            zoneout_drop_c=0.05,  # default 0.15
            attention_cfg=decoder_attention_cfg,
            output_proj_dim=512,
            output_dropout=0.2,  # default 0.3
        )

        model_cfg = ConformerAEDModelConfig(
            feat_extraction_cfg=feat_extraction_cfg,
            encoder_cfg=conformer_encoder_cfg,
            decoder_cfg=decoder_cfg,
        )
        self.aed_model = ConformerAEDModel(cfg=model_cfg)

    def forward(
        self,
        raw_audio_features: torch.Tensor,
        raw_audio_features_lens: torch.Tensor,
        bpe_labels: torch.Tensor,
    ):
        decoder_logits, state, encoder_outputs, encoder_seq_len, ctc_logprobs =  self.aed_model(
            raw_audio_features=raw_audio_features,
            raw_audio_features_lens=raw_audio_features_lens,
            bpe_labels=bpe_labels,
        )

        return decoder_logits, ctc_logprobs, encoder_seq_len

def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    bpe_labels = data["bpe_labels"]  # [B, N] (sparse)
    bpe_labels_len = data["bpe_labels:size1"]  # [B, N]

    unnormalized_logits, ctc_logprobs, encoder_seq_len = model(
        raw_audio_features=raw_audio,
        raw_audio_features_lens=raw_audio_len,
        bpe_labels=bpe_labels,
    )
    
    # CTC Loss
    
    transposed_logprobs = torch.permute(ctc_logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        bpe_labels,
        input_lengths=encoder_seq_len,
        target_lengths=bpe_labels_len,
        blank=model.aed_model.vocab_size,  # CTC is last in vocab
        reduction="sum",
    )
    num_phonemes = torch.sum(bpe_labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)

    # CE Loss

    # ignore padded values in the loss
    targets_packed = nn.utils.rnn.pack_padded_sequence(
        bpe_labels, bpe_labels_len.cpu(), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(
        targets_packed, batch_first=True, padding_value=-100
    )

    ce_loss = nn.functional.cross_entropy(
        unnormalized_logits.transpose(1, 2), targets_masked.long(), reduction="sum"
    )  # [B,N]

    num_labels = torch.sum(bpe_labels_len)

    run_ctx.mark_as_loss(name="bpe_ce", loss=ce_loss, inv_norm_factor=num_labels)


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
    for hyp, tag in zip (hypothesis, tags):
        tokens = hyp[0].tokens.numpy()
        sequence = " ".join([run_ctx.labels[i] for i in tokens[1:-2]])
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
