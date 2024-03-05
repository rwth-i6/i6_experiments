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

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.parts.frontend.common import mask_pool

from returnn.torch.context import get_run_ctx




def apply_static_specaugment(audio_features, num_repeat_time, max_dim_time, num_repeat_feat, max_dim_feat):
    """
    Does fixed masking for each sequence in a batch, where the mask locations and length are drawn uniformly
    but consistent.

    :param Tensor audio_features: the audio_features audio features (B,T,F)
    :param int num_repeat_time: number of repetitions to apply time mask
    :param int max_dim_time: number of columns to be masked on time dimension will be uniformly sampled from [0, mask_param]
    :param int num_repeat_feat: number of repetitions to apply feature mask
    :param int max_dim_feat: number of columns to be masked on feature dimension will be uniformly sampled from [0, mask_param]
    """
    for _ in range(num_repeat_time):
        audio_features = mask_along_axis(audio_features, mask_param=max_dim_time, mask_value=0.0, axis=1)

    for _ in range(num_repeat_feat):
        audio_features = mask_along_axis(audio_features, mask_param=max_dim_feat, mask_value=0.0, axis=2)
    return audio_features


def apply_static_specaugment_by_length(audio_features, repeat_per_n_frames, max_dim_time, num_repeat_feat, max_dim_feat):
    """
    Does fixed masking for each sequence in a batch, where the mask locations and length are drawn uniformly
    but consistent.

    Considers the maximum sequence length in the batch to determine the number of time masks.

    :param input: [B, T, F] audio features
    :param repeat_per_n_frames: create a time mask for each of n frames, but place them uniformly over the batch
        (sequences with a lot of padding might have all masks applied in the padding area)
    :param int max_dim_time: number of columns to be masked on time dimension will be uniformly sampled from [0, mask_param]
    :param int num_repeat_feat: number of repetitions to apply feature mask
    :param int max_dim_feat: number of columns to be masked on feature dimension will be uniformly sampled from [0, mask_param]
    :return:
    """
    return apply_static_specaugment(
        audio_features,
        num_repeat_time=audio_features.size(1) // repeat_per_n_frames,
        max_dim_time=max_dim_time,
        num_repeat_feat=num_repeat_feat,
        max_dim_feat=max_dim_feat)


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


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TwoLayer1DFrontend(nn.Module):
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
        self.pos_encoding = PositionalEncoding(model_cfg.conv2_channels, 0.1)


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
        tensor = self.pos_encoding(tensor)

        return tensor, sequence_mask

    def _calculate_dim(self) -> int:
        return self.conv2.out_channels





class Model(torch.nn.Module):
    def __init__(self, **net_kwargs):
        super().__init__()
        self.net_kwargs = {
            "repeat_per_num_frames": 100,
            "max_dim_feat": 8,
            "num_repeat_feat": 5,
            "max_dim_time": 20,
        }
        fe_config = LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=False,
        )
        frontend_config = VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,  # =same
            pool1_stride=(2, 1),  # pool along the time axis,
            pool1_kernel_size=(2, 1),
            pool1_padding=(1, 0),
            pool2_stride=(2, 2),  # pool along the feature axis
            pool2_kernel_size=(2, 2),
            pool2_padding=(1, 0),
            out_features=512,
            activation=nn.ReLU(),
        )
        frontend_config = TwoLayer1DFrontendConfig(
            in_features=80,
            conv1_channels=512,
            conv1_kernel_size=5,
            conv1_stride=3,
            conv2_channels=512,
            conv2_stride=2,
            conv2_kernel_size=5,
        )
        conformer_size = 512
        conformer_config = ConformerEncoderV1Config(
            num_layers=8,
            frontend=ModuleFactoryV1(module_class=TwoLayer1DFrontend, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=conformer_size,
                    dropout=0.2,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=conformer_size,
                    num_att_heads=4,
                    att_weights_dropout=0.2,
                    dropout=0.2,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=conformer_size, kernel_size=9, dropout=0.2, activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size)
                ),
            ),
        )

        run_ctx = get_run_ctx()
        dataset = run_ctx.engine.train_dataset or run_ctx.engine.forward_dataset
        self.label_target_size = len(dataset.datasets["zip_dataset"].targets.labels)

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=fe_config)
        self.conformer = ConformerEncoderV1(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.label_target_size + 1)  # + CTC blank

        self.export_mode = False
        
        # initialize weights
        self.apply(self._weight_init)

    @staticmethod
    def _weight_init(module: torch.nn.Module):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            print("apply weight init for %s" % str(module))
            nn.init.xavier_uniform_(module.weight)

    def forward(
            self,
            raw_audio: torch.Tensor,
            raw_audio_len: torch.Tensor,
    ): 
        
        squeezed_features = torch.squeeze(raw_audio)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()

            if self.training and run_ctx.epoch >= 10:
                audio_features_masked_2 = apply_static_specaugment_by_length(
                    audio_features,
                    repeat_per_n_frames=self.net_kwargs["repeat_per_num_frames"],
                    max_dim_time=self.net_kwargs["max_dim_time"],
                    num_repeat_feat=self.net_kwargs["num_repeat_feat"],
                    max_dim_feat=self.net_kwargs["max_dim_feat"])
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        conformer_out, out_mask = self.conformer(conformer_in, mask)
        logits = self.final_linear(conformer_out)

        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    bpe_labels = data["bpe_labels"]  # [B, N] (sparse)
    bpe_labels_len = data["bpe_labels:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        bpe_labels,
        input_lengths=audio_features_len,
        target_lengths=bpe_labels_len,
        blank=model.label_target_size,
        reduction="sum",
    )
    num_phonemes = torch.sum(bpe_labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    from torchaudio.models.decoder import ctc_decoder
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")
    import subprocess
    lm = subprocess.check_output(["cf", kwargs["arpa_lm"]]).decode().strip()
    labels = run_ctx.engine.forward_dataset.datasets["zip_dataset"].targets.labels
    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=kwargs["lexicon"],
        lm=lm,
        lm_weight=kwargs["lm_weight"],
        tokens=labels + ["[blank]", "[SILENCE]"],
        blank_token="[blank]",
        sil_token="[SILENCE]",
        unk_word="[UNKNOWN]",
        nbest=1,
        beam_size=kwargs["beam_size"],
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = kwargs.get("blank_log_penalty", None)


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()
    
    # write empty HDF until new ForwardJob exists
    f = open("output.hdf", "wt")
    f.write(" ")
    f.close()


def search_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    tags = data["seq_tag"]

    logprobs_cpu = logprobs.cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty

    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())
    # for hyp, tag in zip (hypothesis, tags):
    #     tokens = hyp[0].tokens.numpy()
    #     sequence = " ".join([run_ctx.labels[i] for i in tokens[1:-2]])
    #     run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
    for hyp, tag in zip (hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))

