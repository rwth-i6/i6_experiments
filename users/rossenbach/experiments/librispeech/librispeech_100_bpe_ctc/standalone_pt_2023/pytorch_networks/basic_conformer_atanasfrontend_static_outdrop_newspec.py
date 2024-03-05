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



def _mask(tensor, batch_axis, axis, pos, max_amount):
    batch_dim = tensor.shape[batch_axis]
    dim = tensor.shape[axis]
    amount = torch.randint(low=1, high=max_amount + 1, size=(batch_dim,), dtype=torch.int32).to(device=tensor.device)
    pos2 = torch.min(pos + amount, torch.tensor([dim] * batch_dim).to(device=tensor.device))
    idxs = torch.arange(0, dim).to(device=tensor.device).unsqueeze(0)  # [1,dim]
    pos_bc = pos.unsqueeze(1)  # [B,1]
    pos2_bc = pos2.unsqueeze(1)  # [B,1]
    cond = torch.logical_and(torch.greater_equal(idxs, pos_bc), torch.less(idxs, pos2_bc))  # [B,dim]
    if batch_axis > axis:
        cond = cond.transpose(0, 1)  # [dim,B]
    cond = torch.reshape(
        cond, shape=[tensor.shape[i] if i in (batch_axis, axis) else 1 for i in range(len(tensor.shape))]
    )
    tensor = torch.where(cond, 0.0, tensor)
    return tensor


def _random_mask(tensor, batch_axis, axis, min_num, max_num, max_dims):
    batch_dim = tensor.shape[batch_axis]
    num_masks = torch.randint(min_num, max_num, size=(batch_dim,)).to(device=tensor.device)  # [B]

    z = -torch.log(-torch.log(torch.rand((batch_dim, tensor.shape[axis])).to(device=tensor.device)))  # [B,dim]
    _, indices = torch.topk(z, num_masks.max().item(), dim=1)

    for i in range(num_masks.max().item()):
        tensor = _mask(
            tensor, batch_axis, axis, indices[:, i], max_dims
        )
    return tensor

def returnn_specaugment(tensor: torch.Tensor, time_num_masks, time_mask_max_size, freq_num_masks, freq_mask_max_size):
    assert len(tensor.shape) == 3
    tensor = _random_mask(tensor, 0, 1, 2, time_num_masks, time_mask_max_size)  # time masking
    tensor = _random_mask(tensor, 0, 2, 2, freq_num_masks, freq_mask_max_size)  # freq masking
    return tensor


def returnn_specaugment_by_length(audio_features, repeat_per_n_frames, max_dim_time, num_repeat_feat, max_dim_feat):
    return returnn_specaugment(
        audio_features,
        time_num_masks=audio_features.size(1) // repeat_per_n_frames,
        time_mask_max_size=max_dim_time,
        freq_num_masks=num_repeat_feat,
        freq_mask_max_size=max_dim_feat)


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




@dataclass
class AtanasVGGFrontendConfigV1(ModelConfiguration):
    num_inputs: int # 80
    conv1_channels: int # 32
    conv2_channels: int # 64
    conv3_channels: int # 64
    conv_kernel_size: int # 3
    conv1_stride: int # 1
    conv2_stride: int # 3
    conv3_stride: int # 2
    pool_size: int # 2
    linear_size: int # 512
    dropout: float # 0.1

    def __post__init__(self):
        super().__post_init__()
        assert self.conv_kernel_size % 2 == 1, "Conv kernel size must be odd."


def subsample_mask(sequence_mask: torch.Tensor, subsampling_factor: int):
    if subsampling_factor == 1:
        return sequence_mask

    max_len = sequence_mask.shape[1]

    padding = 0
    if (overhang := max_len % subsampling_factor) != 0:
        padding = subsampling_factor - overhang

    padded_mask = torch.nn.functional.pad(sequence_mask, pad=(0, padding), value=0)

    reshaped_mask = padded_mask.reshape(padded_mask.shape[0], -1, subsampling_factor)

    subsampled_mask = torch.any(reshaped_mask == 1, dim=2)
    subsampled_mask = subsampled_mask.type(sequence_mask.dtype)

    return subsampled_mask


class AtanasVGGFrontendV1(torch.nn.Module):
    def __init__(self, config: AtanasVGGFrontendConfigV1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=config.conv1_channels, # 32
            kernel_size=(config.conv_kernel_size, 1), # (3, 1)
            padding=(config.conv_kernel_size // 2, 0), # (1, 1)
            stride=(config.conv1_stride, 1), # (1, 1)
        )
        self.pool = torch.nn.MaxPool2d(
            kernel_size=(1, config.pool_size), # (1, 2)
            stride=(1, config.pool_size), # (1, 2)
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=config.conv1_channels, # 32
            out_channels=config.conv2_channels, # 64
            kernel_size=config.conv_kernel_size, # 3
            padding=config.conv_kernel_size // 2, # 1
            stride=(config.conv2_stride, 1), # (3, 1)
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=config.conv2_channels, # 64
            out_channels=config.conv3_channels, # 64
            kernel_size=config.conv_kernel_size, # 3
            padding=config.conv_kernel_size // 2, # 1
            stride=(config.conv3_stride, 1), # (2, 1)
        )
        self.linear = torch.nn.Linear(
            config.conv3_channels * (config.num_inputs // config.pool_size), config.linear_size
        )
        self.subsample_factor = config.conv1_stride * config.conv2_stride * config.conv3_stride
        self.dropout = config.dropout
        self.layer_norm = torch.nn.LayerNorm(config.linear_size)

    def forward(
        self, x: torch.Tensor, sequence_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x[:, None, :, :]  # [B, 1, T, F]
        x = self.conv1(x)  # [B, C_1, T', F]
        x = torch.nn.functional.silu(x)  # [B, C_1, T', F]
        x = self.pool(x)  # [B, C_1, T', F']
        x = self.conv2(x)  # [B, C_2, T'', F']
        x = torch.nn.functional.silu(x)  # [B, C_2, T'', F']
        x = self.conv3(x)  # [B, C_3, T''', F']
        x = torch.nn.functional.silu(x)  # [B, C_3, T''', F']
        x = torch.transpose(x, 1, 2)  # [B, T''', C_3, F']
        x = torch.flatten(x, start_dim=2)  # [B, T''', C_3 * F']
        x = self.linear(x)  # [B, T''', F'']
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)  # [B, T''', F'']
        x = self.layer_norm(x)  # [B, T''', F'']

        if sequence_mask is None:
            subsampled_mask = None
        else:
            subsampled_mask = subsample_mask(sequence_mask, self.subsample_factor)

        return x, subsampled_mask

class Model(torch.nn.Module):
    def __init__(self, **net_kwargs):
        super().__init__()
        self.net_kwargs = {
            "repeat_per_num_frames": 50,
            "max_dim_time": 20,
            "max_dim_feat": 8,
            "num_repeat_feat": 5,
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
        conformer_size = 512
        frontend_config = AtanasVGGFrontendConfigV1(
            num_inputs=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv_kernel_size=3,
            conv1_stride=1,
            conv2_stride=3,
            conv3_stride=2,
            pool_size=2,
            linear_size=512,
            dropout=0.1,
        )
        conformer_config = ConformerEncoderV1Config(
            num_layers=8,
            frontend=ModuleFactoryV1(module_class=AtanasVGGFrontendV1, cfg=frontend_config),
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

            if self.training:
                audio_features_masked_2 = returnn_specaugment_by_length(
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
        conformer_out = nn.functional.dropout(conformer_out, p=0.2, training=self.training)
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
    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))

