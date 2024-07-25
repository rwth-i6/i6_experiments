"""
Like v2, but with i6_models specaugment (v3)
and now controllable start time for when specaugment is applied (v4)
and with the proper feature extraction from i6-models
"""

import numpy as np
import torch
from typing import Tuple, Union, Callable, Optional, List
import torch.functional as F
from torch import nn
from dataclasses import dataclass

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v2 import ConformerBlockV2Config, ConformerBlockV2
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from returnn.torch.context import get_run_ctx

from .conformer_v2_uni_aggr_cfg_v1 import ModelConfig


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


@dataclass
class ConformerAggrEncoderV2Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int
    aggr_layer: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockV2Config

class ConformerAggrEncoderV2(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerAggrEncoderV2Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV2(cfg.block_cfg) for _ in range(cfg.num_layers)])
        self.aggr_layer = cfg.aggr_layer
        self.aggr_lin = nn.Linear(cfg.block_cfg.ff_cfg.input_dim, 1)

    def forward(
        self, data_tensor: torch.Tensor, /, sequence_mask: torch.Tensor, return_layers: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        if return_layers is None:
            return_layers = [len(self.module_list) - 1]

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        outputs = []
        assert (
            max(return_layers) < len(self.module_list) and min(return_layers) >= 0
        ), f"invalid layer index, should be between 0 and {len(self.module_list)-1}"
        for i in range(max(return_layers) + 1):
            if i == self.aggr_layer:
                lengths = torch.sum(sequence_mask, dim=1)
                # from https://github.com/Audio-WestlakeU/UMA-ASR/blob/main/espnet2/asr/uma.py
                batch, length, _ = x.size()
                weights = self.aggr_lin(x)  # [B, T, 1]
                weights = torch.sigmoid(weights)
                scalar_before = weights[:, :-1, :].detach()  # (#batch, L-1, 1)
                scalar_after = weights[:, 1:, :].detach()  # (#batch, L-1, 1)
                scalar_before = torch.nn.functional.pad(scalar_before, (0, 0, 1, 0))  # (#batch, L, 1)
                scalar_after = torch.nn.functional.pad(scalar_after, (0, 0, 0, 1))  # (#batch, L, 1)
                mask = (weights.lt(scalar_before)) & (weights.lt(scalar_after))  # bool tensor (#batch, L, 1)
                mask = mask.reshape(weights.shape[0], -1)  # bool tensor (#batch, L)
                mask[:, 0] = True
                batch_index = mask.nonzero()[:, 0]  # (k,1); [0,0,0,...,1,1,...,2,2,...,#batch-1,...]
                valley_index_start = mask.nonzero()[:, 1]  # (k,1); [0,3,7,...,0,2,...,0,4,...,0,...]
                mask[:, 0] = False
                mask[:, -1] = True
                valley_index_end = mask.nonzero()[:, 1] + 2
                # (k,1); [5,9,...,4,...,6,...]
                valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end),
                                               (length) * torch.ones_like(valley_index_end), valley_index_end)
                _, counts = torch.unique(batch_index,
                                         return_counts=True)  # (#batch, 1); the number of valleys in each sample
                max_counts = (torch.max(counts)).item()
                utri_mat1 = torch.tril(torch.ones(max_counts + 1, max_counts), -1).to(x.device)
                batch_index_mask = utri_mat1[counts]
                batch_index_mask = batch_index_mask.reshape(-1, 1)
                batch_index_mask = batch_index_mask.nonzero()[:, 0]
                valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
                valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),
                                                      1)
                utri_mat = torch.tril(torch.ones(length + 1, length), -1).to(x.device)
                output_mask = (utri_mat[valleys[:, 1]] - utri_mat[valleys[:, 0]]).reshape(batch, max_counts, length)
                output_mask = output_mask.detach()
                alpha_h = torch.mul(weights, x)
                x = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, weights).clamp_(1e-6)
                new_lengths = (lengths / lengths[0] * x.shape[1]).type_as(lengths)
                sequence_mask = mask_tensor(x, new_lengths)
            x = self.module_list[i](x, sequence_mask)  # [B, T, F']
            if i in return_layers:
                outputs.append(x)

        return outputs, sequence_mask


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        conformer_config = ConformerAggrEncoderV2Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV2Config(
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
                modules=self.cfg.module_list,
                scales=self.cfg.module_scales,
            ),
            aggr_layer=self.cfg.aggr_layer,
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerAggrEncoderV2(cfg=conformer_config)
        self.num_output_linears = 1 if self.cfg.aux_ctc_loss_layers is None else len(self.cfg.aux_ctc_loss_layers)
        self.output_linears = nn.ModuleList([
            nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
            for _ in range(self.num_output_linears)
        ])
        self.output_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

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
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
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
        return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]

        conformer_out_layers, out_mask = self.conformer(conformer_in, mask, return_layers=return_layers)
        log_probs_list = []
        for i, (out_layer, scale) in enumerate(zip(conformer_out_layers, self.cfg.aux_ctc_loss_scales)):
            if scale == 0.0:
                continue
            conformer_out = self.output_dropout(out_layer)
            logits = self.output_linears[i](conformer_out)
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs_list.append(log_probs)

        if len(log_probs_list) == 1:
            log_probs_list = log_probs_list[0]

        return log_probs_list, torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    logprobs_list, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    for logprobs, layer_index, scale in zip(logprobs_list, model.cfg.aux_ctc_loss_layers, model.cfg.aux_ctc_loss_scales):
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
        num_phonemes = torch.sum(labels_len)
        run_ctx.mark_as_loss(name=f"ctc_loss_layer{layer_index + 1}", loss=ctc_loss, scale=scale, inv_norm_factor=num_phonemes)

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
