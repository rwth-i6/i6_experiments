"""
Like v2, but with i6_models specaugment (v3)
and now controllable start time for when specaugment is applied (v4)
and with the proper feature extraction from i6-models
"""

import contextlib
import numpy as np
import torch
from torch import nn

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosEncoderV1Config
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosBlockV1Config, ConformerRelPosEncoderV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from i6_models.parts.conformer.convolution import ConformerConvolutionV2Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
from i6_models.parts.dropout import BroadcastDropout
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from returnn.torch.context import get_run_ctx

from .i6models_relposV1_VGG4LayerActFrontendV1_feat_v1_cfg import ModelConfig
from .i6models_relposV1_VGG4LayerActFrontendV1_v1 import (
    mask_tensor, train_step, prior_init_hook, prior_finish_hook, prior_step
)
from ..features.scf import (
    SupervisedConvolutionalFeatureExtractionV1,
    SupervisedConvolutionalFeatureExtractionV2,
)


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        conformer_config = ConformerRelPosEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
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
                    channels=conformer_size, kernel_size=self.cfg.conv_kernel_size, dropout=self.cfg.conv_dropout, activation=nn.functional.silu,
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
        
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad() if not self.training else contextlib.ExitStack():
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
