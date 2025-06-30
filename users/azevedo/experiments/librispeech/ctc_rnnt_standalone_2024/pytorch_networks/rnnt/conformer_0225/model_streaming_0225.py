import torch
from torch import nn
from dataclasses import dataclass

from i6_models.primitives.specaugment import specaugment_v1_by_length
from returnn.torch.context import get_run_ctx

from .model_streaming_0225_cfg import ModelConfig
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config
from i6_models.config import ModuleFactoryV1

# frontend imports
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from ..streaming_conformer.generic_frontend_v2 import GenericFrontendV2

# feature extract and conformer module imports
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config

from .conf_lah_carryover_v4 import (
    StreamableConformerEncoderV1, 
    StreamableFeatureExtractorV1, 
    StreamableJoinerV1,
    StreamableConformerEncoderV1Config
)
from ..conformer_0924.model_streaming_lah_carryover_v4 import train_step

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import (
    prior_step,
    prior_init_hook,
    prior_finish_hook,
    Predictor,
)

from ..auxil.functional import num_samples_to_frames, Mode, TrainingStrategy, mask_tensor


class Model(nn.Module):
    def __init__(self, model_config_dict: ModelConfig, **kwargs):
        super().__init__()
        # net_args passed as dict to returnn and retransformed here into dataclass
        self.cfg = ModelConfig.from_dict(model_config_dict)

        if self.cfg.use_vgg:
            frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=self.cfg.frontend_config)
        else:
            frontend = ModuleFactoryV1(module_class=GenericFrontendV2, cfg=self.cfg.frontend_config)

        dual_mode = False
        if self.cfg.dual_mode and self.cfg.training_strategy in [TrainingStrategy.UNIFIED, TrainingStrategy.SWITCHING]:
            dual_mode = True

        conformer_config = StreamableConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            dual_mode=dual_mode,
            frontend=frontend,
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=self.cfg.conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=self.cfg.conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=self.cfg.conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(self.cfg.conformer_size),
                ),
            ),
        )

        self.feature_extraction = StreamableFeatureExtractorV1(
            cfg=self.cfg.feature_extraction_config,
            specaug_cfg=self.cfg.specaug_config,
            specaug_start_epoch=self.cfg.specauc_start_epoch
        )
        self.conformer = StreamableConformerEncoderV1(cfg=conformer_config)
        self.predictor = Predictor(
            cfg=self.cfg.predictor_config,
            label_target_size=self.cfg.label_target_size + 1,  # ctc blank added
            output_dim=self.cfg.joiner_dim,
        )
        self.joiner = StreamableJoinerV1(
            input_dim=self.cfg.joiner_dim,
            output_dim=self.cfg.label_target_size + 1,
            activation=self.cfg.joiner_activation,
            dropout=self.cfg.joiner_dropout,
            dual_mode=dual_mode,
        )

        self.encoder_out_linear = nn.Linear(self.cfg.conformer_size, self.cfg.joiner_dim)

        if self.cfg.ctc_output_loss > 0:
            self.encoder_ctc_on = nn.Linear(self.cfg.conformer_size, self.cfg.label_target_size + 1) 
            if dual_mode:
                self.encoder_ctc_off = nn.Linear(self.cfg.conformer_size, self.cfg.label_target_size + 1)
            else:
                self.encoder_ctc_off = self.encoder_ctc_on

        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        self.chunk_size = self.cfg.chunk_size
        self.lookahead_size = self.cfg.lookahead_size
        self.carry_over_size = self.cfg.carry_over_size
        self.mode = None

    def forward(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
            labels: torch.Tensor, labels_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, N]
        :param labels_len: length of N as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        assert self.mode is not None

        self.feature_extraction.set_mode(self.mode)
        conformer_in, mask = self.feature_extraction(raw_audio, raw_audio_len, self.chunk_size)

        self.conformer.set_mode_cascaded(self.mode)
        conformer_out, out_mask = self.conformer(conformer_in, mask,
                                                 lookahead_size=self.lookahead_size,
                                                 carry_over_size=self.carry_over_size)
        
        if self.mode == Mode.STREAMING:
            conformer_out = conformer_out.flatten(1, 2)  # [B, C'*N, F']
            out_mask = out_mask.flatten(1, 2)  # [B, C'*N]

        conformer_joiner_out = self.encoder_out_linear(conformer_out)
        conformer_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        self.joiner.set_mode(self.mode)
        output_logits, src_len, tgt_len = self.joiner(
            source_encodings=conformer_joiner_out,
            source_lengths=conformer_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len,
        )  # output is [B, T, N, #vocab]

        if self.cfg.ctc_output_loss > 0:
            encoder_ctc = self.encoder_ctc_on if self.mode == Mode.STREAMING else self.encoder_ctc_off
            ctc_logprobs = torch.log_softmax(encoder_ctc(conformer_out), dim=-1)
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs

