import torch
from torch import nn
from dataclasses import dataclass

from ..conformer_0325.model_dual_0325_v1_cfg import ModelConfig
from i6_models.config import ModuleFactoryV1

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.dropout import BroadcastDropout

from ..conformer_0325.conf_dual_0325_v1 import (
    StreamableConformerEncoderRelPosV2,
    StreamableConformerEncoderRelPosV2Config,

)
from ..conformer_1124.conf_relpos_streaming_v1 import (
    ConformerRelPosBlockV1COV1Config,
    ConformerPositionwiseFeedForwardV2Config,
    ConformerMHSARelPosV1Config,
    ConformerConvolutionV2Config,
)


from ..conformer_0325.conf_dual_0325_v1 import (
    StreamableFeatureExtractorV1,
    StreamableJoinerV1,
)
from ..conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1 import Predictor

# from ..conformer_0924.model_streaming_lah_carryover_v4 import train_step

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import (
    prior_step,
    prior_init_hook,
    prior_finish_hook,
)

from ..auxil.functional import Mode, TrainingStrategy


class Model(nn.Module):
    def __init__(self, model_config_dict: ModelConfig, **kwargs):
        super().__init__()

        self.cfg = ModelConfig.from_dict(model_config_dict)

        dual_mode = False
        if self.cfg.dual_mode and self.cfg.training_strategy in [TrainingStrategy.UNIFIED, TrainingStrategy.SWITCHING]:
            dual_mode = True

        conformer_config = StreamableConformerEncoderRelPosV2Config(
            num_layers=self.cfg.num_layers,
            dual_mode=dual_mode,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=self.cfg.frontend_config),
            block_cfg=ConformerRelPosBlockV1COV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                    input_dim=self.cfg.conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes
                ),
                mhsa_cfg=ConformerMHSARelPosV1Config(
                    input_dim=self.cfg.conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
                    learnable_pos_emb=self.cfg.pos_emb_config.learnable_pos_emb,
                    rel_pos_clip=self.cfg.pos_emb_config.rel_pos_clip,
                    with_linear_pos=self.cfg.pos_emb_config.with_linear_pos,
                    with_pos_bias=self.cfg.pos_emb_config.with_pos_bias,
                    separate_pos_emb_per_head=self.cfg.pos_emb_config.separate_pos_emb_per_head,
                    pos_emb_dropout=self.cfg.pos_emb_config.pos_emb_dropout,
                    with_bias=self.cfg.mhsa_with_bias,
                ),
                conv_cfg=ConformerConvolutionV2Config(
                    channels=self.cfg.conformer_size, kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout, activation=nn.functional.silu,
                    norm=LayerNormNC(self.cfg.conformer_size),
                    dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,

                ),
            ),
        )

        self.feature_extraction = StreamableFeatureExtractorV1(
            cfg=self.cfg.feature_extraction_config,
            specaug_cfg=self.cfg.specaug_config,
            specaug_start_epoch=self.cfg.specauc_start_epoch
        )
        self.conformer = StreamableConformerEncoderRelPosV2(cfg=conformer_config)
        self.predictor = Predictor(
            cfg=self.cfg.predictor_config,
            label_target_size=self.cfg.label_target_size + 1,  # ctc blank added
            output_dim=self.cfg.joiner_dim,
            dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
        )
        self.joiner = StreamableJoinerV1(
            input_dim=self.cfg.joiner_dim,
            output_dim=self.cfg.label_target_size + 1,
            activation=self.cfg.joiner_activation,
            dropout=self.cfg.joiner_dropout,
            dropout_broadcast_axes=self.cfg.dropout_broadcast_axes,
            dual_mode=dual_mode
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

        self.output_dropout = BroadcastDropout(p=self.cfg.final_dropout,
                                               dropout_broadcast_axes=self.cfg.dropout_broadcast_axes)

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



def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    from i6_native_ops import monotonic_rnnt

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    prepended_targets = labels.new_empty([labels.size(0), labels.size(1) + 1])
    prepended_targets[:, 1:] = labels
    prepended_targets[:, 0] = model.cfg.label_target_size  # blank is last index
    prepended_target_lengths = labels_len + 1

    num_phonemes = torch.sum(labels_len)

    def _train_step_mode(encoder_mode: Mode, scale: float):
        model.mode = encoder_mode

        logits, audio_features_len, ctc_logprobs = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
            labels=prepended_targets, labels_len=prepended_target_lengths,
        )

        logprobs = torch.log_softmax(logits, dim=-1)

        rnnt_loss = monotonic_rnnt.monotonic_rnnt_loss(
            acts=logprobs,
            labels=labels,
            input_lengths=audio_features_len.to(dtype=torch.int32),
            label_lengths=labels_len.to(dtype=torch.int32),
            blank_label=model.cfg.label_target_size,
        )

        mode_str = encoder_mode.name.lower()[:3]
        ctc_loss = None
        if ctc_logprobs is not None:
            transposed_logprobs = torch.permute(ctc_logprobs, (1, 0, 2))  # CTC needs [T, B, #vocab + 1]

            ctc_loss = nn.functional.ctc_loss(
                transposed_logprobs,
                labels,
                input_lengths=audio_features_len,
                target_lengths=labels_len,
                blank=model.cfg.label_target_size,
                reduction="sum",
                zero_infinity=True,
            )

        return {
            "rnnt.%s" % mode_str: {
                "loss": rnnt_loss,
                "scale": scale
            },
            "ctc.%s" % mode_str: {
                "loss": ctc_loss,
                "scale": model.cfg.ctc_output_loss * scale
            }
        }

    def _train_step_unified():
        str_loss = _train_step_mode(Mode.STREAMING, scale=model.cfg.online_model_scale)
        off_loss = _train_step_mode(Mode.OFFLINE, scale=1 - model.cfg.online_model_scale)
        return {**str_loss, **off_loss}

    def _train_step_switching():
        encoder_mode = Mode.STREAMING if run_ctx.global_step % 2 == 0 else Mode.OFFLINE
        switch_loss = _train_step_mode(encoder_mode, scale=1)
        return switch_loss

    if model.cfg.training_strategy == TrainingStrategy.UNIFIED:
        loss_dict = _train_step_unified()
    elif model.cfg.training_strategy == TrainingStrategy.SWITCHING:
        loss_dict = _train_step_switching()
    else:
        loss_dict = _train_step_mode(Mode.STREAMING, scale=1)

    for loss_key in loss_dict:
        run_ctx.mark_as_loss(
            name=loss_key,
            loss=loss_dict[loss_key]["loss"],
            inv_norm_factor=num_phonemes,
            scale=loss_dict[loss_key]["scale"]
        )

