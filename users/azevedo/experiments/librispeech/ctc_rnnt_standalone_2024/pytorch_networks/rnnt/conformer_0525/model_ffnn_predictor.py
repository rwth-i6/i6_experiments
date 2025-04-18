import torch
from torch import nn
from typing import Optional, List, Tuple

from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.dropout import BroadcastDropout

from ..conformer_1124.conf_relpos_streaming_v1 import (
    ConformerRelPosBlockV1COV1Config,
    ConformerPositionwiseFeedForwardV2Config,
    ConformerMHSARelPosV1Config,
    ConformerConvolutionV2Config,
)
from ..conformer_0325.conf_dual_0325_v1 import StreamableJoinerV1
from .model_ffnn_predictor_cfg import ModelConfig
from ..conformer_0425.conf_dual_0425_v1 import (
    StreamableConformerEncoderRelPosV3Config,
    StreamableConformerEncoderRelPosV3,
    StreamableModule,
)

from .ffnn_prediction_net import FFNNPredictor
# from ..conformer_0924.model_streaming_lah_carryover_v4 import train_step
from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import (
    prior_step,
    prior_init_hook,
    prior_finish_hook,
)
from ..auxil.functional import TrainingStrategy, Mode


class Model(StreamableModule):
    def __init__(self, model_config_dict: ModelConfig, **kwargs):
        super().__init__()

        self.cfg = ModelConfig.from_dict(model_config_dict)

        dual_mode = False
        if self.cfg.dual_mode and self.cfg.training_strategy in [TrainingStrategy.UNIFIED, TrainingStrategy.SWITCHING]:
            dual_mode = True

        conformer_config = StreamableConformerEncoderRelPosV3Config(
            num_layers=self.cfg.num_layers,
            dual_mode=dual_mode,
            feature_extraction_config=self.cfg.feature_extraction_config,
            specaug_config=self.cfg.specaug_config,
            specauc_start_epoch=self.cfg.specauc_start_epoch,

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
        self.conformer = StreamableConformerEncoderRelPosV3(cfg=conformer_config, out_dim=self.cfg.joiner_dim)
        self.predictor = FFNNPredictor(
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

    def forward_offline(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
            labels: torch.Tensor, labels_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        (b4_final_lin, conformer_out), conformer_out_lengths = self.conformer(raw_audio, raw_audio_len)

        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        output_logits, src_len, tgt_len = self.joiner(
            source_encodings=conformer_out,
            source_lengths=conformer_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len,
        )  # output is [B, T, N, #vocab]

        if self.cfg.ctc_output_loss > 0:
            ctc_logprobs = torch.log_softmax(self.encoder_ctc_off(b4_final_lin), dim=-1)
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs

    def forward_streaming(
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
        (b4_final_lin, conformer_out), conformer_out_lengths = self.conformer(
            raw_audio, raw_audio_len,
            chunk_size=self.chunk_size,
            lookahead_size=self.lookahead_size,
            carry_over_size=self.carry_over_size
        ) # [B, C'*N, F'], [B, C'*N]

        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        output_logits, src_len, tgt_len = self.joiner(
            source_encodings=conformer_out,
            source_lengths=conformer_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len,
        )  # output is [B, T, N, #vocab]

        if self.cfg.ctc_output_loss > 0:
            ctc_logprobs = torch.log_softmax(self.encoder_ctc_on(b4_final_lin), dim=-1)
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size: Optional[float],
            lookahead_size: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:

        if self._mode == Mode.OFFLINE:
            raise ValueError("Model should not be in offline mode during streaming inference.")
        if chunk_size is None or lookahead_size is None:
            raise ValueError("chunk_size or lookahead_size should not be None.")

        with torch.no_grad():
            encoder_out, encoder_out_lengths, state = self.encoder.infer(
                input, lengths, states, chunk_size=chunk_size, lookahead_size=lookahead_size
            )

        return encoder_out, encoder_out_lengths, state


def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    # from i6_native_ops import monotonic_rnnt
    from i6_native_ops import warp_rnnt

    # NOTE: model.config.train_config

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    prepended_targets = labels.new_empty([labels.size(0), labels.size(1) + 1])
    prepended_targets[:, 1:] = labels
    prepended_targets[:, 0] = model.cfg.label_target_size  # blank is last index
    prepended_target_lengths = labels_len + 1

    num_phonemes = torch.sum(labels_len)

    def _train_step_mode(mode: Mode, scale: float):
        model.set_mode_cascaded(mode)

        logits, audio_features_len, ctc_logprobs = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
            labels=prepended_targets, labels_len=prepended_target_lengths,
        )
        logprobs = torch.log_softmax(logits, dim=-1)
        fastemit_lambda = model.cfg.fastemit_lambda

        rnnt_loss = warp_rnnt.rnnt_loss(
            log_probs=logprobs,
            frames_lengths=audio_features_len.to(dtype=torch.int32),
            labels=labels,
            labels_lengths=labels_len.to(dtype=torch.int32),
            blank=model.cfg.label_target_size,
            fastemit_lambda=fastemit_lambda if fastemit_lambda is not None else 0.0,
            reduction="sum",
            gather=True,
        )

        mode_str = mode.name.lower()[:3]
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
        off_loss = _train_step_mode(Mode.OFFLINE, scale=1-model.cfg.online_model_scale)
        return {**str_loss, **off_loss}

    def _train_step_switching():
        encoder_mode = Mode.STREAMING if run_ctx.global_step % 2 == 0 else Mode.OFFLINE
        switch_loss = _train_step_mode(encoder_mode, scale=1)
        return switch_loss

    match model.cfg.training_strategy:
        case TrainingStrategy.UNIFIED:
            loss_dict = _train_step_unified()
        case TrainingStrategy.SWITCHING:
            loss_dict = _train_step_switching()
        case TrainingStrategy.STREAMING:
            loss_dict = _train_step_mode(Mode.STREAMING, scale=1)
        case TrainingStrategy.OFFLINE:
            loss_dict = _train_step_mode(Mode.OFFLINE, scale=1)
        case _:
            NotImplementedError("Training Strategy not available yet.")

    for loss_key in loss_dict:
        run_ctx.mark_as_loss(
            name=loss_key,
            loss=loss_dict[loss_key]["loss"],
            inv_norm_factor=num_phonemes,
            scale=loss_dict[loss_key]["scale"]
        )

