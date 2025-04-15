import numpy as np
import torch
from torch import nn
from returnn.torch.context import get_run_ctx
from typing import List, Tuple, Optional

from .model_dual_0425_cfg import ModelConfig
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.dropout import BroadcastDropout
from i6_models.primitives.specaugment import specaugment_v1_by_length

from ...rnnt.conformer_0325.conf_dual_0325_v1 import (
    StreamableModule,
    StreamableFeatureExtractorV1,
    StreamableConformerEncoderRelPosV2,
    StreamableConformerEncoderRelPosV2Config,
    ConformerRelPosBlockV1COV1Config,
    ConformerPositionwiseFeedForwardV2Config,
    ConformerMHSARelPosV1Config,
)
from ...rnnt.conformer_1124.conf_relpos_streaming_v1 import ConformerConvolutionV2Config
from ..conformer_0125.model_relpos_streaming import (
    prior_init_hook, prior_step, prior_finish_hook
)

from ...rnnt.auxil.functional import Mode, TrainingStrategy


class Model(StreamableModule):
    def __init__(self, model_config_dict, **kwargs):
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
        self.final_linear = nn.Linear(self.cfg.conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
        self.final_dropout = BroadcastDropout(p=self.cfg.final_dropout,
                                              dropout_broadcast_axes=self.cfg.dropout_broadcast_axes)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        self.chunk_size = self.cfg.chunk_size
        self.lookahead_size = self.cfg.lookahead_size
        self.carry_over_size = self.cfg.carry_over_size
        self.mode = None

    def forward_offline(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        conformer_in, mask = self.feature_extraction(raw_audio, raw_audio_len, self.chunk_size)
        conformer_out, out_mask = self.conformer(conformer_in, mask,
                                                 lookahead_size=self.lookahead_size,
                                                 carry_over_size=self.carry_over_size)
        # final linear layer
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        frame_lengths = torch.sum(out_mask, dim=1)
        print(f"> raw_audio_len = \n\t{raw_audio_len}")
        print(f"> frame_lens = \n\t{frame_lengths}")
        audio_len_sec = raw_audio_len / 16e3
        frame_len_sec = frame_lengths * 0.06
        print(f"> audio lens [s] = \n\t{audio_len_sec}")
        print(f"> frame lens [s] = \n\t{frame_len_sec}")
        print(f"> audio lens - frame lens [s] = \n\t{audio_len_sec.cpu() - frame_len_sec.cpu()}\n\n")

        return log_probs, frame_lengths

    def forward_streaming(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        conformer_in, mask = self.feature_extraction(raw_audio, raw_audio_len, self.chunk_size)
        conformer_out, out_mask = self.conformer(conformer_in, mask,
                                                 lookahead_size=self.lookahead_size,
                                                 carry_over_size=self.carry_over_size)

        conformer_out = conformer_out.flatten(1, 2)  # [B, C'*N, F']
        out_mask = out_mask.flatten(1, 2)  # [B, C'*N]

        # final linear layer
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        frame_lengths = torch.sum(out_mask, dim=1)

        print(f"> raw_audio_len = \n\t{raw_audio_len}")
        print(f"> frame_lens = \n\t{frame_lengths}")
        audio_len_sec = raw_audio_len / 16e3
        frame_len_sec = frame_lengths * 0.06
        print(f"> audio lens [s] = \n\t{audio_len_sec}")
        print(f"> frame lens [s] = \n\t{frame_len_sec}")
        print(f"> audio lens - frame lens [s] = \n\t{audio_len_sec.cpu() - frame_len_sec.cpu()}\n\n")

        return log_probs, frame_lengths
    
    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        assert chunk_size is not None
        
        with torch.no_grad():
            chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(chunk_size))
            audio_features, audio_features_len = self.feature_extraction.infer(input, lengths, chunk_size_frames)

            encoder_out, out_mask, state = self.encoder.infer(
                audio_features, audio_features_len,
                states,
                chunk_size=chunk_size_frames,
                lookahead_size=self.lookahead_size
            )

            encoder_out = self.final_linear(encoder_out) # (1, C', V+1)
            encoder_out_lengths = torch.sum(out_mask, dim=1)  # [1, T] -> [1]

        return encoder_out, encoder_out_lengths, [state]



def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    num_phonemes = torch.sum(labels_len)

    def _train_step_mode(encoder_mode: Mode, scale: float):
        model.set_mode_cascaded(encoder_mode)
        logprobs, audio_features_len = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
        )

        mode_str = encoder_mode.name.lower()[:3]
        transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, #vocab + 1]

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
            "ctc.%s" % mode_str: {
                "loss": ctc_loss,
                "scale": scale
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
