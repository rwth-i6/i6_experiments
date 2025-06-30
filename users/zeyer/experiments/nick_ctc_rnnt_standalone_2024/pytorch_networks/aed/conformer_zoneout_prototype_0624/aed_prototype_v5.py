"""
First AED Prototype based on the CTC setup

v4: with added label smoothing start epoch
"""

import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1, ConformerEncoderV1Config, \
    ConformerBlockV1Config
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from i6_models.decoder.attention import AdditiveAttention, AttentionLSTMDecoderV1
from i6_models.decoder.zoneout_lstm import ZoneoutLSTMCell


from returnn.torch.context import get_run_ctx

from .aed_prototype_v5_cfg import ModelConfig


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

class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **net_kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size
        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
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
                    channels=conformer_size, kernel_size=self.cfg.conv_kernel_size, dropout=self.cfg.conv_dropout, activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size)
                ),
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderV1(cfg=conformer_config)
        self.decoder = AttentionLSTMDecoderV1(cfg=self.cfg.decoder_config)
        
        self.final_linear = nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

        self.ctc_loss_scale = self.cfg.ctc_loss_scale
        self.label_smoothing = self.cfg.label_smoothing
        self.label_smoothing_start_epoch = self.cfg.label_smoothing_start_epoch


    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
        bpe_labels: Optional[torch.Tensor],
        do_search: bool = False,
        encoder_only: bool = False,
        ctc_only: bool = False,
    ):
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

        conformer_out, out_mask = self.conformer(conformer_in, mask)

        if encoder_only:
            encoder_seq_len = torch.sum(out_mask, dim=1)  # [B]
            return conformer_out, encoder_seq_len
        elif ctc_only:
            conformer_out = self.final_dropout(conformer_out)
            logits = self.final_linear(conformer_out)

            ctc_log_probs = torch.log_softmax(logits, dim=2)
            encoder_seq_len = torch.sum(out_mask, dim=1)  # [B]

            return ctc_log_probs, encoder_seq_len
        elif not do_search:
            conformer_out = self.final_dropout(conformer_out)
            logits = self.final_linear(conformer_out)

            ctc_log_probs = torch.log_softmax(logits, dim=2)

            decoder_logits, state = self.decoder(
                conformer_out, bpe_labels, audio_features_len.to(device=conformer_out.device)
            )
            encoder_seq_len = torch.sum(out_mask, dim=1)  # [B]

            return decoder_logits, state, conformer_out, encoder_seq_len, ctc_log_probs
        else:
            encoder_outputs = conformer_out
            # search here
            state = None
            labels = encoder_outputs.new_zeros(encoder_outputs.size()[0], 1).long()  # [B, 1] labels
            total_scores = encoder_outputs.new_zeros(encoder_outputs.size()[0])  # [B] labels
            label_stack = []
            for i in range(int(encoder_outputs.size(1)) * 2):
                # single forward
                decoder_logits, state = self.decoder(
                    encoder_outputs, labels, audio_features_len, shift_embeddings=(i == 0), state=state
                )  # [B,1,Vocab] and state of [B, *]
                log_softmax = torch.nn.functional.log_softmax(decoder_logits, dim=-1)
                labels = torch.argmax(log_softmax, dim=-1)
                # print("labels")
                # print(labels.size())
                # print("log_softmax")
                # print(log_softmax.size())
                gathered_scores = torch.gather(log_softmax[:, 0], dim=-1, index=labels)
                total_scores += gathered_scores.squeeze()  # add argmax scores
                label_stack.append(labels.cpu().detach().numpy())
                # print("Search step %i" % (i+1))
                # print("Current max labels: %s" % str(labels))
                # print("Current scores: %s" % str(log_softmax))

            return label_stack, total_scores


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    decoder_logits, state, encoder_outputs, audio_features_len, ctc_logprobs = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        bpe_labels=labels,
    )

    transposed_logprobs = torch.permute(ctc_logprobs, (1, 0, 2))  # CTC needs [T, B, F]
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
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, scale=model.ctc_loss_scale, inv_norm_factor=num_phonemes)
        
    # CE Loss

    # ignore padded values in the loss
    targets_packed = nn.utils.rnn.pack_padded_sequence(
        labels, labels_len.cpu(), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(
        targets_packed, batch_first=True, padding_value=-100
    )

    label_smoothing = 0.0
    if model.training is True:
        label_smoothing = model.label_smoothing if run_ctx.epoch >= model.label_smoothing_start_epoch else 0.0


    ce_loss = nn.functional.cross_entropy(
        decoder_logits.transpose(1, 2), targets_masked.long(), reduction="sum", label_smoothing=label_smoothing,
    )  # [B,N]

    num_labels = torch.sum(labels_len)

    run_ctx.mark_as_loss(name="decoder_ce", loss=ce_loss, inv_norm_factor=num_labels)
