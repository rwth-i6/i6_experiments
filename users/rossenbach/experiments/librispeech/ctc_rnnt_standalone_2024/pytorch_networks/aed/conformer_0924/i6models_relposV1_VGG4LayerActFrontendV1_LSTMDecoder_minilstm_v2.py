"""
First AED Prototype based on the CTC setup

v4: with added label smoothing start epoch
"""

import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosEncoderV1Config, ConformerRelPosEncoderV1, ConformerRelPosBlockV1Config
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1

from i6_models.parts.conformer.convolution import ConformerConvolutionV2, ConformerConvolutionV2Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2, ConformerPositionwiseFeedForwardV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1, ConformerMHSARelPosV1Config
from i6_models.parts.dropout import BroadcastDropout
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.decoder.attention import AttentionLSTMDecoderV1

from returnn.torch.context import get_run_ctx

from .i6models_relposV1_VGG4LayerActFrontendV1_LSTMDecoder_minilstm_v1_cfg import ModelConfig


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



from i6_models.decoder.attention import AttentionLSTMDecoderV1Config, ZoneoutLSTMCell, AdditiveAttention


class AttentionLSTMDecoderWithForcedContext(nn.Module):
    """
    Single-headed Attention decoder with additive attention mechanism.
    """

    def __init__(self, cfg: AttentionLSTMDecoderV1Config):
        super().__init__()

        self.target_embed = nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.target_embed_dim)
        self.target_embed_dropout = nn.Dropout(cfg.target_embed_dropout)

        lstm_cell = nn.LSTMCell(
            input_size=cfg.target_embed_dim + cfg.encoder_dim,
            hidden_size=cfg.lstm_hidden_size,
        )
        self.lstm_hidden_size = cfg.lstm_hidden_size
        # if zoneout drop probs are 0, then it is equivalent to normal LSTMCell
        self.s = ZoneoutLSTMCell(
            cell=lstm_cell,
            zoneout_h=cfg.zoneout_drop_h,
            zoneout_c=cfg.zoneout_drop_c,
        )

        self.s_transformed = nn.Linear(cfg.lstm_hidden_size, cfg.attention_cfg.attention_dim, bias=False)  # query

        # for attention
        self.enc_ctx = nn.Linear(cfg.encoder_dim, cfg.attention_cfg.attention_dim)
        self.attention = AdditiveAttention(cfg.attention_cfg)

        # for weight feedback
        self.inv_fertility = nn.Linear(cfg.encoder_dim, 1, bias=False)  # followed by sigmoid
        self.weight_feedback = nn.Linear(1, cfg.attention_cfg.attention_dim, bias=False)

        self.readout_in = nn.Linear(cfg.lstm_hidden_size + cfg.target_embed_dim + cfg.encoder_dim, cfg.output_proj_dim)
        assert cfg.output_proj_dim % 2 == 0, "output projection dimension must be even for the MaxOut op of 2 pieces"
        self.output = nn.Linear(cfg.output_proj_dim // 2, cfg.vocab_size)
        self.output_dropout = nn.Dropout(cfg.output_dropout)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        labels: torch.Tensor,
        enc_seq_len: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        shift_embeddings: bool = True,
        force_context: bool = False,
    ):
        """
        :param encoder_outputs: encoder outputs of shape [B,T,D], same for training and search
        :param labels:
            training: labels of shape [B,N]
            (greedy-)search: hypotheses last label as [B,1]
        :param enc_seq_len: encoder sequence lengths of shape [B,T], same for training and search
        :param state: decoder state
            training: Usually None, unless decoding should be initialized with a certain state (e.g. for context init)
            search: current state of the active hypotheses
        :param shift_embeddings: shift the embeddings by one position along U, padding with zero in front and drop last
            training: this should be "True", in order to start with a zero target embedding
            search: use True for the first step in order to start with a zero embedding, False otherwise
        :param force_context: encoder_outputs are treated directly as context input
        """
        if state is None:
            zeros = encoder_outputs.new_zeros((encoder_outputs.size(0), self.lstm_hidden_size))
            lstm_state = (zeros, zeros)
            att_context = encoder_outputs.new_zeros((encoder_outputs.size(0), encoder_outputs.size(2)))
            accum_att_weights = encoder_outputs.new_zeros((encoder_outputs.size(0), encoder_outputs.size(1), 1))
        else:
            lstm_state, att_context, accum_att_weights = state

        target_embeddings = self.target_embed(labels)  # [B,N,D]
        target_embeddings = self.target_embed_dropout(target_embeddings)

        if shift_embeddings:
            # pad for BOS and remove last token as this represents history and last token is not used
            target_embeddings = nn.functional.pad(target_embeddings, (0, 0, 1, 0), value=0)[:, :-1, :]  # [B,N,D]

        enc_ctx = self.enc_ctx(encoder_outputs)  # [B,T,D]
        enc_inv_fertility = nn.functional.sigmoid(self.inv_fertility(encoder_outputs))  # [B,T,1]

        num_steps = labels.size(1)  # N
        if force_context:
            assert encoder_outputs.shape[1] == num_steps
            accum_att_weights = None

        # collect for computing later the decoder logits outside the loop
        s_list = []
        att_context_list = []

        # decoder loop
        for step in range(num_steps):
            target_embed = target_embeddings[:, step, :]  # [B,D]
            if force_context:
                att_context = encoder_outputs[:, step]
            lstm_state = self.s(torch.cat([target_embed, att_context], dim=-1), lstm_state)
            lstm_out = lstm_state[0]
            s_transformed = self.s_transformed(lstm_out)  # project query
            s_list.append(lstm_out)

            # attention mechanism
            if not force_context:
                weight_feedback = self.weight_feedback(accum_att_weights)
                att_context, att_weights = self.attention(
                    key=enc_ctx,
                    value=encoder_outputs,
                    query=s_transformed,
                    weight_feedback=weight_feedback,
                    enc_seq_len=enc_seq_len,
                )
                accum_att_weights = accum_att_weights + att_weights * enc_inv_fertility * 0.5

            att_context_list.append(att_context)

        # output layer
        s_stacked = torch.stack(s_list, dim=1)  # [B,N,D]
        att_context_stacked = torch.stack(att_context_list, dim=1)  # [B,N,D]
        readout_in = self.readout_in(torch.cat([s_stacked, target_embeddings, att_context_stacked], dim=-1))  # [B,N,D]

        # maxout layer
        readout_in = readout_in.view(readout_in.size(0), readout_in.size(1), -1, 2)  # [B,N,D/2,2]
        readout, _ = torch.max(readout_in, dim=-1)  # [B,N,D/2]

        readout_drop = self.output_dropout(readout)
        decoder_logits = self.output(readout_drop)

        state = lstm_state, att_context, accum_att_weights

        return decoder_logits, state



class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **net_kwargs):
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

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerRelPosEncoderV1(cfg=conformer_config)
        self.decoder = AttentionLSTMDecoderWithForcedContext(cfg=self.cfg.decoder_config)

        self.num_output_linears = 1 if self.cfg.aux_ctc_loss_layers is None else len(self.cfg.aux_ctc_loss_layers)
        self.output_linears = nn.ModuleList([
            nn.Linear(conformer_size, self.cfg.label_target_size + 1)  # + CTC blank
            for _ in range(self.num_output_linears)
        ])
        self.output_dropout = BroadcastDropout(p=self.cfg.final_dropout,
                                               dropout_broadcast_axes=self.cfg.dropout_broadcast_axes)

        self.return_layers = self.cfg.aux_ctc_loss_layers or [self.cfg.num_layers - 1]
        self.scales = self.cfg.aux_ctc_loss_scales or [1.0]

        self.specaug_start_epoch = self.cfg.specauc_start_epoch
        self.label_smoothing = self.cfg.label_smoothing
        self.label_smoothing_start_epoch = self.cfg.label_smoothing_start_epoch

        self.mini_lstm_do_training = self.cfg.mini_lstm_do_training

        if self.mini_lstm_do_training:
            for param in self.parameters():
                param.requires_grad = False

        self.mini_lstm = torch.nn.LSTM(
            input_size=self.cfg.decoder_config.target_embed_dim,
            hidden_size=self.cfg.mini_lstm_hidden_size,
            batch_first=True
        )
        self.mini_lstm_ff = torch.nn.Linear(self.cfg.mini_lstm_hidden_size, self.cfg.decoder_config.encoder_dim)

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

        conformer_out_layers, out_mask = self.conformer(conformer_in, mask, return_layers=self.return_layers)
        log_probs_list = []
        conformer_out = None
        logits = None
        for i, (out_layer, scale) in enumerate(zip(conformer_out_layers, self.scales)):
            if scale == 0.0:
                continue
            conformer_out = self.output_dropout(out_layer)
            logits = self.output_linears[i](conformer_out)
            log_probs = torch.log_softmax(logits, dim=2)
            log_probs_list.append(log_probs)


        if encoder_only:
            encoder_seq_len = torch.sum(out_mask, dim=1)  # [B]
            return conformer_out, encoder_seq_len
        elif ctc_only:
            ctc_log_probs = torch.log_softmax(logits, dim=2)
            encoder_seq_len = torch.sum(out_mask, dim=1)  # [B]

            return ctc_log_probs, encoder_seq_len
        elif not do_search:
            if self.mini_lstm_do_training:
                embeddings = self.decoder.target_embed(bpe_labels)
                lstm_out, _ = self.mini_lstm(embeddings)
                fake_context = self.mini_lstm_ff(lstm_out)
                # rotate by one and fill front with zeros
                fake_context_shifted = nn.functional.pad(fake_context, (0, 0, 1, 0), value=0)[:, :-1, :]
                conformer_out = fake_context_shifted

            decoder_logits, state = self.decoder(
                conformer_out, bpe_labels, audio_features_len.to(device=conformer_out.device), force_context=self.mini_lstm_do_training,
            )
            encoder_seq_len = torch.sum(out_mask, dim=1)  # [B]

            return decoder_logits, state, conformer_out, encoder_seq_len, log_probs_list
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

    decoder_logits, state, encoder_outputs, audio_features_len, logprobs_list = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
        bpe_labels=labels,
    )

    num_phonemes = torch.sum(labels_len)
    for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
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
        run_ctx.mark_as_loss(name=f"ctc_loss_layer{layer_index + 1}", loss=ctc_loss, scale=scale,
                             inv_norm_factor=num_phonemes)

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


    ce_ls_loss = nn.functional.cross_entropy(
        decoder_logits.transpose(1, 2), targets_masked.long(), reduction="sum", label_smoothing=label_smoothing,
    )  # [B,N]

    ce_loss = nn.functional.cross_entropy(
        decoder_logits.transpose(1, 2), targets_masked.long(), reduction="sum", label_smoothing=0.0,
    )  # [B,N]

    num_labels = torch.sum(labels_len)

    run_ctx.mark_as_loss(name="decoder_ce_smoothed", scale=1.0, loss=ce_ls_loss, inv_norm_factor=num_labels)
    run_ctx.mark_as_loss(name="decoder_ce", scale=0.0, loss=ce_loss, inv_norm_factor=num_labels)
