"""
First AED Prototype based on the CTC setup

v2: with added ctc loss scale, was incorrect in v1
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


from returnn.torch.context import get_run_ctx

from .aed_prototype_v2_cfg import ModelConfig, AdditiveAttentionConfig, AttentionLSTMDecoderV1Config


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


class AdditiveAttention(nn.Module):
    """
    Additive attention mechanism. This is defined as:
        energies = v^T * tanh(h + s + beta)  where beta is weight feedback information
        weights = softmax(energies)
        context = sum_t weights_t * h_t
    """

    def __init__(self, cfg: AdditiveAttentionConfig):
        super().__init__()
        self.linear = nn.Linear(cfg.attention_dim, 1, bias=False)
        self.att_weights_drop = nn.Dropout(cfg.att_weights_dropout)

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        weight_feedback: torch.Tensor,
        enc_seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param key: encoder keys of shape [B,T,D_k]
        :param value: encoder values of shape [B,T,D_v]
        :param query: query of shape [B,D_k]
        :param weight_feedback: shape is [B,T,D_k]
        :param enc_seq_len: encoder sequence lengths [B]
        :return: attention context [B,D_v], attention weights [B,T,1]
        """
        # all inputs are already projected
        energies = self.linear(nn.functional.tanh(key + query.unsqueeze(1) + weight_feedback))  # [B,T,1]
        time_arange = torch.arange(energies.size(1), device=energies.device)  # [T]
        seq_len_mask = torch.less(time_arange[None, :], enc_seq_len[:, None])  # [B,T]
        energies = torch.where(seq_len_mask.unsqueeze(2), energies, energies.new_tensor(-float("inf")))
        weights = nn.functional.softmax(energies, dim=1)  # [B,T,1]
        weights = self.att_weights_drop(weights)
        context = torch.bmm(weights.transpose(1, 2), value)  # [B,1,D_v]
        context = context.reshape(context.size(0), -1)  # [B,D_v]
        return context, weights


class ZoneoutLSTMCell(nn.Module):
    """
    Wrap an LSTM cell with Zoneout regularization (https://arxiv.org/abs/1606.01305)
    """

    def __init__(self, cell: nn.RNNCellBase, zoneout_h: float, zoneout_c: float):
        """
        :param cell: LSTM cell
        :param zoneout_h: zoneout drop probability for hidden state
        :param zoneout_c: zoneout drop probability for cell state
        """
        super().__init__()
        self.cell = cell
        assert 0.0 <= zoneout_h <= 1.0 and 0.0 <= zoneout_c <= 1.0, "Zoneout drop probability must be in [0, 1]"
        self.zoneout_h = zoneout_h
        self.zoneout_c = zoneout_c

    def forward(
        self, inputs: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            h, c = self.cell(inputs)
        prev_h, prev_c = state
        h = self._zoneout(prev_h, h, self.zoneout_h)
        c = self._zoneout(prev_c, c, self.zoneout_c)
        return h, c

    def _zoneout(self, prev_state: torch.Tensor, curr_state: torch.Tensor, factor: float):
        """
        Apply Zoneout.
        :param prev: previous state tensor
        :param curr: current state tensor
        :param factor: drop probability
        """
        if factor == 0.0:
            return curr_state
        if self.training:
            mask = curr_state.new_empty(size=curr_state.size()).bernoulli_(factor)
            return mask * prev_state + (1 - mask) * curr_state
        else:
            return factor * prev_state + (1 - factor) * curr_state


class AttentionLSTMDecoderV1(nn.Module):
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
        shift_embeddings=True,
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
        :return
            decoder_logits as [B, T, #vocab] or [B, 1, #vocab]

        """
        assert len(encoder_outputs.size()) == 3
        assert len(labels.size()) == 2
        assert encoder_outputs.shape[0] == labels.shape[0]

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

        # collect for computing later the decoder logits outside the loop
        s_list = []
        att_context_list = []

        # decoder loop
        for step in range(num_steps):
            target_embed = target_embeddings[:, step, :]  # [B,D]

            lstm_state = self.s(torch.cat([target_embed, att_context], dim=-1), lstm_state)
            lstm_out = lstm_state[0]
            s_transformed = self.s_transformed(lstm_out)  # project query
            s_list.append(lstm_out)

            # attention mechanism
            weight_feedback = self.weight_feedback(accum_att_weights)
            att_context, att_weights = self.attention(
                key=enc_ctx,
                value=encoder_outputs,
                query=s_transformed,
                weight_feedback=weight_feedback,
                enc_seq_len=enc_seq_len,
            )
            att_context_list.append(att_context)
            accum_att_weights = accum_att_weights + att_weights * enc_inv_fertility * 0.5

        # output layer
        s_stacked = torch.stack(s_list, dim=1)  # [B,N,D]
        att_context_stacked = torch.stack(att_context_list, dim=1)  # [B,N,D]
        readout_in = self.readout_in(torch.cat([s_stacked, target_embeddings, att_context_stacked], dim=-1))  # [B,N,D]

        # maxout layer
        readout_in = readout_in.view(readout_in.size(0), readout_in.size(1), -1, 2)  # [B,N,D/2,2]
        readout, _ = torch.max(readout_in, dim=-1)  # [B,N,D/2]

        readout = self.output_dropout(readout)
        decoder_logits = self.output(readout)

        state = lstm_state, att_context, accum_att_weights

        return decoder_logits, state


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


    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
        bpe_labels: Optional[torch.Tensor],
        do_search: bool = False,
        encoder_only: bool = False,
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

    ce_loss = nn.functional.cross_entropy(
        decoder_logits.transpose(1, 2), targets_masked.long(), reduction="sum"
    )  # [B,N]

    num_labels = torch.sum(labels_len)

    run_ctx.mark_as_loss(name="decoder_ce", loss=ce_loss, inv_norm_factor=num_labels)
