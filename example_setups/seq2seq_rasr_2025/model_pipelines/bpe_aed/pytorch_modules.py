__all__ = [
    "AdditiveAttentionConfig",
    "AttentionLSTMDecoderV1Config",
    "AEDConfig",
    "AdditiveAttention",
    "ZoneoutLSTMCell",
    "AttentionLSTMDecoderV1",
    "AEDModel",
    "AEDEncoder",
    "AEDScorer",
    "AEDStateInitializer",
    "AEDStateUpdater",
]

from dataclasses import dataclass
from typing import Tuple

import torch
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import ConformerRelPosEncoderV1, ConformerRelPosEncoderV1Config
from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length

from ..common.pytorch_modules import SpecaugmentByLengthConfig, lengths_to_padding_mask


@dataclass
class AdditiveAttentionConfig(ModelConfiguration):
    attention_dim: int
    att_weights_dropout: float


@dataclass
class AttentionLSTMDecoderV1Config(ModelConfiguration):
    encoder_dim: int
    vocab_size: int
    target_embed_dim: int
    target_embed_dropout: float
    lstm_hidden_size: int
    zoneout_drop_h: float
    zoneout_drop_c: float
    attention_cfg: AdditiveAttentionConfig
    output_proj_dim: int
    output_dropout: float


@dataclass
class AEDConfig(ModelConfiguration):
    logmel_cfg: LogMelFeatureExtractionV1Config
    specaug_cfg: SpecaugmentByLengthConfig
    conformer_cfg: ConformerRelPosEncoderV1Config
    decoder_config: AttentionLSTMDecoderV1Config
    enc_dim: int
    final_dropout: float
    label_target_size: int


class AdditiveAttention(torch.nn.Module):
    """
    Additive attention mechanism. This is defined as:
        energies = v^T * tanh(h + s + beta)  where beta is weight feedback information
        weights = softmax(energies)
        context = sum_t weights_t * h_t
    """

    def __init__(self, cfg: AdditiveAttentionConfig):
        super().__init__()
        self.linear = torch.nn.Linear(cfg.attention_dim, 1, bias=False)
        self.att_weights_drop = torch.nn.Dropout(cfg.att_weights_dropout)

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
        energies = self.linear(torch.nn.functional.tanh(key + query.unsqueeze(1) + weight_feedback))  # [B,T,1]
        time_arange = torch.arange(energies.size(1), device=energies.device)  # [T]
        seq_len_mask = torch.less(time_arange[None, :], enc_seq_len[:, None])  # [B,T]
        energies = torch.where(seq_len_mask.unsqueeze(2), energies, energies.new_tensor(-float("inf")))
        weights = torch.nn.functional.softmax(energies, dim=1)  # [B,T,1]
        weights = self.att_weights_drop(weights)
        context = torch.bmm(weights.transpose(1, 2), value)  # [B,1,D_v]
        context = context.reshape(context.size(0), -1)  # [B,D_v]
        return context, weights


class ZoneoutLSTMCell(torch.nn.Module):
    """
    Wrap an LSTM cell with Zoneout regularization (https://arxiv.org/abs/1606.01305)
    """

    def __init__(self, cell: torch.nn.RNNCellBase, zoneout_h: float, zoneout_c: float):
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
            h, c = self.cell.forward(inputs)
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


class AttentionLSTMDecoderV1(torch.nn.Module):
    """
    Single-headed Attention decoder with additive attention mechanism.
    """

    def __init__(self, cfg: AttentionLSTMDecoderV1Config):
        super().__init__()

        self.target_embed_dim = cfg.target_embed_dim
        self.target_embed = torch.nn.Embedding(
            num_embeddings=cfg.vocab_size, embedding_dim=cfg.target_embed_dim, padding_idx=0
        )
        self.target_embed_dropout = torch.nn.Dropout(cfg.target_embed_dropout)

        lstm_cell = torch.nn.LSTMCell(
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

        self.attention_dim = cfg.attention_cfg.attention_dim
        self.s_transformed = torch.nn.Linear(cfg.lstm_hidden_size, self.attention_dim, bias=False)  # query

        # for attention
        self.enc_ctx = torch.nn.Linear(cfg.encoder_dim, self.attention_dim)
        self.attention = AdditiveAttention(cfg.attention_cfg)

        # for weight feedback
        self.inv_fertility = torch.nn.Linear(cfg.encoder_dim, 1, bias=False)  # followed by sigmoid
        self.weight_feedback = torch.nn.Linear(1, self.attention_dim, bias=False)

        self.readout_in = torch.nn.Linear(
            cfg.lstm_hidden_size + cfg.target_embed_dim + cfg.encoder_dim, cfg.output_proj_dim
        )
        assert cfg.output_proj_dim % 2 == 0, "output projection dimension must be even for the MaxOut op of 2 pieces"
        self.output = torch.nn.Linear(cfg.output_proj_dim // 2, cfg.vocab_size)
        self.output_dropout = torch.nn.Dropout(cfg.output_dropout)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        labels: torch.Tensor,
        enc_seq_len: torch.Tensor,
    ):
        """
        :param encoder_outputs: encoder outputs of shape [B,T,D], same for training and search
        :param labels: labels of shape [B,N]
        :param enc_seq_len: encoder sequence lengths of shape [B,T], same for training and search
        :return
            decoder_logits as [B, T, #vocab] or [B, 1, #vocab]

        """
        assert len(encoder_outputs.size()) == 3
        assert len(labels.size()) == 2
        assert encoder_outputs.shape[0] == labels.shape[0]

        zeros = encoder_outputs.new_zeros((encoder_outputs.size(0), self.lstm_hidden_size))
        lstm_state = (zeros, zeros)
        att_context = encoder_outputs.new_zeros((encoder_outputs.size(0), encoder_outputs.size(2)))
        accum_att_weights = encoder_outputs.new_zeros((encoder_outputs.size(0), encoder_outputs.size(1), 1))

        target_embeddings = self.target_embed(labels)  # [B,N,D]
        target_embeddings = self.target_embed_dropout(target_embeddings)

        target_embeddings = torch.nn.functional.pad(target_embeddings, (0, 0, 1, 0), value=0)[:, :-1, :]  # [B,N,D]

        enc_ctx = self.enc_ctx(encoder_outputs)  # [B,T,D]
        enc_inv_fertility = torch.nn.functional.sigmoid(self.inv_fertility(encoder_outputs))  # [B,T,1]

        num_steps = labels.size(1)  # N

        # collect for computing later the decoder logits outside the loop
        s_list = []
        att_context_list = []

        # decoder loop
        for step in range(num_steps):
            history_embed = target_embeddings[:, step, :]  # [B,D]

            lstm_state = self.s(torch.cat([history_embed, att_context], dim=-1), lstm_state)
            lstm_out = lstm_state[0]
            s_transformed = self.s_transformed(lstm_out)  # project query
            s_list.append(lstm_out)

            # attention mechanism
            weight_feedback = self.weight_feedback(accum_att_weights)
            att_context, att_weights = self.attention.forward(
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

        readout = self.output_dropout.forward(readout)
        decoder_logits = self.output.forward(readout)

        return decoder_logits


class AEDModel(torch.nn.Module):
    def __init__(self, cfg: AEDConfig, **net_kwargs):
        super().__init__()
        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.conformer = ConformerRelPosEncoderV1(cfg.conformer_cfg)
        self.enc_dim = cfg.enc_dim
        self.label_target_size = cfg.label_target_size
        self.final_linear = torch.nn.Linear(cfg.enc_dim, cfg.label_target_size + 1)  # + CTC blank
        self.final_dropout = torch.nn.Dropout(p=cfg.final_dropout)
        self.decoder = AttentionLSTMDecoderV1(cfg=cfg.decoder_config)

    def forward(
        self,
        audio_samples: torch.Tensor,
        audio_samples_size: torch.Tensor,
        bpe_labels: torch.Tensor,
    ):
        squeezed_features = torch.squeeze(audio_samples, dim=-1)

        with torch.no_grad():
            audio_features, audio_features_size = self.feature_extraction.forward(
                squeezed_features, audio_samples_size
            )  # [B, T, F], [B]

            from returnn.torch.context import get_run_ctx  # type: ignore

            if self.training and get_run_ctx().epoch >= self.specaug_config.start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features=audio_features,
                    time_min_num_masks=self.specaug_config.time_min_num_masks,
                    time_max_mask_per_n_frames=self.specaug_config.time_max_mask_per_n_frames,
                    time_mask_max_size=self.specaug_config.time_mask_max_size,
                    freq_min_num_masks=self.specaug_config.freq_min_num_masks,
                    freq_max_num_masks=self.specaug_config.freq_max_num_masks,
                    freq_mask_max_size=self.specaug_config.freq_mask_max_size,
                )  # [B, T, F]
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = lengths_to_padding_mask(audio_features_size)

        conformer_out, out_mask = self.conformer.forward(conformer_in, mask)
        conformer_out = conformer_out[-1]

        conformer_out = self.final_dropout.forward(conformer_out)
        logits = self.final_linear.forward(conformer_out)

        ctc_log_probs = torch.log_softmax(logits, dim=2)

        decoder_logits = self.decoder.forward(
            conformer_out, bpe_labels, audio_features_size.to(device=conformer_out.device)
        )
        encoder_seq_len = torch.sum(out_mask, dim=1)

        return decoder_logits, encoder_seq_len, ctc_log_probs


class AEDEncoder(AEDModel):
    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T', F]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, T, E+A+1]
        squeezed_features = torch.squeeze(audio_samples, dim=-1)

        with torch.no_grad():
            audio_features, audio_features_size = self.feature_extraction.forward(
                squeezed_features, audio_samples_size
            )  # [B, T, F], [B]

            audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = lengths_to_padding_mask(audio_features_size)

        conformer_out, _ = self.conformer.forward(conformer_in, mask)
        conformer_out = conformer_out[-1]
        enc_ctx = self.decoder.enc_ctx.forward(conformer_out)  # [B, T, A]
        enc_inv_fertility = torch.nn.functional.sigmoid(self.decoder.inv_fertility.forward(conformer_out))  # [B,T,1]
        return torch.cat([conformer_out, enc_ctx, enc_inv_fertility], dim=2)  # [B, T, E+A+1]


class AEDScorer(AEDModel):
    def forward(
        self,
        token_embedding: torch.Tensor,  # [B, M]
        lstm_state_h: torch.Tensor,  # [B, H]
        att_context: torch.Tensor,  # [B, E]
    ) -> torch.Tensor:
        readout_in = self.decoder.readout_in.forward(
            torch.cat([lstm_state_h, token_embedding, att_context], dim=1)
        )  # [B, D]

        # maxout layer
        readout_in = readout_in.view(readout_in.size(0), -1, 2)  # [B, D/2, 2]
        readout, _ = torch.max(readout_in, dim=2)  # [B, D/2]

        decoder_logits = self.decoder.output.forward(readout)  # [B, V]
        scores = -torch.log_softmax(decoder_logits, dim=1)  # [B, V]

        return scores


class AEDStateInitializer(AEDModel):
    def forward(
        self,
        encoder_states: torch.Tensor,  # [1, T, E + A]
        encoder_states_size: torch.Tensor,  # [1]
    ) -> Tuple[torch.Tensor, ...]:
        encoder_out, enc_ctx, enc_inv_fertility = torch.split(
            encoder_states,
            [
                self.enc_dim,
                self.decoder.attention_dim,
                1,
            ],
            dim=2,
        )  # [1, T, E], [1, T, A], [1, T, 1]

        lstm_state_h = encoder_out.new_zeros(1, self.decoder.lstm_hidden_size)  # [1, H]
        lstm_state_c = encoder_out.new_zeros(1, self.decoder.lstm_hidden_size)  # [1, H]
        att_context = encoder_out.new_zeros(1, encoder_out.size(2))  # [1, E]
        accum_att_weights = encoder_out.new_zeros((1, encoder_out.size(1), 1))  # [1, T, 1]

        token_embedding = encoder_out.new_zeros(1, self.decoder.target_embed_dim)  # [1, M]

        new_lstm_state_h, new_lstm_state_c = self.decoder.s.forward(
            torch.cat([token_embedding, att_context], dim=1),
            (lstm_state_h, lstm_state_c),
        )  # [1, H], [1, H]
        s_transformed = self.decoder.s_transformed.forward(new_lstm_state_h)  # [1, A]

        weight_feedback = self.decoder.weight_feedback.forward(accum_att_weights)  # [1, T, A]

        new_att_context, new_att_weights = self.decoder.attention(
            key=enc_ctx,
            value=encoder_out,
            query=s_transformed,
            weight_feedback=weight_feedback,
            enc_seq_len=encoder_states_size,
        )  # [1, E], [1, T, 1]

        new_accum_att_weights = accum_att_weights + new_att_weights * enc_inv_fertility * 0.5  # [1, T, 1]

        return token_embedding, new_lstm_state_h, new_lstm_state_c, new_att_context, new_accum_att_weights


class AEDStateUpdater(AEDModel):
    def forward(
        self,
        encoder_states: torch.Tensor,  # [T, E + A]
        encoder_states_size: torch.Tensor,  # []
        token: torch.Tensor,  # [1j
        lstm_state_h: torch.Tensor,  # [1, H]
        lstm_state_c: torch.Tensor,  # [1, H]
        att_context: torch.Tensor,  # [1, E]
        accum_att_weights: torch.Tensor,  # [1, T, 1]
    ) -> Tuple[torch.Tensor, ...]:
        encoder_out, enc_ctx, enc_inv_fertility = torch.split(
            encoder_states,
            [
                self.enc_dim,
                self.decoder.attention_dim,
                1,
            ],
            dim=2,
        )  # [1, T, E], [1, T, A], [1, T, 1]

        new_token_embedding = self.decoder.target_embed.forward(token)  # [1, M]

        new_lstm_state_h, new_lstm_state_c = self.decoder.s.forward(
            torch.cat([new_token_embedding, att_context], dim=1),
            (lstm_state_h, lstm_state_c),
        )  # [1, H], [1, H]
        s_transformed = self.decoder.s_transformed.forward(new_lstm_state_h)  # [1, A]

        weight_feedback = self.decoder.weight_feedback.forward(accum_att_weights)  # [1, T, A]

        new_att_context, new_att_weights = self.decoder.attention(
            key=enc_ctx,
            value=encoder_out,
            query=s_transformed,
            weight_feedback=weight_feedback,
            enc_seq_len=encoder_states_size,
        )  # [1, E], [1, T, 1]

        new_accum_att_weights = accum_att_weights + new_att_weights * enc_inv_fertility * 0.5  # [T, 1]

        return new_token_embedding, new_lstm_state_h, new_lstm_state_c, new_att_context, new_accum_att_weights
