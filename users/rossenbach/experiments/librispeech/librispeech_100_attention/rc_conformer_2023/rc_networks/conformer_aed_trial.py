"""
Implementation of the CTC NAR Network
"""
from returnn_common import nn
from returnn_common.nn.conformer import ConformerEncoder, ConformerConvSubsample, ConformerEncoderLayer
from typing import Tuple, Union, Optional, Sequence

from .rescale_tests import BF16Linear, Memristor

from .features import LogMelFeatureExtractor


class Conv6Subsampling(nn.Module):

    def __init__(
            self,
            in_dim: nn.Dim,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.feature_pool_dim = nn.FeatureDim("conv6sub_conv1_feat", 32)
        self.time_pool1_dim = nn.FeatureDim("conv6sub_conv2_feat", 64)
        self.time_pool2_dim = nn.FeatureDim("conv6sub_conv3_feat", 64)
        self.out_dim = nn.FeatureDim("subsampling_out", dimension=(in_dim.dimension // 2) * 64)

        self.expanded_feature_dim = nn.FeatureDim("conv6sub_new_feature", 1)

        self.feature_conv = nn.Conv2d(
            in_dim=self.expanded_feature_dim,
            out_dim=self.feature_pool_dim,
            filter_size=(3, 3),
            padding="same",
        )

        self.time1_conv = nn.Conv2d(
            in_dim=self.feature_pool_dim,
            out_dim=self.time_pool1_dim,
            filter_size=(3, 3),
            padding="same",
            strides=(3, 1),
        )

        self.time2_conv = nn.Conv2d(
            in_dim=self.time_pool1_dim,
            out_dim=self.time_pool2_dim,
            filter_size=(3, 3),
            padding="same",
            strides=(2, 1),
        )

    def __call__(self, features: nn.Tensor, in_spatial_dim: nn.Dim):
        feature_spatial_dim = nn.SpatialDim("conv6sub_feature_spatial")

        expanded_features = nn.split_dims(features, axis=self.in_dim,
                                          dims=[feature_spatial_dim, self.expanded_feature_dim])

        conv1, [time_dim, feature_spatial_dim] = self.feature_conv(
            expanded_features,
            in_spatial_dims=[in_spatial_dim, feature_spatial_dim]
        )

        pool1, feature_spatial_dim = nn.pool1d(conv1, mode="max", pool_size=2, padding="same",
                                               in_spatial_dim=feature_spatial_dim)
        conv2, [time_dim, feature_spatial_dim] = self.time1_conv(pool1, in_spatial_dims=[time_dim, feature_spatial_dim])
        conv3, [time_dim, feature_spatial_dim] = self.time2_conv(conv2, in_spatial_dims=[time_dim, feature_spatial_dim])

        final, _ = nn.merge_dims(conv3, axes=[feature_spatial_dim, conv3.feature_dim], out_dim=self.out_dim)

        return final, time_dim


class ConvFeedbackMLPAttention(nn.Module):

    def __init__(self, query_feature_dim: nn.Dim, key_feature_dim: nn.Dim, value_feature_dim: nn.Dim):
        super().__init__()
        self.attention_dim = nn.FeatureDim("attention_dim", 1024)
        self.energy_weights_dim = nn.FeatureDim("energy_weights_dim", 1)
        self.query_feature_dim = query_feature_dim
        self.key_feature_dim = key_feature_dim
        self.value_feature_dim = value_feature_dim

        self.encoder_output = None
        self.encoder_time_dim = None
        self.fertility = None

        self.W_q = nn.Linear(in_dim=self.query_feature_dim, out_dim=self.attention_dim)
        self.W_k = nn.Linear(in_dim=self.key_feature_dim, out_dim=self.attention_dim)
        self.v_transposed = nn.Linear(in_dim=self.attention_dim, out_dim=self.energy_weights_dim)
        self.feedback_transform = nn.Linear(in_dim=self.energy_weights_dim, out_dim=self.attention_dim)
        self.fertility_transform = nn.Linear(in_dim=value_feature_dim, out_dim=self.energy_weights_dim)

    def initialize(self, encoder_output: nn.Tensor, encoder_time_dim: nn.Dim):
        self.encoder_time_dim = encoder_time_dim
        self.encoder_output = encoder_output
        self.fertility = self.fertility_transform(self.encoder_output)

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        state = nn.LayerState()
        state.feedback_sum = nn.constant(
            value=0,
            shape=list(batch_dims) + [self.encoder_time_dim, self.energy_weights_dim],
            dtype="float32"
        )
        return state

    def __call__(
            self, query: nn.Tensor, key: nn.Tensor, value: nn.Tensor, key_value_time: nn.Dim, state: nn.LayerState
    ) -> Tuple[nn.Tensor, nn.LayerState]:
        q_prime = self.W_q(query)
        k_prime = self.W_k(key)
        feedback = self.feedback_transform(state.feedback_sum)
        attention_hidden = q_prime + k_prime + feedback * 0.5
        energy = self.v_transposed(attention_hidden)  # {B, kv_time, (q_time), att}
        alpha = nn.softmax(energy, axis=key_value_time)
        state.feedback_sum = state.feedback_sum + alpha * 0.5 * self.fertility
        c = nn.dot(alpha, value, reduce=key_value_time)
        c, _ = nn.merge_dims(source=c, axes=[alpha.feature_dim, value.feature_dim], out_dim=value.feature_dim)
        return c, state


class AttentionLSTMDecoder(nn.Module):

    def __init__(self, batch_dim: nn.Dim, encoder_feature_dim: nn.Dim, target_feature_dim: nn.Dim):
        super().__init__()
        self.batch_dim = batch_dim

        # new feature dims
        self.decoder_feature_dim = nn.FeatureDim("decoder_dim", 1024)
        self.readout_feature_dim = nn.FeatureDim("readout_dim", 1024)
        self.readout_reduce_feature_dim = self.readout_feature_dim // 2
        self.feedback_feature_dim = nn.FeatureDim("feedback_dim", 512)
        self.encoder_feature_dim = encoder_feature_dim
        self.target_dim = target_feature_dim

        self.encoder_output = None
        self.encoder_time = None

        self.target_embedding = nn.Embedding(in_dim=self.target_dim, out_dim=self.feedback_feature_dim)
        self.mlp_attention = ConvFeedbackMLPAttention(
            query_feature_dim=self.decoder_feature_dim,
            key_feature_dim=encoder_feature_dim,
            value_feature_dim=encoder_feature_dim,
        )
        self.lstm = nn.LSTM(in_dim=encoder_feature_dim + self.feedback_feature_dim, out_dim=self.decoder_feature_dim)
        self.readout = nn.Linear(in_dim=encoder_feature_dim + self.decoder_feature_dim + self.feedback_feature_dim,
                                 out_dim=self.readout_feature_dim)
        self.logit_linear = nn.Linear(in_dim=self.readout_reduce_feature_dim, out_dim=self.target_dim)

    def max_seq_len(self) -> nn.Tensor:
        return nn.constant(100)

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        state = nn.LayerState()
        state.lstm_state = self.lstm.default_initial_state(batch_dims=batch_dims)
        state.context_state = nn.constant(0, shape=list(batch_dims) + [self.encoder_feature_dim], dtype="float32")
        state.attention_state = self.mlp_attention.default_initial_state(batch_dims=batch_dims)
        return state

    def initialize(self, encoder_output: nn.Tensor, encoder_time_dim: nn.Dim):
        self.encoder_output = encoder_output
        self.encoder_time = encoder_time_dim
        self.mlp_attention.initialize(encoder_output=encoder_output, encoder_time_dim=encoder_time_dim)

    def __call__(self, prev_target: nn.Tensor, *, state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
        target_embed = self.target_embedding(prev_target)

        output, state.lstm_state = self.lstm(
            source=nn.concat_features(state.context_state, target_embed),
            spatial_dim=nn.single_step_dim,
            state=state.lstm_state
        )

        context, state.attention_state = self.mlp_attention(
            query=output,
            key=self.encoder_output,
            value=self.encoder_output,
            key_value_time=self.encoder_time,
            state=state.attention_state,
        )
        state.context_state = context

        readout = self.readout(nn.concat_features(context, output, target_embed))
        reduced_readout = nn.reduce_out(readout, mode="max", num_pieces=2)
        logits = self.logit_linear(reduced_readout)

        return logits, state


class ConformerAEDModel(nn.Module):
    """
      NAR TTS Model from Timur Schümann implemented in returnn common
      """

    def __init__(
            self,
            bpe_feature_dim: nn.Dim,
            batch_dim: nn.Dim,
            num_layers=12,
            **kwargs,
    ):
        super().__init__()
        self.feature_extractor = LogMelFeatureExtractor()
        self.subsampling = Conv6Subsampling(in_dim=self.feature_extractor.out_feature_dim)

        self.encoder = ConformerEncoder(
            in_dim=self.feature_extractor.out_feature_dim,
            num_layers=num_layers,
            input_layer=self.subsampling,
            ff_activation=nn.swish,
            input_dropout=0.2,
            dropout=0.2,
            conv_norm=nn.BatchNorm,
            num_heads=8,
            att_dropout=0.2,
        )

        self.ctc_out_dim = bpe_feature_dim + 1
        self.bpe_feature_dim = bpe_feature_dim
        # self.ctc_linear = BF16Linear(self.encoder.out_dim, out_dim=self.ctc_out_dim)
        self.ctc_linear = nn.Linear(self.encoder.out_dim, out_dim=self.ctc_out_dim)
        # self.ctc_linear = Memristor(self.encoder.out_dim, out_dim=self.ctc_out_dim)

        self.decoder = AttentionLSTMDecoder(
            batch_dim=batch_dim,
            encoder_feature_dim=self.encoder.out_dim,
            target_feature_dim=bpe_feature_dim,
        )

    def __call__(self, audio_features: nn.Tensor, audio_time: nn.Dim, bpe_labels: nn.Tensor, bpe_time: nn.Dim):
        _, log_mel_features, logmel_time_dim = self.feature_extractor(audio_features, audio_time)

        encoder_output, encoder_time_dim = self.encoder(
            source=log_mel_features,
            in_spatial_dim=logmel_time_dim
        )

        ctc_logits = self.ctc_linear(encoder_output)
        ctc = nn.ctc_loss(logits=ctc_logits, targets=bpe_labels, blank_index=self.ctc_out_dim.dimension - 1)
        ctc.mark_as_loss(name="ctc", custom_inv_norm_factor=nn.length(dim=bpe_time))

        self.decoder.initialize(encoder_output=encoder_output, encoder_time_dim=encoder_time_dim)
        self.decoder.target_spatial_dim = bpe_time

        batch_dims = audio_features.remaining_dims(remove=[audio_time, audio_features.feature_dim])

        loop = nn.Loop(axis=bpe_time)
        loop.max_seq_len = self.decoder.max_seq_len()
        loop.state.decoder = self.decoder.default_initial_state(batch_dims=batch_dims)
        loop.state.target = nn.constant(0, shape=batch_dims, sparse_dim=self.bpe_feature_dim)
        with loop:
            logits, loop.state.decoder = self.decoder(loop.state.target, state=loop.state.decoder)
            loop.state.target = loop.unstack(bpe_labels)
            logits = loop.stack(logits)

        ce_loss = nn.cross_entropy(target=bpe_labels, estimated=logits, estimated_type="logits")
        ce_loss.mark_as_loss("ce_loss", custom_inv_norm_factor=nn.length(bpe_time))

        return logits


def construct_network(
        epoch: int,
        audio_features: nn.Data,
        bpe_labels: nn.Data,
        **kwargs
):
    net = ConformerAEDModel(
        bpe_feature_dim=bpe_labels.sparse_dim,
        batch_dim=audio_features.get_batch_dim_tag(),
        audio_feature_dim=audio_features.dim_tags[audio_features.feature_dim_axis],
        **kwargs
    )

    out = net(
        audio_features=nn.get_extern_data(audio_features),
        audio_time=audio_features.dim_tags[audio_features.time_dim_axis],
        bpe_labels=nn.get_extern_data(bpe_labels),
        bpe_time=bpe_labels.dim_tags[bpe_labels.time_dim_axis],
    )
    out.mark_as_default_output()
    return net
