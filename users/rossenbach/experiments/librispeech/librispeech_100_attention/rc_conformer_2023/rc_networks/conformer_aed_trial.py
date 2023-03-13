"""
Implementation of the CTC NAR Network
"""
from returnn_common import nn
from returnn_common.nn.conformer import ConformerEncoder, ConformerConvSubsample, ConformerEncoderLayer
from typing import Tuple, Union, Optional, Sequence

from .ar_decoder import IDecoder

from .features import LogMelFeatureExtractor


class Conv6Subsampling(nn.Module):

    def __init__(
            self,
            in_dim: nn.Dim,
    ):
        self.in_dim = in_dim
        self.feature_pool_dim = nn.FeatureDim("conv6sub_conv1_feat", 32)
        self.time_pool1_dim = nn.FeatureDim("conv6sub_conv2_feat", 64)
        self.time_pool2_dim = nn.FeatureDim("conv6sub_conv3_feat", 64)
        self.out_dim = nn.FeatureDim("subsampling_out", dimension=(in_dim.dimension//2)*64)

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

        expanded_features = nn.split_dims(features, axis=self.in_dim, dims=[feature_spatial_dim, self.expanded_feature_dim])

        conv1, [time_dim, feature_spatial_dim] = self.feature_conv(
            expanded_features,
            in_spatial_dims=[in_spatial_dim, self.in_dim]
        )

        pool1, feature_spatial_dim = nn.pool1d(conv1, mode="max", pool_size=2, padding="same", in_spatial_dim=feature_spatial_dim)
        conv2, [time_dim, feature_spatial_dim] = self.time1_conv(pool1, in_spatial_dims=[time_dim, feature_spatial_dim])
        conv3, [time_dim, feature_spatial_dim] = self.time2_conv(conv2, in_spatial_dims=[time_dim, feature_spatial_dim])

        final, _ = nn.merge_dims(conv3, axes=[feature_spatial_dim, conv3.feature_dim], out_dim=self.out_dim)

        return final, time_dim




class ConformerAEDModel(nn.Module):
    """
      NAR TTS Model from Timur Schümann implemented in returnn common
      """

    def __init__(
            self,
            bpe_size: nn.Dim,
            **kwargs,
    ):

        self.feature_extractor = LogMelFeatureExtractor()
        self.subsampling = Conv6Subsampling(in_dim=self.feature_extractor.out_feature_dim)

        self.encoder = ConformerEncoder(
            in_dim=self.feature_extractor.out_feature_dim,
            num_layers=12,
            input_layer=self.subsampling,
            ff_activation=nn.swish,
            input_dropout=0.2,
            dropout=0.2,
            conv_norm=nn.BatchNorm,
            num_heads=8,
            att_dropout=0.2,
        )

        self.ctc_out_dim = bpe_size + 1
        self.ctc_linear = nn.Linear(self.encoder.out_dim, out_dim=self.ctc_out_dim)


    def __call__(self, audio_features: nn.Tensor, audio_time: nn.Dim, bpe_labels: nn.Tensor, bpe_time: nn.Dim):

        _, log_mel_features, logmel_time_dim = self.feature_extractor(audio_features, audio_time)

        encoder_output, encoder_time_dim = self.encoder(
            source=log_mel_features,
            in_spatial_dim=logmel_time_dim
        )

        ctc_logits = self.ctc_linear(encoder_output)
        ctc = nn.ctc_loss(logits=ctc_logits, targets=bpe_labels, blank_index=self.ctc_out_dim.dimension - 1)
        ctc.mark_as_loss(name="ctc", custom_inv_norm_factor=nn.length(dim=bpe_time))

        return encoder_output


def construct_network(
        epoch: int,
        audio_features: nn.Data,  # phoneme labels
        bpe_labels: nn.Data,
        **kwargs
):
    net = ConformerAEDModel(
        bpe_size=bpe_labels.sparse_dim,
        **kwargs
    )

    out = net(
        audio_features=nn.get_extern_data(audio_features),
        audio_time=audio_features.dim_tags[audio_features.time_dim_axis],
        bpe_labels=nn.get_extern_data(bpe_labels),
        bpe_time=bpe_labels.dim_tags[bpe_labels.time_dim_axis]
    )
    out.mark_as_default_output()

    return net
