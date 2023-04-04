from __future__ import annotations
from returnn_common.asr.specaugment import specaugment_v2
from returnn_common.nn.encoder.base import ISeqDownsamplingEncoder, ISeqFramewiseEncoder
from ..feature_extraction import FeatureType, make_features
from ..encoder import EncoderType, make_encoder
from returnn_common import nn
from i6_experiments.users.berger.network.models.fullsum_ctc_raw_samples import make_rasr_ctc_loss_opts


class CTCModel(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        feature_type: FeatureType = FeatureType.Gammatone,
        encoder_type: EncoderType = EncoderType.Blstm,
        specaug_args: dict = {},
        feature_args: dict = {},
        encoder_args: dict = {},
        loss_args: dict = {},
    ) -> None:
        self.features, self.feature_dim = make_features(feature_type, **feature_args)

        self.specaug_args = specaug_args

        self.encoder = make_encoder(encoder_type, in_dim=self.feature_dim, **encoder_args)
        self.out_dim = nn.FeatureDim("ctc_out", dimension=num_outputs)
        self.out_projection = nn.Linear(self.encoder.out_dim, self.out_dim)

        self.loss_args = loss_args

    def __call__(
        self,
        source: nn.Tensor,
        *,
        train: bool = False,
    ) -> nn.Tensor:
        assert source.data is not None
        assert source.data.time_dim_axis is not None
        in_spatial_dim = source.data.dim_tags[source.data.time_dim_axis]

        x = source

        if source.data.feature_dim_or_sparse_dim is not None:
            x = nn.squeeze(x, axis=source.data.feature_dim_or_sparse_dim)

        x = self.features(x, in_spatial_dim=in_spatial_dim)
        assert isinstance(x, tuple)
        x, spatial_dim = x
        assert isinstance(spatial_dim, nn.Dim)

        x = specaugment_v2(x, spatial_dim=spatial_dim, feature_dim=self.feature_dim, **self.specaug_args)

        if isinstance(self.encoder, ISeqFramewiseEncoder):
            x = self.encoder(x, spatial_dim=spatial_dim)
        elif isinstance(self.encoder, ISeqDownsamplingEncoder):
            x, spatial_dim = self.encoder(x, in_spatial_dim=spatial_dim)
        else:
            raise TypeError(f"unsupported encoder type {type(self.encoder)}")

        x = self.out_projection(x)

        if train:
            x = nn.make_layer(
                name="ctc_loss",
                layer_dict={
                    "class": "activation",
                    "activation": "softmax",
                    "from": x,
                    "loss": "fast_bw",
                    "loss_opts": make_rasr_ctc_loss_opts(**self.loss_args),
                },
            )
        else:
            x = nn.log_softmax(x, axis=self.out_dim)

        return x


def construct_net_with_data(epoch: int, train: bool, audio_data: nn.Data, **kwargs) -> CTCModel:
    net = CTCModel(**kwargs)

    output = net(source=nn.get_extern_data(audio_data), train=train)
    output.mark_as_default_output()

    return net
