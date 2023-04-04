from __future__ import annotations
from typing import Optional, Tuple

from returnn_common.nn.encoder.base import ISeqDownsamplingEncoder, ISeqFramewiseEncoder
from ..feature_extraction import FeatureType, make_features
from ..encoder import EncoderType, make_encoder
from returnn_common import nn
from returnn_common.nn import hybrid_hmm
from returnn_common.asr.specaugment import specaugment_v2


class HybridModel(hybrid_hmm.IHybridHMM):
    def __init__(
        self,
        out_dim: nn.Dim,
        feature_type: FeatureType = FeatureType.Gammatone,
        encoder_type: EncoderType = EncoderType.Blstm,
        focal_loss_scale: float = 1.0,
        label_smoothing: float = 0.0,
        specaug_args: dict = {},
        feature_args: dict = {},
        encoder_args: dict = {},
    ) -> None:
        self.features, self.feature_dim = make_features(feature_type, **feature_args)
        self.specaug_args = specaug_args

        self.encoder = make_encoder(encoder_type, in_dim=self.feature_dim, **encoder_args)
        self.out_projection = nn.Linear(self.encoder.out_dim, out_dim)
        self.out_dim = out_dim

        self.focal_loss_scale = focal_loss_scale
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        source: nn.Tensor,
        *,
        train: bool = False,
        targets: Optional[nn.Tensor] = None,
    ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
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
            assert targets
            assert spatial_dim in targets.dims_set
            if self.label_smoothing != 0:
                targets = nn.label_smoothing(targets, self.label_smoothing)
            ce_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=x, targets=targets, axis=self.out_dim)
            ce_loss *= (1.0 - nn.exp(-ce_loss)) ** self.focal_loss_scale
            ce_loss.mark_as_loss("ce")
        return nn.log_softmax(x, axis=self.out_dim), None


def construct_net_with_data(epoch: int, train: bool, audio_data: nn.Data, label_data: nn.Data, **kwargs) -> HybridModel:
    label_dim = label_data.feature_dim_or_sparse_dim
    assert label_dim is not None
    net = HybridModel(out_dim=label_dim, **kwargs)

    out, _ = net(train=train, source=nn.get_extern_data(audio_data), targets=nn.get_extern_data(label_data))
    out.mark_as_default_output()

    return net
