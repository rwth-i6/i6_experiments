from __future__ import annotations
from typing import Optional, Tuple
from i6_experiments.users.berger.network.helpers.speech_separation import (
    add_speech_separation,
)

from returnn_common.nn.encoder.base import ISeqDownsamplingEncoder, ISeqFramewiseEncoder
from ..feature_extraction import FeatureType, make_features
from ..encoder import EncoderType, make_encoder
from returnn_common import nn
from returnn_common.nn import hybrid_hmm
from returnn_common.nn.encoder.blstm import BlstmSingleLayer
from returnn_common.asr.specaugment import specaugment_v2


class DualOutputHybridModel(hybrid_hmm.IHybridHMM):
    def __init__(
        self,
        out_dim: nn.Dim,
        feature_type: FeatureType = FeatureType.Gammatone,
        encoder_type: EncoderType = EncoderType.Blstm,
        focal_loss_scale: float = 1.0,
        label_smoothing: float = 0.0,
        specaug_args: dict = {},
        feature_args: dict = {},
        speech_separator_args: dict = {},
        enable_sep: bool = True,
        enable_mix: bool = False,
        enable_mas: bool = False,
        enable_combine: bool = False,
        sep_encoder_args: dict = {},
        mix_encoder_args: dict = {},
        mas_encoder_args: dict = {},
        combine_layer_dim: int = 512,
    ) -> None:
        assert enable_sep or enable_mix
        assert enable_sep or enable_mas
        self.enable_sep = enable_sep
        self.enable_mix = enable_mix
        self.enable_mas = enable_mas
        self.enable_combine = enable_combine

        self.speech_separator_net_dict = {}
        out_layer_name, dim_tags = add_speech_separation(self.speech_separator_net_dict, **speech_separator_args)
        self.speech_separator_net_dict["output"] = {
            "class": "copy",
            "from": out_layer_name,
        }
        self.speaker_dim = dim_tags["speaker"]
        self.features, self.feature_dim = make_features(feature_type, **feature_args)
        self.specaug_args = specaug_args

        self.sep_encoder = make_encoder(encoder_type, in_dim=self.feature_dim, **sep_encoder_args)
        if enable_sep:
            sep_dim = self.sep_encoder.out_dim
        else:
            sep_dim = self.feature_dim
        self.sep_projection = nn.Linear(self.sep_encoder.out_dim, out_dim)

        self.mix_encoder = make_encoder(encoder_type, in_dim=self.feature_dim, **mix_encoder_args)
        if enable_mix:
            mix_dim = self.mix_encoder.out_dim
        else:
            mix_dim = self.feature_dim

        if enable_sep and enable_mix:
            assert self.sep_encoder.out_dim.is_equal(self.mix_encoder.out_dim)
            self.mix_encoder.out_dim.declare_same_as(self.sep_encoder.out_dim)
            mas_in_dim = self.sep_encoder.out_dim
        else:
            mas_in_dim = sep_dim + mix_dim

        self.mas_encoder = make_encoder(encoder_type, in_dim=mas_in_dim, **mas_encoder_args)
        self.mas_projection = nn.Linear(self.mas_encoder.out_dim, out_dim)

        self.combine_layer = BlstmSingleLayer(
            in_dim=self.mas_encoder.out_dim + self.mas_encoder.out_dim,
            out_dim=nn.FeatureDim("combine_out", combine_layer_dim),
        )

        if enable_combine:
            softmax_in = self.combine_layer.out_dim
        elif enable_mas:
            softmax_in = self.mas_encoder.out_dim
        elif enable_mix:
            softmax_in = sep_dim + mix_dim
        else:
            softmax_in = sep_dim

        self.out_projection = nn.Linear(softmax_in, out_dim)
        self.out_dim = out_dim

        self.focal_loss_scale = focal_loss_scale
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        source: nn.Tensor,
        *,
        train: bool = False,
        speaker_idx: Optional[int] = None,
        targets: Optional[nn.Tensor] = None,
    ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
        assert source.data is not None
        assert source.data.time_dim_axis is not None
        in_spatial_dim = source.data.dim_tags[source.data.time_dim_axis]

        x = source

        if source.data.feature_dim_or_sparse_dim is not None:
            x = nn.squeeze(x, axis=source.data.feature_dim_or_sparse_dim)

        x = nn.make_layer(
            {
                "class": "subnetwork",
                "from": x,
                "subnetwork": self.speech_separator_net_dict,
            },
            name="speech_separator",
        )

        x = self.features(x, in_spatial_dim=in_spatial_dim)
        assert isinstance(x, tuple)
        x, spatial_dim = x
        assert isinstance(spatial_dim, nn.Dim)

        x = specaugment_v2(
            x,
            spatial_dim=spatial_dim,
            feature_dim=self.feature_dim,
            **self.specaug_args,
        )

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

    out, _ = net(
        train=train,
        source=nn.get_extern_data(audio_data),
        targets=nn.get_extern_data(label_data),
    )
    out.mark_as_default_output()

    return net
