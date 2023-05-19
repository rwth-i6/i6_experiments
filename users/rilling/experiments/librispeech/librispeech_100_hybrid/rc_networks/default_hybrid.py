from typing import Any, Callable, Optional, Tuple

from returnn_common import nn
from returnn_common.nn.hybrid_hmm import IHybridHMM, EncoderType, ISeqFramewiseEncoder, ISeqDownsamplingEncoder
from returnn_common.asr.specaugment import specaugment_v2, random_mask_v2


def flexible_specaugment(
        x: nn.Tensor, *,
        spatial_dim: nn.Dim,
        feature_dim: nn.Dim = nn.NotSpecified,
        only_on_train: bool = True,
        min_frame_masks=2,
        mask_each_n_frames=25,
        max_frames_per_mask=20,
        min_feature_masks=2,
        max_feature_masks=5,
        max_features_per_mask=8
    ) -> nn.Tensor:
    """
    SpecAugment :func:`specaugment_v2` but with adjustable parameters
    """
    if feature_dim is nn.NotSpecified:
        assert x.feature_dim
        feature_dim = x.feature_dim

    with nn.Cond(nn.train_flag() | (not only_on_train)) as cond:
        x_masked = x
        spatial_len = nn.dim_value(spatial_dim)
        # time mask
        x_masked = random_mask_v2(
            x_masked, mask_axis=spatial_dim, broadcast_axis=feature_dim,
            min_num=nn.minimum(min_frame_masks, spatial_len),
            max_num=nn.minimum(nn.maximum(spatial_len // mask_each_n_frames, min_frame_masks), spatial_len),
            max_dims=max_frames_per_mask)
        # feature mask
        x_masked = random_mask_v2(
            x_masked, mask_axis=feature_dim, broadcast_axis=spatial_dim,
            min_num=min_feature_masks, max_num=max_feature_masks,
            max_dims=max_features_per_mask)
        cond.true = x_masked
        cond.false = x
    return cond.result


class BLSTMLayer(nn.Module):
    """
    BLSTM with time broadcasted dropout
    """
    def __init__(self, size=512, dropout: float = 0.0):
        super().__init__()
        self.lstm_dim = nn.FeatureDim(description="BLSTM-out-dim", dimension=size)
        self.out_dim = self.lstm_dim * 2
        self.fwd_lstm = nn.LSTM(out_dim=self.lstm_dim)
        self.bwd_lstm = nn.LSTM(out_dim=self.lstm_dim)
        self.dropout = dropout

    def __call__(self, source: nn.Tensor, time_dim: nn.Dim):
        fwd, _ = self.fwd_lstm(source, axis=time_dim, direction=1)
        bwd, _ = self.bwd_lstm(source, axis=time_dim, direction=-1)
        concat = nn.concat((fwd, self.lstm_dim), (bwd, self.lstm_dim))
        if self.dropout > 0.0:
            return nn.dropout(concat, self.dropout, axis=[nn.batch_dim, concat.feature_dim])
        else:
            return concat


class BLSTMEncoder(ISeqFramewiseEncoder):
    """
    BLSTM encoder with specaugment
    """

    def __init__(self, label_dim: nn.Dim, num_layers: int, size: int, dropout: float, specaugment_options=None):
        super().__init__()
        self.specaugment_options = specaugment_options
        self.out_dim = label_dim
        self.blstm_stack = nn.Sequential(
            [
                BLSTMLayer(size=size, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def __call__(self, source: nn.Tensor, *, spatial_dim: nn.Dim) -> nn.Tensor:
        if self.specaugment_options:
            source = flexible_specaugment(
                source,
                spatial_dim=spatial_dim,
                feature_dim=source.feature_dim,
                **self.specaugment_options)
        return self.blstm_stack(source, time_dim=spatial_dim)


class HybridHMM(IHybridHMM):
    """
    Hybrid NN-HMM
    """

    def __init__(self, *, encoder: EncoderType, out_dim: nn.Dim, focal_loss_scale: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.out_dim = out_dim
        self.focal_loss_scale = focal_loss_scale
        self.out_projection = nn.Linear(out_dim)

    def __call__(self, source: nn.Tensor, *,
                 state: Optional[nn.LayerState] = None,
                 train: bool = False, targets: Optional[nn.Tensor] = None) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
        assert source.data.time_dim_axis is not None
        in_spatial_dim = source.data.dim_tags[source.data.time_dim_axis]
        assert state is None, f"{self} stateful hybrid HMM not supported yet"
        if isinstance(self.encoder, ISeqFramewiseEncoder):
            encoder_output = self.encoder(source, spatial_dim=in_spatial_dim)
            out_spatial_dim = in_spatial_dim
        elif isinstance(self.encoder, ISeqDownsamplingEncoder):
            encoder_output, out_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim)
        else:
            raise TypeError(f"unsupported encoder type {type(self.encoder)}")
        out_embed = self.out_projection(encoder_output)
        if train:
            assert out_spatial_dim in targets.shape
            ce_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=out_embed, targets=targets, axis=self.out_dim)
            # focal loss (= more emphasis on "low" scores), might not be correct yet
            if self.focal_loss_scale != 1.0:
                ce_loss *= (1.0 - nn.exp(-ce_loss)) ** self.focal_loss_scale
            ce_loss.mark_as_loss(name="default_ce")
        return nn.log_softmax(out_embed, axis=self.out_dim), None


def construct_hybrid_network(
        epoch: int,
        train: bool,
        encoder: Callable[[nn.Dim, Any], EncoderType],
        audio_data: nn.Data,
        label_data: nn.Data,
        **kwargs
):
    """
    :param epoch:
    :param train:
    :param encoder:
    :param audio_data:
    :param label_data:
    :param kwargs: encoder kwargs
    :return:
    """
    label_feature_dim = label_data.sparse_dim
    focal_loss_scale = kwargs.pop("focal_loss_scale", 1.0)
    enc = encoder(label_feature_dim, **kwargs)
    net = HybridHMM(
        encoder=enc,
        out_dim=label_feature_dim,
        focal_loss_scale=focal_loss_scale
    )
    out, _ = net(
        source=nn.get_extern_data(audio_data),
        train=train,
        targets=nn.get_extern_data(label_data)
    )
    out.mark_as_default_output()

    return net
