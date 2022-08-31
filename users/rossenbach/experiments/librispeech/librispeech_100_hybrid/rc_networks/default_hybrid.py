from typing import Any, Callable

from returnn_common import nn
from returnn_common.nn.hybrid_hmm import HybridHMM, EncoderType, ISeqFramewiseEncoder
from returnn_common.asr.specaugment import specaugment_v2

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

    def __init__(self, label_dim: nn.Dim, num_layers: int, size: int, dropout: float):
        super().__init__()

        self.out_dim = label_dim
        self.blstm_stack = nn.Sequential(
            [
                BLSTMLayer(size=size, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def __call__(self, source: nn.Tensor, *, spatial_dim: nn.Dim) -> nn.Tensor:
        spec_source =  specaugment_v2(source, spatial_dim=spatial_dim, feature_dim=source.feature_dim)
        return self.blstm_stack(spec_source, time_dim=spatial_dim)

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
    enc = encoder(label_feature_dim, **kwargs)
    net = HybridHMM(
        encoder=enc,
        out_dim=label_feature_dim,
    )
    out, _ = net(
        source=nn.get_extern_data(audio_data),
        train=train,
        targets=nn.get_extern_data(label_data)
    )
    out.mark_as_default_output()

    return net
