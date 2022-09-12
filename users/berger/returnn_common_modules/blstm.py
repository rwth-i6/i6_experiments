from returnn_common import nn
from returnn_common.nn.encoder.base import ISeqDownsamplingEncoder
from typing import Optional, Tuple, List


class BLSTMPoolModule(nn.Module):
    def __init__(
        self, hidden_size: int, max_pool: Optional[int], dropout: Optional[float] = None
    ):
        super().__init__()
        self.lstm_out_dim = nn.FeatureDim("lstm_out_dim", dimension=hidden_size)
        self.fw_rec = nn.LSTM(out_dim=self.lstm_out_dim)
        self.bw_rec = nn.LSTM(out_dim=self.lstm_out_dim)
        self.max_pool = max_pool
        self.dropout = dropout

    def __call__(self, inp: nn.Tensor, axis: nn.Dim) -> nn.Tensor:
        fw_out, _ = self.fw_rec(inp, direction=1, axis=axis)
        bw_out, _ = self.bw_rec(inp, direction=-1, axis=axis)
        concat = nn.concat((fw_out, self.lstm_out_dim), (bw_out, self.lstm_out_dim))
        if self.max_pool is not None and self.max_pool > 1:
            pool, pool_spatial_dim = nn.pool1d(
                concat,
                mode="max",
                pool_size=self.max_pool,
                padding="same",
                in_spatial_dim=axis,
            )
            inp = nn.dropout(pool, self.dropout, axis=nn.any_feature_dim)
        else:
            inp = nn.dropout(concat, self.dropout, axis=nn.any_feature_dim)
        return inp


class BLSTMNetwork(ISeqDownsamplingEncoder):
    def __init__(
        self,
        num_blstm_layers: int,
        layer_size: int,
        max_pool: Optional[List[int]] = None,
        feature_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()

        self.feature_dropout = feature_dropout
        self.downsample_factor = 1
        self.out_dim = nn.FeatureDim("blstm_out_dim", dimension=2 * layer_size)

        modules = []
        for i in range(num_blstm_layers):
            if max_pool is not None and len(max_pool) > i:
                pool = max_pool[i]
                self.downsample_factor *= pool
            else:
                pool = 1
            modules.append(
                BLSTMPoolModule(hidden_size=layer_size, max_pool=pool, **kwargs)
            )
        self.blstms = nn.Sequential(modules)

    def __call__(
        self, source: nn.Tensor, *, in_spatial_dim: nn.Dim
    ) -> Tuple[nn.Tensor, nn.Dim]:
        if self.feature_dropout > 0:
            source = nn.dropout(input_data, self.feature_dropout)
        out = self.blstms(inp, axis=self.out_dim)
        return out, self.out_dim
