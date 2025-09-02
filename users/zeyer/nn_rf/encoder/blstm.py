"""
Multi layer BLSTM
"""

from typing import Union, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder


class BlstmEncoder(ISeqDownsamplingEncoder):
    """
    multi-layer BLSTM
    """

    def __init__(
        self,
        in_dim: Dim,
        dim: Dim = Dim(1024, name="lstm"),
        *,
        num_layers: int = 6,
        time_reduction: Union[int, Tuple[int, ...]] = 6,
        allow_pool_last: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.num_layers = num_layers

        if isinstance(time_reduction, int):
            n = time_reduction
            time_reduction = []
            for i in range(2, n + 1):
                while n % i == 0:
                    time_reduction.insert(0, i)
                    n //= i
                if n <= 1:
                    break
        assert isinstance(time_reduction, (tuple, list))
        assert num_layers > 0
        if num_layers == 1 and not allow_pool_last:
            assert not time_reduction, f"time_reduction {time_reduction} not supported for single layer"
        while len(time_reduction) > (num_layers if allow_pool_last else (num_layers - 1)):
            time_reduction[:2] = [time_reduction[0] * time_reduction[1]]
        self.time_reduction = time_reduction

        self.dropout = dropout

        in_dims = [in_dim] + [2 * dim] * (num_layers - 1)
        self.layers = rf.ModuleList([BlstmSingleLayer(in_dims[i], dim) for i in range(num_layers)])
        self.out_dim = self.layers[-1].out_dim

    def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        feat_dim = self.in_dim
        for i, lstm in enumerate(self.layers):
            if i > 0:
                if self.dropout:
                    source = rf.dropout(source, drop_prob=self.dropout, axis=feat_dim)
            assert isinstance(lstm, BlstmSingleLayer)
            source = lstm(source, spatial_dim=in_spatial_dim)
            feat_dim = lstm.out_dim
            red = self.time_reduction[i] if i < len(self.time_reduction) else 1
            if red > 1:
                source, in_spatial_dim = rf.pool1d(
                    source, mode="max", padding="same", pool_size=red, in_spatial_dim=in_spatial_dim
                )
        return source, in_spatial_dim


class BlstmSingleLayer(rf.Module):
    """
    single-layer BLSTM
    """

    def __init__(self, in_dim: Dim, out_dim: Dim):
        super(BlstmSingleLayer, self).__init__()
        self.in_dim = in_dim
        self.fw = rf.LSTM(in_dim, out_dim)
        self.bw = rf.LSTM(in_dim, out_dim)
        self.out_dim = 2 * out_dim

    def __call__(self, x: Tensor, *, spatial_dim: Dim) -> Tensor:
        batch_dims = x.remaining_dims((self.in_dim, spatial_dim))
        x_ = rf.reverse_sequence(x, axis=spatial_dim)
        fw, _ = self.fw(x, spatial_dim=spatial_dim, state=self.fw.default_initial_state(batch_dims=batch_dims))
        bw_, _ = self.bw(x_, spatial_dim=spatial_dim, state=self.fw.default_initial_state(batch_dims=batch_dims))
        bw = rf.reverse_sequence(bw_, axis=spatial_dim)
        return rf.concat_features(fw, bw)
