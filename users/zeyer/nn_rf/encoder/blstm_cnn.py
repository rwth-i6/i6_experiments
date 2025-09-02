"""
BLSTM with CNN
"""

from typing import Union, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from .blstm import BlstmEncoder


class BlstmCnnEncoder(BlstmEncoder):
    """
    PreCNN . BLSTM
    """

    def __init__(
        self,
        in_dim: Dim,
        lstm_dim: Union[int, Dim] = Dim(1024, name="lstm"),
        *,
        num_layers: int = 6,
        time_reduction: Union[int, Tuple[int, ...]] = 6,
        allow_pool_last: bool = False,
        dropout=0.3,
    ):
        self.pre_conv_net = PreConvNet(in_dim=in_dim)
        super().__init__(
            in_dim=self.pre_conv_net.out_dim,
            dim=lstm_dim,
            num_layers=num_layers,
            time_reduction=time_reduction,
            allow_pool_last=allow_pool_last,
            dropout=dropout,
        )

    def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        source = self.pre_conv_net(source, spatial_dim=in_spatial_dim)
        source, in_spatial_dim = super().__call__(source, in_spatial_dim=in_spatial_dim)
        return source, in_spatial_dim


class PreConvNet(rf.Module):
    """
    2 layer pre conv net, usually used before a BLSTM
    """

    def __init__(self, in_dim: Dim, dim: Dim = Dim(32, name="feat"), *, filter_size=(3, 3)):
        super().__init__()
        self.in_dim = in_dim
        self._dummy_feat_dim = Dim(1, name="dummy")
        self.conv0 = rf.Conv2d(self._dummy_feat_dim, out_dim=dim, padding="same", filter_size=filter_size)
        self.conv1 = rf.Conv2d(dim, dim, padding="same", filter_size=filter_size)
        self._final_extra_spatial_dim = in_dim.ceildiv_right(2).ceildiv_right(2)
        self.out_dim = self._final_extra_spatial_dim * dim

    def __call__(self, x: Tensor, *, spatial_dim: Dim) -> Tensor:
        assert self.in_dim in x.dims_set
        batch_dims = x.remaining_dims((self.in_dim, spatial_dim))
        extra_spatial_dim = self.in_dim
        x = rf.expand_dim(x, dim=self._dummy_feat_dim)
        x, _ = self.conv0(x, in_spatial_dims=(spatial_dim, extra_spatial_dim))
        feat_dim = self.conv0.out_dim
        x, extra_spatial_dim = rf.pool1d(x, in_spatial_dim=extra_spatial_dim, pool_size=2, mode="max", padding="same")
        x, _ = self.conv1(x, in_spatial_dims=(spatial_dim, extra_spatial_dim))
        x, extra_spatial_dim = rf.pool1d(x, in_spatial_dim=extra_spatial_dim, pool_size=2, mode="max", padding="same")
        x, extra_spatial_dim = rf.replace_dim(x, in_dim=extra_spatial_dim, out_dim=self._final_extra_spatial_dim)
        x, _ = rf.merge_dims(x, dims=(extra_spatial_dim, feat_dim), out_dim=self.out_dim)
        x.verify_out_shape(set(batch_dims) | {self.out_dim, spatial_dim})
        return x
