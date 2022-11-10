"""
Implementation of the CTC Aligner, updated for the new non-lazy init
"""
from typing import List, Union
from returnn_common import nn


class Conv1DBlock(nn.Module):
    """
    1D Convolutional Block
    """

    def __init__(
            self,
            in_dim: nn.Dim,
            dim: Union[int, nn.Dim] = 256,
            filter_size: int = 5,
            bn_epsilon: float = 1e-5,
            dropout: float = 0.5,
            l2: float = 1e-07,
    ):
        """
        :param dim: feature dimension of the convolution
        :param filter_size: filter size of the conv, int because we are doing 1D here
        :param bn_epsilon: batch_normalization epsilon value
        :param dropout: dropout value
        :param l2: weight decay value
        """
        super(Conv1DBlock, self).__init__()
        if isinstance(dim, int):
            self.conv_dim = nn.FeatureDim("conv_dim_%d" % dim, dim)
        elif isinstance(dim, nn.Dim):
            self.conv_dim = dim
        else:
            raise Exception("Wrong Dim given!")
        self.conv = nn.Conv1d(
            in_dim=in_dim,
            out_dim=self.conv_dim,
            filter_size=filter_size,
            padding="same",
            with_bias=False,
        )
        self.bn = nn.BatchNorm(
            in_dim=self.conv_dim,
            epsilon=bn_epsilon, use_mask=False
        )
        self.dropout = dropout
        self.l2 = l2

    def __call__(self, inp: nn.Tensor, time_dim: nn.SpatialDim):
        conv, _ = self.conv(inp, in_spatial_dim=time_dim)
        # set weight decay
        for param in self.conv.parameters():
            param.weight_decay = self.l2

        conv = nn.relu(conv)
        bn = self.bn(conv)
        drop = nn.dropout(
            bn, dropout=self.dropout, axis=[nn.batch_dim, time_dim, bn.feature_dim]
        )

        return drop


class ConvStack(nn.Module):
    """
    Stacks :class:`Conv1DBlock` modules
    """

    def __init__(
            self,
            in_dim: nn.Dim,
            num_layers: int = 5,
            dim_sizes: List[int] = (256,),
            filter_sizes: List[int] = (5, 5, 5, 5, 5),
            bn_epsilon: float = 1e-5,
            dropout: List[float] = (0.35, 0.35, 0.35, 0.35, 0.35),
            l2: float = 1e-07,
    ):
        """
        :param num_layers: number of conv block layers
        :param dim_sizes: dimensions for the convolutions in the block
        :param filter_sizes: sizes for the filters in the block
        :param bn_epsilon: batch_normalization epsilon value
        :param dropout: dropout values
        :param l2: weight decay value
        """
        super(ConvStack, self).__init__()
        assert isinstance(dim_sizes, int) or len(dim_sizes) == num_layers  # mismatch in dim_sizes
        assert len(filter_sizes) == num_layers  # mismatch in filter_sizes
        assert len(dropout) == num_layers  # mismatch in dropout

        self.num_layers = num_layers
        # simplify tags a bit if possible
        if isinstance(dim_sizes, int):
            out_dims = [nn.FeatureDim("conv_dim", dim_sizes)] * num_layers
        elif len(set(dim_sizes)) == 1:  # all sizes equal
            out_dims = [nn.FeatureDim("conv_dim", dim_sizes[0])] * num_layers
        else:
            out_dims = [
                nn.FeatureDim("conv_dim_%s" % str(x), dim_sizes[x]) for x in range(num_layers)
            ]

        sequential_list = []
        temp_in_dim = in_dim
        for x in range(num_layers):
            sequential_list.append(
                Conv1DBlock(
                    in_dim=temp_in_dim,
                    dim=out_dims[x],
                    filter_size=filter_sizes[x],
                    bn_epsilon=bn_epsilon,
                    dropout=dropout[x],
                    l2=l2,
                )
            )
            temp_in_dim = out_dims[x]

        self.stack = nn.Sequential(sequential_list)
        self.out_dim = out_dims[-1]

    def __call__(self, inp: nn.Tensor, time_dim: nn.Dim):
        """
        Applies all conv blocks in sequence

        :param inp: input tensor
        :return:
        """
        out = self.stack(inp, time_dim=time_dim)
        return out