"""
Custom frontends.

In RF `ConformerEncoder`, that would be an argument for `input_layer`.

Default we used in the past::

    input_layer=ConformerConvSubsample(
        in_dim,
        out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
        filter_sizes=[(3, 3), (3, 3), (3, 3)],
        pool_sizes=[(1, 2)],
        strides=[(1, 1), (3, 1), (2, 1)],
    ),

This uses a downsampling factor of 6.
"""

from typing import Tuple
from returnn.tensor import Dim
from returnn.frontend.encoder.conformer import ConformerConvSubsample
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder


# TODO how to serialize variants?


def get_default(*, in_dim: Dim, time_strides: Tuple[int, int, int] = (1, 3, 2)) -> ISeqDownsamplingEncoder:
    assert len(time_strides) == 3
    return ConformerConvSubsample(
        in_dim,
        out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
        filter_sizes=[(3, 3), (3, 3), (3, 3)],
        strides=[(s, 1) for s in time_strides],
        pool_sizes=[(1, 2)],
    )
