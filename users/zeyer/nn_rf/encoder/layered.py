"""
Some layered structure
"""

from typing import Any, Sequence, Tuple, Dict
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder, ISeqFramewiseEncoder, IEncoder


class Layered(ISeqDownsamplingEncoder):
    """
    Any layered structure composed of encoders.
    """

    def __init__(self, layers: Sequence[Dict[str, Any]]):
        super().__init__()
        self.layers = rf.ModuleList(rf.build_from_dict(layer) for layer in layers)

        self.downsample_factor = 1
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ISeqDownsamplingEncoder):
                self.downsample_factor *= layer.downsample_factor
            assert isinstance(layer, (ISeqDownsamplingEncoder, ISeqFramewiseEncoder, IEncoder)), (
                f"{self}: Layer {i} has unsupported type {type(layer)}."
            )
        self.out_dim = self.layers[-1].out_dim

    def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        tensor = source
        spatial_dim = in_spatial_dim
        for layer in self.layers:
            if isinstance(layer, ISeqDownsamplingEncoder):
                tensor, spatial_dim = layer(tensor, in_spatial_dim=spatial_dim)
            elif isinstance(layer, ISeqFramewiseEncoder):
                tensor = layer(tensor, spatial_dim=spatial_dim)
            elif isinstance(layer, IEncoder):
                tensor = layer(tensor)
            else:
                raise TypeError(f"Unsupported layer type: {type(layer)}")
        return tensor, spatial_dim
