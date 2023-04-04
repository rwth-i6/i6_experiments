from typing import Tuple
from returnn_common import nn

from .gammatone import make_gammatone_module

from enum import Enum, auto


class FeatureType(Enum):
    Gammatone = auto()


def make_features(type: FeatureType, *args, **kwargs) -> Tuple[nn.Module, nn.Dim]:
    return {FeatureType.Gammatone: make_gammatone_module}[type](*args, **kwargs)
