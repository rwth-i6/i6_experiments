"""
Model logic
"""

from typing import Protocol, List, Dict
import dataclasses
from sisyphus import tk
from i6_core.returnn.training import Checkpoint
from returnn_common import nn


class ModelForTrainDef(Protocol):
    """
    Creates the model for training, and defines the losses (mark_as_loss).
    Returns the model root module.
    """
    def __call__(self, *,
                 data: nn.Data, data_spatial_dim: nn.Dim,
                 targets: nn.Data, targets_spatial_dim: nn.Dim
                 ) -> nn.Module:
        raise NotImplementedError


class ModelForFramewiseTrainDef(Protocol):
    """
    Creates the model for training, and defines the losses (mark_as_loss).
    Returns the model root module.
    """
    def __call__(self, *,
                 data: nn.Data, data_spatial_dim: nn.Dim,
                 targets: nn.Data, targets_spatial_dim: nn.Dim,
                 align_targets: nn.Data, align_targets_spatial_dim: nn.Dim
                 ) -> nn.Module:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ModelWithCheckpoint:
    """
    Model
    """
    definition: ModelForTrainDef
    checkpoint: Checkpoint


@dataclasses.dataclass(frozen=True)
class Alignment:
    """Alignment, for one specific dataset"""
    hdf_files: List[tk.Path]


@dataclasses.dataclass(frozen=True)
class AlignmentCollection:
    """Alignment for multiple datasets"""
    alignments: Dict[str, Alignment]
