"""
Model logic
"""

from __future__ import annotations
from typing import Protocol, TypeVar, Optional, List, Dict
import dataclasses
from sisyphus import tk
from i6_core.returnn.training import ReturnnTrainingJob, Checkpoint
from returnn_common import nn


ModelT = TypeVar("ModelT", bound=nn.Module)


class ModelDef(Protocol[ModelT]):
    """
    Creates the model, per epoch
    """
    def __call__(self, *, epoch: int, in_dim: nn.Dim, target_dim: nn.Dim) -> ModelT:
        raise NotImplementedError


class TrainDef(Protocol[ModelT]):
    """
    Defines the losses (mark_as_loss).
    """
    def __call__(self, *,
                 model: ModelT,
                 data: nn.Tensor, data_spatial_dim: nn.Dim,
                 targets: nn.Tensor, targets_spatial_dim: nn.Dim
                 ):
        raise NotImplementedError

    learning_rate_control_error_measure: Optional[str] = None


class FramewiseTrainDef(Protocol[ModelT]):
    """
    Defines the losses (mark_as_loss).
    """
    def __call__(self, *,
                 model: ModelT,
                 data: nn.Tensor, data_spatial_dim: nn.Dim,
                 align_targets: nn.Tensor, align_targets_spatial_dim: nn.Dim
                 ):
        raise NotImplementedError

    learning_rate_control_error_measure: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ModelWithCheckpoint:
    """
    Model
    """
    definition: ModelDef
    checkpoint: Checkpoint

    def with_recog(self, recog: RecogDef) -> ModelWithCheckpointAndRecog:
        """add recog def"""
        return ModelWithCheckpointAndRecog(self.definition, self.checkpoint, recog)


@dataclasses.dataclass(frozen=True)
class ModelWithCheckpoints:
    """
    What comes out of training
    """
    definition: ModelDef
    # they will always be available and kept once the training reaches the epoch
    fixed_checkpoints: Dict[int, Checkpoint]  # epoch -> checkpoint
    # when this becomes available, you can check potential other checkpoints
    scores_and_learning_rates: tk.Path  # ReturnnTrainingJob.out_learning_rates
    model_dir: tk.Path  # ReturnnTrainingJob.out_model_dir
    model_name: str = "epoch"  # RETURNN config `model` option; ReturnnTrainingJob has hardcoded "epoch"

    @classmethod
    def from_training_job(cls, definition: ModelDef, training_job: ReturnnTrainingJob) -> ModelWithCheckpoints:
        """model from training job"""
        return ModelWithCheckpoints(
            definition=definition,
            fixed_checkpoints=training_job.out_checkpoints,
            scores_and_learning_rates=training_job.out_learning_rates,
            model_dir=training_job.out_model_dir,
        )

    @property
    def last_fixed_epoch_idx(self) -> int:
        """last epoch"""
        return max(self.fixed_checkpoints.keys())

    def get_epoch(self, epoch: int) -> ModelWithCheckpoint:
        """for one specific epoch"""
        return ModelWithCheckpoint(
            self.definition,
            Checkpoint(index_path=self.model_dir.join_right("%s.%03d.index" % (self.model_name, epoch))))

    def get_last_fixed_epoch(self) -> ModelWithCheckpoint:
        """for the last fixed epoch"""
        return self.get_epoch(self.last_fixed_epoch_idx)


@dataclasses.dataclass(frozen=True)
class Alignment:
    """Alignment, for one specific dataset"""
    hdf_files: List[tk.Path]


@dataclasses.dataclass(frozen=True)
class AlignmentCollection:
    """Alignment for multiple datasets"""
    alignments: Dict[str, Alignment]


class RecogDef(Protocol[ModelT]):
    """
    Defines the recog. It returns the recog output.
    Thus, this includes all the recog details, such as beam size, etc.
    """

    def __call__(self, *,
                 model: ModelT,
                 data: nn.Tensor, data_spatial_dim: nn.Dim,
                 ) -> nn.Tensor:
        """
        :return: recog output, including beam or not, depending on output_with_beam
        """
        raise NotImplementedError

    output_with_beam: bool = True
    output_blank_label: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ModelWithCheckpointAndRecog(ModelWithCheckpoint):
    """Model with recog"""
    recog: RecogDef
