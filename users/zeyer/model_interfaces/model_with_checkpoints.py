"""
Model with checkpoints structure
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Set, Union
import dataclasses
from i6_core.returnn.training import Checkpoint as _TfCheckpoint, PtCheckpoint as _PtCheckpoint
from i6_experiments.users.zeyer.returnn.training import default_returnn_keep_epochs

if TYPE_CHECKING:
    from sisyphus import tk as _tk
    from i6_core.returnn.training import ReturnnTrainingJob
    from .model import ModelDef
    from .recog import RecogDef


Checkpoint = Union[_TfCheckpoint, _PtCheckpoint]


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
    # They will always be available and kept once the training reaches the epoch,
    # and are recommended to perform recognition on.
    # This is a subset of all kept epochs.
    fixed_epochs: Set[int]
    # when this becomes available, you can check potential other checkpoints
    scores_and_learning_rates: _tk.Path  # ReturnnTrainingJob.out_learning_rates
    model_dir: _tk.Path  # ReturnnTrainingJob.out_model_dir
    model_name: str = "epoch"  # RETURNN config `model` option; ReturnnTrainingJob has hardcoded "epoch"
    # Note on num_pretrain_epochs: This is used to get the right model filename.
    # When ModelDef is given and used, this should always be 0,
    # as ModelDef covers pretrain logic internally, as it has `epoch` as an argument,
    # and this is via the RETURNN `get_network` logic, where no epochs are specifically marked as pretrain.
    # However, this structure here is also used when wrapping old configs,
    # where we might have pretrain epochs.
    num_pretrain_epochs: int = 0
    # Note: This object influences the hash of certain jobs, e.g. GetBestRecogTrainExp.
    # Thus, when anything is added, we might want to define our custom _sis_hash here,
    # following the logic of the default sis_hash_helper.

    @classmethod
    def from_training_job(
        cls,
        definition: ModelDef,
        training_job: ReturnnTrainingJob,
        *,
        num_pretrain_epochs: int = 0,
    ) -> ModelWithCheckpoints:
        """
        Model from training job

        It sets fixed_epochs via the cleanup_old_models["keep"] option
        or via the default_returnn_keep_epochs() function.

        Note on num_pretrain_epochs:
        Also see the comment above.
        We could automatically figure that out by parsing the RETURNN config (load_returnn_config_safe).
        However, we want to keep it very simple here, and just have this explicit argument,
        which you need to provide explicitly.
        Usually with a correct ModelDef, it anyway should always be 0.
        """
        num_epochs = training_job.returnn_config.post_config["num_epochs"]
        save_interval = training_job.returnn_config.post_config["save_interval"]
        stored_epochs = set(list(range(save_interval, num_epochs, save_interval)) + [num_epochs])

        # Get the kept epochs, but maybe restrict it when all are kept.
        # The last epoch is always kept.
        # Via keep_last_n, some more would be kept, but we don't consider them here
        # because we don't want to do too many recognition jobs.
        fixed_kept_epochs = {num_epochs}
        # Get the user defined keep_epochs.
        cleanup_old_models = training_job.returnn_config.post_config.get("cleanup_old_models", None)
        keep_epochs = cleanup_old_models.get("keep", None) if isinstance(cleanup_old_models, dict) else None
        if keep_epochs is None:
            # cleanup_old_models is either not enabled.
            # In that case, all epochs are kept.
            # However, we don't want to perform recognition on all, so we fall back to the default kept epochs.
            # In the case it is enabled, but "keep" is not specified, the default is used,
            # so this is correct as well.
            keep_epochs = default_returnn_keep_epochs(num_epochs=num_epochs)
        fixed_kept_epochs.update(keep_epochs)
        # Only the epochs which are also stored are kept.
        fixed_kept_epochs.intersection_update(stored_epochs)

        return ModelWithCheckpoints(
            definition=definition,
            fixed_epochs=fixed_kept_epochs,
            scores_and_learning_rates=training_job.out_learning_rates,
            model_dir=training_job.out_model_dir,
            num_pretrain_epochs=num_pretrain_epochs,
        )

    @property
    def last_fixed_epoch_idx(self) -> int:
        """last epoch"""
        return max(self.fixed_epochs)

    def get_epoch(self, epoch: int) -> ModelWithCheckpoint:
        """for one specific epoch"""
        is_pretrain = epoch <= self.num_pretrain_epochs
        return ModelWithCheckpoint(
            self.definition,
            (
                _PtCheckpoint(
                    self.model_dir.join_right(
                        self.model_name + (".pretrain" if is_pretrain else "") + ".%03d.pt" % epoch
                    )
                )
                if model_def_is_torch(self.definition)
                else _TfCheckpoint(
                    index_path=self.model_dir.join_right(
                        self.model_name + (".pretrain" if is_pretrain else "") + ".%03d.index" % epoch
                    )
                )
            ),
        )

    def get_last_fixed_epoch(self) -> ModelWithCheckpoint:
        """for the last fixed epoch"""
        return self.get_epoch(self.last_fixed_epoch_idx)


@dataclasses.dataclass(frozen=True)
class Alignment:
    """Alignment, for one specific dataset"""

    hdf_files: List[_tk.Path]


@dataclasses.dataclass(frozen=True)
class AlignmentCollection:
    """Alignment for multiple datasets"""

    alignments: Dict[str, Alignment]


@dataclasses.dataclass(frozen=True)
class ModelWithCheckpointAndRecog(ModelWithCheckpoint):
    """Model with recog"""

    recog: RecogDef


def model_def_is_torch(model_def: ModelDef) -> bool:
    """
    Is this a torch model?
    """
    if getattr(model_def, "backend", None) == "torch":
        return True
    return False
