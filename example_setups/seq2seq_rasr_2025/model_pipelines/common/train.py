__all__ = ["TrainOptions", "train"]

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.common.setups.serialization import Collection, ExternalImport, Import
from i6_models.config import ModelConfiguration
from sisyphus import tk

from ...data.base import DataConfig
from ...tools import minireturnn_root, returnn_python_exe
from ..common.learning_rates import LRConfig
from ..common.optimizer import OptimizerConfig
from ..common.serializers import recipe_imports


@dataclass
class TrainOptions:
    train_data_config: DataConfig
    cv_data_config: DataConfig
    save_epochs: List[int]
    batch_size: int
    accum_grad_multiple_step: int
    optimizer_config: OptimizerConfig
    lr_config: LRConfig
    gradient_clip: float
    num_workers_per_gpu: int
    automatic_mixed_precision: bool
    gpu_mem_rqmt: float
    max_seqs: Optional[int]
    max_seq_length: Optional[int]


ModelConfigType = TypeVar("ModelConfigType", bound=ModelConfiguration)


@dataclass
class TrainedModel(Generic[ModelConfigType]):
    model_config: ModelConfigType
    train_job: ReturnnTrainingJob

    def get_checkpoint(self, epoch: Optional[int] = None) -> PtCheckpoint:
        if epoch is None:
            epoch = max(self.train_job.out_checkpoints)
        return self.train_job.out_checkpoints[epoch]  # type: ignore


def train(options: TrainOptions, model_serializers: Collection, train_step_import: Import) -> ReturnnTrainingJob:
    num_epochs = options.save_epochs[-1]

    config_dict = {
        "backend": "torch",
        "batch_size": options.batch_size,
        "accum_grad_multiple_step": options.accum_grad_multiple_step,
        "cleanup_old_models": {  # TODO: This should be moved to post-config and `keep_epochs` in ReturnnTrainingJob should be utilized
            "keep_last_n": 1,
            "keep_best_n": 0,
            "keep": options.save_epochs,
        },
        "gradient_clip_norm": options.gradient_clip,
        "num_workers_per_gpu": options.num_workers_per_gpu,
        "stop_on_nonfinite_train_score": True,
    }
    if options.automatic_mixed_precision:
        config_dict["torch_amp_options"] = {"dtype": "bfloat16"}

    if options.max_seqs:
        config_dict["max_seqs"] = options.max_seqs

    if options.max_seq_length:
        config_dict["max_seq_length"] = options.max_seq_length

    train_returnn_config = ReturnnConfig(
        config=config_dict,
        python_prolog=[
            recipe_imports,
            ExternalImport(minireturnn_root),
        ],
        python_epilog=[
            model_serializers,
            train_step_import,
        ],  # type: ignore
        sort_config=False,
    )

    train_returnn_config.update(options.optimizer_config.get_returnn_config())
    train_returnn_config.update(options.lr_config.get_returnn_config())
    train_returnn_config.update(options.train_data_config.get_returnn_data("train"))
    train_returnn_config.update(options.cv_data_config.get_returnn_data("dev"))

    train_job = ReturnnTrainingJob(
        returnn_config=train_returnn_config,
        log_verbosity=5,
        num_epochs=num_epochs,
        time_rqmt=168,
        mem_rqmt=24,
        cpu_rqmt=6,
        returnn_python_exe=returnn_python_exe,
        returnn_root=minireturnn_root,
    )
    train_job.add_alias("training")
    train_job.rqmt["gpu_mem"] = options.gpu_mem_rqmt

    tk.register_output("training/learning_rates", train_job.out_learning_rates)
    tk.register_output("training/final_checkpoint", train_job.out_checkpoints[num_epochs].path)  # type: ignore

    return train_job
