__all__ = ["TrainOptions", "train"]

from dataclasses import dataclass
from typing import List

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.common.setups.serialization import Collection, ExternalImport, Import
from sisyphus import tk

from ...data.base import DataConfig
from ...tools import minireturnn_root, returnn_python_exe
from ..common.imports import recipe_imports
from ..common.learning_rates import LRConfig
from ..common.optimizer import OptimizerConfig


@dataclass
class TrainOptions:
    descriptor: str
    train_data_config: DataConfig
    cv_data_config: DataConfig
    save_epochs: List[int]
    batch_size: int
    accum_grad_multiple_step: int
    optimizer_config: OptimizerConfig
    lr_config: LRConfig
    gradient_clip: float
    num_workers_per_gpu: int
    stop_on_inf_nan_score: bool


def train(options: TrainOptions, model_serializers: Collection, train_step_import: Import) -> ReturnnTrainingJob:
    num_epochs = options.save_epochs[-1]

    train_returnn_config = ReturnnConfig(
        config={
            "backend": "torch",
            "batch_size": options.batch_size,
            "accum_grad_multiple_step": options.accum_grad_multiple_step,
            "cleanup_old_models": {
                "keep_last_n": 1,
                "keep_best_n": 0,
                "keep": options.save_epochs,
            },
            "gradient_clip_norm": options.gradient_clip,
            "torch_amp_options": {"dtype": "bfloat16"},
            "num_workers_per_gpu": options.num_workers_per_gpu,
            "stop_on_nonfinite_train_score": options.stop_on_inf_nan_score,
        },
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
    train_job.add_alias(f"training/{options.descriptor}")
    train_job.rqmt["gpu_mem"] = 24

    tk.register_output(f"train_{options.descriptor}/learning_rates", train_job.out_learning_rates)
    tk.register_output(f"train_{options.descriptor}/checkpoint", train_job.out_checkpoints[num_epochs].path)  # type: ignore

    return train_job
