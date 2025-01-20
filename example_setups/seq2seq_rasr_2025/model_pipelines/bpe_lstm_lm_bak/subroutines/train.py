from i6_core.returnn.config import ReturnnConfig, WriteReturnnConfigJob
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.common.setups.serialization import ExternalImport, Import
from sisyphus import tk

from ....tools import returnn_python_exe, minireturnn_root
from ..pytorch_modules import LstmLm, LstmLmConfig
from .configs import TrainRoutineConfig
from .returnn_steps.train import train_step
from .util import get_model_serializers, recipe_imports


def train(config: TrainRoutineConfig, model_config: LstmLmConfig) -> ReturnnTrainingJob:
    model_serializers = get_model_serializers(model_class=LstmLm, model_config=model_config)
    num_epochs = config.num_epochs

    train_returnn_config = ReturnnConfig(
        config={
            "backend": "torch",
            "batch_size": config.batch_size,
            "cleanup_old_models": {
                "keep_last_n": 1,
                "keep_best_n": 1,
                "keep": [],
            },
            "optimizer": {
                "class": "RAdam",
            },
            "gradient_clip_global_norm": config.gradient_clip,
            "torch_amp_options": {"dtype": "float16"},
            "num_workers_per_gpu": 1,
            "stop_on_nonfinite_train_score": False,
        },
        python_prolog=[
            recipe_imports,
            ExternalImport(minireturnn_root),
        ],
        python_epilog=[
            *model_serializers,
            Import(f"{train_step.__module__}.{train_step.__name__}"),
        ],  # type: ignore
        sort_config=False,
    )

    train_returnn_config.update(config.lr_config.get_returnn_config())
    train_returnn_config.update(config.train_data_config.get_returnn_data("train"))
    train_returnn_config.update(config.cv_data_config.get_returnn_data("dev"))

    tk.register_output(
        "train/returnn.config",
        WriteReturnnConfigJob(train_returnn_config).out_returnn_config_file,
    )

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
    train_job.rqmt["gpu_mem"] = 11

    tk.register_output("train/learning_rates", train_job.out_learning_rates)

    return train_job
