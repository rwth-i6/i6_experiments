"""
Builder for training job.
"""

from typing import Any, Dict

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from .returnn_config_helpers import get_training_config
from ..data.dataset_commons import TrainingDatasets


def create_training_job(training_name: str,
                        datasets: TrainingDatasets,
                        train_args: Dict[str, Any],
                        num_epochs: int,
                        returnn_root: tk.Path) -> ReturnnTrainingJob:
    """
    :param training_name:
    :param datasets:
    :param train_args:
    :param num_epochs:
    :param returnn_root: Path to a checked out RETURNN repository
    """
    # TODO: separate method in 2 (1 is returnn config creation, other is job creation)

    training_rqmt = {  # TODO: extract as config file?
        # Experiment Length
        "num_epochs": num_epochs,
        "time_rqmt": 168,

        # CPU
        "cpu_rqmt": 6,
        "mem_rqmt": 24,

        # Other
        "log_verbosity": 5,
        "returnn_root": returnn_root,
    }

    # TODO: don't like how this is done. Maybe flag outside the dict?
    num_gpus = train_args["config"].pop("__num_gpus", 1)
    if num_gpus > 1:
        train_args["config"].update({
            "torch_distributed": {
                "param_sync_step": 100,
                "reduce_type": "param"
            },
            "use_horovod": True,
        })
        training_rqmt.update({
            "distributed_launch_cmd": "torchrun",
            "horovod_num_processes": num_gpus,
            "mem_rqmt": 20  # ??
        })

    returnn_config: ReturnnConfig = get_training_config(training_datasets=datasets, **train_args)

    train_job = ReturnnTrainingJob(returnn_config, **training_rqmt)
    train_job.rqmt["gpu_mem"] = 48  # TODO: should come from config file also...
    train_job.add_alias(f"{training_name}/training")
    tk.register_output(f"{training_name}/learning_rates", train_job.out_learning_rates)
    return train_job
