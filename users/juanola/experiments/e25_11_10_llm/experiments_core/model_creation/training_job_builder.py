"""
Builder for training job.
"""

from typing import Any, Dict

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from .returnn_config_helpers import get_training_config
from ...configurations import optimizer_configs, learning_rate_configs


def create_training_job(training_name: str,
                        datasets: TrainingDatasets,
                        num_gpus: int,

                        network_module: str,
                        network_args: Dict[str, Any],

                        train_step_module: str,
                        train_epochs: int,

                        debug_returnn_param: bool,

                        returnn_root: tk.Path) -> ReturnnTrainingJob:
    """
    :param training_name:
    :param datasets:
    :param num_gpus:
    :param network_module:
    :param network_args:
    :param train_step_module:
    :param train_epochs:
    :param debug_returnn_param:
    :param returnn_root: Path to a checked out RETURNN repository
    """
    train_args, training_rqmt = get_training_parameters(num_gpus, debug_returnn_param, network_args, network_module,
                                                        returnn_root, train_epochs, train_step_module)
    returnn_config: ReturnnConfig = get_training_config(training_datasets=datasets, **train_args)
    train_job = ReturnnTrainingJob(returnn_config, **training_rqmt)

    train_job.add_alias(f"{training_name}/training")
    tk.register_output(f"{training_name}/learning_rates", train_job.out_learning_rates)
    return train_job


def get_training_parameters(num_gpus: int, debug_returnn_param: bool, network_args: dict[str, Any], network_module: str,
                            returnn_root: tk.Path, train_epochs: int, train_step_module: str) -> tuple[dict[
    str, Any], dict[str, Any]]:
    # Some values
    batch_size_factor = 160
    batch_size = 15_000

    train_config = {
        **optimizer_configs.v1,
        **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(
            n_ep=train_epochs,
        ),
        "batch_size": batch_size * batch_size_factor,
        "max_seq_length": {"raw_audio": 19.5 * network_args["sampling_rate"]},  # 19.5 seconds
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": 5.0,
        "__num_gpus": num_gpus,
        "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        "torch_amp": "bfloat16", # only for gpus > 11gb
    }

    train_args = { # Params for the get_training_config() method #TODO needed this way?
        "config": train_config,

        "network_module": network_module,
        "net_args": network_args,

        "train_step_module": train_step_module,
        "train_args": {  # TODO: could also be extracted in a file
            "aed_loss_scale": 1.0,
            "aux_loss_scales": (1.0, 1.0),
            "label_smoothing": 0.1,
            "label_smoothing_start_epoch": 0,
        },

        "debug": debug_returnn_param,
        "use_speed_perturbation": True,
    }

    training_rqmt = {  # TODO: extract as config file?
        # Experiment Length
        "num_epochs": train_epochs,
        "time_rqmt": 168,

        # CPU
        "cpu_rqmt": 6,
        "mem_rqmt": 24,

        # Other
        "log_verbosity": 5,
        "returnn_root": returnn_root,
    }

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
    return train_args, training_rqmt
