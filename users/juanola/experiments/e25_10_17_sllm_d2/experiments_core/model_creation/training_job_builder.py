"""
Builder for training job.
"""

from typing import Any, Dict

from sisyphus import tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from .returnn_config_helpers import get_training_config
from ...configurations.pipeline.training_config import TrainingConfig
from ...configurations.pretrained_models import PretrainedConfig, get_encoder_checkpoint, get_decoder_checkpoint


def create_training_job(training_name: str,
                        datasets: TrainingDatasets,
                        batch_size: int,

                        network_import_path: str,
                        network_args: Dict[str, Any],

                        train_step_module: str,
                        train_epochs: int,

                        training_config: TrainingConfig,
                        pretrained_config: PretrainedConfig,

                        returnn_root: tk.Path) -> ReturnnTrainingJob:
    """
    :param training_name:
    :param datasets:
    :param network_import_path:
    :param network_args:
    :param train_step_module:
    :param train_epochs:
    :param returnn_root: Path to a checked out RETURNN repository
    """
    train_args, training_rqmt = get_training_parameters(network_args, network_import_path,
                                                        returnn_root, train_epochs, train_step_module, batch_size, training_config, pretrained_config)
    returnn_config: ReturnnConfig = get_training_config(training_datasets=datasets, **train_args)
    train_job = ReturnnTrainingJob(returnn_config, **training_rqmt)

    train_job.add_alias(f"{training_name}/training")
    tk.register_output(f"{training_name}/learning_rates", train_job.out_learning_rates)
    return train_job


def get_training_parameters(network_args: dict[str, Any], network_import_path: str,
                            returnn_root: tk.Path, train_epochs: int, train_step_module: str, batch_size: int, train_config_obj: TrainingConfig, pretrained_config: PretrainedConfig) -> tuple[
    dict[str, Any], dict[str, Any]]:
    train_config = { # TODO: lots of settings could be moved to configs.
        **train_config_obj.optimizer.get_optimizer_returnn_config(),
        **train_config_obj.dynamic_lr.get_dynamic_lr_returnn_config(train_epochs),
        "batch_size": batch_size * train_config_obj.batch_size_factor,
        "max_seq_length": {"raw_audio": train_config_obj.max_seq_length_seconds * network_args["sampling_rate"]},
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": 5.0,
        "__num_gpus": train_config_obj.num_gpus,
        "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    }
    if train_config_obj.use_torch_amp:
        train_config["torch_amp"] = train_config_obj.torch_amp
    if train_config_obj.use_grad_scaler:
        train_config["grad_scaler"] = train_config_obj.grad_scaler
    if train_config_obj.random_seed is not None:
        train_config["random_seed"] = train_config_obj.random_seed

    if pretrained_config.pretrained_encoder is not None or pretrained_config.pretrained_decoder is not None:
        preload_from_files = {}
        if pretrained_config.pretrained_encoder is not None:
            preload_from_files["ENCODER"] = {
                "filename": get_encoder_checkpoint(pretrained_config),
                "init_for_train": True,
                "ignore_missing": True,
            }
        if pretrained_config.pretrained_decoder is not None:
            preload_from_files["DECODER"] = {
                "filename": get_decoder_checkpoint(pretrained_config),
                "init_for_train": True,
                "ignore_missing": True,
            }
        train_config["preload_from_files"] = preload_from_files



    train_args = {  # Params for the get_training_config() method #TODO needed this way?
        "config": train_config,

        "network_import_path": network_import_path,
        "net_args": network_args,

        "train_step_module": train_step_module,
        "train_args": {  # train step args - # TODO: could also be extracted in a file
            "aed_loss_scale": 1.0,
            "aux_loss_scales": (1.0, 1.0),
            "label_smoothing": 0.1,
            "label_smoothing_start_epoch": 0,
        },

        "debug": train_config_obj.debug_returnn_param,
        "use_speed_perturbation": True,
    }

    training_rqmt = {  # TODO: extract as config file?
        # Experiment Length
        "num_epochs": train_epochs,
        "time_rqmt": 168,

        # CPU
        "cpu_rqmt": train_config_obj.num_cpus, # can be increased if needed (with care)
        "mem_rqmt": train_config_obj.cpu_memory,  # RAM # can be increased if needed (with care) (max 64??)

        # Other
        "log_verbosity": 5,
        "returnn_root": returnn_root,
    }

    if train_config_obj.num_gpus > 1:
        train_args["config"].update({
            "torch_distributed": {
                "param_sync_step": 100,
                "reduce_type": "param"
            },
            "use_horovod": True,
        })
        training_rqmt.update({
            "distributed_launch_cmd": "torchrun",
            "horovod_num_processes": train_config_obj.num_gpus,
        })
    return train_args, training_rqmt
