"""
Universal helpers to create configuration objects (i6_core ReturnnConfig) for RETURNN training/forwarding
"""
import copy
from typing import Any, Dict, Optional

from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.returnn.serialization import get_serializable_config
from .data.common import TrainingDatasets
from .returnn_config_serializer import serialize_training, serialize_forward


def get_training_config(
    training_datasets: TrainingDatasets,
    network_module: str,
    train_step_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    train_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
        include_native_ops: bool = False,
    debug: bool = False,
    use_speed_perturbation: bool = False,
    post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    """
    Get a generic config for training a model

    :param training_datasets: datasets for training
    :param network_module: path to the pytorch config file containing Model
    :param train_step_module: ????? path to the pytorch config file containing TrainingStep
    :param config: config arguments for RETURNN
    :param net_args: extra arguments for constructing the PyTorch model
    :param train_args: ??????? extra arguments for constructing the PyTorch model
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :parm include_native_ops: ????
    :param debug: run training in debug mode (linking from recipe instead of copy)
    :param use_speed_perturbation: Use speed perturbation in the training
    :param post_config: Add non-hashed arguments for RETURNN
    """

    # RC - CONFIG
    base_config = {
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
        },
        "train": copy.deepcopy(training_datasets.train.as_returnn_opts()),
        "dev": training_datasets.cv.as_returnn_opts(),
        "eval_datasets": {"devtrain": training_datasets.devtrain.as_returnn_opts()},
    }
    config = {**base_config, **copy.deepcopy(config)}  # Our base config + parameter config options

    # RC - POST CONFIG
    base_post_config = {"stop_on_nonfinite_train_score": True, "backend": "torch"}
    post_config = {**base_post_config, **copy.deepcopy(post_config or {})}

    # RC - PYTHON PROLOG
    python_prolog = None
    if use_speed_perturbation:  # TODO: maybe make nice (if capability added to RETURNN itself)
        from i6_experiments.users.zeyer.speed_pert.librosa_config import (
            speed_pert_librosa_config,
        )  # TODO: MJ: should be copied and not imported

        # prolog_serializer = TorchCollection(
        #     serializer_objects=[
        #         Import(
        #             code_object_path=PACKAGE + ".extra_code.speed_perturbation.legacy_speed_perturbation",
        #             unhashed_package_root=PACKAGE,
        #         )
        #     ]
        # )
        # python_prolog = [prolog_serializer]
        config["train"]["dataset"]["audio"]["pre_process"] = speed_pert_librosa_config

    # RC - PYTHON EPILOG
    extern_data = {
        "data": {"dim": 1},
        "classes": {
            "dim": training_datasets.datastreams["labels"].vocab_size,
            "sparse": True,
            # important: deepcopy. when extern_data is serialized, path objects (e.g. SPM model file) are converted to
            # strings. we don't want this to affect the original dictionary object
            "vocab": copy.deepcopy(training_datasets.train.dataset.target_options),
        },
    }
    serializer = serialize_training(
        network_module=network_module,
        train_step_module=train_step_module,
        net_args=net_args,
        train_args=train_args,
        unhashed_net_args=unhashed_net_args,
        include_native_ops=include_native_ops,
        debug=debug,
        extern_data=extern_data,
    )

    return get_serializable_config(
        ReturnnConfig(config=config, post_config=post_config, python_prolog=python_prolog, python_epilog=[serializer]),
        serialize_dim_tags=False,
    )


def get_prior_config(
    training_datasets: TrainingDatasets,  # TODO: replace by single dataset
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
):
    """
    Get a generic config for extracting output label priors

    :param training_datasets: datasets for training
    :param network_module: path to the pytorch config file containing Model
    :param config: config arguments for RETURNN
    :param net_args: extra arguments for constructing the PyTorch model
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :param debug: run training in debug mode (linking from recipe instead of copy)
    """
    # RC - CONFIG
    base_config = {
        "batch_size": 500 * 16000,
        "max_seqs": 240,
        "forward": copy.deepcopy(training_datasets.prior.as_returnn_opts()),
    }
    config = {**base_config, **copy.deepcopy(config)}

    # RC - POST CONFIG
    post_config = {
        "num_workers_per_gpu": 2,
        "backend": "torch"
    }

    # RC - PYTHON EPILOG
    serializer = serialize_forward(
        network_module=network_module,
        net_args=net_args,
        unhashed_net_args=unhashed_net_args,
        forward_module=None,  # same as network
        forward_step_name="prior",
        forward_init_args=None,
        unhashed_forward_init_args=None,
        debug=debug,
    )

    return ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])


def get_forward_config(
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    decoder: str,
    decoder_args: Dict[str, Any],
    vocab_opts: Dict,
    unhashed_decoder_args: Optional[Dict[str, Any]] = None,
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> ReturnnConfig:
    """
    Get a generic config for forwarding

    :param network_module: path to the pytorch config file containing Model
    :param config: config arguments for RETURNN
    :param net_args: extra arguments for constructing the PyTorch model
    :param decoder: which (python) file to load which defines the forward, forward_init and forward_finish functions
    :param decoder_args: extra arguments to pass to forward_init
    :param vocab_opts: ????? vocab options
    :param unhashed_decoder_args: unhashed extra arguments for the forward init
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :param debug: run training in debug mode (linking from recipe instead of copy)
    """
    # RC - CONFIG
    base_config = {
        "batch_size": 15_000 * 160,
        "max_seqs": 200,
    }
    config = {**base_config, **copy.deepcopy(config)}

    # RC - POST CONFIG
    post_config = {"backend": "torch"}

    # RC - PYTHON EPILOG
    extern_data = {
        "data": {"dim": 1},
    }
    serializer = serialize_forward(
        network_module=network_module,
        net_args=net_args,
        unhashed_net_args=unhashed_net_args,
        forward_module=decoder,
        forward_init_args=decoder_args,
        unhashed_forward_init_args=unhashed_decoder_args,
        debug=debug,
        extern_data=extern_data,
        vocab_opts=vocab_opts,
    )

    return ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
