"""
Universal helpers to create configuration objects (i6_core ReturnnConfig) for RETURNN training/forwarding
"""
import copy
from typing import Any, Dict, Optional

from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.returnn.serialization import get_serializable_config
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from .returnn_config_serializer import serialize_training, serialize_forward
from ..data.librispeech_lm_utils import get_extern_data_data
from ...constants import DATA_PARAM_NAME, CLASSES_PARAM_NAME


# TODO: make nice and separate


def get_training_config(
        training_datasets: TrainingDatasets,
        network_import_path: str,
        train_step_module: str,
        config: Dict[str, Any],
        net_args: Dict[str, Any],
        train_args: Dict[str, Any],
        unhashed_net_args: Optional[Dict[str, Any]] = None,
        include_native_ops: bool = False,
        debug: bool = False,
        post_config: Optional[Dict[str, Any]] = None,
) -> ReturnnConfig:
    """
    Get a generic config for training a model

    :param include_native_ops:
    :param training_datasets: datasets for training
    :param network_import_path: path to the pytorch config file containing Model
    :param train_step_module: ????? path to the pytorch config file containing TrainingStep
    :param config: config arguments for RETURNN
    :param net_args: extra arguments for constructing the PyTorch model
    :param train_args: ??????? extra arguments for constructing the PyTorch model
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :parm include_native_ops: ????
    :param debug: run training in debug mode (linking from recipe instead of copy)
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
    base_post_config = {
        "stop_on_nonfinite_train_score": True,
        "backend": "torch",

        # For better debugging
        "torch_log_memory_usage": True,  # GPU
        "log_batch_size": True,
        "use_tensorboard": True,
        "log_grad_norm": True,
        "watch_memory": True,  # RAM
    }
    post_config = {**base_post_config, **copy.deepcopy(post_config or {})}

    # RC - PYTHON PROLOG
    python_prolog = None

    # RC - PYTHON EPILOG
    extern_data = {# TODO: all should be encapsulated, not only data
        DATA_PARAM_NAME: get_extern_data_data(),
    }
    serializer = serialize_training(
        network_import_path=network_import_path,
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
        network_import_path: str,
        config: Dict[str, Any],
        net_args: Dict[str, Any],
        unhashed_net_args: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        batch_size: int = 16_000,
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
        "batch_size": batch_size,
        "max_seqs": 240,
        "forward": copy.deepcopy(training_datasets.prior.as_returnn_opts()),
    }
    config = {**base_config, **copy.deepcopy(config)}

    # RC - POST CONFIG
    post_config = {
        "num_workers_per_gpu": 2,
        "backend": "torch",
        "forward_auto_split_batch_on_oom": True,
    }

    forward_step_params = {#TODO: remove
        "beam_size": 12, #TODO: fix this!
        "max_tokens_per_sec": 20,
        "sample_rate": 16_000,
    }

    # RC - PYTHON EPILOG
    serializer = serialize_forward(  # TODO: fix this! 2 more params are needed
        network_import_path=network_import_path,
        net_args=net_args,
        unhashed_net_args=unhashed_net_args,
        forward_step_name="prior_step",
        debug=debug,
        forward_args=forward_step_params,
    )

    return ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])


def get_forward_config(
        network_import_path: str,
        config: Dict[str, Any],
        net_args: Dict[str, Any],
        forward_module: str,
        vocab_opts: Dict,
        forward_method: Optional[str] = None,
        unhashed_net_args: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        forward_args: Dict[str, Any] = None,
) -> ReturnnConfig:
    """
    Get a generic config for forwarding

    :param forward_method:
    :param forward_module:
    :param forward_args:
    :param network_import_path: path to the pytorch config file containing Model
    :param config: config arguments for RETURNN
    :param net_args: extra arguments for constructing the PyTorch model
    :param vocab_opts: ????? vocab options
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :param debug: run training in debug mode (linking from recipe instead of copy)
    """
    # RC - CONFIG
    config = copy.deepcopy(config)

    # RC - POST CONFIG
    post_config = {
        "backend": "torch",
        "forward_auto_split_batch_on_oom": True,
    }

    # RC - PYTHON EPILOG
    extern_data = {# TODO: finish! take the one in training
        DATA_PARAM_NAME: {"dim": 1},
    }
    serializer = serialize_forward(
        network_import_path=network_import_path,
        forward_method=forward_method,
        net_args=net_args,
        extern_data=extern_data,
        vocab_opts=vocab_opts,

        unhashed_net_args=unhashed_net_args,
        forward_module=forward_module,
        debug=debug,
        forward_args=forward_args,
    )

    return ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
