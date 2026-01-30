"""
Universal helpers to create configuration objects (i6_core ReturnnConfig) for RETURNN training/forwarding
"""
import copy
from typing import Any, Dict, Optional, List

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection as TorchCollection,
)

from i6_experiments.common.setups.returnn.serialization import get_serializable_config
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datasets import Dataset

from i6_experiments.common.setups.serialization import Import
from .data.common import TrainingDatasets
from ..serializer import serialize_training, serialize_forward, PACKAGE
from ..serializer_v2 import ReturnnConfigWithNewSerialization


def get_training_config(
    training_datasets: TrainingDatasets,
    network_module: str,
    train_step_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    train_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    include_native_ops=False,
    debug: bool = False,
    use_speed_perturbation: bool = False,
    post_config: Optional[Dict[str, Any]] = None,
    python_prolog: Optional[List] = None,
    use_v2_serialization: bool = False,
) -> ReturnnConfig:
    """
    Get a generic config for training a model

    :param training_datasets: datasets for training
    :param network_module: path to the pytorch config file containing Model
    :param net_args: extra arguments for constructing the PyTorch model
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :param config: config arguments for RETURNN
    :param debug: run training in debug mode (linking from recipe instead of copy)
    :param use_speed_perturbation: Use speedperturbation in the training
    :param post_config: Add non-hashed arguments for RETURNN
    """

    # changing these does not change the hash
    base_post_config = {
        "stop_on_nonfinite_train_score": True,
        "backend": "torch",
        "torch_log_memory_usage": True,
        "log_batch_size": True,
        "use_tensorboard": True,
        "log_grad_norm": True,
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
        },
    }

    base_config = {
        #############
        "train": copy.deepcopy(training_datasets.train.as_returnn_opts()),
        "dev": training_datasets.cv.as_returnn_opts(),
        "eval_datasets": {"devtrain": training_datasets.devtrain.as_returnn_opts()},
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config = {**base_post_config, **copy.deepcopy(post_config or {})}

    extern_data = {
        k: v.as_returnn_extern_data_opts() for k, v in training_datasets.datastreams.items()
    }
    # if set(training_datasets.datastreams.keys()) == {"features", "labels"}:
    #     extern_data = {
    #         "data": training_datasets.datastreams["features"].as_returnn_extern_data_opts(),
    #         "phon_indices": training_datasets.datastreams["labels"].as_returnn_extern_data_opts()
    #     }
    # else:
    #     data_keys = list(training_datasets.datastreams.keys())
    #     assert len(data_keys) == 1
    #     extern_data = {
    #         "data": training_datasets.datastreams[data_keys[0]].as_returnn_extern_data_opts(),
    #     }

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

    if python_prolog is None:
        python_prolog = []

    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_prolog=[serializer] + python_prolog, python_epilog=None
    )
    if use_v2_serialization:
        returnn_config = ReturnnConfigWithNewSerialization.from_cfg(returnn_config)
    else:
        returnn_config = get_serializable_config(returnn_config, serialize_dim_tags=False)
    return returnn_config


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

    # changing these does not change the hash
    post_config = {
        "num_workers_per_gpu": 2,
    }

    base_config = {
        #############
        "batch_size": 500 * 16000,
        "max_seqs": 240,
        #############
        "forward": copy.deepcopy(training_datasets.prior.as_returnn_opts()),
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

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
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    return returnn_config


def get_forward_config(
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    decoder: str,
    callback_module: str,
    decoder_args: Dict[str, Any],
    datastreams: Dict,
    unhashed_decoder_args: Optional[Dict[str, Any]] = None,
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> ReturnnConfig:
    """
    Get a generic config for forwarding

    :param network_module: path to the pytorch config file containing Model
    :param net_args: extra arguments for constructing the PyTorch model
    :param decoder: which (python) file to load which defines the forward, forward_init and forward_finish functions
    :param decoder_args: extra arguments to pass to forward_init
    :param config: config arguments for RETURNN
    :param unhashed_decoder_args: unhashed extra arguments for the forward init
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :param debug: run training in debug mode (linking from recipe instead of copy)
    """

    # changing these does not change the hash
    post_config = {}

    # changeing these does change the hash
    base_config = {
        "batch_size": 9_000,
        "max_seqs": 200,
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    # extern_data = {
    #     # TODO: do not hardcode
    #     "data": {
    #         "shape": (None,),
    #         "dim": 128,
    #         "sparse": True,
    #     }
    # }

    data_keys = datastreams.keys()
    extern_data = {
        data_key: datastreams[data_key].as_returnn_extern_data_opts() for data_key in data_keys
    }

    serializer = serialize_forward(
        network_module=network_module,
        net_args=net_args,
        unhashed_net_args=unhashed_net_args,
        forward_module=decoder,
        callback_module=callback_module,
        forward_init_args=decoder_args,
        unhashed_forward_init_args=unhashed_decoder_args,
        debug=debug,
        extern_data=extern_data,
        vocab_opts=datastreams["labels"].as_returnn_targets_opts(),
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    return returnn_config
