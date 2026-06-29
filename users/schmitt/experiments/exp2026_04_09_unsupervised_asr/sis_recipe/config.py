"""
Universal helpers to create configuration objects (i6_core ReturnnConfig) for RETURNN training/forwarding
"""

import copy
from typing import Any, Dict, Optional, List, Sequence

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from i6_experiments.users.schmitt.returnn.serialization import (
    ReturnnConfigWithNewSerialization,
)
from i6_experiments.common.setups.returnn.datastreams.base import Datastream

from .data.common import TrainingDatasets
from .serializer import serialize_training, serialize_forward


def get_training_config(
    training_datasets: TrainingDatasets,
    network_module: str,
    train_step_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    train_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    use_speed_perturbation: bool = False,
    post_config: Optional[Dict[str, Any]] = None,
    python_prolog: Optional[List[str]] = None,
    train_step_import_as: Optional[str] = None,
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
        # NOTE: stop_on_nonfinite_train_score is functional (it decides whether a non-finite
        # train score aborts training), so it must NOT live in the non-hashed post_config.
        # RETURNN defaults it to True; set it explicitly in the (hashed) training config when
        # a specific experiment needs a different value.
        "backend": "torch",
        "torch_log_memory_usage": True,
        "watch_memory": True,
        "log_batch_size": True,
        "use_tensorboard": True,
        "log_grad_norm": True,
    }

    base_config = {
        #############
        "train": copy.deepcopy(training_datasets.train.as_returnn_opts()),
        "eval_datasets": {key: dataset.as_returnn_opts() for key, dataset in training_datasets.eval_datasets.items()},
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config = {**base_post_config, **copy.deepcopy(post_config or {})}

    default_data_key = config.get("default_data_key", "audio")
    default_target_key = config.get("default_target_key", "text")
    extern_data = {k: v.as_returnn_extern_data_opts() for k, v in training_datasets.datastreams.items()}
    config.update(
        {
            "default_data_key": default_data_key,
            "default_target_key": default_target_key,
        }
    )

    serializer = serialize_training(
        network_module=network_module,
        train_step_module=train_step_module,
        net_args=net_args,
        train_args=train_args,
        unhashed_net_args=unhashed_net_args,
        extern_data=extern_data,
        train_step_import_as=train_step_import_as,
    )

    if use_speed_perturbation:
        from i6_experiments.users.zeyer.speed_pert.librosa_config import (
            speed_pert_librosa_config,
        )

        config["train"]["dataset"]["audio"]["pre_process"] = speed_pert_librosa_config
    else:
        config.pop("speed_pert_discrete_values", None)

    python_prolog = (python_prolog or []) + [serializer]

    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        python_prolog=python_prolog,
    )
    returnn_config = ReturnnConfigWithNewSerialization.from_cfg(returnn_config)
    return returnn_config


def get_forward_config(
    config: Dict[str, Any],
    network_module: str,
    extra_config: ReturnnConfig,
    net_args: Dict[str, Any],
    decoder: str,
    callback_module: str,
    decoder_args: Dict[str, Any],
    datastreams: Dict[str, Datastream],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    add_text_to_extern_data: bool = False,
    callback_opts: Optional[Dict[str, Any]] = None,
    extern_data: Optional[Dict[str, Any]] = None,
    base_config: Optional[Dict[str, Any]] = None,
    vocab_key: Optional[str] = None,
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
    post_config = {
        "torch_log_memory_usage": True,
        "watch_memory": True,
    }

    if base_config is None:
        base_config = {}
    # changeing these does change the hash
    base_config = {
        "max_seqs": 200,
        **base_config,
    }
    config = {**base_config, **config}
    post_config["backend"] = "torch"

    default_data_key = config.get("default_data_key", "audio")
    default_target_key = config.get("default_target_key", "text")
    # which datastream provides the vocab to decode hypotheses / references. Defaults to the text
    # target; for same-modality reconstruction (e.g. audio->audio) the output is a different
    # modality, so its datastream (e.g. "data") must be used instead.
    vocab_key = vocab_key or default_target_key
    if extern_data is None:
        extern_data = {
            default_data_key: datastreams[default_data_key].as_returnn_extern_data_opts(),
        }

    if add_text_to_extern_data:
        label_datastream = datastreams[default_target_key]
        extern_data[default_target_key] = {
            "dim": label_datastream.vocab_size,
            "sparse": True,
            # important: deepcopy. when extern_data is serialized, path objects (e.g. SPM model file) are converted to
            # strings. we don't want this to affect the original dictionary object
            "vocab": label_datastream.as_returnn_targets_opts(),
        }
        config.update(
            {
                "default_target_key": default_target_key,
            }
        )

    serializer = serialize_forward(
        network_module=network_module,
        net_args=net_args,
        unhashed_net_args=unhashed_net_args,
        forward_module=decoder,
        callback_module=callback_module,
        forward_init_args=decoder_args,
        extern_data=extern_data,
        vocab_opts=datastreams[vocab_key].as_returnn_targets_opts(),
        callback_opts=callback_opts,
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_prolog=[serializer])
    returnn_config.update(extra_config)

    returnn_config = ReturnnConfigWithNewSerialization.from_cfg(returnn_config)

    return returnn_config
