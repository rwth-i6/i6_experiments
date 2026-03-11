"""
Universal helpers to create configuration objects (i6_core ReturnnConfig) for RETURNN training/forwarding
"""
import copy
from typing import Any, Dict, Optional, List

from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.common.setups.returnn.serialization import get_serializable_config
from i6_experiments.common.setups.serialization import PartialImport
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from i6_experiments.users.juanola.returnn.serialization import ReturnnConfigWithNewSerialization
from .returnn_config_serializer import serialize_training, serialize_forward
from ...configurations.pipeline.prior_config import PriorConfig
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
        use_speed_perturbation: bool = False,
        post_config: Optional[Dict[str, Any]] = None,
        use_lora_adapted_weights_method: bool = False,
) -> ReturnnConfig:
    """
    Get a generic config for training a model

    :param training_datasets: datasets for training
    :param network_import_path: path to the pytorch config file containing Model
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
    if use_lora_adapted_weights_method:
        qwen_load_lora_adapted_weights = PartialImport(
            code_object_path="i6_experiments.users.juanola.pretraining.custom_missing_load_functions.qwen_load_lora_adapted_weights",
            import_as="qwen_load_lora_adapted_weights",
            hashed_arguments={},
            unhashed_arguments={},
            unhashed_package_root=None,
        )
        python_prolog = [Collection([qwen_load_lora_adapted_weights])]

    if use_speed_perturbation:  # TODO: maybe make nice (if capability added to RETURNN itself)
        from i6_experiments.users.zeyer.speed_pert.librosa_config import \
            speed_pert_librosa_config  # TODO: warning! external import!
        config["train"]["dataset"]["audio"]["pre_process"] = speed_pert_librosa_config

    # RC - PYTHON EPILOG
    extern_data = {
        DATA_PARAM_NAME: {"dim": 1},
        CLASSES_PARAM_NAME: {
            "dim": training_datasets.datastreams["labels"].vocab_size,
            "sparse": True,
            # important: deepcopy. when extern_data is serialized, path objects (e.g. SPM model file) are converted to
            # strings. we don't want this to affect the original dictionary object
            "vocab": copy.deepcopy(training_datasets.train.dataset.target_options),
        },
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
        training_datasets: TrainingDatasets,
        network_import_path: str,
        net_args: Dict[str, Any],
        prior_config: PriorConfig,
        vocab_opts: Dict,
        forward_module: str,

        config: Dict[str, Any] = {},
        unhashed_net_args: Optional[Dict[str, Any]] = None,
):
    """
    Get a generic config for extracting output label priors

    :param training_datasets: datasets for training
    :param network_import_path: path to the pytorch config file containing Model
    :param config: config arguments for RETURNN
    :param net_args: extra arguments for constructing the PyTorch model
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :param debug: run training in debug mode (linking from recipe instead of copy)
    """

    # RC - CONFIG
    base_config = {
        "batch_size": prior_config.batch_size_factor * prior_config.batch_size,
        "max_seqs": 240,
        "forward_data": copy.deepcopy(training_datasets.train.as_returnn_opts()), # over train!!
    }

    if base_config["forward_data"]["num_workers"] > 4:
        base_config["forward_data"]["num_workers"] = 4

    config = {**base_config, **copy.deepcopy(config)}

    # RC - POST CONFIG
    post_config = {
        "num_workers_per_gpu": 2,
        "backend": "torch",
        "forward_auto_split_batch_on_oom": True,
    }

    forward_step_params = {}

    # RC - PYTHON EPILOG
    extern_data = {
        DATA_PARAM_NAME: {"dim": 1},
    }
    serializer = serialize_forward(
        network_import_path=network_import_path,
        net_args=net_args,
        extern_data=extern_data,
        vocab_opts=vocab_opts,

        unhashed_net_args=unhashed_net_args,
        forward_module=forward_module,
        forward_method=prior_config.forward_method,
        debug=prior_config.debug_returnn_param,
        forward_args=forward_step_params,
        callback_name="ReturnnCollectStatsForwardCallbackV1",
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
    extern_data = {
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

def get_forward_config_v2(
        network_import_path: str,
        net_args: Dict[str, Any],
        forward_module: str,
        forward_method: str,
        callback_name: str,
        decoder_args: Dict[str, Any],
        label_datastream: LabelDatastream,
        unhashed_net_args: Optional[Dict[str, Any]] = None,
        # add_text_to_extern_data: bool = False,
        callback_opts: Optional[Dict[str, Any]] = None,
        extern_data: Optional[Dict[str, Any]] = None,
        base_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        extra_configs: List[ReturnnConfig] = None,
        default_data_key: Optional[str] = None,
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
    if base_config is None:
        base_config = {}
    if extra_configs is None:
        extra_configs = []

    # changing these does not change the hash
    post_config = {
        "torch_log_memory_usage": True,
        "watch_memory": True,
        "backend": "torch",
    }

    config = {
        **base_config
    }

    if extern_data is None:
        extern_data = {
            DATA_PARAM_NAME: {"dim": 1},
        }

    if default_data_key is not None:
        config.update({"default_data_key": default_data_key})

    # if extern_data is None:
    #     extern_data = {
    #         default_data_key: {"shape": (None,)},
    #     }
    #

    # if add_text_to_extern_data:
    #     default_target_key = "text"
    #     extern_data[default_target_key] = { # TODO: adapt?
    #         "dim": label_datastream.vocab_size,
    #         "sparse": True,
    #         # important: deepcopy. when extern_data is serialized, path objects (e.g. SPM model file) are converted to
    #         # strings. we don't want this to affect the original dictionary object
    #         "vocab": label_datastream.as_returnn_targets_opts(),
    #     }
    #     config.update(
    #         {
    #             "default_target_key": default_target_key,
    #         }
    #     )

    serializer = serialize_forward(
        network_import_path=network_import_path,
        net_args=net_args,
        extern_data=extern_data,
        vocab_opts=label_datastream.as_returnn_targets_opts(),

        unhashed_net_args=unhashed_net_args,
        forward_module=forward_module,
        forward_method=forward_method,
        forward_args=decoder_args,

        callback_name=callback_name,
        callback_opts=callback_opts,

        debug=debug,
    )

    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    for extra_returnn_config in extra_configs:
        returnn_config.update(extra_returnn_config)
    return returnn_config
