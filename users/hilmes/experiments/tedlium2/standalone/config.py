"""
Universal helpers to create configuration objects (i6_core ReturnnConfig) for RETURNN training/forwarding
"""
import copy
from typing import Any, Dict, Optional, List

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection as TorchCollection,
)
from i6_experiments.common.setups.serialization import Import
from .data.common import TrainingDatasets
from .serializer import serialize_training, serialize_forward, PACKAGE, serialize_quant


def get_training_config(
    training_datasets: TrainingDatasets,
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    use_speed_perturbation: bool = False,
    post_config: Optional[Dict[str, Any]] = None,
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
    base_post_config = {"stop_on_nonfinite_train_score": True, "num_workers_per_gpu": 2, "backend": "torch"}

    base_config = {
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
        },
        #############
        "train": copy.deepcopy(training_datasets.train.as_returnn_opts()),
        "dev": training_datasets.cv.as_returnn_opts(),
        "eval_datasets": {"devtrain": training_datasets.devtrain.as_returnn_opts()},
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config = {**base_post_config, **copy.deepcopy(post_config or {})}

    serializer = serialize_training(
        network_module=network_module,
        net_args=net_args,
        unhashed_net_args=unhashed_net_args,
        debug=debug,
    )
    python_prolog = None

    # TODO: maybe make nice (if capability added to RETURNN itself)
    if use_speed_perturbation:
        prolog_serializer = TorchCollection(
            serializer_objects=[
                Import(
                    code_object_path=PACKAGE + ".extra_code.speed_perturbation.legacy_speed_perturbation",
                    unhashed_package_root=PACKAGE,
                )
            ]
        )
        python_prolog = [prolog_serializer]
        config["train"]["datasets"]["zip_dataset"]["audio"]["pre_process"] = CodeWrapper("legacy_speed_perturbation")

    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_prolog=python_prolog, python_epilog=[serializer]
    )
    return returnn_config


def get_prior_config(
    training_datasets: TrainingDatasets,  # TODO: replace by single dataset
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    import_memristor: bool = False,
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
        import_memristor=import_memristor,
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    return returnn_config


def get_forward_config(
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    decoder: str,
    decoder_args: Dict[str, Any],
    unhashed_decoder_args: Optional[Dict[str, Any]] = None,
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    import_memristor: bool = False,
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
        "batch_size": 1000 * 16000,
        "max_seqs": 240,
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    serializer = serialize_forward(
        network_module=network_module,
        net_args=net_args,
        unhashed_net_args=unhashed_net_args,
        forward_module=decoder,
        forward_init_args=decoder_args,
        unhashed_forward_init_args=unhashed_decoder_args,
        import_memristor=import_memristor,
        debug=debug,
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    return returnn_config


def get_static_quant_config(
    training_datasets: TrainingDatasets,
    network_module: str,
    net_args: Dict[str, Any],
    quant_args: Dict[str, Any],
    config: Dict[str, Any],
    num_samples: int,
    dataset_seed: int,
    dataset_filter_args: Optional[Dict[str, Any]],
    debug: bool = False,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN training config
    """

    # changing these does not change the hash
    post_config = {}

    base_config = {
        #############
        "batch_size": 50000 * 160,
        "max_seqs": 240,
        #############
        "forward": copy.deepcopy(training_datasets.prior.as_returnn_opts()),
    }
    base_config["forward"]["seq_ordering"] = "random"
    base_config["forward"]["datasets"]["zip_dataset"]["fixed_random_subset"] = num_samples
    base_config["forward"]["datasets"]["zip_dataset"]["fixed_random_subset_seed"] = dataset_seed
    if dataset_filter_args is not None:
        base_config["forward"]["datasets"]["zip_dataset"]["random_subset_filter_args"] = dataset_filter_args
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"
    assert net_args.keys().isdisjoint(quant_args.keys())
    serializer = serialize_forward(
        network_module=network_module, net_args=net_args | quant_args, debug=debug, forward_step_name="static_quant"
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    return returnn_config


def get_onnx_export_config(
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    unhashed_net_args: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> ReturnnConfig:
    """
    Get a generic config for forwarding

    :param network_module: path to the pytorch config file containing Model
    :param net_args: extra arguments for constructing the PyTorch model
    :param decoder: which (python) file to load which defines the forward, forward_init and forward_finish functions
    :param config: config arguments for RETURNN
    :param unhashed_net_args: unhashed extra arguments for constructing the PyTorch model
    :param debug: run training in debug mode (linking from recipe instead of copy)
    """

    # changing these does not change the hash
    post_config = {}

    # changeing these does change the hash
    base_config = {
        "batch_size": 1000 * 16000,
        "max_seqs": 240,
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    serializer = serialize_quant(
        network_module=network_module,
        net_args=net_args,
        unhashed_net_args=unhashed_net_args,
        debug=debug,
        export_step_name="export",
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    return returnn_config
