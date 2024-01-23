import copy
from sisyphus import tk
from typing import Any, Dict

from i6_core.returnn import ReturnnConfig

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import (
    GenericDataset,
)
from .data import TrainingDataset
from ..serializer import get_network_serializer, get_pytorch_serializer


def get_training_config(
        returnn_common_root: tk.Path,
        training_datasets: TrainingDataset,
        network_module: str,
        net_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
        pytorch_mode=False,
        use_custom_engine=False,
        keep_epochs: set=None
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN training config
    """

    # changing these does not change the hash
    post_config = {
        "cleanup_old_models": True if keep_epochs is None else {"keep": keep_epochs},
        "stop_on_nonfinite_train_score": True,  # this might break now with True
        "allow_missing_optimizer_checkpoint": True,
    }

    base_config = {
        #############
        "train": training_datasets.train.as_returnn_opts(),
        "dev": training_datasets.cv.as_returnn_opts()
    }
    config = {**base_config, **copy.deepcopy(config)}

    if pytorch_mode:
        get_serializer = get_pytorch_serializer
        post_config["backend"] = "torch"
    else:
        get_serializer = get_network_serializer
        post_config["backend"] = "tensorflow"

    serializer = get_serializer(
        training=True,
        returnn_common_root=returnn_common_root,
        datastreams=training_datasets.datastreams,
        network_module=network_module,
        net_args=net_args,
        debug=debug,
        use_custom_engine=use_custom_engine
    )
    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )
    return returnn_config


def get_extract_durations_forward__config(
        returnn_common_root, forward_dataset: GenericDataset, network_module, net_args, debug=False, pytorch_mode=False,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """
    config = {
        "behavior_version": 16,
        "batch_size": 28000,
        "max_seq_length": {"audio_features": 1000},
        "max_seqs": 200,
        "forward_use_search": True,
        #############
        "eval": forward_dataset.joint.as_returnn_opts()
    }
    get_serializer = get_pytorch_serializer if pytorch_mode else get_network_serializer

    serializer = get_serializer(
        training=False,
        returnn_common_root=returnn_common_root,
        datastreams=forward_dataset.datastreams,
        network_module=network_module,
        net_args=net_args,
        forward=True,
        debug=debug
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config

def get_forward_config(
        returnn_common_root, forward_dataset: GenericDataset, network_module, net_args, config, debug=False, pytorch_mode=False, forward_args={}, target="audio", train_data=False
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """
    base_config = {
        "behavior_version": 16,
        "forward_use_search": True,
        #############
        "forward": forward_dataset.cv.as_returnn_opts() if not train_data else forward_dataset.train.as_returnn_opts()
    }

    config = {**base_config, **copy.deepcopy(config)}
    get_serializer = get_pytorch_serializer if pytorch_mode else get_network_serializer

    serializer = get_serializer(
        training=False,
        returnn_common_root=returnn_common_root,
        datastreams=forward_dataset.datastreams,
        network_module=network_module,
        net_args=net_args,
        forward_args=forward_args,
        forward=True,
        debug=debug,
        target=target
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


