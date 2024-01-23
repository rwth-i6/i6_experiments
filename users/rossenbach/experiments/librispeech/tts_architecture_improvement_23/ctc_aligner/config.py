import copy
from sisyphus import tk
from typing import Any, Dict

from i6_core.returnn import ReturnnConfig

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import (
    GenericDataset,
)
from .data import AlignmentTrainingDatasets
from ..serializer import get_network_serializer, get_pytorch_serializer, get_pytorch_serializer_v2, get_pytorch_serializer_v3


def get_training_config(
        returnn_common_root: tk.Path,
        training_datasets: AlignmentTrainingDatasets,
        network_module: str,
        net_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
        pytorch_mode=False,
        v2_mode=False,
        use_custom_engine=False,
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
        "cleanup_old_models": True,
        "stop_on_nonfinite_train_score": False,  # this might break now with True
        "num_workers_per_gpu": 2,
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
        if v2_mode:
            get_serializer = get_pytorch_serializer_v2
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


def get_forward_config(
        returnn_common_root, forward_dataset: GenericDataset, datastreams, network_module, net_args, debug=False, pytorch_mode=False,
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
        "forward_batch_size": 28000,
        "max_seq_length": {"audio_features": 1000},
        "max_seqs": 200,
        "forward_use_search": True,
        "target": "extract_alignment",
        #############
        "eval": forward_dataset.as_returnn_opts()
    }
    get_serializer = get_pytorch_serializer if pytorch_mode else get_network_serializer

    serializer = get_serializer(
        training=False,
        returnn_common_root=returnn_common_root,
        datastreams=datastreams,
        network_module=network_module,
        net_args=net_args,
        debug=debug
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config

def get_pt_forward_config(
        returnn_common_root, forward_dataset: GenericDataset, datastreams, network_module, net_args, debug=False
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """
    config = {
        "batch_size": 28000,
        "max_seqs": 200,
        #############
        "forward": forward_dataset.as_returnn_opts()
    }
    get_serializer = get_pytorch_serializer

    serializer = get_serializer(
        training=False,
        returnn_common_root=returnn_common_root,
        datastreams=datastreams,
        network_module=network_module,
        forward=True,
        net_args=net_args,
        debug=debug
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


def get_pt_raw_forward_config(
        returnn_common_root, forward_dataset: GenericDataset, datastreams, network_module, net_args, debug=False,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """
    config = {
        "batch_size": 56000*200,
        "max_seqs": 200,
        #############
        "forward": forward_dataset.as_returnn_opts()
    }
    get_serializer = get_pytorch_serializer

    serializer = get_serializer(
        training=False,
        returnn_common_root=returnn_common_root,
        datastreams=datastreams,
        network_module=network_module,
        forward=True,
        net_args=net_args,
        debug=debug
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


def get_pt_raw_forward_config_v2(
        forward_dataset: GenericDataset, network_module, net_args, init_args, debug=False,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """
    config = {
        "batch_size": 56000*200,
        "max_seqs": 200,
        #############
        "forward": forward_dataset.as_returnn_opts()
    }
    get_serializer = get_pytorch_serializer_v3

    serializer = get_serializer(
        training=False,
        network_module=network_module,
        forward=True,
        net_args=net_args,
        init_args=init_args,
        debug=debug
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


def get_pt_raw_prior_config(
        training_dataset: AlignmentTrainingDatasets, network_module, net_args, debug=False
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """
    config = {
        "batch_size": 56000*200,
        "max_seqs": 200,
        #############
        "forward": training_dataset.train.as_returnn_opts()
    }
    get_serializer = get_pytorch_serializer_v3

    serializer = get_serializer(
        training=False,
        network_module=network_module,
        prior=True,
        net_args=net_args,
        debug=debug
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config



def get_pt_raw_search_config(
        returnn_common_root, forward_dataset: GenericDataset, datastreams, network_module, net_args, debug=False, search_args=None,
):
    """
    JUST FOR TESTING

    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """
    config = {
        "batch_size": 56000*200,
        "max_seqs": 200,
        #############
        "forward": forward_dataset.as_returnn_opts()
    }
    get_serializer = get_pytorch_serializer_v2

    serializer = get_serializer(
        training=False,
        returnn_common_root=returnn_common_root,
        datastreams=datastreams,
        network_module=network_module,
        search=True,
        net_args=net_args,
        debug=debug,
        search_args=search_args,
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


