import copy
from sisyphus import tk
from typing import Any, Dict

from i6_core.returnn import ReturnnConfig

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import (
    GenericDataset,
)
from .data import TrainingDataset
from .serializer import get_serializer


def get_training_config(
    training_datasets: TrainingDataset,
    network_module: str,
    net_args: Dict[str, Any],
    config: Dict[str, Any],
    debug: bool = False,
    training_args: Dict[str, Any]={},
    use_custom_engine=False,
    keep_epochs: set = None,
    asr_cv_set: bool = False
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
        "backend": "torch"
    }

    base_config = {
        #############
        "train": training_datasets.train.as_returnn_opts(),
        "dev": training_datasets.cv.as_returnn_opts() if not asr_cv_set else training_datasets.cv_asr.as_returnn_opts(),
        "eval_datasets": {
            "devtrain": training_datasets.devtrain.as_returnn_opts()
        }
    }
    config = {**base_config, **copy.deepcopy(config)}

    serializer = get_serializer(
        training=True,
        datastreams=training_datasets.datastreams,
        network_module=network_module,
        net_args=net_args,
        training_args=training_args,
        debug=debug,
        use_custom_engine=use_custom_engine,
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    return returnn_config


def get_extract_durations_forward__config(
    forward_dataset: GenericDataset,
    network_module,
    net_args,
    debug=False,
    pytorch_mode=False,
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
        "eval": forward_dataset.joint.as_returnn_opts(),
    }

    serializer = get_serializer(
        training=False,
        datastreams=forward_dataset.datastreams,
        network_module=network_module,
        net_args=net_args,
        forward=True,
        debug=debug,
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


def get_forward_config(
    forward_dataset: GenericDataset,
    network_module,
    net_args,
    config,
    debug=False,
    pytorch_mode=False,
    forward_args={},
    target="audio",
    train_data=False,
    joint_data=False,
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
        "batch_size": 100 * 16000,
        #############
        "forward": forward_dataset.devtrain.as_returnn_opts()
        if not (train_data or joint_data)
        else (forward_dataset.joint.as_returnn_opts() if joint_data else forward_dataset.train.as_returnn_opts()),
    }

    config = {**base_config, **copy.deepcopy(config)}

    serializer = get_serializer(
        training=False,
        datastreams=forward_dataset.datastreams,
        network_module=network_module,
        net_args=net_args,
        forward_args=forward_args,
        forward=True,
        debug=debug,
        target=target,
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


def get_search_config(
    network_module: str,
    net_args: Dict[str, Any],
    search_args: Dict[str, Any],
    config: Dict[str, Any],
    debug: bool = False,
    use_custom_engine=False,
    target="text"
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
        "batch_size": 18000 * 160,
        "max_seqs": 60,
        #############
        # dataset is added later in the pipeline
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    serializer = get_serializer(
        network_module=network_module,
        net_args=net_args,
        debug=debug,
        use_custom_engine=use_custom_engine,
        forward=True,
        forward_args=search_args,
        target=target
    )
    returnn_config = ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
    return returnn_config
