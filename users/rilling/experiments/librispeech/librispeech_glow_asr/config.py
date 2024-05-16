import copy
from sisyphus import tk
from typing import Any, Dict

from i6_core.returnn import ReturnnConfig

from .data import TrainingDatasets
from .serializer import get_pytorch_serializer_v3

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset


def get_training_config(
        training_datasets: TrainingDatasets,
        network_module: str,
        net_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
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
        "stop_on_nonfinite_train_score": True,  # this might break now with True
        "allow_missing_optimizer_checkpoint": True,
    }

    base_config = {
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "learning_rates": [0.0001],
        #############
        "batch_size": 300 * 16000,  # batch size in second
        "max_seq_length": {"audio_features": 25 * 16000},  # max seq len in seconds
        "max_seqs": 60,
        #############
        "train": training_datasets.train.as_returnn_opts(),
        "dev": training_datasets.cv.as_returnn_opts()
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    serializer = get_pytorch_serializer_v3(
        network_module=network_module,
        net_args=net_args,
        debug=debug,
        use_custom_engine=use_custom_engine
    )
    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )
    return returnn_config


def get_prior_config(
        training_datasets: TrainingDatasets,
        network_module: str,
        net_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
        use_custom_engine=False,
        **kwargs,
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
    }

    base_config = {
        #############
        "batch_size": 50000 * 160,
        "max_seqs": 60,
        #############
        "forward": training_datasets.prior.as_returnn_opts()

    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    serializer = get_pytorch_serializer_v3(
        network_module=network_module,
        net_args=net_args,
        debug=debug,
        use_custom_engine=use_custom_engine,
        prior=True,
    )
    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )
    return returnn_config

def get_inv_config(
        training_datasets: TrainingDatasets,
        network_module: str,
        net_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
        use_custom_engine=False,
        **kwargs,
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
    }

    base_config = {
        #############
        "batch_size": 50000 * 160,
        "max_seqs": 60,
        #############
        "forward": training_datasets.devtrain.as_returnn_opts()

    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    serializer = get_pytorch_serializer_v3(
        network_module=network_module,
        net_args=net_args,
        debug=debug,
        use_custom_engine=use_custom_engine,
        invertibility=True
    )
    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )
    return returnn_config

def get_search_config(
        network_module: str,
        net_args: Dict[str, Any],
        search_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
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
    }
    base_config = {
        #############
        "batch_size": 18000 * 160,
        "max_seqs": 60,
        #############
        # dataset is added later in the pipeline
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config["backend"] = "torch"

    serializer = get_pytorch_serializer_v3(
        network_module=network_module,
        net_args=net_args,
        debug=debug,
        use_custom_engine=use_custom_engine,
        search=True,
        search_args=search_args
    )
    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )
    return returnn_config
