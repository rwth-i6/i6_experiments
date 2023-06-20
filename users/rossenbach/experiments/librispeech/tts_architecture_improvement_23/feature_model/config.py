import copy
from sisyphus import tk
from typing import Any, Dict

from i6_core.returnn import ReturnnConfig

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import (
    GenericDataset,
)
from .data import FeatureModelTrainingDatasets
from ..serializer import get_pytorch_serializer_v2


def get_training_config(
        returnn_common_root: tk.Path,
        training_datasets: FeatureModelTrainingDatasets,
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
        "stop_on_nonfinite_train_score": False,  # this might break now with True
    }

    base_config = {
        #############
        "train": training_datasets.train.as_returnn_opts(),
        "dev": training_datasets.cv.as_returnn_opts()
    }
    config = {**base_config, **copy.deepcopy(config)}

    get_serializer = get_pytorch_serializer_v2
    post_config["backend"] = "torch"

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


def get_pt_raw_forward_config(
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
        forward=True,
        net_args=net_args,
        debug=debug
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


