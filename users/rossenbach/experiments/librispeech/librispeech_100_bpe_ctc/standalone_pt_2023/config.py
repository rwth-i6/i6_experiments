import copy
from sisyphus import tk
from typing import Any, Dict

from i6_core.returnn import ReturnnConfig

from .data import TrainingDatasets
from .serializer import get_pytorch_serializer_v3

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset


PACKAGE = __package__


def get_training_config(
        training_datasets: TrainingDatasets,
        network_module: str,
        net_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
        use_custom_engine=False,
        speed_perturbation=False,
        with_devtrain=False,
        num_workers=1,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_aligner
    """

    # changing these does not change the hash
    post_config = {
        "cleanup_old_models": True,
        "stop_on_nonfinite_train_score": True,  # this might break now with True
    }

    base_config = {
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "learning_rates": [0.001],
        #############
        "batch_size": 18000 * 160,
        "max_seq_length": {"audio_features": 1600 * 160},
        "max_seqs": 60,
        #############
        "train": copy.deepcopy(training_datasets.train.as_returnn_opts()),
        "dev": training_datasets.cv.as_returnn_opts()
    }
    config = {**base_config, **copy.deepcopy(config)}
    if with_devtrain:
        config["eval_datasets"] = {
            "devtrain": training_datasets.devtrain.as_returnn_opts()
        }
    if num_workers > 1:
        config["num_workers_per_gpu"] = num_workers
    post_config["backend"] = "torch"

    serializer = get_pytorch_serializer_v3(
        network_module=network_module,
        net_args=net_args,
        debug=debug,
        use_custom_engine=use_custom_engine
    )
    python_prolog = None
    if speed_perturbation:
        from i6_experiments.common.setups.returnn_pytorch.serialization import (
            Collection as TorchCollection,
        )
        from i6_experiments.common.setups.serialization import Import
        prolog_serializer = TorchCollection(
            serializer_objects=[Import(
                code_object_path=PACKAGE + ".extra_code.speed_perturbation", unhashed_package_root=PACKAGE
            )]
        )
        python_prolog = [prolog_serializer]
    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_prolog=python_prolog, python_epilog=[serializer]
    )
    return returnn_config


def get_search_config(
        network_module: str,
        net_args: Dict[str, Any],
        search_args: Dict[str, Any],
        config: Dict[str, Any],
        debug: bool = False,
        use_custom_engine=False,
        **kwargs,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_aligner
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
