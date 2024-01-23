import copy
from typing import Any, Dict, Optional, List

import torch

from i6_core.returnn import ReturnnConfig
from i6_models.config import ModelConfiguration

from i6_experiments.users.gruev.pytorch.ctc_data import TrainingDatasets
from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset

from i6_experiments.users.gruev.pytorch.serializers.basic import (
    get_basic_pt_network_train_serializer,
    get_basic_pt_network_recog_serializer,
)


PACKAGE = __package__.rsplit(".", 1)[0]
MODEL_IMPORT_PATH = "models.i6_base_model"


def get_pt_train_config(
    training_datasets: TrainingDatasets,
    config: Dict[str, Any],
    model_config: ModelConfiguration,
    python_prolog: Optional[List] = None,
    debug: bool = False,
):
    """
    :param training_datasets: datasets for training
    :param config: base config to set optimizer, learning rates, etc.
    :param model_config: model settings
    :param python_prolog: add e.g. speed_pert
    :param debug: enable local copy of additional packaged
    :return: RETURNN training config
    """

    # changes to these do not modify the hash
    post_config = {
        "cleanup_old_models": True,
        "stop_on_nonfinite_train_score": False,  # this might break now with True
    }

    base_config = {
        "backend": "torch",
        #############
        "train": training_datasets.train.as_returnn_opts(),
        "dev": training_datasets.cv.as_returnn_opts(),
    }
    config = {**base_config, **copy.deepcopy(config)}

    serializer = get_basic_pt_network_train_serializer(
        module_import_path=f"{PACKAGE}.{MODEL_IMPORT_PATH}.ConformerCTCModel",
        train_step_import_path=f"{PACKAGE}.{MODEL_IMPORT_PATH}.train_step",
        model_config=model_config,
        debug=debug,
    )

    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        python_prolog=python_prolog,
        python_epilog=[serializer],
    )
    return returnn_config


def get_pt_search_config(
    forward_dataset: GenericDataset,
    model_config: ModelConfiguration,
    import_kwargs: Dict[str, Any],
    debug: bool = False,
):
    """
    :param training_datasets: datasets for training
    :param config: base config to set optimizer, learning rates, etc.
    :param model_config: model settings
    :param python_prolog: add e.g. speed_pert
    :param debug: enable local copy of additional packaged
    :return: RETURNN training config
    """
    config = {
        "batch_size": 20_000 * 160,
        "max_seqs": 200,
        #############
        "forward": forward_dataset.as_returnn_opts(),
    }

    serializer = get_basic_pt_network_recog_serializer(
        module_import_path=f"{PACKAGE}.{MODEL_IMPORT_PATH}.ConformerCTCModel",
        recog_step_import_path=f"{PACKAGE}.{MODEL_IMPORT_PATH}",
        model_config=model_config,
        import_kwargs=import_kwargs,
        debug=debug,
    )


    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config
