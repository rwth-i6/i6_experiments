import copy
from sisyphus import tk
from i6_core.returnn import ReturnnConfig
from i6_experiments.common.setups.returnn_common.serialization import (
    Collection,
    ExternData,
    Import,
    Network,
    PythonEnlargeStackWorkaroundNonhashedCode,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datasets import (
    GenericDataset,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream

from typing import Dict


datastream_to_nn_data_mapping = {
    "audio_data": "audio_samples",
    "label_data": "speaker_labels",
    "phoneme_data": "phon_labels",
}


def get_training_config(returnn_common_root, training_datasets, network_kwargs, debug=False):
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
        "use_tensorflow": True,
        "tf_log_memory_usage": True,
        "stop_on_nonfinite_train_score": False,  # this might break now with True
        "log_batch_size": True,
        "debug_print_layer_output_template": True,
        "cache_size": "0",
    }

    config = {
        "behavior_version": 12,
        ############
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "accum_grad_multiple_step": 2,
        "gradient_clip": 1,
        "gradient_noise": 0,
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "use_learning_rate_control_always": True,
        "learning_rate_control_error_measure": "dev_score_reconstruction_output",
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": 28000*160,
        "max_seq_length": {"audio_features": 1600*160},
        "max_seqs": 200,
        #############
        "train": training_datasets.train.as_returnn_opts(),
        "dev": training_datasets.cv.as_returnn_opts()
    }
    serializer = get_network_serializer(
        training=True,
        returnn_common_root=returnn_common_root,
        datastreams=training_datasets.datastreams,
        network_kwargs=network_kwargs,
        debug=debug,
    )
    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )
    return returnn_config


def get_forward_config(
        returnn_common_root, forward_dataset: GenericDataset, datastreams, network_kwargs,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_aligner
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """
    config = {
        "behavior_version": 12,
        "forward_batch_size": 28000*160,
        "max_seq_length": {"audio_features": 1600*160},
        "max_seqs": 200,
        "forward_use_search": True,
        "target": "extract_alignment",
        #############
        "eval": forward_dataset.as_returnn_opts()
    }
    serializer = get_network_serializer(
        training=False,
        returnn_common_root=returnn_common_root,
        datastreams=datastreams,
        network_kwargs=network_kwargs,
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])
    return returnn_config


def get_network_serializer(training: bool, returnn_common_root: tk.Path, datastreams: Dict[str, Datastream], network_kwargs, debug: bool = False) -> Collection:
    """

    :param training
    :param returnn_common_root
    :param datastreams:
    :param use_v2:
    :param kwargs:
    :return:
    """
    extern_data = [
        datastream.as_nnet_constructor_data(key)
        for key, datastream in datastreams.items()
    ]

    rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
    rc_extern_data = ExternData(extern_data=extern_data)

    rc_package = "i6_experiments.users.rossenbach.experiments.unnormalized_tts.rc_networks"
    rc_construction_code = Import(rc_package + ".ctc_aligner_v2.construct_network")

    d = copy.deepcopy(network_kwargs)
    if training is False:
        d["training"] = False

    rc_network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map=datastream_to_nn_data_mapping,
        net_kwargs={**d},
    )

    serializer = Collection(
        serializer_objects=[
            rc_recursionlimit,
            rc_extern_data,
            rc_construction_code,
            rc_network,
        ],
        returnn_common_root=returnn_common_root,
        make_local_package_copy=not debug,
        packages={
            rc_package,
        },
    )

    return serializer