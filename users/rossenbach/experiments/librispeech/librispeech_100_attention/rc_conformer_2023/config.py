"""

"""
import copy
from dataclasses import dataclass
import os
from sisyphus import tk

from i6_core.returnn import ReturnnConfig
from i6_experiments.common.setups.returnn_common.serialization import (
    Collection,
    ExternData,
    Import,
    Network,
    PythonEnlargeStackWorkaroundNonhashedCode,
    SerializerObject
)
from i6_experiments.common.setups.serialization import (
    CodeFromFunction
)

from .data import TrainingDatasets

from .default_tools import RETURNN_COMMON

# @dataclass()
# class OCLRSchedulerSettings:
#     initialLR = {initial_lr}
#     peakLR = {peak_lr}
#     finalLR = {final_lr}
#     cycleEpoch = {cycle_ep}
#     totalEpoch = {total_ep}
#     nStep = {n_step}
#
# class OCLRScheduler(SerializerObject):
#
#     def __init__(self):
#
#

def speed_perturbation(audio, sample_rate, random_state):
    import librosa
    new_sample_rate = int(sample_rate * (1 + random_state.randint(-1, 2) * 0.1))
    if new_sample_rate != sample_rate:
        audio = librosa.core.resample(audio, sample_rate, new_sample_rate, res_type="kaiser_fast")
    return audio


def get_network_serializer(returnn_common_root: tk.Path, rc_extern_data: ExternData, training: bool, network_args, debug=False) -> Collection:

    rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
    model_type = network_args.pop("model_type")
    rc_construction_code = Import(
        __package__ + ".rc_networks.%s.construct_network" % model_type
    )
    net_func_map = {
        "audio_features": "audio_features",
        "bpe_labels": "bpe_labels",
    }

    rc_network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map=net_func_map,
        net_kwargs={"training": training, **network_args},
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
            __package__ + ".rc_networks",
        },
    )

    return serializer


def get_training_config(
        training_datasets: TrainingDatasets, network_args, debug=False, returnn_common_root: tk.Path = RETURNN_COMMON,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_model
    :param training_datasets: datasets for training
    :param network_args: arguments to be passed to the network construction
    :param debug: set to true to import model code in the job directly from the recipes for debugging
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :return: RETURNN training config
    """
    post_config = {
        "cleanup_old_models": True,
        "use_tensorflow": True,
        "tf_log_memory_usage": True,
        "stop_on_nonfinite_train_score": True,
        "log_batch_size": True,
        "debug_print_layer_output_template": True,
        "cache_size": "0",
    }
    config = {
        "behavior_version": 16,
        ############
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "accum_grad_multiple_step": 2,
        "gradient_clip": 1,
        "gradient_noise": 0,
        "learning_rates": [0.001],
        #############
        "batch_size": 18000 * 160,
        "max_seq_length": {"audio_features": 1600*160},
        "max_seqs": 60,
        #############
        "train": training_datasets.train.as_returnn_opts(),
        "dev": training_datasets.cv.as_returnn_opts(),
        "eval_datasets": {'devtrain': training_datasets.devtrain.as_returnn_opts()},
    }

    rc_extern_data = ExternData([
        datastream.as_nnet_constructor_data(key)
        for key, datastream in training_datasets.datastreams.items()
    ])

    serializer = get_network_serializer(
        returnn_common_root=returnn_common_root,
        rc_extern_data=rc_extern_data,
        training=True,
        network_args=network_args,
        debug=debug,
    )

    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer], python_prolog=speed_perturbation,
    )

    return returnn_config