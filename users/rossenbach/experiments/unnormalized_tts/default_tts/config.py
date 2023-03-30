"""
Pipeline file for experiments with the standard CTC TTS model
"""
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
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    TTSTrainingDatasets,
    TTSForwardData,
)


def get_network_serializer(returnn_common_root: tk.Path, rc_extern_data: ExternData, training: bool, debug=False, **kwargs) -> Collection:

    rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
    model_type = kwargs.pop("model_type")
    rc_construction_code = Import(
        "i6_experiments.users.rossenbach.experiments.unnormalized_tts.rc_networks.%s.construct_network" % model_type
    )
    net_func_map = {
        "phoneme_data": "phon_labels",
        "speaker_label_data": "speaker_labels",
    }

    if "audio_samples" in [init_args.name for init_args in rc_extern_data.extern_data]:
        net_func_map["audio_data"] = "audio_samples"
    if "phon_durations" in [init_args.name for init_args in rc_extern_data.extern_data]:
        net_func_map["phoneme_duration_data"] = "phon_durations"

    #if "use_pitch_pred" in kwargs.keys() and kwargs["use_pitch_pred"]:
    #    net_func_map["pitch"] = "pitch_data"

    #if "use_energy_pred" in kwargs.keys() and kwargs["use_energy_pred"]:
    #    net_func_map["energy"] = "energy_data"

    rc_network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map=net_func_map,
        net_kwargs={"training": training, **kwargs},
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
            "i6_experiments.users.rossenbach.experiments.unnormalized_tts.rc_networks",
        },
    )

    return serializer


def get_training_config(
        returnn_common_root: tk.Path, training_datasets: TTSTrainingDatasets, batch_size = 18000, debug=False, **kwargs
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_model
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
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
        "behavior_version": 12,
        ############
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "accum_grad_multiple_step": round(18000 * 2 / batch_size),
        "gradient_clip": 1,
        "gradient_noise": 0,
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "use_learning_rate_control_always": True,
        "learning_rate_control_error_measure": "dev_score_dec_output",
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": batch_size*160,
        "max_seq_length": {"audio_features": 2200*160},
        "max_seqs": 60,
        #############
        "train": training_datasets.train.as_returnn_opts(),
        "dev": training_datasets.cv.as_returnn_opts()
    }

    rc_extern_data = ExternData([
        datastream.as_nnet_constructor_data(key)
        for key, datastream in training_datasets.datastreams.items()
    ])

    serializer = get_network_serializer(
        returnn_common_root=returnn_common_root,
        rc_extern_data=rc_extern_data,
        training=True,
        debug=debug,
        **kwargs
    )

    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )

    return returnn_config


def get_forward_config(
        returnn_common_root,
        forward_dataset: TTSForwardData,
        batch_size: int = 4000,
        **kwargs,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_model
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """

    config = {
        "behavior_version": 12,
        "forward_batch_size": batch_size,
        "max_seqs": 60,
        "forward_use_search": True,
        "target": "dec_output",
        #############
        "eval": forward_dataset.dataset.as_returnn_opts()
    }

    rc_extern_data = ExternData([
        datastream.as_nnet_constructor_data(key)
        for key, datastream in forward_dataset.datastreams.items()
    ])

    serializer = get_network_serializer(
        returnn_common_root=returnn_common_root,
        rc_extern_data=rc_extern_data,
        training=False,
        **kwargs
    )

    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

    return returnn_config

def get_speaker_extraction_config(
        returnn_common_root, forward_dataset: TTSForwardData, **kwargs
):
    """

    :param returnn_common_root:
    :param forward_dataset:
    :param kwargs:
    :return:
    """
    config = {
        "behavior_version": 12,
        "forward_batch_size": 18000,
        "max_seqs": 60,
        "forward_use_search": True,
        "target": "dec_output",
        #############
        "eval": forward_dataset.dataset.as_returnn_opts()
    }
    rc_extern_data = ExternData([
        datastream.as_nnet_constructor_data(key)
        for key, datastream in forward_dataset.datastreams.items()
    ])

    local_kwargs = copy.deepcopy(kwargs)
    local_kwargs["dump_speaker_embeddings"] = True
    local_kwargs["calc_speaker_embeddings"] = True  # needed because we are not training
    serializer = get_network_serializer(
        returnn_common_root=returnn_common_root,
        rc_extern_data=rc_extern_data,
        training=False,
        **local_kwargs,
    )
    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

    return returnn_config