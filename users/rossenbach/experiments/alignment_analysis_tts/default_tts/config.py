"""
Pipeline file for experiments with the standard CTC TTS model
"""
from sisyphus import tk
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob
from i6_experiments.common.setups.returnn_common.serialization import (
    Collection,
    ExternData,
    Import,
    Network,
    PythonEnlargeStackWorkaroundNonhashedCode,
)
from i6_experiments.common.datasets.librispeech import get_corpus_object_dict
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    TTSTrainingDatasets,
    TTSForwardData,
)
from i6_private.users.hilmes.tools.tts import VerifyCorpus, MultiJobCleanup
from i6_experiments.users.hilmes.experiments.librispeech.util.asr_evaluation import (
    asr_evaluation,
)
from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import CalculateSpeakerPriorJob
from i6_experiments.users.hilmes.data.librispeech import get_ls_train_clean_100_tts_silencepreprocessed
from i6_core.returnn.oggzip import BlissToOggZipJob

def get_training_config(
        returnn_common_root: tk.Path, training_datasets: TTSTrainingDatasets, batch_size = 18000, **kwargs
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
        "batch_size": batch_size,
        "max_seq_length": {"audio_features": 1600},
        "max_seqs": 60,
    }

    extern_data = [
        datastream.as_nnet_constructor_data(key)
        for key, datastream in training_datasets.datastreams.items()
    ]
    config["train"] = training_datasets.train.as_returnn_opts()
    config["dev"] = training_datasets.cv.as_returnn_opts()

    rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
    rc_extern_data = ExternData(extern_data=extern_data)
    rc_model = Import(
        "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.NARTTSModel"
    )
    rc_construction_code = Import(
        "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.construct_network"
    )

    net_func_map = {
        "net_module": rc_model.object_name,
        "phoneme_data": "phonemes",
        "duration_data": "duration_data",
        "label_data": "speaker_labels",
        "audio_data": "audio_features",
        "time_dim": "phonemes_time",
        "label_time_dim": "speaker_labels_time",
        "speech_time_dim": "audio_features_time",
        "duration_time_dim": "duration_data_time",
    }

    if "use_pitch_pred" in kwargs.keys() and kwargs["use_pitch_pred"]:
        net_func_map["pitch"] = "pitch_data"
        net_func_map["pitch_time"] = "pitch_data_time"

    if "use_energy_pred" in kwargs.keys() and kwargs["use_energy_pred"]:
        net_func_map["energy"] = "energy_data"
        net_func_map["energy_time"] = "energy_data_time"

    rc_network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map=net_func_map,
        net_kwargs={"training": True, **kwargs},
    )

    serializer = Collection(
        serializer_objects=[
            rc_recursionlimit,
            rc_extern_data,
            rc_model,
            rc_construction_code,
            rc_network,
        ],
        returnn_common_root=returnn_common_root,
        make_local_package_copy=True,
        packages={
            "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks",
            "i6_experiments.users.hilmes.modules",
        },
    )

    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )

    return returnn_config