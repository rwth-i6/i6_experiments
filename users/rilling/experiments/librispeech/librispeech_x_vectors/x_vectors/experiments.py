import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import build_training_dataset, TrainingDatasetSettings, build_test_dataset
from .config import get_training_config, get_forward_config
from .pipeline import x_vector_training, x_vector_forward

from ..default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT

from ..storage import add_x_vector_extraction


def get_pytorch_xvector():
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/x_vector/"

    def run_exp(
        name,
        args,
        dataset,
        num_epochs=100,
        use_custom_engine=False,
        forward_args={},
        keep_epochs=None,
    ):
        exp_dict = {}
        training_config = get_training_config(
            returnn_common_root=RETURNN_COMMON,
            training_datasets=dataset,
            **args,
            use_custom_engine=use_custom_engine,
            pytorch_mode=True,
            keep_epochs=keep_epochs,
        )  # implicit reconstruction loss

        forward_config = get_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=dataset,
            **args,
            forward_args=forward_args,
            pytorch_mode=True,
        )

        train_job = x_vector_training(
            config=training_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
            num_epochs=num_epochs,
        )

        tts_hdf = x_vector_forward(
            checkpoint=train_job.out_checkpoints[num_epochs],
            config=forward_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
        )

        add_x_vector_extraction(name, tts_hdf, average=True)

        return train_job

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    training_datasets = build_training_dataset(
        settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=False
    )
    training_datasets_silence_preprocessed = build_training_dataset(
        settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=True
    )

    test_clean_dataset = build_test_dataset(dataset_key="test-clean")

    from .data import get_tts_log_mel_datastream
    from .feature_config import DbMelFeatureExtractionConfig
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import DBMelFilterbankOptions

    log_mel_datastream = get_tts_log_mel_datastream(silence_preprocessing=False)
    log_mel_datastream_silence_preprocessed = get_tts_log_mel_datastream(silence_preprocessing=True)

    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])
    assert isinstance(log_mel_datastream.options.feature_options, DBMelFilterbankOptions)
    fe_config = DbMelFeatureExtractionConfig(
        sample_rate=log_mel_datastream.options.sample_rate,
        win_size=log_mel_datastream.options.window_len,
        hop_size=log_mel_datastream.options.step_len,
        f_min=log_mel_datastream.options.feature_options.fmin,
        f_max=log_mel_datastream.options.feature_options.fmax,
        min_amp=log_mel_datastream.options.feature_options.min_amp,
        num_filters=log_mel_datastream.options.num_feature_filters,
        center=log_mel_datastream.options.feature_options.center,
        norm=norm,
    )

    log_mel_datastream_silence_preprocessed = get_tts_log_mel_datastream(silence_preprocessing=True)

    assert "norm_mean" in log_mel_datastream_silence_preprocessed.additional_options
    assert "norm_std_dev" in log_mel_datastream_silence_preprocessed.additional_options

    norm_silence_preprocessed = (
        log_mel_datastream_silence_preprocessed.additional_options["norm_mean"],
        log_mel_datastream_silence_preprocessed.additional_options["norm_std_dev"],
    )
    assert isinstance(log_mel_datastream_silence_preprocessed.options.feature_options, DBMelFilterbankOptions)
    fe_config_silence_preprocessed = DbMelFeatureExtractionConfig(
        sample_rate=log_mel_datastream_silence_preprocessed.options.sample_rate,
        win_size=log_mel_datastream_silence_preprocessed.options.window_len,
        hop_size=log_mel_datastream_silence_preprocessed.options.step_len,
        f_min=log_mel_datastream_silence_preprocessed.options.feature_options.fmin,
        f_max=log_mel_datastream_silence_preprocessed.options.feature_options.fmax,
        min_amp=log_mel_datastream_silence_preprocessed.options.feature_options.min_amp,
        num_filters=log_mel_datastream_silence_preprocessed.options.num_feature_filters,
        center=log_mel_datastream_silence_preprocessed.options.feature_options.center,
        norm=norm_silence_preprocessed,
    )

    net_args_spp = {
        "input_dim": fe_config_silence_preprocessed.num_filters,
        "num_classes": training_datasets_silence_preprocessed.datastreams["speaker_labels"].vocab_size,
        "batch_norm": True,
        "fe_config": asdict(fe_config),
    }

    net_args_no_spp = {
        "input_dim": fe_config.num_filters,
        "num_classes": training_datasets.datastreams["speaker_labels"].vocab_size,
        "batch_norm": True,
        "fe_config": asdict(fe_config),
    }

    net_module = "x_vector"
    train_args = {
        "net_args": net_args_spp,
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "radam", "epsilon": 1e-9},
            "num_workers_per_gpu": 2,
            # "learning_rate_control": "newbob_multi_epoch",
            # "learning_rate_control_min_num_epochs_per_new_lr": 5,
            # "learning_rate_control_relative_error_relative_lr": True,
            "learning_rate": 1e-3,
            # "gradient_clip_norm": 2.0,
            "use_learning_rate_control_always": True,
            "learning_rate_control_error_measure": "ce",
            # ############
            # "newbob_learning_rate_decay": 0.9,
            # "newbob_multi_num_epochs": 5,
            # "newbob_multi_update_interval": 1,
            # "newbob_relative_error_threshold": 0,
            #############
            # "batch_size": 56000,
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
        },
    }

    exps_dict = {}
    train_args_spp = copy.deepcopy(train_args)
    train_args_spp["net_args"] = net_args_spp

    train_args_no_spp = copy.deepcopy(train_args)
    train_args_no_spp["net_args"] = net_args_no_spp

    train_job = run_exp("x_vector/1e-3_not_silence_preprocessed", train_args_no_spp, training_datasets, num_epochs=100)
    exps_dict["x_vector/1e-3_not_silence_preprocessed"] = {"train_job": train_job}

    net_module = "x_vector_cnn"
    train_args_no_spp["network_module"] = net_module
    train_job = run_exp(
        "x_vector_cnn/1e-3_not_silence_preprocessed", train_args_no_spp, training_datasets, num_epochs=100
    )
    exps_dict["x_vector_cnn/1e-3_not_silence_preprocessed"] = {"train_job": train_job}

    args_forward_test_clean = copy.deepcopy(train_args_no_spp)
    del args_forward_test_clean["config"]["max_seq_length"]
    forward_config_test_clean = get_forward_config(
        returnn_common_root=RETURNN_COMMON,
        forward_dataset=test_clean_dataset,
        **args_forward_test_clean,
        forward_args={},
        pytorch_mode=True,
    )

    tts_hdf = x_vector_forward(
            checkpoint=train_job.out_checkpoints[100],
            config=forward_config_test_clean,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + net_module + "/test-clean",
        )

    add_x_vector_extraction("x_vector_cnn/1e-3_not_silence_preprocessed/test-clean", tts_hdf, average=True)

    return exps_dict
