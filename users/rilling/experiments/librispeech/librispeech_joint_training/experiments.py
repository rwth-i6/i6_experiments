import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import build_training_dataset, build_test_dataset, TrainingDatasetSettings
from .config import get_training_config, get_extract_durations_forward__config, get_forward_config, get_search_config
from .pipeline import training, forward, search

from .default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT


def get_glow_joint():
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/joint_training/raw_audio/"
    experiments = {}

    def run_exp(
        name,
        args,
        dataset,
        test_dataset,
        num_epochs=100,
        use_custom_engine=False,
        forward_args={},
        search_args={},
        keep_epochs=None,
    ):
        exp = {}

        training_config = get_training_config(
            training_datasets=dataset,
            **args,
            use_custom_engine=use_custom_engine,
            keep_epochs=keep_epochs,
        )  # implicit reconstruction loss

        forward_config = get_forward_config(
            forward_dataset=dataset,
            **args,
            forward_args=forward_args,
        )

        search_config = get_search_config(
            **args,
            search_args=search_args,
        )

        train_job = training(
            config=training_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
            num_epochs=num_epochs,
        )
        exp["train_job"] = train_job

        forward_job = forward(
            checkpoint=train_job.out_checkpoints[num_epochs],
            config=forward_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
        )
        exp["forward_job"] = forward_job

        search(
            name + "/search",
            search_config,
            train_job.out_checkpoints[num_epochs],
            test_dataset,
            RETURNN_PYTORCH_EXE,
            MINI_RETURNN_ROOT,
        )
        return exp

    # def get_lr_scale(dim_model, step_num, warmup_steps):
    #     return np.power(dim_model, -0.5) * np.min(
    #         [np.power(step_num + 1, -0.5), step_num + 1 * np.power(warmup_steps, -1.5)]
    #     )

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    training_datasets = build_training_dataset(
        settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=False
    )
    training_datasets_silence_preprocessed = build_training_dataset(
        settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=True
    )

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
    net_module = "glow_TTS_ASR"

    asr_test_datasets = {}

    asr_test_datasets["dev-other"] = build_test_dataset(librispeech_key="train-clean-100", dataset_key="dev-other")

    print(asr_test_datasets)

    train_args = {
        "net_args": {
            "n_vocab": training_datasets.datastreams["phonemes"].vocab_size,
            "hidden_channels": 192,
            "filter_channels": 768,
            "filter_channels_dp": 256,
            "out_channels": 80,
            "n_speakers": 251,
            "gin_channels": 256,
            "p_dropout": 0.1,
            "p_dropout_decoder": 0.05,
            "dilation_rate": 1,
            "n_sqz": 2,
            "prenet": True,
            "window_size": 4,
            "fe_config": asdict(fe_config),
        },
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "radam", "epsilon": 1e-9},
            "num_workers_per_gpu": 2,
            "learning_rates": list(np.concatenate((np.linspace(1e-4, 5e-4, 50), np.linspace(5e-4, 1e-6, 50)))),
            "gradient_clip_norm": 20.0,
            "use_learning_rate_control_always": True,
            "learning_rate_control_error_measure": "dev_loss_mle",
            "batch_size": 100 * 16000,
            "accum_grad_multiple_step": 3,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
        },
    }

    forward_args = {"noise_scale": 0.66, "length_scale": 1}

    train_args_wrong_LR = copy.deepcopy(train_args)
    train_args_wrong_LR["net_args"]["mean_only"] = True
    train_args_wrong_LR["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-4, 5e-4, 50), np.linspace(5e-4, 1e-6, 150)))
    )  # The training didn't converge anyways but the learning rates were wrong. To keep the training with the old hash in the graph the wrong LR schedule is used here.
    exp_dict = run_exp(net_module, train_args_wrong_LR, training_datasets, asr_test_datasets, 100, forward_args=forward_args)

    experiments[net_module] = exp_dict

    net_module = "glow_TTS_ASR_unjoint_control"
    train_args_control = copy.deepcopy(train_args)
    train_args_control["network_module"] = net_module
    exp_dict = run_exp(
        net_module, train_args_control, training_datasets, asr_test_datasets, 100, forward_args=forward_args
    )

    train_args_ctc_scale2 = copy.deepcopy(train_args)
    net_module = "glow_TTS_ASR_v2"
    train_args_ctc_scale2["network_module"] = net_module
    exp_dict = run_exp(
        net_module, train_args_ctc_scale2, training_datasets, asr_test_datasets, 100, forward_args=forward_args
    )

    return experiments
