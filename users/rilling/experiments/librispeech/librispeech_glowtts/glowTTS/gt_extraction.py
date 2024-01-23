import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import build_training_dataset, TrainingDatasetSettings
from .config import get_training_config, get_extract_durations_forward__config, get_forward_config
from .pipeline import glowTTS_training, glowTTS_forward

from ..default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT

from i6_experiments.users.rossenbach.experiments.alignment_analysis_tts.gl_vocoder.default_vocoder import get_default_vocoder


def get_ground_truth_audio_and_spectrograms():
    """
    Setup to compute audio and spectrograms of ground truth with given pipeline settings and vocoder / feature extraction
    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/tts_architecture/glow_tts/ground_truth/"

    def run_exp(name, args, dataset):
        forward_config = get_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=dataset,
            **args,
            pytorch_mode=True
        )

        forward_config2 = get_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=dataset,
            **args,
            pytorch_mode=True,
            target="spectrograms"
        )
        
        tts_hdf = glowTTS_forward(
            checkpoint=None,
            config=forward_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name
        )

        glowTTS_forward(
            checkpoint=None,
            config=forward_config2,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
            target="spectrograms"
        )

        return tts_hdf
    
    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )
    
    training_datasets = build_training_dataset(settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=False)
    training_datasets_silence_preprocessed = build_training_dataset(settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=True)

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

    norm_silence_preprocessed = (log_mel_datastream_silence_preprocessed.additional_options["norm_mean"], log_mel_datastream_silence_preprocessed.additional_options["norm_std_dev"])
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

    net_module = "gt_extractor"

    args = {
        "net_args": {
            "fe_config": asdict(fe_config)
        },
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-9},
            "num_workers_per_gpu": 2,
            "learning_rates": list(np.concatenate((np.linspace(5e-5, 5e-4, 50), np.linspace(5e-4, 1e-6, 150)))),
            "gradient_clip_norm": 2.0,
            "use_learning_rate_control_always": True,
            "learning_rate_control_error_measure": "dev_loss_mle",
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
        }
    }

    forward_args = {
        "noise_scale": 0.66,
        "length_scale": 1
    }

    run_exp("ground_truth", args, dataset=training_datasets)
    run_exp("ground_truth_silence", args, dataset=training_datasets_silence_preprocessed)
