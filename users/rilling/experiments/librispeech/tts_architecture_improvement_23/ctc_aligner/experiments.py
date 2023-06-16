import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import build_training_dataset
from .config import get_training_config, get_forward_config, get_pt_forward_config
from .pipeline import ctc_training, ctc_forward
from ..data import get_tts_log_mel_datastream

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import DBMelFilterbankOptions

from ..default_tools import RETURNN_EXE, RETURNN_ROOT, RETURNN_COMMON, RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT
from ..storage import add_duration


from ..rc_networks.ctc_aligner.parameters import ConvBlstmRecParams

def get_baseline_ctc_alignment():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    name = "experiments/librispeech/tts_architecture/ctc_aligner/baseline"
    training_datasets = build_training_dataset(silence_preprocessed=True)


    config = {
        "behavior_version": 16,
        ############
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "use_learning_rate_control_always": True,
        ############
        "accum_grad_multiple_step": 2,
        "gradient_clip": 1,
        "gradient_noise": 0,
        "learning_rate_control_error_measure": "dev_score_reconstruction_output",
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": 28000,
        "max_seq_length": {"audio_features": 1600},
        "max_seqs": 200,
    }

    net_module = "ctc_aligner.conv_blstm_rec"
    params = ConvBlstmRecParams(
        audio_emb_size=256,
        speaker_emb_size=256,
        conv_hidden_size=256,
        enc_lstm_size=256,
        rec_lstm_size=512,
        dropout=0.5,
        reconstruction_scale=0.5,
        training=True
    )

    aligner_config = get_training_config(
        returnn_common_root=RETURNN_COMMON,
        training_datasets=training_datasets,
        network_module=net_module,
        net_args=asdict(params),
        config=config,
    )  # implicit reconstruction loss
    params.training = False
    forward_config = get_forward_config(
        returnn_common_root=RETURNN_COMMON,
        forward_dataset=training_datasets.joint,
        datastreams=training_datasets.datastreams,
        network_module=net_module,
        net_args=asdict(params)
    )
    train_job = ctc_training(
        config=aligner_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        prefix=name,
    )
    duration_hdf = ctc_forward(
        checkpoint=train_job.out_checkpoints[100],
        config=forward_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        prefix=name
    )
    return duration_hdf

def get_pytorch_ctc_alignment():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "gradient_clip": 1.0,
        "use_learning_rate_control_always": True,
        "learning_rate_control_error_measure": "dev_ctc",
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": 56000,
        "max_seq_length": {"audio_features": 1600},
        "max_seqs": 200,
    }

    prefix = "experiments/librispeech/tts_architecture/ctc_aligner/pytorch/"
    training_datasets = build_training_dataset(silence_preprocessed=True)

    def run_exp(name, params, net_module, config, use_custom_engine=False, debug=False):
        aligner_config = get_training_config(
            returnn_common_root=RETURNN_COMMON,
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
            pytorch_mode=True
        )  # implicit reconstruction loss
        forward_config = get_pt_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=training_datasets.joint,
            datastreams=training_datasets.datastreams,
            network_module=net_module,
            net_args=params,
            pytorch_mode=True
        )
        train_job = ctc_training(
            config=aligner_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
        )
        duration_hdf = ctc_forward(
            checkpoint=train_job.out_checkpoints[100],
            config=forward_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name
        )
        return duration_hdf

    net_module = "ctc_aligner_v1"
    params = {
        "conv_hidden_size": 256,
        "lstm_size": 512,
        "speaker_embedding_size": 256,
        "dropout": 0.35,
        "target_size": 44
    }

    duration_hdf = run_exp(net_module + "_drop035_bs56k", params, net_module, config, debug=True)

    #net_module = "ctc_aligner_v1_gradaccum"
    config_gradaccum = copy.deepcopy(config)
    #run_exp(net_module + "_drop1d035_bs28k_accum2", params, net_module, config, use_custom_engine=True, debug=True)
    
    
    # net_module = "ctc_aligner_v2"
    # params_v2 = {
    #     "conv_hidden_size": 256,
    #     "lstm_size": 512,
    #     "speaker_embedding_size": 256,
    #     "conv_dropout": 0.5,
    #     "final_dropout": 0.1,
    #     "target_size": 44
    # }
    # run_exp(net_module + "_drop05_01", params_v2, net_module, config, use_custom_engine=False, debug=False)

    #net_module = "ctc_aligner_v1_ctc_sum"
    #run_exp(net_module + "_drop01", params, net_module, config)

    #net_module = "ctc_aligner_v1_ctc_sum_nobroad"
    #params = copy.deepcopy(params)
    #params["dropout"] = 0.35
    #run_exp(net_module + "_drop035", params, net_module, config)

    return duration_hdf


def get_pytorch_raw_ctc_alignment():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    samples_per_frame = int(16000*0.0125)
    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "gradient_clip": 1.0,
        "use_learning_rate_control_always": True,
        "learning_rate_control_error_measure": "dev_ctc",
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": 56000*samples_per_frame,
        "max_seq_length": {"audio_features": 1600*samples_per_frame},
        "max_seqs": 200,
    }

    prefix = "experiments/librispeech/tts_architecture/ctc_aligner/pytorch/"
    training_datasets = build_training_dataset(silence_preprocessed=True, raw_audio=True)

    def run_exp(name, params, net_module, config, use_custom_engine=False, debug=False):
        aligner_config = get_training_config(
            returnn_common_root=RETURNN_COMMON,
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
            pytorch_mode=True
        )  # implicit reconstruction loss
        forward_config = get_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=training_datasets.joint,
            datastreams=training_datasets.datastreams,
            network_module=net_module,
            net_args=params,
            pytorch_mode=True
        )
        train_job = ctc_training(
            config=aligner_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
        )
        duration_hdf = ctc_forward(
            checkpoint=train_job.out_checkpoints[100],
            config=forward_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name
        )
        return duration_hdf

    net_module = "ctc_aligner_v1_fe"
    log_mel_datastream = get_tts_log_mel_datastream()

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ..pytorch_networks.ctc_aligner_v1_fe import DbMelFeatureExtractionConfig, Config
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
        norm=norm
    )
    model_config = Config(
        conv_hidden_size=256,
        lstm_size=512,
        speaker_embedding_size=256,
        dropout=0.35,
        target_size=44,
        feature_extraction_config=fe_config,
    )

    params = {
        "config": asdict(model_config)
    }


    duration_hdf = run_exp(net_module + "_drop035_bs56k", params, net_module, config, debug=True)

    return duration_hdf
