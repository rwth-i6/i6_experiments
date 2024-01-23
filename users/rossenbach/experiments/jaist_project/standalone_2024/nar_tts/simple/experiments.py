import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from ...data.tts_phon import build_durationtts_training_dataset
from ...data.tts_phon import get_tts_log_mel_datastream, get_vocab_datastream

from ...config import get_training_config, get_forward_config
from ...pipeline import training, tts_eval

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...storage import duration_alignments, vocoders



def get_pytorch_raw_ctc_tts():
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
        "batch_size": 225 * 16000,
        "max_seq_length": {"audio_features": 20 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/jaist_project/standalone_2024/nar_tts/simple/"

    def run_exp(name, params, net_module, config, duration_hdf, decoder_options, extra_decoder=None, use_custom_engine=False, debug=False):
        training_datasets = build_durationtts_training_dataset(duration_hdf=duration_hdf)
        training_config = get_training_config(
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
        )  # implicit reconstruction loss
        forward_config = get_forward_config(
            network_module=net_module,
            net_args=params,
            decoder=extra_decoder or net_module,
            decoder_args=decoder_options,
            config={
                "forward": training_datasets.cv.as_returnn_opts()
            },
            debug=debug,
        )
        train_job = training(
            prefix_name=prefix + name,
            returnn_config=training_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            num_epochs=100
        )
        forward_job = tts_eval(
            prefix_name=prefix + name,
            returnn_config=forward_config,
            checkpoint=train_job.out_checkpoints[100],
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(prefix + name + "/audio_files", forward_job.out_files["audio_files"])
        return train_job, forward_job

    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-100")

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    net_module = "nar_tts.legacy.nar_taco_v2"

    from ...pytorch_networks.nar_tts.legacy.nar_taco_v2_config import (
        DbMelFeatureExtractionConfig,
        NarEncoderConfig,
        NarTacoDecoderConfig,
        ConvDurationSigmaPredictorConfig,
        ModelConfig
    )
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

    encoder_config = NarEncoderConfig(
        label_in_dim=get_vocab_datastream(with_blank=True).vocab_size,  # forgot to remove the blank :(
        embedding_size=256,
        conv_hidden_size=256,
        filter_size=3,
        dropout=0.5,
        lstm_size=256
    )
    decoder_config = NarTacoDecoderConfig(
        lstm_size=1024,
        dropout=0.5,
    )
    duration_predictor_config = ConvDurationSigmaPredictorConfig(
        hidden_size=256,
        filter_size=3,
        dropout=0.5,
    )
    model_config = ModelConfig(
        speaker_embedding_size=256,
        dropout=0.5,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        duration_predictor_config=duration_predictor_config,
        feature_extraction_config=fe_config,
    )

    params = {
        "config": asdict(model_config)
    }
    
    decoder_options = {
        "norm_mean": norm[0],
        "norm_std_dev": norm[1]
    }

    decoder_options = copy.deepcopy(decoder_options)
    vocoder = vocoders["blstm_gl_v1"]
    decoder_options["gl_net_checkpoint"] = vocoder.checkpoint
    decoder_options["gl_net_config"] = vocoder.config
    
    # New v7 setup,
    duration_hdf = duration_alignments["ctc.tts_aligner_1223.ctc_aligner_tts_fe_v8_tfstyle_v2_fullength"]
    train, forward = run_exp(net_module + "_ctc_tts_fe_v8_tfstyle_v2_fulllength", params, net_module, config,
                             extra_decoder="nar_tts.legacy.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)
    # train.hold()
    
    duration_hdf = duration_alignments["glow_tts.lukas_baseline_bs600_v2"]
    train, forward = run_exp(net_module + "_fromglowtts1_fe_v8_tfstyle_v2_fulllength", params, net_module, config,
                             extra_decoder="nar_tts.legacy.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)
    train.hold()