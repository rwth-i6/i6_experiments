import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import build_training_dataset
from .config import get_training_config,get_pt_raw_forward_config
from .pipeline import tts_training, tts_forward, tts_forward_v2
from ..data import get_tts_log_mel_datastream, get_vocab_datastream

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import DBMelFilterbankOptions

from ..default_tools import RETURNN_EXE, RETURNN_ROOT, RETURNN_COMMON, RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT
from ..storage import duration_alignments


from ..rc_networks.ctc_aligner.parameters import ConvBlstmRecParams


def get_pytorch_raw_ctc_tts():
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
        "batch_size": 18000*samples_per_frame,
        "max_seq_length": {"audio_features": 1600*samples_per_frame},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/librispeech/tts_architecture/tts_feature_model/pytorch/"

    
    def run_exp(name, params, net_module, config, duration_hdf, use_custom_engine=False, debug=False, do_forward=False):
        training_datasets = build_training_dataset(silence_preprocessed=True, raw_audio=True, duration_hdf=duration_hdf)
        training_config = get_training_config(
            returnn_common_root=RETURNN_COMMON,
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
        )  # implicit reconstruction loss
        forward_config = get_pt_raw_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=training_datasets.joint,
            datastreams=training_datasets.datastreams,
            network_module=net_module,
            net_args=params,
            debug=debug,
        )
        train_job = tts_training(
            config=training_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
        )
        if do_forward:
            forward_job = tts_forward_v2(
                checkpoint=train_job.out_checkpoints[100],
                config=forward_config,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name,
            )

    log_mel_datastream = get_tts_log_mel_datastream()

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ..pytorch_networks.nar_taco_v1_config import (
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
    net_module = "nar_taco_v1"

    duration_hdf = duration_alignments["ctc_aligner_v1_fe_drop_035_bs56k_seriv2"]

    run_exp(net_module + "_ctc_drop035_bs56k", params, net_module, config, duration_hdf=duration_hdf, debug=True)

    config_adamw = copy.deepcopy(config)
    config_adamw["optimizer"] = {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-7}
    run_exp(net_module + "_ctc_drop035_bs56k+adamw_1e7", params, net_module, config_adamw, duration_hdf=duration_hdf, debug=True)

    # explicit_duration_hdf = tk.Path("/work/asr4/rossenbach/sisyphus_work_folders/tts_asr_2021_work/i6_experiments/users/rossenbach/tts/duration_extraction/ViterbiAlignmentToDurationsJob.AyAO6JWXTnVc/output/durations.hdf")
    # run_exp(net_module + "_ctc_drop035_bs56k+custom_dur_tftts", params, net_module, config, duration_hdf=explicit_duration_hdf, debug=True)

    net_module = "nar_taco_v2"
    

    from ..pytorch_networks.nar_taco_v2_config import (
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
    
    run_exp(net_module + "_ctc_drop035_bs56k", params, net_module, config, duration_hdf=duration_hdf, debug=True)

    duration_hdf = duration_alignments["ctc_aligner_v3_fe_drop_035_bs56k_seriv2_adam1e-7"]

    run_exp(net_module + "_ctcv3_drop035_bs56k_adam1e-7", params, net_module, config, duration_hdf=duration_hdf, debug=True)

    duration_hdf = duration_alignments["ctc_aligner_tts_fe_drop_035_bs56k_seriv2_adam1e-7"]

    run_exp(net_module + "_ctcttsv1_drop035_bs56k_adam1e-7", params, net_module, config, duration_hdf=duration_hdf, debug=True)

    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v2_drop_05"]

    run_exp(net_module + "_ctc_tts_fe_v2_drop_05", params, net_module, config, duration_hdf=duration_hdf, debug=True)

    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v3_drop_05_spkemb64"]
    
    run_exp(net_module + "_ctc_tts_fe_v3_drop_05_spkemb64", params, net_module, config, duration_hdf=duration_hdf, debug=True)

    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v3_conv384_drop05_spkemb64_dec512"]

    run_exp(net_module + "_ctc_tts_fe_v3_conv384_drop05_spkemb64_dec512", params, net_module, config, duration_hdf=duration_hdf,
            debug=True)

    ctc_name = "_conv384_drop05_spkemb64_dec512_ep200"
    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v3" + ctc_name]
    run_exp(net_module + "_ctc_tts_fe_v3" + ctc_name, params, net_module, config, duration_hdf=duration_hdf,
            debug=True)
    
    ctc_name = "_conv384_drop05+02_spkemb64_dec512_ep200"
    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v3" + ctc_name]
    run_exp(net_module + "_ctc_tts_fe_v3" + ctc_name, params, net_module, config, duration_hdf=duration_hdf,
            debug=True)

    # NO LSTM CTC
    ctc_name = "_conv384_drop05_spkemb64_dec512"
    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v3_nolstm" + ctc_name]
    run_exp(net_module + "_ctc_tts_fe_v3_nolstm" + ctc_name, params, net_module, config, duration_hdf=duration_hdf,
            debug=True)
    
    # New v6 setup with dual LSTM and some other fixes
    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v6_tfstyle_v2"]
    run_exp(net_module + "_ctc_tts_fe_v6_tfstyle_v2", params, net_module, config, duration_hdf=duration_hdf,
            debug=True)
    
    # New v7 setup,
    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v7_tfstyle_v2"]
    run_exp(net_module + "_ctc_tts_fe_v7_tfstyle_v2", params, net_module, config, duration_hdf=duration_hdf,
            debug=True, do_forward=True)
    
    # New v7 setup + prior
    duration_hdf = duration_alignments["ctc_aligner_tts_fe_v7_tfstyle_v2_prior0.3"]
    run_exp(net_module + "_ctc_tts_fe_v7_tfstyle_v2_prior0.3", params, net_module, config, duration_hdf=duration_hdf,
            debug=True)
    

def get_pytorch_raw_ctc_tts_extern_durations():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    samples_per_frame = int(16000 * 0.0125)
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
        "batch_size": 18000 * samples_per_frame,
        "max_seq_length": {"audio_features": 1600 * samples_per_frame},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/librispeech/tts_architecture/tts_feature_model/pytorch/"

    def run_exp(name, params, net_module, config, duration_hdf, use_custom_engine=False, debug=False):
        training_datasets = build_training_dataset(silence_preprocessed=True, raw_audio=True, duration_hdf=duration_hdf)
        training_config = get_training_config(
            returnn_common_root=RETURNN_COMMON,
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
        )  # implicit reconstruction loss
        forward_config = get_pt_raw_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=training_datasets.cv,
            datastreams=training_datasets.datastreams,
            network_module=net_module,
            net_args=params,
            debug=debug,
        )
        train_job = tts_training(
            config=training_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
        )
        forward_job = tts_forward(
            checkpoint=train_job.out_checkpoints[100],
            config=forward_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
        )

    log_mel_datastream = get_tts_log_mel_datastream()

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    net_module = "nar_taco_v2_olddur"

    from ..pytorch_networks.nar_taco_v2_config import (
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
        center=False,
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

    explicit_duration_hdf = tk.Path("/work/asr4/rossenbach/sisyphus_work_folders/tts_asr_2021_work/i6_experiments/users/rossenbach/tts/duration_extraction/ViterbiAlignmentToDurationsJob.AyAO6JWXTnVc/output/durations.hdf")
    run_exp(net_module + "_ctc_drop035_bs56k+custom_dur_tftts", params, net_module, config, duration_hdf=explicit_duration_hdf, debug=True)
