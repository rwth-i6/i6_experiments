import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import (
    build_training_dataset,
    build_test_dataset,
    TrainingDatasetSettings,
    get_binary_lm,
    get_arpa_lm,
    get_text_lexicon,
)
from .config import get_training_config, get_extract_durations_forward__config, get_forward_config, get_search_config
from .pipeline import training, forward, search

from .default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT
from .pytorch_networks.shared.configs import (
    SpecaugConfig,
    ModelConfigV1,
    ModelConfigV2,
    VGG4LayerActFrontendV1Config_mod,
    TextEncoderConfig,
    FlowDecoderConfig,
    PhonemePredictionConfig,
)


def get_glow_joint(x_vector_exp, joint_exps, tts_exps):
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/joint_training/given_alignments/raw_audio/"
    experiments = {}

    def run_exp(
        name,
        args,
        dataset,
        test_dataset,
        num_epochs=100,
        use_custom_engine=False,
        training_args={},
        forward_args={},
        search_args={},
        keep_epochs=None,
        extract_x_vector=False,
        tts_forward=True,
        asr_search=True,
        asr_cv_set=False,
        given_train_job_for_forward=None
    ):
        exp = {}

        if given_train_job_for_forward is None:
            training_config = get_training_config(
                training_datasets=dataset,
                **args,
                training_args=training_args,
                use_custom_engine=use_custom_engine,
                keep_epochs=keep_epochs,
                asr_cv_set=asr_cv_set,
            )  # implicit reconstruction loss

        if tts_forward:
            forward_config = get_forward_config(
                forward_dataset=dataset,
                **{**args, **{"config": {"batch_size": 50 * 16000}}},
                forward_args=forward_args,
            )

        if asr_search:
            search_config = get_search_config(
                **args,
                search_args=search_args,
            )

        if given_train_job_for_forward is None:
            train_job = training(
                config=training_config,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name,
                num_epochs=num_epochs,
            )
        else: 
            train_job = given_train_job_for_forward
        exp["train_job"] = train_job

        if tts_forward:
            forward_job = forward(
                checkpoint=train_job.out_checkpoints[num_epochs],
                config=forward_config,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name,
            )
            exp["forward_job"] = forward_job

        if extract_x_vector:
            forward_x_vector_config = get_forward_config(
                forward_dataset=dataset, **args, forward_args=forward_args, target="xvector", train_data=True
            )
            forward_xvector_job = forward(
                checkpoint=train_job.out_checkpoints[num_epochs],
                config=forward_x_vector_config,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name,
                target="xvector",
            )
            exp["forward_xvector_job"] = forward_xvector_job
        if asr_search:
            search(
                prefix + name + "/search",
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
        custom_processing_function=None, partition_epoch=3, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    # training_datasets = build_training_dataset(
    #     settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=False
    # )

    glowTTS_durations_job = tts_exps["glowTTS/enc192/200ep/long_cooldown/not_silence_preprocessed"]["forward_job_joint_durations"]
    training_datasets_tts_segments = build_training_dataset(
        settings=train_settings,
        librispeech_key="train-clean-100",
        silence_preprocessing=False,
        use_tts_train_segments=True,
        durations_file=glowTTS_durations_job.out_hdf_files["output.hdf"]
    )
    # training_datasets_silence_preprocessed = build_training_dataset(
    #     settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=True
    # )
    train_settings_pe1 = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )
    training_datasets_pe1_tts_segments = build_training_dataset(
        settings=train_settings_pe1, librispeech_key="train-clean-100", silence_preprocessing=False, use_tts_train_segments=True, durations_file=glowTTS_durations_job.out_hdf_files["output.hdf"]
    )

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    label_datastream_asr = cast(LabelDatastream, training_datasets_tts_segments.datastreams["phonemes_eow"])
    vocab_size_without_blank_asr = label_datastream_asr.vocab_size
    label_datastream_tts = cast(LabelDatastream, training_datasets_tts_segments.datastreams["phonemes"])
    vocab_size_without_blank_tts = label_datastream_tts.vocab_size
    speaker_datastream = cast(LabelDatastream, training_datasets_tts_segments.datastreams["speaker_labels"])

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

    asr_test_datasets = {}

    asr_test_datasets["dev-other"] = build_test_dataset(librispeech_key="train-clean-100", dataset_key="dev-other")

    asr_test_datasets2 = copy.deepcopy(asr_test_datasets)
    asr_test_datasets2["train-clean-100-cv"] = build_test_dataset(
        librispeech_key="train-clean-100", dataset_key="train-clean-100", test_on_tts_cv=True
    )
    asr_test_datasets2["dev-clean"] = build_test_dataset(librispeech_key="train-clean-100", dataset_key="dev-clean")

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=100,
        max_dim_time=20,
        max_dim_feat=8,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=16,
        conv2_channels=16,
        conv3_channels=16,
        conv4_channels=16,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=96,
        activation=None,
    )
    text_encoder_config = TextEncoderConfig(
        n_vocab=label_datastream_tts.vocab_size,
        hidden_channels=192,
        filter_channels=768,
        filter_channels_dp=256,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        window_size=4,
        block_length=None,
        mean_only=False,
        prenet=True,
    )

    flow_decoder_config = FlowDecoderConfig(
        hidden_channels=192,
        kernel_size=5,
        dilation_rate=1,
        n_blocks=12,
        n_layers=4,
        p_dropout=0.05,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
    )

    phoeneme_prediction_config = PhonemePredictionConfig(
        n_channels=512,
        n_layers=3,
        p_dropout=0.1
    )

    model_config_tts_only = ModelConfigV1(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        text_encoder_config=text_encoder_config,
        decoder_config=flow_decoder_config,
        label_target_size=vocab_size_without_blank_asr,
        ffn_layers=3,
        ffn_channels=512,
        specauc_start_epoch=1,
        out_channels=80,
        gin_channels=512,
        n_speakers=speaker_datastream.vocab_size,
    )

    model_config = ModelConfigV2(
        specaug_config=specaug_config,
        text_encoder_config=text_encoder_config,
        decoder_config=flow_decoder_config,
        phoneme_prediction_config=phoeneme_prediction_config,
        label_target_size=vocab_size_without_blank_asr,
        specauc_start_epoch=1,
        out_channels=80,
        gin_channels=256,
        n_speakers=speaker_datastream.vocab_size,
    )

    net_module = "glowTTS_x_vector"

    train_args = {
        "net_args": {"fe_config": asdict(fe_config), "model_config": asdict(model_config_tts_only)},
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-8},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
            + list(np.linspace(7e-4, 7e-5, 110))
            + list(np.linspace(7e-5, 1e-8, 30)),
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
        },
    }

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    label_datastream = cast(LabelDatastream, training_datasets_tts_segments.datastreams["phonemes_eow"])

    forward_args = {"noise_scale": 0.66, "length_scale": 1}
    default_search_args = {
        "lexicon": get_text_lexicon(),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 256,
        "arpa_lm": get_binary_lm(),
        "lm_weight": 5,
        "beam_threshold": 16,
        "asr_data": False,
    }

    x_vect_train_job = x_vector_exp["x_vector_cnn/1e-3_not_silence_preprocessed"]["train_job"]
    train_args["config"]["preload_from_files"] = {
        "x_vector_model": {
            "filename": x_vect_train_job.out_checkpoints[x_vect_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "prefix": "x_vector.",
            "ignore_missing": True,
        }
    }

    train_args_TTS_xvector = copy.deepcopy(train_args)
    net_module = "glowTTS_x_vector"
    train_args_TTS_xvector["network_module"] = net_module
    train_args_TTS_xvector["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50)))
    )
    train_args_TTS_xvector["config"]["optimizer"] = {"class": "radam", "epsilon": 1e-9}
    train_args_TTS_xvector["net_args"]["model_config"]["specaug_config"] = None
    exp_dict = run_exp(
        net_module,
        train_args_TTS_xvector,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )
    experiments[net_module] = exp_dict

    net_module = "glowTTS_x_vector_v2"
    train_args_TTS_xvector["net_args"]["model_config"]["gin_channels"] = 256
    train_args_TTS_xvector["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_TTS_xvector,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )
    experiments[net_module] = exp_dict

    net_module = "glowTTS"
    train_args_TTS = copy.deepcopy(train_args_TTS_xvector)
    del train_args_TTS["config"]["preload_from_files"]
    train_args_TTS["network_module"] = net_module

    exp_dict = run_exp(
        net_module + "/enc768",
        train_args_TTS,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )

    train_args_TTS["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 192
    exp_dict = run_exp(
        net_module + "/enc192",
        train_args_TTS,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )

    net_module = "glowTTS_ASR_ffn_x_vector"
    train_args_joint_training = copy.deepcopy(train_args)
    glowTTS_x_vector_train_job = experiments["glowTTS_x_vector_v2"]["train_job"]
    train_args_joint_training["config"]["preload_from_files"]["glowTTS_xvector"] = {
        "filename": glowTTS_x_vector_train_job.out_checkpoints[glowTTS_x_vector_train_job.returnn_config.get("num_epochs", 100)],
        "init_for_train": True,
        "ignore_missing": True
    }
    train_args_joint_training["network_module"] = net_module
    train_args_joint_training["net_args"]["model_config"] = asdict(model_config)

    exp_dict = run_exp(
        net_module + "/specaug",
        train_args_joint_training,
        training_datasets_tts_segments,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )

    # # train_args_pe1 = copy.deepcopy(train_args)
    # # train_args_pe1["config"]["learning_rates"] = list(
    # #     np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50)))
    # # )

    train_args_joint_training_no_specaug = copy.deepcopy(train_args_joint_training)
    train_args_joint_training_no_specaug["net_args"]["model_config"]["specaug_config"] = None

    exp_dict = run_exp(
        net_module + "/no_specaug",
        train_args_joint_training_no_specaug,
        training_datasets_tts_segments,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )

    net_module = "glowTTS_ASR_blstm_x_vector"
    train_args_joint_training["network_module"] = net_module
    train_args_joint_training_no_specaug["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/specaug",
        train_args_joint_training,
        training_datasets_tts_segments,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )

    exp_dict = run_exp(
        net_module + "/no_specaug",
        train_args_joint_training_no_specaug,
        training_datasets_tts_segments,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )
