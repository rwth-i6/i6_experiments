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
from .pytorch_networks.shared.i6modelsV1_VGG4LayerActFrontendV1_v4_cfg import (
    SpecaugConfig,
    ModelConfig,
    VGG4LayerActFrontendV1Config_mod,
    TextEncoderConfig,
    FlowDecoderConfig,
)


def get_glow_joint(x_vector_exp):
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/joint_training/default/raw_audio/"
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
        use_speaker_labels_in_dev=False,
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
                use_speaker_labels_in_dev=use_speaker_labels_in_dev,
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

    training_datasets = build_training_dataset(
        settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=False
    )
    training_datasets_tts_segments = build_training_dataset(
        settings=train_settings,
        librispeech_key="train-clean-100",
        silence_preprocessing=False,
        use_tts_train_segments=True,
    )
    # training_datasets_silence_preprocessed = build_training_dataset(
    #     settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=True
    # )
    train_settings_pe1 = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )
    training_datasets_pe1 = build_training_dataset(
        settings=train_settings_pe1, librispeech_key="train-clean-100", silence_preprocessing=False
    )

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    label_datastream_asr = cast(LabelDatastream, training_datasets.datastreams["phonemes_eow"])
    vocab_size_without_blank_asr = label_datastream_asr.vocab_size
    label_datastream_tts = cast(LabelDatastream, training_datasets.datastreams["phonemes"])
    vocab_size_without_blank_tts = label_datastream_tts.vocab_size
    speaker_datastream = cast(LabelDatastream, training_datasets.datastreams["speaker_labels"])

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
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=None,
        text_encoder_config=text_encoder_config,
        decoder_config=flow_decoder_config,
        label_target_size=vocab_size_without_blank_asr,
        conformer_size=96,
        num_layers=8,
        num_heads=2,
        ff_dim=384,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=9,
        final_dropout=0.2,
        specauc_start_epoch=1,
        out_channels=80,
        gin_channels=512,
        n_speakers=speaker_datastream.vocab_size,
    )

    net_module = "glowTTS_ASR_conformer_x_vector"

    train_args = {
        "net_args": {"fe_config": asdict(fe_config), "model_config": asdict(model_config)},
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

    label_datastream = cast(LabelDatastream, training_datasets.datastreams["phonemes_eow"])

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
    exp_dict = run_exp(
        net_module,
        train_args,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )

    exp_dict = run_exp(
        net_module + "_ctc_scale_0.1",
        train_args,
        training_datasets,
        asr_test_datasets,
        250,
        training_args={"ctc_scale": 0.1},
        forward_args=forward_args,
        search_args=default_search_args,
    )
    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"_ctc_scale_0.1/tuning/lm_{lm}",
            train_args,
            training_datasets,
            asr_test_datasets,
            250,
            training_args={"ctc_scale": 0.1},
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )

    train_args_spec_augment = copy.deepcopy(train_args)
    train_args_spec_augment["net_args"]["model_config"]["specaug_config"] = asdict(specaug_config)
    exp_dict = run_exp(
        net_module + "_spec_augment",
        train_args_spec_augment,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"_spec_augment/tuning/lm_{lm}",
            train_args_spec_augment,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )

    exp_dict = run_exp(
        net_module + "_spec_augment_ctc_scale_0.1",
        train_args_spec_augment,
        training_datasets,
        asr_test_datasets,
        250,
        training_args={"ctc_scale": 0.1},
        forward_args=forward_args,
        search_args=default_search_args,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"_spec_augment_ctc_scale_0.1/tuning/lm_{lm}",
            train_args_spec_augment,
            training_datasets,
            asr_test_datasets,
            250,
            training_args={"ctc_scale": 0.1},
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )

    exp_dict = run_exp(
        net_module + "_spec_augment_ctc_scale_0.1_tts_segments",
        train_args_spec_augment,
        training_datasets_tts_segments,
        asr_test_datasets2,
        250,
        training_args={"ctc_scale": 0.1},
        forward_args=forward_args,
        search_args=default_search_args,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"_spec_augment_ctc_scale_0.1_tts_segments/tuning/lm_{lm}",
            train_args_spec_augment,
            training_datasets_tts_segments,
            asr_test_datasets,
            250,
            training_args={"ctc_scale": 0.1},
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )

    exp_dict = run_exp(
        net_module + f"_spec_augment_ctc_scale_0.1_tts_segments/tuning/lm_2.5",
        train_args_spec_augment,
        training_datasets_tts_segments,
        asr_test_datasets2,
        250,
        training_args={"ctc_scale": 0.1},
        forward_args=forward_args,
        search_args={**default_search_args, **{"lm_weight": 2.5}},
    )

    train_args_radam = copy.deepcopy(train_args)
    train_args_radam["config"]["optimizer"] = {"class": "radam", "epsilon": 1e-8}
    exp_dict = run_exp(
        net_module + "_radam",
        train_args_radam,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )
    experiments[net_module + "_radam"] = exp_dict

    train_args_control = copy.deepcopy(train_args)
    net_module = "glowTTS_ASR_conformer_x_vector_control"
    train_args_control["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_control,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )

    train_args_control_radam = copy.deepcopy(train_args_control)
    train_args_control_radam["config"]["optimizer"] = {"class": "radam", "epsilon": 1e-8}
    exp_dict = run_exp(
        net_module + "_radam",
        train_args_control_radam,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    experiments[net_module] = exp_dict
    train_args_control_spec_augment = copy.deepcopy(train_args_control)
    train_args_control_spec_augment["net_args"]["model_config"]["specaug_config"] = asdict(specaug_config)
    exp_dict = run_exp(
        net_module + "_spec_augment",
        train_args_control_spec_augment,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )

    net_module = "glowTTS_x_vector"
    train_args_glowTTS_only = copy.deepcopy(train_args)
    train_args_glowTTS_only["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glowTTS_only,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )
    experiments[net_module] = exp_dict

    train_args_glowTTS_only_TTS_lr = copy.deepcopy(train_args_glowTTS_only)
    train_args_glowTTS_only_TTS_lr["config"]["learning_rates"] = list(np.linspace(1e-5, 5e-4, 125)) + list(
        np.linspace(5e-4, 1e-5, 125)
    )
    exp_dict = run_exp(
        net_module + "_TTS_LR_schedule",
        train_args_glowTTS_only_TTS_lr,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )

    train_args_glowTTS_only_radam = copy.deepcopy(train_args_glowTTS_only)
    train_args_glowTTS_only_radam["config"]["optimizer"] = {"class": "radam", "epsilon": 1e-9}
    exp_dict = run_exp(
        net_module + "_radam_1e_9",
        train_args_glowTTS_only_radam,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )

    train_args_glowTTS_only_radam_TTS_lr = copy.deepcopy(train_args_glowTTS_only_radam)
    train_args_glowTTS_only_radam_TTS_lr["config"]["learning_rates"] = list(np.linspace(1e-5, 5e-4, 125)) + list(
        np.linspace(5e-4, 1e-5, 125)
    )
    exp_dict = run_exp(
        net_module + "_TTS_LR_radam_1e_9",
        train_args_glowTTS_only_radam_TTS_lr,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )

    train_args_glowTTS_only_pe1 = copy.deepcopy(train_args_glowTTS_only)
    train_args_glowTTS_only_pe1["config"]["learning_rates"] = list(np.linspace(1e-5, 5e-4, 50)) + list(
        np.linspace(5e-4, 1e-5, 50)
    )
    exp_dict = run_exp(
        net_module + "_pe1",
        train_args_glowTTS_only_pe1,
        training_datasets_pe1,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )

    train_args_glowTTS_only_pe1["config"]["optimizer"] = {"class": "radam", "epsilon": 1e-9}
    exp_dict = run_exp(
        net_module + "_pe1_radam",
        train_args_glowTTS_only_pe1,
        training_datasets_pe1,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )

    train_args_glowTTS_only_no_dec_dropout = copy.deepcopy(train_args_glowTTS_only_pe1)
    train_args_glowTTS_only_no_dec_dropout["net_args"]["model_config"]["decoder_config"]["p_dropout"] = 0.0
    exp_dict = run_exp(
        net_module + "_pe1_radam_no_dec_dropout",
        train_args_glowTTS_only_no_dec_dropout,
        training_datasets_pe1,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )

    net_module = "glowTTS"
    train_args_glowTTS_only_simple_se = copy.deepcopy(train_args_glowTTS_only_pe1)
    train_args_glowTTS_only_simple_se["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glowTTS_only_simple_se,
        training_datasets_pe1,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
        use_speaker_labels_in_dev=True,
    )
    experiments[net_module] = exp_dict

    net_module = "glowTTS_ASR_conformer_x_vector_trainXvector"
    train_args_train_xvector = copy.deepcopy(train_args)
    train_args_train_xvector["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_train_xvector,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        extract_x_vector=True,
    )
    experiments[net_module] = exp_dict

    net_module = "glowTTS_ASR_conformer_x_vector_two_forward_pass"
    train_args_two_forward = copy.deepcopy(train_args_spec_augment)
    train_args_two_forward["network_module"] = net_module

    exp_dict = run_exp(
        net_module,
        train_args_two_forward,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )

    exp_dict = run_exp(
        net_module + "_ctc_scale_0.1",
        train_args_two_forward,
        training_datasets,
        asr_test_datasets,
        250,
        training_args={"ctc_scale": 0.1},
        forward_args=forward_args,
        search_args=default_search_args,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_two_forward,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )

        exp_dict = run_exp(
            net_module + f"_ctc_scale_0.1/tuning/lm_{lm}",
            train_args_two_forward,
            training_datasets,
            asr_test_datasets,
            250,
            training_args={"ctc_scale": 0.1},
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )

    net_module = "glowTTS_ASR_conformer_two_forward_pass"
    train_args_two_forward_no_xvector = copy.deepcopy(train_args_spec_augment)
    train_args_two_forward_no_xvector["network_module"] = net_module
    train_args_two_forward_no_xvector["net_args"]["model_config"]["gin_channels"] = 256
    del train_args_two_forward_no_xvector["config"]["preload_from_files"]

    exp_dict = run_exp(
        net_module,
        train_args_two_forward_no_xvector,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
    )
    exp_dict = run_exp(
        net_module + "_ctc_scale_0.1",
        train_args_two_forward_no_xvector,
        training_datasets,
        asr_test_datasets,
        250,
        training_args={"ctc_scale": 0.1},
        forward_args=forward_args,
        search_args=default_search_args,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_two_forward_no_xvector,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )
        exp_dict = run_exp(
            net_module + f"_ctc_scale_0.1/tuning/lm_{lm}",
            train_args_two_forward_no_xvector,
            training_datasets,
            asr_test_datasets,
            250,
            training_args={"ctc_scale": 0.1},
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
        )

    train_args_conformer_only = copy.deepcopy(train_args)
    train_args_conformer_only["net_args"]["model_config"] = asdict(model_config)
    net_module = "only_conformer"
    train_args_conformer_only["network_module"] = net_module
    del train_args_conformer_only["config"]["preload_from_files"]
    exp_dict = run_exp(
        net_module,
        train_args_conformer_only,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_conformer_only,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    train_args_conformer_only_spec_augment = copy.deepcopy(train_args_spec_augment)
    train_args_conformer_only_spec_augment["network_module"] = net_module
    del train_args_conformer_only_spec_augment["config"]["preload_from_files"]
    exp_dict = run_exp(
        net_module + "_spec_augment",
        train_args_conformer_only_spec_augment,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"_spec_augment/tuning/lm_{lm}",
            train_args_conformer_only_spec_augment,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    exp_dict = run_exp(
        net_module + "_spec_augment_tts_train_segments",
        train_args_conformer_only_spec_augment,
        training_datasets_tts_segments,
        asr_test_datasets2,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"_spec_augment_tts_train_segments/tuning/lm_{lm}",
            train_args_conformer_only_spec_augment,
            training_datasets_tts_segments,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    exp_dict = run_exp(
        net_module + f"_spec_augment_tts_train_segments/tuning/lm_2.5",
        train_args_conformer_only_spec_augment,
        training_datasets_tts_segments,
        asr_test_datasets2,
        250,
        forward_args=forward_args,
        search_args={**default_search_args, **{"lm_weight": 2.5}},
        tts_forward=False,
    )

    net_module = "glow_ASR_conformer"
    train_args_glow_conformer = copy.deepcopy(train_args_spec_augment)
    train_args_glow_conformer["network_module"] = net_module
    del train_args_glow_conformer["config"]["preload_from_files"]
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_glow_conformer,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    net_module = "glow_ASR_conformer_specaugment_before"
    train_args_glow_conformer_specaugment_before = copy.deepcopy(train_args_glow_conformer)
    train_args_glow_conformer_specaugment_before["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_specaugment_before,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_glow_conformer_specaugment_before,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    exp_dict = run_exp(
        net_module + "_tts_train_segments",
        train_args_glow_conformer_specaugment_before,
        training_datasets_tts_segments,
        asr_test_datasets2,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"_tts_train_segments/tuning/lm_{lm}",
            train_args_glow_conformer_specaugment_before,
            training_datasets_tts_segments,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    exp_dict = run_exp(
        net_module + f"_tts_train_segments/tuned/lm_2.5",
        train_args_glow_conformer_specaugment_before,
        training_datasets_tts_segments,
        asr_test_datasets2,
        250,
        forward_args=forward_args,
        search_args={**default_search_args, **{"lm_weight": 2.5}},
        tts_forward=False,
    )

    net_module = "glow_ASR_conformer_specaugment_before_xvector"
    train_args_glow_conformer_specaugment_before_x_vector = copy.deepcopy(train_args_glow_conformer)
    train_args_glow_conformer_specaugment_before_x_vector["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_specaugment_before_x_vector,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_glow_conformer_specaugment_before_x_vector,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    net_module = "glow_ASR_conformer_specaugment_before_xvector_control"
    train_args_glow_conformer_specaugment_before_x_vector_control = copy.deepcopy(train_args_glow_conformer)
    train_args_glow_conformer_specaugment_before_x_vector_control["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_specaugment_before_x_vector_control,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    net_module = "glow_ASR_conformer_specaugment_before_xvector_v2"
    train_args_glow_conformer_specaugment_before_x_vector_v2 = copy.deepcopy(
        train_args_glow_conformer_specaugment_before_x_vector
    )
    train_args_glow_conformer_specaugment_before_x_vector_v2["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_specaugment_before_x_vector_v2,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_glow_conformer_specaugment_before_x_vector_v2,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    net_module = "glow_ASR_conformer_specaugment_before_xvector_v3"
    train_args_glow_conformer_specaugment_before_x_vector_v3 = copy.deepcopy(
        train_args_glow_conformer_specaugment_before_x_vector_v2
    )
    train_args_glow_conformer_specaugment_before_x_vector_v3["network_module"] = net_module
    train_args_glow_conformer_specaugment_before_x_vector_v3["net_args"]["model_config"]["gin_channels"] = 256
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_specaugment_before_x_vector_v3,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_glow_conformer_specaugment_before_x_vector_v3,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    net_module = "glow_ASR_conformer_xvector"
    train_args_glow_conformer_xvector = copy.deepcopy(train_args_spec_augment)
    train_args_glow_conformer_xvector["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_xvector,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    train_args_glow_conformer_xvector["config"]["gradient_clip_norm"] = 10
    exp_dict = run_exp(
        net_module + "grad_clip_10",
        train_args_glow_conformer_xvector,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    for lm in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        exp_dict = run_exp(
            net_module + f"grad_clip_10/tuning/lm_{lm}",
            train_args_glow_conformer_xvector,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
        )

    train_args_glow_conformer_specaug_before_ddi_actnorm = copy.deepcopy(train_args_glow_conformer_specaugment_before)
    net_module = "glow_ASR_conformer_specaugment_before_ddi_actnorm"
    train_args_glow_conformer_specaug_before_ddi_actnorm["network_module"] = net_module
    train_args_glow_conformer_specaug_before_ddi_actnorm["config"]["gradient_clip"] = 0.5
    train_args_glow_conformer_specaug_before_ddi_actnorm["config"]["learning_rates"] = (
        list(np.linspace(1e-9, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_specaug_before_ddi_actnorm,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    train_args_glow_conformer_specaug_before_coupling_eps = copy.deepcopy(train_args_glow_conformer_specaugment_before)
    net_module = "glow_ASR_conformer_specaugment_before_coupling_epsilon"
    train_args_glow_conformer_specaug_before_coupling_eps["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_specaug_before_coupling_eps,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    train_args_glow_conformer_specaug_before_no_jit = copy.deepcopy(train_args_glow_conformer_specaugment_before)
    net_module = "glow_ASR_conformer_specaugment_before_no_jit"
    train_args_glow_conformer_specaug_before_no_jit["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_glow_conformer_specaug_before_no_jit,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
        given_train_job_for_forward=exp_dict["train_job"]
    )

    for lm in [1.5, 2.0, 2.5, 3.0]:
        exp_dict = run_exp(
            net_module + f"/tuning/lm_{lm}",
            train_args_glow_conformer_specaug_before_no_jit,
            training_datasets,
            asr_test_datasets,
            250,
            forward_args=forward_args,
            search_args={**default_search_args, **{"lm_weight": lm}},
            tts_forward=False,
            given_train_job_for_forward=exp_dict["train_job"]
        )

    train_args_spec_augment_ddi_actnorm = copy.deepcopy(train_args_spec_augment)
    net_module = "glowTTS_ASR_conformer_x_vector_ddi_actnorm"
    train_args_spec_augment_ddi_actnorm["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_spec_augment_ddi_actnorm,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )

    train_args_two_forward_no_xvector_ddi_actnorm = copy.deepcopy(train_args_two_forward_no_xvector)
    net_module = "glowTTS_ASR_conformer_two_forward_pass_ddi_actnorm"
    train_args_two_forward_no_xvector_ddi_actnorm["network_module"] = net_module
    exp_dict = run_exp(
        net_module,
        train_args_two_forward_no_xvector_ddi_actnorm,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )   
    return experiments
