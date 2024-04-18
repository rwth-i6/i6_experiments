import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_experiments.users.rilling.experiments.librispeech.librispeech_x_vectors.storage import x_vector_extractions

from ..data import (
    build_training_dataset,
    build_test_dataset,
    build_tts_forward_dataset,
    TrainingDatasetSettings,
    get_binary_lm,
    get_arpa_lm,
    get_text_lexicon,
    get_bliss_corpus_dict,
)
from ..config import get_training_config, get_extract_durations_forward__config, get_forward_config, get_search_config
from ..pipeline import training, forward, search, compute_phoneme_pred_accuracy
from i6_experiments.users.rilling.experiments.librispeech.common.tts_eval import tts_eval

from ..default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, RETURNN_PYTORCH_ASR_SEARCH_EXE, MINI_RETURNN_ROOT
from ..pytorch_networks.shared.configs import (
    SpecaugConfig,
    ModelConfigV1,
    ModelConfigV2,
    VGG4LayerActFrontendV1Config_mod,
    TextEncoderConfig,
    EmbeddingTextEncoderConfig,
    FlowDecoderConfig,
    MultiscaleFlowDecoderConfig,
    ConformerFlowDecoderConfig,
    PhonemePredictionConfig,
)

from ..storage import add_tts_model, TTSModel


def get_glow_tts(x_vector_exp, joint_exps, tts_exps, gl_checkpoint):
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/joint_training/given_alignments/raw_audio/TTS_models/"

    def run_exp(
        name,
        args,
        dataset,
        test_dataset,
        num_epochs=100,
        use_custom_engine=False,
        training_args={},
        forward_args={},
        keep_epochs=None,
        extract_x_vector=False,
        asr_cv_set=False,
        given_train_job_for_forward=None,
        nisqa_evaluation=True,
        swer_evaluation=True,
        tts_eval_datasets=None
    ):
        exp = {}
        assert len(args["config"]["learning_rates"]) == num_epochs, "Length of LR schedule and number of epochs differ."

        if given_train_job_for_forward is None:
            training_config = get_training_config(
                training_datasets=dataset,
                **args,
                training_args=training_args,
                use_custom_engine=use_custom_engine,
                keep_epochs=keep_epochs,
                asr_cv_set=asr_cv_set,
            )  # implicit reconstruction loss

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

        for ds_k, ds in tts_eval_datasets.items():
            forward_config_gl = get_forward_config(
                forward_dataset=ds,
                **{**args, **{"config": {"batch_size": 50 * 16000}}},
                forward_args={
                    **forward_args,
                    "gl_net_checkpoint": gl_checkpoint["checkpoint"],
                    "gl_net_config": gl_checkpoint["config"],
                },
                target="corpus_gl",
            )
            forward_job_gl = tts_eval(
                checkpoint=train_job.out_checkpoints[num_epochs],
                prefix_name=prefix + name,
                returnn_config=forward_config_gl,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_exe_asr=RETURNN_PYTORCH_ASR_SEARCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                vocoder="gl",
                nisqa_eval=nisqa_evaluation,
                swer_eval=swer_evaluation,
                swer_eval_corpus_key=ds_k
            )

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
        return exp

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=3, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    glowTTS_durations_job = tts_exps["glowTTS/enc192/200ep/long_cooldown/not_silence_preprocessed"][
        "forward_job_joint_durations"
    ]
    training_datasets_tts_segments = build_training_dataset(
        settings=train_settings,
        librispeech_key="train-clean-100",
        silence_preprocessing=False,
        use_tts_train_segments=True,
        durations_file=glowTTS_durations_job.out_hdf_files["output.hdf"],
    )

    train_settings_pe1 = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )
    training_datasets_pe1_tts_segments = build_training_dataset(
        settings=train_settings_pe1,
        librispeech_key="train-clean-100",
        silence_preprocessing=False,
        use_tts_train_segments=True,
        durations_file=glowTTS_durations_job.out_hdf_files["output.hdf"],
    )
    forward_datasets_pe1_tts_segments_xvectors = build_training_dataset(
        settings=train_settings_pe1,
        librispeech_key="train-clean-100",
        silence_preprocessing=False,
        use_tts_train_segments=True,
        durations_file=glowTTS_durations_job.out_hdf_files["output.hdf"],
        xvectors_file=x_vector_extractions["x_vector_cnn/1e-3_not_silence_preprocessed"]["hdf"],
    )

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    label_datastream_asr = cast(LabelDatastream, training_datasets_tts_segments.datastreams["phonemes_eow"])
    vocab_size_without_blank_asr = label_datastream_asr.vocab_size
    label_datastream_tts = cast(LabelDatastream, training_datasets_tts_segments.datastreams["phonemes"])
    vocab_size_without_blank_tts = label_datastream_tts.vocab_size
    speaker_datastream = cast(LabelDatastream, training_datasets_tts_segments.datastreams["speaker_labels"])

    from ..data import get_tts_log_mel_datastream
    from ..feature_config import DbMelFeatureExtractionConfig
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

    tts_forward_datasets = {}
    tts_forward_datasets_xvectors = {}

    tts_forward_datasets["test-clean"] = build_tts_forward_dataset(
        librispeech_key="train-clean-100",
        dataset_key="test-clean",
    )

    tts_forward_datasets_xvectors["test-clean"] = build_tts_forward_dataset(
        librispeech_key="train-clean-100",
        dataset_key="test-clean",
        xvectors_file=x_vector_extractions["x_vector_cnn/1e-3_not_silence_preprocessed/test-clean"]["hdf"],
    )

    asr_test_datasets = {}

    asr_test_datasets["dev-other"] = build_test_dataset(librispeech_key="train-clean-100", dataset_key="dev-other")

    asr_test_datasets2 = copy.deepcopy(asr_test_datasets)
    asr_test_datasets2["train-clean-100-cv"] = build_test_dataset(
        librispeech_key="train-clean-100", dataset_key="train-clean-100", test_on_tts_cv=True
    )
    asr_test_datasets2["dev-clean"] = build_test_dataset(librispeech_key="train-clean-100", dataset_key="dev-clean")

    dev_dataset_tuples_with_phon = {}
    for testset in ["train-clean"]:
        dev_dataset_tuples_with_phon[testset] = (
            training_datasets_pe1_tts_segments.cv,
            get_bliss_corpus_dict()["train-clean-100"],
        )

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

    flow_decoder_config_no_dropout = copy.deepcopy(flow_decoder_config)
    flow_decoder_config_no_dropout.p_dropout = 0.0

    phoeneme_prediction_config = PhonemePredictionConfig(n_channels=512, n_layers=3, p_dropout=0.1)

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
        text_encoder_config=text_encoder_config,
        decoder_config=flow_decoder_config,
        label_target_size=vocab_size_without_blank_asr,
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
            "learning_rates": list(np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50)))),
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
        },
    }
    lr_schedule_200ep = list(np.concatenate((np.linspace(1e-5, 5e-4, 100), np.linspace(5e-4, 1e-5, 100))))

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    label_datastream = cast(LabelDatastream, training_datasets_tts_segments.datastreams["phonemes_eow"])

    forward_args = {"noise_scale": 0.66, "length_scale": 1}

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

    train_args_TTS_xvector["config"]["optimizer"] = {"class": "radam", "epsilon": 1e-9}
    train_args_TTS_xvector["net_args"]["model_config"]["specaug_config"] = None
    exp_dict = run_exp(
        net_module + "/enc768/100ep/dec_drop_0.05",
        train_args_TTS_xvector,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
    )

    add_tts_model(
        net_module,
        TTSModel(
            config=ModelConfigV1.from_dict(train_args_TTS_xvector["net_args"]["model_config"]),
            checkpoint=exp_dict["train_job"].out_checkpoints[100],
        ),
    )
    # experiments[net_module] = exp_dict

    net_module = "glowTTS_x_vector_v2"
    train_args_TTS_xvector["net_args"]["model_config"]["gin_channels"] = 256
    train_args_TTS_xvector["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/enc768/100ep/dec_drop_0.05",
        train_args_TTS_xvector,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
    )
    add_tts_model(
        net_module,
        TTSModel(
            config=ModelConfigV1.from_dict(train_args_TTS_xvector["net_args"]["model_config"]),
            checkpoint=exp_dict["train_job"].out_checkpoints[100],
        ),
    )

    train_args_TTS_xvector_200ep = copy.deepcopy(train_args_TTS_xvector)
    train_args_TTS_xvector_200ep["config"]["learning_rates"] = lr_schedule_200ep
    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.05",
        train_args_TTS_xvector_200ep,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        swer_evaluation=True
    )
    add_tts_model(
        net_module + "/enc768/200ep/dec_drop_0.05", TTSModel(ModelConfigV1.from_dict(train_args_TTS_xvector_200ep["net_args"]["model_config"]), exp_dict["train_job"].out_checkpoints[200])
    )

    train_args_TTS_xvector_200ep["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 192
    exp_dict = run_exp(
        net_module + "/enc192/200ep/dec_drop_0.05",
        train_args_TTS_xvector_200ep,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        swer_evaluation=True
    )

    add_tts_model(
        net_module + "/enc192/200ep/dec_drop_0.05",
        TTSModel(
            ModelConfigV1.from_dict(train_args_TTS_xvector_200ep["net_args"]["model_config"]),
            exp_dict["train_job"].out_checkpoints[200],
        ),
    )

    train_args_TTS_xvector_200ep_no_dec_dropout = copy.deepcopy(train_args_TTS_xvector_200ep)
    train_args_TTS_xvector_200ep_no_dec_dropout["net_args"]["model_config"]["text_encoder_config"][
        "filter_channels"
    ] = 768
    train_args_TTS_xvector_200ep_no_dec_dropout["net_args"]["model_config"]["decoder_config"] = asdict(
        flow_decoder_config_no_dropout
    )
    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.0",
        train_args_TTS_xvector_200ep_no_dec_dropout,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        swer_evaluation=True
    )

    train_args_TTS_xvector_200ep_no_dec_dropout["net_args"]["model_config"]["text_encoder_config"][
        "filter_channels"
    ] = 192
    exp_dict = run_exp(
        net_module + "/enc192/200ep/dec_drop_0.0",
        train_args_TTS_xvector_200ep_no_dec_dropout,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        swer_evaluation=True
    )

    train_args_xvector_altLR = copy.deepcopy(train_args_TTS_xvector_200ep)
    train_args_xvector_altLR["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-6, 5e-4, 50), np.linspace(5e-4, 1e-6, 150)))
    )
    # train_args_xvector_altLR["config"]["gradient_clip_norm"] = 2.0
    train_args_xvector_altLR["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 768
    exp_dict = run_exp(
        net_module + "/enc768/200ep_long_cooldown/dec_drop_0.05",
        train_args_xvector_altLR,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        swer_evaluation=True
    )

    train_args_xvector_altLR_no_dec_drop = copy.deepcopy(train_args_xvector_altLR)
    train_args_xvector_altLR_no_dec_drop["net_args"]["model_config"]["decoder_config"] = asdict(flow_decoder_config_no_dropout)
    exp_dict = run_exp(
        net_module + "/enc768/200ep_long_cooldown/dec_drop_0.0",
        train_args_xvector_altLR_no_dec_drop,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        swer_evaluation=True
    )

    net_module = "glowTTS"
    train_args_TTS = copy.deepcopy(train_args_TTS_xvector)
    del train_args_TTS["config"]["preload_from_files"]
    train_args_TTS["network_module"] = net_module

    exp_dict = run_exp(
        net_module + "/enc768/100ep/dec_drop_0.05",
        train_args_TTS,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_TTS_100ep_no_dec_dropout = copy.deepcopy(train_args_TTS)
    train_args_TTS_100ep_no_dec_dropout["net_args"]["model_config"]["decoder_config"] = asdict(flow_decoder_config_no_dropout)
    train_args_TTS_100ep_no_dec_dropout["config"]["gradient_clip_norm"] = 10.0
    exp_dict = run_exp(
        net_module + "/enc768/100ep/dec_drop_0.00",
        train_args_TTS_100ep_no_dec_dropout,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )
    add_tts_model(net_module + "/enc768/100ep/dec_drop_0.00", TTSModel(ModelConfigV1.from_dict(train_args_TTS_100ep_no_dec_dropout["net_args"]["model_config"]), exp_dict["train_job"].out_checkpoints[100]))

    train_args_TTS_200ep = copy.deepcopy(train_args_TTS)
    train_args_TTS_200ep["config"]["learning_rates"] = lr_schedule_200ep
    train_args_TTS_200ep_alt_epsilon = copy.deepcopy(train_args_TTS_200ep)
    train_args_TTS_200ep_alt_epsilon["config"]["optimizer"]["epsilon"] = 1e-8

    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.05",
        train_args_TTS_200ep,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        swer_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )
    add_tts_model(net_module + "/enc768/200ep/dec_drop_0.05", TTSModel(ModelConfigV1.from_dict(train_args_TTS_200ep["net_args"]["model_config"]), exp_dict["train_job"].out_checkpoints[200]))

    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.05_epsilon_1e-8",
        train_args_TTS_200ep_alt_epsilon,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_TTS["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 192
    train_args_TTS_200ep["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 192
    train_args_TTS_200ep_alt_epsilon["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 192
    exp_dict = run_exp(
        net_module + "/enc192/100ep/dec_drop_0.05",
        train_args_TTS,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        swer_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    exp_dict = run_exp(
        net_module + "/enc192/200ep/dec_drop_0.05",
        train_args_TTS_200ep,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )
    add_tts_model(
        net_module + "/enc192/200ep/dec_drop_0.05",
        TTSModel(
            ModelConfigV1.from_dict(train_args_TTS_200ep["net_args"]["model_config"]),
            exp_dict["train_job"].out_checkpoints[200],
        ),
    )

    exp_dict = run_exp(
        net_module + "/enc192/200ep/dec_drop_0.05_epsilon_1e-8",
        train_args_TTS_200ep_alt_epsilon,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_TTS_no_dec_dropout = copy.deepcopy(train_args_TTS_200ep)
    train_args_TTS_no_dec_dropout["net_args"]["model_config"]["decoder_config"] = asdict(flow_decoder_config_no_dropout)
    train_args_TTS_no_dec_dropout["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 768
    train_args_TTS_no_dec_dropout["config"]["optimizer"]["epsilon"] = 1e-8

    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.0/epsilon_1e-8",
        train_args_TTS_no_dec_dropout,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_TTS_no_dec_dropout["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 192
    exp_dict = run_exp(
        net_module + "/enc192/200ep/dec_drop_0.0/epsilon_1e-8",
        train_args_TTS_no_dec_dropout,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_TTS_no_dec_dropout["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 768
    train_args_TTS_no_dec_dropout["config"]["optimizer"]["epsilon"] = 1e-9
    train_args_TTS_no_dec_dropout["config"]["grad"] = 1e-9
    train_args_TTS_no_dec_dropout["config"]["gradient_clip_norm"] = 10
    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.0/grad_clip_10",
        train_args_TTS_no_dec_dropout,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_TTS_no_dec_dropout["net_args"]["model_config"]["text_encoder_config"]["filter_channels"] = 192
    exp_dict = run_exp(
        net_module + "/enc192/200ep/dec_drop_0.0/grad_clip_10",
        train_args_TTS_no_dec_dropout,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets,
    )

    net_module = "glowTTS_simple_encoder"
    train_args_TTS_simple_encoder = copy.deepcopy(train_args_TTS_200ep)
    train_args_TTS_simple_encoder["network_module"] = net_module

    simple_text_encoder_config = EmbeddingTextEncoderConfig(
        n_vocab=label_datastream_tts.vocab_size,
        hidden_channels=192,
        filter_channels_dp=256,
        kernel_size=3,
        p_dropout=0.1,
        mean_only=False,
    )

    flow_decoder_config_20blocks = copy.deepcopy(flow_decoder_config)
    flow_decoder_config_20blocks.n_blocks = 20

    model_config_simple_encoder = ModelConfigV2(
        specaug_config=None,
        text_encoder_config=simple_text_encoder_config,
        decoder_config=flow_decoder_config,
        label_target_size=vocab_size_without_blank_asr,
        specauc_start_epoch=1,
        out_channels=80,
        gin_channels=256,
        n_speakers=speaker_datastream.vocab_size,
    )

    model_config_simple_encoder_20cb = copy.deepcopy(model_config_simple_encoder)
    model_config_simple_encoder_20cb.decoder_config = flow_decoder_config_20blocks

    train_args_TTS_simple_encoder["net_args"]["model_config"] = asdict(model_config_simple_encoder)

    exp_dict = run_exp(
        net_module + "/12cb/200ep/dec_drop_0.05",
        train_args_TTS_simple_encoder,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        swer_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_TTS_simple_encoder["net_args"]["model_config"] = asdict(model_config_simple_encoder_20cb)

    exp_dict = run_exp(
        net_module + "/20cb/200ep/dec_drop_0.05",
        train_args_TTS_simple_encoder,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        swer_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    # net_module = "glowTTS_simple_encoder_x_vector"
    # train_args_TTS_simple_encoder["network_module"] = net_module
    # exp_dict = run_exp(
    #     net_module + "/200ep/dec_drop_0.05",
    #     train_args_TTS_simple_encoder,
    #     training_datasets_pe1_tts_segments,
    #     asr_test_datasets,
    #     200,
    #     forward_args=forward_args,
    # )

    # ==================== Conformer Coupling =======================
    net_module = "glowTTS_x_vector_v2_conformer_coupling"
    train_args_TTS_xvector_200ep_conformer_coupling = copy.deepcopy(train_args_TTS_xvector_200ep)
    train_args_TTS_xvector_200ep_conformer_coupling["network_module"] = net_module
    model_config_conformer_coupling = copy.deepcopy(model_config)
    model_config_conformer_coupling.decoder_config = ConformerFlowDecoderConfig(
        hidden_channels=model_config.decoder_config.hidden_channels,
        kernel_size=model_config.decoder_config.kernel_size,
        dilation_rate=model_config.decoder_config.dilation_rate,
        n_blocks=model_config.decoder_config.n_blocks,
        n_layers=model_config.decoder_config.n_layers,
        n_heads=2,
        p_dropout=model_config.decoder_config.p_dropout,
        n_split=model_config.decoder_config.n_split,
        n_sqz=model_config.decoder_config.n_sqz,
        sigmoid_scale=model_config.decoder_config.sigmoid_scale
    )

    train_args_TTS_xvector_200ep_conformer_coupling["net_args"]["model_config"] = asdict(model_config_conformer_coupling)
    train_args_TTS_xvector_200ep_conformer_coupling["config"]["batch_size"] = 75 * 16000
    train_args_TTS_xvector_200ep_conformer_coupling["config"]["accum_grad_multiple_step"] = 4

    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.05",
        train_args_TTS_xvector_200ep_conformer_coupling,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        swer_evaluation=True
    )

    # ===================== Multi-Scale =======================
    net_module = "glowTTS_x_vector_v2_multiscale"
    train_args_TTS_xvector_200ep_multiscale = copy.deepcopy(train_args_TTS_xvector_200ep)
    train_args_TTS_xvector_200ep_multiscale["network_module"] = net_module

    model_config_multiscale = copy.deepcopy(model_config)
    model_config_multiscale.decoder_config = MultiscaleFlowDecoderConfig(
        hidden_channels=model_config.decoder_config.hidden_channels,
        kernel_size=model_config.decoder_config.kernel_size,
        dilation_rate=model_config.decoder_config.dilation_rate,
        n_blocks=model_config.decoder_config.n_blocks,
        n_layers=model_config.decoder_config.n_layers,
        p_dropout=model_config.decoder_config.p_dropout,
        n_split=model_config.decoder_config.n_split,
        n_sqz=model_config.decoder_config.n_sqz,
        sigmoid_scale=model_config.decoder_config.sigmoid_scale,
        n_early_every=4,
    )

    train_args_TTS_xvector_200ep_multiscale["net_args"]["model_config"] = asdict(model_config_multiscale)
    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.05",
        train_args_TTS_xvector_200ep_multiscale,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        swer_evaluation=True
    )

    # ============= Encoding Distance Loss ========================
    train_args_xvector_dist_loss = copy.deepcopy(train_args_TTS_xvector_200ep)
    net_module = "glowTTS_x_vector_v2_dist_loss"
    train_args_xvector_dist_loss["network_module"] = net_module

    for s in [0.1, 1.0]:
        exp_dict = run_exp(
            net_module + f"/ed_scale_{s}",
            train_args_xvector_dist_loss,
            training_datasets_pe1_tts_segments,
            asr_test_datasets,
            200,
            training_args={"ed_scale": s},
            forward_args=forward_args,
            tts_eval_datasets=tts_forward_datasets_xvectors,
        )

    net_module = "glowTTS_x_vector_v2_logdist_loss"
    train_args_xvector_dist_loss["network_module"] = net_module

    for s in [0.1, 1.0]:
        exp_dict = run_exp(
            net_module + f"/ed_scale_{s}",
            train_args_xvector_dist_loss,
            training_datasets_pe1_tts_segments,
            asr_test_datasets,
            200,
            training_args={"ed_scale": s},
            forward_args=forward_args,
            tts_eval_datasets=tts_forward_datasets_xvectors,
        )

    train_args_xvector_dist_loss["config"]["gradient_clip_norm"] = 10.0
    for s in [0.1, 1.0]:
        exp_dict = run_exp(
            net_module + f"_grad_clip_10/ed_scale_{s}",
            train_args_xvector_dist_loss,
            training_datasets_pe1_tts_segments,
            asr_test_datasets,
            200,
            training_args={"ed_scale": s},
            forward_args=forward_args,
            tts_eval_datasets=tts_forward_datasets_xvectors,
        )

    # ================= InvBatchNorm =================
    net_module = "glowTTS_batch_norm"
    train_args_TTS_200ep_batch_norm = copy.deepcopy(train_args_TTS_200ep)
    train_args_TTS_200ep_batch_norm["network_module"] = net_module
    train_args_TTS_200ep_batch_norm["net_args"]["model_config"] = asdict(model_config)

    exp_dict = run_exp(
        net_module + "/enc768/200ep/dec_drop_0.05",
        train_args_TTS_200ep_batch_norm,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        200,
        forward_args=forward_args,
        swer_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    # ================== 400 EP =======================
    model_config_400ep = ModelConfigV1(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        text_encoder_config=text_encoder_config,
        decoder_config=flow_decoder_config,
        label_target_size=None,
        ffn_layers=None,
        ffn_channels=None,
        specauc_start_epoch=None,
        out_channels=80,
        gin_channels=512,
        n_speakers=speaker_datastream.vocab_size,
    )
    net_module = "glowTTS"
    train_args_400 = {
        "net_args": {"fe_config": asdict(fe_config), "model_config": asdict(model_config_tts_only)},
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "radam", "epsilon": 1e-9},
            "learning_rates": list(np.concatenate((np.linspace(1e-5, 5e-4, 100), np.linspace(5e-4, 1e-6, 300)))),
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
        },
    }

    exp_dict = run_exp(
        net_module + "/enc768/400ep/dec_drop_0.05",
        train_args_400,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        400,
        forward_args=forward_args,
        swer_evaluation=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_400_grad_norm = copy.deepcopy(train_args_400)
    train_args_400_grad_norm["config"]["gradient_clip_norm"] = 10
    exp_dict = run_exp(
        net_module + "/enc768/400ep/grad_clip_10/dec_drop_0.05",
        train_args_400_grad_norm,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        400,
        forward_args=forward_args,
        swer_evaluation=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    net_module = "glowTTS_x_vector_v2"
    train_args_400_xvector = copy.deepcopy(train_args_400)
    train_args_400_xvector["network_module"] = net_module
    train_args_400_xvector["config"]["preload_from_files"] = {
        "x_vector_model": {
            "filename": x_vect_train_job.out_checkpoints[x_vect_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "prefix": "x_vector.",
            "ignore_missing": True,
        }
    }

    exp_dict = run_exp(
        net_module + "/enc768/400ep/dec_drop_0.05",
        train_args_400_xvector,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        400,
        forward_args=forward_args,
        swer_evaluation=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets_xvectors,
    )
