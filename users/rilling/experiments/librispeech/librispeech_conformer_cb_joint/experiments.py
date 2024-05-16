import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict
from torch import nn

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerBlockV1Config
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config

from i6_experiments.users.rilling.experiments.librispeech.librispeech_x_vectors.storage import x_vector_extractions


from .data import (
    build_training_dataset,
    build_test_dataset,
    build_tts_forward_dataset,
    TrainingDatasetSettings,
    get_binary_lm,
    get_arpa_lm,
    get_text_lexicon,
)
from .config import get_training_config, get_extract_durations_forward__config, get_forward_config, get_search_config
from .pipeline import training, forward, search

from i6_experiments.users.rilling.experiments.librispeech.common.tts_eval import tts_eval

from .default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, RETURNN_PYTORCH_ASR_SEARCH_EXE, MINI_RETURNN_ROOT
from .pytorch_networks.shared.model_config import (
    SpecaugConfig,
    ModelConfig,
    VGG4LayerActFrontendV1Config_mod,
    TextEncoderConfig,
    FlowDecoderConfig,
    ConformerCouplingFlowDecoderConfig,
    ConformerASREncoderConfig,
)


def get_conformer_coupling_glow(x_vector_exp, gl_checkpoint):
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/joint_training/conformer_coupling/raw_audio/"
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
        given_train_job_for_forward=None,
        tts_eval_datasets=None,
    ):

        assert not tts_forward or (
            "x_vector" not in name or tts_eval_datasets is not None
        ), "Attempting to evaluate a model with x-vector speaker embeddings, but missing explicit forward dataset with precalculated x-vector speaker embeddings."
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
            for ds_key, ds in tts_eval_datasets.items():
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
                    nisqa_eval=True,
                    swer_eval=True,
                    swer_eval_corpus_key=ds_key
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
        if asr_search:
            search(
                prefix + name + "/search",
                search_config,
                train_job.out_checkpoints[num_epochs],
                test_dataset,
                RETURNN_PYTORCH_ASR_SEARCH_EXE,
                MINI_RETURNN_ROOT,
            )
        return exp

    def tune_lm(
        alias,
        train_args,
        training_datasets,
        asr_test_datasets,
        num_epochs,
        search_args,
        additional_training_args={},
        lm_weights=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    ):
        for lm in lm_weights:
            run_exp(
                alias + f"/tuning/lm_{lm}",
                train_args,
                training_datasets,
                asr_test_datasets,
                num_epochs,
                training_args=additional_training_args,
                search_args={**search_args, **{"lm_weight": lm}},
                tts_forward=False,
            )

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

    forward_dataset_xvector = build_training_dataset(
        settings=train_settings,
        librispeech_key="train-clean-100",
        silence_preprocessing=False,
        xvectors_file=x_vector_extractions["x_vector_cnn/1e-3_not_silence_preprocessed"]["hdf"],
        use_tts_train_segments=True
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
    conformer_flow_decoder_config = ConformerCouplingFlowDecoderConfig(
        hidden_channels=192,
        kernel_size=5,
        dilation_rate=1,
        n_blocks=12,
        n_layers=4,
        p_dropout=0.05,
        n_split=4,
        n_sqz=2,
        n_heads=2,
        sigmoid_scale=False,
        ddi=True,
    )

    conformer_config = ConformerASREncoderConfig(
        conformer_size=96,
        num_layers=8,
        num_heads=2,
        ff_dim=384,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        kernel_size=9,
    )

    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        text_encoder_config=text_encoder_config,
        decoder_config=conformer_flow_decoder_config,
        conformer_asr_encoder_config=conformer_config,
        label_target_size=vocab_size_without_blank_asr,
        specaug_start_epoch=1,
        out_channels=80,
        gin_channels=512,
        final_dropout=0.2,
        n_speakers=speaker_datastream.vocab_size,
    )

    net_module = "glowTTS"

    train_args = {
        "net_args": {"fe_config": asdict(fe_config), "model_config": asdict(model_config)},
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-8},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
            + list(np.linspace(7e-4, 7e-5, 110))
            + list(np.linspace(7e-5, 1e-8, 30)),
            "batch_size": 75 * 16000,
            "accum_grad_multiple_step": 4,
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

    train_args_with_x_vector = copy.deepcopy(train_args)
    x_vect_train_job = x_vector_exp["x_vector_cnn/1e-3_not_silence_preprocessed"]["train_job"]
    train_args_with_x_vector["config"]["preload_from_files"] = {
        "x_vector_model": {
            "filename": x_vect_train_job.out_checkpoints[x_vect_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "prefix": "x_vector.",
            "ignore_missing": True,
        }
    }

    train_args_no_ddi = copy.deepcopy(train_args)
    train_args_no_ddi["net_args"]["model_config"]["decoder_config"]["ddi"] = False

    train_args_with_x_vector_no_ddi = copy.deepcopy(train_args_with_x_vector)
    train_args_with_x_vector_no_ddi["net_args"]["model_config"]["decoder_config"]["ddi"] = False

    experiments = {}

    alias = "ddi/" + net_module
    exp_dict = run_exp(
        alias,
        train_args,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        use_speaker_labels_in_dev=True,
        asr_search=False,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_100ep_TTS = copy.deepcopy(train_args)
    train_args_100ep_TTS["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50)))
    )

    alias = "ddi/" + net_module + "_100ep_pe1"
    exp_dict = run_exp(
        alias,
        train_args_100ep_TTS,
        training_datasets_pe1,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        use_speaker_labels_in_dev=True,
        asr_search=False,
        tts_forward=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_100ep_TTS_no_ddi = copy.deepcopy(train_args_100ep_TTS)
    train_args_100ep_TTS_no_ddi["net_args"]["model_config"]["decoder_config"]["ddi"] = False
    alias = "no_ddi/" + net_module + "_100ep_pe1"
    exp_dict = run_exp(
        alias,
        train_args_100ep_TTS_no_ddi,
        training_datasets_pe1,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        use_speaker_labels_in_dev=True,
        asr_search=False,
        tts_forward=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_100ep_TTS_radam = copy.deepcopy(train_args_100ep_TTS)
    train_args_100ep_TTS_radam["config"]["optimizer"]["class"] = "radam"
    train_args_100ep_TTS_radam["config"]["optimizer"]["epsilon"] = 1e-9
    alias = "ddi/" + net_module + "_100ep_pe1_radam1e-9"
    exp_dict = run_exp(
        alias,
        train_args_100ep_TTS_radam,
        training_datasets_pe1,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        use_speaker_labels_in_dev=True,
        asr_search=False,
        tts_eval_datasets=tts_forward_datasets,
    )

    experiments[alias] = exp_dict

    net_module = "glowTTS_ASR_conformer_two_forward_pass"
    # train_args["network_module"] = net_module
    # alias = "ddi/" + net_module
    # exp_dict = run_exp(
    #     alias,
    #     train_args,
    #     training_datasets,
    #     asr_test_datasets,
    #     250,
    #     forward_args=forward_args,
    #     search_args=default_search_args,
    #     tts_forward=True,
    #     tts_eval_datasets=tts_forward_datasets,
    # )

    # experiments[alias] = exp_dict

    # tune_lm(alias, train_args, training_datasets, asr_test_datasets, 250, search_args=default_search_args)

    train_args_no_ddi["network_module"] = net_module
    alias = "no_ddi/" + net_module
    exp_dict = run_exp(
        alias,
        train_args_no_ddi,
        training_datasets,
        asr_test_datasets,
        250,
        training_args={"ctc_scale": 0.1},
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    experiments[alias] = exp_dict

    tune_lm(alias, train_args_no_ddi, training_datasets, asr_test_datasets, 250, search_args=default_search_args, additional_training_args={"ctc_scale": 0.1})

    net_module = "glow_ASR_conformer"
    train_args["network_module"] = net_module
    alias = "ddi/" + net_module
    exp_dict = run_exp(
        alias,
        train_args,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )
    experiments[alias] = exp_dict

    tune_lm(alias, train_args, training_datasets, asr_test_datasets, 250, search_args=default_search_args)

    train_args_no_ddi["network_module"] = net_module
    alias = "no_ddi/" + net_module
    exp_dict = run_exp(
        alias,
        train_args_no_ddi,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )
    experiments[alias] = exp_dict

    tune_lm(alias, train_args_no_ddi, training_datasets, asr_test_datasets, 250, search_args=default_search_args)

    net_module = "glow_ASR_conformer_specaugment_before"
    train_args_no_ddi["network_module"] = net_module
    alias = "no_ddi/" + net_module
    exp_dict = run_exp(
        alias,
        train_args_no_ddi,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )
    experiments[alias] = exp_dict

    tune_lm(alias, train_args_no_ddi, training_datasets, asr_test_datasets, 250, search_args=default_search_args)

    train_args["network_module"] = net_module
    alias = "ddi/" + net_module
    exp_dict = run_exp(
        alias,
        train_args,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=False,
    )
    experiments[alias] = exp_dict

    tune_lm(alias, train_args, training_datasets, asr_test_datasets, 250, search_args=default_search_args)

    net_module = "glowTTS_x_vector"
    train_args_with_x_vector["network_module"] = net_module
    alias = "ddi/" + net_module
    exp_dict = run_exp(
        alias,
        train_args_with_x_vector,
        training_datasets,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
        tts_forward=True,
        tts_eval_datasets=tts_forward_datasets_xvectors,
    )

    net_module = "glowTTS_ASR_conformer_x_vector"
    train_args_with_x_vector["network_module"] = net_module
    # alias = "ddi/" + net_module
    # exp_dict = run_exp(
    #     alias,
    #     train_args_with_x_vector,
    #     training_datasets,
    #     asr_test_datasets,
    #     250,
    #     forward_args=forward_args,
    #     search_args=default_search_args,
    #     tts_forward=True,
    #     tts_eval_datasets=tts_forward_datasets_xvectors,
    # )
    # tune_lm(alias, train_args_with_x_vector, training_datasets, asr_test_datasets, 250, search_args=default_search_args)

    net_module = "glowTTS_ASR_conformer_x_vector_v2"
    train_args_with_x_vector_no_ddi["network_module"] = net_module
    alias = "no_ddi/" + net_module
    exp_dict = run_exp(
        alias,
        train_args_with_x_vector_no_ddi,
        training_datasets,
        asr_test_datasets,
        250,
        training_args={"ctc_scale": 0.1},
        forward_args=forward_args,
        search_args=default_search_args,
        tts_forward=True,
        tts_eval_datasets=tts_forward_datasets_xvectors,
    )
    tune_lm(alias, train_args_with_x_vector_no_ddi, training_datasets, asr_test_datasets, 250, search_args=default_search_args, additional_training_args={"ctc_scale": 0.1})

    return experiments
