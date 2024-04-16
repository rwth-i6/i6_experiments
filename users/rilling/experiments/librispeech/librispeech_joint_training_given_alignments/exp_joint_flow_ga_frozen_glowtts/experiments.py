import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_experiments.users.rilling.experiments.librispeech.librispeech_x_vectors.storage import x_vector_extractions

from ..data import (
    build_training_dataset,
    build_test_dataset,
    TrainingDatasetSettings,
    get_binary_lm,
    get_arpa_lm,
    get_text_lexicon,
    get_bliss_corpus_dict
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
    PhonemePredictionConfig,
    PhonemePredictionConfigCNN
)

from ..storage import tts_models


def get_glow_joint_flow_ga_frozen_glowtts(x_vector_exp, joint_exps, tts_exps, gl_checkpoint):
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/joint_training/given_alignments/raw_audio/joint_models/flow_given_alignment/"
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
        tts_forward=False,
        asr_search=False,
        phoneme_pred=True,
        asr_cv_set=False,
        given_train_job_for_forward=None,
        eval_invertibility=False,
    ):
        exp = {}

        assert len(args["config"]["learning_rates"]) == num_epochs, "Number of epochs and number of learning rates differ!"

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
            forward_config_gl = get_forward_config(
                forward_dataset=dataset,
                **{**args, **{"config": {"batch_size": 50 * 16000}}},
                forward_args={
                    **forward_args,
                    "gl_net_checkpoint": gl_checkpoint["checkpoint"],
                    "gl_net_config": gl_checkpoint["config"],
                },
                target="corpus_gl",
            )

        if asr_search or phoneme_pred:
            search_config = get_search_config(
                **args,
                search_args=search_args,
            )

        if phoneme_pred:
            phoneme_pred_config_encoder = get_search_config(
                **args,
                search_args={**search_args, "recognition_input": "encoder"},
                target="phoneme"
            )
            phoneme_pred_config_decoder = get_search_config(
                **args,
                search_args={**search_args, "recognition_input": "decoder"},
                target="phoneme"
            )

        if eval_invertibility:
            forward_config_invert = get_forward_config(
                forward_dataset=dataset, **{**args, **{"config": {"batch_size": 50 * 16000}}}, target="invertibility"
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
            forward_job_gl = tts_eval(
                checkpoint=train_job.out_checkpoints[num_epochs],
                prefix_name=prefix + name,
                returnn_config=forward_config_gl,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_exe_asr=RETURNN_PYTORCH_ASR_SEARCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                vocoder="gl",
                nisqa_eval=True, 
                swer_eval=True
            )
            exp["forward_job_gl"] = forward_job_gl

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
        if phoneme_pred:
            compute_phoneme_pred_accuracy(
                prefix + name + "/encoder_eval/",
                phoneme_pred_config_encoder,
                recognition_datasets=test_dataset,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT
            )

            compute_phoneme_pred_accuracy(
                prefix + name + "/decoder_eval/",
                phoneme_pred_config_decoder,
                recognition_datasets=test_dataset,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )

        if eval_invertibility:
            forward_job = forward(
                checkpoint=train_job.out_checkpoints[num_epochs],
                config=forward_config_invert,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name,
                target="invertibility",
            )
            exp["invertibility_job"] = forward_job
        # if "ce_loss_scale" in training_args and training_args["ce_loss_scale"] == 0.01:
        #     breakpoint()
        return exp

    glowTTS_durations_job = tts_exps["glowTTS/enc192/200ep/long_cooldown/not_silence_preprocessed"]["forward_job_joint_durations"]

    train_settings_pe1 = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )
    training_datasets_pe1_tts_segments = build_training_dataset(
        settings=train_settings_pe1,
        librispeech_key="train-clean-100",
        silence_preprocessing=False,
        use_tts_train_segments=True,
        durations_file=glowTTS_durations_job.out_hdf_files["output.hdf"],
        xvectors_file=x_vector_extractions["x_vector_cnn/1e-3_not_silence_preprocessed"]["hdf"],
    )

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    label_datastream_asr = cast(LabelDatastream, training_datasets_pe1_tts_segments.datastreams["phonemes_eow"])
    vocab_size_without_blank_asr = label_datastream_asr.vocab_size
    label_datastream_tts = cast(LabelDatastream, training_datasets_pe1_tts_segments.datastreams["phonemes"])
    vocab_size_without_blank_tts = label_datastream_tts.vocab_size
    speaker_datastream = cast(LabelDatastream, training_datasets_pe1_tts_segments.datastreams["speaker_labels"])

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

    phoeneme_prediction_config = PhonemePredictionConfig(
        n_channels=512,
        n_layers=3,
        p_dropout=0.1
    )

    phoneme_prediction_cnn_config = PhonemePredictionConfigCNN(
        n_channels=512, n_layers=3, kernel_size=5, p_dropout=0.1
    )

    model_config_ffn = ModelConfigV2(
        specaug_config=None,
        text_encoder_config=text_encoder_config,
        decoder_config=flow_decoder_config,
        phoneme_prediction_config=phoeneme_prediction_config,
        label_target_size=vocab_size_without_blank_tts,
        specauc_start_epoch=1,
        out_channels=80,
        gin_channels=256,
        n_speakers=speaker_datastream.vocab_size,
    )

    model_config_cnn = ModelConfigV2(
        specaug_config=None,
        text_encoder_config=text_encoder_config,
        decoder_config=flow_decoder_config,
        phoneme_prediction_config=phoneme_prediction_cnn_config,
        label_target_size=vocab_size_without_blank_tts,
        specauc_start_epoch=1,
        out_channels=80,
        gin_channels=256,
        n_speakers=speaker_datastream.vocab_size,
    )

    # model_config_no_dec_drop = copy.deepcopy(model_config)
    # model_config_no_dec_drop.decoder_config.p_dropout = 0.0

    net_module = "frozen_glowtts.ga_glowTTS_ASR_ffn_x_vector"

    train_args = {
        "net_args": {"fe_config": asdict(fe_config), "model_config": asdict(model_config_ffn)},
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "radam", "epsilon": 1e-9},
            "learning_rates": list(np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50)))),
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
            "preload_from_files": {
                "glowTTS_xvector": {
                    "filename": tts_models["glowTTS_x_vector_v2/enc768/200ep/dec_drop_0.05"].checkpoint,
                    "init_for_train": True,
                    "ignore_missing": True,
                }
            },
        },
    }

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    label_datastream = cast(LabelDatastream, training_datasets_pe1_tts_segments.datastreams["phonemes_eow"])

    forward_args = {"noise_scale": 0.66, "length_scale": 1}
    default_search_args = {}

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/encoder",
        train_args,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "encoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/decoder",
        train_args,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "decoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    train_args_cnn = copy.deepcopy(train_args)
    net_module = "frozen_glowtts.ga_glowTTS_ASR_cnn_x_vector"
    train_args_cnn["network_module"] = net_module
    train_args_cnn["net_args"]["model_config"] = asdict(model_config_cnn)
    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/encoder",
        train_args_cnn,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "encoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/decoder",
        train_args_cnn,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "decoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    train_args_ffn_mas = copy.deepcopy(train_args)
    net_module = "frozen_glowtts.ga_glowTTS_ASR_ffn_x_vector_mas"
    train_args_ffn_mas["network_module"] = net_module

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/encoder",
        train_args_ffn_mas,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "encoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/decoder",
        train_args_ffn_mas,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "decoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    # ===================== Simple Speaker Embeddings ================================
    net_module = "frozen_glowtts.ga_glowTTS_ASR_ffn"
    train_args_no_xvectors = copy.deepcopy(train_args)
    train_args_no_xvectors["network_module"] = net_module
    train_args_no_xvectors["config"]["preload_from_files"] = {
        "glowTTS_xvector": {
            "filename": tts_models["glowTTS/enc768/200ep/dec_drop_0.05"].checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/encoder",
        train_args_no_xvectors,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "encoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/decoder",
        train_args_no_xvectors,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "decoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    train_args_no_xvectors_cnn = copy.deepcopy(train_args_no_xvectors)
    net_module = "frozen_glowtts.ga_glowTTS_ASR_cnn"
    train_args_no_xvectors_cnn["network_module"] = net_module
    train_args_no_xvectors_cnn["net_args"]["model_config"] = asdict(model_config_cnn)
    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/encoder",
        train_args_no_xvectors_cnn,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "encoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/decoder",
        train_args_no_xvectors_cnn,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "decoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    net_module = "frozen_glowtts.ga_glowTTS_ASR_ffn_mas"
    train_args_no_xvectors_mas = copy.deepcopy(train_args_no_xvectors)
    train_args_no_xvectors_mas["network_module"] = net_module
    
    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/encoder",
        train_args_no_xvectors_mas,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "encoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/decoder",
        train_args_no_xvectors_mas,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "decoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    net_module = "frozen_glowtts.ga_glowTTS_ASR_ffn_mas_no_eval"
    train_args_no_xvectors_mas_no_eval = copy.deepcopy(train_args_no_xvectors_mas)
    train_args_no_xvectors_mas_no_eval["network_module"] = net_module

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/encoder",
        train_args_no_xvectors_mas_no_eval,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "encoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )

    exp_dict = run_exp(
        net_module.replace(".", "/") + "/100ep/decoder",
        train_args_no_xvectors_mas_no_eval,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        training_args={"recognition_input": "decoder"},
        search_args=default_search_args,
        phoneme_pred=True,
    )
