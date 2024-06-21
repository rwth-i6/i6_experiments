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
from ..config import get_training_config, get_extract_durations_forward__config, get_forward_config, get_search_config, get_prior_config
from ..pipeline import training, forward, search, compute_phoneme_pred_accuracy, compute_prior

from i6_experiments.users.rilling.experiments.librispeech.common.tts_eval import tts_eval

from ..default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, RETURNN_PYTORCH_ASR_SEARCH_EXE, MINI_RETURNN_ROOT
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config

from ..pytorch_networks.shared.configs import (
    SpecaugConfig,
    ModelConfigV1,
    ModelConfigV2,
    VGG4LayerActFrontendV1Config_mod,
    TextEncoderConfig,
    EmbeddingTextEncoderConfig,
    FlowDecoderConfig,
    PhonemePredictionConfig,
    PhonemePredictionConfigCNN,
    PhonemePredictionConfigBLSTM, 
    ConformerASRConfig
)

from ..storage import tts_models, add_tts_model, TTSModel


def get_glow_joint_2step(x_vector_exp, joint_exps, tts_exps, gl_checkpoint):
    """
    Experiments performing two step training (first block) or further training on pre-trained models from ../exp_joint and ../exp_joint_ga (second and third block)
    The experiments contain further TTS training after a first train step with very strong auxiliary loss (first block)
    but also ASR and phoneme prediction trainings after a pre-training (last blocks)
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
        tts_forward=True,
        asr_search=True,
        phoneme_pred=True,
        asr_cv_set=False,
    ):
        """Creates the jobs to run training, forwarding/TTS evaluation and phoneme prediction evaluation for the given experiment

        :param str name: Name of the experiment used for aliases
        :param dict args: General arguments used for the Returnn Configs
        :param TrainingDataset dataset: Dataset for training
        :param dict test_dataset: Dictionary of datasets used for evaluation 
        :param int num_epochs: Number of epochs for training, defaults to 100
        :param bool use_custom_engine: whether a custom engine is used in Returnn, defaults to False
        :param dict training_args: Additional arguments for training, defaults to {}
        :param dict forward_args: Additional arguments for TTS forward, defaults to {}
        :param dict search_args: Additional arguments for phoneme prediction, defaults to {}
        :param list[int] keep_epochs: List of numbers of checkpoints that are supposed to be kept during training, defaults to None
        :param bool tts_forward: whether TTS evaluation should be run (autoMOS, NISQA), defaults to True
        :param bool asr_search: whether ASR search should be run, defaults to True
        :param bool phoneme_pred: whether phoneme prediction evaluation should be run, defaults to True
        :param bool asr_cv_set: whether the training should use the ASR validation set for CV during training (necessary for ASR trainings), defaults to False
        :return dict: Dictionary containing all the jobs of the experiment
        """        
        exp = {}

        assert num_epochs == len(args["config"]["learning_rates"]), "Number of Epochs and Number of LR steps differs!"

        with_prior = "prior_scale" in search_args

        training_config = get_training_config(
            training_datasets=dataset,
            **args,
            training_args=training_args,
            use_custom_engine=use_custom_engine,
            keep_epochs=keep_epochs,
            asr_cv_set=asr_cv_set,
        )

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
            phoneme_pred_config = get_search_config(
                **args,
                search_args=search_args,
                target="phoneme"
            )

        train_job = training(
            config=training_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
            num_epochs=num_epochs,
        )
        exp["train_job"] = train_job

        if with_prior:
            returnn_config = get_prior_config(training_datasets=dataset, **args)
            prior_file = compute_prior(
                prefix + name,
                returnn_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_PYTORCH_ASR_SEARCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(prefix + name + "/prior.txt", prior_file)
            search_args["prior_file"] = prior_file

        if tts_forward:
            forward_job_gl = tts_eval(
                checkpoint=train_job.out_checkpoints[num_epochs],
                prefix_name=prefix + name,
                returnn_config=forward_config_gl,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_exe_asr=RETURNN_PYTORCH_ASR_SEARCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                vocoder="gl",
            )
            exp["forward_job_gl"] = forward_job_gl

        if asr_search:
            search(
                prefix + name + "/search",
                search_config,
                train_job.out_checkpoints[num_epochs],
                test_dataset,
                RETURNN_PYTORCH_ASR_SEARCH_EXE,
                MINI_RETURNN_ROOT,
            )
        if phoneme_pred:
            compute_phoneme_pred_accuracy(
                prefix + name,
                phoneme_pred_config,
                recognition_datasets=test_dataset,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT
            )
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

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=3, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    training_datasets_pe3 = build_training_dataset(
        settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=False
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

    model_config = ModelConfigV2(
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

    model_config_no_dec_drop = copy.deepcopy(model_config)
    model_config_no_dec_drop.decoder_config.p_dropout = 0.0

    net_module = "ga_glowTTS_ASR_ffn_x_vector_v2"

    train_args_first_step = {
        "net_args": {"fe_config": asdict(fe_config), "model_config": asdict(model_config)},
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "radam", "epsilon": 1e-9},
            "learning_rates": list(np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50)))),
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
        },
    }

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    forward_args = {"noise_scale": 0.66, "length_scale": 1}
    default_search_args = {
        "lexicon": get_text_lexicon(),
        "returnn_vocab": label_datastream_asr.vocab,
        "beam_size": 256,
        "arpa_lm": get_binary_lm(),
        "lm_weight": 5,
        "beam_threshold": 16,
        "asr_data": False,
    }

    default_search_args_tts = {
        "lexicon": get_text_lexicon(tts=True),
        "returnn_vocab": label_datastream_tts.vocab,
        "beam_size": 256,
        "arpa_lm": get_binary_lm(),
        "lm_weight": 5,
        "beam_threshold": 16,
        "asr_data": False,
    }

    x_vect_train_job = x_vector_exp["x_vector_cnn/1e-3_not_silence_preprocessed"]["train_job"]
    train_args_first_step["config"]["preload_from_files"] = {
        "x_vector_model": {
            "filename": x_vect_train_job.out_checkpoints[x_vect_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "prefix": "x_vector.",
            "ignore_missing": True,
        }
    }

    train_args_pretrained_tts = copy.deepcopy(train_args_first_step)
    train_args_pretrained_tts["config"]["preload_from_files"] = {
        "glowTTS_xvector": {
            "filename": tts_models["glowTTS_x_vector_v2"].checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    for ce in [1.0, 2.0, 3.0, 4.0]:
        exp_dict = run_exp(
            "first_step/" + net_module + f"/ce_ls_{ce}",
            train_args_first_step,
            training_datasets_pe1_tts_segments,
            dev_dataset_tuples_with_phon,
            100,
            forward_args=forward_args,
            training_args={"ce_loss_scale": ce},
            search_args=default_search_args,
            asr_search=False,
        )
        add_tts_model(name="first_step/" + net_module + f"/ce_ls_{ce}", model=TTSModel(config=model_config, checkpoint=exp_dict["train_job"].out_checkpoints[100]))

    # ========== 2nd Step FFN CE ============
    net_module = "ga_glowTTS_ASR_ffn_x_vector_v2_2ndstep"
    train_args_2nd_step = copy.deepcopy(train_args_first_step)
    train_args_2nd_step["network_module"] = net_module
    train_args_2nd_step["config"]["preload_from_files"] = {
        "glowTTS_xvector": {
            "filename": tts_models["first_step/ga_glowTTS_ASR_ffn_x_vector_v2/ce_ls_1.0"].checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    exp_dict = run_exp(
        "second_step/" + net_module + f"/ce_ls_1.0",
        train_args_2nd_step,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )
    # ========== 2nd Step BLSTM CTC =========
    model_config_blstm_ctc = copy.deepcopy(model_config)
    model_config_blstm_ctc.phoneme_prediction_config = PhonemePredictionConfigBLSTM(
        n_channels=512, n_layers=2, p_dropout=0.1, subsampling_factor=4
    )

    model_config_blstm_ctc_specaug = copy.deepcopy(model_config_blstm_ctc)
    model_config_blstm_ctc_specaug.specaug_config = specaug_config
    model_config_blstm_ctc_specaug.specauc_start_epoch = 1

    net_module = "ga_glowTTS_ASR_ffn_x_vector_v2_blstm_2ndstep"
    train_args_blstm_ctc = copy.deepcopy(train_args_first_step)
    train_args_blstm_ctc["network_module"] = net_module
    train_args_blstm_ctc["net_args"]["model_config"] = asdict(model_config_blstm_ctc)
    train_args_blstm_ctc["config"]["preload_from_files"] = {
        "glowTTS_xvector": {
            "filename": tts_models["first_step/ga_glowTTS_ASR_ffn_x_vector_v2/ce_ls_1.0"].checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
            "ignore_params_prefixes": ["asr_ouput"]
        }
    }
    train_args_blstm_ctc_specaug = copy.deepcopy(train_args_blstm_ctc)
    train_args_blstm_ctc_specaug["net_args"]["model_config"] = asdict(model_config_blstm_ctc_specaug)

    exp_dict = run_exp(
        "second_step/" + net_module + f"/no_specaug/ce_ls_1.0",
        train_args_blstm_ctc,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args_tts,
        tts_forward=False,
        phoneme_pred=False
    )

    exp_dict = run_exp(
        "second_step/" + net_module + f"/specaug/ce_ls_1.0",
        train_args_blstm_ctc_specaug,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args_tts,
        tts_forward=False,
        phoneme_pred=False,
    )

    net_module = "ga_glowTTS_ASR_ffn_x_vector_v2_blstm_2ndstep_v2"
    train_args_blstm_ctc["network_module"] = net_module
    exp_dict = run_exp(
        "second_step/" + net_module + f"/no_specaug/ce_ls_1.0",
        train_args_blstm_ctc,
        training_datasets_pe1_tts_segments,
        asr_test_datasets,
        100,
        forward_args=forward_args,
        search_args=default_search_args_tts,
        tts_forward=False,
        phoneme_pred=False,
    )

    # ============= 2nd step TTS flow training =====================
    train_args_2nd_step_tts = copy.deepcopy(train_args_2nd_step)
    net_module = "ga_glowTTS_ASR_ffn_x_vector_v2_2ndstep_tts"
    train_args_2nd_step_tts["network_module"] = net_module

    exp_dict = run_exp(
        "second_step_tts/" + net_module + f"/ce_ls_1.0",
        train_args_2nd_step_tts,
        training_datasets_pe1_tts_segments,
        dev_dataset_tuples_with_phon,
        100,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=False,
    )


    # ==================== CNN pretrained 2nd step Conformer ======================
    net_module = "frozen_glowtts.glowASR_conformer_x_vector"
    first_step_cnn = tts_models["glowTTS_ASR_cnn_x_vector/tts_pretrained/no_specaug/tts_target_size/ce_ls_0.1"]
    model_config = first_step_cnn.config

    specaug_config_conf = SpecaugConfig(
        repeat_per_n_frames=100,
        max_dim_feat=8,
        num_repeat_feat=5,
        max_dim_time=20,
    )

    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        out_features=384,
        activation_str="ReLU",
    )

    model_config.phoneme_prediction_config = ConformerASRConfig(
        frontend_config=frontend_config,
        label_target_size=vocab_size_without_blank_asr,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )
    
    model_config.specaug_config = specaug_config_conf
    model_config.specauc_start_epoch = 1

    train_args = {
        "net_args": {"fe_config": asdict(fe_config), "model_config": asdict(model_config)},
        "network_module": net_module,
        "debug": True,
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-8},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            "batch_size": 360 * 16000,
            # "gradient_clip_norm": 1.0,
            "stop_on_nonfinite_train_score": False,
            # "max_seq_length": {"audio_features": 25 * 16000},
            # "max_seqs": 60,
            "preload_from_files": {
                "glowTTS": {
                    "filename": first_step_cnn.checkpoint,
                    "init_for_train": True,
                    "ignore_missing": True,
                    "ignore_params_prefixes": ["encoder", "phoneme_pred_cnn", "phoneme_pred_output"],
                }
            }
        },
    }

    exp_dict = run_exp(
        "second_step_asr/" + net_module.replace(".", "/"),
        train_args,
        training_datasets_pe3,
        asr_test_datasets,
        250,
        forward_args=forward_args,
        search_args=default_search_args,
        asr_search=True,
        asr_cv_set=True,
        phoneme_pred=False,
    )

    for lm_w in [2, 2.5, 3.0, 3.5, 4.0, 4.5]:
        for ps in [0, 0.3, 0.5]:
            additional_search_args = {"lm_weight": lm_w} if ps == 0 else {"lm_weight": lm_w, "prior_scale": ps}
            suffix = f"/tuning/lm_{lm_w}" if ps == 0 else f"/tuning/lm_{lm_w}_ps_{ps}"
            exp_dict = run_exp(
                "second_step_asr/" + net_module.replace(".", "/") + suffix,
                train_args,
                training_datasets_pe3,
                asr_test_datasets,
                250,
                forward_args=forward_args,
                search_args={**default_search_args, **additional_search_args},
                asr_search=True,
                asr_cv_set=True,
                phoneme_pred=False,
            )