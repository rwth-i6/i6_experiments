from sisyphus import tk

from dataclasses import asdict

import numpy as np

import copy

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from .i6modelsV1_VGG4LayerActFrontendV1_v4_cfg import SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig

from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset, get_binary_lm, get_arpa_lm
from .data import get_text_lexicon as get_text_lexicon_tts
from .data_ctc import build_training_datasets_normal_ctc, build_test_dataset_normal_ctc
from .data_ctc import get_text_lexicon as get_text_lexicon_asr
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from .pipeline import training, search, compute_prior

from .config import get_training_config, get_search_config, get_prior_config


def glowASR(TTS_experiments: dict):
    prefix_name = "experiments/librispeech/librispeech_glow_asr/pytorch/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=3, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets("train-clean-100", settings=train_settings)
    train_data_normal_ctc = build_training_datasets_normal_ctc("train-clean-100", settings=train_settings)

    # build testing datasets
    dev_dataset_tuples = {}
    dev_dataset_normal_ctc_tuples = {}
    # for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )

        dev_dataset_normal_ctc_tuples[testset] = build_test_dataset_normal_ctc(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )
    dev2_dataset_tuples = {}
    dev2_dataset_normal_ctc_tuples = {}
    for testset in ["dev-other", "dev-clean"]:
        dev2_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )

        dev2_dataset_normal_ctc_tuples[testset] = build_test_dataset_normal_ctc(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )

    test_dataset_tuples = {}
    test_dataset_normal_ctc_tuples = {}
    for testset in ["test-clean"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )

        test_dataset_normal_ctc_tuples[testset] = build_test_dataset_normal_ctc(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )

        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

    label_datastream = cast(LabelDatastream, train_data.datastreams["phon_labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    label_datastream_normal_ctc = cast(LabelDatastream, train_data_normal_ctc.datastreams["phon_labels"])
    vocab_size_without_blank_normal_ctc = label_datastream_normal_ctc.vocab_size

    from .data import get_tts_log_mel_datastream

    log_mel_datastream = get_tts_log_mel_datastream(silence_preprocessing=False)

    from .pytorch_networks.shared.configs import DbMelFeatureExtractionConfig
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import DBMelFilterbankOptions

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

    config = {}

    def run_exp(
        ft_name,
        datasets,
        train_args,
        search_args=None,
        with_prior=False,
        num_epochs=100,
        extra_evaluate_epoch=None,
        test_datasets=dev_dataset_tuples,
        large_gpu_training=False,
    ):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        returnn_search_config = get_search_config(**train_args, search_args=search_args)
        train_job = training(
            ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs, large_gpu=large_gpu_training
        )

        if with_prior:
            returnn_config = get_prior_config(training_datasets=datasets, **train_args)
            prior_file = compute_prior(
                ft_name,
                returnn_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(ft_name + "/prior.txt", prior_file)
            search_args["prior_file"] = prior_file

        # averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        # best_checkpoint = get_best_checkpoint(train_job)
        if not "test" in ft_name:
            search(
                ft_name + "/default_250",
                returnn_search_config,
                train_job.out_checkpoints[num_epochs],
                test_datasets,
                RETURNN_EXE,
                MINI_RETURNN_ROOT,
            )

        if extra_evaluate_epoch is not None:
            search(
                ft_name + "/default_250/extra",
                returnn_search_config,
                train_job.out_checkpoints[extra_evaluate_epoch],
                test_datasets,
                RETURNN_EXE,
                MINI_RETURNN_ROOT,
            )
        # search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        # search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    train_args = {
        "net_args": {"n_vocab": vocab_size_without_blank, "gin_channels": 256, "fe_config": asdict(fe_config)},
        "network_module": "basic_glowASR_linear",
        "debug": True,
        "config": {
            "preload_from_files": {
                "existing-model": {
                    "filename": "/u/lukas.rilling/experiments/glow_tts_asr_v2/alias/experiments/librispeech/tts_architecture/glow_tts/pytorch/glowTTS_warmup/training/output/models/epoch.100.pt",
                    "init_for_train": True,
                    "ignore_params_prefixes": ["encoder"],
                    "ignore_missing": True,
                }
            }
        },
    }

    default_search_args = {
        # "lexicon": get_text_lexicon_tts(),
        "lexicon": "/u/lukas.rilling/experiments/glow_tts_asr_v2/lexicon.txt",
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 32,
        "arpa_lm": get_arpa_lm(),
        "lm_weight": 5,
        "beam_threshold": 50,
    }
    # run_exp(prefix_name + "test", datasets=train_data, train_args=train_args)

    train_args2 = copy.deepcopy(train_args)
    train_args2["net_args"]["final_n_layers"] = 5
    train_args2["net_args"]["final_hidden_channels"] = 512

    run_exp(prefix_name + "/test/linear", datasets=train_data, train_args=train_args2, search_args=default_search_args)

    # train_args2["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))
    # run_exp(prefix_name + "test/linear", datasets=train_data, train_args=train_args2)

    train_args_conv = copy.deepcopy(train_args)
    train_args_conv["network_module"] = "basic_glowASR_conv"
    train_args_conv["net_args"]["final_n_layers"] = 4
    train_args_conv["net_args"]["final_hidden_channels"] = 512

    run_exp(
        prefix_name + "/test/conv/no_warmup",
        datasets=train_data,
        train_args=train_args_conv,
        search_args=default_search_args,
    )

    train_args_conv["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5 * 1e-4, 50), np.linspace(5 * 1e-4, 1e-5, 50)))
    )

    run_exp(
        prefix_name + "/test/conv/warmup",
        datasets=train_data,
        train_args=train_args_conv,
        search_args=default_search_args,
    )

    train_args_blstm = copy.deepcopy(train_args)
    train_args_blstm["network_module"] = "basic_glowASR_blstm"
    train_args_blstm["net_args"]["final_n_layers"] = 4

    run_exp(
        prefix_name + "/test/blstm/no_warmup",
        datasets=train_data,
        train_args=train_args_blstm,
        search_args=default_search_args,
    )

    train_args_blstm2 = copy.deepcopy(train_args_blstm)
    train_args_blstm2["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5 * 1e-4, 50), np.linspace(5 * 1e-4, 1e-5, 50)))
    )
    run_exp(
        prefix_name + "/test/blstm/warmup",
        datasets=train_data,
        train_args=train_args_blstm2,
        search_args=default_search_args,
    )

    train_args_blstm3 = copy.deepcopy(train_args_blstm)
    train_args_blstm3["net_args"]["final_hidden_channels"] = 512
    run_exp(prefix_name + "/test/blstm512/no_warmup", datasets=train_data, train_args=train_args_blstm3)

    train_args_blstm4 = copy.deepcopy(train_args_blstm3)
    train_args_blstm4["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5 * 1e-4, 50), np.linspace(5 * 1e-4, 1e-5, 50)))
    )

    # train_args_blstm["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))
    run_exp(
        prefix_name + "/test/blstm512/warmup/default_search",
        datasets=train_data,
        train_args=train_args_blstm4,
        search_args=default_search_args,
    )

    additional_search_args = {}
    additional_search_args["lm_weight"] = 10
    additional_search_args["beam_size"] = 32
    run_exp(
        prefix_name + "/test/blstm512/warmup/search_lmW10",
        datasets=train_data,
        train_args=train_args_blstm4,
        search_args={**default_search_args, **additional_search_args},
    )

    train_args_blstm4["net_args"]["p_dropout"] = 0.2

    run_exp(
        prefix_name + "/test/blstm512/warmup/d0.2_b300",
        datasets=train_data,
        train_args=train_args_blstm4,
        search_args=default_search_args,
    )

    train_args_blstm4["config"]["batch_size"] = 100 * 16000

    run_exp(
        prefix_name + "/test/blstm512/warmup/d0.2_b100",
        datasets=train_data,
        train_args=train_args_blstm4,
        search_args=default_search_args,
        extra_evaluate_epoch=72,
    )

    train_args_blstm4["config"]["batch_size"] = 600 * 16000

    run_exp(
        prefix_name + "/test/blstm512/warmup/d0.2_b600",
        datasets=train_data,
        train_args=train_args_blstm4,
        search_args=default_search_args,
    )

    train_args_blstm4["config"]["accum_grad_multiple_step"] = 4
    run_exp(
        prefix_name + "/test/blstm512/warmup/d0.2_b600_ga4",
        datasets=train_data,
        train_args=train_args_blstm4,
        search_args=default_search_args,
    )

    train_args_blstm_frame_stack = copy.deepcopy(train_args_blstm3)
    train_args_blstm_frame_stack["network_module"] = "glowASR_blstm_frame_stack"
    train_args_blstm_frame_stack["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5 * 1e-4, 50), np.linspace(5 * 1e-4, 1e-5, 50)))
    )
    train_args_blstm_frame_stack["net_args"]["p_dropout"] = 0.2
    train_args_blstm_frame_stack["net_args"]["final_hidden_channels"] = 512
    train_args_blstm_frame_stack["net_args"]["final_n_layers"] = 2
    train_args_blstm_frame_stack["net_args"]["subsampling_factor"] = 4

    # default_search_args["beam_size"] = 256
    # default_search_args["lm_weight"] = 5
    # default_search_args["beam_threshold"] = 16
    # default_search_args_tts = copy.deepcopy(default_search_args)
    # default_search_args_asr = copy.deepcopy(default_search_args)

    default_search_args = {
        "beam_size": 256,
        "arpa_lm": get_binary_lm(),
        "lm_weight": 5,
        "beam_threshold": 16,
    }

    default_search_args_tts = copy.deepcopy(default_search_args)
    # default_search_args_tts["lexicon"] = get_text_lexicon_tts()
    default_search_args_tts[
        "lexicon"
    ] = "/u/lukas.rilling/experiments/glow_tts_asr_v2/lexicon.txt"  # TODO: Fix this and use lexicon from Job with deleted [silence] token
    default_search_args_tts["arpa_lm"] = get_arpa_lm()  # TODO: Delete this when fix above is done.
    default_search_args_tts["returnn_vocab"] = label_datastream.vocab
    # default_search_args_tts["asr_data"] = False

    default_search_args_asr = copy.deepcopy(default_search_args)
    default_search_args_asr["lexicon"] = get_text_lexicon_asr()
    default_search_args_asr["returnn_vocab"] = label_datastream_normal_ctc.vocab
    default_search_args_asr["asr_data"] = True

    default_search_args_tts_fix = copy.deepcopy(default_search_args_tts)
    default_search_args_tts_fix["lexicon"] = get_text_lexicon_tts()
    default_search_args_tts_fix["arpa_lm"] = get_binary_lm()
    default_search_args_tts_fix["asr_data"] = False
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/default",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts_fix,
    )
    # run_exp(
    #     prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/search_param/beam128",
    #     datasets=train_data,
    #     train_args=train_args_blstm_frame_stack,
    #     search_args={**default_search_args_tts, **{"beam_size": 128}},
    # )
    # run_exp(
    #     prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/search_params/lm4",
    #     datasets=train_data,
    #     train_args=train_args_blstm_frame_stack,
    #     search_args={**default_search_args_tts, **{"lm_weight": 4}},
    # )
    # run_exp(
    #     prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/search_params/lm6",
    #     datasets=train_data,
    #     train_args=train_args_blstm_frame_stack,
    #     search_args={**default_search_args_tts, **{"lm_weight": 6}},
    # )

    # for lm_w in [2.0, 2.5, 3.0, 3.5]:
    #     run_exp(
    #         prefix_name + f"blstm_2x512_d0.2_b300_fs4/tts_dataset/search_params/lm{lm_w}",
    #         datasets=train_data,
    #         train_args=train_args_blstm_frame_stack,
    #         search_args={**default_search_args_tts, **{"lm_weight": lm_w}},
    #     )

    # for t in [14, 16, 18, 20, 25, 30]:
    #     run_exp(
    #         prefix_name + f"blstm_2x512_d0.2_b300_fs4/tts_dataset/search_params/beam1024_bt{t}",
    #         datasets=train_data,
    #         train_args=train_args_blstm_frame_stack,
    #         search_args={**default_search_args_tts, **{"beam_size": 1024, "beam_threshold": t}},
    #     )

    # run_exp(
    #     prefix_name + f"blstm_2x512_d0.2_b300_fs4/tts_dataset/search_params/bt25_lm3_5",
    #     datasets=train_data,
    #     train_args=train_args_blstm_frame_stack,
    #     search_args={**default_search_args_tts, **{"beam_threshold": 25, "lm_weight": 3.5}},
    # )

    train_args_blstm_frame_stack["net_args"]["n_vocab"] = vocab_size_without_blank_normal_ctc
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/default",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )

    train_args_blstm_frame_stack_dropout = copy.deepcopy(train_args_blstm_frame_stack)
    train_args_blstm_frame_stack_dropout["net_args"]["dropout_around_blstm"] = True
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/drop_around_blstm/lm5",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack_dropout,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )
    # run_exp(
    #     prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/drop_around_blstm/lm4",
    #     datasets=train_data_normal_ctc,
    #     train_args=train_args_blstm_frame_stack_dropout,
    #     search_args={**default_search_args_asr, **{"lm_weight": 4}},
    #     test_datasets=dev_dataset_normal_ctc_tuples,
    # )
    # run_exp(
    #     prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/drop_around_blstm/lm6",
    #     datasets=train_data_normal_ctc,
    #     train_args=train_args_blstm_frame_stack_dropout,
    #     search_args={**default_search_args_asr, **{"lm_weight": 6}},
    #     test_datasets=dev_dataset_normal_ctc_tuples,
    # )

    train_args_blstm_frame_stack_dropout["net_args"]["spec_augment"] = True
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/drop_around_blstm/spec_augment",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack_dropout,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.v8r3HeTPk9q5/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_enc768",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.w4RgUEr0Et25/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_enc768/not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )

    train_args_4blstm_frame_stack = copy.deepcopy(train_args_blstm_frame_stack)
    train_args_4blstm_frame_stack["net_args"]["final_n_layers"] = 4
    run_exp(
        prefix_name + "blstm_4x512_d0.2_b300_fs4/asr_dataset/glow_enc768/not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_4blstm_frame_stack,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )

    train_args_blstm1024_frame_stack = copy.deepcopy(train_args_blstm_frame_stack)
    train_args_blstm1024_frame_stack["net_args"]["final_hidden_channels"] = 1024
    run_exp(
        prefix_name + "blstm_2x1024_d0.2_b300_fs4/asr_dataset/glow_enc768/not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm1024_frame_stack,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.8PSFfojvDJ2D/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc768/with_sigma/not_silence_preprocessed/lm5",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_enc768/with_sigma/not_silence_preprocessed/lm5",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )
    # run_exp(
    #     prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_enc768/with_sigma/not_silence_preprocessed/lm4",
    #     datasets=train_data_normal_ctc,
    #     train_args=train_args_blstm_frame_stack,
    #     search_args={**default_search_args_asr, **{"lm_weight": 4}},
    #     test_datasets=dev_dataset_normal_ctc_tuples,
    # )
    # run_exp(
    #     prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_enc768/with_sigma/not_silence_preprocessed/lm6",
    #     datasets=train_data_normal_ctc,
    #     train_args=train_args_blstm_frame_stack,
    #     search_args={**default_search_args_asr, **{"lm_weight": 6}},
    #     test_datasets=dev_dataset_normal_ctc_tuples,
    # )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.576hp04ASmp5/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc768/with_sigma/silence_preprocessing",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_enc768/with_sigma/silence_preprocessing",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )

    # for t in [14, 16, 18, 20, 25, 30]:
    #     run_exp(
    #         prefix_name + f"blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc768/with_sigma/silence_preprocessing_bt_comparison/beam1024_bt{t}",
    #         datasets=train_data,
    #         train_args=train_args_blstm_frame_stack,
    #         search_args={**default_search_args_tts, **{"beam_size": 1024, "beam_threshold": t}},
    #     )

    train_args_blstm_only = copy.deepcopy(train_args_blstm_frame_stack)
    del train_args_blstm_only["config"]["preload_from_files"]
    train_args_blstm_only["network_module"] = "only_blstm_frame_stack"
    run_exp(
        prefix_name + "only_blstm_2x512_d0.2_b300_fs4/asr_dataset",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_only,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )

    for lm_w in [2.5, 3.0, 3.5, 4.0]:
        for ps in [0, 0.3, 0.5]:
            additional_search_args = {"lm_weight": lm_w, "prior_scale": ps}
            run_exp(
                prefix_name + f"only_blstm_2x512_d0.2_b300_fs4/asr_dataset/tuning/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_blstm_only,
                search_args={**default_search_args_asr, **additional_search_args},
                test_datasets=dev_dataset_normal_ctc_tuples,
            )

    run_exp(
        prefix_name + "only_blstm_2x512_d0.2_b300_fs4/tts_dataset",
        datasets=train_data,
        train_args=train_args_blstm_only,
        search_args=default_search_args_tts,
        test_datasets=dev_dataset_tuples,
    )

    # for lm_w in [2.0, 2.5, 3.0, 3.5]:
    #     run_exp(
    #         prefix_name + f"only_blstm_2x512_d0.2_b300_fs4/tts_dataset/search_params/lm{lm_w}",
    #         datasets=train_data,
    #         train_args=train_args_blstm_only,
    #         search_args={**default_search_args_tts, **{"lm_weight": lm_w}},
    #     )

    # for t in [14, 16, 18, 20, 25, 30]:
    #     run_exp(
    #         prefix_name + f"only_blstm_2x512_d0.2_b300_fs4/tts_dataset/search_params/beam1024_bt{t}",
    #         datasets=train_data,
    #         train_args=train_args_blstm_only,
    #         search_args={**default_search_args_tts, **{"beam_size": 1024, "beam_threshold": t}},
    #     )

    train_args_linear = copy.deepcopy(train_args_blstm_frame_stack)
    train_args_linear["network_module"] = "glowASR_linear_frame_stack"
    train_args_linear["net_args"]["final_n_layers"] = 1
    run_exp(
        prefix_name + "linear_1x512_d0.2_b300_fs4/glow_enc768/tts_dataset",
        datasets=train_data,
        train_args=train_args_linear,
        search_args=default_search_args_tts,
        test_datasets=dev_dataset_tuples,
    )

    train_args_linear["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.resp1KPTqtf7/output/models/epoch.100.pt"
    run_exp(
        prefix_name + "linear_1x512_d0.2_b300_fs4/glow_nar_taco_encoder_16blocks",
        datasets=train_data,
        train_args=train_args_linear,
        search_args=default_search_args_tts,
    )

    # ------------------------------  BLSTM ------------------------------ #

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.cHG7DjqrOKtx/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_with_small_enc/silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.GdZl97LaScpY/output/models/epoch.100.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc192/not_silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_enc192/not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_asr,
        test_datasets=dev_dataset_normal_ctc_tuples,
    )
    for lm_w in [2.5, 3.0, 3.5, 4.0]:
        for ps in [0, 0.3, 0.5]:
            additional_search_args = {"lm_weight": lm_w, "prior_scale": ps}
            run_exp(
                prefix_name + f"blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_enc192/not_silence_preprocessed/tuned/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_blstm_frame_stack,
                search_args={**default_search_args_asr, **additional_search_args},
                test_datasets=dev_dataset_normal_ctc_tuples,
            )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.WsV7aygCG89f/output/models/epoch.100.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc192/100epTTS/silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.AzntDeTvU6Qa/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc192/200epsTTS/silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.AzntDeTvU6Qa/output/models/epoch.100.pt"
    run_exp(
        prefix_name
        + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc192/200epsTTS_early_eval_ep100/silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.nmK7Mhq7biaG/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc192/200epsTTS/not_silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.nmK7Mhq7biaG/output/models/epoch.100.pt"
    run_exp(
        prefix_name
        + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_enc192/200epsTTS_early_eval_ep100/not_silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.ZPejnErmVbBS/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_nar_taco_encoder/silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.q0TNvar5rBFy/output/models/epoch.200.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_nar_taco_encoder/not_silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.resp1KPTqtf7/output/models/epoch.100.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_nar_taco_encoder_16blocks/not_silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.7DYhobwqP2vc/output/models/epoch.084.pt"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_simple_encoder_epoch84/silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack["config"]["preload_from_files"]["existing-model"]["filename"] = tk.Path(
        "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.OA7TIuJfvLsR/output/models/epoch.100.pt"
    )
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/tts_dataset/glow_simple_encoder/silence_preprocessed",
        datasets=train_data,
        train_args=train_args_blstm_frame_stack,
        search_args=default_search_args_tts,
    )

    train_args_blstm_frame_stack_asr = copy.deepcopy(train_args_blstm_frame_stack)
    train_args_blstm_frame_stack_asr["net_args"]["vocab"] = vocab_size_without_blank_normal_ctc
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/glow_simple_encoder/silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_frame_stack_asr,
        search_args=default_search_args_asr,
    )

    # ------------------------------ Conformer ------------------------------ #

    train_args_conformer = copy.deepcopy(train_args_blstm_frame_stack)
    train_args_conformer["network_module"] = "glowASR_conformer"
    train_args_conformer["net_args"]["p_dropout"] = 0.2
    train_args_conformer["net_args"]["spec_augment"] = True
    train_args_conformer["config"]["learning_rates"] = (
        list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    )
    # glowTTS enc192 200ep not-silence-preprocessed:
    train_args_conformer["config"]["max_seq_length"] = None
    train_args_conformer["config"]["batch_size"] = 360 * 16000
    train_args_conformer["config"]["preload_from_files"]["existing-model"]["filename"] = tk.Path(
        "/work/asr3/rossenbach/rilling/sisyphus_work_dirs/glow_tts_asr_v2/i6_core/returnn/training/ReturnnTrainingJob.nmK7Mhq7biaG/output/models/epoch.200.pt"
    )
    train_args_conformer["net_args"]["n_vocab"] = vocab_size_without_blank
    run_exp(
        prefix_name + "conformer/tts_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed",
        datasets=train_data,
        train_args=train_args_conformer,
        search_args=default_search_args_tts,
        large_gpu_training=True,
        num_epochs=250,
    )

    train_args_conformer_asr_data = copy.deepcopy(train_args_conformer)
    train_args_conformer_asr_data["net_args"]["n_vocab"] = vocab_size_without_blank_normal_ctc
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_asr_data,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )

    train_args_glow100ep_conformer_asr_data = copy.deepcopy(train_args_conformer_asr_data)
    tts_exp_name = "glowTTS/enc192/100ep/not_silence_preprocessed"
    tts_train_job = TTS_experiments[tts_exp_name]["train_job"]
    train_args_glow100ep_conformer_asr_data["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = tts_train_job.out_checkpoints[
        tts_train_job.returnn_config.get("num_epochs", 100)
    ]

    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc192_100ep_not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_glow100ep_conformer_asr_data,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )

    train_args_glow_enc768_200ep_conformer_asr_data = copy.deepcopy(train_args_conformer_asr_data)
    tts_exp_name = "glowTTS/enc768/200ep/not_silence_preprocessed"
    tts_train_job = TTS_experiments[tts_exp_name]["train_job"]
    train_args_glow_enc768_200ep_conformer_asr_data["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = tts_train_job.out_checkpoints[
        tts_train_job.returnn_config.get("num_epochs", 200)
    ]
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc768_200ep_not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_glow_enc768_200ep_conformer_asr_data,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )
    
    # train_args_glow_enc768_100ep_conformer_asr_data = copy.deepcopy(train_args_conformer_asr_data)
    # tts_exp_name = "glowTTS/enc768/100ep/not_silence_preprocessed"
    # tts_train_job = TTS_experiments[tts_exp_name]["train_job"]
    # train_args_glow_enc768_100ep_conformer_asr_data["config"]["preload_from_files"]["existing-model"][
    #     "filename"
    # ] = tts_train_job.out_checkpoints[
    #     tts_train_job.returnn_config.get("num_epochs", 100)
    # ]
    # run_exp(
    #     prefix_name + "conformer/asr_dataset/spec_augment/glow_enc768_100ep_not_silence_preprocessed",
    #     datasets=train_data_normal_ctc,
    #     train_args=train_args_glow_enc768_100ep_conformer_asr_data,
    #     search_args=default_search_args_asr,
    #     large_gpu_training=True,
    #     num_epochs=250,
    # )

    train_args_conformer_asr_data_layer_norm = copy.deepcopy(train_args_conformer_asr_data)
    train_args_conformer_asr_data_layer_norm["net_args"]["layer_norm"] = True
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed/layer_norm",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_asr_data_layer_norm,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )

    train_args_conformer_asr_data_batch_norm = copy.deepcopy(train_args_conformer_asr_data)
    train_args_conformer_asr_data_batch_norm["net_args"]["batch_norm"] = True
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed/batch_norm",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_asr_data_batch_norm,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )

    train_args_conformer_no_spec_augment = copy.deepcopy(train_args_conformer)
    train_args_conformer_no_spec_augment["net_args"]["spec_augment"] = False
    run_exp(
        prefix_name + "conformer/tts_dataset/no_spec_augment/glow_enc192_200ep_not_silence_preprocessed",
        datasets=train_data,
        train_args=train_args_conformer_no_spec_augment,
        search_args=default_search_args_tts,
        large_gpu_training=True,
        num_epochs=250,
    )

    train_args_conformer_no_spec_augment_asr_data = copy.deepcopy(train_args_conformer_no_spec_augment)
    train_args_conformer_no_spec_augment_asr_data["net_args"]["n_vocab"] = vocab_size_without_blank_normal_ctc

    run_exp(
        prefix_name + "conformer/asr_dataset/no_spec_augment/glow_enc192_200ep_not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_no_spec_augment_asr_data,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )


    train_args_conformer_only = copy.deepcopy(train_args_conformer)
    train_args_conformer_only["network_module"] = "only_conformer"
    train_args_conformer_only["config"].pop("preload_from_files")
    run_exp(
        prefix_name + "conformer/tts_dataset/spec_augment/no_glow",
        datasets=train_data,
        train_args=train_args_conformer_only,
        search_args=default_search_args_tts,
        large_gpu_training=True,
        num_epochs=250,
    )

    train_args_conformer_only_asr_data = copy.deepcopy(train_args_conformer_asr_data)
    train_args_conformer_only_asr_data["network_module"] = "only_conformer"

    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/no_glow",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_only_asr_data,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )

    train_args_conformer_only_asr_data_no_spec_augment = copy.deepcopy(train_args_conformer_only_asr_data)
    train_args_conformer_only_asr_data_no_spec_augment["net_args"]["spec_augment"] = False
    run_exp(
        prefix_name + "conformer/asr_dataset/no_spec_augment/no_glow",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_only_asr_data_no_spec_augment,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )

    for lm_w in [2.5, 3.0, 3.5, 4.0]:
        for ps in [0, 0.3, 0.5]:
            additional_search_args = {"lm_weight": lm_w, "prior_scale": ps}
            run_exp(
                prefix_name
                + f"conformer/asr_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed/search_params/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                with_prior=True,
                num_epochs=250,
                test_datasets=dev_dataset_normal_ctc_tuples,
            )

            run_exp(
                prefix_name + f"conformer/asr_dataset/spec_augment/glow_enc192_100ep_not_silence_preprocessed/tuning/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_glow100ep_conformer_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                num_epochs=250,
                with_prior=True,
                test_datasets=dev_dataset_normal_ctc_tuples
            )

            run_exp(
                prefix_name + f"conformer/asr_dataset/spec_augment/glow_enc768_200ep_not_silence_preprocessed/tuning/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_glow_enc768_200ep_conformer_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                num_epochs=250,
                with_prior=True,
                test_datasets=dev_dataset_normal_ctc_tuples
            )
            
            # run_exp(
            #     prefix_name + f"conformer/asr_dataset/spec_augment/glow_enc768_100ep_not_silence_preprocessed/tuning/lm_{lm_w}_ps_{ps}",
            #     datasets=train_data_normal_ctc,
            #     train_args=train_args_glow_enc768_100ep_conformer_asr_data,
            #     search_args={**default_search_args_asr, **additional_search_args},
            #     large_gpu_training=True,
            #     num_epochs=250,
            #     with_prior=True,
            #     test_datasets=dev_dataset_normal_ctc_tuples
            # )
                    # run_exp(
            #     prefix_name
            #     + f"conformer/asr_dataset/spec_augment/glow_enc192_100ep_not_silence_preprocessed/search_params/lm_{lm_w}_ps_{ps}",
            #     datasets=train_data_normal_ctc,
            #     train_args=train_args_conformer_asr_data,
            #     search_args={**default_search_args_asr, **additional_search_args},
            #     large_gpu_training=True,
            #     with_prior=True,
            #     num_epochs=250,
            #     test_datasets=dev_dataset_normal_ctc_tuples,
            # )
            run_exp(
                prefix_name + f"conformer/asr_dataset/spec_augment/no_glow/search_params/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_only_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                with_prior=True,
                num_epochs=250,
                test_datasets=dev_dataset_normal_ctc_tuples,
            )
            run_exp(
                prefix_name + f"conformer/asr_dataset/no_spec_augment/no_glow/search_params/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_only_asr_data_no_spec_augment,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                with_prior=True,
                num_epochs=250,
                test_datasets=dev_dataset_normal_ctc_tuples,
            )
            run_exp(
                prefix_name + f"conformer/asr_dataset/no_spec_augment/glow_enc192_200ep_not_silence_preprocessed/search_params/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_no_spec_augment_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                num_epochs=250,
            )
    optimized_search_args = {"lm_weight": 3.5, "prior_scale": 0.5}
    run_exp(
        prefix_name + f"conformer/asr_dataset/spec_augment/no_glow/tuned",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_only_asr_data,
        search_args={**default_search_args_asr, **optimized_search_args},
        large_gpu_training=True,
        with_prior=True,
        num_epochs=250,
        test_datasets=test_dataset_normal_ctc_tuples,
    )
    run_exp(
        prefix_name + f"conformer/asr_dataset/spec_augment/no_glow/tuned",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_only_asr_data,
        search_args={**default_search_args_asr, **optimized_search_args},
        large_gpu_training=True,
        with_prior=True,
        num_epochs=250,
        test_datasets=dev2_dataset_normal_ctc_tuples,
    )
    run_exp(
        prefix_name + f"conformer/asr_dataset/spec_augment/no_glow/tuned_no_prior",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_only_asr_data,
        search_args={**default_search_args_asr, **{"lm_weight": 3.5, "prior_scale": 0.0}},
        large_gpu_training=True,
        with_prior=True,
        num_epochs=250,
        test_datasets=dev2_dataset_normal_ctc_tuples,
    )

    optimized_search_args = {"lm_weight": 3.0, "prior_scale": 0.5}
    run_exp(
        prefix_name + f"conformer/asr_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed/tuned",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_asr_data,
        search_args={**default_search_args_asr, **optimized_search_args},
        large_gpu_training=True,
        with_prior=True,
        num_epochs=250,
        test_datasets=test_dataset_normal_ctc_tuples,
    )
    run_exp(
        prefix_name + f"conformer/asr_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed/tuned",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_asr_data,
        search_args={**default_search_args_asr, **optimized_search_args},
        large_gpu_training=True,
        with_prior=True,
        num_epochs=250,
        test_datasets=dev2_dataset_normal_ctc_tuples,
    )
    train_args_conformer_speaker_drop = copy.deepcopy(train_args_conformer)
    train_args_conformer_speaker_drop_asr_data = copy.deepcopy(train_args_conformer_asr_data)

    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 1]:
        TTS_exp_name = f"glowTTS/enc192/100ep/speaker_drop/p_speaker_drop_{p}_not_silence_preprocessed"
        assert TTS_exp_name in TTS_experiments, "Experiment reference not found!"

        TTS_exp_train_job = TTS_experiments[TTS_exp_name]["train_job"]
        train_args_conformer_speaker_drop_asr_data["config"]["preload_from_files"]["existing-model"][
            "filename"
        ] = TTS_exp_train_job.out_checkpoints[TTS_exp_train_job.returnn_config.get("num_epochs", 100)]

        run_exp(
            prefix_name
            + f"conformer/asr_dataset/spec_augment/glow_enc192_100ep_not_silence_preprocessed_speaker_drop_{p}",
            datasets=train_data_normal_ctc,
            train_args=train_args_conformer_speaker_drop_asr_data,
            search_args=default_search_args_asr,
            large_gpu_training=True,
            num_epochs=250,
        )

    speaker_drop_05_TTS_exp_name = "glowTTS/enc192/100ep/speaker_drop/p_speaker_drop_0.5_not_silence_preprocessed"
    speaker_drop_05_TTS_exp_train_job = TTS_experiments[speaker_drop_05_TTS_exp_name]["train_job"]
    train_args_conformer_speaker_drop_asr_data["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = speaker_drop_05_TTS_exp_train_job.out_checkpoints[
        speaker_drop_05_TTS_exp_train_job.returnn_config.get("num_epochs", 100)
    ]

    for lm_w in [2.5, 3.0, 3.5, 4.0]:
        for ps in [0, 0.3, 0.5]:
            additional_search_args = {"lm_weight": lm_w, "prior_scale": ps}
            run_exp(
                prefix_name
                + f"conformer/asr_dataset/spec_augment/glow_enc192_100ep_not_silence_preprocessed_speaker_drop_0.5/search_params/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_speaker_drop_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                num_epochs=250,
                with_prior=True,
            )
    speaker_drop_1_TTS_exp_name = "glowTTS/enc192/100ep/speaker_drop/p_speaker_drop_1_not_silence_preprocessed"
    speaker_drop_1_TTS_exp_train_job = TTS_experiments[speaker_drop_1_TTS_exp_name]["train_job"]
    train_args_conformer_speaker_drop_asr_data["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = speaker_drop_1_TTS_exp_train_job.out_checkpoints[
        speaker_drop_1_TTS_exp_train_job.returnn_config.get("num_epochs", 100)
    ]

    for lm_w in [2.5, 3.0, 3.5, 4.0]:
        for ps in [0, 0.3, 0.5]:
            additional_search_args = {"lm_weight": lm_w, "prior_scale": ps}
            run_exp(
                prefix_name
                + f"conformer/asr_dataset/spec_augment/glow_enc192_100ep_not_silence_preprocessed_speaker_drop_1/search_params/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_speaker_drop_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                num_epochs=250,
                with_prior=True,
            )
    # ------------------------------ No Freezing Experiments ------------------------------

    train_args_conformer_no_freeze_asr_data = copy.deepcopy(train_args_conformer_asr_data)
    train_args_conformer_no_freeze_asr_data["config"]["batch_size"] = 150 * 16000
    train_args_conformer_no_freeze_asr_data["config"]["accum_grad_multiple_step"] = 2
    train_args_conformer_no_freeze_asr_data["network_module"] = "glowASR_conformer_no_freeze_spec_augment"
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed_not_freezed",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_no_freeze_asr_data,
        search_args=default_search_args_asr,
        large_gpu_training=False,
        num_epochs=250,
    )
    
    train_args_conformer_no_freeze_asr_data_spec_aug_before = copy.deepcopy(train_args_conformer_no_freeze_asr_data)
    train_args_conformer_no_freeze_asr_data_spec_aug_before["network_module"] = "glowASR_conformer_no_freeze_spec_augment_before"
    train_args_conformer_no_freeze_asr_data_spec_aug_before["net_args"]["spec_augment"] = True
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment_before/glow_enc192_200ep_not_silence_preprocessed_not_freezed",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_no_freeze_asr_data_spec_aug_before,
        search_args=default_search_args_asr,
        large_gpu_training=False,
        num_epochs=250,
    )
    
    train_args_conformer_asr_data_spec_aug_before = copy.deepcopy(train_args_conformer_asr_data)
    train_args_conformer_asr_data_spec_aug_before["network_module"] = "glowASR_conformer_specaug_before"
    train_args_conformer_asr_data_spec_aug_before["net_args"]["spec_augment"] = True
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment_before/glow_enc192_200ep_not_silence_preprocessed",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_asr_data_spec_aug_before,
        search_args=default_search_args_asr,
        large_gpu_training=False,
        num_epochs=250,
    )

    train_args_conformer_no_freeze_asr_data_no_spec_augment = copy.deepcopy(train_args_conformer_no_freeze_asr_data)
    train_args_conformer_no_freeze_asr_data_no_spec_augment["net_args"]["spec_augment"] = False
    train_args_conformer_no_freeze_asr_data_no_spec_augment["network_module"] = "glowASR_conformer_no_freeze"
    run_exp(
        prefix_name + "conformer/asr_dataset/no_spec_augment/glow_enc192_200ep_not_silence_preprocessed_not_freezed",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_no_freeze_asr_data_no_spec_augment,
        search_args=default_search_args_asr,
        large_gpu_training=False,
        num_epochs=250,
    )

    train_args_conformer_no_pretrained_asr_data = copy.deepcopy(train_args_conformer_no_freeze_asr_data)
    del train_args_conformer_no_pretrained_asr_data["config"]["preload_from_files"]
    train_args_conformer_no_pretrained_asr_data["network_module"] = "glowASR_conformer_no_pretrained"
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc192_not_pretrained",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_no_pretrained_asr_data,
        search_args=default_search_args_asr,
        large_gpu_training=False,
        num_epochs=250,
    )
    train_args_conformer_no_pretrained_asr_data_no_spec_augment = copy.deepcopy(
        train_args_conformer_no_pretrained_asr_data
    )
    train_args_conformer_no_pretrained_asr_data_no_spec_augment["net_args"]["spec_augment"] = False
    train_args_conformer_no_pretrained_asr_data_no_spec_augment[
        "network_module"
    ] = "glowASR_conformer_no_freeze"  # This is only because it takes the same file as "no_freeze" and we don't want to lose the hash of an already done experiment
    run_exp(
        prefix_name + "conformer/asr_dataset/no_spec_augment/glow_enc192_not_pretrained",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_no_pretrained_asr_data_no_spec_augment,
        search_args=default_search_args_asr,
        large_gpu_training=False,
        num_epochs=250,
    )

    train_args_conformer_no_pretrained_control = copy.deepcopy(train_args_conformer_no_pretrained_asr_data)
    train_args_conformer_no_pretrained_control["network_module"] = "glowASR_conformer"
    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc192_not_pretrained_control",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_no_pretrained_control,
        search_args=default_search_args_asr,
        large_gpu_training=False,
        num_epochs=250,
    )

    train_args_blstm_asr_no_freeze = copy.deepcopy(train_args_blstm_frame_stack_asr)
    TTS_train_job = TTS_experiments["glowTTS/enc192/200ep/long_cooldown/not_silence_preprocessed"]["train_job"]
    train_args_blstm_asr_no_freeze["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = TTS_train_job.out_checkpoints[TTS_train_job.returnn_config.get("num_epochs", 200)]
    train_args_blstm_asr_no_freeze["network_module"] = "glowASR_blstm_frame_stack_no_freeze"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/no_spec_augment/glow_enc192_200ep_not_freezed",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_asr_no_freeze,
        search_args=default_search_args_asr,
    )

    train_args_blstm_asr_no_freeze_spec_augment = copy.deepcopy(train_args_blstm_asr_no_freeze)
    train_args_blstm_asr_no_freeze_spec_augment["net_args"]["spec_augment"] = True
    train_args_blstm_asr_no_freeze_spec_augment["config"]["gradient_clip_norm"] = 10
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/spec_augment/glow_enc192_200ep_not_freezed",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_asr_no_freeze_spec_augment,
        search_args=default_search_args_asr,
    )
    
    train_args_blstm_asr_no_freeze_spec_augment_before = copy.deepcopy(train_args_blstm_asr_no_freeze_spec_augment)
    train_args_blstm_asr_no_freeze_spec_augment_before["network_module"] = "glowASR_blstm_frame_stack_no_freeze_specaug_before"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/spec_augment_before/glow_enc192_200ep_not_freezed",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_asr_no_freeze_spec_augment_before,
        search_args=default_search_args_asr,
    )

    train_args_blstm_asr_spec_augment_before = copy.deepcopy(train_args_blstm_asr_no_freeze_spec_augment)
    train_args_blstm_asr_spec_augment_before["network_module"] = "glowASR_blstm_frame_stack"
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/spec_augment_before/glow_enc192_200ep",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_asr_spec_augment_before,
        search_args=default_search_args_asr,
    )

    train_args_blstm_asr_no_pretrained_spec_augment = copy.deepcopy(train_args_blstm_asr_no_freeze_spec_augment)
    train_args_blstm_asr_no_pretrained_spec_augment["network_module"] = "glowASR_blstm_frame_stack_no_pretrained"
    del train_args_blstm_asr_no_pretrained_spec_augment["config"]["preload_from_files"]
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/spec_augment/glow_enc192_200ep_not_pretrained",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_asr_no_pretrained_spec_augment,
        search_args=default_search_args_asr,
    )

    train_args_blstm_asr_no_pretrained_no_spec_augment = copy.deepcopy(train_args_blstm_asr_no_pretrained_spec_augment)
    train_args_blstm_asr_no_pretrained_no_spec_augment["net_args"]["spec_augment"] = False
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/no_spec_augment/glow_enc192_200ep_not_pretrained",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_asr_no_pretrained_no_spec_augment,
        search_args=default_search_args_asr,
    )
    
    for lm_w in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        for ps in [0, 0.3, 0.5]:
            additional_search_args = {"lm_weight": lm_w, "prior_scale": ps}
            run_exp(
                prefix_name + f"conformer/asr_dataset/spec_augment/glow_enc192_200ep_not_silence_preprocessed_not_freezed/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_no_freeze_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=False,
                num_epochs=250,
                with_prior=True
            )
            run_exp(
                prefix_name + f"conformer/asr_dataset/no_spec_augment/glow_enc192_200ep_not_silence_preprocessed_not_freezed/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_no_freeze_asr_data_no_spec_augment,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=False,
                num_epochs=250,
                with_prior=True
            )
            run_exp(
                prefix_name + f"conformer/asr_dataset/spec_augment/glow_enc192_not_pretrained/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_no_pretrained_asr_data,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=False,
                num_epochs=250,
                with_prior=True
            )
            run_exp(
                prefix_name + f"conformer/asr_dataset/no_spec_augment/glow_enc192_not_pretrained/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_no_pretrained_asr_data_no_spec_augment,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=False,
                num_epochs=250,
                with_prior=True
            )
            run_exp(
                prefix_name + f"conformer/asr_dataset/spec_augment/glow_enc192_not_pretrained_control/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_no_pretrained_control,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=False,
                num_epochs=250,
                with_prior=True
            )
            run_exp(
                prefix_name + f"blstm_2x512_d0.2_b300_fs4/asr_dataset/no_spec_augment/glow_enc192_not_pretrained/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_blstm_asr_no_pretrained_no_spec_augment,
                search_args={**default_search_args_asr, **additional_search_args},
                with_prior=True
            )
            run_exp(
                prefix_name + f"blstm_2x512_d0.2_b300_fs4/asr_dataset/spec_augment/glow_enc192_not_pretrained/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_blstm_asr_no_pretrained_spec_augment,
                search_args={**default_search_args_asr, **additional_search_args},
                with_prior=True
            )
            run_exp(
                prefix_name + f"blstm_2x512_d0.2_b300_fs4/asr_dataset/spec_augment/glow_enc192_200ep_not_freezed/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_blstm_asr_no_freeze_spec_augment,
                search_args={**default_search_args_asr, **additional_search_args},
                with_prior=True
            )
            run_exp(
                prefix_name + f"blstm_2x512_d0.2_b300_fs4/asr_dataset/no_spec_augment/glow_enc192_200ep_not_freezed/lm_tuning/lm{lm_w}_ps{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_blstm_asr_no_freeze,
                search_args={**default_search_args_asr, **additional_search_args},
                with_prior=True
            )       
    #  ------------------------------ X-Vectors ------------------------------
    train_args_conformer_asr_data_xvector = copy.deepcopy(train_args_conformer_asr_data)
    train_args_conformer_asr_data_xvector["network_module"] = "glowASR_conformer_x_vector"
    train_args_conformer_asr_data_xvector["net_args"]["gin_channels"] = 512
    train_args_conformer_asr_data_xvector["net_args"]["n_speakers"] = 251
    glowTTS_xvector_train_job = TTS_experiments["glowTTS_x_vector/enc768/100ep/not_silence_preprocessed"]["train_job"]
    train_args_conformer_asr_data_xvector["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = glowTTS_xvector_train_job.out_checkpoints[glowTTS_xvector_train_job.returnn_config.get("num_epochs", 100)]

    run_exp(
        prefix_name + "conformer/asr_dataset/spec_augment/glow_enc768_100ep_xvector",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_asr_data_xvector,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )

    for lm_w in [2.5, 3.0, 3.5, 4.0]:
        for ps in [0, 0.3, 0.5]:
            additional_search_args = {"lm_weight": lm_w, "prior_scale": ps}
            run_exp(
                prefix_name
                + f"conformer/asr_dataset/spec_augment/glow_enc768_100ep_xvector/search_params/lm_{lm_w}_ps_{ps}",
                datasets=train_data_normal_ctc,
                train_args=train_args_conformer_asr_data_xvector,
                search_args={**default_search_args_asr, **additional_search_args},
                large_gpu_training=True,
                num_epochs=250,
                with_prior=True,
            )

    train_args_blstm_asr_xvector = copy.deepcopy(train_args_blstm_frame_stack_asr)
    train_args_blstm_asr_xvector["network_module"] = "glowASR_blstm_frame_stack_x_vector"
    train_args_blstm_asr_xvector["net_args"]["gin_channels"] = 512
    train_args_blstm_asr_xvector["net_args"]["n_speakers"] = 251
    train_args_blstm_asr_xvector["config"]["preload_from_files"]["existing-model"][
        "filename"
    ] = glowTTS_xvector_train_job.out_checkpoints[glowTTS_xvector_train_job.returnn_config.get("num_epochs", 100)]
    run_exp(
        prefix_name + "blstm_2x512_d0.2_b300_fs4/asr_dataset/spec_augment/glow_enc768_100ep_xvector",
        datasets=train_data_normal_ctc,
        train_args=train_args_blstm_asr_xvector,
        search_args=default_search_args_asr,
    )

    # ============================== Weaker Conformer Baseline ==============================
    train_args_conformer_baseline = copy.deepcopy(train_args_conformer_only_asr_data)

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=50,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=3,
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
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank_normal_ctc,
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
    )

    train_args_conformer_baseline["net_args"]["conformer_model_config"] = asdict(model_config)
    train_args_conformer_baseline["config"]["eval_datasets"] = {
        "devtrain": train_data_normal_ctc.devtrain.as_returnn_opts()
    }
    del train_args_conformer_baseline["config"]["preload_from_files"]
    run_exp(
        prefix_name + "conformer/asr_dataset/weak_baseline",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_baseline,
        search_args=default_search_args_asr,
        large_gpu_training=True,
        num_epochs=250,
    )

    for lm_w in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        run_exp(
            prefix_name + f"conformer/asr_dataset/weak_baseline/lm_tuning/lm{lm_w}",
            datasets=train_data_normal_ctc,
            train_args=train_args_conformer_baseline,
            search_args={**default_search_args_asr, **{"lm_weight": lm_w}},
            large_gpu_training=True,
            num_epochs=250,
        )

    run_exp(
        prefix_name + f"conformer/asr_dataset/weak_baseline/lm_tuning/lm2.5",
        datasets=train_data_normal_ctc,
        train_args=train_args_conformer_baseline,
        search_args={**default_search_args_asr, **{"lm_weight": 2.5}},
        large_gpu_training=True,
        num_epochs=250,
        test_datasets=dev2_dataset_normal_ctc_tuples
    )
