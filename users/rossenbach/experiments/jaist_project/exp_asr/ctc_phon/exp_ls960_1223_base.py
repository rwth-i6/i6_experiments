from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.phon import build_eow_phon_training_datasets, TrainingDatasetSettings, get_text_lexicon
from ...data.common import build_test_dataset
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm

from ...pipeline import search, ctc_training

from ...config import get_forward_config

from ...storage import add_asr_recognizer, ASRRecognizerSystem

from i6_experiments.users.rossenbach.tools.parameter_tuning import PickOptimalParametersJob


def eow_phon_ls960_1023_base():
    prefix_name = "experiments/jaist_project/asr/ls960_ctc_eow_phon/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=10,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    # build testing datasets
    dev_dataset_tuples = {}
    # for testset in ["dev", "test"]:
    for testset in ["dev-clean", "dev-other"]:
            dev_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                preemphasis=train_settings.preemphasis,
                peak_normalization=train_settings.peak_normalization,
            )
        
    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                preemphasis=train_settings.preemphasis,
                peak_normalization=train_settings.peak_normalization,
            )

    arpa_4gram_lm = get_4gram_binary_lm()

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp_search(ft_name, train_args, checkpoint, search_args=None, prior_file=None, decoder="ctc.decoder.flashlight_phoneme_ctc", store_system_with_name=None):
        search_args = search_args if search_args is not None else {}
        if prior_file is not None:
            search_args["prior_file"] = prior_file

        returnn_search_config = get_forward_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)

        format_string_report, values_report, search_jobs, wers = search(ft_name, returnn_search_config,
                          checkpoint, dev_dataset_tuples, RETURNN_EXE,
                           MINI_RETURNN_ROOT, return_wers=True)

        if store_system_with_name is not None:
            system = ASRRecognizerSystem(
                config=returnn_search_config,
                checkpoint=checkpoint,
                preemphasis=0.97,
                peak_normalization=True
            )
            add_asr_recognizer(name=store_system_with_name, system=system)

        return search_jobs, wers


    def dedicated_search(ft_name, dataset_key, checkpoint, train_args, search_args, decoder="ctc.decoder.flashlight_phoneme_ctc"):
        returnn_search_config = get_forward_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)
        dataset_tuples = {dataset_key: test_dataset_tuples[dataset_key]}
        ret_vals = search(ft_name + "/" + dataset_key, returnn_search_config,
                                   checkpoint, dataset_tuples, RETURNN_EXE,
                                   MINI_RETURNN_ROOT, with_confidence=True)
    
    
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
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
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
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
        specauc_start_epoch=1,
    )

    train_args_adamw03_accum2_jjlr = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
                np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 360 * 16000,  # no grad accum needed within Kagayaki
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
        },
        "debug": False,
    }

    default_search_args = {
        "lexicon": get_text_lexicon(librispeech_key="train-other-960"),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 1024,
        "beam_size_token": 128,
        "arpa_lm": arpa_4gram_lm,
        "beam_threshold": 14,
    }

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config)},
    }
    # diverged with hiccup
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #         }
    #         run_exp(
    #             prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR/lm%.1f_prior%.2f_bs1024_th14" % (
    #                 lm_weight, prior_scale),
    #             datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
            
    train_args_gc1 = copy.deepcopy(train_args)
    train_args_gc1["config"]["gradient_clip"] = 1.0
    train_args_gc1["config"]["torch_amp_options"] = {"dtype": "bfloat16"}
    name = prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16"
    train, checkpoint, prior_file = ctc_training(
        training_name=name,
        datasets=train_data,
        train_args=train_args_gc1,
        with_prior=True,
        num_epochs=250,
    )
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _ , wers = run_exp_search(
                ft_name = name + "_last250/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                train_args=train_args_gc1,
                checkpoint=checkpoint,
                search_args=search_args,
                prior_file=prior_file
            )

    frontend_config_large = VGG4LayerActFrontendV1Config_mod(
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
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )

    model_config_large = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_large,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config_large)},
    }
    train_args_gc1 = copy.deepcopy(train_args)
    train_args_gc1["config"]["gradient_clip"] = 1.0
    train_args_gc1["config"]["torch_amp_options"] = {"dtype": "bfloat16"}
    train_args_gc1["config"]["learning_rates"] = list(np.linspace(7e-6, 5e-4, 120)) + list(
                np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-8, 10))
    name = prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_large_LRv2_peaknorm_gc1_amp16"
    train, checkpoint, prior_file = ctc_training(
        training_name=name,
        datasets=train_data,
        train_args=train_args_gc1,
        with_prior=True,
        num_epochs=250,
    )
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _ , wers = run_exp_search(
                ft_name = name + "_last250/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                train_args=train_args_gc1,
                checkpoint=checkpoint,
                search_args=search_args,
                prior_file=prior_file
            )
            
    for lm_weight in [1.8, 2.0, 2.2]:
        for prior_scale in [0.2, 0.3, 0.4]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _ , wers = run_exp_search(
                ft_name = name + "_last250/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                train_args=train_args_gc1,
                checkpoint=checkpoint,
                search_args=search_args,
                prior_file=prior_file
            )
    
    specaug_config_half = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,
        num_repeat_feat=5,
    )
    model_config_large_halfspec = copy.deepcopy(model_config_large)
    model_config_large_halfspec.specaug_config = specaug_config_half
    train_args_gc1_50eps = copy.deepcopy(train_args_gc1)
    train_args_gc1_50eps["net_args"] = {"model_config_dict": asdict(model_config_large_halfspec)}
    train_args_gc1_50eps["config"]["learning_rates"] = list(np.linspace(7e-6, 5e-4, 240)) + list(
                np.linspace(5e-4, 5e-5, 240)) + list(np.linspace(5e-5, 1e-7, 20))
    train_args_gc1_50eps["post_config"] = {"cleanup_old_models": {'keep_last_n': 10}}
    
    name = prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_large_LRv2_peaknorm_gc1_amp16"
    train, checkpoint, prior_file = ctc_training(
        training_name=name,
        datasets=train_data,
        train_args=train_args_gc1_50eps,
        with_prior=True,
        num_epochs=500,
    )
    
    _, checkpoint_avg10, prior_file_avg10 = ctc_training(
        training_name=name,
        datasets=train_data,
        train_args=train_args_gc1_50eps,
        with_prior=True,
        num_epochs=500,
        average_checkpoints=10,
    )

    tune_parameters = []
    tune_values_clean = []
    tune_values_other = []
    for lm_weight in [2.0, 2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            _ , wers = run_exp_search(
                ft_name = name + "_last500/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                train_args=train_args_gc1_50eps,
                checkpoint=checkpoint,
                search_args=search_args,
                prior_file=prior_file
            )
            tune_parameters.append((lm_weight, prior_scale))
            tune_values_clean.append((wers["dev-clean"]))
            tune_values_other.append((wers["dev-other"]))
            if lm_weight == 2.5 and prior_scale == 0.3:
                _, wers = run_exp_search(
                    ft_name=name + "_average10/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                    train_args=train_args_gc1_50eps,
                    checkpoint=checkpoint_avg10,
                    search_args=search_args,
                    prior_file=prior_file_avg10,
                )

                search_args_fast_v1 = copy.deepcopy(search_args)
                search_args_fast_v1["beam_size_token"] = 20
                _, wers = run_exp_search(
                    ft_name=name + "_last500/lm%.1f_prior%.2f_bs1024_th14_fastsearch_v1" % (lm_weight, prior_scale),
                    train_args=train_args_gc1_50eps,
                    checkpoint=checkpoint,
                    search_args=search_args_fast_v1,
                    prior_file=prior_file,
                    store_system_with_name="ls960eow_phon_ctc_50eps_fastsearch",
                )
    for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
        pick_optimal_params_job = PickOptimalParametersJob(parameters=tune_parameters, values=tune_values)
        pick_optimal_params_job.add_alias(
            prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_large_LRv2_50epsJJ_halfspec_amp16/pick_best_{key}")
        search_args = copy.deepcopy(default_search_args)
        search_args["lm_weight"] = pick_optimal_params_job.optimal_parameters[0]
        search_args["prior_scale"] = pick_optimal_params_job.optimal_parameters[1]
        search_args["prior_file"] = prior_file
        dedicated_search(
            ft_name=prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_large_LRv2_50epsJJ_halfspec_amp16",
            dataset_key=key,
            checkpoint=train.out_checkpoints[500],
            train_args=train_args_gc1_50eps,
            search_args=search_args
        )

    search_args_lowlm = {
        **default_search_args,
        "lm_weight": 0.01,
        "beam_size_token": 2, # pick only 2 best candidates
        "beam_size": 64,
        "prior_scale": 0.0,
    }
    _, wers = run_exp_search(
        ft_name=name + "_last500/lowlm_pick2",
        train_args=train_args_gc1_50eps,
        checkpoint=checkpoint,
        search_args=search_args_lowlm,
        prior_file=None,
    )

    for blank_log_penalty in [0.1, 0.2, 0.3]:
        search_args_lowlm_prior = copy.deepcopy(search_args_lowlm)
        search_args_lowlm_prior["blank_log_penalty"] = blank_log_penalty
        _, wers = run_exp_search(
            ft_name=name + "_last500/lowlm_pick2_epspen%.1f" % blank_log_penalty,
            train_args=train_args_gc1_50eps,
            checkpoint=checkpoint,
            search_args=search_args_lowlm_prior,
            prior_file=None,
        )
        
    search_args_narrow_beam_normal_lm = {
        **default_search_args,
        "lm_weight": 2.5,
        "beam_size_token": 2, # pick only 2 best candidates
        "beam_size": 64,
        "prior_scale": 0.3,
    }
    _, wers = run_exp_search(
        ft_name=name + "_last500/narrow_beam_normal_lm",
        train_args=train_args_gc1_50eps,
        checkpoint=checkpoint,
        search_args=search_args_narrow_beam_normal_lm,
        prior_file=None,
    )

    search_args_normal_beam_no_lm = {
        **default_search_args,
        "lm_weight": 0.01,
        "prior_scale": 0.3
    }
    _, wers = run_exp_search(
        ft_name=name + "_last500/normal_beam_no_lm",
        train_args=train_args_gc1_50eps,
        checkpoint=checkpoint,
        search_args=search_args_normal_beam_no_lm,
        prior_file=None,
    )