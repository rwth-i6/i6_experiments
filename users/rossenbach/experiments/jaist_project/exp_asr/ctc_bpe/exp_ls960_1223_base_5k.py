from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast


from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.experiments.jaist_project.lm import get_4gram_binary_lm
from i6_experiments.users.rossenbach.experiments.jaist_project.data.bpe import build_bpe_training_datasets, TrainingDatasetSettings, get_text_lexicon
from i6_experiments.users.rossenbach.experiments.jaist_project.data.common import build_test_dataset
from i6_experiments.users.rossenbach.experiments.jaist_project.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, KENLM_BINARY_PATH

from i6_experiments.users.rossenbach.experiments.jaist_project.pipeline import training, search, compute_prior

from i6_experiments.users.rossenbach.experiments.jaist_project.config import get_training_config, get_forward_config, get_prior_config
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import add_ctc_model

from i6_experiments.users.rossenbach.tools.parameter_tuning import PickOptimalParametersJob


def conformer_baseline_5k():
    prefix_name = "experiments/jaist_project/asr/ls960_ctc_bpe/"

    BPE_SIZE = 5000

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=10,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_bpe_training_datasets(
        librispeech_key="train-other-960",
        bpe_size=BPE_SIZE,
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

    def run_exp(ft_name, datasets, train_args, search_args=None, with_prior=False, num_epochs=250, decoder="ctc.decoder.flashlight_bpe_ctc", return_wers_and_model=False):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if with_prior:
            returnn_config = get_prior_config(training_datasets=datasets, **train_args)
            prior_file = compute_prior(
                ft_name,
                returnn_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(training_name + "/prior.txt", prior_file)
            search_args["prior_file"] = prior_file

        returnn_search_config = get_forward_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)

        ret_vals = search(ft_name + "/last_%i" % num_epochs, returnn_search_config,
                                   train_job.out_checkpoints[num_epochs], dev_dataset_tuples, RETURNN_EXE,
                                   MINI_RETURNN_ROOT, return_wers=return_wers_and_model)
        
        if return_wers_and_model:
            return train_job, ret_vals[2], ret_vals[3], prior_file

        return train_job, ret_vals[2]
    
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
        out_features=512,
        activation=None,
    )
    model_config_start11 = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
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
        specauc_start_epoch=11,
    )

    train_args_adamw03_jjlr = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
                np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 2 * 16000,  # no grad accum needed for JAIST Kagayaki
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
        },
        "debug": False,
    }

    default_search_args = {
        "lexicon": get_text_lexicon(librispeech_key="train-other-960", bpe_size=BPE_SIZE),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 1024,
        "beam_size_token": 128,
        "arpa_lm": arpa_4gram_lm,
        "beam_threshold": 14,
    }

    train_args = {
        **copy.deepcopy(train_args_adamw03_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config_start11)},
    }
    train_args["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_1223_5k/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_start11_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)


    train_args = {
        **copy.deepcopy(train_args_adamw03_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config_start11)},
    }
    train_args["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 5e-4, 120)) + list(
                np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-8, 10))
    train_args["config"]["gradient_clip"] = 1.0
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job, _ = run_exp(
                prefix_name + "conformer_1223_5k/i6modelsV1_VGG4LayerActFrontendV1_v6_LRv2_preaknorm_start11_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    add_ctc_model("bpe5k_i6modelsLV1_LRv2", train_job.out_checkpoints[250])
            
    for lm_weight in [1.2, 1.4, 1.6]:
        for prior_scale in [0.1, 0.2, 0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_1223_5k/i6modelsV1_VGG4LayerActFrontendV1_v6_LRv2_preaknorm_start11_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
    
    
    
    # ------------------------------------
    # Subsampling 6
    # ------------------------------------
    
    specaug_half_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,
        num_repeat_feat=5,
    )
    frontend_config_sub6 = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )
    model_config_sub6_halfspec_start11 = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        specaug_config=specaug_half_config,
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
        specauc_start_epoch=11,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config_sub6_halfspec_start11)},
    }
    train_args["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 5e-4, 120)) + list(
                np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-8, 10))
    train_args["config"]["gradient_clip"] = 1.0
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job, _ = run_exp(
                prefix_name + "conformer_1223_5k/i6modelsV1_VGG4LayerActFrontendV1_v6_sub6_LRv2_halfspec_start11_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
            
    train_args = {
        **copy.deepcopy(train_args_adamw03_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config_sub6_halfspec_start11)},
    }
    train_args_gc1_50eps = copy.deepcopy(train_args)
    train_args_gc1_50eps["net_args"] = {"model_config_dict": asdict(model_config_sub6_halfspec_start11)}
    train_args_gc1_50eps["config"]["learning_rates"] = list(np.linspace(7e-6, 5e-4, 240)) + list(
                np.linspace(5e-4, 5e-5, 240)) + list(np.linspace(5e-5, 1e-7, 20))
    train_args_gc1_50eps["post_config"] = {"cleanup_old_models": {'keep_last_n': 10}}
    train_args_gc1_50eps["config"]["gradient_clip"] = 1.0

    tune_parameters = []
    tune_values_clean = []
    tune_values_other = []
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job, _, wers, prior_file = run_exp(
                prefix_name + "conformer_1223_5k/i6modelsV1_VGG4LayerActFrontendV1_v6_sub6_LRv2_halfspec_start11_50eps_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_gc1_50eps, search_args=search_args, with_prior=True, num_epochs=500, return_wers_and_model=True)
            tune_parameters.append((lm_weight, prior_scale))
            tune_values_clean.append((wers["dev-clean"]))
            tune_values_other.append((wers["dev-other"]))
    for lm_weight in [1.4, 1.6, 1.8]:
        for prior_scale in [0.2, 0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job, _, wers, prior_file = run_exp(
                prefix_name + "conformer_1223_5k/i6modelsV1_VGG4LayerActFrontendV1_v6_sub6_LRv2_halfspec_start11_50eps_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_gc1_50eps, search_args=search_args, with_prior=True, num_epochs=500, return_wers_and_model=True)
            tune_parameters.append((lm_weight, prior_scale))
            tune_values_clean.append((wers["dev-clean"]))
            tune_values_other.append((wers["dev-other"]))
            
    for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
        pick_optimal_params_job = PickOptimalParametersJob(parameters=tune_parameters, values=tune_values)
        pick_optimal_params_job.add_alias(
            prefix_name + f"conformer_1223_5k/i6modelsV1_VGG4LayerActFrontendV1_v6_sub6_LRv2_halfspec_start11_50eps_amp16/pick_best_{key}")
        search_args = copy.deepcopy(default_search_args)
        search_args["lm_weight"] = pick_optimal_params_job.optimal_parameters[0]
        search_args["prior_scale"] = pick_optimal_params_job.optimal_parameters[1]
        search_args["prior_file"] = prior_file
        dedicated_search(
            ft_name=prefix_name + f"conformer_1223_5k/i6modelsV1_VGG4LayerActFrontendV1_v6_sub6_LRv2_halfspec_start11_50eps_amp16",
            dataset_key=key,
            checkpoint=train_job.out_checkpoints[500],
            train_args=train_args_gc1_50eps,
            search_args=search_args
        )

    add_ctc_model("bpe5k_i6modelsLV1_LRv2_sub6_ep50", train_job.out_checkpoints[500])
