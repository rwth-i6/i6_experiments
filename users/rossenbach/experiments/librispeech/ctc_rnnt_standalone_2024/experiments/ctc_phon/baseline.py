from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search



def eow_phon_ls960_1023_base():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )
    train_settings_part20 = copy.deepcopy(train_settings)
    train_settings_part20.train_partition_epoch = 20

    train_settings_part5 = copy.deepcopy(train_settings)
    train_settings_part5.train_partition_epoch = 5

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    train_data_part20 = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings_part20,
    )
    train_data_part5 = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings_part5,
    )

    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    def tune_and_evaluate_helper(training_name, asr_model, base_decoder_config, lm_scales, prior_scales, decoder_module="ctc.decoder.flashlight_ctc_v1"):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name, forward_config={}, asr_model=asr_model, decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)}, test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn
            )


    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

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
        max_dim_feat=8,  # Jingjing style
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

    model_config = ModelConfig(
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
        specauc_start_epoch=1,
    )

    
    learning_rates_default_500 = list(np.linspace(7e-6, 5e-4, 240)) + list(
        np.linspace(5e-4, 5e-5, 240)) + list(np.linspace(5e-5, 1e-7, 20))

    learning_rates_default_250 = list(np.linspace(7e-6, 5e-4, 120)) + list(
        np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10))
    
    
    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 240)) + list(
            np.linspace(5e-4, 5e-5, 240)) + list(np.linspace(5e-5, 1e-7, 20)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".512dim_sub4_24gbgpu_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])

    asr_model_best4 = prepare_asr_model(
        training_name+ "/best4", train_job, train_args, with_prior=True, datasets=train_data, get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    )
    tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])
    
    
    # Conv first + Quantization
    from ...pipeline import QuantArgs
    from ...pytorch_networks.ctc.conformer_1023.quant.baseline_quant_v1_cfg import QuantModelConfigV1
    model_config_quant_v1 = QuantModelConfigV1(
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor",
        moving_average=0.01,
        weight_bit_prec=8,
        activation_bit_prec=8,
        linear_quant_output=True,
    )

    for num_samples in [10, 100, 1000, 10000]:
        for i in range(10):
            q_args_test = QuantArgs(
                quant_config_dict={
                    "quant_model_config_dict": asdict(model_config_quant_v1)
                },
                num_samples=num_samples,
                seed=i,
                datasets=train_data,
                network_module="ctc.conformer_1023.quant.baseline_quant_v2",
                filter_args=None
            )
            quant_asr_model = prepare_asr_model(
                training_name + "_test_quant_num%i_run%i" % (num_samples, i), train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500, quant_args=q_args_test
            )
            # tune_and_evaluate_helper(training_name + "_test_quant", quant_asr_model, default_decoder_config, lm_scales=[2.5],
            #                         prior_scales=[0.3], decoder_module="ctc.decoder.flashlight_quant_stat_phoneme_ctc")
            decoder_config = copy.deepcopy(default_decoder_config)
            decoder_config.lm_weight = 2.5
            decoder_config.prior_scale = 0.3
            search_name = training_name + "_test_quant_num%i_run%i/search_lm%.1f_prior%.1f" % (num_samples, i, 2.5, 0.3)
            search_jobs, wers = search(
                search_name,
                forward_config={},
                asr_model=quant_asr_model,
                decoder_module="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                **default_returnn
            )


    # Compile degrades speed

     # decoder_config_with_compile = copy.deepcopy(default_decoder_config)
     # decoder_config_with_compile.use_torch_compile = True
     # decoder_config_with_compile.torch_compile_options = {"dynamic": True}
     # tune_and_evaluate_helper(training_name + "/dynamic_compile_test", asr_model, decoder_config_with_compile, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])

    # decoder_config = copy.deepcopy(debug_decoder_config)
    # decoder_config.use_torch_compile = True
    # decoder_config.torch_compile_options = {"dynamic": True, "fullgraph": True}
    # search_name = training_name + "/dynamic_compile_test_fullgraph"
    # search_jobs, wers = search(
    #     search_name, forward_config={}, asr_model=asr_model,
    #     decoder_module="ctc.decoder.flashlight_ctc_v1",
    #     decoder_args={"config": asdict(decoder_config)},
    #     test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
    #     **default_returnn
    # )

    debug_decoder_config = copy.deepcopy(default_decoder_config)
    debug_decoder_config.lm_weight = 2.3
    debug_decoder_config.prior_scale = 0.3


    decoder_config = copy.deepcopy(debug_decoder_config)
    search_name = training_name + "/search_onnx_test"
    search_jobs, wers = search(
        search_name,
        forward_config={},
        asr_model=asr_model,
        decoder_module="ctc.decoder.flashlight_ctc_v1_onnx",
        decoder_args={"config": asdict(decoder_config)},
        test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
        **default_returnn,
        debug=True,
    )


    # extra dropout test
    model_config_extra_dropout = copy.deepcopy(model_config)
    model_config_extra_dropout.ff_dropout = 0.2
    model_config_extra_dropout.conv_dropout = 0.2
    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args_extra_dropout = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config_extra_dropout)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".512dim_sub4_24gbgpu_50eps_extra_dropout"
    train_job = training(training_name, train_data, train_args_extra_dropout, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_extra_dropout, with_prior=True, datasets=train_data, get_specific_checkpoint=500
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])


    # Conv first test
    network_module_conv_first = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
    train_args_conv_first = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module_conv_first,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module_conv_first + ".512dim_sub4_24gbgpu_50eps"
    train_job = training(training_name, train_data, train_args_conv_first, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_conv_first, with_prior=True, datasets=train_data, get_specific_checkpoint=500
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7],
                             prior_scales=[0.2, 0.3, 0.4])






    # Conv first test + extra dropout
    network_module_conv_first = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
    train_args_conv_first_extra_dropout = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module_conv_first,
        "net_args": {"model_config_dict": asdict(model_config_extra_dropout)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module_conv_first + ".512dim_sub4_24gbgpu_50eps_extra_dropout"
    train_job = training(training_name, train_data, train_args_conv_first_extra_dropout, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_conv_first_extra_dropout, with_prior=True, datasets=train_data, get_specific_checkpoint=500
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7],
                             prior_scales=[0.2, 0.3, 0.4])




    # speed perturbation
    training_name = prefix_name + "/" + network_module + ".512dim_sub4_24gbgpu_50eps_speed"
    train_args_speed_pert = copy.deepcopy(train_args)
    train_args_speed_pert["use_speed_perturbation"] = True
    train_args_speed_pert["config"]["gradient_clip"] = 1.0
    train_job = training(training_name, train_data, train_args_speed_pert, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])

    # efficiency tune
    training_name = prefix_name + "/" + network_module + ".512dim_sub4_24gbgpu_eff_tune"
    train_args_eff_tune = copy.deepcopy(train_args)
    train_args_eff_tune["config"]["batch_size"] = 720 * 16000
    train_args_eff_tune["config"]["learning_rates"] = list(np.linspace(7e-6, 5e-4, 270)) + list(
            np.linspace(5e-4, 5e-5, 270)) + list(np.linspace(5e-5, 1e-7, 20))
    train_job = training(training_name, train_data, train_args_eff_tune, num_epochs=560, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_eff_tune, with_prior=True, datasets=train_data, get_specific_checkpoint=560
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])



    # training with 250 subepochs
    train_args_250eps = copy.deepcopy(train_args)
    train_args_250eps["config"]["learning_rates"] = learning_rates_default_250
    training_name = prefix_name + "/" + network_module + ".512dim_sub4_24gbgpu_50eps_part5"
    train_job = training(training_name, train_data_part5, train_args_250eps, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )

    # No improvement, just as example
    # asr_model_best4 = prepare_asr_model(
    #     training_name+ "/best4", train_job, train_args, with_prior=True, datasets=train_data, get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    # )
    # tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])

    train_config_11gbgpu = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": learning_rates_default_250,
        #############
        "batch_size": 180 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
    }
    train_args = {
        "config": train_config_11gbgpu,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    # first with 250 eps and split 10
    training_name = prefix_name + "/" + network_module + ".512dim_sub4_11gbgpu_25eps"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )

    # first with 250 eps and split 10
    train_args_schedule_test = copy.deepcopy(train_args)
    train_args_schedule_test["config"]["learning_rates"] = learning_rates_default_500[::2]
    training_name = prefix_name + "/" + network_module + ".512dim_sub4_11gbgpu_25eps_schedule_test"
    train_job = training(training_name, train_data, train_args_schedule_test, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )

    # and once with 500 eps and split 20
    train_args = copy.deepcopy(train_args)
    train_args["config"]["learning_rates"] = learning_rates_default_500
    training_name = prefix_name + "/" + network_module + ".512dim_sub4_11gbgpu_25eps_part20"
    train_job = training(training_name, train_data_part20, train_args, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )

    # run the "better" 250/10 also with jingjings 15k batch size
    train_args_batch15k = copy.deepcopy(train_args)
    train_args_batch15k["config"]["batch_size"] = 150 * 16000
    training_name = prefix_name + "/" + network_module + ".512dim_sub4_11gbgpu_25eps_batch15k"
    train_job = training(training_name, train_data, train_args_batch15k, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )
    
    
    # Faster Convergence RAdam style
    train_config_11gbgpu = {
        "optimizer": {"class": "radam", "epsilon": 1e-16, "decoupled_weight_decay": True, "weight_decay": 1e-3},
        "learning_rates": learning_rates_default_250,
        #############
        "batch_size": 180 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
    }
    train_args = {
        "config": train_config_11gbgpu,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    
    training_name = prefix_name + "/" + network_module + ".512dim_sub4_11gbgpu_25eps_radamv1"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )


    train_args_update_lr_for_adam = copy.deepcopy(train_args)
    train_args_update_lr_for_adam["config"]["learning_rates"] = list(np.linspace(1e-4, 5e-4, 40)) + list(
        np.linspace(5e-4, 5e-5, 200)) + list(np.linspace(5e-5, 1e-7, 10))

    training_name = prefix_name + "/" + network_module + ".512dim_sub4_11gbgpu_25eps_radamv1_newlr"
    train_job = training(training_name, train_data, train_args_update_lr_for_adam, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_update_lr_for_adam, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )
    
    train_args_update_lr_for_adam = copy.deepcopy(train_args)
    train_args_update_lr_for_adam["config"]["learning_rates"] = list(np.linspace(1e-4, 5e-4, 40)) + list(
        np.linspace(5e-4, 5e-4, 80)) + list(np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10))

    training_name = prefix_name + "/" + network_module + ".512dim_sub4_11gbgpu_25eps_radamv1_newlrv2"
    train_job = training(training_name, train_data, train_args_update_lr_for_adam, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_update_lr_for_adam, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )
    
    train_args_update_lr_for_adam = copy.deepcopy(train_args)
    train_args_update_lr_for_adam["config"]["learning_rates"] = list(np.linspace(1e-4, 5e-4, 40)) + list(
        np.linspace(5e-4, 5e-4, 80)) + list(np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10))

    training_name = prefix_name + "/" + network_module + ".512dim_sub4_11gbgpu_25eps_radamv1_newlrv2"
    train_job = training(training_name, train_data, train_args_update_lr_for_adam, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 11
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_update_lr_for_adam, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )
