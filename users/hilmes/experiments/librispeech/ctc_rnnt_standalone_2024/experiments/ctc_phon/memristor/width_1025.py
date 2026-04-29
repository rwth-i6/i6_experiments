from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast
import os
from functools import partial

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....data.common import DatasetSettings, build_test_dataset
from ....data.phon import build_eow_phon_training_datasets, get_bliss_phoneme_lexicon
from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....lm import get_arpa_lm_config
from ....pipeline import training
from ....report import generate_report, build_qat_report, build_qat_report_v2

from ..tune_eval import eval_model


def eow_phon_ls960_1025_memristor_width():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon_memristor/width"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
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

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ....pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
    from ....rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bliss_phoneme_lexicon(),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=2048,
        score_threshold=18.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("4gram", lexicon_file=get_bliss_phoneme_lexicon(), scale=0.0),
    )

    as_training_rasr_config = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,  # this will be overwritten internally
        prior_file=None,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )
    rasr_config_memristor = copy.deepcopy(as_training_rasr_config)
    rasr_config_memristor.turn_off_quant = False

    rasr_prior_scales = [0.2, 0.3, 0.4, 0.5]
    rasr_lm_scales = [0.9, 1.0, 1.1]


    network_module_mem_v9 = "ctc.qat_0711.memristor_v9"
    network_module_mem_v10 = "ctc.qat_0711.memristor_v10"
    from ....pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8
    from torch_memristor.memristor_modules import DacAdcHardwareSettings
    from ....pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        LogMelFeatureExtractionV1Config,
        SpecaugConfig,
        ConformerPosEmbConfig,
        VGG4LayerActFrontendV1Config_mod
    )
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

    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Normal Style
        num_repeat_feat=5,
    )

    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=False,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )
    memristor_prior = 0.5
    memristor_lm = 1.0

    memristor_report = {}
    for activation_bit in [8, 6, 5, 4]:
        # 128, 256, 1280, 1792
        for dim in [384, 512, 768, 1024, 1536, 2048]:
            for num_layers in [10, 12, 14, 16]:
                for weight_bit in [4, 6, 8]:
                    # Diverged list format: act, weight, dim, seed
                    diverged_list = [
                        (4, 8, 512, 0), # Epoch 328, symmetric
                        (4, 8, 512, 1), # Epoch 408, symmetric
                        (4, 8, 1024, 0), # Epoch 135
                        (4, 6, 512, 0), # Epoch 128
                        (4, 6, 512, 1), # Epoch 282
                        (4, 6, 1024, 0), # Epoch 193
                        (4, 4, 512, 0), # Epoch 328
                        (4, 4, 512, 1), # Epoch 292
                        (4, 4, 1024, 0), # Epoch 128
                        (5, 8, 1024, 0) # Epoch 656, got initial convergence but diverged as LR increased
                    ]
                    if num_layers not in [12]:
                        if dim not in [512]:
                            continue
                        if activation_bit not in [8]:
                            continue
                        if weight_bit not in [4, 8]:
                            continue
                    if dim not in [512, 1024]:
                        if dim in [1024] and weight_bit < 6:
                            continue
                        if weight_bit not in [4, 8]:
                            continue
                        if activation_bit not in [8]:
                            continue
                    seeds = 2
                    if weight_bit not in [4] or activation_bit not in [8]:
                        if dim not in [512]:
                            seeds = 1
                    if num_layers not in [12]:
                        seeds = 1

                    frontend_config_dim = VGG4LayerActFrontendV1Config_mod(
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
                        out_features=dim,
                        activation=None,
                    )
                    prior_train_dac_settings = DacAdcHardwareSettings(
                        input_bits=0,
                        output_precision_bits=0,
                        output_range_bits=0,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )

                    quant_scheme = "per_tensor_symmetric"

                    model_config = MemristorModelTrainConfigV8(
                        feature_extraction_config=fe_config,
                        frontend_config=frontend_config_dim,
                        specaug_config=specaug_config_full,
                        label_target_size=vocab_size_without_blank,
                        conformer_size=dim,
                        num_layers=num_layers,
                        num_heads=8,
                        ff_dim=dim * 4,
                        att_weights_dropout=0.1,
                        conv_dropout=0.1,
                        ff_dropout=0.1,
                        mhsa_dropout=0.1,
                        conv_kernel_size=31,
                        final_dropout=0.1,
                        specauc_start_epoch=1,
                        weight_quant_dtype="qint8",
                        weight_quant_method=quant_scheme,
                        activation_quant_dtype="qint8",
                        activation_quant_method=quant_scheme,
                        dot_quant_dtype="qint8",
                        dot_quant_method=quant_scheme,
                        Av_quant_dtype="qint8",
                        Av_quant_method=quant_scheme,
                        moving_average=None,
                        weight_bit_prec=weight_bit,
                        activation_bit_prec=activation_bit,
                        quantize_output=False,
                        converter_hardware_settings=prior_train_dac_settings,
                        quant_in_linear=True,
                        num_cycles=0,
                        correction_settings=None,
                        weight_noise_func=None,
                        weight_noise_values=None,
                        weight_noise_start_epoch=None,
                        pos_emb_config=pos_emb_cfg,
                        module_list=["ff", "conv", "mhsa", "ff"],
                        module_scales=[0.5, 1.0, 1.0, 0.5],
                        aux_ctc_loss_layers=None,
                        aux_ctc_loss_scales=None,
                        dropout_broadcast_axes=None,
                    )
                    res_seeds_total = {}
                    for seed in range(seeds):
                        if (activation_bit, weight_bit, dim, seed) in diverged_list:
                            memristor_report[prefix_name + "/" + network_module_mem_v9 + f"_{1000//10}eps_{num_layers}lay_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"] = "Diverged"
                            continue

                        train_config_24gbgpu = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-12,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (1000 // 2 - 20)))
                                              + list(np.linspace(5e-4, 5e-5, (1000 // 2 - 20)))
                                              + list(np.linspace(5e-5, 1e-7, 40)),
                            #############
                            "batch_size": 360 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "gradient_clip_norm": 1.0,
                            "seed": seed,
                            "torch_amp_options": {"dtype": "bfloat16"},
                        }
                        train_args = {
                            "config": train_config_24gbgpu,
                            "network_module": network_module_mem_v9,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }
                        training_name = prefix_name + "/" + network_module_mem_v9 + f"_{1000//10}eps_{num_layers}lay_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                        train_job = training(training_name, train_data, train_args, num_epochs=1000, **default_returnn)
                        if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.rqmt['cpu'] = 12
                            train_job.hold()
                            train_job.move_to_hpc = True
                        # train_job.rqmt["gpu_mem"] = 24
                        # train_job.rqmt['mem'] = 36

                        results = {}
                        results, best_params_job = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data,
                            decoder_config=as_training_rasr_config,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            result_dict=results,
                            decoder_module="ctc.decoder.rasr_ctc_v1",
                            prior_scales=rasr_prior_scales,
                            lm_scales=rasr_lm_scales,
                            import_memristor=True,
                            get_best_params=True,
                            run_rasr=True,
                            run_best_4=False,
                            run_best=False,
                        )
                        generate_report(results=results, exp_name=training_name + "/non_memristor")
                        memristor_report[training_name] = results

                        res_conv = {}
                        recog_dac_settings = DacAdcHardwareSettings(
                            input_bits=activation_bit,
                            output_precision_bits=4,
                            output_range_bits=4,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=8020.0,
                        )
                        memristor_runs = 3
                        for num_cycles in range(1, memristor_runs+1):
                            model_config_recog = copy.deepcopy(model_config)
                            model_config_recog.converter_hardware_settings = recog_dac_settings
                            model_config_recog.num_cycles = num_cycles

                            prior_args = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v9,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }

                            train_args_recog = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v10,
                                "net_args": {"model_config_dict": asdict(model_config_recog)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            recog_name = prefix_name + "/" + network_module_mem_v10 + f"_{1000//10}eps_{num_layers}lay_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                            res_conv = eval_model(
                                training_name=recog_name  + f"_{num_cycles}",
                                train_job=train_job,
                                train_args=train_args_recog,
                                train_data=train_data,
                                decoder_config=rasr_config_memristor,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                result_dict=res_conv,
                                decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                prior_scales=[memristor_prior],
                                lm_scales=[memristor_lm],
                                use_gpu=True,
                                import_memristor=True,
                                extra_forward_config={
                                    "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                },
                                run_best_4=False,
                                run_best=False,
                                prior_args=prior_args,
                                run_search_on_hpc=False,
                                run_rasr=True,
                                split_mem_init = True,
                                search_gpu=48 if dim > 512 else 11,
                            )

                        recog_name = prefix_name + "/" + network_module_mem_v10 + f"_{1000 // 10}eps_{num_layers}lay_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}_cycle"
                        generate_report(results=res_conv, exp_name=recog_name)
                        memristor_report[recog_name] = copy.deepcopy(res_conv)

                        if weight_bit == 4 and dim == 512 and num_layers == 12:
                            for adc_bit in [3, 2]:
                                res_conv = {}
                                recog_dac_settings_smaller = DacAdcHardwareSettings(
                                    input_bits=activation_bit,
                                    output_precision_bits=adc_bit,
                                    output_range_bits=adc_bit,
                                    hardware_input_vmax=0.6,
                                    hardware_output_current_scaling=8020.0,
                                )
                                for num_cycles in range(1, memristor_runs + 1):
                                    model_config_recog = copy.deepcopy(model_config)
                                    model_config_recog.converter_hardware_settings = recog_dac_settings_smaller
                                    model_config_recog.num_cycles = num_cycles

                                    prior_args = {
                                        "config": train_config_24gbgpu,
                                        "network_module": network_module_mem_v9,
                                        "net_args": {"model_config_dict": asdict(model_config)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }

                                    train_args_recog = {
                                        "config": train_config_24gbgpu,
                                        "network_module": network_module_mem_v10,
                                        "net_args": {"model_config_dict": asdict(model_config_recog)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }
                                    recog_name = prefix_name + "/" + network_module_mem_v10 + f"_{1000 // 10}eps_{num_layers}lay_{dim}dim_w{weight_bit}_a{activation_bit}_adc{adc_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                    res_conv = eval_model(
                                        training_name=recog_name + f"_{num_cycles}",
                                        train_job=train_job,
                                        train_args=train_args_recog,
                                        train_data=train_data,
                                        decoder_config=rasr_config_memristor,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        result_dict=res_conv,
                                        decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                        prior_scales=[memristor_prior],
                                        lm_scales=[memristor_lm],
                                        use_gpu=True,
                                        import_memristor=True,
                                        extra_forward_config={
                                            "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                        },
                                        run_best_4=False,
                                        run_best=False,
                                        prior_args=prior_args,
                                        run_search_on_hpc=False,
                                        run_rasr=True,
                                        split_mem_init=True,
                                        search_gpu=48 if dim > 512 else 11,
                                    )

                                recog_name = prefix_name + "/" + network_module_mem_v10 + f"_{1000 // 10}eps_{num_layers}lay_{dim}dim_w{weight_bit}_a{activation_bit}_adc{adc_bit}_seed_{seed}_cycle"
                                generate_report(results=res_conv, exp_name=recog_name)
                                memristor_report[recog_name] = copy.deepcopy(res_conv)

                    #training_name = (
                    #    prefix_name
                    #    + "/"
                    #    + network_module_mem_v10
                    #    + f"_{1000//10}eps_{dim}dim_{weight_bit}_{activation_bit}_seeds_combined_cycle"
                    #)
                    # generate_report(results=res_seeds_total, exp_name=training_name)
                    # memristor_report[training_name] = copy.deepcopy(res_seeds_total)
    tk.register_report("reports/lbs/memristor_width_report", partial(build_qat_report, memristor_report), required=memristor_report, update_frequency=400)
    tk.register_report("reports/lbs/v2/memristor_width", partial(build_qat_report_v2, memristor_report),
                       required=memristor_report, update_frequency=400)