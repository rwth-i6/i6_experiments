import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Dict
from functools import partial
import os

from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon, get_bliss_phoneme_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm, get_arpa_lm_config
from ...pipeline import training
from ...report import generate_report
from .tune_eval import eval_model, build_report, build_qat_report, RTFArgs


def get_ted_data_classes(prefix_name):
    train_settings_4k = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.4000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_4k = build_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=train_settings_4k,
    )

    label_datastream = cast(LabelDatastream, train_data_4k.datastreams["labels"])

    dev_dataset_tuples = {}
    for testset in ["dev"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings_4k,
        )

    test_dataset_tuples = {}
    for testset in ["test"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings_4k,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    return train_data_4k, label_datastream, dev_dataset_tuples, test_dataset_tuples, arpa_4gram_lm

def eow_phon_ted_0825_mem_noise():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/memristor/noise"

    train_data_4k, label_datastream, dev_dataset_tuples, test_dataset_tuples, arpa_4gram_lm = get_ted_data_classes(prefix_name)
    vocab_size_without_blank = label_datastream.vocab_size

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_qat_phoneme_ctc import DecoderConfig

    default_decoder_config = DecoderConfig(  # this has memristor enabled
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
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
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    default_frontend_config = VGG4LayerActFrontendV1Config_mod(
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
    as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )

    network_module_mem_v7 = "ctc.qat_0711.memristor_v7"
    network_module_mem_v5 = "ctc.qat_0711.memristor_v5"
    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from ...pytorch_networks.ctc.qat_0711.memristor_v5_cfg import QuantModelTrainConfigV5 as MemristorModelTrainConfigV5
    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    ####################################################################################################################
    ## Weight noise
    qat_report = {}

    for activation_bit in [8]:
        dac_settings = DacAdcHardwareSettings(
            input_bits=activation_bit,
            output_precision_bits=4,
            output_range_bits=4,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
        for weight_bit in [3, 4, 5]:
            for dropout in [0.2]:
                for seed in [0, 1, 2]:
                    model_config = MemristorModelTrainConfigV5(
                        feature_extraction_config=fe_config,
                        frontend_config=default_frontend_config,
                        specaug_config=specaug_config,
                        label_target_size=vocab_size_without_blank,
                        conformer_size=384,
                        num_layers=12,
                        num_heads=4,
                        ff_dim=1536,
                        att_weights_dropout=dropout,
                        conv_dropout=dropout,
                        ff_dropout=dropout,
                        mhsa_dropout=dropout,
                        conv_kernel_size=31,
                        final_dropout=dropout,
                        specauc_start_epoch=1,
                        weight_quant_dtype="qint8",
                        weight_quant_method="per_tensor_symmetric",
                        activation_quant_dtype="qint8",
                        activation_quant_method="per_tensor_symmetric",
                        dot_quant_dtype="qint8",
                        dot_quant_method="per_tensor_symmetric",
                        Av_quant_dtype="qint8",
                        Av_quant_method="per_tensor_symmetric",
                        moving_average=None,
                        weight_bit_prec=weight_bit,
                        activation_bit_prec=activation_bit,
                        quantize_output=False,
                        converter_hardware_settings=dac_settings,
                        quant_in_linear=True,
                        num_cycles=10,  # hash compat
                    )
                    training_name = (
                        prefix_name
                        + "/"
                        + network_module_mem_v7
                        + f"_{weight_bit}_{8}_baseline_drop{dropout}_seed_{seed}"
                    )
                    train_config = {
                        "optimizer": {
                            "class": "radam",
                            "epsilon": 1e-16,
                            "weight_decay": 1e-2,
                            "decoupled_weight_decay": True,
                        },
                        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                        + list(np.linspace(5e-4, 5e-5, 110))
                        + list(np.linspace(5e-5, 1e-7, 30)),
                        #############
                        "batch_size": 180 * 16000,
                        "max_seq_length": {"audio_features": 35 * 16000},
                        "accum_grad_multiple_step": 1,
                        "gradient_clip_norm": 1.0,
                        "seed": seed,  # hash compat
                    }
                    train_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v5,
                        "net_args": {"model_config_dict": asdict(model_config)},
                        "debug": True,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }
                    train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
                    train_job.rqmt["gpu_mem"] = 11
                    results = {}
                    lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
                    prior_scales = [0.3, 0.5, 0.7]

                    results = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data_4k,
                        decoder_config=as_training_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        result_dict=results,
                        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                        prior_scales=prior_scales,
                        lm_scales=lm_scales,
                        import_memristor=True,
                        run_best=False,
                        run_best_4=False,
                    )
                    generate_report(results=results, exp_name=training_name)
                    qat_report[training_name] = results

                    prior_config = copy.deepcopy(model_config)
                    prior_config.num_cycles = 0
                    prior_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v5,
                        "net_args": {"model_config_dict": asdict(prior_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }

                    res_conv = {}
                    for num_cycle in range(1, 11):
                        if weight_bit in [1.5]:
                            continue
                        model_config_recog = copy.deepcopy(model_config)
                        model_config_recog.num_cycles = num_cycle

                        train_args = {
                            "config": train_config,
                            "network_module": network_module_mem_v5,
                            "net_args": {"model_config_dict": asdict(model_config_recog)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }

                        training_name = (
                            prefix_name
                            + "/"
                            + network_module_mem_v5
                            + f"_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycle // 11}"
                        )
                        res_conv = eval_model(
                            training_name=training_name + f"_{num_cycle}",
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data_4k,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            result_dict=res_conv,
                            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                            prior_scales=[0.5],
                            lm_scales=[2.0],
                            prior_args=prior_args,
                            run_best=False,
                            run_best_4=False,
                            import_memristor=not train_args["debug"],
                            use_gpu=True,
                            extra_forward_config={
                                "batch_size": 7000000,
                            },
                        )
                        if num_cycle % 10 == 0 and num_cycle > 0:
                            generate_report(results=res_conv, exp_name=training_name)
                            qat_report[training_name] = copy.deepcopy(res_conv)
                    for start_epoch in [1, 11]:
                        for dev in [0.0125, 0.05, 0.025, 0.1]:
                            if not seed == 0 and not weight_bit == 5:
                                continue
                            model_config = MemristorModelTrainConfigV7(
                                feature_extraction_config=fe_config,
                                frontend_config=default_frontend_config,
                                specaug_config=specaug_config,
                                label_target_size=vocab_size_without_blank,
                                conformer_size=384,
                                num_layers=12,
                                num_heads=4,
                                ff_dim=1536,
                                att_weights_dropout=dropout,
                                conv_dropout=dropout,
                                ff_dropout=dropout,
                                mhsa_dropout=dropout,
                                conv_kernel_size=31,
                                final_dropout=dropout,
                                specauc_start_epoch=1,
                                weight_quant_dtype="qint8",
                                weight_quant_method="per_tensor_symmetric",
                                activation_quant_dtype="qint8",
                                activation_quant_method="per_tensor_symmetric",
                                dot_quant_dtype="qint8",
                                dot_quant_method="per_tensor_symmetric",
                                Av_quant_dtype="qint8",
                                Av_quant_method="per_tensor_symmetric",
                                moving_average=None,
                                weight_bit_prec=weight_bit,
                                activation_bit_prec=activation_bit,
                                quantize_output=False,
                                converter_hardware_settings=dac_settings,
                                quant_in_linear=True,
                                num_cycles=0,
                                weight_noise_func="gauss",
                                weight_noise_values={"dev": dev},
                                weight_noise_start_epoch=start_epoch,
                                correction_settings=None,
                            )

                            train_config = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-16,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                                + list(np.linspace(5e-4, 5e-5, 110))
                                + list(np.linspace(5e-5, 1e-7, 30)),
                                #############
                                "batch_size": 180 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                            }
                            if seed > 0:
                                train_config["seed"] = seed
                            train_args = {
                                "config": train_config,
                                "network_module": network_module_mem_v7,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": True,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }

                            if seed > 0:
                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_mem_v7
                                    + f"_{weight_bit}_{8}_noise{start_epoch}_{dev}_drop{dropout}_seed_{seed}"
                                )
                            else:
                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_mem_v7
                                    + f"_{weight_bit}_{8}_noise{start_epoch}_{dev}_drop{dropout}"
                                )
                            train_job = training(
                                training_name, train_data_4k, train_args, num_epochs=250, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 11

                            prior_config = copy.deepcopy(model_config)
                            prior_config.weight_noise_func = None
                            prior_config.weight_noise_values = None
                            prior_config.weight_noise_start_epoch = None

                            prior_args = copy.deepcopy(train_args)
                            prior_args["net_args"] =  {"model_config_dict": asdict(prior_config)}

                            results = {}
                            lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
                            prior_scales = [0.3, 0.5, 0.7]

                            results, best_params_noise = eval_model(
                                training_name=training_name + "/with_noise",
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_4k,
                                decoder_config=as_training_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                result_dict=results,
                                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                import_memristor=True,
                                prior_args=prior_args,
                                get_best_params=True,
                            )
                            generate_report(results=results, exp_name=training_name + "/with_noise")
                            qat_report[training_name + "/with_noise"] = results

                            results = {}
                            results, best_params_no_noise = eval_model(
                                training_name=training_name + "/without_noise",
                                train_job=train_job,
                                train_args=prior_args,
                                train_data=train_data_4k,
                                decoder_config=as_training_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                result_dict=results,
                                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                import_memristor=True,
                                prior_args=prior_args,
                                get_best_params=True,
                            )
                            generate_report(results=results, exp_name=training_name + "/without_noise")
                            qat_report[training_name + "/without_noise"] = results
                            res_conv = {}
                            if seed > 0:
                                continue
                            for num_cycle in range(1, 11):
                                mem_config = copy.deepcopy(prior_config)
                                mem_config.num_cycles = num_cycle
                                train_args_recog = copy.deepcopy(train_args)
                                train_args_recog["net_args"] = {"model_config_dict": asdict(mem_config)}
                                train_args_recog["debug"] = False

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_mem_v7
                                    + f"_{weight_bit}_{8}_noise{start_epoch}_{dev}_drop{dropout}/cycle_{num_cycle // 11}"
                                )
                                res_conv = eval_model(
                                    training_name=training_name + f"_{num_cycle}",
                                    train_job=train_job,
                                    train_args=train_args_recog,
                                    train_data=train_data_4k,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    result_dict=res_conv,
                                    decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                                    prior_scales=[best_params_no_noise.out_optimal_parameters[1]],
                                    lm_scales=[(best_params_no_noise.out_optimal_parameters[0], "best_no_noise")],
                                    prior_args=prior_args,
                                    run_best=False,
                                    run_best_4=False,
                                    import_memristor=not train_args_recog["debug"],
                                    use_gpu=True,
                                    extra_forward_config={
                                        "batch_size": 7000000,
                                    },
                                    run_search_on_hpc=True,
                                )
                            training_name = (
                                prefix_name
                                + "/"
                                + network_module_mem_v7
                                + f"_{weight_bit}_{8}_noise{start_epoch}_{dev}_drop{dropout}/cycle_combined"
                            )
                            generate_report(results=res_conv, exp_name=training_name)
                            qat_report[training_name] = copy.deepcopy(res_conv)

    tk.register_report(
        "reports/ted/memristor_noise_phon", partial(build_qat_report, qat_report), required=qat_report, update_frequency=100
    )


def eow_phon_ted_0825_mem_correction():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/memristor/correction"

    train_data_4k, label_datastream, dev_dataset_tuples, test_dataset_tuples, arpa_4gram_lm = get_ted_data_classes(
        prefix_name)
    vocab_size_without_blank = label_datastream.vocab_size

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_qat_phoneme_ctc import DecoderConfig

    default_decoder_config = DecoderConfig(  # this has memristor enabled
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
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
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    default_frontend_config = VGG4LayerActFrontendV1Config_mod(
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
    as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )

    rasr_prior_scales = [0.5, 0.7, 0.9]
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

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
    mem_rasr_config = copy.deepcopy(as_training_rasr_config)
    mem_rasr_config.turn_off_quant = False

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bliss_phoneme_lexicon(),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=4096,
        score_threshold=20.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("4gram", lexicon_file=get_bliss_phoneme_lexicon(), scale=0.0),
    )

    as_training_rasr_config_larger = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,  # this will be overwritten internally
        prior_file=None,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )
    mem_rasr_config_larger = copy.deepcopy(as_training_rasr_config_larger)
    mem_rasr_config_larger.turn_off_quant = False

    network_module_mem_v7 = "ctc.qat_0711.memristor_v7"
    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    qat_report = {}
    for activation_bit in [8]:
        dac_settings_wrong = (
            DacAdcHardwareSettings(  # just to restore the training hash, does not have influence on training
                input_bits=8,
                output_precision_bits=8,
                output_range_bits=10,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
        )
        dac_settings = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=4,
            output_range_bits=4,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )

        for weight_bit in [3, 4, 5]:
            model_train_config = MemristorModelTrainConfigV7(
                feature_extraction_config=fe_config,
                frontend_config=default_frontend_config,
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
                weight_quant_dtype="qint8",
                weight_quant_method="per_tensor_symmetric",
                activation_quant_dtype="qint8",
                activation_quant_method="per_tensor_symmetric",
                dot_quant_dtype="qint8",
                dot_quant_method="per_tensor_symmetric",
                Av_quant_dtype="qint8",
                Av_quant_method="per_tensor_symmetric",
                moving_average=None,
                weight_bit_prec=weight_bit,
                activation_bit_prec=activation_bit,
                quantize_output=False,
                converter_hardware_settings=dac_settings_wrong,
                quant_in_linear=True,
                num_cycles=0,
                weight_noise_func=None,
                weight_noise_values=None,
                weight_noise_start_epoch=None,
                correction_settings=None,
            )

            train_config = {
                "optimizer": {
                    "class": "radam",
                    "epsilon": 1e-16,
                    "weight_decay": 1e-2,
                    "decoupled_weight_decay": True,
                },
                "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                + list(np.linspace(5e-4, 5e-5, 110))
                + list(np.linspace(5e-5, 1e-7, 30)),
                #############
                "batch_size": 180 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 1,
                "gradient_clip_norm": 1.0,
            }
            train_args = {
                "config": train_config,
                "network_module": network_module_mem_v7,
                "net_args": {"model_config_dict": asdict(model_train_config)},
                "debug": True,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            training_name = prefix_name + "/" + network_module_mem_v7 + f"_{weight_bit}_{8}_correction_baseline"
            train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
            train_job.rqmt["gpu_mem"] = 11

            results = {}
            training_name = prefix_name + "/" + network_module_mem_v7 + f"rasr_{weight_bit}_{8}_correction_baseline"
            results, best_params_rasr_larger = eval_model(
                training_name=training_name + "_rasr",
                train_job=train_job,
                train_args=train_args,
                train_data=train_data_4k,
                decoder_config=as_training_rasr_config_larger,
                dev_dataset_tuples=dev_dataset_tuples,
                result_dict=results,
                decoder_module="ctc.decoder.rasr_ctc_v1",
                prior_scales=rasr_prior_scales,
                lm_scales=rasr_lm_scales,
                import_memristor=True,
                prior_args=train_args,
                get_best_params=True,
                run_rasr=True
            )
            generate_report(results=results, exp_name=training_name+ "_rasr")
            qat_report[training_name] = results

            # rasr recog without correction
            prior_args = copy.deepcopy(train_args)
            res_conv = {}
            for num_cycle in range(1, 11):
                mem_config = copy.deepcopy(model_train_config)
                mem_config.num_cycles = num_cycle
                mem_config.converter_hardware_settings = dac_settings

                train_args_recog = copy.deepcopy(train_args)
                train_args_recog["net_args"] = {"model_config_dict": asdict(mem_config)}
                train_args_recog["debug"] = False
                training_name = (
                    prefix_name
                    + "/"
                    + network_module_mem_v7
                    + f"_{weight_bit}_{8}_no_correction_rasr_larger/cycle_{num_cycle // 11}"
                )

                res_conv = eval_model(
                    training_name=training_name + f"_{num_cycle}",
                    train_job=train_job,
                    train_args=train_args_recog,
                    train_data=train_data_4k,
                    decoder_config=mem_rasr_config_larger,
                    dev_dataset_tuples=dev_dataset_tuples,
                    result_dict=res_conv,
                    decoder_module="ctc.decoder.rasr_ctc_v1",
                    prior_scales=[best_params_rasr_larger.out_optimal_parameters[1]],
                    lm_scales=[(best_params_rasr_larger.out_optimal_parameters[0], "best")],
                    prior_args=prior_args,
                    run_best=False,
                    run_best_4=False,
                    import_memristor=not train_args_recog["debug"],
                    use_gpu=True,
                    extra_forward_config={
                        "batch_size": 7000000,
                    },
                    run_search_on_hpc=False,
                    run_rasr=True,
                )
            training_name = (
                prefix_name + "/" + network_module_mem_v7 + f"rasr_{weight_bit}_{8}_no_correction/cycle_combined"
            )
            generate_report(results=res_conv, exp_name=training_name)
            qat_report[training_name] = copy.deepcopy(res_conv)
            for num_cycles_correction in [1, 10, 100]:
                for test_input_value in [0.6, 0.4, 0.3]:
                    for relative_deviation in [0.0001, 0.025, 0.05, 0.1, 0.2]:
                        if weight_bit not in [4] and not (
                                num_cycles_correction == 10 and test_input_value == 0.4
                        ):
                            continue
                        cycle_correction_settings = CycleCorrectionSettings(
                            num_cycles=num_cycles_correction,
                            test_input_value=test_input_value,
                            relative_deviation=relative_deviation,
                        )
                        res_conv = {}
                        for num_cycle in range(1, 11):
                            mem_config = copy.deepcopy(model_train_config)
                            mem_config.num_cycles = num_cycle
                            mem_config.converter_hardware_settings = dac_settings
                            mem_config.correction_settings = cycle_correction_settings

                            train_args_recog = copy.deepcopy(train_args)
                            train_args_recog["net_args"] = {"model_config_dict": asdict(mem_config)}
                            train_args_recog["debug"] = False

                            training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_mem_v7
                                    + f"rasr_{weight_bit}_{8}_correction_{num_cycles_correction}_{test_input_value}_{relative_deviation}/cycle_{num_cycle // 11}"
                            )
                            priors = [best_params_rasr_larger.out_optimal_parameters[1]]
                            lms =[(best_params_rasr_larger.out_optimal_parameters[0], "best")]
                            res_conv = eval_model(
                                training_name=training_name + f"_{num_cycle}",
                                train_job=train_job,
                                train_args=train_args_recog,
                                train_data=train_data_4k,
                                decoder_config=mem_rasr_config_larger,
                                dev_dataset_tuples=dev_dataset_tuples,
                                result_dict=res_conv,
                                decoder_module="ctc.decoder.rasr_ctc_v1",
                                prior_scales=priors,
                                lm_scales=lms,
                                prior_args=prior_args,
                                run_best=False,
                                run_best_4=False,
                                import_memristor=not train_args_recog["debug"],
                                use_gpu=True,
                                extra_forward_config={
                                    "batch_size": 7000000,
                                },
                                run_search_on_hpc=False,
                                run_rasr=True
                            )
                        training_name = (
                                prefix_name
                                + "/"
                                + network_module_mem_v7
                                + f"rasr_{weight_bit}_{8}_correction_{num_cycles_correction}_{test_input_value}_{relative_deviation}/cycle_combined"
                        )
                        generate_report(results=res_conv, exp_name=training_name)
                        qat_report[training_name] = copy.deepcopy(res_conv)
    tk.register_report(
        "reports/ted/memristor_corr_phon", partial(build_qat_report, qat_report), required=qat_report, update_frequency=600
    )




def eow_phon_ted_0925_mem_width():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/memristor/width"

    train_data_4k, label_datastream, dev_dataset_tuples, test_dataset_tuples, arpa_4gram_lm = get_ted_data_classes(
        prefix_name)
    vocab_size_without_blank = label_datastream.vocab_size

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_qat_phoneme_ctc import DecoderConfig

    as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )

    larger_as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=4096,
        beam_size_token=20,
        arpa_lm=arpa_4gram_lm,
        beam_threshold=20,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )

    largerer_as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=6144,
        beam_size_token=30,
        arpa_lm=arpa_4gram_lm,
        beam_threshold=30,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )

    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

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

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bliss_phoneme_lexicon(),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=4096,
        score_threshold=20.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("4gram", lexicon_file=get_bliss_phoneme_lexicon(), scale=0.0),
    )

    as_training_rasr_config_larger = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,  # this will be overwritten internally
        prior_file=None,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )

    default_decoder_config = DecoderConfig(  # this has memristor enabled
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,  # TODO: this might need to be larger
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )
    rasr_prior_scales = [0.5, 0.7, 0.9]
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
    )
    from ...pytorch_networks.ctc.qat_0711.memristor_v4_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV4
    from ...pytorch_networks.ctc.qat_0711.memristor_v5_cfg import QuantModelTrainConfigV5 as MemristorModelTrainConfigV5

    from torch_memristor.memristor_modules import DacAdcHardwareSettings

    qat_report = {}

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
    fe_config_higher_ma = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-7,
        num_filters=80,
        center=False,
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )

    network_module_mem_v5 = "ctc.qat_0711.memristor_v5"

    dac_settings_lower = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=4,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    dim_frontend_config = VGG4LayerActFrontendV1Config_mod(
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

    # test bfloat16 training
    model_config = MemristorModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=dim_frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=384 * 4,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=1,
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor_symmetric",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor_symmetric",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor_symmetric",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor_symmetric",
        moving_average=None,
        weight_bit_prec=8,
        activation_bit_prec=8,
        quantize_output=True,
        converter_hardware_settings=dac_settings_lower,
        quant_in_linear=True,
        num_cycles=10,  # error from before, should not matter
    )
    for epochs in [250, 500, 1000]:
        train_config = {
            "optimizer": {
                "class": "radam",
                "epsilon": 1e-16,
                "weight_decay": 1e-2,
                "decoupled_weight_decay": True,
            },
            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
            + list(np.linspace(5e-5, 1e-7, 30)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "seed": 0,  # random param, this does nothing but generate a new hash!!!
            "torch_amp_options": {"dtype": "bfloat16"},
        }
        train_args = {
            "config": train_config,
            "network_module": network_module_mem_v5,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = (
            prefix_name + "/" + network_module_mem_v5 + f"_{8}_{8}_dim{384}_quant_out_eps{epochs}_seed_{0}_bfloat16"
        )
        train_job = training(training_name, train_data_4k, train_args, num_epochs=epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        results = {}
        results, best_params_job = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_4k,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.3, 0.5, 0.7],
            lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8],
            import_memristor=True,
            get_best_params=True,
        )
        generate_report(results=results, exp_name=training_name + "/non_memristor")
        qat_report[training_name] = results

        results, _ = eval_model(
            training_name=training_name + "_larger_search",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_4k,
            decoder_config=larger_as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.3, 0.5, 0.7],
            lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8],
            import_memristor=True,
            get_best_params=True,
        )
        generate_report(results=results, exp_name=training_name + "/non_memristor_larger_search")
        qat_report[training_name + "_larger_search"] = results

        results = {}
        results, _ = eval_model(
            training_name=training_name + "_rasr",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_4k,
            decoder_config=as_training_rasr_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.rasr_ctc_v1",
            prior_scales=rasr_prior_scales,
            lm_scales=rasr_lm_scales,
            import_memristor=True,
            get_best_params=True,
            run_rasr=True
        )
        generate_report(results=results, exp_name=training_name + "/non_memristor_rasr")
        qat_report[training_name + "_rasr"] = results

        results = {}
        results, _ = eval_model(
            training_name=training_name + "_rasr_larger",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_4k,
            decoder_config=as_training_rasr_config_larger,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.rasr_ctc_v1",
            prior_scales=rasr_prior_scales,
            lm_scales=rasr_lm_scales,
            import_memristor=True,
            get_best_params=True,
            run_rasr=True
        )
        generate_report(results=results, exp_name=training_name + "/non_memristor_rasr_larger")
        qat_report[training_name + "_rasr_larger_search"] = results

    for activation_bit in [8]:
        for weight_bit in [4, 3]:
            # TODO: remove 128, 256
            # TODO: add 1000 epochs
            for dim in [384, 512, 768, 1024]:
                for epochs in [1000]:
                    res_seeds_total = {}
                    for seed in range(2):
                        dac_settings_lower = DacAdcHardwareSettings(
                            input_bits=activation_bit,
                            output_precision_bits=4,
                            output_range_bits=4,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=8020.0,
                        )
                        dim_frontend_config = VGG4LayerActFrontendV1Config_mod(
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
                        model_config = MemristorModelTrainConfigV4(
                            feature_extraction_config=fe_config,
                            frontend_config=dim_frontend_config,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=dim,
                            num_layers=12,
                            num_heads=4,
                            ff_dim=dim * 4,
                            att_weights_dropout=0.2,
                            conv_dropout=0.2,
                            ff_dropout=0.2,
                            mhsa_dropout=0.2,
                            conv_kernel_size=31,
                            final_dropout=0.2,
                            specauc_start_epoch=1,
                            weight_quant_dtype="qint8",
                            weight_quant_method="per_tensor_symmetric",
                            activation_quant_dtype="qint8",
                            activation_quant_method="per_tensor_symmetric",
                            dot_quant_dtype="qint8",
                            dot_quant_method="per_tensor_symmetric",
                            Av_quant_dtype="qint8",
                            Av_quant_method="per_tensor_symmetric",
                            moving_average=None,
                            weight_bit_prec=weight_bit,
                            activation_bit_prec=activation_bit,
                            quantize_output=True,
                            converter_hardware_settings=dac_settings_lower,
                            quant_in_linear=True,
                            num_cycles=10,  # error from before, should not matter
                        )
                        train_config = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-16,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                            + list(np.linspace(5e-5, 1e-7, 30)),
                            #############
                            "batch_size": 180 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "gradient_clip_norm": 1.0,
                            "seed": seed,  # random param, this does nothing but generate a new hash!!!
                        }
                        if dim > 1024:
                            train_config['batch_size'] /= 2
                            train_config['accum_grad_multiple_step'] *= 2

                        train_args = {
                            "config": train_config,
                            "network_module": network_module_mem_v5,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }

                        training_name = (
                            prefix_name
                            + "/"
                            + network_module_mem_v5
                            + f"_{weight_bit}_{activation_bit}_dim{dim}_quant_out_eps{epochs}_seed_{seed}"
                        )
                        train_job = training(
                            training_name, train_data_4k, train_args, num_epochs=epochs, **default_returnn
                        )
                        if dim > 500:
                            train_job.rqmt["gpu_mem"] = 24
                        if dim > 1024:
                            train_job.rqmt["gpu_mem"] = 48
                            train_job.rqmt["mem_rqmt"] = 36
                        if dim == 2048:
                            train_job.rqmt["mem_rqmt"] = 56

                        results = {}
                        results = eval_model(
                            training_name=training_name + "_rasr_larger",
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data_4k,
                            decoder_config=as_training_rasr_config_larger,
                            dev_dataset_tuples=dev_dataset_tuples,
                            result_dict=results,
                            decoder_module="ctc.decoder.rasr_ctc_v1",
                            prior_scales=rasr_prior_scales,
                            lm_scales=rasr_lm_scales,
                            import_memristor=True,
                            run_rtf=False,
                            rtf_args=None,
                            run_rasr=True,
                            run_best_4=False,
                            run_best=False,
                        )
                        generate_report(results=results, exp_name=training_name + "_rasr_larger")
                        qat_report[training_name + "_rasr"] = results

                        res_conv = {}
                        for num_cycle in range(1, 11):
                            continue  # TODO: might need to make search space larger
                            model_config_recog = copy.deepcopy(model_config)
                            model_config_recog.num_cycles = num_cycle

                            train_args_recog = {
                                "config": train_config,
                                "network_module": network_module_mem_v5,
                                "net_args": {"model_config_dict": asdict(model_config_recog)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }

                            prior_args = copy.deepcopy(train_args)

                            training_name = (
                                prefix_name
                                + "/"
                                + network_module_mem_v5
                                + f"_{weight_bit}_{activation_bit}_dim{dim}_quant_out_eps{epochs}_{seed}/cycle_{num_cycle // 11}"
                            )
                            res_conv = eval_model(
                                training_name=training_name + f"_{num_cycle}",
                                train_job=train_job,
                                train_args=train_args_recog,
                                train_data=train_data_4k,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                result_dict=res_conv,
                                decoder_module="ctc.decoder.decoder.rasr_ctc_v1",
                                prior_scales=[best_params_job.out_optimal_parameters[1]],
                                lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                prior_args=prior_args,
                                run_best=False,
                                run_best_4=False,
                                import_memristor=not train_args_recog["debug"],
                                use_gpu=True,
                                extra_forward_config={
                                    "batch_size": 7000000,
                                },
                                run_rasr=True,
                            )
                            res_seeds_total.update(res_conv)
                            if num_cycle % 10 == 0 and num_cycle > 0:
                                generate_report(results=res_conv, exp_name=training_name)
                                qat_report[training_name] = copy.deepcopy(res_conv)
                    # training_name = (
                    #     prefix_name
                    #     + "/"
                    #     + network_module_mem_v5
                    #     + f"_{weight_bit}_{activation_bit}_dim{dim}_quant_out_eps{epochs}_combined_cycle"
                    # )
                    # generate_report(results=res_seeds_total, exp_name=training_name)
                    # qat_report[training_name] = copy.deepcopy(res_seeds_total)

    tk.register_report(
        "reports/ted/memristor_width_phon", partial(build_qat_report, qat_report, False), required=qat_report, update_frequency=300
    )
