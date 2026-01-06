import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Dict
import os

from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from .tune_eval import QuantArgs
from ...data.common import DatasetSettings, build_test_dataset, build_st_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon, get_bliss_phoneme_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm, get_arpa_lm_config
from ...pipeline import training, prepare_asr_model
from ...report import generate_report
from .tune_eval import tune_and_evaluate_helper, eval_model, build_report, build_qat_report, RTFArgs
from functools import partial


def eow_phon_ted_1023_qat():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/qat"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=train_settings,
    )

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

    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

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
    rasr_prior_scales = [0.5, 0.7, 0.9]
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

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

    qat_report = {}
    # try out lower bits
    as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )
    network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    ############################################################################################
    # No specific changes for memristor, this would be the "best" we can do

    model_config = QuantModelTrainConfigV4(
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
        weight_quant_method="per_tensor",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor",
        moving_average=None,
        weight_bit_prec=8,
        activation_bit_prec=8,
        quantize_output=False,
        extra_act_quant=False,
        quantize_bias=None,
        observer_only_in_train=False,
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
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
    }
    train_args = {
        "config": train_config,
        "network_module": network_module_v4,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v4 + f"_8_8_phon"
    train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    rtf_args = RTFArgs(
        beam_sizes=[256, 512, 1024, 4096],
        beam_size_tokens=[4, 8, 12, 20, 30],
        beam_thresholds=[4, 8, 20, 30],
        include_gpu=True,
    )
    results = eval_model(
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
        run_rtf=True,
        rtf_args=rtf_args,
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    # DEPRECATED
    # training_name = prefix_name + "/" + network_module_v4 + f"_8_8_real_quant"
    # train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
    # train_job.rqmt["gpu_mem"] = 48
    # results = {}
    # rtf_args = RTFArgs(
    #     beam_sizes=[1024],
    #     beam_size_tokens=[12],  # makes it much faster
    #     beam_thresholds=[14],
    #     decoder_module="ctc.decoder.flashlight_ctc_v6_rescale_measure",
    # )
    # results = eval_model(
    #     training_name=training_name,
    #     train_job=train_job,
    #     train_args=train_args,
    #     train_data=train_data_4k,
    #     decoder_config=default_decoder_config,
    #     dev_dataset_tuples=dev_dataset_tuples,
    #     result_dict=results,
    #     decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
    #     prior_scales=[0.3, 0.5, 0.7],
    #     lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8],
    #     run_rtf=True,
    #     rtf_args=rtf_args,
    # )
    # generate_report(results=results, exp_name=training_name)
    # qat_report[training_name] = results

    ############################################################################################
    # This also does not have specific changes yet, but quantizes everything
    network_module_full_v1 = "ctc.qat_0711.full_qat_v1"
    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
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
        weight_quant_method="per_tensor",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor",
        moving_average=None,
        weight_bit_prec=8,
        activation_bit_prec=8,
        quantize_output=True,
        quantize_bias=True,
        extra_act_quant=False,
        observer_only_in_train=False,
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
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
    }
    train_args = {
        "config": train_config,
        "network_module": network_module_full_v1,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_full_v1 + f"_8_8"
    train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    results = eval_model(
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
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    network_module_mem_v1 = "ctc.qat_0711.memristor_v1"
    network_module_mem_v2 = "ctc.qat_0711.memristor_v2"
    network_module_mem_v3 = "ctc.qat_0711.memristor_v3"
    network_module_mem_v4 = "ctc.qat_0711.memristor_v4"
    network_module_mem_v5 = "ctc.qat_0711.memristor_v5"
    network_module_mem_v6 = "ctc.qat_0711.memristor_v6"
    network_module_mem_v7 = "ctc.qat_0711.memristor_v7"
    # network_module_mem_v8 = "ctc.qat_0711.memristor_v8"
    network_module_mem_v9 = "ctc.qat_0711.memristor_v9"
    from ...pytorch_networks.ctc.qat_0711.memristor_v1_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV1
    from ...pytorch_networks.ctc.qat_0711.memristor_v4_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV4
    from ...pytorch_networks.ctc.qat_0711.memristor_v5_cfg import QuantModelTrainConfigV5 as MemristorModelTrainConfigV5
    from ...pytorch_networks.ctc.qat_0711.memristor_v6_cfg import QuantModelTrainConfigV6 as MemristorModelTrainConfigV6
    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8
    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import ConformerPosEmbConfig
    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    for activation_bit in [8]:
        for weight_bit in [8, 7, 6, 5, 4, 3, 2]:
            dac_settings = DacAdcHardwareSettings(
                input_bits=activation_bit,
                output_precision_bits=8,
                output_range_bits=10,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            model_config = MemristorModelTrainConfigV1(
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
                converter_hardware_settings=dac_settings,
                quant_in_linear=True,
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
                "batch_size": 300 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 1,
                "gradient_clip_norm": 1.0,
            }
            train_args = {
                "config": train_config,
                "network_module": network_module_mem_v1,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            training_name = prefix_name + "/" + network_module_mem_v1 + f"_{weight_bit}_{activation_bit}"
            train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            results = {}
            results = eval_model(
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
            )
            generate_report(results=results, exp_name=training_name)
            qat_report[training_name] = results

            ##################################################################################
            # Arbitrarily high precision
            res_cyc = {}
            res_conv = {}
            for num_cycle in range(1, 11):
                continue
                if num_cycle > 10 and not weight_bit == 4:
                    continue
                model_config = MemristorModelTrainConfigV4(
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
                    converter_hardware_settings=dac_settings,
                    quant_in_linear=True,
                    num_cycles=num_cycle,
                )

                train_args = {
                    "config": train_config,
                    "network_module": network_module_mem_v4,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": True,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }

                prior_config = MemristorModelTrainConfigV4(
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
                    converter_hardware_settings=dac_settings,
                    quant_in_linear=True,
                    num_cycles=0,
                )
                prior_args = {
                    "config": train_config,
                    "network_module": network_module_mem_v4,
                    "net_args": {"model_config_dict": asdict(prior_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }

                training_name = (
                    prefix_name
                    + "/"
                    + network_module_mem_v4
                    + f"_{weight_bit}_{activation_bit}_cycle/{num_cycle // 11}"
                )
                res_cyc = eval_model(
                    training_name=training_name + f"_{num_cycle}",
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data_4k,
                    decoder_config=default_decoder_config,
                    dev_dataset_tuples=dev_dataset_tuples,
                    result_dict=res_cyc,
                    decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                    prior_scales=[0.5],
                    lm_scales=[2.0],
                    prior_args=prior_args,
                    run_best=False,
                    run_best_4=False,
                    import_memristor=not train_args["debug"],
                    use_gpu=True,
                )
                if num_cycle % 10 == 0:
                    generate_report(results=res_cyc, exp_name=training_name)
                    qat_report[training_name] = copy.deepcopy(res_cyc)

            ##################################################################################
            # Reasonable precision
            res_cyc = {}
            # for num_cycle in range(1, 11):
            dac_settings_lower = DacAdcHardwareSettings(
                input_bits=activation_bit,
                output_precision_bits=4,
                output_range_bits=4,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            for num_cycle in range(1, 11):
                if weight_bit in [1.5, 2]:
                    continue
                model_config = MemristorModelTrainConfigV5(
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
                    converter_hardware_settings=dac_settings_lower,
                    quant_in_linear=True,
                    num_cycles=num_cycle,
                )

                train_args = {
                    "config": train_config,
                    "network_module": network_module_mem_v5,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }

                prior_config = MemristorModelTrainConfigV5(
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
                    converter_hardware_settings=dac_settings_lower,
                    quant_in_linear=True,
                    num_cycles=0,
                )
                prior_args = {
                    "config": train_config,
                    "network_module": network_module_mem_v5,
                    "net_args": {"model_config_dict": asdict(prior_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }

                training_name = (
                    prefix_name
                    + "/"
                    + network_module_mem_v5
                    + f"_{weight_bit}_{activation_bit}_lower_output_prec_cycle/{num_cycle // 11}"
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
                        "batch_size": 7000003,
                    },
                )
                if num_cycle % 10 == 0 and num_cycle > 0:
                    generate_report(results=res_conv, exp_name=training_name)
                    qat_report[training_name] = copy.deepcopy(res_conv)

            ####################################################################################################################
            ## Multiple runs
            if weight_bit in [1.5, 2]:
                continue
            res_seeds_total = {}
            res_seeds_total_best = {}
            for seed in range(3):
                model_config = MemristorModelTrainConfigV4(
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
                    "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                    + list(np.linspace(5e-4, 5e-5, 110))
                    + list(np.linspace(5e-5, 1e-7, 30)),
                    #############
                    "batch_size": 180 * 16000,
                    "max_seq_length": {"audio_features": 35 * 16000},
                    "accum_grad_multiple_step": 1,
                    "gradient_clip_norm": 1.0,
                    "seed": seed,  # random param, this does nothing but generate a new hash!!!
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
                    prefix_name + "/" + network_module_mem_v5 + f"_{weight_bit}_{activation_bit}_seed_{seed}"
                )
                train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
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

                results = {}
                results = eval_model(
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
                    run_rasr=True,
                    run_best_4=False,
                )
                generate_report(results=results, exp_name=training_name + "/non_memristor_rasr")
                qat_report[training_name + "_rasr"] = results

                res_conv = {}
                res_best = {}
                for num_cycle in range(1, 11):
                    if weight_bit in [1.5]:
                        continue
                    model_config = MemristorModelTrainConfigV5(
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
                        converter_hardware_settings=dac_settings_lower,
                        quant_in_linear=True,
                        num_cycles=num_cycle,
                    )

                    train_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v5,
                        "net_args": {"model_config_dict": asdict(model_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }

                    prior_config = MemristorModelTrainConfigV5(
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
                        converter_hardware_settings=dac_settings_lower,
                        quant_in_linear=True,
                        num_cycles=0,
                    )
                    prior_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v5,
                        "net_args": {"model_config_dict": asdict(prior_config)},
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
                    res_seeds_total.update(res_conv)
                    if num_cycle % 10 == 0 and num_cycle > 0:
                        generate_report(results=res_conv, exp_name=training_name)
                        qat_report[training_name] = copy.deepcopy(res_conv)

                    training_name = (
                        prefix_name
                        + "/"
                        + network_module_mem_v5
                        + f"_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycle // 11}_best"
                    )
                    res_best = eval_model(
                        training_name=training_name + f"_{num_cycle}",
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data_4k,
                        decoder_config=default_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        result_dict=res_best,
                        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                        prior_scales=[best_params_job.out_optimal_parameters[1]],
                        lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                        prior_args=prior_args,
                        run_best=False,
                        run_best_4=False,
                        import_memristor=not train_args["debug"],
                        use_gpu=True,
                        extra_forward_config={
                            "batch_size": 7000000,
                        },
                    )
                    res_seeds_total_best.update(res_best)
                    if num_cycle % 10 == 0 and num_cycle > 0:
                        generate_report(results=res_best, exp_name=training_name)
                        qat_report[training_name] = copy.deepcopy(res_best)

            training_name = (
                prefix_name + "/" + network_module_mem_v5 + f"_{weight_bit}_{activation_bit}_seeds_combined_cycle"
            )
            generate_report(results=res_seeds_total, exp_name=training_name)
            qat_report[training_name] = copy.deepcopy(res_seeds_total)
            training_name = (
                prefix_name + "/" + network_module_mem_v5 + f"_{weight_bit}_{activation_bit}_seeds_combined_best_cycle"
            )
            generate_report(results=res_seeds_total_best, exp_name=training_name)
            qat_report[training_name] = copy.deepcopy(res_seeds_total_best)

            for seed in range(3):
                if weight_bit in [1.5, 2, 7, 6]:
                    continue
                model_config = MemristorModelTrainConfigV4(
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
                    "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                    + list(np.linspace(5e-4, 5e-5, 110))
                    + list(np.linspace(5e-5, 1e-7, 30)),
                    #############
                    "batch_size": 180 * 16000,
                    "max_seq_length": {"audio_features": 35 * 16000},
                    "accum_grad_multiple_step": 1,
                    "gradient_clip_norm": 1.0,
                    "seed": seed,  # random param, this does nothing but generate a new hash!!!
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
                    prefix_name
                    + "/"
                    + network_module_mem_v5
                    + f"_{weight_bit}_{activation_bit}_quantize_out_seed_{seed}"
                )
                train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
                if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                    train_job.rqmt["cpu"] = 24
                    train_job.hold()
                    train_job.move_to_hpc = True
                results = {}
                results = eval_model(
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
                )
                generate_report(results=results, exp_name=training_name + "/non_memristor")
                qat_report[training_name] = results

                results = {}
                results = eval_model(
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
                    run_rasr=True,
                    run_best_4=False,
                )
                generate_report(results=results, exp_name=training_name + "/non_memristor_rasr")
                qat_report[training_name] = results

                res_conv = {}
                for num_cycle in range(1, 11):
                    if weight_bit in [1.5]:
                        continue
                    model_config = MemristorModelTrainConfigV5(
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
                        quantize_output=True,
                        converter_hardware_settings=dac_settings_lower,
                        quant_in_linear=True,
                        num_cycles=num_cycle,
                    )

                    train_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v5,
                        "net_args": {"model_config_dict": asdict(model_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }

                    prior_config = MemristorModelTrainConfigV5(
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
                        quantize_output=True,
                        converter_hardware_settings=dac_settings_lower,
                        quant_in_linear=True,
                        num_cycles=0,
                    )
                    prior_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v5,
                        "net_args": {"model_config_dict": asdict(prior_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }

                    training_name = (
                        prefix_name
                        + "/"
                        + network_module_mem_v5
                        + f"_{weight_bit}_{activation_bit}_quantize_out_seed_{seed}/cycle_{num_cycle // 11}"
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
                    res_seeds_total.update(res_conv)
                    if num_cycle % 10 == 0 and num_cycle > 0:
                        generate_report(results=res_conv, exp_name=training_name)
                        qat_report[training_name] = copy.deepcopy(res_conv)

            training_name = (
                prefix_name
                + "/"
                + network_module_mem_v5
                + f"_{weight_bit}_{activation_bit}_quant_out_seeds_combined_cycle"
            )
            generate_report(results=res_seeds_total, exp_name=training_name)
            qat_report[training_name] = copy.deepcopy(res_seeds_total)

            for epochs in [250, 1000]:
                for seed in range(3):
                    if epochs == 250 and not seed == 0:
                        continue
                    dim = 384
                    pos_emb_cfg = ConformerPosEmbConfig(
                        learnable_pos_emb=False,
                        rel_pos_clip=16,
                        with_linear_pos=True,
                        with_pos_bias=True,
                        separate_pos_emb_per_head=True,
                        pos_emb_dropout=0.0,
                    )
                    if weight_bit == 8 and epochs in [1000]:
                        stronger_specaug_config = SpecaugConfig(
                            repeat_per_n_frames=25,
                            max_dim_time=20,
                            max_dim_feat=16,  # Jingjing style
                            num_repeat_feat=5,
                        )
                        model_config = MemristorModelTrainConfigV8(
                            feature_extraction_config=fe_config,
                            frontend_config=default_frontend_config,
                            specaug_config=stronger_specaug_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=384,
                            num_layers=12,
                            num_heads=8,
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
                            weight_bit_prec=weight_bit,
                            activation_bit_prec=activation_bit,
                            quantize_output=True,
                            converter_hardware_settings=None,
                            quant_in_linear=True,
                            num_cycles=0,
                            pos_emb_config=pos_emb_cfg,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                            dropout_broadcast_axes=None,
                            correction_settings=None,
                            weight_noise_func=None,
                            weight_noise_values=None,
                            weight_noise_start_epoch=None,
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
                            "torch_amp_options": {"dtype": "bfloat16"},
                        }

                        train_args = {
                            "config": train_config,
                            "network_module": network_module_mem_v9,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }

                        training_name = (
                                prefix_name
                                + "/"
                                + network_module_mem_v9
                                + f"_{weight_bit}_{activation_bit}_dim{dim}_better_params_quant_out_eps{epochs}_seed_{seed}"
                        )
                        train_job = training(
                            training_name, train_data_4k, train_args, num_epochs=epochs, **default_returnn
                        )
                        train_job.rqmt["gpu_mem"] = 48

                    model_config = MemristorModelTrainConfigV8(
                        feature_extraction_config=fe_config,
                        frontend_config=default_frontend_config,
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
                        weight_bit_prec=weight_bit,
                        activation_bit_prec=activation_bit,
                        quantize_output=True,
                        converter_hardware_settings=None,
                        quant_in_linear=True,
                        num_cycles=0,
                        pos_emb_config=pos_emb_cfg,
                        module_list=["ff", "conv", "mhsa", "ff"],
                        module_scales=[0.5, 1.0, 1.0, 0.5],
                        aux_ctc_loss_layers=None,
                        aux_ctc_loss_scales=None,
                        dropout_broadcast_axes=None,
                        correction_settings=None,
                        weight_noise_func=None,
                        weight_noise_values=None,
                        weight_noise_start_epoch=None,
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
                        "torch_amp_options": {"dtype": "bfloat16"},
                    }

                    train_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v9,
                        "net_args": {"model_config_dict": asdict(model_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }

                    training_name = (
                            prefix_name
                            + "/"
                            + network_module_mem_v9
                            + f"_{weight_bit}_{activation_bit}_dim{dim}_quant_out_eps{epochs}_seed_{seed}"
                    )
                    train_job = training(
                        training_name, train_data_4k, train_args, num_epochs=epochs, **default_returnn
                    )
                    train_job.rqmt["gpu_mem"] = 24

                    # results = {}
                    # results, best_params_job = eval_model(
                    #     training_name=training_name,
                    #     train_job=train_job,
                    #     train_args=train_args,
                    #     train_data=train_data_4k,
                    #     decoder_config=as_training_decoder_config,
                    #     dev_dataset_tuples=dev_dataset_tuples,
                    #     result_dict=results,
                    #     decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                    #     prior_scales=[0.3, 0.5, 0.7],
                    #     lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8],
                    #     import_memristor=True,
                    #     get_best_params=True,
                    #     loss_name="dev_loss_ctc_loss_layer12"
                    # )
                    # generate_report(results=results, exp_name=training_name + "/non_memristor")
                    # qat_report[training_name] = results

                    results = {}
                    results, best_params_job = eval_model(
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
                        loss_name="dev_loss_ctc_loss_layer12",
                        run_rasr=True,
                        run_best_4=False,
                    )
                    generate_report(results=results, exp_name=training_name + "/non_memristor_rasr")
                    qat_report[training_name + "_rasr"] = results

                    res_conv = {}
                    continue
                    for num_cycle in range(1, 11):
                        model_config = MemristorModelTrainConfigV8(
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
                            num_cycles=num_cycle,
                        )

                        train_args = {
                            "config": train_config,
                            "network_module": network_module_mem_v5,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }

                        prior_config = MemristorModelTrainConfigV8(
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
                            num_cycles=0,
                        )
                        prior_args = {
                            "config": train_config,
                            "network_module": network_module_mem_v9,
                            "net_args": {"model_config_dict": asdict(prior_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }

                        training_name = (
                                prefix_name
                                + "/"
                                + network_module_mem_v9
                                + f"_{weight_bit}_{activation_bit}_dim{dim}_quant_out_eps{epochs}_{seed}/cycle_{num_cycle // 11}"
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
                            prior_scales=[best_params_job.out_optimal_parameters[1]],
                            lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                            prior_args=prior_args,
                            run_best=False,
                            run_best_4=False,
                            import_memristor=not train_args["debug"],
                            use_gpu=True,
                            extra_forward_config={
                                "batch_size": 7000000,
                            },
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
        "reports/ted/qat_report_phon", partial(build_qat_report, qat_report), required=qat_report, update_frequency=100
    )
