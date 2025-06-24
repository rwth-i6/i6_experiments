import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Dict


from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from .tune_eval import QuantArgs
from ...data.common import DatasetSettings, build_test_dataset, build_st_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
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

    training_name = prefix_name + "/" + network_module_v4 + f"_8_8"
    train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    rtf_args = RTFArgs(
        beam_sizes=[1024],
        beam_size_tokens=[12],  # makes it much faster
        beam_thresholds=[14],
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
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        run_rtf=True,
        rtf_args=rtf_args,
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    training_name = prefix_name + "/" + network_module_v4 + f"_8_8_real_quant"
    train_job = training(training_name, train_data_4k, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    rtf_args = RTFArgs(
        beam_sizes=[1024],
        beam_size_tokens=[12],  # makes it much faster
        beam_thresholds=[14],
        decoder_module="ctc.decoder.flashlight_ctc_v2_rescale_measure",
    )
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_4k,
        decoder_config=default_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        run_rtf=True,
        rtf_args=rtf_args,
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

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
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
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
    from ...pytorch_networks.ctc.qat_0711.memristor_v1_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV1
    from ...pytorch_networks.ctc.qat_0711.memristor_v4_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV4
    from ...pytorch_networks.ctc.qat_0711.memristor_v5_cfg import QuantModelTrainConfigV5 as MemristorModelTrainConfigV5
    from ...pytorch_networks.ctc.qat_0711.memristor_v6_cfg import QuantModelTrainConfigV6 as MemristorModelTrainConfigV6
    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    for activation_bit in [8]:
        for weight_bit in [8, 7, 6, 5, 4, 3, 2, 1.5]:
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
                prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
                lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
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
            for num_cycle in range(1, 11):
                if weight_bit in [1.5, 2]:
                    continue
                dac_settings_lower = DacAdcHardwareSettings(
                    input_bits=activation_bit,
                    output_precision_bits=4,
                    output_range_bits=4,
                    hardware_input_vmax=0.6,
                    hardware_output_current_scaling=8020.0,
                )
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
                    num_cycles=num_cycle,
                )

                train_args = {
                    "config": train_config,
                    "network_module": network_module_mem_v4,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": False,
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
                    + f"_{weight_bit}_{activation_bit}_lower_output_prec_cycle/{num_cycle // 11}"
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
                    extra_forward_config={
                        "batch_size": 7000000,
                    },
                )
                if num_cycle % 10 == 0:
                    generate_report(results=res_cyc, exp_name=training_name)
                    qat_report[training_name] = copy.deepcopy(res_cyc)
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
            if weight_bit in [1.5]:
                continue
            res_seeds_total = {}
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
                results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data_4k,
                    decoder_config=as_training_decoder_config,
                    dev_dataset_tuples=dev_dataset_tuples,
                    result_dict=results,
                    decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                    prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
                    lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                    import_memristor=True,
                )
                generate_report(results=results, exp_name=training_name + "/non_memristor")
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
                prefix_name + "/" + network_module_mem_v5 + f"_{weight_bit}_{activation_bit}_seeds_combined_cycle"
            )
            generate_report(results=res_seeds_total, exp_name=training_name)
            qat_report[training_name] = copy.deepcopy(res_seeds_total)
    ####################################################################################################################
    ## Weight noise
    for noise_module in [network_module_mem_v7]:
        for activation_bit in [8]:
            for weight_bit in [3, 4, 5]:
                for dropout in [0.0, 0.2]:
                    if dropout == 0.0 and weight_bit not in [4]:
                        continue
                    for start_epoch in [1, 11]:
                        for dev in [0.05, 0.025, 0.1]:
                            dac_settings = DacAdcHardwareSettings(
                                input_bits=8,
                                output_precision_bits=8,
                                output_range_bits=10,
                                hardware_input_vmax=0.6,
                                hardware_output_current_scaling=8020.0,
                            )
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
                            train_args = {
                                "config": train_config,
                                "network_module": noise_module,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": True,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }

                            training_name = (
                                prefix_name
                                + "/"
                                + noise_module
                                + f"_{weight_bit}_{8}_noise{start_epoch}_{dev}_drop{dropout}"
                            )
                            train_job = training(
                                training_name, train_data_4k, train_args, num_epochs=250, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 11

                            prior_config = MemristorModelTrainConfigV7(
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
                                weight_noise_func=None,
                                weight_noise_values=None,
                                weight_noise_start_epoch=None,
                                correction_settings=None,
                            )
                            prior_args = {
                                "config": train_config,
                                "network_module": noise_module,
                                "net_args": {"model_config_dict": asdict(prior_config)},
                                "debug": True,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }

                            results = {}

                            results, best_params_noise = eval_model(
                                training_name=training_name + "/with_noise",
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_4k,
                                decoder_config=as_training_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                result_dict=results,
                                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                                prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
                                import_memristor=True,
                                prior_args=prior_args,
                                get_best_params=True
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
                                prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
                                import_memristor=True,
                                prior_args=prior_args,
                                get_best_params=True,
                            )
                            generate_report(results=results, exp_name=training_name + "/without_noise")
                            qat_report[training_name + "/without_noise"] = results
                            res_conv = {}
                            for num_cycle in range(1, 11):
                                mem_config = MemristorModelTrainConfigV7(
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
                                    converter_hardware_settings=dac_settings_lower,
                                    quant_in_linear=True,
                                    num_cycles=num_cycle,
                                    weight_noise_func=None,
                                    weight_noise_values=None,
                                    weight_noise_start_epoch=None,
                                    correction_settings=None,
                                )

                                train_args = {
                                    "config": train_config,
                                    "network_module": noise_module,
                                    "net_args": {"model_config_dict": asdict(mem_config)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }

                                MemristorModelTrainConfigV7(
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
                                    weight_noise_func=None,
                                    weight_noise_values=None,
                                    weight_noise_start_epoch=None,
                                    correction_settings=None,
                                )

                                prior_args = {
                                    "config": train_config,
                                    "network_module": noise_module,
                                    "net_args": {"model_config_dict": asdict(prior_config)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + noise_module
                                    + f"_{weight_bit}_{8}_noise{start_epoch}_{dev}_drop{dropout}/cycle_{num_cycle // 11}"
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
                                res_conv = eval_model(
                                    training_name=training_name + f"_{num_cycle}",
                                    train_job=train_job,
                                    train_args=train_args,
                                    train_data=train_data_4k,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    result_dict=res_conv,
                                    decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                                    prior_scales=[best_params_noise.out_optimal_parameters[1]],
                                    lm_scales=[(best_params_noise.out_optimal_parameters[0], "best_noise")],
                                    prior_args=prior_args,
                                    run_best=False,
                                    run_best_4=False,
                                    import_memristor=not train_args["debug"],
                                    use_gpu=True,
                                    extra_forward_config={
                                        "batch_size": 7000000,
                                    },
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
                                    prior_scales=[best_params_no_noise.out_optimal_parameters[1]],
                                    lm_scales=[(best_params_no_noise.out_optimal_parameters[0], "best_no_noise")],
                                    prior_args=prior_args,
                                    run_best=False,
                                    run_best_4=False,
                                    import_memristor=not train_args["debug"],
                                    use_gpu=True,
                                    extra_forward_config={
                                        "batch_size": 7000000,
                                    },
                                )
                            training_name = (
                                prefix_name
                                + "/"
                                + noise_module
                                + f"_{weight_bit}_{8}_noise{start_epoch}_{dev}_drop{dropout}/cycle_combined"
                            )
                            generate_report(results=res_conv, exp_name=training_name)
                            qat_report[training_name] = copy.deepcopy(res_conv)

    ####################################################################################################################
    ## Error Correction
    for activation_bit in [8]:
        for weight_bit in [4]:
            dac_settings = DacAdcHardwareSettings(
                input_bits=8,
                output_precision_bits=8,
                output_range_bits=10,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            model_train_config = MemristorModelTrainConfigV7(
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
            results = eval_model(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args,
                train_data=train_data_4k,
                decoder_config=as_training_decoder_config,
                dev_dataset_tuples=dev_dataset_tuples,
                result_dict=results,
                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
                lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],
                import_memristor=True,
                prior_args=train_args,
            )
            generate_report(results=results, exp_name=training_name)
            qat_report[training_name] = results

            prior_config = MemristorModelTrainConfigV7(
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
                weight_noise_func=None,
                weight_noise_values=None,
                weight_noise_start_epoch=None,
                correction_settings=None,
            )

            prior_args = {
                "config": train_config,
                "network_module": network_module_mem_v7,
                "net_args": {"model_config_dict": asdict(prior_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            # recog without correction
            res_conv = {}
            for num_cycle in range(1, 11):
                mem_config = MemristorModelTrainConfigV7(
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
                    converter_hardware_settings=dac_settings_lower,
                    quant_in_linear=True,
                    num_cycles=num_cycle,
                    weight_noise_func=None,
                    weight_noise_values=None,
                    weight_noise_start_epoch=None,
                    correction_settings=None,
                )
                train_args = {
                    "config": train_config,
                    "network_module": network_module_mem_v7,
                    "net_args": {"model_config_dict": asdict(mem_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }
                training_name = (
                    prefix_name
                    + "/"
                    + network_module_mem_v7
                    + f"_{weight_bit}_{8}_no_correction/cycle_{num_cycle // 11}"
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
            training_name = (
                prefix_name
                + "/"
                + network_module_mem_v7
                + f"_{weight_bit}_{8}_no_correction/cycle_combined"
            )
            generate_report(results=res_conv, exp_name=training_name)
            qat_report[training_name] = copy.deepcopy(res_conv)

            for num_cycles_correction in [1, 10, 20]:
                for test_input_value in [0.6, 0.4]:
                    for relative_deviation in [0.05, 0.1, 0.2]:
                        cycle_correction_settings = CycleCorrectionSettings(
                            num_cycles=num_cycles_correction,
                            test_input_value=test_input_value,
                            relative_deviation=relative_deviation,
                        )
                        res_conv = {}
                        for num_cycle in range(1, 11):
                            mem_config = MemristorModelTrainConfigV7(
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
                                converter_hardware_settings=dac_settings_lower,
                                quant_in_linear=True,
                                num_cycles=num_cycle,
                                weight_noise_func=None,
                                weight_noise_values=None,
                                weight_noise_start_epoch=None,
                                correction_settings=cycle_correction_settings,
                            )
                            train_args = {
                                "config": train_config,
                                "network_module": network_module_mem_v7,
                                "net_args": {"model_config_dict": asdict(mem_config)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }

                            training_name = (
                                prefix_name
                                + "/"
                                + network_module_mem_v7
                                + f"_{weight_bit}_{8}_correction_{num_cycles_correction}_{test_input_value}_{relative_deviation}/cycle_{num_cycle // 11}"
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
                                    "batch_size": 7000001,
                                },
                            )
                        training_name = (
                            prefix_name
                            + "/"
                            + network_module_mem_v7
                            + f"_{weight_bit}_{8}_correction_{num_cycles_correction}_{test_input_value}_{relative_deviation}/cycle_combined"
                        )
                        generate_report(results=res_conv, exp_name=training_name)
                        qat_report[training_name] = copy.deepcopy(res_conv)

    tk.register_report(
        "reports/qat_report_phon", partial(build_qat_report, qat_report), required=qat_report, update_frequency=100
    )
