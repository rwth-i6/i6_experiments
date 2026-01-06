from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List
from functools import partial

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon, get_bpe_lexicon, get_bpe_bliss_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm, get_arpa_lm_config
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...report import generate_report
from ...experiments.ctc_phon.tune_eval import build_qat_report

from ..ctc_phon.tune_eval import eval_model


def bpe_ted_0125_qat():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/bpe_ctc_bpe/256/qat"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.4000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe256 = build_bpe_training_datasets(
        prefix=prefix_name,
        bpe_size=256,  # TODO tune
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe256 = cast(LabelDatastream, train_data_bpe256.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe256.vocab_size

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

    default_decoder_config_bpe256 = DecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=256),
        returnn_vocab=label_datastream_bpe256.vocab,
        beam_size=1024,  # Untuned
        beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,  # Untuned
    )
    as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=256),
        returnn_vocab=label_datastream_bpe256.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        turn_off_quant="leave_as_is",
    )

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyCTCDecoderConfig

    as_training_greedy_decoder_config = GreedyCTCDecoderConfig(
        returnn_vocab=label_datastream_bpe256.vocab,
        turn_off_quant="leave_as_is",
    )

    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import (
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
    frontend_config_sub4 = VGG4LayerActFrontendV1Config_mod(  # TODO: this might be subsampling 4
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
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

    ####################################################################################################
    # QAT Baseline
    network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
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
        specauc_start_epoch=11,
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

    training_name = prefix_name + "/" + network_module_v4 + f"_8_8_later_spec"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    results = {}
    results = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
    )
    generate_report(results=results, exp_name=training_name + "_greedy")
    qat_report[training_name + "_greedy"] = results

    # STOP READING HERE :)

    #########################################################################################

    ####################################################################################################
    # QAT Baseline
    network_module_v5 = "ctc.qat_0711.baseline_qat_v5"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v5_cfg import QuantModelTrainConfigV5
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v5_cfg import VGG4LayerActFrontendV1Config_mod as QuantFrontendConfig

    quant_frontend_config_sub4 = QuantFrontendConfig(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor",
        moving_average=None,
        weight_bit_prec=8,
        activation_bit_prec=8,
        observer_only_in_train=False,
    )

    model_config = QuantModelTrainConfigV5(
        feature_extraction_config=fe_config,
        frontend_config=quant_frontend_config_sub4,
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
        specauc_start_epoch=11,
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
        "torch_amp_options": {"dtype": "bfloat16"},
    }
    train_args = {
        "config": train_config,
        "network_module": network_module_v5,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v5 + f"_8_8"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    results = {}
    results = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
    )
    generate_report(results=results, exp_name=training_name + "_greedy")
    qat_report[training_name + "_greedy"] = results

    # STOP READING HERE :)

    #########################################################################################

    # Full Quant Baseline
    network_module_v1 = "ctc.qat_0711.full_qat_v1"
    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
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
        specauc_start_epoch=11,
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
        "network_module": network_module_v1,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v1 + f"_{8}_{8}"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    results = {}
    results = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
    )
    generate_report(results=results, exp_name=training_name + "_greedy")
    qat_report[training_name + "_greedy"] = results

    #########################################################################################
    # Full Quant With Sym
    network_module_v1 = "ctc.qat_0711.full_qat_v1"
    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
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
        specauc_start_epoch=11,
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
        "network_module": network_module_v1,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v1 + f"_{8}_{8}_sym"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    results = {}
    results = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
    )
    generate_report(results=results, exp_name=training_name + "_greedy")
    qat_report[training_name + "_greedy"] = results

    # memristor BPE
    from ...pytorch_networks.ctc.qat_0711.memristor_v5_cfg import QuantModelTrainConfigV5 as MemristorModelTrainConfigV5

    network_module_mem_v5 = "ctc.qat_0711.memristor_v5"
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

    for activation_bit in [8]:
        dac_settings_lower = DacAdcHardwareSettings(
            input_bits=activation_bit,
            output_precision_bits=4,
            output_range_bits=4,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
        for weight_bit in [4]:
            res_seeds_total = {}
            for seed in range(3):
                model_config = MemristorModelTrainConfigV5(
                    feature_extraction_config=fe_config,
                    frontend_config=frontend_config_sub4,
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
                    specauc_start_epoch=11,
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
                    converter_hardware_settings=None,
                    quant_in_linear=True,
                    num_cycles=0,
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
                train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
                train_job.rqmt["gpu_mem"] = 24
                results = {}
                results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data_bpe256,
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
                        frontend_config=frontend_config_sub4,
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
                        frontend_config=frontend_config_sub4,
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
                        converter_hardware_settings=None,
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
                        train_data=train_data_bpe256,
                        decoder_config=default_decoder_config_bpe256,
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

    tk.register_report("reports/ted/qat_report_bpe", partial(build_qat_report, qat_report), required=qat_report)


def bpe_ted_qat_comparisons():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/bpe_ctc_bpe/256/qat_comparison"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe256 = build_bpe_training_datasets(
        prefix=prefix_name,
        bpe_size=256,  # TODO tune
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe256 = cast(LabelDatastream, train_data_bpe256.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe256.vocab_size

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

    default_decoder_config_bpe256 = DecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=256),
        returnn_vocab=label_datastream_bpe256.vocab,
        beam_size=1024,  # Untuned
        beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,  # Untuned
    )
    as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=256),
        returnn_vocab=label_datastream_bpe256.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        turn_off_quant="leave_as_is",
    )

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyCTCDecoderConfig

    as_training_greedy_decoder_config = GreedyCTCDecoderConfig(
        returnn_vocab=label_datastream_bpe256.vocab,
        turn_off_quant="leave_as_is",
    )

    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import (
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
    frontend_config_sub6 = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
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
    from ..ctc_phon.tune_eval import RTFArgs, RasrRTFArgs


    ####################################################################################################
    # QAT Baseline
    network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
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
        specauc_start_epoch=11,
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

    train_args = {
        "config": train_config,
        "network_module": network_module_v4,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v4 + f"_8_8_bpe"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    rtf_args = RTFArgs(
        beam_sizes=[256, 512, 1024, 4096],
        beam_size_tokens=[4, 8, 12, 20, 30],
        beam_thresholds=[4, 8, 20, 30],
        decoder_module="ctc.decoder.flashlight_ctc_v5_rescale_measure",
        include_gpu=True,
    )
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
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

    rtf_args_max_1 = RTFArgs(
        beam_sizes=[256, 512, 1024, 4096, 6144],
        beam_size_tokens=[4, 8, 12, 20, 30, 40, 100],
        beam_thresholds=[4, 8, 20, 30, 40],
        decoder_module="ctc.decoder.flashlight_ctc_v7_rescale_measure",
        include_gpu=False,
        forward_args={"max_seqs": 1}
    )
    results = {}
    results = eval_model(
        training_name=training_name + "_max_1",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        run_rtf=True,
        rtf_args=rtf_args_max_1,
    )

    rtf_args_max_1_v8 = RTFArgs(
        beam_sizes=[256, 512, 1024, 4096, 6144],
        beam_size_tokens=[4, 8, 12, 20, 30, 40, 100],
        beam_thresholds=[4, 8, 20, 30, 40],
        decoder_module="ctc.decoder.flashlight_ctc_v8_rescale_measure",
        include_gpu=False,
        forward_args={"max_seqs": 1},
        run_quant=True
    )
    results = {}
    results = eval_model(
        training_name=training_name + "_max_1_v8",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        run_rtf=True,
        rtf_args=rtf_args_max_1_v8,
    )

    # RASR SEARCH TRIAL 1
    results = {}
    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bpe_bliss_lexicon(bpe_size=256, add_blank=True),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=1024,
        score_threshold=14.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("4gram", get_bpe_bliss_lexicon(bpe_size=256, add_blank=True), scale=0.0),
    )

    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig

    as_training_rasr_config = DecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,  # this will be overwritten internally
        prior_file=None,
        turn_off_quant="leave_as_is",
    )
    rasr_rtf = RasrRTFArgs(
        max_beam_size=[128, 512, 1024, 2048, 4096, 8192, 16384],
        score_threshold=[4.0, 8.0, 10.0, 12.0, 14.0, 20.0, 30.0],
        decoder_module="ctc.decoder.rasr_ctc_v1_rescale_measure_v2",
        include_gpu=False,
        include_cpu=True,
        run_quant=True,
    )

    results = eval_model(
        training_name=training_name + "_rasr",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_rasr_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.rasr_ctc_v1",
        prior_scales=[0.5, 0.7],
        lm_scales=[0.6, 0.7, 0.8, 0.9],  # tuned
        run_rtf=True,
        rtf_args=rasr_rtf,
        run_rasr=True,
        run_best_4=False,
    )
    generate_report(results=results, exp_name=training_name + "_rasr")
    qat_report[training_name + "_rasr"] = results

    rtf_args_greedy = RTFArgs(
        beam_sizes=None,
        beam_size_tokens=None,
        beam_thresholds=None,
        decoder_module="ctc.decoder.greedy_bpe_ctc_rescale_measure_v2",
        type="greedy",
        include_gpu=True,
    )
    results = {}
    results = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
        run_rtf=True,
        rtf_args=rtf_args_greedy,
    )
    generate_report(results=results, exp_name=training_name + "_greedy")
    qat_report[training_name + "_greedy"] = results


    rtf_args_greedy = RTFArgs(
        beam_sizes=None,
        beam_size_tokens=None,
        beam_thresholds=None,
        decoder_module="ctc.decoder.greedy_bpe_ctc_rescale_measure_v2",
        type="greedy",
        include_gpu=True,
        forward_args={"max_seqs": 1}
    )
    results = {}
    results = eval_model(
        training_name=training_name + "/greedy" + "_max_1",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
        run_rtf=True,
        rtf_args=rtf_args_greedy,
    )

    # Neural LM
    from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v3 import DecoderConfig as BeamSearchDecoderConfig
    from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v4 import DecoderConfig as BeamSearchDecoderConfigv4
    from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v3 import DecoderExtraConfig
    from ... import PACKAGE

    from ...pytorch_networks.lm.kazuki_lstm_zijian_variant_v1_cfg import ModelConfig

    default_init_args = {
        "init_args_w": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
        "init_args_b": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
    }

    lm_config = ModelConfig(
        vocab_dim=vocab_size_without_blank,
        embed_dim=512,
        hidden_dim=2048,
        n_lstm_layers=2,
        use_bottle_neck=False,
        dropout=0.2,
        init_args=default_init_args,
    )
    lm = "/work/asr3/zyang/share/zhan/torch_setup/work/i6_core/returnn/training/ReturnnTrainingJob.qIIaRxZQmaBL/output/models/epoch.200.pt"
    lm_net_args = asdict(lm_config)

    rtf_lm = RTFArgs(
        beam_sizes=None,
        beam_size_tokens=None,
        beam_thresholds=None,
        include_gpu=True,
        type="nn_lm",
        decoder_module="ctc.decoder.beam_search_bpe_ctc_v4_rescale_measure_v3",
        include_cpu=False,
    )

    rtf_lm_max_1 = RTFArgs(
        beam_sizes=None,
        beam_size_tokens=None,
        beam_thresholds=None,
        include_gpu=True,
        type="nn_lm",
        decoder_module="ctc.decoder.beam_search_bpe_ctc_v4_rescale_measure_v3",
        include_cpu=False,
        forward_args={"max_seqs": 1},
    )

    for beam in [10, 30, 128, 200, 256, 300, 315]:
        beam_search_decoder_config_v4_lstmlm = BeamSearchDecoderConfigv4(
            returnn_vocab=label_datastream_bpe256.vocab,
            beam_size=beam,
            lm_model_args=lm_net_args,
            lm_checkpoint=lm,
            lm_module="pytorch_networks.lm.kazuki_lstm_zijian_variant_v1.Model",
            lm_states_need_label_axis=False,
        )
        decoder_unhashed_config_v3 = DecoderExtraConfig(
            lm_package=PACKAGE,
        )
        train_args["debug"] = True
        results = {}
        results = eval_model(
            training_name=training_name + f"/lstm_lm/{beam}",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=beam_search_decoder_config_v4_lstmlm,
            unhashed_decoder_args=decoder_unhashed_config_v3,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
            prior_scales=[0.3, 0.5, 0.7, 0.9, 1.0],
            lm_scales=[0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.4],
            # prior_scales=[0.5],
            # lm_scales=[2.0],
            with_prior=True,
            run_rtf=True,
            use_gpu=True,
            run_best=False,
            run_best_4=False,
            extra_forward_config={"batch_size": 200 * 16000},
            rtf_args=rtf_lm,
        )
        generate_report(results=results, exp_name=training_name + f"_lstm_lm_{beam}")
        qat_report[training_name + f"_lstm_lm_{beam}"] = results

        results = {}
        eval_model(
            training_name=training_name + f"/lstm_lm/{beam}_max_1",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=beam_search_decoder_config_v4_lstmlm,
            unhashed_decoder_args=decoder_unhashed_config_v3,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
            prior_scales=[0.3, 0.5, 0.7, 0.9, 1.0],
            lm_scales=[0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.4],
            # prior_scales=[0.5],
            # lm_scales=[2.0],
            with_prior=True,
            run_rtf=True,
            use_gpu=True,
            run_best=False,
            run_best_4=False,
            extra_forward_config={"batch_size": 200 * 16000},
            rtf_args=rtf_lm_max_1,
        )

    from ...pytorch_networks.lm.kazuki_trafo_zijian_variant_v1_cfg import (
        TransformerLMConfig,
        TransformerMHSAConfig,
        TransformerLinearConfig,
        TransformerBlockConfig,
    )

    hidden_dim = 768
    ff_dim = 4096
    input_dim = hidden_dim
    output_dim = hidden_dim
    linear_config = TransformerLinearConfig(
        input_dim=input_dim, ff_dim=ff_dim, output_dim=output_dim, dropout=0.0, batch_first=True
    )
    mhsa_config = TransformerMHSAConfig(input_dim=input_dim, num_heads=4, dropout=0.0, batch_first=True)
    block_config = TransformerBlockConfig(linear_config=linear_config, mhsa_config=mhsa_config)
    trafo_base_config = TransformerLMConfig(
        embed_dim=128,
        hidden_dim=hidden_dim,
        vocab_dim=vocab_size_without_blank,
        num_layers=12,
        block_config=block_config,
        batch_first=True,  # very important, state management in decoder does not work otherwise
        dropout=0.0,
    )
    lm = "/work/asr3/zyang/share/zhan/torch_setup/work/i6_core/returnn/training/ReturnnTrainingJob.JiXjvdOlsRMv/output/models/epoch.200.pt"
    lm_net_args = asdict(trafo_base_config)
    for beam in [32, 64, 128, 200, 256]:
        beam_search_decoder_config_v4_trafo = BeamSearchDecoderConfigv4(
            returnn_vocab=label_datastream_bpe256.vocab,
            beam_size=beam,
            lm_model_args=lm_net_args,
            lm_checkpoint=lm,
            lm_module="pytorch_networks.lm.kazuki_trafo_zijian_variant_v1_decoding.Model",
            lm_states_need_label_axis=True,
        )
        decoder_unhashed_config_v3 = DecoderExtraConfig(
            lm_package=PACKAGE,
        )
        train_args["debug"] = True
        results = {}
        eval_model(
            training_name=training_name + f"/trafo_lm/{beam}",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=beam_search_decoder_config_v4_trafo,
            unhashed_decoder_args=decoder_unhashed_config_v3,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
            prior_scales=[0.3, 0.5, 0.7, 0.9],
            lm_scales=[0.5, 0.7, 0.9, 1.0, 1.2],
            with_prior=True,
            run_rtf=True,
            rtf_args=rtf_lm,
            use_gpu=True,
            run_best=False,
            run_best_4=False,
            extra_forward_config={"batch_size": 150 * 16000} if beam <= 128 else {"max_seqs": 1},
        )
        generate_report(results=results, exp_name=training_name + f"_trafo_lm_{beam}")
        qat_report[training_name + f"_trafo_lm_{beam}"] = results

        results = {}
        eval_model(
            training_name=training_name + f"/trafo_lm/{beam}_max_1",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=beam_search_decoder_config_v4_trafo,
            unhashed_decoder_args=decoder_unhashed_config_v3,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
            prior_scales=[0.3, 0.5, 0.7, 0.9],
            lm_scales=[0.5, 0.7, 0.9, 1.0, 1.2],
            with_prior=True,
            run_rtf=True,
            rtf_args=rtf_lm_max_1,
            use_gpu=True,
            run_best=False,
            run_best_4=False,
            extra_forward_config={"batch_size": 150 * 16000} if beam <= 128 else {"max_seqs": 1},
        )

    from ...pytorch_networks.lm.kazuki_trafo_zijian_variant_v1_cfg import (
        TransformerLMConfig,
        TransformerMHSAConfig,
        TransformerLinearConfig,
        TransformerBlockConfig,
    )

    hidden_dim = 768
    ff_dim = 4096
    input_dim = hidden_dim
    output_dim = hidden_dim
    linear_config = TransformerLinearConfig(
        input_dim=input_dim, ff_dim=ff_dim, output_dim=output_dim, dropout=0.0, batch_first=True
    )
    mhsa_config = TransformerMHSAConfig(input_dim=input_dim, num_heads=4, dropout=0.0, batch_first=True)
    block_config = TransformerBlockConfig(linear_config=linear_config, mhsa_config=mhsa_config)
    trafo_base_config = TransformerLMConfig(
        embed_dim=128,
        hidden_dim=hidden_dim,
        vocab_dim=vocab_size_without_blank,
        num_layers=24,
        block_config=block_config,
        batch_first=True,  # very important, state management in decoder does not work otherwise
        dropout=0.0,
    )
    lm = "/work/asr3/zyang/share/zhan/torch_setup/work/i6_core/returnn/training/ReturnnTrainingJob.L1cJPJEMZufI/output/models/epoch.200.pt"
    lm_net_args = asdict(trafo_base_config)
    for beam in [32, 64, 128, 150]:
        beam_search_decoder_config_v4_trafo = BeamSearchDecoderConfigv4(
            returnn_vocab=label_datastream_bpe256.vocab,
            beam_size=beam,
            lm_model_args=lm_net_args,
            lm_checkpoint=lm,
            lm_module="pytorch_networks.lm.kazuki_trafo_zijian_variant_v1_decoding.Model",
            lm_states_need_label_axis=True,
        )
        decoder_unhashed_config_v3 = DecoderExtraConfig(
            lm_package=PACKAGE,
        )
        train_args["debug"] = True
        results = {}
        eval_model(
            training_name=training_name + f"/trafo_24l_lm/{beam}",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=beam_search_decoder_config_v4_trafo,
            unhashed_decoder_args=decoder_unhashed_config_v3,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
            prior_scales=[0.3, 0.5, 0.7, 0.9],
            lm_scales=[0.5, 0.7, 0.9, 1.0, 1.2],
            with_prior=True,
            run_rtf=beam < 100,
            rtf_args=rtf_lm if beam < 100 else None,
            use_gpu=True,
            run_best=False,
            run_best_4=False,
            extra_forward_config={"batch_size": 150 * 16000} if beam < 100 else {"max_seqs": 1},
        )
        generate_report(results=results, exp_name=training_name + f"_trafo_24l_lm_{beam}")
        qat_report[training_name + f"_trafo_24l_lm_{beam}"] = results

        results = {}
        eval_model(
            training_name=training_name + f"/trafo_24l_lm/{beam}_max_1",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=beam_search_decoder_config_v4_trafo,
            unhashed_decoder_args=decoder_unhashed_config_v3,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
            prior_scales=[0.3, 0.5, 0.7, 0.9],
            lm_scales=[0.5, 0.7, 0.9, 1.0, 1.2],
            with_prior=True,
            run_rtf=beam < 100,
            rtf_args=rtf_lm_max_1 if beam < 100 else None,
            use_gpu=True,
            run_best=False,
            run_best_4=False,
            extra_forward_config={"batch_size": 150 * 16000} if beam < 128 else {"max_seqs": 1},
        )

    #########################################################################################
    # Full Quant Baseline
    network_module_v1 = "ctc.qat_0711.full_qat_v1"
    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
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
        specauc_start_epoch=11,
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
    train_args = {
        "config": train_config,
        "network_module": network_module_v1,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v1 + f"_{8}_{8}"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        run_rtf=False,
        rtf_args=None,
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    results = {}
    results = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
        run_rtf=False,
    )
    generate_report(results=results, exp_name=training_name + "_greedy")
    qat_report[training_name + "_greedy"] = results

    for ff_dim in [512, 1024]:
        # ########################################################################
        # # FF 512 and 1024
        # frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
        #     in_features=80,
        #     conv1_channels=32,
        #     conv2_channels=64,
        #     conv3_channels=64,
        #     conv4_channels=32,
        #     conv_kernel_size=(3, 3),
        #     conv_padding=None,
        #     pool1_kernel_size=(3, 1),
        #     pool1_stride=(2, 1),
        #     pool1_padding=None,
        #     pool2_kernel_size=(2, 1),
        #     pool2_stride=(2, 1),
        #     pool2_padding=None,
        #     activation_str="ReLU",
        #     out_features=512,
        #     activation=None,
        # )
        #
        # model_config = QuantModelTrainConfigV4(
        #     feature_extraction_config=fe_config,
        #     frontend_config=frontend_config_sub6_512,
        #     specaug_config=specaug_config,
        #     label_target_size=vocab_size_without_blank,
        #     conformer_size=512,
        #     num_layers=12,
        #     num_heads=4,
        #     ff_dim=1024,
        #     att_weights_dropout=0.2,
        #     conv_dropout=0.2,
        #     ff_dropout=0.2,
        #     mhsa_dropout=0.2,
        #     conv_kernel_size=31,
        #     final_dropout=0.2,
        #     specauc_start_epoch=11,
        #     weight_quant_dtype="qint8",
        #     weight_quant_method="per_tensor",
        #     activation_quant_dtype="qint8",
        #     activation_quant_method="per_tensor",
        #     dot_quant_dtype="qint8",
        #     dot_quant_method="per_tensor",
        #     Av_quant_dtype="qint8",
        #     Av_quant_method="per_tensor",
        #     moving_average=None,
        #     weight_bit_prec=8,
        #     activation_bit_prec=8,
        #     quantize_output=False,
        #     extra_act_quant=False,
        #     quantize_bias=None,
        #     observer_only_in_train=False,
        # )
        # train_args = {
        #     "config": train_config,
        #     "network_module": network_module_v1,
        #     "net_args": {"model_config_dict": asdict(model_config)},
        #     "debug": False,
        #     "post_config": {"num_workers_per_gpu": 8},
        #     "use_speed_perturbation": True,
        # }
        #
        # training_name = prefix_name + "/" + network_module_v1 + f"_8_8_512_1024"
        # train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        # train_job.rqmt["gpu_mem"] = 48
        # results = {}
        # results = eval_model(
        #     training_name=training_name,
        #     train_job=train_job,
        #     train_args=train_args,
        #     train_data=train_data_bpe256,
        #     decoder_config=as_training_decoder_config,
        #     dev_dataset_tuples=dev_dataset_tuples,
        #     result_dict=results,
        #     decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        #     prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        #     lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        # )
        # generate_report(results=results, exp_name=training_name)
        # qat_report[training_name] = results
        #
        # results = {}
        # results = eval_model(
        #     training_name=training_name + "/greedy",
        #     train_job=train_job,
        #     train_args=train_args,
        #     train_data=train_data_bpe256,
        #     decoder_config=as_training_greedy_decoder_config,
        #     dev_dataset_tuples=dev_dataset_tuples,
        #     result_dict=results,
        #     decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        #     prior_scales=[0.0],
        #     lm_scales=[0.0],
        #     with_prior=False,
        # )
        # generate_report(results=results, exp_name=training_name + "_greedy")
        # qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
        train_args = {
            "config": train_config,
            "network_module": network_module_v1,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1 + f"_8_8_512_{ff_dim}"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results
        ########################################################################
        # FF 512 and 512 with mean abs
        network_module_v1_mean = "ctc.qat_0711.full_qat_v1_mean_abs_norm"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean + f"_8_8_512_{ff_dim}"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu
        network_module_v1_mean_relu = "ctc.qat_0711.full_qat_v1_relu_mean_abs"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # sym and means abs and ReLu and shared observers (faulty old)
        network_module_v1_mean_relu_shared_obs = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu_shared_obs,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu_shared_obs + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # sym and means abs and ReLu and shared observers
        network_module_v1_mean_relu_shared_obs_v2 = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs_v2"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu_shared_obs_v2,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu_shared_obs_v2 + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu 16 L
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=16,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym_16l"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu 20 L
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=20,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym_20l"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

    tk.register_report("reports/ted/qat_report_bpe_comparison", partial(build_qat_report, qat_report), required=qat_report)


def bpe_ted_qat_comparisons_new():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/bpe_ctc_bpe/256/qat_comparison_new"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe256 = build_bpe_training_datasets(
        prefix=prefix_name,
        bpe_size=256,  # TODO tune
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe256 = cast(LabelDatastream, train_data_bpe256.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe256.vocab_size

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

    default_decoder_config_bpe256 = DecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=256),
        returnn_vocab=label_datastream_bpe256.vocab,
        beam_size=1024,  # Untuned
        beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,  # Untuned
    )
    as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=256),
        returnn_vocab=label_datastream_bpe256.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        turn_off_quant="leave_as_is",
    )

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyCTCDecoderConfig

    as_training_greedy_decoder_config = GreedyCTCDecoderConfig(
        returnn_vocab=label_datastream_bpe256.vocab,
        turn_off_quant="leave_as_is",
    )

    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import (
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
    frontend_config_sub4 = VGG4LayerActFrontendV1Config_mod(
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
    from ..ctc_phon.tune_eval import RTFArgs, RasrRTFArgs


    ####################################################################################################
    # QAT Baseline
    network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
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
        specauc_start_epoch=11,
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

    train_args = {
        "config": train_config,
        "network_module": network_module_v4,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v4 + f"_8_8_bpe"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    rtf_args = RTFArgs(
        beam_sizes=[256, 512, 1024, 4096],
        beam_size_tokens=[4, 8, 12, 20, 30],
        beam_thresholds=[4, 8, 20, 30],
        decoder_module="ctc.decoder.flashlight_ctc_v7_rescale_measure",
        include_gpu=False,
    )
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        run_rtf=False,
        rtf_args=None,
    )

    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    # RASR SEARCH TRIAL 1
    results = {}
    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bpe_bliss_lexicon(bpe_size=256, add_blank=True),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=1024,
        score_threshold=14.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("4gram", get_bpe_bliss_lexicon(bpe_size=256, add_blank=True), scale=0.0),
    )

    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig

    as_training_rasr_config = DecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,  # this will be overwritten internally
        prior_file=None,
        turn_off_quant="leave_as_is",
    )
    rasr_rtf = RasrRTFArgs(
        max_beam_size=[128, 512, 1024, 2048, 4096, 8192, 16384],
        score_threshold=[4.0, 8.0, 10.0, 12.0, 14.0, 20.0, 30.0],
        decoder_module="ctc.decoder.rasr_ctc_v1_rescale_measure_v2",
        include_gpu=False,
        include_cpu=True,
        run_quant=True,
    )

    results = eval_model(
        training_name=training_name + "_rasr",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_rasr_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.rasr_ctc_v1",
        prior_scales=[0.5, 0.7],
        lm_scales=[0.6, 0.7, 0.8, 0.9],  # tuned
        run_rtf=False,
        rtf_args=None,
        run_rasr=True,
        run_best_4=False,
    )
    generate_report(results=results, exp_name=training_name + "_rasr")
    qat_report[training_name + "_rasr"] = results

    #########################################################################################
    # Full Quant Baseline
    network_module_v1 = "ctc.qat_0711.full_qat_v1"
    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
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
        specauc_start_epoch=11,
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
    train_args = {
        "config": train_config,
        "network_module": network_module_v1,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v1 + f"_{8}_{8}"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        run_rtf=False,
        rtf_args=None,
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    results = {}
    results = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
        run_rtf=False,
    )
    generate_report(results=results, exp_name=training_name + "_greedy")
    qat_report[training_name + "_greedy"] = results

    return
    for ff_dim in [512, 1024]:
        # ########################################################################
        # # FF 512 and 1024
        # frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
        #     in_features=80,
        #     conv1_channels=32,
        #     conv2_channels=64,
        #     conv3_channels=64,
        #     conv4_channels=32,
        #     conv_kernel_size=(3, 3),
        #     conv_padding=None,
        #     pool1_kernel_size=(3, 1),
        #     pool1_stride=(2, 1),
        #     pool1_padding=None,
        #     pool2_kernel_size=(2, 1),
        #     pool2_stride=(2, 1),
        #     pool2_padding=None,
        #     activation_str="ReLU",
        #     out_features=512,
        #     activation=None,
        # )
        #
        # model_config = QuantModelTrainConfigV4(
        #     feature_extraction_config=fe_config,
        #     frontend_config=frontend_config_sub6_512,
        #     specaug_config=specaug_config,
        #     label_target_size=vocab_size_without_blank,
        #     conformer_size=512,
        #     num_layers=12,
        #     num_heads=4,
        #     ff_dim=1024,
        #     att_weights_dropout=0.2,
        #     conv_dropout=0.2,
        #     ff_dropout=0.2,
        #     mhsa_dropout=0.2,
        #     conv_kernel_size=31,
        #     final_dropout=0.2,
        #     specauc_start_epoch=11,
        #     weight_quant_dtype="qint8",
        #     weight_quant_method="per_tensor",
        #     activation_quant_dtype="qint8",
        #     activation_quant_method="per_tensor",
        #     dot_quant_dtype="qint8",
        #     dot_quant_method="per_tensor",
        #     Av_quant_dtype="qint8",
        #     Av_quant_method="per_tensor",
        #     moving_average=None,
        #     weight_bit_prec=8,
        #     activation_bit_prec=8,
        #     quantize_output=False,
        #     extra_act_quant=False,
        #     quantize_bias=None,
        #     observer_only_in_train=False,
        # )
        # train_args = {
        #     "config": train_config,
        #     "network_module": network_module_v1,
        #     "net_args": {"model_config_dict": asdict(model_config)},
        #     "debug": False,
        #     "post_config": {"num_workers_per_gpu": 8},
        #     "use_speed_perturbation": True,
        # }
        #
        # training_name = prefix_name + "/" + network_module_v1 + f"_8_8_512_1024"
        # train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        # train_job.rqmt["gpu_mem"] = 48
        # results = {}
        # results = eval_model(
        #     training_name=training_name,
        #     train_job=train_job,
        #     train_args=train_args,
        #     train_data=train_data_bpe256,
        #     decoder_config=as_training_decoder_config,
        #     dev_dataset_tuples=dev_dataset_tuples,
        #     result_dict=results,
        #     decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        #     prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        #     lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        # )
        # generate_report(results=results, exp_name=training_name)
        # qat_report[training_name] = results
        #
        # results = {}
        # results = eval_model(
        #     training_name=training_name + "/greedy",
        #     train_job=train_job,
        #     train_args=train_args,
        #     train_data=train_data_bpe256,
        #     decoder_config=as_training_greedy_decoder_config,
        #     dev_dataset_tuples=dev_dataset_tuples,
        #     result_dict=results,
        #     decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        #     prior_scales=[0.0],
        #     lm_scales=[0.0],
        #     with_prior=False,
        # )
        # generate_report(results=results, exp_name=training_name + "_greedy")
        # qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
        train_args = {
            "config": train_config,
            "network_module": network_module_v1,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1 + f"_8_8_512_{ff_dim}"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results
        ########################################################################
        # FF 512 and 512 with mean abs
        network_module_v1_mean = "ctc.qat_0711.full_qat_v1_mean_abs_norm"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean + f"_8_8_512_{ff_dim}"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu
        network_module_v1_mean_relu = "ctc.qat_0711.full_qat_v1_relu_mean_abs"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # sym and means abs and ReLu and shared observers (faulty old)
        network_module_v1_mean_relu_shared_obs = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu_shared_obs,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu_shared_obs + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # sym and means abs and ReLu and shared observers
        network_module_v1_mean_relu_shared_obs_v2 = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs_v2"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu_shared_obs_v2,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu_shared_obs_v2 + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu 16 L
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=16,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym_16l"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu 20 L
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,
            pool1_kernel_size=(3, 1),
            pool1_stride=(2, 1),
            pool1_padding=None,
            pool2_kernel_size=(2, 1),
            pool2_stride=(2, 1),
            pool2_padding=None,
            activation_str="ReLU",
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=20,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym_20l"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

    tk.register_report("reports/qat_report_bpe_comparison_new", partial(build_qat_report, qat_report), required=qat_report)
