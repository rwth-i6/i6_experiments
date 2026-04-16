import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Dict
import os


from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training
from ...report import generate_report
from ..ctc_phon.tune_eval import eval_model, build_qat_report
from functools import partial


def bpe_ted_0725_memristor():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/bpe_ctc_bpe/256/memristor"

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
    frontend_config_sub4 = VGG4LayerActFrontendV1Config_mod(
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
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )

    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7

    network_module_mem_v7 = "ctc.qat_0711.memristor_v7"
    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    memristor_report = {}
    for activation_bit in [8]:
        dac_settings_lower = DacAdcHardwareSettings(
            input_bits=activation_bit,
            output_precision_bits=4,
            output_range_bits=4,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
        for weight_bit in [3, 4, 5, 6, 7, 8]:
            res_seeds_total = {}
            res_seeds_greedy = {}
            for seed in range(3):
                model_config = MemristorModelTrainConfigV7(
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
                    "learning_rates": list(np.linspace(7e-5, 7e-4, 110))
                    + list(np.linspace(7e-4, 7e-5, 110))
                    + list(np.linspace(7e-5, 1e-7, 30)),
                    #############
                    "batch_size": 180 * 16000,
                    "max_seq_length": {"audio_features": 35 * 16000},
                    "accum_grad_multiple_step": 1,
                    "gradient_clip_norm": 1.0,
                    "seed": seed,  # random param, this does nothing but generate a new hash!!!
                }
                train_args = {
                    "config": train_config,
                    "network_module": network_module_mem_v7,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }

                training_name = (
                    prefix_name + "/" + network_module_mem_v7 + f"_{weight_bit}_{activation_bit}_seed_{seed}"
                )
                train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
                if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                    train_job.hold()
                    train_job.move_to_hpc = True

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
                memristor_report[training_name] = results
                if seed == 0:
                    memristor_report[
                        prefix_name
                        + "/"
                        + network_module_mem_v7
                        + f"_{weight_bit}_{activation_bit}_correction_baseline"
                    ] = results

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
                    import_memristor=True,
                )
                generate_report(results=results, exp_name=training_name + "/greedy" + "/non_memristor")
                memristor_report[training_name + "_greedy"] = results

                res_conv = {}
                results_greedy = {}
                for num_cycle in range(1, 11):
                    if weight_bit in [1.5]:
                        continue
                    model_config = MemristorModelTrainConfigV7(
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
                        correction_settings=None,
                        weight_noise_func=None,
                        weight_noise_values=None,
                        weight_noise_start_epoch=None,
                    )

                    train_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v7,
                        "net_args": {"model_config_dict": asdict(model_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }

                    prior_config = MemristorModelTrainConfigV7(
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
                        correction_settings=None,
                        weight_noise_func=None,
                        weight_noise_values=None,
                        weight_noise_start_epoch=None,
                    )
                    prior_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v7,
                        "net_args": {"model_config_dict": asdict(prior_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }

                    training_name = (
                        prefix_name
                        + "/"
                        + network_module_mem_v7
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
                            "batch_size": 7000000 * 4,
                        },
                        run_search_on_hpc=True,
                    )
                    res_seeds_total.update(res_conv)

                    results_greedy = eval_model(
                        training_name=training_name + f"_{num_cycle}" + "_greedy",
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data_bpe256,
                        decoder_config=as_training_greedy_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        result_dict=results_greedy,
                        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
                        prior_scales=[0.0],
                        lm_scales=[0.0],
                        with_prior=False,
                        import_memristor=True,
                        run_best=False,
                        run_best_4=False,
                        use_gpu=True,
                        extra_forward_config={
                            "batch_size": 7000000 * 4,
                        },
                        run_search_on_hpc=True,
                    )
                    res_seeds_greedy.update(results_greedy)

                    if num_cycle % 10 == 0 and num_cycle > 0:
                        generate_report(results=res_conv, exp_name=training_name)
                        memristor_report[training_name] = copy.deepcopy(res_conv)
                        if seed == 0:
                            memristor_report[
                                prefix_name
                                + "/"
                                + network_module_mem_v7
                                + f"_{weight_bit}_{activation_bit}_no_correction/cycle_combined"
                            ] = copy.deepcopy(res_conv)
                        generate_report(results=results_greedy, exp_name=training_name + "_greedy")
                        memristor_report[training_name + "_greedy"] = copy.deepcopy(results_greedy)

                if seed == 0 and weight_bit in [3, 4, 8]:
                    for num_cycles_correction in [1, 20]:
                        for test_input_value in [0.6, 0.4]:
                            for relative_deviation in [0.025, 0.05, 0.1]:
                                cycle_correction_settings = CycleCorrectionSettings(
                                    num_cycles=num_cycles_correction,
                                    test_input_value=test_input_value,
                                    relative_deviation=relative_deviation,
                                )
                                res_conv = {}
                                for num_cycle in range(1, 11):
                                    if weight_bit in [1.5]:
                                        continue
                                    model_config = MemristorModelTrainConfigV7(
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
                                        correction_settings=cycle_correction_settings,
                                        weight_noise_func=None,
                                        weight_noise_values=None,
                                        weight_noise_start_epoch=None,
                                    )
                                    train_args = {
                                        "config": train_config,
                                        "network_module": network_module_mem_v7,
                                        "net_args": {"model_config_dict": asdict(model_config)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }

                                    prior_config = MemristorModelTrainConfigV7(
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
                                        correction_settings=None,
                                        weight_noise_func=None,
                                        weight_noise_values=None,
                                        weight_noise_start_epoch=None,
                                    )
                                    prior_args = {
                                        "config": train_config,
                                        "network_module": network_module_mem_v7,
                                        "net_args": {"model_config_dict": asdict(prior_config)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }

                                    training_name = (
                                        prefix_name
                                        + "/"
                                        + network_module_mem_v7
                                        + f"_{weight_bit}_{activation_bit}_correction_{num_cycles_correction}_{test_input_value}_{relative_deviation}/cycle_{num_cycle // 11}"
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
                                            "batch_size": 7000000 * 4,
                                        },
                                        run_search_on_hpc=True,
                                    )
                                    # res_seeds_total.update(res_conv)

                                    # results_greedy = eval_model(
                                    #     training_name=training_name + f"_{num_cycle}" + "_greedy",
                                    #     train_job=train_job,
                                    #     train_args=train_args,
                                    #     train_data=train_data_bpe256,
                                    #     decoder_config=as_training_greedy_decoder_config,
                                    #     dev_dataset_tuples=dev_dataset_tuples,
                                    #     result_dict=results_greedy,
                                    #     decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
                                    #     prior_scales=[0.0],
                                    #     lm_scales=[0.0],
                                    #     with_prior=False,
                                    #     import_memristor=True,
                                    #     run_best=False,
                                    #     run_best_4=False,
                                    #     use_gpu=True,
                                    #     extra_forward_config={
                                    #         "batch_size": 7000000 * 4,
                                    #     },
                                    #     run_search_on_hpc=True,
                                    # )
                                    # res_seeds_greedy.update(results_greedy)

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_mem_v7
                                    + f"_{weight_bit}_{activation_bit}_correction_{num_cycles_correction}_{test_input_value}_{relative_deviation}/cycle_combined"
                                )
                                generate_report(results=res_conv, exp_name=training_name)
                                memristor_report[training_name] = copy.deepcopy(res_conv)
                                # generate_report(results=results_greedy, exp_name=training_name + "_greedy")
                                # memristor_report[training_name + "_greedy"] = copy.deepcopy(results_greedy)

            training_name = (
                prefix_name + "/" + network_module_mem_v7 + f"_{weight_bit}_{activation_bit}_seeds_combined_cycle"
            )
            generate_report(results=res_seeds_total, exp_name=training_name)
            memristor_report[training_name] = copy.deepcopy(res_seeds_total)
            generate_report(results=res_seeds_greedy, exp_name=training_name + "_greedy")
            memristor_report[training_name + "_greedy"] = copy.deepcopy(res_seeds_greedy)

    tk.register_report(
        "reports/ted/memristor_report_bpe",
        partial(build_qat_report, memristor_report),
        required=memristor_report,
        update_frequency=10,
    )
