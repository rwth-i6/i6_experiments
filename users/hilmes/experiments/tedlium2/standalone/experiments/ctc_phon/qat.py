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
from .tune_eval import tune_and_evaluate_helper, eval_model, build_report, build_qat_report
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

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v1_cfg import (
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
        turn_off_quant="leave_as_is",
    )
    network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

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
    from ...pytorch_networks.ctc.qat_0711.memristor_v1_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV1
    from ...pytorch_networks.ctc.qat_0711.memristor_v4_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV4
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

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
            )
            generate_report(results=results, exp_name=training_name)
            qat_report[training_name] = results
            if weight_bit == 4:
                training_name = prefix_name + "/" + network_module_mem_v1 + f"_{weight_bit}_{activation_bit}_memristor"
                results = {}
                results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data_4k,
                    decoder_config=default_decoder_config,
                    dev_dataset_tuples=dev_dataset_tuples,
                    result_dict=results,
                    decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                    prior_scales=[0.5],  # [0.5, 0.7],
                    lm_scales=[2.0],  # , 2.2, 2.4, 2.6],
                    use_gpu=True,
                    import_memristor=True,
                    extra_forward_config={
                        "batch_size": 1000000,
                    },
                    run_best=False,
                    run_best_4=False,
                )
                generate_report(results=results, exp_name=training_name)
                qat_report[training_name] = results

            train_args = {
                "config": train_config,
                "network_module": network_module_mem_v3,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            training_name = prefix_name + "/" + network_module_mem_v3 + f"_{weight_bit}_{activation_bit}_memristor"
            results = {}
            results = eval_model(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args,
                train_data=train_data_4k,
                decoder_config=default_decoder_config,
                dev_dataset_tuples=dev_dataset_tuples,
                result_dict=results,
                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                prior_scales=[0.5],  # [0.5, 0.7],
                lm_scales=[2.0],  # , 2.2, 2.4, 2.6],
                use_gpu=True,
                import_memristor=True,
                extra_forward_config={
                    "batch_size": 1000000,
                },
                run_best=False,
                run_best_4=False,
            )
            generate_report(results=results, exp_name=training_name)
            qat_report[training_name] = results

            res_cyc = {}
            for num_cycle in range(1, 11):
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
                    "debug": True,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }

                training_name = prefix_name + "/" + network_module_mem_v4 + f"_{weight_bit}_{activation_bit}_cycle"
                res_cyc = eval_model(
                    training_name=training_name + f"{num_cycle}",
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data_4k,
                    decoder_config=default_decoder_config,
                    dev_dataset_tuples=dev_dataset_tuples,
                    result_dict=None,
                    decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                    prior_scales=[0.5],
                    lm_scales=[2.0],
                    prior_args=prior_args,
                    run_best=False,
                    run_best_4=False,
                    import_memristor=not train_args["debug"],
                    use_gpu=True,
                )
            generate_report(results=res_cyc, exp_name=training_name)
            qat_report[training_name] = res_cyc

            if weight_bit in [4, 6, 8]:
                for prec_bit, range_bit in [(8, 8), (8, 6), (6, 8), (6, 6), (4, 6), (4, 4), (4, 2), (2, 2), (2, 1)]:
                    dac_settings = DacAdcHardwareSettings(
                        input_bits=activation_bit,
                        output_precision_bits=prec_bit,
                        output_range_bits=range_bit,
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
                        "network_module": network_module_mem_v3,
                        "net_args": {"model_config_dict": asdict(model_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }
                    prior_dac_settings = DacAdcHardwareSettings(
                        input_bits=0,
                        output_precision_bits=0,
                        output_range_bits=0,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    prior_model_config = MemristorModelTrainConfigV1(
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
                        converter_hardware_settings=prior_dac_settings,
                        quant_in_linear=True,
                    )
                    prior_args = {
                        "config": train_config,
                        "network_module": network_module_mem_v3,
                        "net_args": {"model_config_dict": asdict(prior_model_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }
                    training_name = (
                        prefix_name
                        + "/"
                        + network_module_mem_v3
                        + f"_{weight_bit}_{activation_bit}_memristor_p{prec_bit}_r{range_bit}"
                    )
                    results = {}
                    results = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data_4k,
                        decoder_config=default_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        result_dict=results,
                        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                        prior_scales=[0.5],  # TODO 0.7
                        lm_scales=[2.0],
                        use_gpu=True,
                        import_memristor=True,
                        extra_forward_config={
                            "batch_size": 1000000,
                        },
                        run_best_4=False,
                        run_best=False,
                        prior_args=prior_args,
                    )
                    generate_report(results=results, exp_name=training_name)
                    qat_report[training_name] = results

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
                    ff_dim=768,
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

                training_name = prefix_name + "/" + network_module_mem_v1 + f"_{weight_bit}_{activation_bit}_smaller"
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

                for prec_bit, range_bit in [
                    (8, 10),
                    (8, 8),
                    (8, 6),
                    (6, 8),
                    (6, 6),
                    (4, 6),
                    (4, 4),
                    (4, 2),
                    (2, 2),
                    (2, 1),
                ]:
                    dac_settings = DacAdcHardwareSettings(
                        input_bits=activation_bit,
                        output_precision_bits=prec_bit,
                        output_range_bits=range_bit,
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
                        ff_dim=2 * 384,
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
                    training_name = (
                        prefix_name
                        + "/"
                        + network_module_mem_v1
                        + f"_{weight_bit}_{activation_bit}_smaller_memristor_p{prec_bit}_r{range_bit}"
                    )
                    results = {}
                    results = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data_4k,
                        decoder_config=default_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        result_dict=results,
                        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                        prior_scales=[0.5, 0.7],
                        lm_scales=[2.0, 2.2, 2.4, 2.6],
                        use_gpu=True,
                        import_memristor=True,
                        extra_forward_config={
                            "batch_size": 500000,
                        },
                        run_best_4=False,
                        run_best=False,
                    )
                    generate_report(results=results, exp_name=training_name)
                    qat_report[training_name] = results

    tk.register_report("reports/qat_report", partial(build_qat_report, qat_report), required=qat_report)
