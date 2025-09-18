from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast
import os

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search
from ...report import generate_report, build_qat_report

from .tune_eval import tune_and_evaluate_helper, eval_model
from functools import partial
from sisyphus import tk
import numpy as np
from i6_core.report.report import GenerateReportStringJob, MailJob, _Report_Type
import copy
from typing import Dict
from i6_core.util import instanciate_delayed


def eow_phon_ls960_0725_memristor():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon_memristor"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
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

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.flashlight_qat_phoneme_ctc import DecoderConfig as DecoderConfigMemristor

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    decoder_config_memristor = DecoderConfigMemristor(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    decoder_config_no_memristor = copy.deepcopy(decoder_config_memristor)
    decoder_config_no_memristor.turn_off_quant = "leave_as_is"

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

    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Normal Style
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

    memristor_report = {}
    # Normal QAT
    base_network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
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


    train_config_24gbgpu = {
        "optimizer": {
            "class": "radam",
            "epsilon": 1e-12,
            "weight_decay": 1e-2,
            "decoupled_weight_decay": True,
        },
        "learning_rates": list(np.linspace(7e-6, 5e-4, 480))
                          + list(np.linspace(5e-4, 5e-5, 480))
                          + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
    }

    train_args_base_qat = {
        "config": train_config_24gbgpu,
        "network_module": base_network_module_v4,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 8}
    }

    name = ".512dim_sub4_48gbgpu_100eps_radam_bs360_sp_8_8"
    training_name = prefix_name + "/" + base_network_module_v4 + name
    train_job = training(training_name, train_data, train_args_base_qat, num_epochs=1000, **default_returnn)
    if not os.path.exists(
        f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
        train_job.hold()
        train_job.move_to_hpc = True
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args_base_qat,
        train_data=train_data,
        decoder_config=decoder_config_no_memristor,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
    )
    generate_report(results=results, exp_name=training_name + "/non_memristor")
    memristor_report[training_name] = results

    network_module_mem_v7 = "ctc.qat_0711.memristor_v7"
    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    for activation_bit in [8]:
        for weight_bit in [3, 4, 5, 6, 7, 8]:
            prior_train_dac_settings = DacAdcHardwareSettings(
                input_bits=0,
                output_precision_bits=0,
                output_range_bits=0,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            model_config = MemristorModelTrainConfigV7(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config_full,
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
                converter_hardware_settings=prior_train_dac_settings,
                quant_in_linear=True,
                num_cycles=0,
                correction_settings=None,
                weight_noise_func=None,
                weight_noise_values=None,
                weight_noise_start_epoch=None,
            )
            res_seeds_total = {}
            for seed in range(3):
                train_config_24gbgpu = {
                    "optimizer": {
                        "class": "radam",
                        "epsilon": 1e-12,
                        "weight_decay": 1e-2,
                        "decoupled_weight_decay": True,
                    },
                    "learning_rates": list(np.linspace(7e-6, 5e-4, 480))
                                      + list(np.linspace(5e-4, 5e-5, 480))
                                      + list(np.linspace(5e-5, 1e-7, 40)),
                    #############
                    "batch_size": 360 * 16000,
                    "max_seq_length": {"audio_features": 35 * 16000},
                    "accum_grad_multiple_step": 1,
                    "gradient_clip_norm": 1.0,
                    "seed": seed,
                }
                train_args = {
                    "config": train_config_24gbgpu,
                    "network_module": network_module_mem_v7,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }
                training_name = prefix_name + "/" + network_module_mem_v7 + f"_{weight_bit}_{activation_bit}_seed_{seed}"
                train_job = training(training_name, train_data, train_args, num_epochs=1000, **default_returnn)
                if not os.path.exists(
                    f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                    train_job.hold()
                    train_job.move_to_hpc = True

                results = {}
                results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data,
                    decoder_config=decoder_config_no_memristor,
                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                    result_dict=results,
                    decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                    prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
                    lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                    import_memristor=True,
                )
                generate_report(results=results, exp_name=training_name + "/non_memristor")
                memristor_report[training_name] = results

                res_conv = {}
                res_smaller = {}
                for num_cycles in range(1, 11):
                    recog_dac_settings = DacAdcHardwareSettings(
                        input_bits=8,
                        output_precision_bits=4,
                        output_range_bits=4,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    model_config_recog = copy.deepcopy(model_config)
                    model_config_recog.converter_hardware_settings = recog_dac_settings
                    model_config_recog.num_cycles = num_cycles


                    prior_args = {
                        "config": train_config_24gbgpu,
                        "network_module": network_module_mem_v7,
                        "net_args": {"model_config_dict": asdict(model_config)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }

                    train_args_recog = {
                        "config": train_config_24gbgpu,
                        "network_module": network_module_mem_v7,
                        "net_args": {"model_config_dict": asdict(model_config_recog)},
                        "debug": False,
                        "post_config": {"num_workers_per_gpu": 8},
                        "use_speed_perturbation": True,
                    }
                    recog_name = prefix_name + "/" + network_module_mem_v7 + f"_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                    res_conv = eval_model(
                        training_name=recog_name  + f"_{num_cycles}",
                        train_job=train_job,
                        train_args=train_args_recog,
                        train_data=train_data,
                        decoder_config=decoder_config_memristor,
                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                        result_dict=res_conv,
                        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                        prior_scales=[0.3],  # TODO 0.7
                        lm_scales=[2.0],
                        use_gpu=True,
                        import_memristor=True,
                        extra_forward_config={
                            "batch_size": 7000000 * 2,
                        },
                        run_best_4=False,
                        run_best=False,
                        prior_args=prior_args,
                        run_search_on_hpc=True,
                    )
                    res_seeds_total.update(res_conv)
                    if num_cycles % 10 == 0 and num_cycles > 0:
                        generate_report(results=res_conv, exp_name=recog_name)
                        memristor_report[recog_name] = copy.deepcopy(res_conv)

                    if weight_bit <= 4:

                        recog_name = prefix_name + "/" + network_module_mem_v7 + f"_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}_smaller_batch"
                        res_smaller = eval_model(
                            training_name=recog_name  + f"_{num_cycles}",
                            train_job=train_job,
                            train_args=train_args_recog,
                            train_data=train_data,
                            decoder_config=decoder_config_memristor,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            result_dict=res_smaller,
                            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                            prior_scales=[0.3],  # TODO 0.7
                            lm_scales=[2.0],
                            use_gpu=True,
                            import_memristor=True,
                            extra_forward_config={
                                "batch_size": 200 * 16000,
                            },
                            run_best_4=False,
                            run_best=False,
                            prior_args=prior_args,
                            run_search_on_hpc=True,
                        )
                        res_seeds_total.update(res_smaller)
                        if num_cycles % 10 == 0 and num_cycles > 0:
                            generate_report(results=res_smaller, exp_name=recog_name)
                            memristor_report[recog_name] = copy.deepcopy(res_smaller)


            training_name = (
                prefix_name
                + "/"
                + network_module_mem_v7
                + f"_{weight_bit}_{activation_bit}_seeds_combined_cycle"
            )
            generate_report(results=res_seeds_total, exp_name=training_name)
            memristor_report[training_name] = copy.deepcopy(res_seeds_total)

    tk.register_report("reports/memristor_report_phon_lbs", partial(build_qat_report, memristor_report), required=memristor_report)
