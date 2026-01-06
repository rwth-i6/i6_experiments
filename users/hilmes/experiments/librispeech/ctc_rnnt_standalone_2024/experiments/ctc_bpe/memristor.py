from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast
import os

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm, get_arpa_lm_config
from ...pipeline import training, prepare_asr_model, search
from ...report import generate_report, build_qat_report

from ..ctc_phon.tune_eval import tune_and_evaluate_helper, eval_model
from functools import partial
from sisyphus import tk
import numpy as np
from i6_core.report.report import GenerateReportStringJob, MailJob, _Report_Type
import copy
from typing import Dict
from i6_core.util import instanciate_delayed
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon, get_bpe_bliss_lexicon

def bpe_ls960_1225_memristor():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/bpe_ls960_memristor"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe128 = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=128,
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe128 = cast(LabelDatastream, train_data_bpe128.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe128.vocab_size

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

    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, librispeech_key="train-other-960"),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=2048,
        score_threshold=18.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("4gram", lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, librispeech_key="train-other-960"), scale=0.0),
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
    rasr_lm_scales = [0.9, 1.0, 1.1, 1.2]

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

    from ...pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ModelConfig as RelPosModelConfigV1,
        ConformerPosEmbConfig,
    )

    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    memristor_report = {}
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
    baseline_network_module = "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"

    model_config = RelPosModelConfigV1(
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
        specauc_start_epoch=11,
        pos_emb_config=pos_emb_cfg,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
        dropout_broadcast_axes=None,
        mhsa_with_bias=True,
    )
    train_args_base = {
        "config": train_config_24gbgpu,
        "network_module": baseline_network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 8}
    }
    name = ".baseline_512dim_sub4_48gbgpu_100eps_radam_bs360_sp"
    training_name = prefix_name + "/" + baseline_network_module + name
    train_job = training(training_name, train_data_bpe128, train_args_base, num_epochs=1000, **default_returnn)

    if not os.path.exists(
        f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
        train_job.hold()
        train_job.move_to_hpc = True
    results = {}
    results, best_params_job = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args_base,
        train_data=train_data_bpe128,
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
    generate_report(results=results, exp_name=training_name)
    memristor_report[training_name] = results

    network_module_mem_v7 = "ctc.qat_0711.memristor_v7"
    network_module_mem_v10 = "ctc.qat_0711.memristor_v10"
    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8

    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    for activation_bit in [8]:
        for epochs in [1000]:
            for dim in [512]:
                for weight_bit in [4, 5, 8]:
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
                    model_config = MemristorModelTrainConfigV8(
                        feature_extraction_config=fe_config,
                        frontend_config=frontend_config_dim,
                        specaug_config=specaug_config_full,
                        label_target_size=vocab_size_without_blank,
                        conformer_size=dim,
                        num_layers=12,
                        num_heads=8,
                        ff_dim=dim * 4,
                        att_weights_dropout=0.1,
                        conv_dropout=0.1,
                        ff_dropout=0.1,
                        mhsa_dropout=0.1,
                        conv_kernel_size=31,
                        final_dropout=0.1,
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
                    for seed in range(1):
                        train_config_24gbgpu = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-12,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                              + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
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
                            "network_module": network_module_mem_v10,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }
                        training_name = prefix_name + "/" + network_module_mem_v10 + f"_{epochs//10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}"
                        train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)
                        if not os.path.exists(
                            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.rqmt['cpu'] = 8
                            train_job.hold()
                            train_job.move_to_hpc = True

                        results = {}
                        results, best_params_job = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data_bpe128,
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
                                "network_module": network_module_mem_v10,
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
                            recog_name = prefix_name + "/" + network_module_mem_v10 + f"_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                            res_conv = eval_model(
                                training_name=recog_name + f"_{num_cycles}",
                                train_job=train_job,
                                train_args=train_args_recog,
                                train_data=train_data_bpe128,
                                decoder_config=rasr_config_memristor,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                result_dict=res_conv,
                                decoder_module="ctc.decoder.rasr_ctc_v1",
                                prior_scales=[best_params_job.out_optimal_parameters[1]],
                                lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                use_gpu=True,
                                import_memristor=True,
                                extra_forward_config={
                                    "batch_size": 7000000 * 2,
                                },
                                run_best_4=False,
                                run_best=False,
                                prior_args=prior_args,
                                run_search_on_hpc=False,
                                run_rasr=True,
                                split_mem_init=True,
                            )
                            res_seeds_total.update(res_conv)
                            if num_cycles % 10 == 0 and num_cycles > 0:
                                generate_report(results=res_conv, exp_name=recog_name)
                                memristor_report[recog_name] = copy.deepcopy(res_conv)

                    training_name = (
                        prefix_name
                        + "/"
                        + network_module_mem_v10
                        + f"_{epochs//10}eps_{dim}dim_{weight_bit}_{activation_bit}_seeds_combined_cycle"
                    )
                    generate_report(results=res_seeds_total, exp_name=training_name)
                    memristor_report[training_name] = copy.deepcopy(res_seeds_total)

    tk.register_report("reports/lbs/memristor_report_bpe", partial(build_qat_report, memristor_report),
        required=memristor_report, update_frequency=400)


