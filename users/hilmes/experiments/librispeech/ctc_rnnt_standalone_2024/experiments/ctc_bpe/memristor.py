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
from ...report import generate_report, build_qat_report, build_qat_report_v2

from ..ctc_phon.tune_eval import tune_and_evaluate_helper, eval_model
from functools import partial
from sisyphus import tk
import numpy as np
from i6_core.report.report import GenerateReportStringJob, MailJob, _Report_Type
import copy
from typing import Dict
from i6_core.util import instanciate_delayed
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon, get_bpe_bliss_lexicon

def run_memristor_cycle_eval(
    train_job, train_data, train_config, model_config,
    recog_name_prefix, rasr_config, greedy_config, dev_dataset_tuples,
    prior_scales, lm_scales, batch_size, max_runs, report_dict,
    prior_network_module, recog_network_module,
    recog_model_config_class=None,
    final_name=None,
    recog_dac_settings=None,
    posenc_dac_settings=None,
):
    from torch_memristor.memristor_modules import DacAdcHardwareSettings
    if recog_dac_settings is None:
        recog_dac_settings = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=4,
            output_range_bits=4,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
    if posenc_dac_settings is None:
        posenc_dac_settings = DacAdcHardwareSettings(
            input_bits=8,
            output_precision_bits=1,
            output_range_bits=7,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )
    prior_args = {
        "config": train_config,
        "network_module": prior_network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }
    res_fixed, res_greedy = {}, {}
    for num_cycles in range(1, max_runs + 1):
        if recog_model_config_class is not None:
            model_config_recog = recog_model_config_class(
                **model_config.__dict__, pos_enc_converter_hardware_settings=None
            )
        else:
            model_config_recog = copy.deepcopy(model_config)
        model_config_recog.converter_hardware_settings = recog_dac_settings
        model_config_recog.num_cycles = num_cycles
        model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings
        train_args_recog = {
            "config": train_config,
            "network_module": recog_network_module,
            "net_args": {"model_config_dict": asdict(model_config_recog)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }
        recog_name = recog_name_prefix + f"/cycle_{num_cycles // 11}"
        res_fixed = eval_model(
            training_name=recog_name + f"_{num_cycles}",
            train_job=train_job,
            train_args=train_args_recog,
            train_data=train_data,
            decoder_config=rasr_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=res_fixed,
            decoder_module="ctc.decoder.rasr_ctc_v1_batched",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            use_gpu=True,
            import_memristor=True,
            extra_forward_config={"batch_size": batch_size},
            run_best_4=False,
            run_best=False,
            prior_args=prior_args,
            run_search_on_hpc=False,
            run_rasr=True,
            split_mem_init=True,
        )
        res_greedy = eval_model(
            training_name=recog_name + f"_{num_cycles}",
            train_job=train_job,
            train_args=train_args_recog,
            train_data=train_data,
            decoder_config=greedy_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=res_greedy,
            decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
            prior_scales=[0.0],
            lm_scales=[0.0],
            use_gpu=True,
            import_memristor=True,
            extra_forward_config={"batch_size": batch_size},
            run_best_4=False,
            run_best=False,
            prior_args=None,
            with_prior=False,
            run_search_on_hpc=False,
            run_rasr=False,
            split_mem_init=True,
        )
        if num_cycles == max_runs:
            report_name = final_name if final_name is not None else recog_name
            generate_report(results=res_fixed, exp_name=report_name)
            generate_report(results=res_greedy, exp_name=report_name + "_greedy")
            report_dict[report_name] = copy.deepcopy(res_fixed)
            report_dict[report_name + "_greedy"] = copy.deepcopy(res_greedy)


def run_non_memristor_eval(
    training_name, train_job, train_args, train_data,
    rasr_config, greedy_config, dev_dataset_tuples,
    rasr_prior_scales, rasr_lm_scales, report_dict,
):
    results, best_params_job = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=rasr_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict={},
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
    report_dict[training_name] = results

    results, _ = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=greedy_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict={},
        decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
        prior_scales=[0.0],
        lm_scales=[0.0],
        import_memristor=True,
        get_best_params=True,
        run_rasr=False,
        run_best_4=False,
        run_best=False,
        with_prior=False,
    )
    generate_report(results=results, exp_name=training_name + "/greedy/non_memristor")
    report_dict[training_name + "_greedy"] = results
    return best_params_job


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
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_quant_v1 import DecoderConfig as GreedyDecoderConfig
    as_training_greedy_decoder_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe128.vocab,
        turn_off_quant="leave_as_is",
    )
    greedy_decoder_memristor = copy.deepcopy(as_training_greedy_decoder_config)
    greedy_decoder_memristor.turn_off_quant = False

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

    for epochs in [500, 1000, 2000]:
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
        }
        train_args_base = {
            "config": train_config_24gbgpu,
            "network_module": baseline_network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "use_speed_perturbation": True,
            "post_config": {"num_workers_per_gpu": 8}
        }
        name = f".baseline_512dim_sub4_48gbgpu_{epochs//10}eps_radam_bs360_sp"
        training_name = prefix_name + "/" + baseline_network_module + name
        train_job = training(training_name, train_data_bpe128, train_args_base, num_epochs=epochs, **default_returnn)
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

        results = {}
        results, _ = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args_base,
            train_data=train_data_bpe128,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
            prior_scales=[0.0],
            lm_scales=[0.0],
            import_memristor=True,
            get_best_params=True,
            run_rasr=False,
            run_best_4=False,
            run_best=False,
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        memristor_report[training_name + '_greedy'] = results

    def _make_frontend_config(out_features):
        return VGG4LayerActFrontendV1Config_mod(
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
            out_features=out_features,
            activation=None,
        )

    def _make_model_config_kwargs(frontend_config, conformer_size, activation_bit, converter_hardware_settings):
        return dict(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config_full,
            label_target_size=vocab_size_without_blank,
            conformer_size=conformer_size,
            num_layers=12,
            num_heads=8,
            ff_dim=conformer_size * 4,
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
            activation_bit_prec=activation_bit,
            quantize_output=False,
            converter_hardware_settings=converter_hardware_settings,
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

    network_module_mem_v7 = "ctc.qat_0711.memristor_v7"
    network_module_mem_v10 = "ctc.qat_0711.memristor_v10"
    network_module_mem_v11 = "ctc.qat_0711.memristor_v11"
    network_module_mem_v12 = "ctc.qat_0711.memristor_v12"
    network_module_mem_v13 = "ctc.qat_0711.memristor_v13"

    from ...pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8

    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    for activation_bit in [8]:
        for epochs in [500, 1000, 2000]:
            for dim in [128, 256, 384, 512, 768, 1024]:
                for weight_bit in [4, 5, 6, 7, 8]:
                    if weight_bit in [6, 7]:
                        if not dim == 512 or not epochs == 1000:
                            continue
                    test_dims = [128, 256, 768]
                    if dim in test_dims and epochs not in [500]:
                        continue
                    if dim not in [512, 1024]:
                        if weight_bit not in [4, 8]:
                            continue
                    if epochs not in [1000]:
                        if dim > 768 or weight_bit not in [4, 8]:
                            continue
                    frontend_config_dim = _make_frontend_config(dim)
                    prior_train_dac_settings = DacAdcHardwareSettings(
                        input_bits=0,
                        output_precision_bits=0,
                        output_range_bits=0,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    model_config = MemristorModelTrainConfigV8(
                        **_make_model_config_kwargs(frontend_config_dim, dim, activation_bit, prior_train_dac_settings),
                        weight_bit_prec=weight_bit,
                    )
                    seeds = 3
                    if dim not in [512]:
                        seeds = 2
                        if dim not in [1024]:
                            seeds = 1
                    elif epochs not in [1000]:
                        seeds = 2
                    if weight_bit in [6, 7]:
                        seeds = 1
                    for seed in range(seeds):
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
                        if dim in test_dims:
                            train_job.rqmt['cpu'] = 8
                            train_job.rqmt['gpu_mem'] = 48
                        elif not os.path.exists(
                            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.rqmt['cpu'] = 8
                            train_job.hold()
                            train_job.move_to_hpc = True

                        _ = run_non_memristor_eval(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data_bpe128,
                            rasr_config=as_training_rasr_config,
                            greedy_config=as_training_greedy_decoder_config,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            rasr_prior_scales=rasr_prior_scales,
                            rasr_lm_scales=rasr_lm_scales,
                            report_dict=memristor_report,
                        )

                        max_runs = 10
                        from ...pytorch_networks.ctc.qat_0711.memristor_v11_cfg import \
                            QuantModelTrainConfigV11 as MemristorModelTrainConfigV11
                        run_memristor_cycle_eval(
                            train_job=train_job,
                            train_data=train_data_bpe128,
                            train_config=train_config_24gbgpu,
                            model_config=model_config,
                            recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}",
                            rasr_config=rasr_config_memristor,
                            greedy_config=greedy_decoder_memristor,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            prior_scales=[0.5],
                            lm_scales=[0.8],
                            batch_size=3500000 if weight_bit not in [8] else 2500000,
                            max_runs=max_runs,
                            report_dict=memristor_report,
                            prior_network_module=network_module_mem_v10,
                            recog_network_module=network_module_mem_v11,
                            recog_model_config_class=MemristorModelTrainConfigV11,
                        )


                        model_config_ideal = copy.deepcopy(model_config)
                        train_dac_settings_ideal = DacAdcHardwareSettings(
                            input_bits=8,
                            output_precision_bits=4,
                            output_range_bits=4,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=5476.0,
                        )

                        posenc_dac_settings_ideal = DacAdcHardwareSettings(
                            input_bits=8,
                            output_precision_bits=1,
                            output_range_bits=7,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=5476.0,
                        )
                        from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
                        ideal = CycleCorrectionSettings(
                            num_cycles=None,
                            test_input_value=None,
                            relative_deviation=None,
                            ideal_programming=True
                        )
                        model_config_ideal.correction_settings = ideal

                        if dim == 512:
                            run_memristor_cycle_eval(
                                train_job=train_job,
                                train_data=train_data_bpe128,
                                train_config=train_config_24gbgpu,
                                model_config=model_config_ideal,
                                recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_ideal_seed_{seed}",
                                rasr_config=rasr_config_memristor,
                                greedy_config=greedy_decoder_memristor,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                prior_scales=[0.5],
                                lm_scales=[0.8],
                                batch_size=3500000 if weight_bit not in [8] else 2500000,
                                max_runs=3,
                                report_dict=memristor_report,
                                prior_network_module=network_module_mem_v10,
                                recog_network_module=network_module_mem_v11,
                                recog_model_config_class=MemristorModelTrainConfigV11,
                                recog_dac_settings=train_dac_settings_ideal,
                                posenc_dac_settings=posenc_dac_settings_ideal,
                            )


    from ...pytorch_networks.ctc.qat_0711.memristor_v12_cfg import QuantModelTrainConfigV12 as MemristorModelTrainConfigV12
    weight_bit = 4
    activation_bit = 8
    epochs = 1000
    frontend_config_dim = _make_frontend_config(512)
    prior_train_dac_settings = DacAdcHardwareSettings(
        input_bits=0,
        output_precision_bits=0,
        output_range_bits=0,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    model_config = MemristorModelTrainConfigV12(
        **_make_model_config_kwargs(frontend_config_dim, 512, activation_bit, prior_train_dac_settings),
        weight_bit_prec=[4] * 12,
        pos_enc_converter_hardware_settings=prior_train_dac_settings,
    )
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
            "network_module": network_module_mem_v12,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }
        training_name = prefix_name + "/" + network_module_mem_v12 + f"_{epochs // 10}eps_test_conv_order_{512}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
        train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)

        train_job.rqmt['gpu_mem'] = 48
        train_job.rqmt['cpu'] = 8
        train_job.rqmt['mem'] = 36

        _ = run_non_memristor_eval(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe128,
            rasr_config=as_training_rasr_config,
            greedy_config=as_training_greedy_decoder_config,
            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
            rasr_prior_scales=rasr_prior_scales,
            rasr_lm_scales=rasr_lm_scales,
            report_dict=memristor_report,
        )

    for weight_precisions in [
        [8] * 3 + [4] * 6 + [8] * 3,
        [8] * 6 + [4] * 6,
        [8] * 3 + [4] * 2 + [3] * 2 + [4] * 2 + [8] * 3,
    ]:
        model_config = MemristorModelTrainConfigV12(
            **_make_model_config_kwargs(frontend_config_dim, 512, activation_bit, prior_train_dac_settings),
            weight_bit_prec=weight_precisions,
            pos_enc_converter_hardware_settings=prior_train_dac_settings,
        )
        for seed in range(2 if weight_precisions == [8] * 3 + [4] * 6 + [8] * 3 else 1):
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
                "network_module": network_module_mem_v12,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            from itertools import groupby as _groupby
            w_str = "_".join(f"{v}x{sum(1 for _ in g)}" for v, g in _groupby(weight_precisions))
            training_name = prefix_name + "/" + network_module_mem_v12 + f"_{epochs // 10}eps_{512}dim_w{w_str}_a{activation_bit}_seed_{seed}"
            train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)


            if weight_precisions == [8] * 3 + [4] * 6 + [8] * 3:
                train_job.rqmt['gpu_mem'] = 48
                train_job.rqmt['cpu'] = 8
            elif not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                train_job.rqmt['cpu'] = 8
                train_job.hold()
                train_job.move_to_hpc = True

            _ = run_non_memristor_eval(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args,
                train_data=train_data_bpe128,
                rasr_config=as_training_rasr_config,
                greedy_config=as_training_greedy_decoder_config,
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                rasr_prior_scales=rasr_prior_scales,
                rasr_lm_scales=rasr_lm_scales,
                report_dict=memristor_report,
            )

            run_memristor_cycle_eval(
                train_job=train_job,
                train_data=train_data_bpe128,
                train_config=train_config_24gbgpu,
                model_config=model_config,
                recog_name_prefix=prefix_name + "/" + network_module_mem_v12 +  f"_{epochs // 10}eps_{512}dim_w{w_str}_a{activation_bit}_seed_{seed}",
                rasr_config=rasr_config_memristor,
                greedy_config=greedy_decoder_memristor,
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                prior_scales=[0.5],
                lm_scales=[0.8],
                batch_size=3500000 if weight_bit not in [8] else 2500000,
                max_runs=5,
                report_dict=memristor_report,
                prior_network_module=network_module_mem_v10,
                recog_network_module=network_module_mem_v11,
            )

    from ...pytorch_networks.ctc.qat_0711.memristor_v13_cfg import QuantModelTrainConfigV13 as MemristorModelTrainConfigV13
    for weight_bit in [8]:
        for activation_bit in [8]:
            for epochs in [1000]:
                for dim in [512]:
                    for weight_dropout in [0.0, 0.1, 0.2]:
                        frontend_config_dim = _make_frontend_config(dim)
                        prior_train_dac_settings = DacAdcHardwareSettings(
                            input_bits=0,
                            output_precision_bits=0,
                            output_range_bits=0,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=8020.0,
                        )

                        model_config = MemristorModelTrainConfigV13(
                            **_make_model_config_kwargs(frontend_config_dim, 512, activation_bit, prior_train_dac_settings),
                            weight_bit_prec=weight_bit,
                            pos_enc_converter_hardware_settings=prior_train_dac_settings,
                            weight_dropout=weight_dropout,
                        )

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
                                "network_module": network_module_mem_v13,
                                "net_args": {"model_config_dict": asdict(model_config)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = prefix_name + "/" + network_module_mem_v13 + f"_{epochs // 10}eps_wdrop{weight_dropout}_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                            train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs,
                                **default_returnn)

                            if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True

                            _ = run_non_memristor_eval(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_bpe128,
                                rasr_config=as_training_rasr_config,
                                greedy_config=as_training_greedy_decoder_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                rasr_prior_scales=rasr_prior_scales,
                                rasr_lm_scales=rasr_lm_scales,
                                report_dict=memristor_report,
                            )

                            max_runs = 5
                            run_memristor_cycle_eval(
                                train_job=train_job,
                                train_data=train_data_bpe128,
                                train_config=train_config_24gbgpu,
                                model_config=model_config,
                                recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_wdrop{weight_dropout}_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}",
                                rasr_config=rasr_config_memristor,
                                greedy_config=greedy_decoder_memristor,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                prior_scales=[0.5],
                                lm_scales=[0.8],
                                batch_size=3500000 if weight_bit not in [8] else 2500000,
                                max_runs=max_runs,
                                report_dict=memristor_report,
                                prior_network_module=network_module_mem_v10,
                                recog_network_module=network_module_mem_v11,
                            )


    # tk.register_report("reports/lbs/memristor_report_bpe", partial(build_qat_report, memristor_report),
    #     required=memristor_report, update_frequency=400)
    tk.register_report("reports/lbs/v2/memristor_bpe", partial(build_qat_report_v2, memristor_report),
                       required=memristor_report, update_frequency=400)

