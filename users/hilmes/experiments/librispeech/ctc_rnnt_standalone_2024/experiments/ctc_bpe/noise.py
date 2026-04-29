from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast
import os
from functools import partial

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_bpe_bliss_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_arpa_lm_config
from ...pipeline import training
from ...report import generate_report, build_qat_report, build_qat_report_v2

from ..ctc_phon.tune_eval import eval_model
from .memristor import run_non_memristor_eval, run_memristor_cycle_eval


def bpe_ls960_0426_noise():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/bpe_ls960_memristor/noise"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

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
        turn_off_quant="leave_as_is",
    )
    rasr_config_memristor = copy.deepcopy(as_training_rasr_config)
    rasr_config_memristor.turn_off_quant = False

    rasr_prior_scales = [0.2, 0.3, 0.4, 0.5]
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    rasr_noise_prior_scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rasr_noise_lm_scales = [0.7, 0.8, 0.9, 1.0, 1.1]

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_quant_v1 import DecoderConfig as GreedyDecoderConfig
    as_training_greedy_decoder_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe128.vocab,
        turn_off_quant="leave_as_is",
    )
    greedy_decoder_memristor = copy.deepcopy(as_training_greedy_decoder_config)
    greedy_decoder_memristor.turn_off_quant = False

    network_module_mem_v10 = "ctc.qat_0711.memristor_v10"
    network_module_mem_v11 = "ctc.qat_0711.memristor_v11"

    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8
    from ...pytorch_networks.ctc.qat_0711.memristor_v11_cfg import QuantModelTrainConfigV11 as MemristorModelTrainConfigV11
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, LogMelFeatureExtractionV1Config,
    )
    from ...pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ConformerPosEmbConfig,
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
        max_dim_feat=16,
        num_repeat_feat=5,
    )

    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    prior_train_dac_settings = DacAdcHardwareSettings(
        input_bits=0,
        output_precision_bits=0,
        output_range_bits=0,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    memristor_report = {}
    activation_bits = [8]
    dims = [384, 512, 1024]
    weight_bits = [4, 8]
    memristor_runs = 5

    def _make_frontend_config(dim):
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
            out_features=dim,
            activation=None,
        )

    def _make_model_config_kwargs(dim, weight_noise_func=None, weight_noise_values=None, weight_noise_start_epoch=None):
        return dict(
            feature_extraction_config=fe_config,
            frontend_config=_make_frontend_config(dim),
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
            quantize_output=False,
            converter_hardware_settings=prior_train_dac_settings,
            quant_in_linear=True,
            num_cycles=0,
            correction_settings=None,
            weight_noise_func=weight_noise_func,
            weight_noise_values=weight_noise_values,
            weight_noise_start_epoch=weight_noise_start_epoch,
            pos_emb_config=pos_emb_cfg,
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
            dropout_broadcast_axes=None,
        )

    # --- Baseline runs (no noise) ---
    for epochs in [1000]:
        for activation_bit in activation_bits:
            for dim in dims:
                for weight_bit in weight_bits:
                    seeds = 2
                    model_config = MemristorModelTrainConfigV8(
                        **_make_model_config_kwargs(dim),
                        weight_bit_prec=weight_bit,
                        activation_bit_prec=activation_bit,
                    )
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
                        training_name = prefix_name + "/" + network_module_mem_v10 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                        train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)
                        if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                            train_job.rqmt['cpu'] = 12
                            train_job.hold()
                            train_job.move_to_hpc = True

                        best_params_job = run_non_memristor_eval(
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
                            recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}",
                            rasr_config=rasr_config_memristor,
                            greedy_config=greedy_decoder_memristor,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            prior_scales=[best_params_job.out_optimal_parameters[1]],
                            lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                            batch_size=3500000 if weight_bit not in [8] else 2500000,
                            max_runs=memristor_runs,
                            report_dict=memristor_report,
                            prior_network_module=network_module_mem_v10,
                            recog_network_module=network_module_mem_v11,
                            recog_model_config_class=MemristorModelTrainConfigV11,
                            final_name=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}_cycle",
                        )

    # --- Noise runs ---
    for epochs in [1000]:
        for activation_bit in activation_bits:
            for dim in dims:
                for weight_bit in weight_bits:
                    for dropout in [0.1]:
                        seeds = 1
                        for start_epoch in [1]:
                            for dev in [0.05]:
                                model_config = MemristorModelTrainConfigV8(
                                    **_make_model_config_kwargs(
                                        dim,
                                        weight_noise_func="gauss",
                                        weight_noise_values={"dev": dev},
                                        weight_noise_start_epoch=start_epoch,
                                    ),
                                    weight_bit_prec=weight_bit,
                                    activation_bit_prec=activation_bit,
                                )
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
                                    training_name = prefix_name + "/" + network_module_mem_v10 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise{start_epoch}_{dev}_drop{dropout}_seed_{seed}"
                                    train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)
                                    if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                                        train_job.rqmt['cpu'] = 12
                                        train_job.hold()
                                        train_job.move_to_hpc = True

                                    prior_config = copy.deepcopy(model_config)
                                    prior_config.weight_noise_func = None
                                    prior_config.weight_noise_values = None
                                    prior_config.weight_noise_start_epoch = None
                                    prior_args = copy.deepcopy(train_args)
                                    prior_args["net_args"] = {"model_config_dict": asdict(prior_config)}

                                    results = {}
                                    results, _ = eval_model(
                                        training_name=training_name + "_with_noise",
                                        train_job=train_job,
                                        train_args=train_args,
                                        train_data=train_data_bpe128,
                                        decoder_config=as_training_rasr_config,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        result_dict=results,
                                        decoder_module="ctc.decoder.rasr_ctc_v1",
                                        prior_scales=rasr_noise_prior_scales,
                                        lm_scales=rasr_noise_lm_scales,
                                        prior_args=prior_args,
                                        import_memristor=True,
                                        get_best_params=True,
                                        run_rasr=True,
                                        run_best_4=False,
                                        run_best=False,
                                    )
                                    generate_report(results=results, exp_name=training_name + "/with_noise")
                                    memristor_report[training_name + "/with_noise"] = results

                                    results = {}
                                    results, best_params_job = eval_model(
                                        training_name=training_name + "_without_noise",
                                        train_job=train_job,
                                        train_args=prior_args,
                                        train_data=train_data_bpe128,
                                        decoder_config=as_training_rasr_config,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        result_dict=results,
                                        decoder_module="ctc.decoder.rasr_ctc_v1",
                                        prior_scales=rasr_prior_scales,
                                        lm_scales=rasr_lm_scales,
                                        prior_args=prior_args,
                                        import_memristor=True,
                                        get_best_params=True,
                                        run_rasr=True,
                                        run_best_4=False,
                                        run_best=False,
                                    )
                                    generate_report(results=results, exp_name=training_name + "/without_noise")
                                    memristor_report[training_name + "/without_noise"] = results

                                    run_memristor_cycle_eval(
                                        train_job=train_job,
                                        train_data=train_data_bpe128,
                                        train_config=train_config_24gbgpu,
                                        model_config=prior_config,
                                        recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise{start_epoch}_{dev}_drop{dropout}_seed_{seed}",
                                        rasr_config=rasr_config_memristor,
                                        greedy_config=greedy_decoder_memristor,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        prior_scales=[best_params_job.out_optimal_parameters[1]],
                                        lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                        batch_size=3500000 if weight_bit not in [8] else 2500000,
                                        max_runs=memristor_runs,
                                        report_dict=memristor_report,
                                        prior_network_module=network_module_mem_v10,
                                        recog_network_module=network_module_mem_v11,
                                        recog_model_config_class=MemristorModelTrainConfigV11,
                                        final_name=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise{start_epoch}_{dev}_drop{dropout}_seed_{seed}_cycle",
                                    )
                                    run_memristor_cycle_eval(
                                        train_job=train_job,
                                        train_data=train_data_bpe128,
                                        train_config=train_config_24gbgpu,
                                        model_config=prior_config,
                                        recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise{start_epoch}_{dev}_drop{dropout}_seed_{seed}",
                                        rasr_config=rasr_config_memristor,
                                        greedy_config=greedy_decoder_memristor,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        prior_scales=[0.5],
                                        lm_scales=[0.8],
                                        batch_size=3500000 if weight_bit not in [8] else 2500000,
                                        max_runs=memristor_runs,
                                        report_dict=memristor_report,
                                        prior_network_module=network_module_mem_v10,
                                        recog_network_module=network_module_mem_v11,
                                        recog_model_config_class=MemristorModelTrainConfigV11,
                                        final_name=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise{start_epoch}_{dev}_drop{dropout}_seed_{seed}_cycle",
                                    )

    tk.register_report("reports/lbs/v2/memristor_noise_bpe", partial(build_qat_report_v2, memristor_report),
                       required=memristor_report, update_frequency=400)