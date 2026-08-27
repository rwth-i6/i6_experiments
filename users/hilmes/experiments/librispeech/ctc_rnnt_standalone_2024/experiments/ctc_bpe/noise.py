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
    network_module_mem_v15 = "ctc.qat_0711.memristor_v15"

    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8
    from ...pytorch_networks.ctc.qat_0711.memristor_v11_cfg import QuantModelTrainConfigV11 as MemristorModelTrainConfigV11
    from ...pytorch_networks.ctc.qat_0711.memristor_v15_cfg import (
        QuantModelTrainConfigV15 as MemristorModelTrainConfigV15,
        GaussianWeightNoiseConfig,
        BitFlipWeightNoiseConfig,
        GaussianWeightLevelNoiseConfig,
        UniformBitNoiseConfig,
        UniformWeightLevelNoiseConfig,
        RelativeGaussianWeightNoiseConfig,
        SynaptogenPoolNoiseConfig,
        AsymmetricBitNoiseConfig,
        StudentTWeightNoiseConfig,
        BitMixingWeightNoiseConfig,
        BitFlipSTEWeightNoiseConfig,
        HeldGaussianNoiseConfig,
        RampGaussianNoiseConfig,
        RandomAmplitudeGaussianNoiseConfig,
        ColumnGainNoiseConfig,
        StuckAtNoiseConfig,
        MixtureNoiseConfig,
        TruncatedCauchyNoiseConfig,
        TileGainNoiseConfig,
        RowGainDistNoiseConfig,
        DecoupledGaussianNoiseConfig,
        SumNoiseConfig,
        AsymmetricBitFlipSTENoiseConfig,
    )
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

    _NO_WEIGHT_NOISE = object()  # sentinel: distinguish "not passed" from an explicit None (no-noise control)

    def _make_model_config_kwargs(dim, weight_noise=_NO_WEIGHT_NOISE):
        d = dict(
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
            pos_emb_config=pos_emb_cfg,
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
            dropout_broadcast_axes=None,
        )
        if weight_noise is not _NO_WEIGHT_NOISE:
            d["weight_noise"] = weight_noise
        return d

    # --- Baseline runs (no noise) ---
    FINETUNE_MODELS = {}
    for epochs in [1000, 1250, 1500, 2000]:
        for activation_bit in activation_bits:
            for dim in dims:
                for weight_bit in weight_bits:
                    if epochs > 1000 and dim not in [512]:
                        continue
                    seeds = 2
                    model_config = MemristorModelTrainConfigV8(
                        **_make_model_config_kwargs(dim),
                        weight_bit_prec=weight_bit,
                        activation_bit_prec=activation_bit,
                        weight_noise_func=None,
                        weight_noise_values=None,
                        weight_noise_start_epoch=None,
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
                        FINETUNE_MODELS[training_name] = train_job.out_checkpoints[epochs]
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
                            max_runs=memristor_runs if dim <= 512 else 3,
                            report_dict=memristor_report,
                            prior_network_module=network_module_mem_v10,
                            recog_network_module=network_module_mem_v11,
                            recog_model_config_class=MemristorModelTrainConfigV11,
                            final_name=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}_best_cycle",
                            search_gpu=11 if dim <= 512 else 48,
                            search_gpu_type="rtx_2080" if dim <= 512 else None,
                            fast_inference=True,
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
                            prior_scales=[0.5],
                            lm_scales=[0.8],
                            batch_size=3500000 if weight_bit not in [8] else 2500000,
                            max_runs=memristor_runs if dim <= 512 else 3,
                            report_dict=memristor_report,
                            prior_network_module=network_module_mem_v10,
                            recog_network_module=network_module_mem_v11,
                            recog_model_config_class=MemristorModelTrainConfigV11,
                            final_name=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}_fixed_cycle",
                            search_gpu=11 if dim <= 512 else 48,
                            search_gpu_type="rtx_2080" if dim <= 512 else None,
                            fast_inference=True,
                        )
                        if dim == 512 and epochs == 1000:
                            # single-forward multi-scale sweep (paper numbers): one forward per
                            # cycle, all (lm, prior) combos applied to the same posteriors.
                            # fast_inference keeps the forward cheap (~27 min); search on 48gb L40S.
                            # Shares the cached conversion/prior jobs with the fixed cycle above.
                            # lm parallelised over 9 workers (free); prior trimmed to the baseline
                            # empirical good region [0.3-0.7] (sweep showed optima at prior 0.4-0.7).
                            run_memristor_cycle_eval(
                                train_job=train_job,
                                train_data=train_data_bpe128,
                                train_config=train_config_24gbgpu,
                                model_config=model_config,
                                recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_fixed_multi_sweep_seed_{seed}",
                                rasr_config=rasr_config_memristor,
                                greedy_config=None,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                prior_scales=[0.3, 0.4, 0.5, 0.6, 0.7],
                                lm_scales=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                                batch_size=3500000,
                                max_runs=memristor_runs,
                                report_dict=memristor_report,
                                prior_network_module=network_module_mem_v10,
                                recog_network_module=network_module_mem_v11,
                                recog_model_config_class=MemristorModelTrainConfigV11,
                                search_gpu=48,
                                run_rasr_multi=True,
                                num_search_workers=9,
                                fast_inference=True,
                            )
                        prior_config = copy.deepcopy(model_config)
                        prior_args = copy.deepcopy(train_args)
                        prior_args["net_args"] = {"model_config_dict": asdict(prior_config)}
                        model_config_ideal = copy.deepcopy(prior_config)
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
                        run_memristor_cycle_eval(
                            train_job=train_job,
                            train_data=train_data_bpe128,
                            train_config=train_config_24gbgpu,
                            model_config=model_config_ideal,
                            recog_name_prefix=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_ideal_seed_{seed}",
                            rasr_config=rasr_config_memristor,
                            greedy_config=None,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            prior_scales=[0.5],
                            lm_scales=[0.8],
                            batch_size=3500000 if weight_bit not in [8] else 2500000,
                            max_runs=2,
                            report_dict=memristor_report,
                            prior_network_module=network_module_mem_v10,
                            recog_network_module=network_module_mem_v11,
                            recog_model_config_class=MemristorModelTrainConfigV11,
                            final_name=prefix_name + "/" + network_module_mem_v11 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}_fixed_ideal",
                            search_gpu=11 if dim <= 512 else 48,
                            search_gpu_type="rtx_2080" if dim <= 512 else None,
                            fast_inference=True,
                            recog_dac_settings=train_dac_settings_ideal,
                            posenc_dac_settings=posenc_dac_settings_ideal,
                        )

    from ...pytorch_networks.ctc.qat_0711.memristor_v13_cfg import \
        QuantModelTrainConfigV13 as MemristorModelTrainConfigV13
    # TODO: finetune
    network_module_mem_v13 = "ctc.qat_0711.memristor_v13"
    for weight_bit in [4, 8]:
        for activation_bit in [8]:
            for epochs in [1000]:
                for dim in [512]:
                    for weight_dropout in [0.1, 0.2]:
                        frontend_config_dim = _make_frontend_config(dim)
                        prior_train_dac_settings = DacAdcHardwareSettings(
                            input_bits=0,
                            output_precision_bits=0,
                            output_range_bits=0,
                            hardware_input_vmax=0.6,
                            hardware_output_current_scaling=8020.0,
                        )

                        model_config = MemristorModelTrainConfigV13(
                            **_make_model_config_kwargs(dim),
                            weight_bit_prec=weight_bit,
                            pos_enc_converter_hardware_settings=prior_train_dac_settings,
                            weight_dropout=weight_dropout,
                            activation_bit_prec=activation_bit,
                            weight_noise_func=None,
                            weight_noise_values=None,
                            weight_noise_start_epoch=None,
                        )

                        for seed in range(2):
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

                            if not os.path.exists(
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

                            max_runs = 5
                            run_memristor_cycle_eval(
                                train_job=train_job,
                                train_data=train_data_bpe128,
                                train_config=train_config_24gbgpu,
                                model_config=model_config,
                                recog_name_prefix=prefix_name + "/" + network_module_mem_v13 + f"_{epochs // 10}eps_wdrop{weight_dropout}_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}",
                                rasr_config=rasr_config_memristor,
                                greedy_config=greedy_decoder_memristor,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                prior_scales=[0.5],
                                lm_scales=[0.8],
                                batch_size=3500000 if weight_bit not in [8] else 2500000,
                                max_runs=max_runs,
                                report_dict=memristor_report,
                                prior_network_module=network_module_mem_v13,
                                recog_network_module=network_module_mem_v13,
                                search_gpu_type="rtx_2080",
                                fast_inference=True,
                            )

    # --- Noise runs ---
    noise_configs = [
        (GaussianWeightNoiseConfig(dev=0.05, start_epoch=1), "gauss0.05_ep1"),
        (BitFlipWeightNoiseConfig(p=0.01, start_epoch=1), "bitflip0.01_ep1"),
    ]
    dims = [512]
    for epochs in [1000]:
        for activation_bit in activation_bits:
            for dim in dims:
                for weight_bit in weight_bits:
                    for dropout in [0.1]:
                        seeds = 1
                        for noise_cfg, noise_name in noise_configs:
                            memristor_runs = 5 if dim <= 512 else 3
                            model_config = MemristorModelTrainConfigV15(
                                **_make_model_config_kwargs(
                                    dim,
                                    weight_noise=noise_cfg,
                                ),
                                weight_bit_prec=weight_bit,
                                activation_bit_prec=activation_bit,
                                weight_dropout=0.0,
                                weight_pruning=None,
                                pos_enc_converter_hardware_settings=prior_train_dac_settings
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
                                    "network_module": network_module_mem_v15,
                                    "net_args": {"model_config_dict": asdict(model_config)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }
                                training_name = prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}"
                                train_job = training(training_name, train_data_bpe128, train_args, num_epochs=epochs, **default_returnn)
                                if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                                    train_job.rqmt['cpu'] = 12
                                    train_job.hold()
                                    train_job.move_to_hpc = True

                                prior_config = copy.deepcopy(model_config)
                                prior_config.weight_noise = None
                                prior_args = copy.deepcopy(train_args)
                                prior_args["net_args"] = {"model_config_dict": asdict(prior_config)}

                                results = {}
                                results, best_params_job_noise = eval_model(
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

                                # run_memristor_cycle_eval(
                                #     train_job=train_job,
                                #     train_data=train_data_bpe128,
                                #     train_config=train_config_24gbgpu,
                                #     model_config=prior_config,
                                #     recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}",
                                #     rasr_config=rasr_config_memristor,
                                #     greedy_config=None,
                                #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                #     prior_scales=[best_params_job.out_optimal_parameters[1]],
                                #     lm_scales=[(best_params_job.out_optimal_parameters[0], "best_nonoise")],
                                #     batch_size=3500000 if weight_bit not in [8] else 2500000,
                                #     max_runs=memristor_runs,
                                #     report_dict=memristor_report,
                                #     prior_network_module=network_module_mem_v15,
                                #     recog_network_module=network_module_mem_v15,
                                #     recog_model_config_class=MemristorModelTrainConfigV15,
                                #     final_name=prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_best_nonoise_cycle",
                                #     search_gpu=11 if dim <= 512 else 24,
                                # )
                                # run_memristor_cycle_eval(
                                #     train_job=train_job,
                                #     train_data=train_data_bpe128,
                                #     train_config=train_config_24gbgpu,
                                #     model_config=prior_config,
                                #     recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}",
                                #     rasr_config=rasr_config_memristor,
                                #     greedy_config=None,
                                #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                #     prior_scales=[best_params_job_noise.out_optimal_parameters[1]],
                                #     lm_scales=[(best_params_job_noise.out_optimal_parameters[0], "best_noise")],
                                #     batch_size=3500000 if weight_bit not in [8] else 2500000,
                                #     max_runs=memristor_runs,
                                #     report_dict=memristor_report,
                                #     prior_network_module=network_module_mem_v15,
                                #     recog_network_module=network_module_mem_v15,
                                #     recog_model_config_class=MemristorModelTrainConfigV15,
                                #     final_name=prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_best_noise_cycle",
                                #     search_gpu=11 if dim <= 512 else 24,
                                # )
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=prior_config,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=memristor_runs,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v15,
                                    recog_network_module=network_module_mem_v15,
                                    recog_model_config_class=MemristorModelTrainConfigV15,
                                    final_name=prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_cycle",
                                    search_gpu=11 if dim <= 512 else 24,
                                    fast_inference=True,
                                )

                                model_config_ideal = copy.deepcopy(prior_config)
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
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=model_config_ideal,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_ideal_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=memristor_runs,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v15,
                                    recog_network_module=network_module_mem_v15,
                                    recog_model_config_class=MemristorModelTrainConfigV15,
                                    final_name=prefix_name + "/" + network_module_mem_v15 + f"_{epochs // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_ideal_fixed_cycle",
                                    recog_dac_settings=train_dac_settings_ideal,
                                    posenc_dac_settings=posenc_dac_settings_ideal,
                                    search_gpu=11 if dim <= 512 else 24,
                                    fast_inference=True,
                                )

    # --- Noise finetune runs ---
    noise_configs_finetune = [
    (GaussianWeightNoiseConfig(dev=0.05, start_epoch=1), "gauss0.05_ep1"),
    (BitFlipWeightNoiseConfig(p=0.01, start_epoch=1), "bitflip0.01_ep1"),
    (GaussianWeightNoiseConfig(dev=0.1, start_epoch=1), "gauss0.1_ep1"),
    (GaussianWeightNoiseConfig(dev=0.01, start_epoch=1), "gauss0.01_ep1"),
    ]
    # New analog weight-noise variants; only run in the 25-epoch finetune (finetune_epochs == 250)
    noise_configs_finetune_new = [
        (GaussianWeightLevelNoiseConfig(weight_dev=0.05, start_epoch=1), "gaussw0.05_ep1"),
        (UniformBitNoiseConfig(bit_amplitude=0.05, start_epoch=1), "unifbit0.05_ep1"),
        (UniformWeightLevelNoiseConfig(weight_amplitude=0.05, start_epoch=1), "unifw0.05_ep1"),
        (RelativeGaussianWeightNoiseConfig(rel_dev=0.05, start_epoch=1), "relgauss0.05_ep1"),
        # Removed 2026-07-29: deterministic given `mix`, so the model just learns the inverse
        # (F6). Successor: AsymmetricBitNoiseConfig. Re-enable only with mixing in eval too.
        # (BitMixingWeightNoiseConfig(mix=0.12, start_epoch=1), "bitmix0.12_ep1"),
        (BitFlipSTEWeightNoiseConfig(flip_p=0.01, start_epoch=1), "bitflipste0.01_ep1"),
        # No-noise control: same finetune (LR schedule, epochs, checkpoint) with noise disabled,
        # to separate the finetune effect from the noise effect on hardware WER.
        (None, "nonoise"),
        # Bracket the magnitude optimum: dev 0.05 is best of {0.01, 0.05, 0.1} on device,
        # 0.03/0.07 give sigma/rms(w) ~ 0.19/0.43 around the ~0.3 optimum.
        (GaussianWeightNoiseConfig(dev=0.03, start_epoch=1), "gauss0.03_ep1"),
        (GaussianWeightNoiseConfig(dev=0.07, start_epoch=1), "gauss0.07_ep1"),
        # densify the lower flank: sigma/rms(w) ~ 0.12/0.25 at w8
        (GaussianWeightNoiseConfig(dev=0.02, start_epoch=1), "gauss0.02_ep1"),
        (GaussianWeightNoiseConfig(dev=0.04, start_epoch=1), "gauss0.04_ep1"),
        # close the 0.05 -> 0.07 gap, where the U-curve turns back up
        (GaussianWeightNoiseConfig(dev=0.06, start_epoch=1), "gauss0.06_ep1"),
        # Device-matched noise: empirical per-bit-state pool sampled from the Synaptogen
        # device ensemble (observations/synaptogen_explicit_noise_benchmark.md);
        # sigma = 0.22*rms(w) at w8 / 0.24 at w4, cost ~unifbit class (+3-5% step time).
        (
            SynaptogenPoolNoiseConfig(
                pool_size=1000000,
                num_cycles=1,
                read_noise="none",
                strength=1.0,
                parametric=False,
                pool_seed=0,
                refresh_per_epoch=False,
                start_epoch=1,
            ),
            "synpool1.0_ep1",
        ),
        # Magnitude-matched Gaussian control for synpool: dev*sqrt(2*sum 4^i) = 3.66 LSB
        # vs synpool 3.63 at w8 (0.7% match, same at w4), so WER differences between the
        # two isolate the noise *shape* (bit-state mixture, +1% gain, HRS tails) from the
        # magnitude. Also the first run of the fused (single-draw) gauss implementation.
        (GaussianWeightNoiseConfig(dev=0.035, start_epoch=1), "gauss0.035_ep1"),
        # Bit-flip STE at sane rates (analysis F7): the old p=0.01 was a ~3x magnitude
        # overdose (sigma 0.87-0.98*rms(w)), so discrete noise has never actually been
        # tested. sigma = sqrt(p*sum_{i<n}4^i): p=1e-3 -> 0.28*rms(w8) (~ the F2 optimum),
        # 3e-4/3e-3 bracket it at 0.15/0.48.
        (BitFlipSTEWeightNoiseConfig(flip_p=0.0003, start_epoch=1), "bitflipste0.0003_ep1"),
        (BitFlipSTEWeightNoiseConfig(flip_p=0.001, start_epoch=1), "bitflipste0.001_ep1"),
        (BitFlipSTEWeightNoiseConfig(flip_p=0.003, start_epoch=1), "bitflipste0.003_ep1"),
        # Noise variance depending on the bit state, at fixed total power: allocation sweep at
        # matched magnitude. r1.34 = device-measured ratio, r1.0 = exactly gauss0.05 (null).
        (
            AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.34, off_scale=1.0, normalize=True, start_epoch=1),
            "asymbit0.05_r1.34_ep1",
        ),
        (
            AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.0, off_scale=1.0, normalize=True, start_epoch=1),
            "asymbit0.05_r1.0_ep1",
        ),
        (
            AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.0, off_scale=0.0, normalize=True, start_epoch=1),
            "asymbit0.05_onlyon_ep1",
        ),
        # r=0 control: spares exactly the large weights (only they have high bits ON). For the
        # bulk of small weights this is ~gauss, so it tests whether the benefit is carried by
        # noise on the large weights (onlyoff < gauss) or by magnitude alone (onlyoff = gauss).
        (
            AsymmetricBitNoiseConfig(dev=0.05, on_scale=0.0, off_scale=1.0, normalize=True, start_epoch=1),
            "asymbit0.05_onlyoff_ep1",
        ),
        # Same magnitude as gauss0.035 but the device's tails (nu fitted by moments to the
        # device noise: 5.88 at w8 / 5.96 at w4). Isolates the tail effect from everything else.
        (StudentTWeightNoiseConfig(dev=0.035, nu=5.9, start_epoch=1), "studentt0.035_nu5.9_ep1"),
        # Allocation axis (analysis Tier 1 #3b/#7b/#3). F5 predicts relgauss0.3 lands near
        # wdrop0.1 (~6.2), not near gauss0.05 (6.0).
        (RelativeGaussianWeightNoiseConfig(rel_dev=0.3, start_epoch=1), "relgauss0.3_ep1"),
        # dropout-only anchors, sigma/rms = sqrt(p/(1-p)) = 0.229 / 0.333 / 0.5
        (None, "wdrop0.05"),
        (None, "wdrop0.1"),
        (None, "wdrop0.2"),
        # split at ~matched total sigma (quadrature): 0.315 / 0.355 vs gauss0.05 alone 0.308
        (GaussianWeightNoiseConfig(dev=0.035, start_epoch=1), "wdrop0.05_gauss0.035_ep1"),
        (GaussianWeightNoiseConfig(dev=0.02, start_epoch=1), "wdrop0.1_gauss0.02_ep1"),
        # swapped pairings complete the 2x2 split grid (quadrature totals 0.260 / 0.397)
        (GaussianWeightNoiseConfig(dev=0.02, start_epoch=1), "wdrop0.05_gauss0.02_ep1"),
        (GaussianWeightNoiseConfig(dev=0.035, start_epoch=1), "wdrop0.1_gauss0.035_ep1"),
        # dev 0.05 column extends it to the full 2x3 grid (quadrature totals 0.384 / 0.454)
        (GaussianWeightNoiseConfig(dev=0.05, start_epoch=1), "wdrop0.05_gauss0.05_ep1"),
        (GaussianWeightNoiseConfig(dev=0.05, start_epoch=1), "wdrop0.1_gauss0.05_ep1"),
        # The two best configs refinetuned with the conformer (activation) dropouts off: the
        # weight noise may already be doing that regularization.
        (GaussianWeightNoiseConfig(dev=0.05, start_epoch=1), "gauss0.05_actdrop0_ep1"),
        (GaussianWeightNoiseConfig(dev=0.035, start_epoch=1), "gauss0.035_actdrop0_ep1"),
        # studentt, synpool and unifbit were only ever run at one setting. Sweep them too,
        # so every family gets the same treatment as gauss and bitflipste.
        # dev means the same here as for gauss, so 0.02/0.035/0.05 brackets the optimum.
        (StudentTWeightNoiseConfig(dev=0.02, nu=5.9, start_epoch=1), "studentt0.02_nu5.9_ep1"),
        (StudentTWeightNoiseConfig(dev=0.05, nu=5.9, start_epoch=1), "studentt0.05_nu5.9_ep1"),
        # +-40% around the device's own noise level: is the empirical distribution best at
        # exactly the magnitude the device produces, or somewhere else?
        (
            SynaptogenPoolNoiseConfig(
                pool_size=1000000,
                num_cycles=1,
                read_noise="none",
                strength=0.7,
                parametric=False,
                pool_seed=0,
                refresh_per_epoch=False,
                start_epoch=1,
            ),
            "synpool0.7_ep1",
        ),
        (
            SynaptogenPoolNoiseConfig(
                pool_size=1000000,
                num_cycles=1,
                read_noise="none",
                strength=1.4,
                parametric=False,
                pool_seed=0,
                refresh_per_epoch=False,
                start_epoch=1,
            ),
            "synpool1.4_ep1",
        ),
        # unifbit0.05 stayed above the floor. Is that the bounded shape, or just too little
        # noise? 0.1 lands inside the gauss optimum, 0.15 goes past it.
        (UniformBitNoiseConfig(bit_amplitude=0.1, start_epoch=1), "unifbit0.1_ep1"),
        (UniformBitNoiseConfig(bit_amplitude=0.15, start_epoch=1), "unifbit0.15_ep1"),
        # --- Wave 2: every entry breaks one assumption of the tuned-sweep floor
        # (iid / fresh-per-step / per-tensor-uniform / static amplitude). Hypotheses and
        # predictions: observations/memristor_noise_methods_analysis.md, Wave 2 section.
        # Held draw, redrawn per subepoch: deployment error is one persistent draw.
        (HeldGaussianNoiseConfig(held_dev=0.035, start_epoch=1), "gausshold0.035_ep1"),
        (HeldGaussianNoiseConfig(held_dev=0.05, start_epoch=1), "gausshold0.05_ep1"),
        # Amplitude ramp 0 -> dev over the whole finetune (250 subepochs).
        (RampGaussianNoiseConfig(dev_end=0.05, ramp_epochs=250, start_epoch=1), "gaussramp0.05_ep1"),
        (RampGaussianNoiseConfig(dev_end=0.07, ramp_epochs=250, start_epoch=1), "gaussramp0.07_ep1"),
        # dev ~ U(0.02, 0.08) per call: robustness across a sigma range, not one point.
        (RandomAmplitudeGaussianNoiseConfig(dev_min=0.02, dev_max=0.08, start_epoch=1), "gaussrand0.02-0.08_ep1"),
        # One gain draw per output channel: correlated periphery error, not iid.
        (ColumnGainNoiseConfig(rel_gain=0.02, start_epoch=1), "colgain0.02_ep1"),
        (ColumnGainNoiseConfig(rel_gain=0.05, start_epoch=1), "colgain0.05_ep1"),
        # Stuck-at extremes: outlier tail robustness, separate from bulk sigma.
        (StuckAtNoiseConfig(stuck_p=0.0001, start_epoch=1), "stuck0.0001_ep1"),
        (StuckAtNoiseConfig(stuck_p=0.0003, start_epoch=1), "stuck0.0003_ep1"),
        # Random family per call, from the three per-block winners.
        (
            MixtureNoiseConfig(
                configs=(
                    GaussianWeightNoiseConfig(dev=0.05, start_epoch=1),
                    UniformBitNoiseConfig(bit_amplitude=0.1, start_epoch=1),
                    BitFlipSTEWeightNoiseConfig(flip_p=0.001, start_epoch=1),
                ),
                start_epoch=1,
            ),
            "mixfam_v1_ep1",
        ),
        # Tails heavier than the device (nu=3, infinite kurtosis) at matched variance.
        (StudentTWeightNoiseConfig(dev=0.035, nu=3.0, start_epoch=1), "studentt0.035_nu3.0_ep1"),
        # Does the actdrop0 -0.1 (best 8-bit number, 5.9) stack on other families?
        (UniformBitNoiseConfig(bit_amplitude=0.1, start_epoch=1), "unifbit0.1_actdrop0_ep1"),
        (BitFlipSTEWeightNoiseConfig(flip_p=0.001, start_epoch=1), "bitflipste0.001_actdrop0_ep1"),
        # CLT probes (observations/noise_equivalence_argument.md #5). Clipped Cauchy:
        # truncated variance matched to gauss0.05 (gamma=0.34 LSB, clip q_max at w8);
        # unclipped: alpha-stable, tails survive ANY fan-in -- the door-1 test.
        (TruncatedCauchyNoiseConfig(cauchy_dev=0.00325, clip_qmax=1.0, start_epoch=1), "cauchydev0.00325_clip1_ep1"),
        (TruncatedCauchyNoiseConfig(cauchy_dev=0.0335, clip_qmax=0.0, start_epoch=1), "cauchydev0.0335_noclip_ep1"),
        # sigma-matched to gauss0.05 like clip1, but clipped at 10*q_max: kappa_w ~ 2e4
        # (100x clip1), still bf16-safe. The trainable stand-in for the noclip probe.
        (TruncatedCauchyNoiseConfig(cauchy_dev=0.000322, clip_qmax=10.0, start_epoch=1), "cauchydev0.000322_clip10_ep1"),
        # Per-tile ADC gain (128x128 = the eval tiling): n_eff 2048 -> 16, the
        # device-faithful midpoint between colgain (n_eff=1) and iid noise.
        (TileGainNoiseConfig(tile_gain=0.05, tile_size=128, start_epoch=1), "tilegain0.05_ep1"),
        (TileGainNoiseConfig(tile_gain=0.1, tile_size=128, start_epoch=1), "tilegain0.1_ep1"),
        # n_eff=1 shape triple: colgain0.05 (Gaussian, above) is the control arm; these
        # two put an undiluted uniform edge / the device's t-tails AT the pre-activation.
        (RowGainDistNoiseConfig(gain_dist="unif", gain_sigma=0.05, gain_nu=0.0, start_epoch=1), "rowgain_unif0.05_ep1"),
        (RowGainDistNoiseConfig(gain_dist="studentt", gain_sigma=0.05, gain_nu=5.9, start_epoch=1), "rowgain_t0.05_nu5.9_ep1"),
        # --- Wave 3 (2026-08-23): mechanism probes for the CLT story
        # (observations/wave3_clt_probes.md). LibriSpeech only for now.
        # Decoupled fwd/bwd: THE mechanistic test of "the backward shape is what matters".
        # forward_only = model sees noise, gradients never do; backward_only = the reverse.
        # dL/dW is untouched either way (additive eps); costs one extra matmul per noised op.
        (DecoupledGaussianNoiseConfig(decoupled_dev=0.05, mode="forward_only", start_epoch=1), "gauss0.05_fwdonly_ep1"),
        (DecoupledGaussianNoiseConfig(decoupled_dev=0.05, mode="backward_only", start_epoch=1), "gauss0.05_bwdonly_ep1"),
        # Tail-cliff bisect: nu=5.9 is at the floor, nu=3 (infinite kurtosis) killed even the
        # digital model. nu=4 sits exactly on the finite-kurtosis boundary.
        (StudentTWeightNoiseConfig(dev=0.035, nu=4.0, start_epoch=1), "studentt0.035_nu4.0_ep1"),
        # Correlation length: ts=128 (n_eff 16) failed like colgain (n_eff 1); iid (n_eff 2048)
        # is at the floor. ts=32 / ts=8 (n_eff 64 / 256) locate the recovery threshold.
        (TileGainNoiseConfig(tile_gain=0.05, tile_size=32, start_epoch=1), "tilegain0.05_ts32_ep1"),
        (TileGainNoiseConfig(tile_gain=0.05, tile_size=8, start_epoch=1), "tilegain0.05_ts8_ep1"),
        # Useless vs harmful: if correlated noise merely fails to help, colgain+gauss recovers
        # the iid floor (~6.0); if it actively damages training, the sum stays at ~7.5.
        (
            SumNoiseConfig(
                sum_configs=(
                    ColumnGainNoiseConfig(rel_gain=0.05, start_epoch=1),
                    GaussianWeightNoiseConfig(dev=0.05, start_epoch=1),
                ),
                start_epoch=1,
            ),
            "colgain0.05_gauss0.05_ep1",
        ),
        # Zero-mean probe with a hardware face: direction-asymmetric SET/RESET flips give a
        # non-zero-mean discrete noise; expected flip count matches bitflipste0.001 (~6.0/6.3).
        (AsymmetricBitFlipSTENoiseConfig(flip_p_set=0.002, flip_p_reset=0.0, start_epoch=1), "asymflip0.002_set_ep1"),
        (AsymmetricBitFlipSTENoiseConfig(flip_p_set=0.0, flip_p_reset=0.002, start_epoch=1), "asymflip0.002_reset_ep1"),
    ]
    # att_weights/conv/ff/mhsa/final dropout -> 0.0 for these variants (default stays 0.1).
    FINETUNE_NO_ACT_DROPOUT = {
        "gauss0.05_actdrop0_ep1", "gauss0.035_actdrop0_ep1",
        "unifbit0.1_actdrop0_ep1", "bitflipste0.001_actdrop0_ep1",
    }
    # Wave 2 control: seed-1 replicate of the reference config (w4 baseline seed spread
    # is 0.5, so no wave "win" is claimable without a training-seed error bar).
    FINETUNE_TWO_SEEDS = {"gauss0.05_ep1"}
    # Finetune weight dropout per variant; unlisted stays 0.0, so existing hashes are untouched.
    # NB the "_drop0.1" in the job names is a naming artefact -- it was hard-coded 0.0 for all.
    FINETUNE_WEIGHT_DROPOUT = {
        "wdrop0.05": 0.05,
        "wdrop0.1": 0.1,
        "wdrop0.2": 0.2,
        "wdrop0.05_gauss0.035_ep1": 0.05,
        "wdrop0.1_gauss0.02_ep1": 0.1,
        "wdrop0.05_gauss0.02_ep1": 0.05,
        "wdrop0.1_gauss0.035_ep1": 0.1,
        "wdrop0.05_gauss0.05_ep1": 0.05,
        "wdrop0.1_gauss0.05_ep1": 0.1,
    }
    # Format: finetune_epochs, dim, weight_bit, dropout, seed, noise_name
    diverged_list = [
        # unclipped alpha-stable noise: extreme draws -> inf activations -> quantizer
        # observer nan assert. Recorded outcome, not restartable (2026-08-18).
        (250, 512, 8, 0.1, 0, "cauchydev0.0335_noclip_ep1"),
        (250, 512, 4, 0.1, 0, "cauchydev0.0335_noclip_ep1"),
    ]
    for finetune_epochs in [10, 50, 100, 200, 250, 500, 750, 1000]:
        for activation_bit in activation_bits:
            for dim in [512]:
                for weight_bit in [8, 4]:
                    for dropout in [0.1]:
                        active_noise_configs_finetune = noise_configs_finetune + (
                            noise_configs_finetune_new if finetune_epochs == 250 else []
                        )
                        for noise_cfg, noise_name in active_noise_configs_finetune:
                            # was: seeds = 1 (all noise finetunes seed 0 only)
                            seeds = 2 if (finetune_epochs == 250 and noise_name in FINETUNE_TWO_SEEDS) else 1
                            if finetune_epochs not in [250]:
                                if "gauss" in noise_name and not "gauss0.05_ep1" in noise_name:
                                    continue
                            memristor_runs = 5 if dim <= 512 else 3
                            cfg_kwargs = _make_model_config_kwargs(dim, weight_noise=noise_cfg)
                            if noise_name in FINETUNE_NO_ACT_DROPOUT:
                                cfg_kwargs.update(
                                    att_weights_dropout=0.0, conv_dropout=0.0, ff_dropout=0.0,
                                    mhsa_dropout=0.0, final_dropout=0.0,
                                )
                            model_config = MemristorModelTrainConfigV15(
                                **cfg_kwargs,
                                weight_bit_prec=weight_bit,
                                activation_bit_prec=activation_bit,
                                pos_enc_converter_hardware_settings=prior_train_dac_settings,
                                weight_pruning=None,
                                # was: weight_dropout=0.0 (hard-coded for every variant)
                                weight_dropout=FINETUNE_WEIGHT_DROPOUT.get(noise_name, 0.0),
                            )
                            model_config.module_list =["ff", "mhsa","conv", "ff"]
                            for seed in range(seeds):
                                if (finetune_epochs, dim, weight_bit, dropout, seed, noise_name) in diverged_list or finetune_epochs < 200:
                                    memristor_report[
                                        prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}"] = "Diverged"
                                    continue
                                if "bitflip0" in noise_name:
                                    memristor_report[
                                        prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}"] = "Diverged"
                                    continue
                                baseline_prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/bpe_ls960_memristor/noise"
                                base_checkpoint_name = baseline_prefix + "/" + network_module_mem_v10 + f"_{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                                train_config_24gbgpu = {
                                    "optimizer": {
                                        "class": "radam",
                                        "epsilon": 1e-12,
                                        "weight_decay": 1e-2,
                                        "decoupled_weight_decay": True,
                                    },
                                    "learning_rates": list(np.linspace(7e-6, 1e-4, finetune_epochs // 2)) + list(np.linspace(1e-4, 1e-7, finetune_epochs // 2)),
                                    "batch_size": 360 * 16000,
                                    "max_seq_length": {"audio_features": 35 * 16000},
                                    "accum_grad_multiple_step": 1,
                                    "gradient_clip_norm": 1.0,
                                    "seed": seed,
                                    "torch_amp_options": {"dtype": "bfloat16"},
                                    "preload_from_files": {
                                        "model": {
                                            "filename": FINETUNE_MODELS[base_checkpoint_name],
                                            "init_for_train": True,
                                            "ignore_missing": False,
                                        }
                                    },
                                }
                                train_args = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v15,
                                    "net_args": {"model_config_dict": asdict(model_config)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }
                                training_name = prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}"
                                train_args_training = train_args
                                if isinstance(noise_cfg, SynaptogenPoolNoiseConfig):
                                    # the lazy pool build needs synaptogen_ml importable inside the
                                    # training job; same pinned clone as the cycle evals. Kept out of
                                    # train_args so the prior/eval configs stay unchanged.
                                    train_args_training = {**train_args, "import_memristor": "new_v3"}
                                train_job = training(training_name, train_data_bpe128, train_args_training,
                                                     num_epochs=finetune_epochs, **default_returnn)
                                # HPC back up (2026-08-25): everything returns to the held
                                # HPC chain path EXCEPT the five finetunes already mid-run on
                                # the local 48 GB GPUs (started 2026-08-24, ~55% done) --
                                # those keep their local settings so they finish in place.
                                LOCAL_RUNNING = {
                                    ("rowgain_unif0.05_ep1", 4),
                                    ("colgain0.05_gauss0.05_ep1", 4),
                                    ("colgain0.05_gauss0.05_ep1", 8),
                                    ("tilegain0.05_ts8_ep1", 8),
                                    ("asymflip0.002_set_ep1", 4),
                                }
                                if not os.path.exists(f"{train_job._sis_path()}/finished.run.1") and (noise_name, weight_bit) in LOCAL_RUNNING:
                                    # keep the local 48 GB settings from the HPC-down window
                                    train_job.rqmt['gpu_mem'] = 48
                                    train_job.rqmt['mem'] = 36
                                    train_job.rqmt['cpu'] = 8
                                    train_job.has_priority = True
                                    train_job.rqmt['time'] = 96
                                elif not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):
                                    train_job.rqmt['cpu'] = 12
                                    train_job.hold()
                                    # 25-ep finetunes need ~25-30 h on CLAIX (July 2026 speed regime)
                                    train_job.rqmt['time'] = 96
                                    if isinstance(
                                        noise_cfg,
                                        (SynaptogenPoolNoiseConfig, AsymmetricBitNoiseConfig,
                                         StudentTWeightNoiseConfig),
                                    ) or noise_name in [
                                        "gauss0.035_ep1",
                                        "gauss0.02_ep1",
                                        "gauss0.04_ep1",
                                        "gauss0.06_ep1",
                                        "bitflipste0.0003_ep1",
                                        "bitflipste0.001_ep1",
                                        "bitflipste0.003_ep1",
                                        # allocation-axis batch: fused gauss / dropout / relgauss
                                        "relgauss0.3_ep1",
                                        "wdrop0.05",
                                        "wdrop0.1",
                                        "wdrop0.2",
                                        "wdrop0.05_gauss0.035_ep1",
                                        "wdrop0.1_gauss0.02_ep1",
                                        "wdrop0.05_gauss0.02_ep1",
                                        "wdrop0.1_gauss0.035_ep1",
                                        "wdrop0.05_gauss0.05_ep1",
                                        "wdrop0.1_gauss0.05_ep1",
                                        "gauss0.05_actdrop0_ep1",
                                        "gauss0.035_actdrop0_ep1",
                                        # unifbit runs at 0.30 s/step, same as synpool and
                                        # asymbit, so 3 segments are enough here too.
                                        "unifbit0.1_ep1",
                                        "unifbit0.15_ep1",
                                        # Wave 2 fused-cost variants (stuck/mixfam stay at
                                        # the 96h default: bit-level / mixed cost)
                                        "gausshold0.035_ep1",
                                        "gausshold0.05_ep1",
                                        "gaussramp0.05_ep1",
                                        "gaussramp0.07_ep1",
                                        "gaussrand0.02-0.08_ep1",
                                        "colgain0.02_ep1",
                                        "colgain0.05_ep1",
                                        "unifbit0.1_actdrop0_ep1",
                                        "bitflipste0.001_actdrop0_ep1",
                                        "cauchydev0.00325_clip1_ep1",
                                        "cauchydev0.0335_noclip_ep1",
                                        "cauchydev0.000322_clip10_ep1",
                                        "tilegain0.05_ep1",
                                        "tilegain0.1_ep1",
                                        "rowgain_unif0.05_ep1",
                                        "rowgain_t0.05_nu5.9_ep1",
                                        # Wave 3 fused-cost variants (decoupled gauss doubles the
                                        # noised matmuls -> stays at the 96h default; asymflip is
                                        # bitflipste-cost, tile/sum are fused-gauss-cost)
                                        "tilegain0.05_ts32_ep1",
                                        "tilegain0.05_ts8_ep1",
                                        "colgain0.05_gauss0.05_ep1",
                                        "asymflip0.002_set_ep1",
                                        "asymflip0.002_reset_ep1",
                                    ]:
                                        # fast noises (~0.30-0.32 s/step: pool noise, fused gauss,
                                        # bitflipste at +10%) finish in ~26-28 h -> 36 h = 3 chain segments
                                        train_job.rqmt['time'] = 36
                                    train_job.move_to_hpc = True
                                    # submit as a chain of ceil(36/12)=3 x 12h segments on the
                                    # c25g queue (--dependency=afterany, see sis_itc_helper.submit_flags):
                                    # instant scheduling, no manual resubmission at walltime kills
                                    train_job.use_new_partition = True

                                prior_config = copy.deepcopy(model_config)
                                prior_config.weight_noise = None
                                prior_args = copy.deepcopy(train_args)
                                prior_args["net_args"] = {"model_config_dict": asdict(prior_config)}

                                if noise_cfg is not None:
                                    # with_noise eval is meaningless for the no-noise control
                                    # (identical config to without_noise)
                                    results = {}
                                    results, best_params_job_noise = eval_model(
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

                                # run_memristor_cycle_eval(
                                #     train_job=train_job,
                                #     train_data=train_data_bpe128,
                                #     train_config=train_config_24gbgpu,
                                #     model_config=prior_config,
                                #     recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}",
                                #     rasr_config=rasr_config_memristor,
                                #     greedy_config=None,
                                #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                #     prior_scales=[best_params_job.out_optimal_parameters[1]],
                                #     lm_scales=[(best_params_job.out_optimal_parameters[0], "best_nonoise")],
                                #     batch_size=3500000 if weight_bit not in [8] else 2500000,
                                #     max_runs=memristor_runs,
                                #     report_dict=memristor_report,
                                #     prior_network_module=network_module_mem_v15,
                                #     recog_network_module=network_module_mem_v15,
                                #     recog_model_config_class=MemristorModelTrainConfigV15,
                                #     final_name=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_best_nonoise_cycle",
                                # )
                                # run_memristor_cycle_eval(
                                #     train_job=train_job,
                                #     train_data=train_data_bpe128,
                                #     train_config=train_config_24gbgpu,
                                #     model_config=prior_config,
                                #     recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}",
                                #     rasr_config=rasr_config_memristor,
                                #     greedy_config=None,
                                #     dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                #     prior_scales=[best_params_job_noise.out_optimal_parameters[1]],
                                #     lm_scales=[(best_params_job_noise.out_optimal_parameters[0], "best_noise")],
                                #     batch_size=3500000 if weight_bit not in [8] else 2500000,
                                #     max_runs=memristor_runs,
                                #     report_dict=memristor_report,
                                #     prior_network_module=network_module_mem_v15,
                                #     recog_network_module=network_module_mem_v15,
                                #     recog_model_config_class=MemristorModelTrainConfigV15,
                                #     final_name=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_best_noise_cycle",
                                # )
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=prior_config,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=memristor_runs,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v15,
                                    recog_network_module=network_module_mem_v15,
                                    recog_model_config_class=MemristorModelTrainConfigV15,
                                    final_name=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_cycle",
                                    fast_inference=True,
                                )
                                if finetune_epochs == 250 and noise_name in [
                                    "gauss0.05_ep1",
                                    "gauss0.1_ep1",
                                    "gauss0.01_ep1",
                                    "gaussw0.05_ep1",
                                    "unifbit0.05_ep1",
                                    "unifw0.05_ep1",
                                    "relgauss0.05_ep1",
                                    # bitmix0.12_ep1 NOT listed: its config is commented
                                    # out of the training loop above, so a guard entry
                                    # would never match
                                    "bitflipste0.01_ep1",
                                ]:
                                    # cell-programming speedup A/B (v5 pin via the
                                    # "_newsynap_progfast_" match + SYN_FAST_PROG in the
                                    # conversion job): re-programs the 9 monitored 250ep noise
                                    # finetunes with the parallel programming path, full 5
                                    # cycles per weight bit (NOT the unguarded 250ep block --
                                    # that would twin ~20 configs and double the intended job
                                    # count). New conversion jobs by design (independent RNG
                                    # realization); WERs compare against the finished fast
                                    # runs above. Rounds 1+2 (gauss0.05/unifbit0.05/
                                    # gaussw0.05): conv x6.3-9.7, per-config mean |dWER|
                                    # <= 0.06. bitmix/bitflipste deltas are not interpretable
                                    # (broken configs, WER 43-98). rasr-only.
                                    run_memristor_cycle_eval(
                                        train_job=train_job,
                                        train_data=train_data_bpe128,
                                        train_config=train_config_24gbgpu,
                                        model_config=prior_config,
                                        recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_newsynap_progfast_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}",
                                        rasr_config=rasr_config_memristor,
                                        greedy_config=None,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        prior_scales=[0.5],
                                        lm_scales=[0.8],
                                        batch_size=3500000 if weight_bit not in [8] else 2500000,
                                        max_runs=memristor_runs,
                                        report_dict=memristor_report,
                                        prior_network_module=network_module_mem_v15,
                                        recog_network_module=network_module_mem_v15,
                                        recog_model_config_class=MemristorModelTrainConfigV15,
                                        final_name=prefix_name + "/" + network_module_mem_v15 + f"_newsynap_progfast_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_cycle",
                                        fast_inference=True,
                                    )
                                if (
                                    finetune_epochs == 250 and weight_bit == 4 and seed == 0
                                    and noise_name in ["gauss0.05_ep1", "nonoise"]
                                ):
                                    # parallel no_outq eval (out-quants -> nn.Identity) for the
                                    # best noise finetune and its no-noise control; conversion/
                                    # prior jobs shared with the run above, results land under
                                    # <prefix>_no_outq
                                    run_memristor_cycle_eval(
                                        train_job=train_job,
                                        train_data=train_data_bpe128,
                                        train_config=train_config_24gbgpu,
                                        model_config=prior_config,
                                        recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}",
                                        rasr_config=rasr_config_memristor,
                                        greedy_config=greedy_decoder_memristor,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        prior_scales=[0.5],
                                        lm_scales=[0.8],
                                        batch_size=3500000,
                                        max_runs=memristor_runs,
                                        report_dict=memristor_report,
                                        prior_network_module=network_module_mem_v15,
                                        recog_network_module=network_module_mem_v15,
                                        recog_model_config_class=MemristorModelTrainConfigV15,
                                        final_name=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_cycle",
                                        fast_inference=True,
                                        no_outq=True,
                                    )
                                if finetune_epochs == 250 and noise_name == "gauss0.05_ep1":
                                    # single-forward multi-scale sweep (paper numbers) for the best
                                    # (Gaussian) noise finetune. Recognises with prior_config
                                    # (weight_noise off) and reuses the cached conversion/prior jobs
                                    # of the single-point cycle above. Prior trimmed to the noise
                                    # good region [0.2-0.6] (gauss sweep optima at prior 0.3-0.4).
                                    run_memristor_cycle_eval(
                                        train_job=train_job,
                                        train_data=train_data_bpe128,
                                        train_config=train_config_24gbgpu,
                                        model_config=prior_config,
                                        recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_multi_sweep_seed_{seed}",
                                        rasr_config=rasr_config_memristor,
                                        greedy_config=None,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        prior_scales=[0.2, 0.3, 0.4, 0.5, 0.6],
                                        lm_scales=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                                        batch_size=3500000,
                                        max_runs=memristor_runs,
                                        report_dict=memristor_report,
                                        prior_network_module=network_module_mem_v15,
                                        recog_network_module=network_module_mem_v15,
                                        recog_model_config_class=MemristorModelTrainConfigV15,
                                        search_gpu=48,
                                        run_rasr_multi=True,
                                        num_search_workers=9,
                                        fast_inference=True,
                                    )

                                model_config_ideal = copy.deepcopy(prior_config)
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
                                run_memristor_cycle_eval(
                                    train_job=train_job,
                                    train_data=train_data_bpe128,
                                    train_config=train_config_24gbgpu,
                                    model_config=model_config_ideal,
                                    recog_name_prefix=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_ideal_seed_{seed}",
                                    rasr_config=rasr_config_memristor,
                                    greedy_config=greedy_decoder_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    prior_scales=[0.5],
                                    lm_scales=[0.8],
                                    batch_size=3500000 if weight_bit not in [8] else 2500000,
                                    max_runs=2,
                                    report_dict=memristor_report,
                                    prior_network_module=network_module_mem_v15,
                                    recog_network_module=network_module_mem_v15,
                                    recog_model_config_class=MemristorModelTrainConfigV15,
                                    final_name=prefix_name + "/" + network_module_mem_v15 + f"_{finetune_epochs // 10}eps_from{1000 // 10}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}_ideal_fixed_cycle",
                                    recog_dac_settings=train_dac_settings_ideal,
                                    posenc_dac_settings=posenc_dac_settings_ideal,
                                    fast_inference=True,
                                )

    tk.register_report("reports/lbs/v2/memristor_noise_bpe", partial(build_qat_report_v2, memristor_report),
                       required=memristor_report, update_frequency=400)
