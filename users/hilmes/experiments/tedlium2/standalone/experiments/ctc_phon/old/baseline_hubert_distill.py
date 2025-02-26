from dataclasses import asdict
import numpy as np
from typing import cast
import copy
import itertools

from sisyphus import tk
from onnxruntime.quantization.quantize import QuantType, QuantFormat
from onnxruntime.quantization.calibrate import CalibrationMethod

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, generate_kd_hypothesis, calculate_blank_counts, calculate_blank_ratios
from ...report import generate_report
from .tune_eval import eval_model, build_hubert_report, build_hubert_distill_report
from functools import partial


def get_quant_str(num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops, quant_format):
    if quant_mode == CalibrationMethod.MinMax:
        mode_str = "/quant/min_max"
    elif quant_mode == CalibrationMethod.Entropy:
        mode_str = "quant/entropy"
    else:
        mode_str = "quant/percentile"
    mode_str += f"/{num_seqs}"
    for x in [activation_type, weight_type]:
        if x == QuantType.QInt8:
            mode_str += "_QInt8"
        elif x == QuantType.QUInt8:
            mode_str += "_QUint8"
    if average:
        mode_str += "_avg"
    if sym:
        mode_str += "_sym"
    if quant_ops is not None:
        mode_str += "_" + "_".join(quant_ops)
    else:
        mode_str += "_full"
    if quant_format == QuantFormat.QDQ:
        mode_str += "_QDQ"
    else:
        mode_str += "QOperator"
    return mode_str


def eow_phon_ted_tune_hubert():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/tune_hubert"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=train_settings,
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

    train_dataset_tuples = {}
    for testset in ["train"]:
        train_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )
    # no_lm_decoder_config = DecoderConfig(
    #     lexicon=get_text_lexicon(),
    #     returnn_vocab=label_datastream.vocab,
    #     beam_size=1024,
    #     beam_size_token=12,  # makes it much faster
    #     arpa_lm=None,
    #     beam_threshold=14,
    # )

    from ...pytorch_networks.ctc.hubert_tune_0711.hubert_tune_v1_cfg import ModelConfig

    # "large-ls960-ft" "large-ll60k" "xlarge-ll60k" "base-ll60k"
    hubert_report = {}
    checkpoints = {}
    # for model in ["base-ls960", "large-ls960-ft", "large-ll60k", "xlarge-ll60k", "xlarge-ls960-ft", , "xlarge-ls960-ft"]:
    for model in ["large-ll60k"]:
        model_config = ModelConfig(
            label_target_size=vocab_size_without_blank,
            final_dropout=0.2,
            model_name=model,
            finetune_layer=True,
            keep_layers=None,
        )
        network_module = "ctc.hubert_tune_0711.hubert_tune_v1"
        keep_epochs = [10, 20, 30, 40, 50]
        train_config_24gbgpu_amp = {
            "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
            + list(np.linspace(5e-4, 5e-5, 110))
            + list(np.linspace(5e-5, 1e-7, 30)),
            #############
            "batch_size": 120 * 16000 if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 60 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 3 if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 6,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 4,
                "keep": keep_epochs,
            },
            "torch_amp_options": {"dtype": "bfloat16"},
        }
        if model in ["xlarge-ll60k", "xlarge-ls960-ft"]:
            train_config_24gbgpu_amp["max_seqs"] = 1
        train_args_amp = {
            "config": train_config_24gbgpu_amp,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
        }

        training_name = prefix_name + "/" + network_module + f"_{model}"
        train_job = training(training_name, train_data, train_args_amp, num_epochs=50, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args_amp,
            train_data=train_data,
            decoder_config=default_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            specific_epoch=keep_epochs,
            prior_scales=[0.1, 0.2, 0.3, 0.4],
            lm_scales=[1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 1.9, 2.0],
            test_dataset_tuples=test_dataset_tuples,
            run_test=True,
        )
        generate_report(results=results, exp_name=training_name)
        hubert_report[training_name] = results
        del results
        if model == "large-ll60k" or model == "xlarge-ll60k":
            from ...pytorch_networks.ctc.hubert_tune_0711.hubert_tune_v2_cfg import ModelConfig as ModelConfigV2

            network_module_v2 = "ctc.hubert_tune_0711.hubert_tune_v2"
            model_config = ModelConfigV2(
                label_target_size=vocab_size_without_blank,
                final_dropout=0.2,
                model_name=model,
                finetune_layer=True,
                keep_layers=None,
                downsample_factor=2,
            )
            keep_epochs = [10, 20, 30, 40, 50]
            train_config_24gbgpu_amp = {
                "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
                "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                + list(np.linspace(5e-4, 5e-5, 110))
                + list(np.linspace(5e-5, 1e-7, 30)),
                #############
                "batch_size": 120 * 16000 if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 15 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 3 if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 24,
                "cleanup_old_models": {
                    "keep_last_n": 4,
                    "keep_best_n": 4,
                    "keep": keep_epochs,
                },
                "torch_amp_options": {"dtype": "bfloat16"},
            }
            train_args_amp = {
                "config": train_config_24gbgpu_amp,
                "network_module": network_module_v2,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": True,
            }
            training_name = prefix_name + "/" + network_module_v2 + f"_{model}_sub4"
            train_job = training(training_name, train_data, train_args_amp, num_epochs=50, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            results = eval_model(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args_amp,
                train_data=train_data,
                decoder_config=default_decoder_config,
                dev_dataset_tuples=dev_dataset_tuples,
                specific_epoch=keep_epochs,
                prior_scales=[0.1, 0.3, 0.4, 0.5],
                lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                test_dataset_tuples=test_dataset_tuples,
                run_test=True,
            )
            hubert_report[training_name] = results
            generate_report(results=results, exp_name=training_name)

            del results

            if model == "large-ll60k":
                # for checkpoint in [33, 50, "best4"]:
                for checkpoint in [33]:
                    for n_best in [10]:
                        threshold = 30 if n_best == 5 else 50
                        beam_size_token = 30 if n_best == 5 else 40
                        from ...pytorch_networks.ctc.decoder.flashlight_ctc_kdhyps import (
                            DecoderConfig as DecoderConfigV2,
                        )

                        kd_decoder_config = DecoderConfigV2(
                            lexicon=get_text_lexicon(),
                            returnn_vocab=label_datastream.vocab,
                            beam_size=1024,
                            beam_size_token=14,  # makes it much faster
                            arpa_lm=arpa_4gram_lm,
                            beam_threshold=threshold,
                            n_best_probs=n_best,
                            add_reference=True,
                            length_norm=True,
                        )
                        pref_name = training_name + f"/{checkpoint}_{n_best}"
                        kd_hyp, prior_file, model_checkpoint = generate_kd_hypothesis(
                            prefix_name=pref_name,
                            train_job=train_job,
                            train_args=train_args_amp,
                            train_data=train_data,
                            checkpoint=checkpoint,
                            decoder_config=kd_decoder_config,
                            prior_scale=0.5,
                            lm_scale=2.0,
                            train_referece=train_dataset_tuples["train"][1],
                            debug=True,
                        )
                        checkpoints[model + f"_{checkpoint}_{n_best}"] = (model_checkpoint, kd_hyp, prior_file, 0.5)
                    blank_counts = calculate_blank_counts(
                        prefix_name=training_name,
                        train_job=train_job,
                        train_args=train_args_amp,
                        train_data=train_data,
                        checkpoint=checkpoint,
                        debug=True,
                    )
                    tk.register_output(training_name + "/" + "blank_counts", blank_counts)

    tk.register_report(
        "reports/finetune_hubert_report",
        partial(build_hubert_report, hubert_report),
        required=hubert_report,
        update_frequency=900,
    )

    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v2_cfg import (
        ModelConfig as StudentConfigV2,
        DistillConfig as TeacherConfigV2,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        SpecaugConfig,
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
        max_dim_feat=8,
        num_repeat_feat=5,  # Jingjing style
    )
    distill_module_v3 = "ctc.hubert_tune_0711.distill_hubert_v3"
    distill_module_v4 = "ctc.hubert_tune_0711.distill_hubert_v4"
    distill_module_v5 = "ctc.hubert_tune_0711.distill_hubert_v5"
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v5_cfg import DistillConfig as TeacherConfigV5
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v6_cfg import DistillConfig as TeacherConfigV6
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v7_cfg import DistillConfig as TeacherConfigV7
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v8_cfg import DistillConfig as TeacherConfigV8
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v9_cfg import DistillConfig as TeacherConfigV9
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v10_cfg import DistillConfig as TeacherConfigV10
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v11_cfg import DistillConfig as TeacherConfigV11
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v12_cfg import DistillConfig as TeacherConfigV12
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v13_cfg import DistillConfig as TeacherConfigV13

    distill_module_v6 = "ctc.hubert_tune_0711.distill_hubert_v6"
    distill_module_v7 = "ctc.hubert_tune_0711.distill_hubert_v7"
    distill_module_v8 = "ctc.hubert_tune_0711.distill_hubert_v8"
    distill_module_v9 = "ctc.hubert_tune_0711.distill_hubert_v9"
    distill_module_v10 = "ctc.hubert_tune_0711.distill_hubert_v10"
    distill_module_v11 = "ctc.hubert_tune_0711.distill_hubert_v11"
    distill_module_v12 = "ctc.hubert_tune_0711.distill_hubert_v12"
    distill_module_v13 = "ctc.hubert_tune_0711.distill_hubert_v13"
    distill_report = {}
    distill_report["baselines"] = {}
    from .distill_auxloss import eow_phon_ted_auxloss_distill

    baselines, base_checkpoints = eow_phon_ted_auxloss_distill(get_report=True)
    baseline_prefix = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/distill_auxloss"
    baseline_module = "ctc.conformer_0106.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1"
    for name, (chkpt, kd_hyps, prior_file, prior_scale) in checkpoints.items():
        name, num, n_best = name.split("_")
        for dim in [384]:
            for layer_count in [12]:
                for distill_scale in [0.25, 1.0, 0.9]:
                    for T in [2]:
                        distill_report["baselines"][
                            baseline_prefix + "/" + baseline_module + f"_{layer_count}_{dim}"
                        ] = baselines[baseline_prefix + "/" + baseline_module + f"_{layer_count}_{dim}"]
                        for drop in [0.1, 0.0]:
                            distill_report["baselines"][
                                baseline_prefix + "/" + baseline_module + f"_{layer_count}_{dim}_drop{drop}"
                            ] = baselines[baseline_prefix + "/" + baseline_module + f"_{layer_count}_{dim}_drop{drop}"]
                        teacher_config = TeacherConfigV2(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            exp_targets=False,
                            eliminate_blanks=False,
                            model_name=name,
                            kd_hyps=None,
                            normalize_stud=False,
                        )
                        frontend_config_student = VGG4LayerActFrontendV1Config_mod(
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

                        student_config = StudentConfigV2(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config_student,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=dim,
                            num_layers=layer_count,
                            num_heads=4,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.2,
                            conv_dropout=0.2,
                            ff_dropout=0.2,
                            mhsa_dropout=0.2,
                            conv_kernel_size=31,
                            final_dropout=0.2,
                            specauc_start_epoch=1,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                            aux_ctc_loss_scales=[0.3, 0.3, 1.0],
                        )

                        train_config_distill = {
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
                        }
                        if dim == 384:
                            train_config_distill["batch_size"] /= 2
                            train_config_distill["accum_grad_multiple_step"] *= 2
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v3,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                        }
                        train_args_distill["config"]["preload_from_files"] = {
                            "teacher": {
                                "filename": chkpt,
                                "init_for_train": True,
                                "ignore_missing": False,
                                "prefix": "teacher.",
                                "ignore_params_prefixes": [],
                            }
                        }
                        model_config_decoding = copy.deepcopy(student_config)
                        model_config_decoding.aux_ctc_loss_scales = [
                            0.0,
                            0.0,
                            1.0,
                        ]  # for decoding use result only of last layer
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(model_config_decoding),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"chkpt_{num}"
                            + f"_{layer_count}_{dim}_{distill_scale}_{T}"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=250,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=[0.3, 0.5, 0.7, 0.9],
                            lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                        teacher_config = TeacherConfigV2(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            exp_targets=False,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v3,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": True,
                        }
                        train_args_distill["config"]["preload_from_files"] = {
                            "teacher": {
                                "filename": chkpt,
                                "init_for_train": True,
                                "ignore_missing": False,
                                "prefix": "teacher.",
                                "ignore_params_prefixes": [],
                            }
                        }
                        model_config_decoding = copy.deepcopy(student_config)
                        model_config_decoding.aux_ctc_loss_scales = [
                            0.0,
                            0.0,
                            1.0,
                        ]  # for decoding use result only of last layer
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(model_config_decoding),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"chkpt_{num}"
                            + f"_{layer_count}_{dim}_{distill_scale}_{T}_elim_blank"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=250,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=[0.3, 0.5, 0.7, 0.9],
                            lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                        for dropout in [0.0, 0.1]:
                            teacher_config = TeacherConfigV2(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                exp_targets=False,
                                eliminate_blanks=False,
                                model_name=name,
                                normalize_stud=False,
                                kd_hyps=None,
                            )
                            student_config_drop = StudentConfigV2(
                                feature_extraction_config=fe_config,
                                frontend_config=frontend_config_student,
                                specaug_config=specaug_config,
                                label_target_size=vocab_size_without_blank,
                                conformer_size=dim,
                                num_layers=layer_count,
                                num_heads=4,
                                ff_dim=4 * dim,
                                att_weights_dropout=dropout,
                                conv_dropout=dropout,
                                ff_dropout=dropout,
                                mhsa_dropout=dropout,
                                conv_kernel_size=31,
                                final_dropout=dropout,
                                specauc_start_epoch=1,
                                module_list=["ff", "conv", "mhsa", "ff"],
                                module_scales=[0.5, 1.0, 1.0, 0.5],
                                aux_ctc_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                                aux_ctc_loss_scales=[0.3, 0.3, 1.0],
                            )

                            train_config_distill = {
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
                            }
                            if dim == 384:
                                train_config_distill["batch_size"] /= 2
                                train_config_distill["accum_grad_multiple_step"] *= 2
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v3,
                                "net_args": {
                                    "model_config_dict": asdict(student_config_drop),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config_drop)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v3
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_drop{dropout}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                            teacher_config = TeacherConfigV2(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                exp_targets=False,
                                eliminate_blanks=True,
                                model_name=name,
                                normalize_stud=False,
                                kd_hyps=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v3,
                                "net_args": {
                                    "model_config_dict": asdict(student_config_drop),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config_drop)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v3
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_drop{dropout}_elim_blank"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                            for keep in [1, 3, 5]:
                                teacher_config = TeacherConfigV6(
                                    distill_scale=distill_scale,
                                    ctc_scale=1 - distill_scale,
                                    t=T,
                                    eliminate_blanks=True,
                                    keep_some_blanks=keep,
                                    model_name=name,
                                    kd_hyps=None,
                                    normalize_stud=False,
                                    prior_file=None,
                                    prior_scale=None,
                                    warmup_loss=None,
                                    mask_padding=False,
                                )
                                train_args_distill = {
                                    "config": train_config_distill,
                                    "network_module": distill_module_v6,
                                    "net_args": {
                                        "model_config_dict": asdict(student_config_drop),
                                        "distill_config_dict": asdict(teacher_config),
                                    },
                                    "debug": True,
                                }
                                train_args_distill["config"]["preload_from_files"] = {
                                    "teacher": {
                                        "filename": chkpt,
                                        "init_for_train": True,
                                        "ignore_missing": False,
                                        "prefix": "teacher.",
                                        "ignore_params_prefixes": [],
                                    }
                                }
                                model_config_decoding = copy.deepcopy(student_config_drop)
                                model_config_decoding.aux_ctc_loss_scales = [
                                    0.0,
                                    0.0,
                                    1.0,
                                ]  # for decoding use result only of last layer
                                train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                train_args_distill_decoding["net_args"] = {
                                    "model_config_dict": asdict(model_config_decoding),
                                    "distill_config_dict": None,
                                }
                                del train_args_distill_decoding["config"]["preload_from_files"]

                                decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + distill_module_v6
                                    + f"chkpt_{num}"
                                    + f"_{layer_count}_{dim}_{distill_scale}_{T}_drop{dropout}_keepsome{keep}"
                                )
                                train_job = training(
                                    training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                                )
                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args_distill_decoding,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    specific_epoch=250,
                                    decoder_module=decoder_module,
                                    loss_name=f"ctc_loss_layer{layer_count}",
                                    prior_scales=[0.3, 0.5, 0.7, 0.9],
                                    lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                )
                                generate_report(results=results, exp_name=training_name)
                                distill_report[training_name] = results
                                del results

                                teacher_config = TeacherConfigV10(
                                    distill_scale=distill_scale,
                                    ctc_scale=1 - distill_scale,
                                    t=T,
                                    eliminate_blanks=True,
                                    keep_some_blanks=(keep, keep),
                                    model_name=name,
                                    kd_hyps=None,
                                    normalize_stud=False,
                                    prior_file=None,
                                    prior_scale=None,
                                    warmup_loss=None,
                                    mask_padding=False,
                                    trim_blanks=False,
                                    mix_nonblank=None,
                                    mix_blank=None,
                                )
                                train_args_distill = {
                                    "config": train_config_distill,
                                    "network_module": distill_module_v10,
                                    "net_args": {
                                        "model_config_dict": asdict(student_config_drop),
                                        "distill_config_dict": asdict(teacher_config),
                                    },
                                    "debug": True,
                                }
                                train_args_distill["config"]["preload_from_files"] = {
                                    "teacher": {
                                        "filename": chkpt,
                                        "init_for_train": True,
                                        "ignore_missing": False,
                                        "prefix": "teacher.",
                                        "ignore_params_prefixes": [],
                                    }
                                }
                                model_config_decoding = copy.deepcopy(student_config_drop)
                                model_config_decoding.aux_ctc_loss_scales = [
                                    0.0,
                                    0.0,
                                    1.0,
                                ]  # for decoding use result only of last layer
                                train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                train_args_distill_decoding["net_args"] = {
                                    "model_config_dict": asdict(model_config_decoding),
                                    "distill_config_dict": None,
                                }
                                del train_args_distill_decoding["config"]["preload_from_files"]

                                decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + distill_module_v6
                                    + f"chkpt_{num}"
                                    + f"_{layer_count}_{dim}_{distill_scale}_{T}_drop{dropout}_keepsome{keep}_sym"
                                )
                                train_job = training(
                                    training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                                )
                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args_distill_decoding,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    specific_epoch=250,
                                    decoder_module=decoder_module,
                                    loss_name=f"ctc_loss_layer{layer_count}",
                                    prior_scales=[0.3, 0.5, 0.7, 0.9],
                                    lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                )
                                generate_report(results=results, exp_name=training_name)
                                distill_report[training_name] = results
                                del results

                        teacher_config = TeacherConfigV2(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            exp_targets=False,
                            eliminate_blanks=False,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                        )
                        student_config = StudentConfigV2(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config_student,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=dim,
                            num_layers=layer_count,
                            num_heads=4,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.2,
                            conv_dropout=0.2,
                            ff_dropout=0.2,
                            mhsa_dropout=0.2,
                            conv_kernel_size=31,
                            final_dropout=0.2,
                            specauc_start_epoch=1,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                            aux_ctc_loss_scales=[0.3, 0.3, 1.0],
                        )

                        train_config_distill_nowd = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-16,
                                # "weight_decay": 1e-2,
                                # "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                            + list(np.linspace(5e-4, 5e-5, 110))
                            + list(np.linspace(5e-5, 1e-7, 30)),
                            #############
                            "batch_size": 180 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                        }
                        if dim == 384:
                            train_config_distill_nowd["batch_size"] /= 2
                            train_config_distill_nowd["accum_grad_multiple_step"] *= 2
                        train_args_distill = {
                            "config": train_config_distill_nowd,
                            "network_module": distill_module_v3,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                        }
                        train_args_distill["config"]["preload_from_files"] = {
                            "teacher": {
                                "filename": chkpt,
                                "init_for_train": True,
                                "ignore_missing": False,
                                "prefix": "teacher.",
                                "ignore_params_prefixes": [],
                            }
                        }
                        model_config_decoding = copy.deepcopy(student_config)
                        model_config_decoding.aux_ctc_loss_scales = [
                            0.0,
                            0.0,
                            1.0,
                        ]  # for decoding use result only of last layer
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(model_config_decoding),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"chkpt_{num}"
                            + f"_{layer_count}_{dim}_{distill_scale}_{T}_no_wdecay"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=250,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=[0.3, 0.5, 0.7, 0.9],
                            lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                        teacher_config = TeacherConfigV2(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            exp_targets=False,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                        )
                        train_args_distill = {
                            "config": train_config_distill_nowd,
                            "network_module": distill_module_v3,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": True,
                        }
                        train_args_distill["config"]["preload_from_files"] = {
                            "teacher": {
                                "filename": chkpt,
                                "init_for_train": True,
                                "ignore_missing": False,
                                "prefix": "teacher.",
                                "ignore_params_prefixes": [],
                            }
                        }
                        model_config_decoding = copy.deepcopy(student_config)
                        model_config_decoding.aux_ctc_loss_scales = [
                            0.0,
                            0.0,
                            1.0,
                        ]  # for decoding use result only of last layer
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(model_config_decoding),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"chkpt_{num}"
                            + f"_{layer_count}_{dim}_{distill_scale}_{T}_no_wdecay_elim_blank"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=250,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=[0.3, 0.5, 0.7, 0.9],
                            lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                        for feat, time in [(2, 12)]:
                            teacher_config = TeacherConfigV2(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                exp_targets=False,
                                eliminate_blanks=False,
                                model_name=name,
                                normalize_stud=False,
                                kd_hyps=None,
                            )
                            specaug_config_less = SpecaugConfig(
                                repeat_per_n_frames=time,
                                max_dim_time=20,
                                max_dim_feat=8,
                                num_repeat_feat=feat,  # Jingjing style
                            )
                            student_config_spec = StudentConfigV2(
                                feature_extraction_config=fe_config,
                                frontend_config=frontend_config_student,
                                specaug_config=specaug_config_less,
                                label_target_size=vocab_size_without_blank,
                                conformer_size=dim,
                                num_layers=layer_count,
                                num_heads=4,
                                ff_dim=4 * dim,
                                att_weights_dropout=0.2,
                                conv_dropout=0.2,
                                ff_dropout=0.2,
                                mhsa_dropout=0.2,
                                conv_kernel_size=31,
                                final_dropout=0.2,
                                specauc_start_epoch=1,
                                module_list=["ff", "conv", "mhsa", "ff"],
                                module_scales=[0.5, 1.0, 1.0, 0.5],
                                aux_ctc_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                                aux_ctc_loss_scales=[0.3, 0.3, 1.0],
                            )

                            train_config_distill = {
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
                            }
                            if dim == 384:
                                train_config_distill["batch_size"] /= 2
                                train_config_distill["accum_grad_multiple_step"] *= 2
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v3,
                                "net_args": {
                                    "model_config_dict": asdict(student_config_spec),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config_spec)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v3
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_lessspec_f{feat}_t{time}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                            teacher_config = TeacherConfigV2(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                exp_targets=False,
                                eliminate_blanks=True,
                                model_name=name,
                                normalize_stud=False,
                                kd_hyps=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v3,
                                "net_args": {
                                    "model_config_dict": asdict(student_config_spec),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config_spec)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v3
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_lessspec_f{feat}_t{time}_elim_blank"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                        if dim == 384 and distill_scale == 0.25 and num == "33":
                            for start_epoch in []:
                                if start_epoch > 0:
                                    teacher_config = TeacherConfigV6(
                                        distill_scale=distill_scale,
                                        ctc_scale=1 - distill_scale,
                                        t=T,
                                        eliminate_blanks=start_epoch,
                                        model_name=name,
                                        normalize_stud=False,
                                        kd_hyps=None,
                                        keep_some_blanks=None,
                                        mask_padding=False,
                                        prior_file=None,
                                        prior_scale=None,
                                        warmup_loss=None,
                                    )
                                    module = distill_module_v6
                                else:
                                    teacher_config = TeacherConfigV7(
                                        distill_scale=distill_scale,
                                        ctc_scale=1 - distill_scale,
                                        t=T,
                                        eliminate_blanks=start_epoch,
                                        model_name=name,
                                        normalize_stud=False,
                                        kd_hyps=None,
                                        keep_some_blanks=None,
                                        mask_padding=False,
                                        prior_file=None,
                                        prior_scale=None,
                                        warmup_loss=None,
                                        trim_blanks=False,
                                    )
                                    module = distill_module_v7
                                train_args_distill = {
                                    "config": train_config_distill,
                                    "network_module": module,
                                    "net_args": {
                                        "model_config_dict": asdict(student_config),
                                        "distill_config_dict": asdict(teacher_config),
                                    },
                                    "debug": True,
                                }
                                train_args_distill["config"]["preload_from_files"] = {
                                    "teacher": {
                                        "filename": chkpt,
                                        "init_for_train": True,
                                        "ignore_missing": False,
                                        "prefix": "teacher.",
                                        "ignore_params_prefixes": [],
                                    }
                                }
                                model_config_decoding = copy.deepcopy(student_config)
                                model_config_decoding.aux_ctc_loss_scales = [
                                    0.0,
                                    0.0,
                                    1.0,
                                ]  # for decoding use result only of last layer
                                train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                train_args_distill_decoding["net_args"] = {
                                    "model_config_dict": asdict(model_config_decoding),
                                    "distill_config_dict": None,
                                }
                                del train_args_distill_decoding["config"]["preload_from_files"]

                                decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + module
                                    + f"chkpt_{num}"
                                    + f"_{layer_count}_{dim}_{distill_scale}_{T}_elim_blank_{start_epoch}"
                                )
                                train_job = training(
                                    training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                                )
                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args_distill_decoding,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    specific_epoch=250,
                                    decoder_module=decoder_module,
                                    loss_name=f"ctc_loss_layer{layer_count}",
                                    prior_scales=[0.3, 0.5, 0.7, 0.9],
                                    lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                )
                                generate_report(results=results, exp_name=training_name)
                                distill_report[training_name] = results
                                del results
                        teacher_config = TeacherConfigV7(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                            keep_some_blanks=None,
                            mask_padding=False,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=None,
                            trim_blanks=True,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v7,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": True,
                        }
                        train_args_distill["config"]["preload_from_files"] = {
                            "teacher": {
                                "filename": chkpt,
                                "init_for_train": True,
                                "ignore_missing": False,
                                "prefix": "teacher.",
                                "ignore_params_prefixes": [],
                            }
                        }
                        model_config_decoding = copy.deepcopy(student_config)
                        model_config_decoding.aux_ctc_loss_scales = [
                            0.0,
                            0.0,
                            1.0,
                        ]  # for decoding use result only of last layer
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(model_config_decoding),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v7
                            + f"chkpt_{num}"
                            + f"_{layer_count}_{dim}_{distill_scale}_{T}_trim_blanks"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=250,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=[0.3, 0.5, 0.7, 0.9],
                            lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        for warmup in []:
                            teacher_config = TeacherConfigV7(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=True,
                                model_name=name,
                                normalize_stud=False,
                                kd_hyps=None,
                                keep_some_blanks=None,
                                mask_padding=False,
                                prior_file=None,
                                prior_scale=None,
                                warmup_loss=warmup,
                                trim_blanks=True,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v7,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v7
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_trim_blanks_warmup{warmup}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                        if dim == 384:
                            # teacher_config = TeacherConfigV2(
                            #     distill_scale=distill_scale,
                            #     ctc_scale=1,
                            #     t=T,
                            #     exp_targets=False,
                            #     eliminate_blanks=False,
                            #     model_name=name,
                            #     kd_hyps=None,
                            #     normalize_stud=False,
                            # )
                            # train_args_distill = {
                            #     "config": train_config_distill,
                            #     "network_module": distill_module_v3,
                            #     "net_args": {
                            #         "model_config_dict": asdict(student_config),
                            #         "distill_config_dict": asdict(teacher_config)
                            #     },
                            #     "debug": True,
                            # }
                            # train_args_distill['config']['preload_from_files'] = {
                            #     "teacher": {
                            #         "filename": chkpt,
                            #         "init_for_train": True,
                            #         "ignore_missing": False,
                            #         "prefix": 'teacher.',
                            #         "ignore_params_prefixes": [],
                            #     }
                            # }
                            # model_config_decoding = copy.deepcopy(student_config)
                            # model_config_decoding.aux_ctc_loss_scales = [0.0, 0.0,
                            #                                              1.0]  # for decoding use result only of last layer
                            # train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            # train_args_distill_decoding["net_args"] = {"model_config_dict": asdict(model_config_decoding), "distill_config_dict": None}
                            # del train_args_distill_decoding['config']['preload_from_files']

                            # decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"
                            #
                            # training_name = prefix_name + "/" + distill_module_v3 + f"chkpt_{num}" + f"_{layer_count}_{dim}_{distill_scale}_{T}_more_loss"
                            # train_job = training(training_name, train_data, train_args_distill, num_epochs=250,
                            #                      **default_returnn)
                            # results = eval_model(
                            #     training_name=training_name,
                            #     train_job=train_job,
                            #     train_args=train_args_distill_decoding,
                            #     train_data=train_data,
                            #     decoder_config=default_decoder_config,
                            #     dev_dataset_tuples=dev_dataset_tuples,
                            #     specific_epoch=250,
                            #     decoder_module=decoder_module,
                            #     loss_name=f"ctc_loss_layer{layer_count}"
                            # )
                            # generate_report(results=results, exp_name=training_name)
                            # distill_report[training_name] = results
                            # del results

                            # teacher_config = TeacherConfigV5(
                            #     distill_scale=distill_scale,
                            #     ctc_scale=1-distill_scale,
                            #     t=T,
                            #     eliminate_blanks=False,
                            #     model_name=name,
                            #     kd_hyps=None,
                            #     normalize_stud=False,
                            #     prior_scale=None,
                            #     prior_file=None,
                            #     warmup_loss=None,
                            #     mask_padding=True
                            # )
                            # train_args_distill = {
                            #     "config": train_config_distill,
                            #     "network_module": distill_module_v5,
                            #     "net_args": {
                            #         "model_config_dict": asdict(student_config),
                            #         "distill_config_dict": asdict(teacher_config)
                            #     },
                            #     "debug": True,
                            # }
                            # train_args_distill['config']['preload_from_files'] = {
                            #     "teacher": {
                            #         "filename": chkpt,
                            #         "init_for_train": True,
                            #         "ignore_missing": False,
                            #         "prefix": 'teacher.',
                            #         "ignore_params_prefixes": [],
                            #     }
                            # }
                            # model_config_decoding = copy.deepcopy(student_config)
                            # model_config_decoding.aux_ctc_loss_scales = [0.0, 0.0,
                            #     1.0]  # for decoding use result only of last layer
                            # train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            # train_args_distill_decoding["net_args"] = {
                            #     "model_config_dict": asdict(model_config_decoding), "distill_config_dict": None}
                            # del train_args_distill_decoding['config']['preload_from_files']
                            #
                            # decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            # training_name = prefix_name + "/" + distill_module_v3 + f"chkpt_{num}" + f"_{layer_count}_{dim}_{distill_scale}_{T}_maskpad"
                            # train_job = training(training_name, train_data, train_args_distill, num_epochs=250,
                            #     **default_returnn)
                            # results = eval_model(
                            #     training_name=training_name,
                            #     train_job=train_job,
                            #     train_args=train_args_distill_decoding,
                            #     train_data=train_data,
                            #     decoder_config=default_decoder_config,
                            #     dev_dataset_tuples=dev_dataset_tuples,
                            #     specific_epoch=250,
                            #     decoder_module=decoder_module,
                            #     loss_name=f"ctc_loss_layer{layer_count}"
                            # )
                            # generate_report(results=results, exp_name=training_name)
                            # distill_report[training_name] = results
                            # del results

                            from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v4_cfg import (
                                DistillConfig as TeacherConfigV4,
                            )

                            teacher_config = TeacherConfigV4(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                exp_targets=False,
                                eliminate_blanks=True,
                                model_name=name,
                                normalize_stud=False,
                                kd_hyps=None,
                                prior_scale=prior_scale,
                                prior_file=prior_file,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v4,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v4
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_elim_blank_prior"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                            from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v2_cfg import (
                                DistillConfig as TeacherConfigV2,
                            )

                            teacher_config = TeacherConfigV7(
                                distill_scale=distill_scale,
                                ctc_scale=1,
                                t=T,
                                eliminate_blanks=False,
                                model_name=name,
                                kd_hyps=kd_hyps,
                                normalize_stud=False,
                                keep_some_blanks=False,
                                trim_blanks=False,
                                prior_scale=None,
                                prior_file=None,
                                warmup_loss=None,
                                mask_padding=False,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v7,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v7
                                + f"chkpt_{num}_{n_best}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_kdhyps"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
                            for warmup in []:
                                for lrs in [
                                    list(np.linspace(7e-6, 5e-4, 110))
                                    + list(np.linspace(5e-4, 5e-5, 110))
                                    + list(np.linspace(5e-5, 1e-7, 30))
                                ]:
                                    for pad in [False]:
                                        train_config_distill = {
                                            "optimizer": {
                                                "class": "radam",
                                                "epsilon": 1e-16,
                                                "weight_decay": 1e-2,
                                                "decoupled_weight_decay": True,
                                            },
                                            "learning_rates": lrs,
                                            #############
                                            "batch_size": 90 * 16000,
                                            "max_seq_length": {"audio_features": 35 * 16000},
                                            "accum_grad_multiple_step": 2,
                                        }
                                        if lrs == [0.0001]:
                                            tmp = "const"
                                        else:
                                            tmp = "oclr"
                                        if warmup <= 50:
                                            teacher_config = TeacherConfigV5(
                                                distill_scale=distill_scale,
                                                ctc_scale=1 - distill_scale,
                                                t=T,
                                                eliminate_blanks=False,
                                                model_name=name,
                                                kd_hyps=None,
                                                normalize_stud=False,
                                                prior_file=None,
                                                prior_scale=None,
                                                warmup_loss=warmup,
                                                mask_padding=pad,
                                            )
                                            train_args_distill = {
                                                "config": train_config_distill,
                                                "network_module": distill_module_v5,
                                                "net_args": {
                                                    "model_config_dict": asdict(student_config),
                                                    "distill_config_dict": asdict(teacher_config),
                                                },
                                                "debug": True,
                                            }
                                            train_args_distill["config"]["preload_from_files"] = {
                                                "teacher": {
                                                    "filename": chkpt,
                                                    "init_for_train": True,
                                                    "ignore_missing": False,
                                                    "prefix": "teacher.",
                                                    "ignore_params_prefixes": [],
                                                }
                                            }
                                            model_config_decoding = copy.deepcopy(student_config)
                                            model_config_decoding.aux_ctc_loss_scales = [
                                                0.0,
                                                0.0,
                                                1.0,
                                            ]  # for decoding use result only of last layer
                                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                            train_args_distill_decoding["net_args"] = {
                                                "model_config_dict": asdict(model_config_decoding),
                                                "distill_config_dict": None,
                                            }
                                            del train_args_distill_decoding["config"]["preload_from_files"]

                                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                            training_name = (
                                                prefix_name
                                                + "/"
                                                + distill_module_v5
                                                + f"chkpt_{num}"
                                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_warm{warmup}_{tmp}_{pad}"
                                            )
                                            train_job = training(
                                                training_name,
                                                train_data,
                                                train_args_distill,
                                                num_epochs=250,
                                                **default_returnn,
                                            )
                                            results = eval_model(
                                                training_name=training_name,
                                                train_job=train_job,
                                                train_args=train_args_distill_decoding,
                                                train_data=train_data,
                                                decoder_config=default_decoder_config,
                                                dev_dataset_tuples=dev_dataset_tuples,
                                                specific_epoch=250,
                                                decoder_module=decoder_module,
                                                loss_name=f"ctc_loss_layer{layer_count}",
                                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                            )
                                            generate_report(results=results, exp_name=training_name)
                                            distill_report[training_name] = results
                                            del results
                                        lr_long = (
                                            list(np.linspace(7e-6, 5e-4, warmup))
                                            + list(np.linspace(7e-6, 5e-4, 110))
                                            + list(np.linspace(5e-4, 5e-5, 110))
                                            + list(np.linspace(5e-5, 1e-7, 30))
                                        )
                                        train_config_distill = {
                                            "optimizer": {
                                                "class": "radam",
                                                "epsilon": 1e-16,
                                                "weight_decay": 1e-2,
                                                "decoupled_weight_decay": True,
                                            },
                                            "learning_rates": lr_long,
                                            #############
                                            "batch_size": 90 * 16000,
                                            "max_seq_length": {"audio_features": 35 * 16000},
                                            "accum_grad_multiple_step": 2,
                                        }
                                        teacher_config = TeacherConfigV5(
                                            distill_scale=distill_scale,
                                            ctc_scale=1 - distill_scale,
                                            t=T,
                                            eliminate_blanks=False,
                                            model_name=name,
                                            kd_hyps=None,
                                            normalize_stud=False,
                                            prior_file=None,
                                            prior_scale=None,
                                            warmup_loss=warmup,
                                            mask_padding=pad,
                                        )
                                        train_args_distill = {
                                            "config": train_config_distill,
                                            "network_module": distill_module_v5,
                                            "net_args": {
                                                "model_config_dict": asdict(student_config),
                                                "distill_config_dict": asdict(teacher_config),
                                            },
                                            "debug": True,
                                        }
                                        train_args_distill["config"]["preload_from_files"] = {
                                            "teacher": {
                                                "filename": chkpt,
                                                "init_for_train": True,
                                                "ignore_missing": False,
                                                "prefix": "teacher.",
                                                "ignore_params_prefixes": [],
                                            }
                                        }
                                        model_config_decoding = copy.deepcopy(student_config)
                                        model_config_decoding.aux_ctc_loss_scales = [
                                            0.0,
                                            0.0,
                                            1.0,
                                        ]  # for decoding use result only of last layer
                                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                        train_args_distill_decoding["net_args"] = {
                                            "model_config_dict": asdict(model_config_decoding),
                                            "distill_config_dict": None,
                                        }
                                        del train_args_distill_decoding["config"]["preload_from_files"]

                                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                        training_name = (
                                            prefix_name
                                            + "/"
                                            + distill_module_v5
                                            + f"chkpt_{num}"
                                            + f"_{layer_count}_{dim}_{distill_scale}_{T}_warm{warmup}_long_{tmp}_{pad}"
                                        )
                                        train_job = training(
                                            training_name,
                                            train_data,
                                            train_args_distill,
                                            num_epochs=250 + warmup,
                                            **default_returnn,
                                        )
                                        results = eval_model(
                                            training_name=training_name,
                                            train_job=train_job,
                                            train_args=train_args_distill_decoding,
                                            train_data=train_data,
                                            decoder_config=default_decoder_config,
                                            dev_dataset_tuples=dev_dataset_tuples,
                                            specific_epoch=250 + warmup,
                                            decoder_module=decoder_module,
                                            loss_name=f"ctc_loss_layer{layer_count}",
                                            prior_scales=[0.3, 0.5, 0.7, 0.9],
                                            lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                        )
                                        generate_report(results=results, exp_name=training_name)
                                        distill_report[training_name] = results
                                        del results
                            train_config_distill = {
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
                                "batch_size": 90 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 2,
                            }
                            for keep in []:  # 2 and 4 should be fine to cut
                                if distill_scale == 0.9:
                                    teacher_config = TeacherConfigV6(
                                        distill_scale=distill_scale,
                                        ctc_scale=1 - distill_scale,
                                        t=T,
                                        eliminate_blanks=True,
                                        keep_some_blanks=keep,
                                        model_name=name,
                                        kd_hyps=None,
                                        normalize_stud=False,
                                        prior_file=None,
                                        prior_scale=None,
                                        warmup_loss=None,
                                        mask_padding=False,
                                    )
                                    train_args_distill = {
                                        "config": train_config_distill,
                                        "network_module": distill_module_v6,
                                        "net_args": {
                                            "model_config_dict": asdict(student_config),
                                            "distill_config_dict": asdict(teacher_config),
                                        },
                                        "debug": True,
                                    }
                                    train_args_distill["config"]["preload_from_files"] = {
                                        "teacher": {
                                            "filename": chkpt,
                                            "init_for_train": True,
                                            "ignore_missing": False,
                                            "prefix": "teacher.",
                                            "ignore_params_prefixes": [],
                                        }
                                    }
                                    model_config_decoding = copy.deepcopy(student_config)
                                    model_config_decoding.aux_ctc_loss_scales = [
                                        0.0,
                                        0.0,
                                        1.0,
                                    ]  # for decoding use result only of last layer
                                    train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                    train_args_distill_decoding["net_args"] = {
                                        "model_config_dict": asdict(model_config_decoding),
                                        "distill_config_dict": None,
                                    }
                                    del train_args_distill_decoding["config"]["preload_from_files"]

                                    decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                    training_name = (
                                        prefix_name
                                        + "/"
                                        + distill_module_v6
                                        + f"chkpt_{num}"
                                        + f"_{layer_count}_{dim}_{distill_scale}_{T}_keepsome{keep}"
                                    )
                                    train_job = training(
                                        training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                                    )
                                    results = eval_model(
                                        training_name=training_name,
                                        train_job=train_job,
                                        train_args=train_args_distill_decoding,
                                        train_data=train_data,
                                        decoder_config=default_decoder_config,
                                        dev_dataset_tuples=dev_dataset_tuples,
                                        specific_epoch=250,
                                        decoder_module=decoder_module,
                                        loss_name=f"ctc_loss_layer{layer_count}",
                                        prior_scales=[0.3, 0.5, 0.7, 0.9],
                                        lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                    )
                                    generate_report(results=results, exp_name=training_name)
                                    distill_report[training_name] = results
                                    del results
                                if distill_scale >= 0.9:
                                    teacher_config = TeacherConfigV10(
                                        distill_scale=distill_scale,
                                        ctc_scale=1 - distill_scale,
                                        t=T,
                                        eliminate_blanks=True,
                                        keep_some_blanks=(keep, keep),
                                        model_name=name,
                                        kd_hyps=None,
                                        normalize_stud=False,
                                        prior_file=None,
                                        prior_scale=None,
                                        warmup_loss=None,
                                        mask_padding=False,
                                        trim_blanks=False,
                                        mix_nonblank=None,
                                        mix_blank=None,
                                    )
                                    train_args_distill = {
                                        "config": train_config_distill,
                                        "network_module": distill_module_v10,
                                        "net_args": {
                                            "model_config_dict": asdict(student_config),
                                            "distill_config_dict": asdict(teacher_config),
                                        },
                                        "debug": True,
                                    }
                                    train_args_distill["config"]["preload_from_files"] = {
                                        "teacher": {
                                            "filename": chkpt,
                                            "init_for_train": True,
                                            "ignore_missing": False,
                                            "prefix": "teacher.",
                                            "ignore_params_prefixes": [],
                                        }
                                    }
                                    model_config_decoding = copy.deepcopy(student_config)
                                    model_config_decoding.aux_ctc_loss_scales = [
                                        0.0,
                                        0.0,
                                        1.0,
                                    ]  # for decoding use result only of last layer
                                    train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                    train_args_distill_decoding["net_args"] = {
                                        "model_config_dict": asdict(model_config_decoding),
                                        "distill_config_dict": None,
                                    }
                                    del train_args_distill_decoding["config"]["preload_from_files"]

                                    decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                    training_name = (
                                        prefix_name
                                        + "/"
                                        + distill_module_v6
                                        + f"chkpt_{num}"
                                        + f"_{layer_count}_{dim}_{distill_scale}_{T}_keepsome{keep}_sym"
                                    )
                                    train_job = training(
                                        training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                                    )
                                    results = eval_model(
                                        training_name=training_name,
                                        train_job=train_job,
                                        train_args=train_args_distill_decoding,
                                        train_data=train_data,
                                        decoder_config=default_decoder_config,
                                        dev_dataset_tuples=dev_dataset_tuples,
                                        specific_epoch=250,
                                        decoder_module=decoder_module,
                                        loss_name=f"ctc_loss_layer{layer_count}",
                                        prior_scales=[0.3, 0.5, 0.7, 0.9],
                                        lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                    )
                                    generate_report(results=results, exp_name=training_name)
                                    distill_report[training_name] = results
                                    del results

                        for increase in []:
                            teacher_config = TeacherConfigV11(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=True,
                                keep_some_blanks=(1, 1),
                                model_name=name,
                                kd_hyps=None,
                                normalize_stud=False,
                                prior_file=None,
                                prior_scale=None,
                                warmup_loss=None,
                                mask_padding=False,
                                trim_blanks=False,
                                mix_nonblank=None,
                                mix_blank=None,
                                increase_keepsome_epochs=increase,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v11,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v11
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_keepsome_increase{increase}_sym"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                        for perc in [0.5, 1.0, 2.0]:
                            if not distill_scale == 0.9:
                                continue
                            teacher_config = TeacherConfigV12(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=False,
                                keep_some_blanks=None,
                                model_name=name,
                                kd_hyps=None,
                                normalize_stud=False,
                                prior_file=None,
                                prior_scale=None,
                                warmup_loss=None,
                                mask_padding=False,
                                trim_blanks=False,
                                mix_nonblank=None,
                                mix_blank=None,
                                increase_keepsome_epochs=None,
                                keep_random=perc,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v12,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v12
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_keep_random{perc}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
                        for threshold in [0.5, 0.25, 0.1]:
                            if not distill_scale == 0.9:
                                continue
                            teacher_config = TeacherConfigV13(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=False,
                                keep_some_blanks=None,
                                model_name=name,
                                kd_hyps=None,
                                normalize_stud=False,
                                prior_file=None,
                                prior_scale=None,
                                warmup_loss=None,
                                mask_padding=False,
                                trim_blanks=False,
                                mix_nonblank=None,
                                mix_blank=None,
                                increase_keepsome_epochs=None,
                                keep_random=None,
                                keep_threshold=threshold,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v13,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v13
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_keep_thresh{threshold}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                        from ...pytorch_networks.ctc.hubert_tune_0711.distill_hubert_v6_cfg import (
                            DistillConfig as TeacherConfigV6,
                            DistillConfig as TeacherConfigV6,
                        )

                        for keep in []:
                            teacher_config = TeacherConfigV6(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=True,
                                keep_some_blanks=keep,
                                model_name=name,
                                kd_hyps=None,
                                normalize_stud=False,
                                prior_file=None,
                                prior_scale=None,
                                warmup_loss=25,
                                mask_padding=False,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v6,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v6
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_warm25_keepsome{keep}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                        # teacher_config = TeacherConfigV6(
                        #     distill_scale=distill_scale,
                        #     ctc_scale=1 - distill_scale,
                        #     t=T,
                        #     eliminate_blanks=True,
                        #     keep_some_blanks=None,
                        #     model_name=name,
                        #     kd_hyps=None,
                        #     normalize_stud=False,
                        #     prior_file=None,
                        #     prior_scale=None,
                        #     warmup_loss=25,
                        #     mask_padding=False,
                        # )
                        # train_args_distill = {
                        #     "config": train_config_distill,
                        #     "network_module": distill_module_v6,
                        #     "net_args": {
                        #         "model_config_dict": asdict(student_config),
                        #         "distill_config_dict": asdict(teacher_config),
                        #     },
                        #     "debug": True,
                        # }
                        # train_args_distill["config"]["preload_from_files"] = {
                        #     "teacher": {
                        #         "filename": chkpt,
                        #         "init_for_train": True,
                        #         "ignore_missing": False,
                        #         "prefix": "teacher.",
                        #         "ignore_params_prefixes": [],
                        #     }
                        # }
                        # model_config_decoding = copy.deepcopy(student_config)
                        # model_config_decoding.aux_ctc_loss_scales = [
                        #     0.0,
                        #     0.0,
                        #     1.0,
                        # ]  # for decoding use result only of last layer
                        # train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        # train_args_distill_decoding["net_args"] = {
                        #     "model_config_dict": asdict(model_config_decoding),
                        #     "distill_config_dict": None,
                        # }
                        # del train_args_distill_decoding["config"]["preload_from_files"]
                        #
                        # decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"
                        #
                        # training_name = (
                        #     prefix_name
                        #     + "/"
                        #     + distill_module_v6
                        #     + f"chkpt_{num}"
                        #     + f"_{layer_count}_{dim}_{distill_scale}_{T}_warm25_elim_blank"
                        # )
                        # train_job = training(
                        #     training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        # )
                        # results = eval_model(
                        #     training_name=training_name,
                        #     train_job=train_job,
                        #     train_args=train_args_distill_decoding,
                        #     train_data=train_data,
                        #     decoder_config=default_decoder_config,
                        #     dev_dataset_tuples=dev_dataset_tuples,
                        #     specific_epoch=250,
                        #     decoder_module=decoder_module,
                        #     loss_name=f"ctc_loss_layer{layer_count}",
                        #     prior_scales=[0.3, 0.5, 0.7, 0.9],
                        #     lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                        # )
                        # generate_report(results=results, exp_name=training_name)
                        # distill_report[training_name] = results
                        # del results

                        for mix in []:
                            if not distill_scale == 0.9:
                                continue
                            teacher_config = TeacherConfigV8(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=False,
                                keep_some_blanks=None,
                                model_name=name,
                                kd_hyps=None,
                                normalize_stud=False,
                                prior_file=None,
                                prior_scale=None,
                                warmup_loss=None,
                                mask_padding=False,
                                trim_blanks=False,
                                mix_nonblank_blank=mix,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v8,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": True,
                            }
                            train_args_distill["config"]["preload_from_files"] = {
                                "teacher": {
                                    "filename": chkpt,
                                    "init_for_train": True,
                                    "ignore_missing": False,
                                    "prefix": "teacher.",
                                    "ignore_params_prefixes": [],
                                }
                            }
                            model_config_decoding = copy.deepcopy(student_config)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(model_config_decoding),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v6
                                + f"chkpt_{num}"
                                + f"_{layer_count}_{dim}_{distill_scale}_{T}_mix{mix}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=250,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=[0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                            # teacher_config = TeacherConfigV9(
                            #     distill_scale=distill_scale,
                            #     ctc_scale=1 - distill_scale,
                            #     t=T,
                            #     eliminate_blanks=False,
                            #     keep_some_blanks=None,
                            #     model_name=name,
                            #     kd_hyps=None,
                            #     normalize_stud=False,
                            #     prior_file=None,
                            #     prior_scale=None,
                            #     warmup_loss=None,
                            #     mask_padding=False,
                            #     trim_blanks=False,
                            #     mix_nonblank=1.0,
                            #     mix_blank=mix,
                            # )
                            # train_args_distill = {
                            #     "config": train_config_distill,
                            #     "network_module": distill_module_v9,
                            #     "net_args": {
                            #         "model_config_dict": asdict(student_config),
                            #         "distill_config_dict": asdict(teacher_config),
                            #     },
                            #     "debug": True,
                            # }
                            # train_args_distill["config"]["preload_from_files"] = {
                            #     "teacher": {
                            #         "filename": chkpt,
                            #         "init_for_train": True,
                            #         "ignore_missing": False,
                            #         "prefix": "teacher.",
                            #         "ignore_params_prefixes": [],
                            #     }
                            # }
                            # model_config_decoding = copy.deepcopy(student_config)
                            # model_config_decoding.aux_ctc_loss_scales = [
                            #     0.0,
                            #     0.0,
                            #     1.0,
                            # ]  # for decoding use result only of last layer
                            # train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            # train_args_distill_decoding["net_args"] = {
                            #     "model_config_dict": asdict(model_config_decoding),
                            #     "distill_config_dict": None,
                            # }
                            # del train_args_distill_decoding["config"]["preload_from_files"]
                            #
                            # decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"
                            #
                            # training_name = (
                            #     prefix_name
                            #     + "/"
                            #     + distill_module_v9
                            #     + f"chkpt_{num}"
                            #     + f"_{layer_count}_{dim}_{distill_scale}_{T}_blank_mix{mix}"
                            # )
                            # train_job = training(
                            #     training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                            # )
                            # results = eval_model(
                            #     training_name=training_name,
                            #     train_job=train_job,
                            #     train_args=train_args_distill_decoding,
                            #     train_data=train_data,
                            #     decoder_config=default_decoder_config,
                            #     dev_dataset_tuples=dev_dataset_tuples,
                            #     specific_epoch=250,
                            #     decoder_module=decoder_module,
                            #     loss_name=f"ctc_loss_layer{layer_count}",
                            #     prior_scales=[0.3, 0.5, 0.7, 0.9],
                            #     lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                            # )
                            # generate_report(results=results, exp_name=training_name)
                            # distill_report[training_name] = results
                            # del results
    tmp_rep = {}
    tmp_rep["baselines"] = distill_report["baselines"]
    for exp, dic in distill_report.items():
        tmp = {}
        for x in dic:
            if "250" in x:
                tmp[x] = dic[x]
        tmp_rep[exp] = tmp
    tk.register_report(
        "reports/distill_hubert_report_last",
        partial(build_hubert_distill_report, tmp_rep),
        required=distill_report,
        update_frequency=3600,
    )
    tmp_rep = {}
    tmp_rep["baselines"] = distill_report["baselines"]
    for exp, dic in distill_report.items():
        tmp = {}
        for x in dic:
            if "best" in x and not "best4" in x:
                tmp[x] = dic[x]
        tmp_rep[exp] = tmp
    tk.register_report(
        "reports/distill_hubert_report_best",
        partial(build_hubert_distill_report, tmp_rep),
        required=distill_report,
        update_frequency=3600,
    )
    tmp_rep = {}
    tmp_rep["baselines"] = distill_report["baselines"]
    for exp, dic in distill_report.items():
        tmp = {}
        for x in dic:
            if "best4" in x:
                tmp[x] = dic[x]
        tmp_rep[exp] = tmp
    tk.register_report(
        "reports/distill_hubert_report_best4",
        partial(build_hubert_distill_report, tmp_rep),
        required=distill_report,
        update_frequency=3600,
    )


"""
old quant stuff
            if not model == "base-ls960":
                continue
            results = {}
            epochs = [250]
            num_seqs_ls = [100]
            quant_modes = [CalibrationMethod.MinMax]
            activation_types = [QuantType.QInt8]
            weight_types = [QuantType.QInt8]
            average_modes = [True, False]
            sym_modes = [False, True]
            quant_ops_ls = [["Conv", "MatMul"]]
            quant_formats = [QuantFormat.QDQ]
            quant_decoder_module = "ctc.decoder.flashlight_quant_onnx_ctc"
            from ...pytorch_networks.ctc.decoder.flashlight_quant_onnx_ctc import DecoderConfig

            quant_decoder_config = DecoderConfig(
                lexicon=get_text_lexicon(),
                returnn_vocab=label_datastream.vocab,
                beam_size=1024,
                beam_size_token=12,  # makes it much faster
                arpa_lm=arpa_4gram_lm,
                beam_threshold=14,
            )
            for num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops, quant_format, epoch in (
                    itertools.product(
                        num_seqs_ls, quant_modes, activation_types, weight_types, average_modes,
                        sym_modes, quant_ops_ls, quant_formats, epochs)):
                quant_str = get_quant_str(num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops,
                                          quant_format)

                returnn_export_config = get_onnx_export_config(
                    network_module=network_module,
                    config={},
                    net_args=train_args_amp["net_args"],
                )
                onnx_job = TorchOnnxExportJob(
                    returnn_config=returnn_export_config,
                    checkpoint=train_job.out_checkpoints[epoch],
                    returnn_root=MINI_RETURNN_ROOT,
                    returnn_python_exe=RETURNN_EXE,
                )
                onnx_job.set_keep_value(5)
                onnx_job.add_alias(training_name + f"/onnx_export_{epoch}")
                for lm_weight in [2.2]: # TODO set proper scales
                    for prior_scale in [0.7]:
                        decoder_config = copy.deepcopy(quant_decoder_config)
                        decoder_config.lm_weight = lm_weight
                        decoder_config.prior_scale = prior_scale
                        decoder_args = {
                            "quantized_model": onnx_job.out_onnx_model,
                            "config": asdict(decoder_config)
                        }
                        returnn_search_config = get_forward_config(
                            network_module=network_module,
                            config={},
                            net_args=train_args_amp['net_args'],
                            decoder_args=decoder_args,
                            decoder=quant_decoder_module,
                            debug=train_args_amp['debug'],
                        )
                        wers = {}
                        search_jobs = []
                        search_prefix = training_name + "/onnx" + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                        for key, (dev_dataset, dev_dataset_reference) in dev_dataset_tuples.items():
                            search_name = search_prefix + "/%s" % key
                            wers[search_name], search_job = search_single(
                                search_name,
                                returnn_search_config,
                                train_job.out_checkpoints[epoch],  # dummy placeholder, decoder replaces it by onnx checkpoint
                                dev_dataset,
                                dev_dataset_reference,
                                RETURNN_EXE,
                                MINI_RETURNN_ROOT,
                                mem_rqmt=30 if "hubert_tune" in search_name else 10,
                                use_gpu=False,
                            )
                            search_jobs.append(search_job)
                            results.update(wers)
                        for random_seed in [0, 1]:
                            quant_name = training_name + quant_str + f"_seed_{random_seed}"
                            quant_data = copy.deepcopy(train_data.train.as_returnn_opts())
                            quant_data['datasets']['zip_dataset']['partition_epoch'] = 1
                            quant_data['datasets']['zip_dataset']['seq_ordering'] = "random"
                            quant_data['datasets']['zip_dataset']['fixed_random_seed'] = random_seed
                            quant_job = ModelQuantizeStaticJob(
                                dataset=quant_data,
                                model=onnx_job.out_onnx_model,
                                num_seqs=num_seqs,
                                calibrate_method=quant_mode,
                                activation_type=activation_type,
                                weight_type=weight_type,
                                moving_average=average,
                                symmetric=sym,
                                ops_to_quant=quant_ops,
                                quant_format=quant_format,
                                num_parallel_seqs=None,
                            )
                            quant_job.set_keep_value(5)
                            quant_job.add_alias(quant_name + f"/quantization_{epoch}")
                            decoder_args = {
                                "quantized_model": quant_job.out_model,
                                "config": asdict(decoder_config)
                            }
                            returnn_search_config = get_forward_config(
                                network_module=network_module,
                                config={},
                                net_args=train_args_amp['net_args'],
                                decoder_args=decoder_args,
                                decoder=quant_decoder_module,
                                debug=train_args_amp['debug'],
                            )
                            wers = {}
                            search_jobs = []
                            search_prefix = quant_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                            for key, (dev_dataset, dev_dataset_reference) in dev_dataset_tuples.items():
                                search_name = search_prefix + "/%s" % key
                                wers[search_name], search_job = search_single(
                                    search_name,
                                    returnn_search_config,
                                    train_job.out_checkpoints[epoch],  # dummy placeholder, decoder replaces it by onnx checkpoint
                                    dev_dataset,
                                    dev_dataset_reference,
                                    RETURNN_EXE,
                                    MINI_RETURNN_ROOT,
                                    mem_rqmt=30 if "hubert_tune" in search_name else 10,
                                    use_gpu=False,
                                )
                                search_jobs.append(search_job)
                                results.update(wers)
            generate_report(results=results, exp_name=training_name + "_quantized")
            del results
            """
