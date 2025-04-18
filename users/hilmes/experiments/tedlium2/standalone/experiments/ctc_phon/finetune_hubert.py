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
from ...pipeline import (
    training,
    generate_kd_hypothesis,
    calculate_blank_counts,
    calculate_blank_ratios,
    prepare_asr_model,
)
from ...report import generate_report
from .tune_eval import eval_model, build_hubert_report, build_hubert_distill_report
from functools import partial
import os

def eow_phon_ted_tune_pos_enc_w_hubert():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/distill_pos_enc_w_hubert"

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

    from ...pytorch_networks.ctc.hubert_tune_0711.hubert_tune_v2_cfg import ModelConfig

    hubert_report = {}
    checkpoints = {}
    for model in ["large-ll60k"]:
        model_config = ModelConfig(
            label_target_size=vocab_size_without_blank,
            final_dropout=0.2,
            model_name=model,
            finetune_layer=True,
            keep_layers=None,
            downsample_factor=2,
        )
        network_module = "ctc.hubert_tune_0711.hubert_tune_v2"
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
            lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        )
        generate_report(results=results, exp_name=training_name)
        hubert_report[training_name] = results
        del results
        if model == "large-ll60k":
            asr_model = prepare_asr_model(
                training_name,
                train_job,
                train_args_amp,
                with_prior=True,
                datasets=train_data,
                get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
            )
            checkpoints[model + f"_best4"] = (asr_model.checkpoint, asr_model.prior_file)
            blank_counts = calculate_blank_counts(
                prefix_name=training_name,
                train_job=train_job,
                train_args=train_args_amp,
                train_data=train_data,
                checkpoint="best4",
                debug=True,
            )
            tk.register_output(training_name + "/" + "blank_counts", blank_counts)
            blank_ratios = calculate_blank_ratios(
                prefix_name=training_name,
                train_job=train_job,
                train_args=train_args_amp,
                train_data=train_data,
                checkpoint="best4",
                debug=True,
            )
            tk.register_output(training_name + "/" + "blank_ratios", blank_ratios)

    from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v1_cfg import (
        ModelConfig as StudentConfigV1,
        DistillConfig as TeacherConfigV1,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        SpecaugConfig,
        ConformerPosEmbConfig,
    )
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v4_cfg import (
        DistillConfig as TeacherConfigV4,
    )
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v5_cfg import (
        DistillConfig as TeacherConfigV5,
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
    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )
    # v1 is broken
    # v2 kinda too
    # distill_module_v2 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v2"
    distill_module_v3 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v3"
    distill_module_v4 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v4"
    distill_module_v5 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v5"

    from .pos_enc_baseline import eow_phon_ted_pos_enc_baseline

    baselines, base_checkpoints = eow_phon_ted_pos_enc_baseline(get_report=True)
    baseline_prefix = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/pos_enc_baseline"
    baseline_module = "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
    prior_scales = [0.3, 0.5, 0.7]
    lm_scales = [1.6, 1.8, 2.0]
    for name, (chkpt, prior_file) in checkpoints.items():
        name, num = name.split("_")
        train_epochs = [500]
        for epochs in train_epochs:
            distill_report = {}
            distill_report["baselines"] = {}
            for dim, spec_start, spec, heads, layer_count in [
                (384, 1, 16, 8, 12),
            ]:
                for distill_scale in [0.25, 0.9, 1.0]:
                    for T in [2]:
                        for teacher_ep in [500, 1000]:
                            distill_report["baselines"][
                                baseline_prefix
                                + "/"
                                + baseline_module
                                + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}"
                            ] = baselines[
                                baseline_prefix
                                + "/"
                                + baseline_module
                                + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}"
                            ]
                        teacher_config = TeacherConfigV1(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=False,
                            model_name=name,
                            kd_hyps=None,
                            normalize_stud=False,
                            keep_some_blanks=False,
                            trim_blanks=False,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=None,
                            mask_padding=False,
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

                        student_config = StudentConfigV1(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config_student,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            pos_emb_config=pos_emb_cfg,
                            conformer_size=dim,
                            num_layers=layer_count,
                            num_heads=heads,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.2,
                            conv_dropout=0.2,
                            ff_dropout=0.2,
                            mhsa_dropout=0.2,
                            mhsa_with_bias=True,
                            conv_kernel_size=31,
                            final_dropout=0.2,
                            dropout_broadcast_axes=None,
                            specauc_start_epoch=spec_start,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                        )
                        ###############################################################################################
                        # Baseline
                        train_config_distill = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-16,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )

                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=prior_scales,
                            lm_scales=lm_scales,
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        blank_counts = calculate_blank_counts(
                            prefix_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            checkpoint=epochs,
                            debug=True,
                        )
                        tk.register_output(training_name + "/" + "blank_counts", blank_counts)
                        ###############################################################################################
                        # Eliminate Blank
                        teacher_config = TeacherConfigV1(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                            keep_some_blanks=False,
                            trim_blanks=False,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=None,
                            mask_padding=False,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=prior_scales,
                            lm_scales=lm_scales,
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        ###############################################################################################
                        # Prior Correction
                        teacher_config = TeacherConfigV1(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                            prior_scale=0.5,
                            prior_file=prior_file,
                            keep_some_blanks=False,
                            warmup_loss=None,
                            mask_padding=False,
                            trim_blanks=False,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank_prior"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
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
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                            + list(np.linspace(5e-5, 1e-7, 30)),
                            #############
                            "batch_size": 90 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 2,
                        }
                        ###############################################################################################
                        # Symmetric Selection
                        for keep in [1, 2, 3, 4, 5]:
                            teacher_config = TeacherConfigV4(
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
                                increase_keepsome_epochs=None,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v4
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keepsome{keep}_sym"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
                            blank_counts = calculate_blank_counts(
                                prefix_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                checkpoint=epochs,
                                debug=True,
                            )
                            tk.register_output(training_name + "/" + "blank_counts", blank_counts)
                        ###############################################################################################
                        # Thresholding Selection
                        for thresh in [0.1, 0.2]:
                            teacher_config = TeacherConfigV5(
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
                                increase_keepsome_epochs=None,
                                keep_random=None,
                                keep_threshold=thresh,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v3
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keep_thresh{thresh}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                        ###############################################################################################
                        # Random Selection
                        for rdn in [0.5, 1.0, 2.0]:
                            teacher_config = TeacherConfigV5(
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
                                increase_keepsome_epochs=None,
                                keep_random=rdn,
                                keep_threshold=None,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v3
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keep_rdn{rdn}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

            tmp_rep = {}
            for exp, dic in distill_report.items():
                tmp = {}
                for x in dic:
                    if any(str(y) in x.split("/")[6:] for y in train_epochs):
                        tmp[x] = dic[x]
                assert len(tmp) > 0 or "baselines" in exp, exp
                tmp_rep[exp] = tmp
            tmp_rep["baselines"] = distill_report["baselines"]
            tk.register_report(
                f"reports/distill_pos_enc_w_hubert_report_last_{epochs}",
                partial(build_hubert_distill_report, tmp_rep),
                required=distill_report,
                update_frequency=900,
            )


def eow_phon_ted_tune_pos_enc_w_hubert_new():
    # TODO: replace this once v2 is done
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/distill_pos_enc_w_hubert_new"

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

    from ...pytorch_networks.ctc.hubert_tune_0711.hubert_tune_v2_cfg import ModelConfig

    hubert_report = {}
    checkpoints = {}
    for model in ["large-ll60k"]:
        model_config = ModelConfig(
            label_target_size=vocab_size_without_blank,
            final_dropout=0.2,
            model_name=model,
            finetune_layer=True,
            keep_layers=None,
            downsample_factor=2,
        )
        network_module = "ctc.hubert_tune_0711.hubert_tune_v2"
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
            lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        )
        generate_report(results=results, exp_name=training_name)
        hubert_report[training_name] = results
        del results
        if model == "large-ll60k":
            asr_model = prepare_asr_model(
                training_name,
                train_job,
                train_args_amp,
                with_prior=True,
                datasets=train_data,
                get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
            )
            checkpoints[model + f"_best4"] = (asr_model.checkpoint, asr_model.prior_file)
            blank_counts = calculate_blank_counts(
                prefix_name=training_name,
                train_job=train_job,
                train_args=train_args_amp,
                train_data=train_data,
                checkpoint="best4",
                debug=True,
            )
            tk.register_output(training_name + "/" + "blank_counts", blank_counts)
            blank_ratios = calculate_blank_ratios(
                prefix_name=training_name,
                train_job=train_job,
                train_args=train_args_amp,
                train_data=train_data,
                checkpoint="best4",
                debug=True,
            )
            tk.register_output(training_name + "/" + "blank_ratios", blank_ratios)

    from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v1_cfg import (
        ModelConfig as StudentConfigV1,
        DistillConfig as TeacherConfigV1,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        SpecaugConfig,
        ConformerPosEmbConfig,
    )
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v4_cfg import (
        DistillConfig as TeacherConfigV4,
    )
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v5_cfg import (
        DistillConfig as TeacherConfigV5,
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
    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )
    # v1 is broken
    # v2 kinda too
    # distill_module_v2 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v2"
    distill_module_v3 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v3"
    distill_module_v4 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v4"
    distill_module_v5 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v5"
    distill_module_v6 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v6"
    # TODO: activate v6 for everything
    decoder_module_v1 = "ctc.decoder.flashlight_ctc_distill_v1"

    from .pos_enc_baseline import eow_phon_ted_pos_enc_baseline

    baselines, base_checkpoints = eow_phon_ted_pos_enc_baseline(get_report=True)
    baseline_prefix = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/pos_enc_baseline"
    baseline_module = "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
    prior_scales = [0.3, 0.5, 0.7]
    lm_scales = [1.6, 1.8, 2.0]
    for name, (chkpt, prior_file) in checkpoints.items():
        name, num = name.split("_")
        train_epochs = [500]  # TODO: 1000
        for epochs in train_epochs:
            distill_report = {}
            distill_report["baselines"] = {}
            train_config_distill = {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                + list(np.linspace(5e-5, 1e-7, 30)),
                #############
                "batch_size": 300 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 1,
                "gradient_clip_norm": 1.0,
                "torch_amp_options": {"dtype": "bfloat16"},
            }
            #######################################################################################################
            ## TODO: fix base train config

            for dim, spec_start, spec, heads, layer_count in [
                (384, 1, 16, 8, 12),
            ]:
                specaug_config = SpecaugConfig(
                    repeat_per_n_frames=25,
                    max_dim_time=20,
                    max_dim_feat=spec,
                    num_repeat_feat=5,  # Jingjing style
                )
                for distill_scale in [0.25, 0.9, 1.0]:
                    for T in [2]:
                        for teacher_ep in [500, 1000]:
                            distill_report["baselines"][
                                baseline_prefix
                                + "/"
                                + baseline_module
                                + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}"
                            ] = baselines[
                                baseline_prefix
                                + "/"
                                + baseline_module
                                + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}_0.2_radam_180bs_amp"
                            ]
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

                        student_config = StudentConfigV1(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config_student,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            pos_emb_config=pos_emb_cfg,
                            conformer_size=dim,
                            num_layers=layer_count,
                            num_heads=heads,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.2,
                            conv_dropout=0.2,
                            ff_dropout=0.2,
                            mhsa_dropout=0.2,
                            mhsa_with_bias=True,
                            conv_kernel_size=31,
                            final_dropout=0.2,
                            dropout_broadcast_axes=None,
                            specauc_start_epoch=spec_start,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                        )
                        ###############################################################################################
                        # Baseline
                        teacher_config = TeacherConfigV1(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=False,
                            model_name=name,
                            kd_hyps=None,
                            normalize_stud=False,
                            keep_some_blanks=False,
                            trim_blanks=False,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=None,
                            mask_padding=False,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v3,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                            "use_speed_perturbation": True,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]
                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        train_job.rqmt["gpu_mem"] = 48

                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module_v1,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=prior_scales,
                            lm_scales=lm_scales,
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        blank_counts = calculate_blank_counts(
                            prefix_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            checkpoint=epochs,
                            debug=True,
                        )
                        tk.register_output(training_name + "/" + "blank_counts", blank_counts)
                        if distill_scale == 0.25:
                            teacher_config = TeacherConfigV5(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=False,
                                model_name=name,
                                kd_hyps=None,
                                normalize_stud=False,
                                keep_some_blanks=None,
                                trim_blanks=False,
                                prior_file=None,
                                prior_scale=None,
                                warmup_loss=None,
                                mask_padding=False,
                                keep_random=None,
                                keep_threshold=None,
                                increase_keepsome_epochs=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v6,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]
                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v6
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 48

                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module_v1,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
                        ###############################################################################################
                        # Eliminate Blank
                        teacher_config = TeacherConfigV1(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                            keep_some_blanks=False,
                            trim_blanks=False,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=None,
                            mask_padding=False,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v3,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                            "use_speed_perturbation": True,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        train_job.rqmt["gpu_mem"] = 48
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module_v1,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=prior_scales,
                            lm_scales=lm_scales,
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        if distill_scale == 0.25:
                            teacher_config = TeacherConfigV5(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=True,
                                model_name=name,
                                normalize_stud=False,
                                kd_hyps=None,
                                keep_some_blanks=None,
                                trim_blanks=False,
                                prior_file=None,
                                prior_scale=None,
                                warmup_loss=None,
                                mask_padding=False,
                                increase_keepsome_epochs=None,
                                keep_threshold=None,
                                keep_random=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v6,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v6
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module_v1,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                            ###############################################################################################
                            # Trim Blank
                            if distill_scale == 0.25:
                                teacher_config = TeacherConfigV1(
                                    distill_scale=distill_scale,
                                    ctc_scale=1 - distill_scale,
                                    t=T,
                                    eliminate_blanks=True,
                                    model_name=name,
                                    normalize_stud=False,
                                    kd_hyps=None,
                                    keep_some_blanks=False,
                                    trim_blanks=True,
                                    prior_file=None,
                                    prior_scale=None,
                                    warmup_loss=None,
                                    mask_padding=False,
                                )
                                train_args_distill = {
                                    "config": train_config_distill,
                                    "network_module": distill_module_v3,
                                    "net_args": {
                                        "model_config_dict": asdict(student_config),
                                        "distill_config_dict": asdict(teacher_config),
                                    },
                                    "debug": False,
                                    "use_speed_perturbation": True,
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
                                train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                train_args_distill_decoding["net_args"] = {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": None,
                                }
                                del train_args_distill_decoding["config"]["preload_from_files"]

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + distill_module_v3
                                    + f"_chkpt_{num}"
                                    + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_trim_blanks"
                                )
                                train_job = training(
                                    training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                                )
                                train_job.rqmt["gpu_mem"] = 48
                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args_distill_decoding,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    specific_epoch=epochs,
                                    decoder_module=decoder_module_v1,
                                    loss_name=f"ctc_loss_layer{layer_count}",
                                    prior_scales=prior_scales,
                                    lm_scales=lm_scales,
                                    run_test=True,
                                    test_dataset_tuples=test_dataset_tuples,
                                )
                                generate_report(results=results, exp_name=training_name)
                                distill_report[training_name] = results
                                del results
                                if distill_scale == 0.25:
                                    teacher_config = TeacherConfigV5(
                                        distill_scale=distill_scale,
                                        ctc_scale=1 - distill_scale,
                                        t=T,
                                        eliminate_blanks=True,
                                        model_name=name,
                                        normalize_stud=False,
                                        kd_hyps=None,
                                        keep_some_blanks=None,
                                        trim_blanks=True,
                                        prior_file=None,
                                        prior_scale=None,
                                        warmup_loss=None,
                                        mask_padding=False,
                                        increase_keepsome_epochs=None,
                                        keep_threshold=None,
                                        keep_random=None,
                                    )
                                    train_args_distill = {
                                        "config": train_config_distill,
                                        "network_module": distill_module_v6,
                                        "net_args": {
                                            "model_config_dict": asdict(student_config),
                                            "distill_config_dict": asdict(teacher_config),
                                        },
                                        "debug": False,
                                        "use_speed_perturbation": True,
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
                                    train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                    train_args_distill_decoding["net_args"] = {
                                        "model_config_dict": asdict(student_config),
                                        "distill_config_dict": None,
                                    }
                                    del train_args_distill_decoding["config"]["preload_from_files"]

                                    training_name = (
                                        prefix_name
                                        + "/"
                                        + distill_module_v6
                                        + f"_chkpt_{num}"
                                        + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_trim_blanks"
                                    )
                                    train_job = training(
                                        training_name,
                                        train_data,
                                        train_args_distill,
                                        num_epochs=epochs,
                                        **default_returnn,
                                    )
                                    train_job.rqmt["gpu_mem"] = 48
                                    results = eval_model(
                                        training_name=training_name,
                                        train_job=train_job,
                                        train_args=train_args_distill_decoding,
                                        train_data=train_data,
                                        decoder_config=default_decoder_config,
                                        dev_dataset_tuples=dev_dataset_tuples,
                                        specific_epoch=epochs,
                                        decoder_module=decoder_module_v1,
                                        loss_name=f"ctc_loss_layer{layer_count}",
                                        prior_scales=prior_scales,
                                        lm_scales=lm_scales,
                                        run_test=True,
                                        test_dataset_tuples=test_dataset_tuples,
                                    )
                                    generate_report(results=results, exp_name=training_name)
                                    distill_report[training_name] = results
                                    del results

                        ###############################################################################################
                        # Prior Correction, maybe remove?
                        teacher_config = TeacherConfigV1(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                            prior_scale=0.5,
                            prior_file=prior_file,
                            keep_some_blanks=False,
                            warmup_loss=None,
                            mask_padding=False,
                            trim_blanks=False,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v3,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                            "use_speed_perturbation": True,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v3
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank_prior"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        train_job.rqmt["gpu_mem"] = 48
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module_v1,
                            loss_name=f"ctc_loss_layer{layer_count}",
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        if distill_scale == 0.25:
                            teacher_config = TeacherConfigV5(
                                distill_scale=distill_scale,
                                ctc_scale=1 - distill_scale,
                                t=T,
                                eliminate_blanks=True,
                                model_name=name,
                                normalize_stud=False,
                                kd_hyps=None,
                                prior_scale=0.5,
                                prior_file=prior_file,
                                keep_some_blanks=None,
                                warmup_loss=None,
                                mask_padding=False,
                                trim_blanks=False,
                                increase_keepsome_epochs=None,
                                keep_random=None,
                                keep_threshold=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v6,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v6
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank_prior"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module_v1,
                                loss_name=f"ctc_loss_layer{layer_count}",
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                        ###############################################################################################
                        # Symmetric Selection
                        for keep in [1, 2, 3, 4, 5]:
                            teacher_config = TeacherConfigV4(
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
                                increase_keepsome_epochs=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v4,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v4
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keepsome{keep}_sym"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
                            blank_counts = calculate_blank_counts(
                                prefix_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                checkpoint=epochs,
                                debug=True,
                            )
                            tk.register_output(training_name + "/" + "blank_counts", blank_counts)

                            if keep == 3:
                                teacher_config = TeacherConfigV5(
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
                                    increase_keepsome_epochs=None,
                                    keep_random=None,
                                    keep_threshold=None,
                                )
                                train_args_distill = {
                                    "config": train_config_distill,
                                    "network_module": distill_module_v6,
                                    "net_args": {
                                        "model_config_dict": asdict(student_config),
                                        "distill_config_dict": asdict(teacher_config),
                                    },
                                    "debug": True,
                                    "use_speed_perturbation": True,
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
                                train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                train_args_distill_decoding["net_args"] = {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": None,
                                }
                                del train_args_distill_decoding["config"]["preload_from_files"]

                                decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + distill_module_v6
                                    + f"_chkpt_{num}"
                                    + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keepsome{keep}_sym"
                                )
                                train_job = training(
                                    training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                                )
                                train_job.rqmt["gpu_mem"] = 48
                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args_distill_decoding,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    specific_epoch=epochs,
                                    decoder_module=decoder_module,
                                    loss_name=f"ctc_loss_layer{layer_count}",
                                    prior_scales=prior_scales,
                                    lm_scales=lm_scales,
                                    run_test=True,
                                    test_dataset_tuples=test_dataset_tuples,
                                )
                                generate_report(results=results, exp_name=training_name)
                                distill_report[training_name] = results
                                del results
                        ###############################################################################################
                        # Thresholding Selection
                        for thresh in [0.1, 0.2]:
                            teacher_config = TeacherConfigV5(
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
                                increase_keepsome_epochs=None,
                                keep_random=None,
                                keep_threshold=thresh,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v5,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v3
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keep_thresh{thresh}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
                            if distill_scale == 0.25 and thresh == 0.1:
                                train_args_distill = {
                                    "config": train_config_distill,
                                    "network_module": distill_module_v6,
                                    "net_args": {
                                        "model_config_dict": asdict(student_config),
                                        "distill_config_dict": asdict(teacher_config),
                                    },
                                    "debug": False,
                                    "use_speed_perturbation": True,
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
                                train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                train_args_distill_decoding["net_args"] = {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": None,
                                }
                                del train_args_distill_decoding["config"]["preload_from_files"]

                                decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + distill_module_v6
                                    + f"_chkpt_{num}"
                                    + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keep_thresh{thresh}"
                                )
                                train_job = training(
                                    training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                                )
                                train_job.rqmt["gpu_mem"] = 48
                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args_distill_decoding,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    specific_epoch=epochs,
                                    decoder_module=decoder_module,
                                    loss_name=f"ctc_loss_layer{layer_count}",
                                    prior_scales=prior_scales,
                                    lm_scales=lm_scales,
                                    run_test=True,
                                    test_dataset_tuples=test_dataset_tuples,
                                )
                                generate_report(results=results, exp_name=training_name)
                                distill_report[training_name] = results
                                del results
                        ###############################################################################################
                        # Random Selection
                        for rdn in [0.5, 1.0, 2.0]:
                            teacher_config = TeacherConfigV5(
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
                                increase_keepsome_epochs=None,
                                keep_random=rdn,
                                keep_threshold=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v5,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v3
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keep_rdn{rdn}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

                            if distill_scale == 0.25 and rdn == 1.0:
                                train_args_distill = {
                                    "config": train_config_distill,
                                    "network_module": distill_module_v6,
                                    "net_args": {
                                        "model_config_dict": asdict(student_config),
                                        "distill_config_dict": asdict(teacher_config),
                                    },
                                    "debug": False,
                                    "use_speed_perturbation": True,
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
                                train_args_distill_decoding = copy.deepcopy(train_args_distill)
                                train_args_distill_decoding["net_args"] = {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": None,
                                }
                                del train_args_distill_decoding["config"]["preload_from_files"]

                                decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                                training_name = (
                                    prefix_name
                                    + "/"
                                    + distill_module_v6
                                    + f"_chkpt_{num}"
                                    + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keep_rdn{rdn}"
                                )
                                train_job = training(
                                    training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                                )
                                train_job.rqmt["gpu_mem"] = 48
                                results = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args_distill_decoding,
                                    train_data=train_data,
                                    decoder_config=default_decoder_config,
                                    dev_dataset_tuples=dev_dataset_tuples,
                                    specific_epoch=epochs,
                                    decoder_module=decoder_module,
                                    loss_name=f"ctc_loss_layer{layer_count}",
                                    prior_scales=prior_scales,
                                    lm_scales=lm_scales,
                                    run_test=True,
                                    test_dataset_tuples=test_dataset_tuples,
                                )
                                generate_report(results=results, exp_name=training_name)
                                distill_report[training_name] = results
                                del results

            tmp_rep = {}
            for exp, dic in distill_report.items():
                tmp = {}
                for x in dic:
                    if any(str(y) in x.split("/")[6:] for y in train_epochs):
                        tmp[x] = dic[x]
                assert len(tmp) > 0 or "baselines" in exp, exp
                tmp_rep[exp] = tmp
            tmp_rep["baselines"] = distill_report["baselines"]
            tk.register_report(
                f"reports/distill_pos_enc_w_hubert_report_last_{epochs}",
                partial(build_hubert_distill_report, tmp_rep),
                required=distill_report,
                update_frequency=900,
            )
    T = 2
    epochs = 500
    keep = 3
    name = "large-ll60k"
    spec = 16
    dim = 384
    layer_count = 12
    heads = 8
    spec_start = 1
    chkpt = checkpoints[name + f"_best4"][0]
    size_report = {}
    size_report["baselines"] = {}
    for teacher_ep in [500]:
        for dim, heads,spec, spec_start in [(384, 8, 16, 1)]:
            size_report["baselines"][
                baseline_prefix
                + "/"
                + baseline_module
                + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}"
                ] = baselines[
                baseline_prefix
                + "/"
                + baseline_module
                + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}_0.2_radam_180bs_amp"
                ]
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=spec,
        num_repeat_feat=5,  # Jingjing style
    )
    for dim, layer_count, heads in [(384, 12, 8), (384, 8, 8), (384, 4, 8), (384, 16, 8), (384, 6, 8)]:
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

        student_config = StudentConfigV1(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_student,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            pos_emb_config=pos_emb_cfg,
            conformer_size=dim,
            num_layers=layer_count,
            num_heads=heads,
            ff_dim=4 * dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            mhsa_with_bias=True,
            conv_kernel_size=31,
            final_dropout=0.2,
            dropout_broadcast_axes=None,
            specauc_start_epoch=spec_start,
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
        )

        train_config_distill = {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
            + list(np.linspace(5e-5, 1e-7, 30)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "torch_amp_options": {"dtype": "bfloat16"},
        }
        for distill_scale in [0.25, 0.9, 1.0]:
            teacher_config = TeacherConfigV4(
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
                increase_keepsome_epochs=None,
            )
            train_args_distill = {
                "config": train_config_distill,
                "network_module": distill_module_v4,
                "net_args": {
                    "model_config_dict": asdict(student_config),
                    "distill_config_dict": asdict(teacher_config),
                },
                "debug": False,
                "use_speed_perturbation": True,
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
            train_args_distill_decoding = copy.deepcopy(train_args_distill)
            train_args_distill_decoding["net_args"] = {
                "model_config_dict": asdict(student_config),
                "distill_config_dict": None,
            }
            del train_args_distill_decoding["config"]["preload_from_files"]

            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

            training_name = (
                prefix_name
                + "/"
                + distill_module_v4
                + f"_chkpt_best4"
                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keepsome{keep}_sym"
            )
            train_job = training(training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48
            results = eval_model(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args_distill_decoding,
                train_data=train_data,
                decoder_config=default_decoder_config,
                dev_dataset_tuples=dev_dataset_tuples,
                specific_epoch=epochs,
                decoder_module=decoder_module,
                loss_name=f"ctc_loss_layer{layer_count}",
                prior_scales=prior_scales,
                lm_scales=lm_scales,
                run_test=True,
                test_dataset_tuples=test_dataset_tuples,
            )
            generate_report(results=results, exp_name=training_name)
            size_report[training_name] = results
            del results

    tmp_rep = {}
    for exp, dic in size_report.items():
        tmp = {}
        for x in dic:
            if any(str(y) in x.split("/")[6:] for y in [epochs]):
                tmp[x] = dic[x]
        assert len(tmp) > 0 or "baselines" in exp, exp
        tmp_rep[exp] = tmp
        # tmp_rep["baselines"] = size_report["baselines"]
    tk.register_report(
        f"reports/distill_sizes_report_last_{epochs}",
        partial(build_hubert_distill_report, tmp_rep),
        required=size_report,
        update_frequency=900,
    )

def eow_phon_ted_tune_pos_enc_w_hubert_new2():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/distill_pos_enc_w_hubert_new2"

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

    from ...pytorch_networks.ctc.hubert_tune_0711.hubert_tune_v2_cfg import ModelConfig

    hubert_report = {}
    checkpoints = {}
    for model in ["large-ll60k"]:
        model_config = ModelConfig(
            label_target_size=vocab_size_without_blank,
            final_dropout=0.2,
            model_name=model,
            finetune_layer=True,
            keep_layers=None,
            downsample_factor=2,
        )
        network_module = "ctc.hubert_tune_0711.hubert_tune_v2"
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
            lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        )
        generate_report(results=results, exp_name=training_name)
        hubert_report[training_name] = results
        del results
        if model == "large-ll60k":
            asr_model = prepare_asr_model(
                training_name,
                train_job,
                train_args_amp,
                with_prior=True,
                datasets=train_data,
                get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
            )
            checkpoints[model + f"_best4"] = (asr_model.checkpoint, asr_model.prior_file)
            blank_counts = calculate_blank_counts(
                prefix_name=training_name,
                train_job=train_job,
                train_args=train_args_amp,
                train_data=train_data,
                checkpoint="best4",
                debug=True,
            )
            tk.register_output(training_name + "/" + "blank_counts", blank_counts)
            blank_ratios = calculate_blank_ratios(
                prefix_name=training_name,
                train_job=train_job,
                train_args=train_args_amp,
                train_data=train_data,
                checkpoint="best4",
                debug=True,
            )
            tk.register_output(training_name + "/" + "blank_ratios", blank_ratios)

    from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v1_cfg import (
        ModelConfig as StudentConfigV1,
    #    DistillConfig as TeacherConfigV1,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        SpecaugConfig,
        ConformerPosEmbConfig,
    )
    #from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v4_cfg import (
    #    DistillConfig as TeacherConfigV4,
    #)
    from ...pytorch_networks.ctc.hubert_tune_0711.distill_pos_enc_hubert_v5_cfg import (
       DistillConfig as TeacherConfigV5,
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
    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )
    # v1 is broken
    # v2 kinda too
    # distill_module_v2 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v2"
    #distill_module_v3 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v3"
    #distill_module_v4 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v4"
    #distill_module_v5 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v5"
    distill_module_v6 = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v6"
    # TODO: activate v6 for everything
    decoder_module_v1 = "ctc.decoder.flashlight_ctc_distill_v1"

    from .pos_enc_baseline import eow_phon_ted_pos_enc_baseline

    baselines, base_checkpoints = eow_phon_ted_pos_enc_baseline(get_report=True)
    baseline_prefix = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/pos_enc_baseline"
    baseline_module = "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
    prior_scales = [0.3, 0.5, 0.7]
    lm_scales = [1.6, 1.8, 2.0]
    for name, (chkpt, prior_file) in checkpoints.items():
        name, num = name.split("_")
        train_epochs = [500]  # TODO: 1000
        for epochs in train_epochs:
            distill_report = {}
            distill_report["baselines"] = {}
            train_config_distill = {
                "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                + list(np.linspace(5e-5, 1e-7, 30)),
                #############
                "batch_size": 180 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 1,
                "gradient_clip_norm": 1.0,
                "torch_amp_options": {"dtype": "bfloat16"},
            }
            #######################################################################################################
            ##
            for dim, spec_start, spec, heads, layer_count in [
                (384, 1, 16, 8, 12),
            ]:
                specaug_config = SpecaugConfig(
                    repeat_per_n_frames=25,
                    max_dim_time=20,
                    max_dim_feat=spec,
                    num_repeat_feat=5,  # Jingjing style
                )
                for distill_scale in [0.25, 0.9, 1.0]:
                    for T in [2]:
                        for teacher_ep in [500, 1000]:
                            distill_report["baselines"][
                                baseline_prefix
                                + "/"
                                + baseline_module
                                + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}"
                            ] = baselines[
                                baseline_prefix
                                + "/"
                                + baseline_module
                                + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}_0.2_radam_180bs_amp"
                            ]
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

                        student_config = StudentConfigV1(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config_student,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            pos_emb_config=pos_emb_cfg,
                            conformer_size=dim,
                            num_layers=layer_count,
                            num_heads=heads,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.2,
                            conv_dropout=0.2,
                            ff_dropout=0.2,
                            mhsa_dropout=0.2,
                            mhsa_with_bias=True,
                            conv_kernel_size=31,
                            final_dropout=0.2,
                            dropout_broadcast_axes=None,
                            specauc_start_epoch=spec_start,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                        )
                        ###############################################################################################
                        # Baseline
                        teacher_config = TeacherConfigV5(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=False,
                            model_name=name,
                            kd_hyps=None,
                            normalize_stud=False,
                            keep_some_blanks=None,
                            trim_blanks=False,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=None,
                            mask_padding=False,
                            increase_keepsome_epochs=None,
                            keep_random=None,
                            keep_threshold=None,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v6,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                            "use_speed_perturbation": True,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]
                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v6
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        if not os.path.exists(
                            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.hold()
                            train_job.move_to_hpc = True
                        train_job.rqmt["gpu_mem"] = 48

                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module_v1,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=prior_scales,
                            lm_scales=lm_scales,
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                            run_best=False,
                            run_best_4=False,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        blank_counts = calculate_blank_counts(
                            prefix_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            checkpoint=epochs,
                            debug=True,
                        )
                        tk.register_output(training_name + "/" + "blank_counts", blank_counts)
                        ###############################################################################################
                        # Eliminate Blank
                        teacher_config = TeacherConfigV5(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                            keep_some_blanks=None,
                            trim_blanks=False,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=None,
                            mask_padding=False,
                            increase_keepsome_epochs=None,
                            keep_random=None,
                            keep_threshold=None,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v6,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                            "use_speed_perturbation": True,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v6
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        if not os.path.exists(
                            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.hold()
                            train_job.move_to_hpc = True
                        train_job.rqmt["gpu_mem"] = 48
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module_v1,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=prior_scales,
                            lm_scales=lm_scales,
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                            run_best=False,
                            run_best_4=False,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        ###############################################################################################
                        # Trim Blank
                        teacher_config = TeacherConfigV5(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                            keep_some_blanks=None,
                            trim_blanks=True,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=None,
                            mask_padding=False,
                            increase_keepsome_epochs=None,
                            keep_random=None,
                            keep_threshold=None,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v6,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                            "use_speed_perturbation": True,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v6
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_trim_blanks"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        if not os.path.exists(
                            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.hold()
                            train_job.move_to_hpc = True
                        train_job.rqmt["gpu_mem"] = 48
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module_v1,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            prior_scales=prior_scales,
                            lm_scales=lm_scales,
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                        ###############################################################################################
                        # Prior Correction, maybe remove?
                        teacher_config = TeacherConfigV5(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            normalize_stud=False,
                            kd_hyps=None,
                            prior_scale=0.5,
                            prior_file=prior_file,
                            keep_some_blanks=None,
                            warmup_loss=None,
                            mask_padding=False,
                            trim_blanks=False,
                            increase_keepsome_epochs=None,
                            keep_random=None,
                            keep_threshold=None,
                        )
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module_v6,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                            "use_speed_perturbation": True,
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
                        train_args_distill_decoding = copy.deepcopy(train_args_distill)
                        train_args_distill_decoding["net_args"] = {
                            "model_config_dict": asdict(student_config),
                            "distill_config_dict": None,
                        }
                        del train_args_distill_decoding["config"]["preload_from_files"]

                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module_v6
                            + f"_chkpt_{num}"
                            + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank_prior"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                        )
                        if not os.path.exists(
                            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.hold()
                            train_job.move_to_hpc = True
                        train_job.rqmt["gpu_mem"] = 48
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill_decoding,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=epochs,
                            decoder_module=decoder_module_v1,
                            loss_name=f"ctc_loss_layer{layer_count}",
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                        ###############################################################################################
                        # Symmetric Selection
                        for keep in [1, 2, 3, 4, 5]:
                            teacher_config = TeacherConfigV5(
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
                                increase_keepsome_epochs=None,
                                keep_random=None,
                                keep_threshold=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v6,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v6
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keepsome{keep}_sym"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.hold()
                                train_job.move_to_hpc = True
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
                            blank_counts = calculate_blank_counts(
                                prefix_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                checkpoint=epochs,
                                debug=True,
                            )
                            tk.register_output(training_name + "/" + "blank_counts", blank_counts)

                        ###############################################################################################
                        # Thresholding Selection
                        for thresh in [0.1, 0.2]:
                            teacher_config = TeacherConfigV5(
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
                                increase_keepsome_epochs=None,
                                keep_random=None,
                                keep_threshold=thresh,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v6,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v6
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keep_thresh{thresh}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.hold()
                                train_job.move_to_hpc = True
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
                        ###############################################################################################
                        # Random Selection
                        for rdn in [0.5, 1.0, 2.0]:
                            teacher_config = TeacherConfigV5(
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
                                increase_keepsome_epochs=None,
                                keep_random=rdn,
                                keep_threshold=None,
                            )
                            train_args_distill = {
                                "config": train_config_distill,
                                "network_module": distill_module_v6,
                                "net_args": {
                                    "model_config_dict": asdict(student_config),
                                    "distill_config_dict": asdict(teacher_config),
                                },
                                "debug": False,
                                "use_speed_perturbation": True,
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
                            train_args_distill_decoding = copy.deepcopy(train_args_distill)
                            train_args_distill_decoding["net_args"] = {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": None,
                            }
                            del train_args_distill_decoding["config"]["preload_from_files"]

                            decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"

                            training_name = (
                                prefix_name
                                + "/"
                                + distill_module_v6
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keep_rdn{rdn}"
                            )
                            train_job = training(
                                training_name, train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.hold()
                                train_job.move_to_hpc = True
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"ctc_loss_layer{layer_count}",
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results

            tmp_rep = {}
            for exp, dic in distill_report.items():
                tmp = {}
                for x in dic:
                    if any(str(y) in x.split("/")[6:] for y in train_epochs):
                        tmp[x] = dic[x]
                assert len(tmp) > 0 or "baselines" in exp, exp
                tmp_rep[exp] = tmp
            tmp_rep["baselines"] = distill_report["baselines"]
            tk.register_report(
                f"reports/distill_pos_enc_w_hubert_report_lastv2_{epochs}",
                partial(build_hubert_distill_report, tmp_rep),
                required=distill_report,
                update_frequency=900,
            )
