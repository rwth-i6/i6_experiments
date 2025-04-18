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
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon, build_combined_eow_phon_training_datasets
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, generate_kd_hypothesis, calculate_blank_counts, prepare_asr_model
from ...report import generate_report
from .tune_eval import eval_model, build_hubert_report, build_hubert_distill_report
from functools import partial


def eow_phon_ted_distill_more_data():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/distill_more_data"

    ted_train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    ted_train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=ted_train_settings,
    )
    ted_label_datastream = cast(LabelDatastream, ted_train_data.datastreams["labels"])
    ted_vocab_size_without_blank = ted_label_datastream.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=ted_train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=ted_train_settings,
        )
    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=ted_label_datastream.vocab,
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
            label_target_size=ted_vocab_size_without_blank,
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
        train_job = training(training_name, ted_train_data, train_args_amp, num_epochs=50, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args_amp,
            train_data=ted_train_data,
            decoder_config=default_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            specific_epoch=keep_epochs,
            prior_scales=[0.1, 0.2, 0.3, 0.4],
            lm_scales=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        )
        generate_report(results=results, exp_name=training_name)
        hubert_report[training_name] = results
        del results
        asr_model = prepare_asr_model(
            prefix_name,
            train_job,
            train_args_amp,
            with_prior=True,
            datasets=ted_train_data,
            get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
        )
        checkpoints[model + f"_best4"] = asr_model.checkpoint

    comb_train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=20,
        train_seq_ordering="laplace:.1000",
    )

    comb_train_data = build_combined_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=comb_train_settings,
    )

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
    distill_module_v5_no_lab = "ctc.hubert_tune_0711.distill_pos_enc_hubert_v5_no_lab"

    from .pos_enc_baseline import eow_phon_ted_pos_enc_baseline

    baselines, base_checkpoints = eow_phon_ted_pos_enc_baseline(get_report=True)
    baseline_prefix = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/pos_enc_baseline"
    baseline_module = "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"

    for name, chkpt in checkpoints.items():
        name, num = name.split("_")
        for epochs in [250, 500, 1000]:
            distill_report = {}
            distill_report["baselines"] = {}
            for dim, spec_start, spec, heads, layer_count, teacher_ep in [
                # (250, 384, 11, 8, 4, 12, 500),
                (384, 1, 16, 8, 12, 500),
            ]:
                for distill_scale in [1.0]:
                    for T in [2]:
                        distill_report["baselines"][
                            baseline_prefix + "/" + baseline_module + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}"
                        ] = baselines[
                            baseline_prefix + "/" + baseline_module + f"_{teacher_ep}_{dim}_{heads}_{spec}_{spec_start}"
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
                            label_target_size=ted_vocab_size_without_blank,
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
                        for keep in [2]:
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
                                "accum_grad_multiple_step": 2,
                                "torch_amp_options": {"dtype": "bfloat16"},
                            }
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
                                "network_module": distill_module_v5_no_lab,
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
                                + distill_module_v5_no_lab
                                + f"_chkpt_{num}"
                                + f"_{epochs}_{layer_count}_{dim}_{heads}_{spec}_{spec_start}_{distill_scale}_{T}_keepsome{keep}_sym"
                            )
                            train_job = training(
                                training_name, comb_train_data, train_args_distill, num_epochs=epochs, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_distill_decoding,
                                train_data=comb_train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                specific_epoch=epochs,
                                decoder_module=decoder_module,
                                loss_name=f"KL",
                                prior_scales=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                                lm_scales=[1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            distill_report[training_name] = results
                            del results
