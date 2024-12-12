import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from sisyphus import tk
from functools import partial
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model
from .tune_eval import build_base_report, eval_model, build_hubert_distill_report
from ...report import generate_report


def eow_phon_ls960_distill_base():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon/baselines"

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

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )
    from ...pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
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

    from ...pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ModelConfig as RelPosModelConfigV1,
        ConformerPosEmbConfig,
        VGG4LayerActFrontendV1Config_mod,
        SpecaugConfig,
    )

    report = {}
    for dim in [384, 512]:
        for spec_start in [1]:
            for epochs in [500, 1000]:
                for spec in [16]:
                    for num_heads in [8]:
                        if dim == 512 and num_heads == 12:
                            continue
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
                            out_features=dim,
                            activation=None,
                        )
                        specaug_config_test = SpecaugConfig(
                            repeat_per_n_frames=25,
                            max_dim_time=20,
                            max_dim_feat=spec,
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
                        model_config_pos_enc = RelPosModelConfigV1(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config,
                            specaug_config=specaug_config_test,
                            label_target_size=vocab_size_without_blank,
                            pos_emb_config=pos_emb_cfg,
                            conformer_size=dim,
                            num_layers=12,
                            num_heads=num_heads,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.1,
                            conv_dropout=0.1,
                            ff_dropout=0.1,
                            mhsa_dropout=0.1,
                            mhsa_with_bias=True,
                            conv_kernel_size=31,
                            final_dropout=0.1,
                            dropout_broadcast_axes=None,
                            specauc_start_epoch=spec_start,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                        )
                        network_module_pos_enc = (
                            "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
                        )
                        train_config = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-16,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))  # try higher start
                            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                            + list(np.linspace(5e-5, 1e-7, 30)),
                            #############
                            "batch_size": 180 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                        }
                        train_args = {
                            "config": train_config,
                            "network_module": network_module_pos_enc,
                            "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                            "debug": True,
                        }
                        results = {}
                        training_name = (
                            prefix_name
                            + "/"
                            + network_module_pos_enc
                            + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}"
                        )
                        train_job = training(
                            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                        )

                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            result_dict=results,
                            loss_name=f"ctc_loss_layer12",
                            specific_epoch=epochs,
                            lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0],
                            prior_scales=[0.1, 0.2, 0.3, 0.5],
                        )
                        generate_report(results=results, exp_name=training_name)
                        report[training_name] = results
                        del results
    tk.register_report("reports/ls_baselines_report", partial(build_base_report, report), required=report)
    report = {}
    for dim in [384, 512]:
        for spec_start in [1]:
            for epochs in [500, 1000]:
                for spec in [16]:
                    for num_heads in [8, 12]:
                        if dim == 512 and num_heads == 12:
                            continue
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
                            out_features=dim,
                            activation=None,
                        )
                        specaug_config_test = SpecaugConfig(
                            repeat_per_n_frames=25,
                            max_dim_time=20,
                            max_dim_feat=spec,
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
                        model_config_pos_enc = RelPosModelConfigV1(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config,
                            specaug_config=specaug_config_test,
                            label_target_size=vocab_size_without_blank,
                            pos_emb_config=pos_emb_cfg,
                            conformer_size=dim,
                            num_layers=12,
                            num_heads=num_heads,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.1,
                            conv_dropout=0.1,
                            ff_dropout=0.1,
                            mhsa_dropout=0.1,
                            mhsa_with_bias=True,
                            conv_kernel_size=31,
                            final_dropout=0.1,
                            dropout_broadcast_axes=None,
                            specauc_start_epoch=spec_start,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                        )
                        network_module_pos_enc = (
                            "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
                        )
                        train_config = {
                            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 40) // 2))  # try higher start
                            + list(np.linspace(5e-4, 5e-5, (epochs - 40) // 2))
                            + list(np.linspace(5e-5, 1e-7, 40)),
                            #############
                            "batch_size": 300 * 16000,  # TODO check if batch causes issues
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "torch_amp_options": {"dtype": "bfloat16"},
                            "gradient_clip_norm": 1.0,
                        }
                        # batch size, adamw, speed pert, gradient clip,
                        train_args = {
                            "config": train_config,
                            "network_module": network_module_pos_enc,
                            "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                            "debug": True,
                            "use_speed_perturbation": True,
                        }
                        results = {}
                        training_name = (
                            prefix_name
                            + "/"
                            + network_module_pos_enc
                            + f"_better_params_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}"
                        )
                        train_job = training(
                            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                        )
                        if dim == 512:
                            train_job.rqmt["gpu_mem"] = 24
                        elif dim == 384:
                            train_job.rqmt["gpu_mem"] = 48

                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            result_dict=results,
                            loss_name=f"ctc_loss_layer12",
                            specific_epoch=epochs,
                            lm_scales=[2.0, 2.2, 2.4, 2.6],
                            prior_scales=[0.1, 0.2, 0.3],
                        )
                        generate_report(results=results, exp_name=training_name)
                        report[training_name] = results
                        del results
    tk.register_report("reports/ls_best_report", partial(build_base_report, report), required=report)
    return report


def eow_phon_ls960_distill_hubert():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon/hubert"

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

    train_dataset_tuples = {}
    for testset in ["train-other-960"]:
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

    from ...pytorch_networks.ctc.conformer_distill_1007.hubert_tune_v2_cfg import ModelConfig

    hubert_report = {}
    checkpoints = {}
    for model in [
        "large-ll60k",
        "large-ls960-ft",
    ]:
        for factor in [1, 2]:
            model_config = ModelConfig(
                label_target_size=vocab_size_without_blank,
                final_dropout=0.2,
                model_name=model,
                finetune_layer=True,
                keep_layers=None,
                downsample_factor=factor,
            )
            network_module = "ctc.conformer_distill_1007.hubert_tune_v2"
            train_config_24gbgpu_amp = {
                "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
                "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                + list(np.linspace(5e-4, 5e-5, 110))
                + list(np.linspace(5e-5, 1e-7, 30)),
                #############
                "batch_size": 120 * 16000 if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 60 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 3 if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 6,
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

            training_name = prefix_name + "/" + network_module + f"_{model}_{factor}"
            train_job = training(training_name, train_data, train_args_amp, num_epochs=50, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            results = eval_model(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args_amp,
                train_data=train_data,
                decoder_config=default_decoder_config,
                dev_dataset_tuples=dev_dataset_tuples,
                prior_scales=[0.1, 0.2],
                lm_scales=[1.6, 1.8, 2.0],
            )
            generate_report(results=results, exp_name=training_name)
            hubert_report[training_name] = results
            del results
            if model == "large-ll60k" and factor == 2:
                for checkpoint in ["best4"]:
                    for n_best in [10]:
                        if checkpoint == "best4":
                            asr_model = prepare_asr_model(
                                prefix_name,
                                train_job,
                                train_args_amp,
                                with_prior=True,
                                datasets=train_data,
                                get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
                            )
                        else:
                            asr_model = prepare_asr_model(
                                prefix_name,
                                train_job,
                                train_args_amp,
                                with_prior=True,
                                datasets=train_data,
                                get_specific_checkpoint=checkpoint,
                            )
                        checkpoints[model + f"_{checkpoint}_{n_best}"] = (
                            asr_model.checkpoint,
                            asr_model.prior_file,
                            0.5,
                        )
    tk.register_report("reports/ls_hubert_report", partial(build_base_report, hubert_report), required=hubert_report)

    from ...pytorch_networks.ctc.conformer_distill_1007.distill_pos_enc_hubert_v2_cfg import (
        ModelConfig as StudentConfigV1,
        DistillConfig as TeacherConfigV1,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        SpecaugConfig,
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
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon/baseline_distill/"
    distill_module_v2 = "ctc.conformer_distill_1007.distill_pos_enc_hubert_v2"
    baselines = eow_phon_ls960_distill_base()
    baseline_prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon/baselines"
    baseline_module = "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
    for name, (chkpt, prior_file, prior_scale) in checkpoints.items():
        name, num, n_best = name.split("_")
        for epochs, dim, spec_start, spec, num_heads, layer_count in [
            (500, 512, 1, 16, 8, 12),
            (1000, 512, 1, 16, 8, 12),
        ]:
            distill_report = {}
            distill_report["baselines"] = {}
            base_train_config_distill = {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 40) // 2))  # try higher start
                + list(np.linspace(5e-4, 5e-5, (epochs - 40) // 2))
                + list(np.linspace(5e-5, 1e-7, 40)),
                #############
                "batch_size": 300 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 1,
                "torch_amp_options": {"dtype": "bfloat16"},
                "gradient_clip_norm": 1.0,
            }
            for distill_scale in [0.0, 0.1, 0.25, 0.9, 1.0]:
                if distill_scale < 0.25 and epochs > 500:
                    continue
                for T in [2]:
                    specaug_config = SpecaugConfig(
                        repeat_per_n_frames=25,
                        max_dim_time=20,
                        max_dim_feat=spec,
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
                    distill_report["baselines"][
                        baseline_prefix + "/" + baseline_module + f"_500_{dim}_{num_heads}_{spec}_{spec_start}"
                    ] = baselines[
                        baseline_prefix
                        + "/"
                        + baseline_module
                        + f"_better_params_500_{dim}_{num_heads}_{spec}_{spec_start}"
                    ]
                    distill_report["baselines"][
                        baseline_prefix + "/" + baseline_module + f"_1000_{dim}_{num_heads}_{spec}_{spec_start}"
                    ] = baselines[
                        baseline_prefix
                        + "/"
                        + baseline_module
                        + f"_better_params_1000_{dim}_{num_heads}_{spec}_{spec_start}"
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
                        num_heads=num_heads,
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
                    train_args_distill = {
                        "config": base_train_config_distill,
                        "network_module": distill_module_v2,
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
                        + distill_module_v2
                        + f"_chkpt_{num}"
                        + f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}"
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
                        lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                        prior_scales=[0.1, 0.2, 0.3],
                    )
                    generate_report(results=results, exp_name=training_name)
                    distill_report[training_name] = results
                    del results
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
                        "config": base_train_config_distill,
                        "network_module": distill_module_v2,
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
                        + distill_module_v2
                        + f"_chkpt_{num}"
                        + f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank"
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
                        lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                        prior_scales=[0.1, 0.2, 0.3],
                    )
                    generate_report(results=results, exp_name=training_name)
                    distill_report[training_name] = results
                    del results
                    if distill_scale < 0.25:
                        continue
                    for keep in [1, 3, 5, 100]:
                        teacher_config = TeacherConfigV1(
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
                            trim_blanks=False,
                        )
                        train_args_distill = {
                            "config": base_train_config_distill,
                            "network_module": distill_module_v2,
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
                            prefix_name + "/" + distill_module_v2 + f"_chkpt_{num}"
                            f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_keepsome{keep}"
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
                            lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                            prior_scales=[0.1, 0.2, 0.3],
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                    continue  # TODO: not more for now

                    teacher_config = TeacherConfigV1(
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
                        "config": base_train_config_distill,
                        "network_module": distill_module_v2,
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
                        prefix_name + "/" + distill_module_v2 + f"_chkpt_{num}"
                        f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_trim_blanks"
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
                        lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                        prior_scales=[0.1, 0.2, 0.3],
                    )
                    generate_report(results=results, exp_name=training_name)
                    distill_report[training_name] = results
                    del results

                    for start_epoch in [5, 10, 25, 50, -5, -10, -25, -50, -100, -200]:
                        teacher_config = TeacherConfigV1(
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
                        train_args_distill = {
                            "config": base_train_config_distill,
                            "network_module": distill_module_v2,
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
                            prefix_name + "/" + distill_module_v2 + f"_chkpt_{num}"
                            f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank_{start_epoch}"
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
                            lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                            prior_scales=[0.1, 0.2, 0.3],
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                    teacher_config = TeacherConfigV1(
                        distill_scale=distill_scale,
                        ctc_scale=1 - distill_scale,
                        t=T,
                        eliminate_blanks=True,
                        model_name=name,
                        normalize_stud=False,
                        kd_hyps=None,
                        prior_scale=prior_scale,
                        prior_file=prior_file,
                        keep_some_blanks=False,
                        warmup_loss=None,
                        mask_padding=False,
                        trim_blanks=False,
                    )
                    train_args_distill = {
                        "config": base_train_config_distill,
                        "network_module": distill_module_v2,
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
                        prefix_name + "/" + distill_module_v2 + f"_chkpt_{num}"
                        f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blank_prior"
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
                    )
                    generate_report(results=results, exp_name=training_name)
                    distill_report[training_name] = results
                    del results

                    teacher_config = TeacherConfigV1(
                        distill_scale=distill_scale,
                        ctc_scale=1,
                        t=T,
                        eliminate_blanks=False,
                        model_name=name,
                        kd_hyps=kd_hyps,
                        normalize_stud=False,
                        keep_some_blanks=False,
                        trim_blanks=False,
                        prior_file=None,
                        prior_scale=None,
                        warmup_loss=None,
                        mask_padding=False,
                    )
                    train_args_distill = {
                        "config": base_train_config_distill,
                        "network_module": distill_module_v2,
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
                        prefix_name + "/" + distill_module_v2 + f"_chkpt_{num}_{n_best}"
                        f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_kdhyps"
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
                    )
                    generate_report(results=results, exp_name=training_name)
                    distill_report[training_name] = results
                    del results
                    for warmup in [10, 25, 50, 100]:
                        teacher_config = TeacherConfigV1(
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
                            eliminate_blanks=True,
                            model_name=name,
                            kd_hyps=None,
                            normalize_stud=False,
                            prior_file=None,
                            prior_scale=None,
                            warmup_loss=warmup,
                            mask_padding=False,
                            keep_some_blanks=False,
                            trim_blanks=False,
                        )
                        train_args_distill = {
                            "config": base_train_config_distill,
                            "network_module": distill_module_v2,
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
                            prefix_name + "/" + distill_module_v2 + f"_chkpt_{num}"
                            f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_elim_blanks"
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
                            lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                            prior_scales=[0.1, 0.2, 0.3, 0.5],
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                        lr_long = (
                            list(np.linspace(7e-6, 5e-4, warmup))
                            + list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                            + list(np.linspace(5e-5, 1e-7, 30)),
                        )
                        longer_train_config_distill = copy.deepcopy(base_train_config_distill)
                        longer_train_config_distill["learning_rates"] = lr_long
                        teacher_config = TeacherConfigV1(
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
                            mask_padding=False,
                            keep_some_blanks=False,
                            trim_blanks=False,
                        )
                        train_args_distill = {
                            "config": longer_train_config_distill,
                            "network_module": distill_module_v2,
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
                            prefix_name + "/" + distill_module_v2 + f"_chkpt_{num}"
                            f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_warm{warmup}_long"
                        )
                        train_job = training(
                            training_name,
                            train_data,
                            train_args_distill,
                            num_epochs=epochs + warmup,
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
                            specific_epoch=epochs + warmup,
                            decoder_module=decoder_module,
                            loss_name=f"ctc_loss_layer{layer_count}",
                            lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                            prior_scales=[0.1, 0.2, 0.3, 0.5],
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
                    for keep in [1, 5]:
                        teacher_config = TeacherConfigV1(
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
                            trim_blanks=False,
                        )
                        train_args_distill = {
                            "config": base_train_config_distill,
                            "network_module": distill_module_v2,
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
                            prefix_name + "/" + distill_module_v2 + f"_chkpt_{num}"
                            f"_{epochs}_{layer_count}_{dim}_{num_heads}_{spec}_{spec_start}_{distill_scale}_{T}_warm25_keepsome{keep}"
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
                            prior_scales=[0.3, 0.5, 0.7, 0.9],
                            lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results
            tk.register_report(
                f"reports/ls_distill_pos_enc_w_hubert_report_{epochs}",
                partial(build_hubert_distill_report, distill_report),
                required=distill_report,
            )
