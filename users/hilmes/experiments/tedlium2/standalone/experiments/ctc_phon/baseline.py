from dataclasses import asdict
import numpy as np
from typing import cast, Dict
from sisyphus import tk
from functools import partial

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from .tune_eval import QuantArgs
from ...data.common import DatasetSettings, build_test_dataset, build_st_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model
from ...report import generate_report, build_memristor_base_report
from .tune_eval import tune_and_evaluate_helper, eval_model, build_report, build_distill_report


def eow_phon_ted_1023_base(full=False):
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon"

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

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
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

    model_config = ModelConfig(
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
    )

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_384dim_sub4_24gbgpu_50eps_amp"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=default_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
    )
    generate_report(results=results, exp_name=training_name)
    del results

    train_config_24gbgpu_amp_sb = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args = {
        "config": train_config_24gbgpu_amp_sb,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_384dim_sub4_24gbgpu_50eps_amp_sb"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=default_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
    )
    generate_report(results=results, exp_name=training_name)
    del results

    train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 180 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
    }
    train_args = {
        "config": train_config,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_384dim_sub4_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    PRETRAIN_CHECKPOINT_DISTILL_V1 = asr_model.checkpoint
    lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    prior_scales = [0.7, 0.9]
    res, _ = tune_and_evaluate_helper(
        training_name,
        asr_model,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
    )
    res, _ = tune_and_evaluate_helper(
        training_name + "/best4",
        asr_model_best4,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc"),
    )
    res, _ = tune_and_evaluate_helper(
        training_name + "/best",
        asr_model_best,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    generate_report(results=results, exp_name=training_name)  # TODO current best with 7.083
    del results
    from ...pytorch_networks.ctc.conformer_1023.quant.baseline_quant_v1_cfg import QuantModelConfigV1

    num_iterations = 1
    results = {}
    for activation_bit, weight_bit in [
        (8, 8),
        (8, 4),
        (8, 3),
        (8, 2),
        (8, 1),
        (6, 6),
        (6, 4),
        (6, 3),
        (6, 2),
        (6, 1),
        (4, 4),
        (4, 3),
        (4, 2),
        (4, 1),
    ]:
        model_config_quant_v1 = QuantModelConfigV1(
            weight_quant_dtype="qint8",
            weight_quant_method="per_tensor",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor",
            moving_average=0.01,
            weight_bit_prec=weight_bit,
            activation_bit_prec=activation_bit,
            linear_quant_output=True,
        )
        quant_args = QuantArgs(
            sample_ls=[100] if weight_bit < 8 or activation_bit < 8 else [100],
            quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
            decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
            num_iterations=num_iterations,
            datasets=train_data,
            network_module="ctc.conformer_1023.quant.baseline_quant_v2",
        )
        quant_str = f"/quantize/weight_{weight_bit}_act_{activation_bit}"
        asr_model = prepare_asr_model(
            training_name + quant_str,
            train_job,
            train_args,
            with_prior=True,
            datasets=train_data,
            get_specific_checkpoint=250,
        )
        res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
            training_name,
            asr_model,
            default_decoder_config,
            lm_scales=[2.8],
            prior_scales=[0.7],
            quant_args=quant_args,
            quant_str=quant_str,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=None,
        )
        results.update(res)
    generate_report(results=results, exp_name=training_name + f"_quantize/combined_results")
    quant_results = {}
    quant_results["baselines"] = {}
    quant_results["baselines"][training_name] = results
    del results
    results = {}
    for activation_bit, weight_bit in [
        (8, 8),
        (8, 6),
        (8, 5),
        (8, 4),
        (8, 3),
        (8, 2),
        (8, 1.5),
    ]:
        model_config_quant_v1 = QuantModelConfigV1(
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
            linear_quant_output=True,
        )
        quant_args = QuantArgs(
            sample_ls=[100],
            quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
            decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
            num_iterations=num_iterations,
            datasets=train_data,
            network_module="ctc.conformer_1023.quant.baseline_quant_v2_mem",
        )
        quant_str = f"/quantize/weight_{weight_bit}_act_{activation_bit}_mem"
        asr_model = prepare_asr_model(
            training_name + quant_str,
            train_job,
            train_args,
            with_prior=True,
            datasets=train_data,
            get_specific_checkpoint=250,
        )
        res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
            training_name,
            asr_model,
            default_decoder_config,
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            prior_scales=[0.5, 0.7, 0.9, 1.1],
            quant_args=quant_args,
            quant_str=quant_str,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=None,
        )
        results.update(res)
    generate_report(results=results, exp_name=training_name + f"_quantize/qat_memristor_results")
    quant_results[training_name] = results
    del results

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
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_better384dim_sub4_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    prior_scales = [0.1, 0.3, 0.5, 0.7, 0.9]
    lm_scales = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    res, _ = tune_and_evaluate_helper(
        training_name,
        asr_model,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
    )
    res, _ = tune_and_evaluate_helper(
        training_name + "/best4",
        asr_model_best4,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc"),
    )
    res, _ = tune_and_evaluate_helper(
        training_name + "/best",
        asr_model_best,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    generate_report(results=results, exp_name=training_name)
    quant_results["baselines"][training_name] = results
    del results

    results = {}
    for activation_bit, weight_bit in [
        (8, 8),
        (8, 6),
        (8, 5),
        (8, 4),
        (8, 3),
        (8, 2),
        (8, 1.5),
    ]:
        model_config_quant_v1 = QuantModelConfigV1(
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
            linear_quant_output=True,
        )
        quant_args = QuantArgs(
            sample_ls=[100],
            quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
            decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
            num_iterations=num_iterations,
            datasets=train_data,
            network_module="ctc.conformer_1023.quant.baseline_quant_v2_mem",
        )
        quant_str = f"/quantize/weight_{weight_bit}_act_{activation_bit}_mem"
        asr_model = prepare_asr_model(
            training_name + quant_str,
            train_job,
            train_args,
            with_prior=True,
            datasets=train_data,
            get_specific_checkpoint=250,
        )
        res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
            training_name,
            asr_model,
            default_decoder_config,
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            prior_scales=[0.5, 0.7, 0.9, 1.1],
            quant_args=quant_args,
            quant_str=quant_str,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=None,
        )
        results.update(res)
    generate_report(results=results, exp_name=training_name + f"_quantize/qat_memristor_results")
    quant_results[training_name] = results
    del results

    results = {}
    for activation_bit, weight_bit in [
        (8, 8),
        (8, 6),
        (8, 5),
        (8, 4),
        (8, 3),
        (8, 2),
        (8, 1.5),
    ]:
        model_config_quant_v1 = QuantModelConfigV1(
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
            linear_quant_output=True,
        )
        quant_args = QuantArgs(
            sample_ls=[100],
            quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
            decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
            num_iterations=num_iterations,
            datasets=train_data,
            network_module="ctc.conformer_1023.quant.baseline_quant_v3_mem",
        )
        quant_str = f"/quantizev3/weight_{weight_bit}_act_{activation_bit}_mem"
        asr_model = prepare_asr_model(
            training_name + quant_str,
            train_job,
            train_args,
            with_prior=True,
            datasets=train_data,
            get_specific_checkpoint=250,
        )
        res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
            training_name,
            asr_model,
            default_decoder_config,
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            prior_scales=[0.5, 0.7, 0.9, 1.1],
            quant_args=quant_args,
            quant_str=quant_str,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=None,
        )
        results.update(res)
    generate_report(results=results, exp_name=training_name + f"_quantize/qat_memristor_results_v3")
    quant_results[training_name + "_quant_with_conv"] = results
    tk.register_report(
        "reports/baseline_quant", partial(build_memristor_base_report, quant_results), required=quant_results
    )
    del results

    report = {}
    if full is True:
        for dim in [64, 128, 256, 384, 512, 768, 1024]:
            for layer_count in [4, 6, 8, 12, 16, 20]:
                # Baseline: 7.0
                #       64    128   256   384   512  768  1024
                # 4:    16.5  11.5  9.1   8.3   8.3  7.9  7.8
                # 6:    13.7  10.3  8.3   8.1   7.7  7.3  7.5
                # 8:    13.0  9.7   8.0   7.1   7.3  7.3  7.4
                # 12:   11.4  9.0   7.3   7.2   7.3  7.2
                # 16:         7.9         6.8   7.1  8.9  9.2

                # Dropout 0.1:
                #       64    128   256   384   512
                # 4:    14.4  10.6
                # 6:    12.8  9.7
                # 8:    11.8  9.0
                # 12:   10.9  8.4
                # 16:   10.3  7.9

                # Dropout 0.0:
                #       64    128   256   384   512
                # 4:    13.0  10.5
                # 6:    12.1  9.4
                # 8:    11.3  9.2
                # 12:   10.6  8.5
                # 16:   9.9   8.3

                dropout = 0.2  # if dim > 128 else 0.0
                frontend_config_dims = VGG4LayerActFrontendV1Config_mod(
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

                model_config_dim = ModelConfig(
                    feature_extraction_config=fe_config,
                    frontend_config=frontend_config_dims,
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
                )
                small = layer_count > 16 and dim > 768
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
                    "batch_size": 180 * 16000 if not small else 90 * 16000,
                    "max_seq_length": {"audio_features": 35 * 16000},
                    "accum_grad_multiple_step": 1 if not small else 2,
                }
                train_args = {
                    "config": train_config,
                    "network_module": network_module,
                    "net_args": {"model_config_dict": asdict(model_config_dim)},
                    "debug": False,
                }
                results = {}
                training_name = prefix_name + "/" + network_module + f"_{layer_count}_{dim}_sub4_50eps"
                train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
                if dim > 768 or layer_count > 8:
                    train_job.rqmt["gpu_mem"] = 24
                if dim == 384 and layer_count == 16:
                    PRETRAIN_CHECKPOINT_DISTILL_V2 = train_job.out_checkpoints[250]
                results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data,
                    decoder_config=default_decoder_config,
                    dev_dataset_tuples=dev_dataset_tuples,
                    result_dict=results,
                )
                generate_report(results=results, exp_name=training_name)
                report[training_name] = results
                del results

                if dim == 384 and layer_count == 16:
                    train_config = {
                        "optimizer": {
                            "class": "radam",
                            "epsilon": 1e-16,
                            "weight_decay": 1e-2,
                            "decoupled_weight_decay": True,
                        },
                        "learning_rates": list(np.linspace(7e-6, 5e-4, 220))
                        + list(np.linspace(5e-4, 5e-5, 220))
                        + list(np.linspace(5e-5, 1e-7, 60)),
                        #############
                        "batch_size": 180 * 16000,
                        "max_seq_length": {"audio_features": 35 * 16000},
                        "accum_grad_multiple_step": 1,
                    }
                    train_args = {
                        "config": train_config,
                        "network_module": network_module,
                        "net_args": {"model_config_dict": asdict(model_config_dim)},
                        "debug": False,
                    }
                    training_name = prefix_name + "/" + network_module + f"_{layer_count}_{dim}_sub4_100eps"
                    train_job = training(training_name, train_data, train_args, num_epochs=500, **default_returnn)
                    if dim > 768 or layer_count > 8:
                        train_job.rqmt["gpu_mem"] = 24
                    results = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data,
                        decoder_config=default_decoder_config,
                        dev_dataset_tuples=dev_dataset_tuples,
                        specific_epoch=500,
                    )
                    generate_report(results=results, exp_name=training_name)
                    report[training_name] = results
                    del results

        tk.register_report("reports/size_report", partial(build_report, report), required=report)

        from ...pytorch_networks.ctc.conformer_distill_1206.self_distill_conformer_v1_cfg import (
            ModelConfig as StudentConfig,
            DistillConfig as TeacherConfig,
        )

        distill_report = {}
        distill_report["baselines"] = {}
        no_drop_stud_report = {}
        no_drop_stud_report["baselines"] = {}
        larger_distill_report = {}
        larger_distill_report["baselines"] = {}
        for dim in [64, 128, 256]:
            for layer_count in [4, 8, 12]:
                # for distill_scale in [0.35, 0.25, 1.0]:
                for distill_scale in [0.25]:
                    # for T in [1, 2, 3]:
                    for T in [2]:
                        distill_report["baselines"][
                            prefix_name + "/" + network_module + f"_{layer_count}_{dim}_sub4_50eps"
                        ] = report[prefix_name + "/" + network_module + f"_{layer_count}_{dim}_sub4_50eps"]
                        no_drop_stud_report["baselines"][
                            prefix_name + "/" + network_module + f"_{layer_count}_{dim}_sub4_50eps"
                        ] = report[prefix_name + "/" + network_module + f"_{layer_count}_{dim}_sub4_50eps"]
                        larger_distill_report["baselines"][
                            prefix_name + "/" + network_module + f"_{layer_count}_{dim}_sub4_50eps"
                        ] = report[prefix_name + "/" + network_module + f"_{layer_count}_{dim}_sub4_50eps"]

                        distill_module = "ctc.conformer_distill_1206.self_distill_conformer_v4"
                        teacher_config = TeacherConfig(
                            frontend_config=default_frontend_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=384,
                            num_layers=12,
                            num_heads=4,
                            ff_dim=1536,
                            att_weights_dropout=0.0,
                            conv_dropout=0.0,
                            ff_dropout=0.0,
                            mhsa_dropout=0.0,
                            conv_kernel_size=31,
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
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

                        student_config = StudentConfig(
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
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                        }
                        train_args_distill["config"]["preload_from_files"] = {
                            "teacher": {
                                "filename": PRETRAIN_CHECKPOINT_DISTILL_V1,
                                "init_for_train": True,
                                "ignore_missing": False,
                                "prefix": "teacher.",
                                "ignore_params_prefixes": ["teacher.feature_extraction"],
                            }
                        }
                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"
                        training_name = prefix_name + "/" + distill_module + f"_{layer_count}_{dim}_{distill_scale}_{T}"
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=250,
                            decoder_module=decoder_module,
                        )
                        generate_report(results=results, exp_name=training_name)
                        distill_report[training_name] = results
                        del results

                        teacher_config = TeacherConfig(
                            frontend_config=default_frontend_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=384,
                            num_layers=12,
                            num_heads=4,
                            ff_dim=1536,
                            att_weights_dropout=0.0,
                            conv_dropout=0.0,
                            ff_dropout=0.0,
                            mhsa_dropout=0.0,
                            conv_kernel_size=31,
                            distill_scale=distill_scale,
                            ctc_scale=1 - distill_scale,
                            t=T,
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

                        student_config = StudentConfig(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config_student,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=dim,
                            num_layers=layer_count,
                            num_heads=4,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.0,
                            conv_dropout=0.0,
                            ff_dropout=0.0,
                            mhsa_dropout=0.0,
                            conv_kernel_size=31,
                            final_dropout=0.0,
                            specauc_start_epoch=1,
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
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                        }
                        train_args_distill["config"]["preload_from_files"] = {
                            "teacher": {
                                "filename": PRETRAIN_CHECKPOINT_DISTILL_V1,
                                "init_for_train": True,
                                "ignore_missing": False,
                                "prefix": "teacher.",
                                "ignore_params_prefixes": ["teacher.feature_extraction"],
                            }
                        }
                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"
                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module
                            + f"_{layer_count}_{dim}_{distill_scale}_{T}_no_stud_drop"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=250,
                            decoder_module=decoder_module,
                        )
                        generate_report(results=results, exp_name=training_name)
                        no_drop_stud_report[
                            prefix_name + "/" + distill_module + f"_{layer_count}_{dim}_{distill_scale}_{T}"
                        ] = results
                        del results

                        teacher_config = TeacherConfig(
                            frontend_config=default_frontend_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=384,
                            num_layers=12,
                            num_heads=4,
                            ff_dim=1536,
                            att_weights_dropout=0.0,
                            conv_dropout=0.0,
                            ff_dropout=0.0,
                            mhsa_dropout=0.0,
                            conv_kernel_size=31,
                            distill_scale=distill_scale + 0.000001,
                            ctc_scale=1 - distill_scale + 0.000001,
                            t=T,
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

                        student_config = StudentConfig(
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
                        train_args_distill = {
                            "config": train_config_distill,
                            "network_module": distill_module,
                            "net_args": {
                                "model_config_dict": asdict(student_config),
                                "distill_config_dict": asdict(teacher_config),
                            },
                            "debug": False,
                        }
                        train_args_distill["config"]["preload_from_files"] = {
                            "teacher": {
                                "filename": PRETRAIN_CHECKPOINT_DISTILL_V2,
                                "init_for_train": True,
                                "ignore_missing": False,
                                "prefix": "teacher.",
                                "ignore_params_prefixes": ["teacher.feature_extraction"],
                            }
                        }
                        decoder_module = "ctc.decoder.flashlight_ctc_distill_v1"
                        training_name = (
                            prefix_name
                            + "/"
                            + distill_module
                            + f"_{layer_count}_{dim}_{distill_scale}_{T}_larger_teacher"
                        )
                        train_job = training(
                            training_name, train_data, train_args_distill, num_epochs=250, **default_returnn
                        )
                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args_distill,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            specific_epoch=250,
                            decoder_module=decoder_module,
                        )
                        generate_report(results=results, exp_name=training_name)
                        larger_distill_report[
                            prefix_name + "/" + distill_module + f"_{layer_count}_{dim}_{distill_scale}_{T}"
                        ] = results
                        del results

        tk.register_report(
            "reports/distill_report", partial(build_distill_report, distill_report), required=distill_report
        )
        tk.register_report(
            "reports/distill_no_drop_stud_report",
            partial(build_distill_report, no_drop_stud_report),
            required=no_drop_stud_report,
        )
        tk.register_report(
            "reports/distill_larger_report",
            partial(build_distill_report, larger_distill_report),
            required=larger_distill_report,
        )
    train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 180 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
    }
    frontend_config_sub2 = VGG4LayerActFrontendV1Config_mod(
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
        pool2_kernel_size=(1, 1),
        pool2_stride=(1, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )

    model_config_sub2 = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub2,
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
    )
    train_args = {
        "config": train_config,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config_sub2)},
        "debug": False,
    }
    training_name = prefix_name + "/" + network_module + "_384dim_sub2_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=default_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        specific_epoch=250,
    )
    generate_report(results=results, exp_name=training_name)
    del results
    train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 90 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 2,
    }
    frontend_config_sub2_768 = VGG4LayerActFrontendV1Config_mod(
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
        out_features=768,
        activation=None,
    )
    model_config_sub2_768 = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub2_768,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=768,
        num_layers=12,
        num_heads=4,
        ff_dim=768 * 4,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=1,
    )
    train_args = {
        "config": train_config,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config_sub2_768)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + "_768dim_sub4_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    prior_scales = [0.7, 0.9]
    res, _ = tune_and_evaluate_helper(
        training_name,
        asr_model,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
    )
    res, _ = tune_and_evaluate_helper(
        training_name + "/best4",
        asr_model_best4,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc"),
    )
    res, _ = tune_and_evaluate_helper(
        training_name + "/best",
        asr_model_best,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)  # 7.3
    generate_report(results=results, exp_name=training_name)

    # E-Branchformer
    branchformer_module = "ctc.conformer_1023.i6models_ebranchformer_v1"
    train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 180 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
    }
    train_args = {
        "config": train_config,
        "network_module": branchformer_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + branchformer_module + "_384dim_sub4_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    prior_scales = [0.7, 0.9]
    res, _ = tune_and_evaluate_helper(
        training_name,
        asr_model,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(4, "dev_loss_ctc"),
    )
    res, _ = tune_and_evaluate_helper(
        training_name + "/best4",
        asr_model_best4,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best",
        train_job,
        train_args,
        with_prior=True,
        datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc"),
    )
    res, _ = tune_and_evaluate_helper(
        training_name + "/best",
        asr_model_best,
        default_decoder_config,
        lm_scales=lm_scales,
        prior_scales=prior_scales,
        dev_dataset_tuples=dev_dataset_tuples,
    )
    results.update(res)
    generate_report(results=results, exp_name=training_name)  # TODO current best with 6.99
    del results
