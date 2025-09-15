import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Dict


from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training
from ...report import generate_report
from .tune_eval import eval_model, build_qat_report, RTFArgs
from functools import partial

def eow_phon_ted_0625_full_qat():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/full_qat"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

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

    from ...pytorch_networks.ctc.decoder.flashlight_qat_phoneme_ctc import DecoderConfig

    as_training_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
        turn_off_quant="leave_as_is",
    )

    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
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
    qat_report = {}

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
        "batch_size": 180 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
    }
    rtf_args = RTFArgs(
        beam_sizes=[256, 512, 1024],
        # beam_size_tokens=[4, 6, 8, 10, 12, 20, 30],  # makes it much faster
        beam_size_tokens=[8, 12],
        beam_thresholds=[10, 14, 18],
        decoder_module="ctc.decoder.flashlight_ctc_v5_rescale_measure",
    )
    prior_scales = [0.3, 0.5, 0.7]
    lm_scales = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]

    ####################################################################################################
    # QAT Baseline
    network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
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

    train_args = {
        "config": train_config,
        "network_module": network_module_v4,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v4 + f"_8_8"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=prior_scales,
        lm_scales=lm_scales,
        run_rtf=True,
        rtf_args=rtf_args,
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    #########################################################################################
    # Full Quant Baseline
    network_module_v1 = "ctc.qat_0711.full_qat_v1"
    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
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
    train_args = {
        "config": train_config,
        "network_module": network_module_v1,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v1 + f"_{8}_{8}"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=prior_scales,
        lm_scales=lm_scales,
        run_rtf=True,
        rtf_args=rtf_args,
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    for ff_dim in [512, 1024]:
        ########################################################################
        # FF 512 and ff_dim
        frontend_config_sub4_512 = VGG4LayerActFrontendV1Config_mod(
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

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
        train_args = {
            "config": train_config,
            "network_module": network_module_v1,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1 + f"_8_8_512_{ff_dim}"
        train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            run_best_4=False,
            run_rtf=True,
            rtf_args=rtf_args,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        ########################################################################
        # FF 512 and ff_dim with mean abs
        network_module_v1_mean = "ctc.qat_0711.full_qat_v1_mean_abs_norm"
        frontend_config_sub4_512 = VGG4LayerActFrontendV1Config_mod(
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

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean + f"_8_8_512_{ff_dim}"
        train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            run_best_4=False,
            run_rtf=True,
            rtf_args=rtf_args,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        ########################################################################
        # FF 512 and ff_dim with sym and means abs
        frontend_config_sub4_512 = VGG4LayerActFrontendV1Config_mod(
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

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            run_best_4=False,
            run_rtf=True,
            rtf_args=rtf_args,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        ########################################################################
        # FF 512 and ff_dim with sym and means abs and ReLu
        network_module_v1_mean_relu = "ctc.qat_0711.full_qat_v1_relu_mean_abs"
        frontend_config_sub4_512 = VGG4LayerActFrontendV1Config_mod(
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

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            run_rtf=True,
            run_best_4=False,
            rtf_args=rtf_args,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        ########################################################################
        # sym and means abs and ReLu and shared observers (faulty old)
        network_module_v1_mean_relu_shared_obs = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs"
        frontend_config_sub4_512 = VGG4LayerActFrontendV1Config_mod(
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

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu_shared_obs,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu_shared_obs + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            run_best_4=False,
            run_rtf=True,
            rtf_args=rtf_args,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        ########################################################################
        # sym and means abs and ReLu and shared observers
        network_module_v1_mean_relu_shared_obs_v2 = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs_v2"
        frontend_config_sub4_512 = VGG4LayerActFrontendV1Config_mod(
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

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu_shared_obs_v2,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu_shared_obs_v2 + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            run_best_4=False,
            run_rtf=True,
            rtf_args=rtf_args,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        ########################################################################
        # FF 512 and ff_dim with sym and means abs and ReLu 16 L
        frontend_config_sub4_512 = VGG4LayerActFrontendV1Config_mod(
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

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=16,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym_16l"
        train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            run_best_4=False,
            run_rtf=True,
            rtf_args=rtf_args,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu 20 L
        frontend_config_sub4_512 = VGG4LayerActFrontendV1Config_mod(
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

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=20,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym_20l"
        train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=prior_scales,
            lm_scales=lm_scales,
            run_best_4=False,
            run_rtf=True,
            rtf_args=rtf_args,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results


    tk.register_report("reports/qat_report_phon_comparison", partial(build_qat_report, qat_report), required=qat_report)
