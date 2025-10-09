from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List
from functools import partial

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....data.common import DatasetSettings, build_test_dataset
from ....data.bpe import build_bpe_training_datasets, get_text_lexicon
from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....lm import get_4gram_binary_lm
from ....pipeline import training, prepare_asr_model, search, ASRModel
# from ....report import generate_report
from ....report import tune_and_evalue_report, build_qat_report
# from ...experiments.ctc_phon.tune_eval import build_qat_report

# from ..ctc_phon.tune_eval import eval_model

from ....pytorch_networks.common import Mode
from ....pytorch_networks.trainers.train_handler import TrainMode



def bpe_lib_qat_comparisons():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2025/ctc_bpe/128/qat_comparison"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    bpe_size = 128
    train_data_bpe = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=bpe_size,
        settings=train_settings,
        use_postfix=True,  # RNN-T now, use postfix
    )
    label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe.vocab_size

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
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

    def tune_and_evaluate_helper(
            training_name, dev_dataset_tuples, test_dataset_tuples,
            asr_model, base_decoder_config, lm_scales, prior_scales, decoder_module,
            unhashed_decoder_config=None, debug=False, use_gpu=False, extra_forward_config=None
    ):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config: DecoderConfig = copy.deepcopy(base_decoder_config)
                search_config: CTCBeamSearchConfig = decoder_config.search_config
                search_config.lm_scale = lm_weight
                search_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.2f_prior%.2f" % (lm_weight, prior_scale)
                asr_model.prior_file = None
                search_jobs, wers = search(
                    search_name,
                    forward_config=extra_forward_config if extra_forward_config else {},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    unhashed_decoder_args={
                        "extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                    test_dataset_tuples=dev_dataset_tuples,
                    debug=debug,
                    use_gpu=use_gpu,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

                report_key = training_name[len(prefix_name)+1:].replace("/", "_")
                if report_key not in qat_report:
                    qat_report[report_key] = {decoder_module: []}

                qat_report[report_key][decoder_module].extend({
                    "wer": wers[search_name + "/dev-other"], 
                    "lm_scale": lm_weight, 
                    "prior_scale": prior_scale, 
                })

        # for key, tune_values in [("test-other", tune_values_other)]:
        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values,
                                                                        mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            if hasattr(decoder_config, "lm_scale"):
                decoder_config.lm_scale = pick_optimal_params_job.out_optimal_parameters[0]
            else:
                decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config=extra_forward_config if extra_forward_config else {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                unhashed_decoder_args={
                    "extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                test_dataset_tuples={key: test_dataset_tuples[key]},
                use_gpu=use_gpu,
                **default_returnn
            )
            report_values[key] = wers[training_name + "/" + key]

        tune_and_evalue_report(
            training_name=training_name,
            tune_parameters=tune_parameters,
            tuning_names=["LM", "Prior"],
            tune_values_clean=tune_values_clean,
            tune_values_other=tune_values_other,
            report_values=report_values
        )

    from ....pytorch_networks.ctc.qat_2509.full_qat_v1_cfg import (
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
        center=True,
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    frontend_config_sub6 = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )

    # default configs for continued training
    train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 480))
                          + list(np.linspace(5e-4, 5e-5, 480))
                          + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 240 * 16000,  # GPU MEM still very moderate, but larger batch did not help
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    from ....pytorch_networks.search.decoder_module import DecoderConfig, ExtraConfig
    from ....pytorch_networks.ctc.search import CTCBeamSearchConfig

    chunk_size = 1.67  # seconds corresponding to 28 subs-frames
    fac_size = 8  # number of future subs-frames
    carry_over_size = 1  # number of past chunks each chunk can (immediately) depend on

    num_epochs = 1000

    qat_report = {}

    ####################################################################################################
    # No-QAT Baseline
    from ....pytorch_networks.ctc.qat_2509.baseline_no_qat_v1_streamable_cfg import ModelTrainNoQuantConfigV1
    network_module = "ctc.qat_2509.baseline_no_qat_v1_streamable"
    model_config = ModelTrainNoQuantConfigV1(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=1024,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=11,

        # streaming params
        chunk_size=chunk_size * 16000,  # samples corresponding to 28 sub-frames
        lookahead_size=fac_size,
        carry_over_size=carry_over_size,
        dual_mode=None,
        streaming_scale=None,
        train_mode=str(TrainMode.STREAMING),
    )

    train_args = {
        "config": train_config,
        "network_module": network_module,
        "include_native_ops": True,
        "debug": True,
        "net_args": {"model_config_dict": asdict(model_config)}
    }
    training_name = prefix_name + "/" + network_module + f"_512_1024_streaming"
    train_job = training(training_name, train_data_bpe, train_args,
                         num_epochs=num_epochs, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
        get_specific_checkpoint=num_epochs,
    )
    search_config = CTCBeamSearchConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
        beam_size_token=16,
        beam_threshold=14,  # Untuned,
        lm_package=arpa_4gram_lm,
        prior_file=asr_model.prior_file,
    )
    decoder_config_offline = DecoderConfig(
        beam_size=1024,
        returnn_vocab=label_datastream_bpe.vocab,
        search_config=search_config,

        mode=Mode.OFFLINE.name,
        test_version=0.0,
    )
    decoder_config_streaming = DecoderConfig(
        beam_size=1024,
        returnn_vocab=label_datastream_bpe.vocab,

        search_config=search_config,

        mode=Mode.STREAMING.name,
        chunk_size=int(model_config.chunk_size),
        lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
        carry_over_size=model_config.carry_over_size,
        test_version=0.0,
    )

    tune_and_evaluate_helper(
        training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        asr_model=asr_model,
        base_decoder_config=decoder_config_offline,
        lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
        prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        decoder_module="search.decoder_module",
        debug=True,
        use_gpu=False,
    )
    tune_and_evaluate_helper(
        training_name + "/streaming/4gram_lm/search_bs%i" % decoder_config_streaming.beam_size,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        asr_model=asr_model,
        base_decoder_config=decoder_config_streaming,
        lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
        prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        decoder_module="search.decoder_module",
        debug=True,
        use_gpu=False,
    )

    # offline
    model_config = ModelTrainNoQuantConfigV1(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        specaug_config=specaug_config,
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

        # streaming params
        chunk_size=None,  # samples corresponding to 28 sub-frames
        lookahead_size=None,
        carry_over_size=None,
        dual_mode=None,
        streaming_scale=None,
        train_mode=str(TrainMode.OFFLINE),
    )

    train_args = {
        "config": train_config,
        "network_module": network_module,
        "include_native_ops": True,
        "debug": True,
        "net_args": {"model_config_dict": asdict(model_config)}
    }
    training_name = prefix_name + "/" + network_module + f"_512_2048_offline"
    train_job = training(training_name, train_data_bpe, train_args,
                         num_epochs=num_epochs, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
        get_specific_checkpoint=num_epochs,
    )
    search_config = CTCBeamSearchConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
        beam_size_token=16,
        beam_threshold=14,  # Untuned,
        lm_package=arpa_4gram_lm,
        prior_file=asr_model.prior_file,
    )
    decoder_config_offline = DecoderConfig(
        beam_size=1024,
        returnn_vocab=label_datastream_bpe.vocab,
        search_config=search_config,

        mode=Mode.OFFLINE.name,
        test_version=0.0,
    )

    tune_and_evaluate_helper(
        training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        asr_model=asr_model,
        base_decoder_config=decoder_config_offline,
        lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
        prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        decoder_module="search.decoder_module",
        debug=True,
        use_gpu=False,
    )

    ####################################################################################################
    # QAT Baseline
    from ....pytorch_networks.ctc.qat_2509.baseline_qat_v4_streamable_cfg import QuantModelTrainConfigV4

    network_module = "ctc.qat_2509.baseline_qat_v4_streamable"
    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=1024,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=11,
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

        # streaming params
        chunk_size=chunk_size * 16000,  # samples corresponding to 28 sub-frames
        lookahead_size=fac_size,
        carry_over_size=carry_over_size,
        dual_mode=None,
        streaming_scale=None,
        train_mode=str(TrainMode.STREAMING),
    )

    train_config_no_amp = copy.deepcopy(train_config)
    train_config_no_amp.pop("torch_amp_options")
    train_args = {
        "config": train_config_no_amp,
        "network_module": network_module,
        "include_native_ops": True,
        "debug": True,
        "net_args": {"model_config_dict": asdict(model_config)}
    }
    training_name = prefix_name + "/" + network_module + f"_8_8_512_1024_streaming"
    train_job = training(training_name, train_data_bpe, train_args,
                         num_epochs=num_epochs, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
        get_specific_checkpoint=num_epochs,
    )
    search_config = CTCBeamSearchConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
        beam_size_token=16,
        beam_threshold=14,  # Untuned,
        lm_package=arpa_4gram_lm,
        prior_file=asr_model.prior_file,
    )
    decoder_config_offline = DecoderConfig(
        beam_size=1024,
        returnn_vocab=label_datastream_bpe.vocab,
        search_config=search_config,

        mode=Mode.OFFLINE.name,
        test_version=0.0,
    )
    decoder_config_streaming = DecoderConfig(
        beam_size=1024,
        returnn_vocab=label_datastream_bpe.vocab,

        search_config=search_config,

        mode=Mode.STREAMING.name,
        chunk_size=int(model_config.chunk_size),
        lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
        carry_over_size=model_config.carry_over_size,
        test_version=0.0,
    )

    tune_and_evaluate_helper(
        training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        asr_model=asr_model,
        base_decoder_config=decoder_config_offline,
        lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
        prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        decoder_module="search.decoder_module",
        debug=True,
        use_gpu=False,
    )
    tune_and_evaluate_helper(
        training_name + "/streaming/4gram_lm/search_bs%i" % decoder_config_streaming.beam_size,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        asr_model=asr_model,
        base_decoder_config=decoder_config_streaming,
        lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
        prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        decoder_module="search.decoder_module",
        debug=True,
        use_gpu=False,
    )

    # offline
    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        specaug_config=specaug_config,
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

        # streaming params
        chunk_size=None,  # samples corresponding to 28 sub-frames
        lookahead_size=None,
        carry_over_size=None,
        dual_mode=None,
        streaming_scale=None,
        train_mode=str(TrainMode.OFFLINE),
    )

    train_config_no_amp = copy.deepcopy(train_config)
    train_config_no_amp.pop("torch_amp_options")
    train_args = {
        "config": train_config_no_amp,
        "network_module": network_module,
        "include_native_ops": True,
        "debug": True,
        "net_args": {"model_config_dict": asdict(model_config)}
    }
    training_name = prefix_name + "/" + network_module + f"_8_8_512_2048_offline"
    train_job = training(training_name, train_data_bpe, train_args,
                         num_epochs=num_epochs, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
        get_specific_checkpoint=num_epochs,
    )
    search_config = CTCBeamSearchConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
        beam_size_token=16,
        beam_threshold=14,  # Untuned,
        lm_package=arpa_4gram_lm,
        prior_file=asr_model.prior_file,
    )
    decoder_config_offline = DecoderConfig(
        beam_size=1024,
        returnn_vocab=label_datastream_bpe.vocab,
        search_config=search_config,

        mode=Mode.OFFLINE.name,
        test_version=0.0,
    )
    tune_and_evaluate_helper(
        training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
        dev_dataset_tuples=dev_dataset_tuples,
        test_dataset_tuples=test_dataset_tuples,
        asr_model=asr_model,
        base_decoder_config=decoder_config_offline,
        lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
        prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        decoder_module="search.decoder_module",
        debug=True,
        use_gpu=False,
    )

    from ....pytorch_networks.ctc.qat_2509.full_qat_v1_streamable_cfg import QuantModelTrainConfigV4

    for ff_dim in [512, 1024]:
        #########################################################################################
        # Full Quant Baseline
        network_module = "ctc.qat_2509.full_qat_v1_streamable"

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=8,
            ff_dim=ff_dim,
            att_weights_dropout=0.1,
            conv_dropout=0.1,
            ff_dropout=0.1,
            mhsa_dropout=0.1,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=11,
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

            # streaming params
            chunk_size=chunk_size * 16000,  # samples corresponding to 28 sub-frames
            lookahead_size=fac_size,
            carry_over_size=carry_over_size,
            dual_mode=None,
            streaming_scale=None,
            train_mode=str(TrainMode.STREAMING),
        )

        train_args = {
            "config": train_config_no_amp,
            "network_module": network_module,
            "include_native_ops": True,
            "debug": True,
            "net_args": {"model_config_dict": asdict(model_config)}
        }
        training_name = prefix_name + "/" + network_module + f"_8_8_512_{ff_dim}_streaming"
        train_job = training(training_name, train_data_bpe, train_args,
                            num_epochs=num_epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=num_epochs,
        )
        search_config = CTCBeamSearchConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
            beam_size_token=16,
            beam_threshold=14,  # Untuned,
            lm_package=arpa_4gram_lm,
            prior_file=asr_model.prior_file,
        )
        decoder_config_offline = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,
            search_config=search_config,

            mode=Mode.OFFLINE.name,
            test_version=0.0,
        )
        decoder_config_streaming = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,

            search_config=search_config,

            mode=Mode.STREAMING.name,
            chunk_size=int(model_config.chunk_size),
            lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
            carry_over_size=model_config.carry_over_size,
            test_version=0.0,
        )

        tune_and_evaluate_helper(
            training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_offline,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
        )
        tune_and_evaluate_helper(
            training_name + "/streaming/4gram_lm/search_bs%i" % decoder_config_streaming.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_streaming,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
        )

        #########################################################################################
        # FF 512 and [512, 1024] with mean abs
        network_module = "ctc.qat_2509.full_qat_v1_mean_abs_norm_streamable"

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=8,
            ff_dim=ff_dim,
            att_weights_dropout=0.1,
            conv_dropout=0.1,
            ff_dropout=0.1,
            mhsa_dropout=0.1,
            conv_kernel_size=31,
            final_dropout=0.1,
            specauc_start_epoch=11,
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

            # streaming params
            chunk_size=chunk_size * 16000,  # samples corresponding to 28 sub-frames
            lookahead_size=fac_size,
            carry_over_size=carry_over_size,
            dual_mode=None,
            streaming_scale=None,
            train_mode=str(TrainMode.STREAMING),
        )

        train_args = {
            "config": train_config_no_amp,
            "network_module": network_module,
            "include_native_ops": True,
            "debug": True,
            "net_args": {"model_config_dict": asdict(model_config)}
        }
        training_name = prefix_name + "/" + network_module + f"_8_8_512_{ff_dim}_streaming"
        train_job = training(training_name, train_data_bpe, train_args,
                            num_epochs=num_epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=num_epochs,
        )
        search_config = CTCBeamSearchConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
            beam_size_token=16,
            beam_threshold=14,  # Untuned,
            lm_package=arpa_4gram_lm,
            prior_file=asr_model.prior_file,
        )
        decoder_config_offline = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,
            search_config=search_config,

            mode=Mode.OFFLINE.name,
            test_version=0.0,
        )
        decoder_config_streaming = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,

            search_config=search_config,

            mode=Mode.STREAMING.name,
            chunk_size=int(model_config.chunk_size),
            lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
            carry_over_size=model_config.carry_over_size,
            test_version=0.0,
        )

        tune_and_evaluate_helper(
            training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_offline,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
        )
        tune_and_evaluate_helper(
            training_name + "/streaming/4gram_lm/search_bs%i" % decoder_config_streaming.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_streaming,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
        )

        #########################################################################################
        # FF 512 and [512, 1024] with mean abs symmetric
        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=8,
            ff_dim=ff_dim,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,

            # streaming params
            chunk_size=chunk_size * 16000,  # samples corresponding to 28 sub-frames
            lookahead_size=fac_size,
            carry_over_size=carry_over_size,
            dual_mode=None,
            streaming_scale=None,
            train_mode=str(TrainMode.STREAMING),
        )

        train_args = {
            "config": train_config_no_amp,
            "network_module": network_module,
            "include_native_ops": True,
            "debug": True,
            "net_args": {"model_config_dict": asdict(model_config)}
        }
        training_name = prefix_name + "/" + network_module + f"_8_8_512_{ff_dim}_sym_streaming"
        train_job = training(training_name, train_data_bpe, train_args,
                            num_epochs=num_epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=num_epochs,
        )
        search_config = CTCBeamSearchConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
            beam_size_token=16,
            beam_threshold=14,  # Untuned,
            lm_package=arpa_4gram_lm,
            prior_file=asr_model.prior_file,
        )
        decoder_config_offline = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,
            search_config=search_config,

            mode=Mode.OFFLINE.name,
            test_version=0.0,
        )
        decoder_config_streaming = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,

            search_config=search_config,

            mode=Mode.STREAMING.name,
            chunk_size=int(model_config.chunk_size),
            lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
            carry_over_size=model_config.carry_over_size,
            test_version=0.0,
        )

        tune_and_evaluate_helper(
            training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_offline,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
        )
        tune_and_evaluate_helper(
            training_name + "/streaming/4gram_lm/search_bs%i" % decoder_config_streaming.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_streaming,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
        )

        #########################################################################################
        # FF [512, 1024] with mean abs symmetric and ReLU
        network_module = "ctc.qat_2509.full_qat_v1_relu_mean_abs_streamable"

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=8,
            ff_dim=ff_dim,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,

            # streaming params
            chunk_size=chunk_size * 16000,  # samples corresponding to 28 sub-frames
            lookahead_size=fac_size,
            carry_over_size=carry_over_size,
            dual_mode=None,
            streaming_scale=None,
            train_mode=str(TrainMode.STREAMING),
        )

        train_args = {
            "config": train_config_no_amp,
            "network_module": network_module,
            "include_native_ops": True,
            "debug": True,
            "net_args": {"model_config_dict": asdict(model_config)}
        }
        training_name = prefix_name + "/" + network_module + f"_8_8_512_{ff_dim}_sym_streaming"
        train_job = training(training_name, train_data_bpe, train_args,
                             num_epochs=num_epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=num_epochs,
        )
        search_config = CTCBeamSearchConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
            beam_size_token=16,
            beam_threshold=14,  # Untuned,
            lm_package=arpa_4gram_lm,
            prior_file=asr_model.prior_file,
        )
        decoder_config_offline = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,
            search_config=search_config,

            mode=Mode.OFFLINE.name,
            test_version=0.0,
        )
        decoder_config_streaming = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,

            search_config=search_config,

            mode=Mode.STREAMING.name,
            chunk_size=int(model_config.chunk_size),
            lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
            carry_over_size=model_config.carry_over_size,
            test_version=0.0,
        )

        tune_and_evaluate_helper(
            training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_offline,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
            )
        tune_and_evaluate_helper(
            training_name + "/streaming/4gram_lm/search_bs%i" % decoder_config_streaming.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_streaming,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
            )

        #########################################################################################
        # FF [512, 1024] with mean abs symmetric and ReLU and shared observers
        network_module = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs_v2_streamable"

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=8,
            ff_dim=ff_dim,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,

            # streaming params
            chunk_size=chunk_size * 16000,  # samples corresponding to 28 sub-frames
            lookahead_size=fac_size,
            carry_over_size=carry_over_size,
            dual_mode=None,
            streaming_scale=None,
            train_mode=str(TrainMode.STREAMING),
        )

        train_args = {
            "config": train_config_no_amp,
            "network_module": network_module,
            "include_native_ops": True,
            "debug": True,
            "net_args": {"model_config_dict": asdict(model_config)}
        }
        training_name = prefix_name + "/" + network_module + f"_8_8_512_{ff_dim}_sym_streaming"
        train_job = training(training_name, train_data_bpe, train_args,
                             num_epochs=num_epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=num_epochs,
        )
        search_config = CTCBeamSearchConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
            beam_size_token=16,
            beam_threshold=14,  # Untuned,
            lm_package=arpa_4gram_lm,
            prior_file=asr_model.prior_file,
        )
        decoder_config_offline = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,
            search_config=search_config,

            mode=Mode.OFFLINE.name,
            test_version=0.0,
        )
        decoder_config_streaming = DecoderConfig(
            beam_size=1024,
            returnn_vocab=label_datastream_bpe.vocab,

            search_config=search_config,

            mode=Mode.STREAMING.name,
            chunk_size=int(model_config.chunk_size),
            lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
            carry_over_size=model_config.carry_over_size,
            test_version=0.0,
        )

        tune_and_evaluate_helper(
            training_name + "/offline/4gram_lm/search_bs%i" % decoder_config_offline.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_offline,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
            )
        tune_and_evaluate_helper(
            training_name + "/streaming/4gram_lm/search_bs%i" % decoder_config_streaming.beam_size,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_streaming,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
            )

    # report string
    tk.register_report("reports/qat/streamable_qat_comparison", partial(build_qat_report, qat_report), required=qat_report)
