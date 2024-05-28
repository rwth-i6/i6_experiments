from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from .tune_eval import QuantArgs
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model
from ...report import generate_report
from .tune_eval import tune_and_evaluate_helper


def eow_phon_ted_1023_base(quant=False):
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
        out_features=384,
        activation=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
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
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
    )
    res, _ = tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config,
        lm_scales=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4], prior_scales=[0.0, 0.3, 0.5, 0.7, 1.0],
        dev_dataset_tuples=dev_dataset_tuples
    )
    results.update(res)
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
    lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    prior_scales = [0.7, 0.9]
    res, _ = tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=lm_scales,
        prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config,
                                   lm_scales=lm_scales, prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples)
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best", asr_model_best, default_decoder_config,
                                   lm_scales=lm_scales, prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples)
    results.update(res)
    generate_report(results=results, exp_name=training_name)  # TODO current best with 7.083
    del results
    if quant is True:
        from ...pytorch_networks.ctc.conformer_1023.quant.baseline_quant_v1_cfg import QuantModelConfigV1
        num_iterations = 100
        # what if we give more information to the activation instead?
        for activation_bit in [8, 7, 6, 5, 4, 3, 2, 1]:
            for weight_bit in [8, 7, 6, 5, 4, 3, 2, 1]:
                results = {}
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
                    linear_quant_output=False,
                )
                quant_args = QuantArgs(
                    sample_ls=[10] if weight_bit < 8 or activation_bit < 8 else [10, 100, 1000, 10000],
                    quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
                    decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
                    num_iterations=num_iterations,
                    datasets=train_data,
                    network_module="ctc.conformer_1023.quant.baseline_quant_v1",
                )
                quant_str = f"_weight_{weight_bit}_act_{activation_bit}"
                asr_model = prepare_asr_model(
                    training_name+quant_str,
                    train_job,
                    train_args,
                    with_prior=True,
                    datasets=train_data,
                    get_specific_checkpoint=250,
                )
                res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
                    training_name, asr_model, default_decoder_config, lm_scales=[2.8],
                    prior_scales=[0.7], quant_args=quant_args, quant_str=quant_str,
                    dev_dataset_tuples=dev_dataset_tuples,
                )
                results.update(res)
                generate_report(results=results, exp_name=training_name + quant_str)
                del results
        num_iterations = 250
        for filter in [
            ({"unique_tags": 0.0}, "unique"),
            ({"single_tag": 0.0}, "single"),
            ({"max_dur": 1.0}, "max_dur_1"),
            ({"min_dur": 15.0}, "min_dur_15")
            ]:
            for activation_bit in [8]:
                for weight_bit in [8]:
                    results = {}
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
                        linear_quant_output=False,
                    )
                    quant_args = QuantArgs(
                        sample_ls=[1], #§, 10, 25, 5],
                        quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
                        decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
                        num_iterations=num_iterations,
                        datasets=train_data,
                        network_module="ctc.conformer_1023.quant.baseline_quant_v1",
                        filter_args=filter[0],
                    )
                    quant_str = f"_weight_{weight_bit}_act_{activation_bit}_{filter[1]}"
                    asr_model = prepare_asr_model(
                        training_name+quant_str,
                        train_job,
                        train_args,
                        with_prior=True,
                        datasets=train_data,
                        get_specific_checkpoint=250,
                    )
                    res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
                        training_name, asr_model, default_decoder_config, lm_scales=[2.8],
                        prior_scales=[0.7], quant_args=quant_args, quant_str=quant_str, dev_dataset_tuples=dev_dataset_tuples
                    )
                    results.update(res)
                    generate_report(results=results, exp_name=training_name + quant_str)
                    del results

        num_iterations = 100
        for activation_bit in [8]:
            for weight_bit in [8, 7, 6, 5, 4, 3, 2, 1]:
                results = {}
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
                    sample_ls=[10] if weight_bit < 8 or activation_bit < 8 else [10, 100, 1000, 10000],
                    quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
                    decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
                    num_iterations=num_iterations,
                    datasets=train_data,
                    network_module="ctc.conformer_1023.quant.baseline_quant_v1",
                )
                quant_str = f"_weight_{weight_bit}_act_{activation_bit}_qlin"
                asr_model = prepare_asr_model(
                    training_name+quant_str, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=250
                )
                res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
                    training_name, asr_model, default_decoder_config, lm_scales=[2.8],
                    prior_scales=[0.7], quant_args=quant_args, quant_str=quant_str, dev_dataset_tuples=dev_dataset_tuples
                )
                results.update(res)
                generate_report(results=results, exp_name=training_name+quant_str)
                del results

        for activation_bit in [8]:
            for weight_bit in [8]:
                results = {}
                model_config_quant_v1 = QuantModelConfigV1(
                    weight_quant_dtype="qint8",
                    weight_quant_method="per_tensor",
                    activation_quant_dtype="qint8",
                    activation_quant_method="per_tensor",
                    dot_quant_dtype="qint8",
                    dot_quant_method="per_tensor",
                    Av_quant_dtype="qint8",
                    Av_quant_method="per_tensor",
                    moving_average=None,
                    weight_bit_prec=weight_bit,
                    activation_bit_prec=activation_bit,
                    linear_quant_output=False,
                )
                quant_args = QuantArgs(
                    sample_ls=[10] if weight_bit < 8 or activation_bit < 8 else [10, 100, 1000, 10000],
                    quant_config_dict={"quant_config_dict": asdict(model_config_quant_v1)},
                    decoder="ctc.decoder.flashlight_quant_stat_phoneme_ctc",
                    num_iterations=num_iterations,
                    datasets=train_data,
                    network_module="ctc.conformer_1023.quant.baseline_quant_v1",
                )
                quant_str = f"_weight_{weight_bit}_act_{activation_bit}_no_avg"
                asr_model = prepare_asr_model(
                    training_name+quant_str,
                    train_job,
                    train_args,
                    with_prior=True,
                    datasets=train_data,
                    get_specific_checkpoint=250,
                )
                res, _ = tune_and_evaluate_helper(  # only take best for now, since otherwise too many searches
                    training_name, asr_model, default_decoder_config, lm_scales=[2.8],
                    prior_scales=[0.7], quant_args=quant_args, quant_str=quant_str, dev_dataset_tuples=dev_dataset_tuples,
                )
                results.update(res)
                generate_report(results=results, exp_name=training_name + quant_str)
                del results

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
        training_name, asr_model, default_decoder_config, lm_scales=lm_scales,
        prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config,
                                      lm_scales=lm_scales, prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples)
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best", asr_model_best, default_decoder_config,
                                      lm_scales=lm_scales, prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples)
    results.update(res)
    generate_report(results=results, exp_name=training_name)  # TODO current best with 6.99
    del results
    unimod_module = "ctc.conformer_1023.conformer_v1_uni_aggr_v1"
    from ...pytorch_networks.ctc.conformer_1023.conformer_v1_uni_aggr_cfg_v1 import ModelConfig as UniAggrConfig
    uni_aggr_model_config = UniAggrConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
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
        aggr_layer=9,
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
    # Unimodal Aggregation
    train_args = {
        "config": train_config,
        "network_module": unimod_module,
        "net_args": {"model_config_dict": asdict(uni_aggr_model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + unimod_module + "_384dim_sub4_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=111
    )
    lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
    prior_scales = [0.7, 0.9]
    res, _ = tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=lm_scales,
        prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples
    )
    results.update(res)
    asr_model_best4 = prepare_asr_model(
        training_name + "/best4", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config,
                                      lm_scales=lm_scales, prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples)
    results.update(res)
    asr_model_best = prepare_asr_model(
        training_name + "/best", train_job, train_args, with_prior=True, datasets=train_data,
        get_best_averaged_checkpoint=(1, "dev_loss_ctc")
    )
    res, _ = tune_and_evaluate_helper(training_name + "/best", asr_model_best, default_decoder_config,
                                      lm_scales=lm_scales, prior_scales=prior_scales, dev_dataset_tuples=dev_dataset_tuples)
    results.update(res)
    generate_report(results=results, exp_name=training_name)
    del results
