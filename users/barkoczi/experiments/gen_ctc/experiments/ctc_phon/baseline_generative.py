import copy
from dataclasses import asdict
from typing import cast

import numpy as np
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import MINI_RETURNN_ROOT, RETURNN_EXE
from ...lm import get_4gram_binary_lm
from ...pipeline import prepare_asr_model, search, training
from ...report import tune_and_evalue_report
from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig
from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_cfg import (
    LogMelFeatureExtractionV1Config,
    ModelConfig,
    SpecaugConfig,
    VGG4LayerActFrontendV1ConfigMod,
)


def eow_phon_ls960_1023_generative_nce():
    prefix_name = "users/barkoczi/experiments/gen_ctc/ls960_ctc_eow_phon_generative_nce"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_dataset_tuples = {
        testset: build_test_dataset(dataset_key=testset, settings=train_settings)
        for testset in ["dev-clean", "dev-other"]
    }
    test_dataset_tuples = {
        testset: build_test_dataset(dataset_key=testset, settings=train_settings)
        for testset in ["test-clean", "test-other"]
    }

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }
    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

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
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1ConfigMod(
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

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
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
        specauc_start_epoch=1,
        sampling_type="batch",
        sampling_ratio=0.1,
        share_samples=False,
        ratio_corrector=1.0,
    )

    train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 480))
        + list(np.linspace(5e-4, 5e-5, 480))
        + list(np.linspace(5e-5, 1e-7, 40)),
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
        "gradient_clip_norm": 1.0,
    }

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first"
    train_args = {
        "config": train_config,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "use_speed_perturbation": True,
        "debug": False,
    }

    run_name = ".512dim_sub4_24gbgpu_100eps_lp_fullspec_gradnorm_smallbatch_sp_gennce"
    training_name = prefix_name + "/" + network_module + run_name
    train_job = training(training_name, train_data, train_args, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24

    asr_model = prepare_asr_model(
        training_name,
        train_job,
        train_args,
        with_prior=False,
        datasets=None,
        get_specific_checkpoint=1000,
    )

    tune_parameters = []
    tune_values_clean = []
    tune_values_other = []
    report_values = {}
    for lm_scale in [0.6, 0.8, 1.0, 1.2, 1.6, 2.0]:
        decoder_config = DecoderConfig(
            lexicon=get_text_lexicon(),
            returnn_vocab=label_datastream.vocab,
            beam_size=1024,
            beam_size_token=12,
            arpa_lm=arpa_4gram_lm,
            beam_threshold=14,
            lm_scale=lm_scale,
            prior_scale=0.0,
            prior_file=None,
        )
        search_name = training_name + "/search_lm%.1f" % lm_scale
        _search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=copy.deepcopy(asr_model),
            decoder_module="ctc.decoder.flashlight_ctc_v2",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples=dev_dataset_tuples,
            **default_returnn,
        )
        tune_parameters.append((lm_scale,))
        tune_values_clean.append(wers[search_name + "/dev-clean"])
        tune_values_other.append(wers[search_name + "/dev-other"])

    for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
        pick_optimal_params_job = GetOptimalParametersAsVariableJob(
            parameters=tune_parameters,
            values=tune_values,
            mode="minimize",
        )
        pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
        decoder_config = DecoderConfig(
            lexicon=get_text_lexicon(),
            returnn_vocab=label_datastream.vocab,
            beam_size=1024,
            beam_size_token=12,
            arpa_lm=arpa_4gram_lm,
            beam_threshold=14,
            lm_scale=pick_optimal_params_job.out_optimal_parameters[0],
            prior_scale=0.0,
            prior_file=None,
        )
        test_search_name = training_name + f"/best_{key}"
        _search_jobs, wers = search(
            test_search_name,
            forward_config={},
            asr_model=copy.deepcopy(asr_model),
            decoder_module="ctc.decoder.flashlight_ctc_v2",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={key: test_dataset_tuples[key]},
            **default_returnn,
        )
        report_values[key] = wers[test_search_name + "/" + key]

    tune_and_evalue_report(
        training_name=training_name,
        tune_parameters=tune_parameters,
        tuning_names=["LM"],
        tune_values_clean=tune_values_clean,
        tune_values_other=tune_values_other,
        report_values=report_values,
    )


py = eow_phon_ls960_1023_generative_nce