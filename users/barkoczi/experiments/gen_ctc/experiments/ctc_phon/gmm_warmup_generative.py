import copy
from dataclasses import asdict
from typing import cast

import numpy as np
from sisyphus import tk

from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....hsmm.data.common import build_oggzip_dataset_with_optional_hdf
from ....hsmm.data.hdf_seq_whitelist import ExtractSeqListFromHDFJob
from ...data.common import DatasetSettings, TrainingDatasets, build_test_dataset, get_audio_raw_datastream
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import MINI_RETURNN_ROOT, RETURNN_EXE
from ...lm import get_4gram_binary_lm
from ...pipeline import prepare_asr_model, search, training
from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_cfg import (
    LogMelFeatureExtractionV1Config,
    ModelConfig,
    SpecaugConfig,
    VGG4LayerActFrontendV1ConfigMod,
)
from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig
from ...report import tune_and_evalue_report


GMM_ALIGNMENT_HDFS = [
    tk.Path(f"/u/zyang/setups/mini/output/lbs_mono_phone_eow_lexicon/alignment_{i}.hdf")
    for i in range(1, 201)
]
GMM_ALIGNMENT_LABEL_MAP = tk.Path(
    "/u/zyang/setups/mini/output/lbs_mono_phone_eow/phoneme_to_idx.txt"
)


def _build_gmm_alignment_training_datasets(
    *,
    prefix_name: str,
    settings: DatasetSettings,
    label_datastream: LabelDatastream,
) -> TrainingDatasets:
    alignment_seq_whitelist = ExtractSeqListFromHDFJob(GMM_ALIGNMENT_HDFS).out_seq_list
    split_job = ShuffleAndSplitSegmentsJob(
        segment_file=alignment_seq_whitelist,
        split={"train": 0.99, "cv": 0.01},
        shuffle=True,
    )
    split_job.add_alias(prefix_name + "/train-other-960/gmm_train_cv_99_1_split")

    alignment_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=label_datastream.vocab,
        vocab_size=label_datastream.vocab_size,
        unk_label=label_datastream.unk_label,
    )
    train_ogg = get_ogg_zip_dict(
        prefix_name,
        returnn_root=MINI_RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
    )["train-other-960"]
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    train_dataset, datastreams = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        hdf_file=GMM_ALIGNMENT_HDFS,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="data",
        partition_epoch=settings.train_partition_epoch,
        segment_file=split_job.out_segments["train"],
        seq_ordering=settings.train_seq_ordering,
        additional_options=settings.train_additional_options,
    )
    eval_dataset, _ = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        hdf_file=GMM_ALIGNMENT_HDFS,
        hdf_datastream=alignment_datastream,
        hdf_stream_name="alignments",
        hdf_data_key="data",
        partition_epoch=1,
        segment_file=split_job.out_segments["cv"],
        seq_ordering="sorted_reverse",
    )
    return TrainingDatasets(
        train=train_dataset,
        cv=eval_dataset,
        devtrain=eval_dataset,
        datastreams={**datastreams, "labels": label_datastream},
        prior=eval_dataset,
    )


def eow_phon_ls960_1023_gmm_warmup_generative_nce():
    prefix_name = "users/barkoczi/experiments/gen_ctc/ls960_ctc_eow_phon_gmm_warmup_generative_nce"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )
    ctc_train_data = build_eow_phon_training_datasets(
        prefix=prefix_name + "/ctc_soft_targets",
        librispeech_key="train-other-960",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, ctc_train_data.datastreams["labels"])
    gmm_train_data = _build_gmm_alignment_training_datasets(
        prefix_name=prefix_name + "/gmm_hard_targets",
        settings=train_settings,
        label_datastream=label_datastream,
    )

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
        label_target_size=label_datastream.vocab_size,
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

    full_learning_rates = (
        list(np.linspace(7e-6, 5e-4, 144))
        + list(np.linspace(5e-4, 5e-5, 144))
        + list(np.linspace(5e-5, 1e-7, 12))
    )
    common_train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
        "gradient_clip_norm": 1.0,
    }

    # train_partition_epoch=10, so 50 RETURNN epochs correspond to five full LS960 passes.
    total_num_epochs = 300
    gmm_num_epochs = 50
    ctc_num_epochs = total_num_epochs - gmm_num_epochs
    gmm_network_module = (
        "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_gmm"
    )
    gmm_train_args = {
        "config": {
            **common_train_config,
            "learning_rates": full_learning_rates[:gmm_num_epochs],
            "cleanup_old_models": {"keep_last_n": 1, "keep_best_n": 1, "keep": [gmm_num_epochs]},
        },
        "network_module": gmm_network_module,
        "net_args": {
            "model_config_dict": asdict(model_config),
            "alignment_label_map": GMM_ALIGNMENT_LABEL_MAP,
            "target_vocab": label_datastream.vocab,
            "alignment_subsampling_factor": 4,
        },
        "use_speed_perturbation": False,
        "debug": False,
    }
    gmm_training_name = (
        prefix_name
        + "/"
        + gmm_network_module
        + ".512dim_sub4_24gbgpu_5full-epochs_gmm-hard-targets_gennce"
    )
    gmm_train_job = training(
        gmm_training_name,
        gmm_train_data,
        gmm_train_args,
        num_epochs=gmm_num_epochs,
        **default_returnn,
    )
    gmm_train_job.rqmt["gpu_mem"] = 24

    ctc_network_module = (
        "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first"
    )
    ctc_train_args = {
        "config": {
            **common_train_config,
            "learning_rates": full_learning_rates[gmm_num_epochs:],
            "import_model_train_epoch1": gmm_train_job.out_checkpoints[gmm_num_epochs],
        },
        "network_module": ctc_network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "use_speed_perturbation": True,
        "debug": False,
    }
    ctc_training_name = (
        prefix_name
        + "/"
        + ctc_network_module
        + ".512dim_sub4_24gbgpu_25eps_sp_ctc-soft-targets_after_5eps-gmm_gennce"
    )
    ctc_train_job = training(
        ctc_training_name,
        ctc_train_data,
        ctc_train_args,
        num_epochs=ctc_num_epochs,
        **default_returnn,
    )
    ctc_train_job.rqmt["gpu_mem"] = 24

    def _format_scale_for_name(value):
        sign = "m" if value < 0 else "p"
        return sign + ("%.1f" % abs(value)).replace(".", "p")

    def tune_and_evaluate_helper(
        *,
        tuning_name,
        asr_model,
        lm_scales,
        prior_scales,
        blank_log_biases,
        posterior_temperatures,
    ):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_scale in lm_scales:
            for prior_scale in prior_scales:
                for blank_log_bias in blank_log_biases:
                    for posterior_temperature in posterior_temperatures:
                        decoder_config = DecoderConfig(
                            lexicon=get_text_lexicon(),
                            returnn_vocab=label_datastream.vocab,
                            beam_size=1024,
                            beam_size_token=12,
                            arpa_lm=arpa_4gram_lm,
                            beam_threshold=14,
                            lm_scale=lm_scale,
                            prior_scale=prior_scale,
                            prior_file=None,
                            blank_log_penalty=None,
                            normalize_log_probs=False,
                            generative_score_conversion=True,
                            blank_log_bias=blank_log_bias,
                            posterior_temperature=posterior_temperature,
                        )
                        search_name = (
                            tuning_name
                            + "/search_lm%.1f_prior%.1f_blank%s_temp%s"
                            % (
                                lm_scale,
                                prior_scale,
                                _format_scale_for_name(blank_log_bias),
                                _format_scale_for_name(posterior_temperature),
                            )
                        )
                        _search_jobs, wers = search(
                            search_name,
                            forward_config={},
                            asr_model=copy.deepcopy(asr_model),
                            decoder_module="ctc.decoder.flashlight_ctc_v2",
                            decoder_args={"config": asdict(decoder_config)},
                            test_dataset_tuples=dev_dataset_tuples,
                            **default_returnn,
                        )
                        tune_parameters.append(
                            (lm_scale, prior_scale, blank_log_bias, posterior_temperature)
                        )
                        tune_values_clean.append(wers[search_name + "/dev-clean"])
                        tune_values_other.append(wers[search_name + "/dev-other"])

        for key, tune_values in [
            ("test-clean", tune_values_clean),
            ("test-other", tune_values_other),
        ]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters,
                values=tune_values,
                mode="minimize",
            )
            pick_optimal_params_job.add_alias(tuning_name + f"/pick_best_{key}")
            decoder_config = DecoderConfig(
                lexicon=get_text_lexicon(),
                returnn_vocab=label_datastream.vocab,
                beam_size=1024,
                beam_size_token=12,
                arpa_lm=arpa_4gram_lm,
                beam_threshold=14,
                lm_scale=pick_optimal_params_job.out_optimal_parameters[0],
                prior_scale=pick_optimal_params_job.out_optimal_parameters[1],
                prior_file=None,
                blank_log_penalty=None,
                normalize_log_probs=False,
                generative_score_conversion=True,
                blank_log_bias=pick_optimal_params_job.out_optimal_parameters[2],
                posterior_temperature=pick_optimal_params_job.out_optimal_parameters[3],
            )
            test_search_name = tuning_name + f"/best_{key}"
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
            training_name=tuning_name,
            tune_parameters=tune_parameters,
            tuning_names=["LM", "Prior", "BlankBias", "Temp"],
            tune_values_clean=tune_values_clean,
            tune_values_other=tune_values_other,
            report_values=report_values,
        )

    asr_model_with_prior = prepare_asr_model(
        ctc_training_name,
        ctc_train_job,
        ctc_train_args,
        with_prior=True,
        datasets=ctc_train_data,
        get_specific_checkpoint=ctc_num_epochs,
    )
    tune_and_evaluate_helper(
        tuning_name=ctc_training_name + "/decode_generative_posterior_v2",
        asr_model=asr_model_with_prior,
        lm_scales=[1.6, 2.0, 2.4],
        prior_scales=[0.8, 1.0, 1.2],
        blank_log_biases=[-2.0, 0.0, 2.0],
        posterior_temperatures=[0.8, 1.0, 1.2],
    )


py = eow_phon_ls960_1023_gmm_warmup_generative_nce
