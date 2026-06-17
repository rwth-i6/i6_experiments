"""

"""
from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon, synthetic_librispeech_bliss_to_ogg_zip
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search
from ...report import tune_and_evalue_report
from ...storage import get_synthetic_data_lexicon


def eow_phon_ls100_2603_synth_compare():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls100_ctc_eow_phon"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=3,
        train_seq_ordering="laplace:.1000",
    )

    def get_training_settings_with_partition(partition):
        return DatasetSettings(
            preemphasis=0.97,
            peak_normalization=True,
            # training
            train_partition_epoch=partition,
            train_seq_ordering="laplace:.1000",
        )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-clean-100",
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

    def tune_and_evaluate_helper(training_name, asr_model, base_decoder_config, lm_scales, prior_scales, decoder_module="ctc.decoder.flashlight_ctc_v1"):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name, forward_config={}, asr_model=asr_model, decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)}, test_dataset_tuples={key: test_dataset_tuples[key]},
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


    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )
    

    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config, ConformerPosEmbConfig

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
    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Normal Style
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

    # Try to do like returnn frontend
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        pos_emb_config=posemb_config,
        specaug_config=specaug_config_full,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=8,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        mhsa_with_bias=True,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=1,  # Fine for phoneme
        dropout_broadcast_axes=None,  # No dropout broadcast yet to properly compare
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
    )

    train_config_24gbgpu = {
        "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        # There is a higher LR, because this model is only 384 in dimension
        "learning_rates": list(np.linspace(7e-5, 7e-4, 140)) + list(
            np.linspace(7e-4, 7e-5, 140)) + list(np.linspace(7e-5, 1e-7, 20)),
        #############
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    # for a first try still use old model with conv first
    network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    train_args = {
        "config": train_config_24gbgpu,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "use_speed_perturbation": True,
        "debug": False,
    }


    name = ".384dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_lr07"
    training_name = prefix_name + "/" + network_module + name
    train_job = training(training_name, train_data, train_args, num_epochs=300, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data,
        get_specific_checkpoint=300
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[2.5, 3.0, 3.5, 4.0], prior_scales=[0.1, 0.2, 0.3, 0.4])


    training_name = prefix_name + "/" + network_module + name + "_resume"
    train_args_resume = copy.deepcopy(train_args)
    train_args_resume["config"]["import_model_train_epoch1"] = asr_model.checkpoint  # only get checkpoint, rest should be identical

    train_job_resume = training(training_name, train_data, train_args_resume, num_epochs=300, **default_returnn)
    train_job_resume.rqmt["gpu_mem"] = 24
    asr_model_resume = prepare_asr_model(
        training_name, train_job_resume, train_args_resume, with_prior=True, datasets=train_data,
        get_specific_checkpoint=300
    )
    tune_and_evaluate_helper(training_name, asr_model_resume, default_decoder_config, lm_scales=[2.5, 3.0, 3.5, 4.0], prior_scales=[0.1, 0.2, 0.3, 0.4])


    # Take over data from JAIST experiment, combined training

    synth_keys = [
        #"ar_tts.tacotron2_decoding.tacotron2_decoding_v2_base320_fromglowbase256_400eps_gl32_syn_train-clean-360",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360",
        #"nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-360",
        #"nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-360",
        #"glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-360",
        # "glow_tts.glow_tts_v1_glow256align_400eps_oclr_nodrop_gl32_noise0.7_syn_train-clean-360",
        # "glow_tts.glow_tts_v1_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn_train-clean-360",
        #"grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-360",
        # ------------------------------------
        # "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360",
        # "glow_tts.glow_tts_v1_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn_train-clean-360",
    ]

    # resume training, this is the "reference" model for combined training, but has even worse WER than none resume

    from i6_experiments.users.rossenbach.experiments.jaist_project.storage import synthetic_bliss_data
    for synth_key in synth_keys:
        name = "/combine_3to1_synth/" + synth_key
        lexicon_key = "train-clean-460" if "train-clean-360" in synth_key else "train-clean-100"
        synth_ogg = synthetic_librispeech_bliss_to_ogg_zip(
            prefix=prefix_name + "/synth_data_prep/" + synth_key,
            bliss=synthetic_bliss_data[synth_key],
            lexicon_librispeech_key=lexicon_key)

        train_data_synth = build_eow_phon_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-clean-100",
            settings=get_training_settings_with_partition(12),
            extra_train_ogg_zips=[synth_ogg],
            data_repetition_factors=[3, 1], ## 3x original + 1x synth
        )

        training_name = prefix_name + "/" + network_module + name + "_resume" + name
        train_job = training(training_name, train_data_synth, train_args_resume, num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24  # just move to 24 gb, even without mixed precision training
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_resume, with_prior=True, datasets=train_data,
            get_specific_checkpoint=300
        )
        tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[3.0, 3.5, 4.0], prior_scales=[0.2, 0.3, 0.4])


