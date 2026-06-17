from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.experiments.jaist_project.data.phon import build_eow_phon_training_datasets, TrainingDatasetSettings, get_text_lexicon
from i6_experiments.users.rossenbach.experiments.jaist_project.data.common import build_test_dataset
from i6_experiments.users.rossenbach.experiments.jaist_project.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.jaist_project.lm import get_4gram_binary_lm

from i6_experiments.users.rossenbach.experiments.jaist_project.pipeline import training, search, compute_prior

from i6_experiments.users.rossenbach.experiments.jaist_project.config import get_training_config, get_forward_config, get_prior_config
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import synthetic_bliss_data

from i6_experiments.users.rossenbach.tools.parameter_tuning import PickOptimalParametersJob


def eow_phon_ls100_1023_synthetic():
    prefix_name = "experiments/jaist_project/asr/ls100_ctc_eow_phon/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )
    
    train_settings_syn_training_ls460 = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=18,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )
    
    train_settings_syn_training_ls100 = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=6,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )
    
    train_settings_syn_training_ls360only = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=9,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        librispeech_key="train-clean-100",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    # build testing datasets
    dev_dataset_tuples = {}
    for devset in ["dev-clean", "dev-other"]:
            dev_dataset_tuples[devset] = build_test_dataset(
                dataset_key=devset,
                preemphasis=train_settings.preemphasis,
                peak_normalization=train_settings.peak_normalization,
            )
    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                dataset_key=testset,
                preemphasis=train_settings.preemphasis,
                peak_normalization=train_settings.peak_normalization,
            )

    arpa_4gram_lm = get_4gram_binary_lm()

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp(ft_name, datasets, train_args, search_args=None, with_prior=False, num_epochs=250, decoder="ctc.decoder.flashlight_phoneme_ctc", eval_mode="dev", use_best=False, return_wers_and_model=False):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if use_best:
            from i6_core.returnn.training import GetBestPtCheckpointJob
            checkpoint = GetBestPtCheckpointJob(
                model_dir=train_job.out_model_dir,
                learning_rates=train_job.out_learning_rates,
                key="dev_loss_ctc",
            ).out_checkpoint
        else:
            checkpoint = train_job.out_checkpoints[num_epochs]

        if with_prior:
            returnn_config = get_prior_config(training_datasets=datasets, **train_args)
            prior_file = compute_prior(
                ft_name,
                returnn_config,
                checkpoint=checkpoint,
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(training_name + "/prior.txt", prior_file)
            search_args["prior_file"] = prior_file
        else:
            prior_file = None

        returnn_search_config = get_forward_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)

        assert eval_mode in ["dev", "test"]
        dataset_tuples = dev_dataset_tuples if eval_mode == "dev" else test_dataset_tuples
        ret_vals = search(ft_name + "/last_%i" % num_epochs, returnn_search_config,
                                   checkpoint, dataset_tuples, RETURNN_EXE,
                                   MINI_RETURNN_ROOT, return_wers=return_wers_and_model)
        if return_wers_and_model:
            return train_job, ret_vals[2], ret_vals[3], prior_file
        return train_job, ret_vals[2]


    def dedicated_search(ft_name, dataset_key, checkpoint, train_args, search_args, decoder="ctc.decoder.flashlight_phoneme_ctc"):
        returnn_search_config = get_forward_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)
        dataset_tuples = {dataset_key: test_dataset_tuples[dataset_key]}
        ret_vals = search(ft_name + "/" + dataset_key, returnn_search_config,
                                   checkpoint, dataset_tuples, RETURNN_EXE,
                                   MINI_RETURNN_ROOT, with_confidence=True)
    
    
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config

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

    train_args_adamw03_accum2_jjlr = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110)) + list(
                np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 360 * 16000,  # no grad accum needed within Kagayaki
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
        },
        "debug": False,
    }

    default_search_args = {
        "lexicon": get_text_lexicon(librispeech_key="train-clean-100"),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 1024,
        "beam_size_token": 128,
        "arpa_lm": arpa_4gram_lm,
        "beam_threshold": 14,
    }

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config)},
    }

    train_args_gc1 = copy.deepcopy(train_args)
    train_args_gc1["config"]["gradient_clip"] = 1.0
    train_args_gc1["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}

    tune_parameters = []
    tune_values_clean = []
    tune_values_other = []
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job_base, _, wers, prior_file = run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_gc1, search_args=search_args, with_prior=True, return_wers_and_model=True)
            tune_parameters.append((lm_weight, prior_scale))
            tune_values_clean.append((wers["dev-clean"]))
            tune_values_other.append((wers["dev-other"]))
    for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
        pick_optimal_params_job = PickOptimalParametersJob(parameters=tune_parameters, values=tune_values)
        pick_optimal_params_job.add_alias(
            prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16/pick_best_{key}")
        search_args = copy.deepcopy(default_search_args)
        search_args["lm_weight"] = pick_optimal_params_job.out_optimal_parameters[0]
        search_args["prior_scale"] = pick_optimal_params_job.out_optimal_parameters[1]
        search_args["prior_file"] = prior_file
        dedicated_search(
            ft_name=prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16",
            dataset_key=key,
            checkpoint=train_job_base.out_checkpoints[250],
            train_args=train_args_gc1,
            search_args=search_args
        )


    # Resume

    # resume training, this is the "reference" model for combined training, but has even worse WER than none resume
    train_args_resume = copy.deepcopy(train_args_gc1)
    train_args_resume["config"]["import_model_train_epoch1"] = train_job_base.out_checkpoints[250]

    tune_parameters = []
    tune_values_clean = []
    tune_values_other = []
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job, _, wers, prior_file = run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_resume, search_args=search_args, with_prior=True, return_wers_and_model=True)
            tune_parameters.append((lm_weight, prior_scale))
            tune_values_clean.append((wers["dev-clean"]))
            tune_values_other.append((wers["dev-other"]))
    for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
        pick_optimal_params_job = PickOptimalParametersJob(parameters=tune_parameters, values=tune_values)
        pick_optimal_params_job.add_alias(
            prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume/pick_best_{key}")
        search_args = copy.deepcopy(default_search_args)
        search_args["lm_weight"] = pick_optimal_params_job.out_optimal_parameters[0]
        search_args["prior_scale"] = pick_optimal_params_job.out_optimal_parameters[1]
        search_args["prior_file"] = prior_file
        dedicated_search(
            ft_name=prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume",
            dataset_key=key,
            checkpoint=train_job_base.out_checkpoints[250],
            train_args=train_args_gc1,
            search_args=search_args
        )


    # Synthetic solo training
    syn_names = [
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.0_syn_train-clean-100",
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.3_syn_train-clean-100",
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.5_syn_train-clean-100",
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.7_syn_train-clean-100",
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise1.0_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_newgl_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_newgl_noise0.7_cont100_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise0.0_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise0.3_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise0.5_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise0.7_syn_fixspk_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise1.0_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise0.7_syn_train-clean-360-sub100",
        "glow_tts.glow_tts_v1_bs600_v2_longer_base256_newgl_extdur_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_longer_noam_base256_newgl_extdur_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl16_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl64_noise0.7_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromctc_v1_halfbatch_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromctc_v1_halfbatch_fixlr_fp16_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromglow_v1_halfbatch_fixlr_fp16_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_200eps_bs300_oclr_fp16_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_200eps_bs300_oclr_fp16_syn_fixspk_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_200eps_bs300_oclr_fp16_syn_train-clean-360-sub100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_noam_fp16_syn_train-clean-100",
        "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_glow256align_200eps_bs600_oclr_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurtest_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.3_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.5_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.7_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.7_10steps_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise1.0_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.5_syn_fixspk_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.5_syn_train-clean-360-sub100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_oclr_noise0.7_syn_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglow_v1_syn_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_syn_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_gl32_syn_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_syn_fixspk_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_syn_train-clean-360-sub100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_v1_syn_train-clean-100",
        # --------------------------------------------------------------------------------------------------------------

        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_fixspk_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360-sub100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_base320_fromglowbase256_400eps_gl32_syn_train-clean-360-sub100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_fixspk_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-360-sub100",
        "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-100",
        "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_fixspk_train-clean-100",
        "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-360-sub100",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_nodrop_gl32_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_fixspk_train-clean-100",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-360-sub100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_fixspk_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-360-sub100",
        ([
                "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-100",
                "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-100",
                "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-100",
                "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-100",
        ], True, "all_except_narblstm_400eps_normal_random_merged"),
    ]

    for syn_name in syn_names:
        if isinstance(syn_name, tuple):
            extra_bliss_names, random_merge_extra_bliss, new_name = syn_name
            extra_bliss = [synthetic_bliss_data[n] for n in extra_bliss_names]
            syn_name = new_name
        else:
            extra_bliss = [synthetic_bliss_data[syn_name]]
            random_merge_extra_bliss = False
        syn_train_data = build_eow_phon_training_datasets(
            librispeech_key="train-clean-100",
            settings=train_settings,
            real_data_weight=0,
            extra_bliss=extra_bliss,
            # This is a tricky one, since we are having data from LibriSpeech 360 we also need that g2p vocab in order for it to work
            lexicon_librispeech_key="train-clean-460" if syn_name.endswith("sub100") else "train-clean-100",
            random_merge_extra_bliss=random_merge_extra_bliss,
        )
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        for lm_weight in [2.5, 3.0, 3.5]:
            for prior_scale in [0.0, 0.3, 0.5]:
                search_args = {
                    **default_search_args,
                    "lm_weight": lm_weight,
                    "prior_scale": prior_scale,
                }
                train_args_tmp = copy.deepcopy(train_args_gc1)
                # somehow training diverged, run with a new seed
                if syn_name ==  "glow_tts.lukas_baseline_bs600_v2_newgl_noise1.0_syn_train-clean-100":
                    train_args_tmp["config"]["random_seed"] = 43  # default is obviously 42
                if syn_name ==  "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_fixspk_train-clean-100":
                    train_args_tmp["config"]["random_seed"] = 43
                train_job, _, wers, prior_file = run_exp(
                    prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_syn/{syn_name}/lm%.1f_prior%.2f_bs1024_th14" % (
                        lm_weight, prior_scale),
                    datasets=syn_train_data, train_args=train_args_tmp, search_args=search_args, with_prior=True, return_wers_and_model=True)
                if syn_name == "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.7_syn_train-clean-100" and lm_weight == 3.5:
                    train_job, _, wers, prior_file = run_exp(
                        prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_syn/{syn_name}/lm%.1f_prior%.2f_bs1024_th14_best" % (
                            lm_weight, prior_scale),
                        datasets=syn_train_data, train_args=train_args_tmp, search_args=search_args, with_prior=True, use_best=True, return_wers_and_model=True)

                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers["dev-clean"]))
                tune_values_other.append((wers["dev-other"]))
        if isinstance(syn_name, str) and "400eps" in syn_name:
            for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
                pick_optimal_params_job = PickOptimalParametersJob(parameters=tune_parameters, values=tune_values)
                pick_optimal_params_job.add_alias(prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_syn/{syn_name}/pick_best_{key}")
                search_args = copy.deepcopy(default_search_args)
                search_args["lm_weight"] = pick_optimal_params_job.out_optimal_parameters[0]
                search_args["prior_scale"] = pick_optimal_params_job.out_optimal_parameters[1]
                search_args["prior_file"] = prior_file
                dedicated_search(
                    ft_name=prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_syn/{syn_name}",
                    dataset_key=key,
                    checkpoint=train_job.out_checkpoints[250],
                    train_args=train_args_gc1,
                    search_args=search_args
                )


    # Resume
    
    # resume training, this is the "reference" model for combined training, but has even worse WER than none resume
    train_args_resume = copy.deepcopy(train_args_gc1)
    train_args_resume["config"]["import_model_train_epoch1"] = train_job_base.out_checkpoints[250]


    # Synthetic combined training
    syn_names = [
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.3_syn_train-clean-360",
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.7_syn_train-clean-360",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_noise0.7_syn_train-clean-360",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise0.7_syn_train-clean-360",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromctc_v1_halfbatch_fixlr_fp16_syn_train-clean-360",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromglow_v1_halfbatch_fixlr_fp16_syn_train-clean-360",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_200eps_bs300_oclr_fp16_syn_train-clean-360",
        # "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurtest_syn_train-clean-360",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.5_syn_train-clean-360",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_syn_train-clean-360",
        # --------------------------------------------------------------------------------------------------------------
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-360",
        "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-360",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-360",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-360",
    ]
    for syn_name in syn_names:
        syn_bliss = synthetic_bliss_data[syn_name]
        syn_train_data = build_eow_phon_training_datasets(
            librispeech_key="train-clean-100",
            settings=train_settings_syn_training_ls460,
            real_data_weight=3,
            extra_bliss=[syn_bliss],
            lexicon_librispeech_key="train-clean-460",
        )
        
        
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        for lm_weight in [2.5, 3.0, 3.5]:
            for prior_scale in [0.0, 0.3, 0.5]:
                search_args = {
                    **default_search_args,
                    "lm_weight": lm_weight,
                    "prior_scale": prior_scale,
                }
                train_job, _, wers, prior_file = run_exp(
                    prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume_syn/{syn_name}/lm%.1f_prior%.2f_bs1024_th14" % (
                        lm_weight, prior_scale),
                    datasets=syn_train_data, train_args=train_args_resume, search_args=search_args, with_prior=True, return_wers_and_model=True)
                
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers["dev-clean"]))
                tune_values_other.append((wers["dev-other"]))
        
        if isinstance(syn_name, str) and "400eps" in syn_name:
            for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
                pick_optimal_params_job = PickOptimalParametersJob(parameters=tune_parameters, values=tune_values)
                pick_optimal_params_job.add_alias(prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume_syn/{syn_name}/pick_best_{key}")
                search_args = copy.deepcopy(default_search_args)
                search_args["lm_weight"] = pick_optimal_params_job.out_optimal_parameters[0]
                search_args["prior_scale"] = pick_optimal_params_job.out_optimal_parameters[1]
                search_args["prior_file"] = prior_file
                dedicated_search(
                    ft_name=prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume_syn/{syn_name}",
                    dataset_key=key,
                    checkpoint=train_job.out_checkpoints[250],
                    train_args=train_args_resume,
                    search_args=search_args)




    # Synthetic combined training but with ls-100
    syn_names = [
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-100",
        "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-100",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-100",
        #-------------------------------------------------------------------------------------------------------------
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_400eps_gl32_syn_train-clean-360-sub100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_glow256align_400eps_bs300_oclr_fp16_gl32_syn_train-clean-360-sub100",
        "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm_size512_glow256align_400eps_bs600_oclr_gl32_syn_train-clean-360-sub100",
        "glow_tts.glow_tts_v1_glow256align_400eps_oclr_gl32_noise0.7_syn_train-clean-360-sub100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_400eps_noise0.7_step10_gl32_syn_train-clean-360-sub100",
    ]


    for syn_name in syn_names:
        syn_bliss = synthetic_bliss_data[syn_name]
        syn_train_data = build_eow_phon_training_datasets(
            librispeech_key="train-clean-100",
            settings=train_settings_syn_training_ls100,
            real_data_weight=1,
            extra_bliss=[syn_bliss],
            lexicon_librispeech_key="train-clean-460" if syn_name.endswith("sub100") else "train-clean-100",
        )

        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        for lm_weight in [2.5, 3.0, 3.5]:
            for prior_scale in [0.0, 0.3, 0.5]:
                search_args = {
                    **default_search_args,
                    "lm_weight": lm_weight,
                    "prior_scale": prior_scale,
                }
                train_job, _, wers, prior_file = run_exp(
                    prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume_syn/{syn_name}/lm%.1f_prior%.2f_bs1024_th14" % (
                        lm_weight, prior_scale),
                    datasets=syn_train_data, train_args=train_args_resume, search_args=search_args, with_prior=True, return_wers_and_model=True)
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers["dev-clean"]))
                tune_values_other.append((wers["dev-other"]))
        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = PickOptimalParametersJob(parameters=tune_parameters, values=tune_values)
            pick_optimal_params_job.add_alias(prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume_syn/{syn_name}/pick_best_{key}")
            search_args = copy.deepcopy(default_search_args)
            search_args["lm_weight"] = pick_optimal_params_job.out_optimal_parameters[0]
            search_args["prior_scale"] = pick_optimal_params_job.out_optimal_parameters[1]
            search_args["prior_file"] = prior_file
            dedicated_search(
                ft_name=prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume_syn/{syn_name}",
                dataset_key=key,
                checkpoint=train_job.out_checkpoints[250],
                train_args=train_args_resume,
                search_args=search_args
            )
