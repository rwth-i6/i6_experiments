from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from ..data.phon import build_eow_phon_training_datasets, TrainingDatasetSettings, get_text_lexicon
from ..data.common import build_test_dataset
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ..lm import get_4gram_binary_lm

from ..pipeline import training, search, compute_prior

from ..config import get_training_config, get_forward_config, get_prior_config
from ..storage import synthetic_bliss_data


def eow_phon_ls100_1023_base():
    prefix_name = "experiments/jaist_project/standalone_2024/ls100_ctc_eow_phon/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True, # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )
    
    train_settings_syn_training = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=18,
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

    def run_exp(ft_name, datasets, train_args, search_args=None, with_prior=False, num_epochs=250, decoder="ctc.decoder.flashlight_phoneme_ctc", eval_mode="dev", use_best=False):
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

        returnn_search_config = get_forward_config(**train_args, decoder_args=search_args,
                                                  decoder=decoder)

        assert eval_mode in ["dev", "test"]
        dataset_tuples = dev_dataset_tuples if eval_mode == "dev" else test_dataset_tuples
        _, _, search_jobs = search(ft_name + "/last_%i" % num_epochs, returnn_search_config,
                                   checkpoint, dataset_tuples, RETURNN_EXE,
                                   MINI_RETURNN_ROOT)

        return train_job, search_jobs
    
    
    from ..pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
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
    # diverged with hiccup
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #         }
    #         run_exp(
    #             prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR/lm%.1f_prior%.2f_bs1024_th14" % (
    #                 lm_weight, prior_scale),
    #             datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
            
    train_args_gc1 = copy.deepcopy(train_args)
    train_args_gc1["config"]["gradient_clip"] = 1.0
    train_args_gc1["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job_base, _ = run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_gc1, search_args=search_args, with_prior=True)
            if lm_weight == 3.5 and prior_scale == 0.5:
                # run test
                run_exp(
                    prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                        lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args_gc1, search_args=search_args, with_prior=True, eval_mode="test")

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
        "glow_tts.glow_tts_v1_bs600_v2_longer_base256_newgl16_extdur_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_longer_base256_newgl32_extdur_noise0.7_syn_train-clean-100",
        "glow_tts.glow_tts_v1_bs600_v2_longer_base256_newgl64_extdur_noise0.7_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromctc_v1_halfbatch_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromctc_v1_halfbatch_fixlr_fp16_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromglow_v1_halfbatch_fixlr_fp16_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromglowbase256_v1_halfbatch_fixlr_fp16_syn_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromglowbase256_v1_halfbatch_fixlr_fp16_syn_fixspk_train-clean-100",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromglowbase256_v1_halfbatch_fixlr_fp16_syn_train-clean-360-sub100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurtest_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.3_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.5_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.7_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise0.7_10steps_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_noise1.0_syn_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_syn_fixspk_train-clean-100",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_syn_train-clean-360-sub100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglow_v1_syn_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_syn_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_gl32_syn_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_syn_fixspk_train-clean-100",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_syn_train-clean-360-sub100",
    ]
    for syn_name in syn_names:
        syn_bliss = synthetic_bliss_data[syn_name]
        syn_train_data = build_eow_phon_training_datasets(
            librispeech_key="train-clean-100",
            settings=train_settings,
            real_data_weight=0,
            extra_bliss=[syn_bliss],
            # This is a tricky one, since we are having data from LibriSpeech 360 we also need that g2p vocab in order for it to work
            lexicon_librispeech_key="train-clean-460" if syn_name.endswith("sub100") else "train-clean-100",
        )
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
                train_job, _ = run_exp(
                    prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_syn/{syn_name}/lm%.1f_prior%.2f_bs1024_th14" % (
                        lm_weight, prior_scale),
                    datasets=syn_train_data, train_args=train_args_tmp, search_args=search_args, with_prior=True)
                if syn_name == "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.7_syn_train-clean-100" and lm_weight == 3.5:
                    train_job, _ = run_exp(
                        prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_syn/{syn_name}/lm%.1f_prior%.2f_bs1024_th14_best" % (
                            lm_weight, prior_scale),
                        datasets=syn_train_data, train_args=train_args_tmp, search_args=search_args, with_prior=True, use_best=True)

    # Resume
    
    # resume training
    train_args_resume = copy.deepcopy(train_args_gc1)
    train_args_resume["config"]["import_model_train_epoch1"] = train_job_base.out_checkpoints[250]
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job, _ = run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_resume, search_args=search_args, with_prior=True)

    # Synthetic combined training
    syn_names = [
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.3_syn_train-clean-360",
        "glow_tts.lukas_baseline_bs600_v2_newgl_noise0.7_syn_train-clean-360",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_noise0.7_syn_train-clean-360",
        "glow_tts.glow_tts_v1_bs600_v2_base256_newgl_extdur_noise0.7_syn_train-clean-360",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromctc_v1_halfbatch_fixlr_fp16_syn_train-clean-360",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromglow_v1_halfbatch_fixlr_fp16_syn_train-clean-360",
        "nar_tts.fastspeech_like.fastspeech_like_v1_fromglowbase256_v1_halfbatch_fixlr_fp16_syn_train-clean-360",
        # "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurtest_syn_train-clean-360",
        "grad_tts.grad_tts_v2_ext_dur_bs300_newgl_extdurglowbase256_syn_train-clean-360",
        "ar_tts.tacotron2_decoding.tacotron2_decoding_v2_fromglowbase256_v1_syn_train-clean-360",
    ]
    for syn_name in syn_names:
        syn_bliss = synthetic_bliss_data[syn_name]
        syn_train_data = build_eow_phon_training_datasets(
            librispeech_key="train-clean-100",
            settings=train_settings_syn_training,
            real_data_weight=3,
            extra_bliss=[syn_bliss],
            lexicon_librispeech_key="train-clean-460",
        )

        for lm_weight in [2.5, 3.0, 3.5]:
            for prior_scale in [0.0, 0.3, 0.5]:
                search_args = {
                    **default_search_args,
                    "lm_weight": lm_weight,
                    "prior_scale": prior_scale,
                }
                train_job, _ = run_exp(
                    prefix_name + f"conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_resume_syn/{syn_name}/lm%.1f_prior%.2f_bs1024_th14" % (
                        lm_weight, prior_scale),
                    datasets=syn_train_data, train_args=train_args_resume, search_args=search_args, with_prior=True)


    # longer training

    train_args_gc1_300ep = copy.deepcopy(train_args)
    train_args_gc1_300ep["config"]["gradient_clip"] = 1.0
    train_args_gc1_300ep["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    train_args_gc1_300ep["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 140)) + list(
                np.linspace(7e-4, 7e-5, 140)) + list(np.linspace(7e-5, 1e-8, 30))
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_amp16_ep300/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_gc1_300ep, search_args=search_args, with_prior=True, num_epochs=300)


    # longer training V2, this was not good....
    model_config_fixspec = copy.deepcopy(model_config)
    model_config_fixspec.specaug_config.max_dim_feat = 8

    train_args_gc1_300ep = copy.deepcopy(train_args)
    train_args_gc1_300ep["net_args"] =  {"model_config_dict": asdict(model_config_fixspec)}
    train_args_gc1_300ep["config"]["gradient_clip"] = 1.0
    train_args_gc1_300ep["config"]["torch_amp_options"] =  {"dtype": "bfloat16"}
    train_args_gc1_300ep["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 140)) + list(
                np.linspace(7e-4, 7e-5, 140)) + list(np.linspace(7e-5, 1e-8, 30))
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_halfspec_amp16_ep300/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_gc1_300ep, search_args=search_args, with_prior=True, num_epochs=300)


    train_args_gc1_speedpert = copy.deepcopy(train_args_gc1)
    train_args_gc1_speedpert["use_speed_perturbation"] = True
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1_sp_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args_gc1_speedpert, search_args=search_args, with_prior=True, num_epochs=250)



    frontend_config_smaller = copy.deepcopy(frontend_config)
    frontend_config_smaller.out_features = 144
    model_config_conformer_s = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_smaller,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=144,
        num_layers=16,
        num_heads=4,
        ff_dim=144 * 4,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config_conformer_s)},
    }
    train_args["config"]["torch_amp_options"] = {"dtype": "bfloat16"}
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train, _ = run_exp(
                prefix_name + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_conformer_s_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            # train.hold()

    # Test Glow-TTS MHSA

    from ..pytorch_networks.ctc.conformer_1023.i6modelsV1_TTSRelMHSA_VGG4LayerActFrontendV1_v1_cfg import \
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
        max_dim_feat=12,
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
        mhsa_window_size=16,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2_jjlr),
        "network_module": "ctc.conformer_1023.i6modelsV1_TTSRelMHSA_VGG4LayerActFrontendV1_v1",
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": True,
    }
    train_args["config"]["torch_amp_options"] = {"dtype": "bfloat16"}
    train_args["config"]["gradient_clip"] = 0.005
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train, _ = run_exp(
                prefix_name + "conformer_1023/i6modelsV1_TTSRelMHSA_VGG4LayerActFrontendV1_v1_JJLR_amp16/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
