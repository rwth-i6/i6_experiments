import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import build_training_dataset, TrainingDatasetSettings, build_tts_forward_dataset
from .config import get_training_config, get_extract_durations_forward__config, get_forward_config
from .pipeline import glowTTS_training, glowTTS_forward
from i6_experiments.users.rilling.experiments.librispeech.common.tts_eval import tts_eval

from ..default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, RETURNN_PYTORCH_ASR_SEARCH_EXE, MINI_RETURNN_ROOT

from i6_experiments.users.rossenbach.experiments.alignment_analysis_tts.gl_vocoder.default_vocoder import (
    get_default_vocoder,
)

from i6_experiments.users.rilling.experiments.librispeech.librispeech_x_vectors.storage import x_vector_extractions

from ..pytorch_networks.glowTTS_nar_taco_encoder import NarEncoderConfig


def get_pytorch_glowTTS(x_vector_exp: dict, gl_checkpoint: dict):
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/tts_architecture/glow_tts/raw_audio/"
    experiments = {}

    def run_exp(
        name,
        args,
        dataset,
        num_epochs=100,
        use_custom_engine=False,
        extra_evaluate_epoch=None,
        forward_args={},
        further_training=False,
        spectrogram_foward=False,
        durations_forward=False,
        latent_space_forward=False,
        joint_data_forward=False,
        train_data_forward=False,
        joint_durations_forward=False,
        keep_epochs=None,
        skip_forward=False,
        nisqa_evaluation=False,
        tts_eval_datasets=None,
        forward_device="gpu",
    ):
        exp = {}

        assert not nisqa_evaluation or (nisqa_evaluation and not skip_forward), "NISQA evaluation with skipping forward jobs is not possible"
        assert not nisqa_evaluation or ("x_vector" not in name or tts_eval_datasets is not None), "Attempting to evaluate a model with x-vector speaker embeddings, but missing explicit forward dataset with precalculated x-vector speaker embeddings."

        training_config = get_training_config(
            returnn_common_root=None,
            training_datasets=dataset,
            **args,
            use_custom_engine=use_custom_engine,
            pytorch_mode=True,
            keep_epochs=keep_epochs,
        )

        train_job = glowTTS_training(
            config=training_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
            num_epochs=num_epochs,
        )
        exp["train_job"] = train_job

        if further_training:
            further_training_args = copy.deepcopy(args)
            further_training_args["config"]["load"] = train_job.out_checkpoints[num_epochs]

            training_config2 = get_training_config(
                returnn_common_root=RETURNN_COMMON,
                training_datasets=dataset,
                **further_training_args,
                use_custom_engine=use_custom_engine,
                pytorch_mode=True,
            )

            train_job2 = glowTTS_training(
                config=training_config2,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name + "_further_training",
                num_epochs=num_epochs,
            )
            exp["train_job2"] = train_job2

        if not skip_forward:
            if not nisqa_evaluation:
                forward_config = get_forward_config(
                    returnn_common_root=RETURNN_COMMON,
                    forward_dataset=dataset,
                    **args,
                    forward_args=forward_args,
                    pytorch_mode=True,
                )
                forward_job = glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name,
                    device=forward_device
                )
                exp["forward_job"] = forward_job

                if extra_evaluate_epoch is not None:
                    if extra_evaluate_epoch < num_epochs:
                        forward_job = glowTTS_forward(
                            checkpoint=train_job.out_checkpoints[extra_evaluate_epoch],
                            config=forward_config,
                            returnn_exe=RETURNN_PYTORCH_EXE,
                            returnn_root=MINI_RETURNN_ROOT,
                            prefix=prefix + name,
                            extra_evaluation_epoch=extra_evaluate_epoch,
                        )
                        exp["forward_job_extra"] = forward_job
                if further_training:
                    forward_job = glowTTS_forward(
                        checkpoint=train_job2.out_checkpoints[num_epochs],
                        config=forward_config,
                        returnn_exe=RETURNN_PYTORCH_EXE,
                        returnn_root=MINI_RETURNN_ROOT,
                        prefix=prefix + name + "_further_training",
                    )
                    exp["forward_job2"] = forward_job
            else:
                # forward_config_univnet = get_forward_config(
                #     returnn_common_root=RETURNN_COMMON,
                #     forward_dataset=forward_dataset or dataset,
                #     **args,
                #     forward_args=forward_args,
                #     pytorch_mode=True,
                #     target="corpus_univnet"
                # )
                # forward_job_univnet = tts_eval(
                #     checkpoint=train_job.out_checkpoints[num_epochs],
                #     prefix_name=prefix + name,
                #     returnn_config=forward_config_univnet,
                #     returnn_exe=RETURNN_PYTORCH_EXE,
                #     returnn_root=MINI_RETURNN_ROOT,
                #     vocoder="univnet"
                # )
                # exp["forward_job_univnet"] = forward_job_univnet
                
                for ds_k, ds in tts_eval_datasets.items():
                    forward_config_gl = get_forward_config(
                        returnn_common_root=RETURNN_COMMON,
                        forward_dataset=ds,
                        **args,
                        forward_args={
                            **forward_args,
                            "gl_net_checkpoint": gl_checkpoint["checkpoint"],
                            "gl_net_config": gl_checkpoint["config"],
                        },
                        pytorch_mode=True,
                        target="corpus_gl",
                    )
                    forward_job_gl = tts_eval(
                        checkpoint=train_job.out_checkpoints[num_epochs],
                        prefix_name=prefix + name,
                        returnn_config=forward_config_gl,
                        returnn_exe=RETURNN_PYTORCH_EXE,
                        returnn_exe_asr=RETURNN_PYTORCH_ASR_SEARCH_EXE,
                        returnn_root=MINI_RETURNN_ROOT,
                        vocoder="gl", 
                        swer_eval=True,
                        nisqa_eval=True,
                        swer_eval_corpus_key=ds_k
                    )

            if spectrogram_foward:
                forward_config2 = get_forward_config(
                    returnn_common_root=RETURNN_COMMON,
                    forward_dataset=dataset,
                    **args,
                    forward_args=forward_args,
                    pytorch_mode=True,
                    target="spectrograms",
                )
                forward_job = glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config2,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name,
                    target="spectrograms",
                )
                exp["forward_job_spec"] = forward_job

            if joint_data_forward:
                forward_config_train_data = get_forward_config(
                    returnn_common_root=RETURNN_COMMON,
                    forward_dataset=dataset,
                    **args,
                    forward_args=forward_args,
                    pytorch_mode=True,
                    target="spectrograms",
                    joint_data=True,
                )

                forward_job = glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config_train_data,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name + "_joint_data",
                    target="spectrograms",
                )
                exp["forward_job_spec_joint_data"] = forward_job

            if train_data_forward:
                forward_config_train_data = get_forward_config(
                    returnn_common_root=RETURNN_COMMON,
                    forward_dataset=dataset,
                    **args,
                    forward_args=forward_args,
                    pytorch_mode=True,
                    target="spectrograms",
                    train_data=True,
                )

                forward_job = glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config_train_data,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name + "_train_data",
                    target="spectrograms",
                )
                exp["forward_job_spec_train_data"] = forward_job

            if durations_forward:
                forward_config_durations = get_forward_config(
                    returnn_common_root=RETURNN_COMMON,
                    forward_dataset=dataset,
                    **args,
                    forward_args=forward_args,
                    pytorch_mode=True,
                    target="durations",
                    train_data=True,
                )

                forward_job = glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config_durations,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name,
                    target="durations",
                )
                exp["forward_job_durations"] = forward_job

            if joint_durations_forward:
                forward_config_durations = get_forward_config(
                    returnn_common_root=RETURNN_COMMON,
                    forward_dataset=dataset,
                    **args,
                    forward_args=forward_args,
                    pytorch_mode=True,
                    target="durations",
                    joint_data=True,
                )

                forward_job = glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config_durations,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name + "/joint_dataset",
                    target="durations",
                )
                exp["forward_job_joint_durations"] = forward_job

            if latent_space_forward:
                forward_config_latent_space = get_forward_config(
                    returnn_common_root=RETURNN_COMMON,
                    forward_dataset=dataset,
                    **args,
                    forward_args=forward_args,
                    pytorch_mode=True,
                    target="latent_space",
                )

                forward_job = glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config_latent_space,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name,
                    target="latent_space",
                )
                exp["forward_job_latent_space"] = forward_job

        return exp

    def get_lr_scale(dim_model, step_num, warmup_steps):
        return np.power(dim_model, -0.5) * np.min(
            [np.power(step_num + 1, -0.5), step_num + 1 * np.power(warmup_steps, -1.5)]
        )

    net_module = "glowTTS"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    training_datasets = build_training_dataset(
        settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=False
    )
    training_datasets_silence_preprocessed = build_training_dataset(
        settings=train_settings, librispeech_key="train-clean-100", silence_preprocessing=True
    )

    tts_forward_datasets = {}
    tts_forward_datasets_xvectors = {}

    tts_forward_datasets["test-clean"] = build_tts_forward_dataset(
        librispeech_key="train-clean-100",
        dataset_key="test-clean",
    )

    tts_forward_datasets_xvectors["test-clean"] = build_tts_forward_dataset(
        librispeech_key="train-clean-100",
        dataset_key="test-clean",
        xvectors_file=x_vector_extractions["x_vector_cnn/1e-3_not_silence_preprocessed/test-clean"]["hdf"],
    )

    from .data import get_tts_log_mel_datastream
    from .feature_config import DbMelFeatureExtractionConfig
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import DBMelFilterbankOptions

    log_mel_datastream = get_tts_log_mel_datastream(silence_preprocessing=False)
    log_mel_datastream_silence_preprocessed = get_tts_log_mel_datastream(silence_preprocessing=True)

    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])
    assert isinstance(log_mel_datastream.options.feature_options, DBMelFilterbankOptions)
    fe_config = DbMelFeatureExtractionConfig(
        sample_rate=log_mel_datastream.options.sample_rate,
        win_size=log_mel_datastream.options.window_len,
        hop_size=log_mel_datastream.options.step_len,
        f_min=log_mel_datastream.options.feature_options.fmin,
        f_max=log_mel_datastream.options.feature_options.fmax,
        min_amp=log_mel_datastream.options.feature_options.min_amp,
        num_filters=log_mel_datastream.options.num_feature_filters,
        center=log_mel_datastream.options.feature_options.center,
        norm=norm,
    )

    log_mel_datastream_silence_preprocessed = get_tts_log_mel_datastream(silence_preprocessing=True)

    assert "norm_mean" in log_mel_datastream_silence_preprocessed.additional_options
    assert "norm_std_dev" in log_mel_datastream_silence_preprocessed.additional_options

    norm_silence_preprocessed = (
        log_mel_datastream_silence_preprocessed.additional_options["norm_mean"],
        log_mel_datastream_silence_preprocessed.additional_options["norm_std_dev"],
    )
    assert isinstance(log_mel_datastream_silence_preprocessed.options.feature_options, DBMelFilterbankOptions)
    fe_config_silence_preprocessed = DbMelFeatureExtractionConfig(
        sample_rate=log_mel_datastream_silence_preprocessed.options.sample_rate,
        win_size=log_mel_datastream_silence_preprocessed.options.window_len,
        hop_size=log_mel_datastream_silence_preprocessed.options.step_len,
        f_min=log_mel_datastream_silence_preprocessed.options.feature_options.fmin,
        f_max=log_mel_datastream_silence_preprocessed.options.feature_options.fmax,
        min_amp=log_mel_datastream_silence_preprocessed.options.feature_options.min_amp,
        num_filters=log_mel_datastream_silence_preprocessed.options.num_feature_filters,
        center=log_mel_datastream_silence_preprocessed.options.feature_options.center,
        norm=norm_silence_preprocessed,
    )

    train_args = {
        "net_args": {
            "n_vocab": training_datasets.datastreams["phonemes"].vocab_size,
            "hidden_channels": 192,
            "filter_channels": 768,
            "filter_channels_dp": 256,
            "out_channels": 80,
            "n_speakers": 251,
            "gin_channels": 256,
            "p_dropout": 0.1,
            "p_dropout_decoder": 0.05,
            "dilation_rate": 1,
            "n_sqz": 2,
            "prenet": True,
            "window_size": 4,
            "fe_config": asdict(fe_config),
        },
        "network_module": "glowTTS",
        "debug": True,
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-9},
            "num_workers_per_gpu": 2,
            # "learning_rate_control": "newbob_multi_epoch",
            # "learning_rate_control_min_num_epochs_per_new_lr": 5,
            # "learning_rate_control_relative_error_relative_lr": True,
            "learning_rates": list(np.concatenate((np.linspace(5e-5, 5e-4, 50), np.linspace(5e-4, 1e-6, 150)))),
            "gradient_clip_norm": 2.0,
            "use_learning_rate_control_always": True,
            "learning_rate_control_error_measure": "dev_loss_mle",
            # ############
            # "newbob_learning_rate_decay": 0.9,
            # "newbob_multi_num_epochs": 5,
            # "newbob_multi_update_interval": 1,
            # "newbob_relative_error_threshold": 0,
            #############
            # "batch_size": 56000,
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 25 * 16000},
            "max_seqs": 60,
        },
    }

    forward_args = {"noise_scale": 0.66, "length_scale": 1}

    # params = {
    #     "n_vocab": training_datasets.datastreams["phonemes"].vocab_size,
    #     "hidden_channels": 192,
    #     "filter_channels": 192,
    #     "filter_channels_dp": 256,
    #     "out_channels": 80,
    #     "n_speakers": 251,
    #     "gin_channels": 256,
    #     "p_dropout": 0.1,
    #     "p_dropout_decoder": 0.05,
    #     "dilation_rate": 1,
    #     "n_sqz": 2,
    #     "prenet": True,
    #     "window_size": 4
    # }

    # exp_dict = run_exp(net_module, params, net_module, config, dataset=training_datasets, debug=True)

    # net_module = "glowTTS_v2"

    # training_datasets2 = build_training_dataset(silence_preprocessed=True, durations_file="/work/asr4/rossenbach/sisyphus_work_folders/tts_asr_2021_work/i6_experiments/users/rossenbach/tts/duration_extraction/ViterbiAlignmentToDurationsJob.AyAO6JWXTnVc/output/durations.hdf", center=False)

    # exp_dict = run_exp_2_steps(name=net_module + "injected_durations", params=params, net_module=net_module, config=config, dataset=training_datasets2, debug=True)
    # exp_dict = run_exp(name=net_module + "further_training", params=params, net_module=net_module, config=config, dataset=training_datasets, debug=True)

    # train_args["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))
    # net_module = "glowTTS_vMF"

    # args_vMF = train_args.copy()
    # args_vMF["net_args"]["mean_only"] = True
    # exp_dict = run_exp(name=net_module, params=params, net_module=net_module, config=config, dataset=training_datasets, debug=True)

    net_module = "glowTTS"
    # exp_dict = run_exp(net_module + "_warmup", train_args, dataset=training_datasets)

    # config["learning_rates"] = [get_lr_scale(params["hidden_channels"], x, 80) for x in np.arange(0, 200)]
    train_args["net_args"]["mean_only"] = True
    # exp_dict = run_exp(net_module + "_warmup_fc768", params, net_module, config, dataset=training_datasets, debug=True, num_epochs=200)

    exp_dict = run_exp(
        net_module + "/enc768/mean_only/not_silence_preprocessed",
        train_args,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    experiments[net_module + "/enc768/mean_only/not_silence_preprocessed"] = exp_dict

    train_args_silence = copy.deepcopy(train_args)
    train_args_silence["net_args"]["fe_config"] = asdict(fe_config_silence_preprocessed)
    exp_dict = run_exp(
        net_module + "/enc768/mean_only/silence_preprocessed",
        train_args_silence,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args["net_args"]["mean_only"] = False
    train_args_silence["net_args"]["mean_only"] = False

    exp_dict = run_exp(
        net_module + "/enc768/with_sigma/not_silence_preprocessed",
        train_args,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        further_training=True,
        spectrogram_foward=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
        forward_device="cpu"
    )
    experiments[net_module + "/enc768/200ep/not_silence_preprocessed"] = exp_dict
    experiments[net_module + "/enc768/with_sigma/not_silence_preprocessed"] = exp_dict

    exp_dict = run_exp(
        net_module + "/enc768/with_sigma/silence_preprocessed",
        train_args_silence,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args["config"]["gradient_clip_norm"] = 10
    exp_dict = run_exp(
        net_module + "/enc768/with_sigma/grad_clip_norm_10",
        train_args,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
    )

    train_args_betas = copy.deepcopy(train_args)
    train_args_betas["config"]["optimizer"] = {"class": "adam", "epsilon": 1e-9, "betas": (0.9, 0.98)}
    exp_dict = run_exp(
        net_module + "/enc768/with_sigma/beta2_0.98",
        train_args_betas,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
    )

    train_args["net_args"]["filter_channels"] = 256
    exp_dict = run_exp(
        net_module + "/enc256/not_silence_preprocessed",
        train_args,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
        forward_device="cpu"
    )
    exp_dict = run_exp(
        net_module + "/enc256/silence_preprocessed",
        train_args,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_alternative_lr = copy.deepcopy(train_args)
    train_args_alternative_lr["net_args"]["filter_channels"] = 192
    train_args_alternative_lr["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-6, 5e-4, 50), np.linspace(5e-4, 1e-6, 150)))
    )

    exp_dict = run_exp(
        net_module + "/enc192/lr1e-6_5e-4_1e-6",
        train_args_alternative_lr,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_alternative_lr["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50)))
    )
    exp_dict = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed",
        train_args_alternative_lr,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
        forward_device="cpu"
    )
    experiments[net_module + "/enc192/100ep/not_silence_preprocessed"] = exp_dict

    exp_dict = run_exp(
        net_module + "/enc192/100ep/silence_preprocessed",
        train_args_alternative_lr,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_newbob = copy.deepcopy(train_args_alternative_lr)
    del train_args_newbob["config"]["learning_rates"]
    new_bob_settings = {
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        "learning_rate": 5e-4,
    }

    train_args_newbob["config"] = {**train_args_newbob["config"], **new_bob_settings}
    exp_dict = run_exp(
        net_module + "/enc192/100ep/newbob/Adam/not_silence_preprocessed",
        train_args_newbob,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        skip_forward=True,
    )

    train_args_newbob["config"]["optimizer"]["class"] = "radam"
    exp_dict = run_exp(
        net_module + "/enc192/100ep/newbob/RAdam/not_silence_preprocessed",
        train_args_newbob,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        skip_forward=True,
    )

    train_args_drop_speaker = copy.deepcopy(train_args_alternative_lr)

    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 1]:
        train_args_drop_speaker["net_args"]["p_speaker_drop"] = p

        exp_dict = run_exp(
            net_module + f"/enc192/100ep/speaker_drop/p_speaker_drop_{p}_not_silence_preprocessed",
            train_args_drop_speaker,
            dataset=training_datasets,
            num_epochs=100,
            forward_args=forward_args,
            spectrogram_foward=True,
        )

        experiments[net_module + f"/enc192/100ep/speaker_drop/p_speaker_drop_{p}_not_silence_preprocessed"] = exp_dict

    train_args_long_cooldown = copy.deepcopy(train_args_alternative_lr)
    train_args_long_cooldown["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50), np.linspace(1e-5, 1e-7, 100)))
    )
    train_args_long_cooldown["config"]["optimizer"]["epsilon"] = 1e-8
    exp_dict = run_exp(
        net_module + "/enc192/200ep/long_cooldown/not_silence_preprocessed",
        train_args_long_cooldown,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        durations_forward=True,
        latent_space_forward=True,
        joint_data_forward=True,
        train_data_forward=True,
        joint_durations_forward=True,
        keep_epochs={100},
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )
    experiments[net_module + "/enc192/200ep/long_cooldown/not_silence_preprocessed"] = exp_dict

    exp_dict = run_exp(
        net_module + "/enc192/200ep/long_cooldown/silence_preprocessed",
        train_args_long_cooldown,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        keep_epochs={100},
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
    )

    train_args_fs4 = copy.deepcopy(train_args_long_cooldown)
    train_args_fs4["net_args"]["n_sqz"] = 4

    exp_dict = run_exp(
        net_module + "/enc192/200ep/long_cooldown/fs4/silence_preprocessed",
        train_args_fs4,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_radam = copy.deepcopy(train_args_long_cooldown)
    train_args_radam["config"]["optimizer"]["class"] = "radam"
    del train_args_radam["config"]["learning_rates"]
    train_args_radam["config"]["learning_rate"] = 1e-3
    exp_dict = run_exp(
        net_module + "/enc192/100ep/RAdam/lr1e-3/silence_preprocessed",
        train_args_radam,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )
    train_args_radam["config"]["learning_rate"] = 1e-4
    exp_dict = run_exp(
        net_module + "/enc192/100ep/RAdam/lr1e-4/silence_preprocessed",
        train_args_radam,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_radam["config"]["learning_rate"] = 1e-5
    exp_dict = run_exp(
        net_module + "/enc192/100ep/RAdam/lr1e-5/silence_preprocessed",
        train_args_radam,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_radam["config"]["learning_rate"] = 1e-6
    exp_dict = run_exp(
        net_module + "/enc192/100ep/RAdam/lr1e-6/silence_preprocessed",
        train_args_radam,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    net_module = "glowTTS_nar_taco_encoder"
    train_args_nar_taco_encoder = copy.deepcopy(train_args)
    train_args_nar_taco_encoder["network_module"] = net_module
    train_args_nar_taco_encoder["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 150)))
    )
    encoder_config = NarEncoderConfig(
        label_in_dim=training_datasets.datastreams["phonemes"].vocab_size,
        embedding_size=256,
        conv_hidden_size=256,
        filter_size=3,
        dropout=0.5,
        lstm_size=256,
    )
    train_args_nar_taco_encoder["net_args"] = {
        "n_vocab": training_datasets.datastreams["phonemes"].vocab_size,
        "hidden_channels": 192,
        "filter_channels_dp": 256,
        "out_channels": 80,
        "n_speakers": 251,
        "gin_channels": 256,
        "p_dropout": 0.1,
        "p_dropout_decoder": 0.05,
        "dilation_rate": 1,
        "n_sqz": 2,
        "fe_config": asdict(fe_config),
        "encoder_config": asdict(encoder_config),
    }

    exp_dict = run_exp(
        net_module + "/not_silence_preprocessed",
        train_args_nar_taco_encoder,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
        extra_evaluate_epoch=107,  # was to late so 100 was already deleted
    )
    exp_dict = run_exp(
        net_module + "/silence_preprocessed",
        train_args_nar_taco_encoder,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        extra_evaluate_epoch=100,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
        forward_device="cpu",
    )

    net_module = "glowTTS_nar_taco_encoder_no_blstm"
    train_args_nar_taco_encoder_no_blstm = copy.deepcopy(train_args_nar_taco_encoder)
    train_args_nar_taco_encoder_no_blstm["network_module"] = net_module
    train_args_nar_taco_encoder_no_blstm["config"]["optimizer"]["class"] = "radam"
    train_args_nar_taco_encoder_no_blstm["config"]["learning_rates"] = list(
        np.concatenate(
            (np.linspace(1e-4, 5e-4, 50), np.linspace(1e-4, 1e-6, 50))
        )  # TODO: Huge step between warmup and cooldown. Delete when not needed anymore...
    )

    exp_dict = run_exp(
        net_module + "/silence_preprocessed",
        train_args_nar_taco_encoder_no_blstm,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
        forward_device="cpu",
    )

    train_args_nar_taco_encoder_no_blstm["net_args"]["n_blocks_dec"] = 16

    exp_dict = run_exp(
        net_module + "/16blocks/silence_preprocessed",
        train_args_nar_taco_encoder_no_blstm,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        durations_forward=True,
    )

    # train_args_nar_taco_encoder_no_blstm_no_warmup = copy.deepcopy(train_args_nar_taco_encoder_no_blstm)
    # train_args_nar_taco_encoder_no_blstm_no_warmup["config"]["learning_rates"] = list(
    #     np.concatenate((np.linspace(5e-4, 5e-4, 50), np.linspace(5e-4, 1e-6, 50)))
    # )

    # exp_dict = run_exp(
    #     net_module + "/16blocks/radam_no_warmup/silence_preprocessed",
    #     train_args_nar_taco_encoder_no_blstm_no_warmup,
    #     dataset=training_datasets_silence_preprocessed,
    #     num_epochs=100,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )

    net_module = "glowTTS_simple_encoder"
    train_args_simple_encoder = copy.deepcopy(train_args_nar_taco_encoder_no_blstm)
    train_args_simple_encoder["network_module"] = net_module
    train_args_simple_encoder["net_args"]["n_blocks_dec"] = 20

    exp_dict = run_exp(
        net_module + "/wrong_LR_schedule/silence_preprocessed",
        train_args_simple_encoder,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        extra_evaluate_epoch=67,
        durations_forward=True,
    )

    train_args_simple_encoder["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-4, 5e-4, 50), np.linspace(5e-4, 1e-6, 50)))
    )

    exp_dict = run_exp(
        net_module + "/silence_preprocessed",
        train_args_simple_encoder,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        extra_evaluate_epoch=67,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
        forward_device="cpu",
    )

    experiments[net_module + "/silence_preprocessed"] = exp_dict

    # net_module = "glowTTS_one_hot_encoder_mean"
    # train_args_one_hot_encoder = copy.deepcopy(train_args_simple_encoder)
    # train_args_one_hot_encoder["network_module"] = net_module

    # exp_dict = run_exp(
    #     net_module + "/silence_preprocessed",
    #     train_args_one_hot_encoder,
    #     dataset=training_datasets_silence_preprocessed,
    #     num_epochs=100,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )

    # net_module = "glowTTS_one_hot_encoder_std"
    # train_args_one_hot_encoder["network_module"] = net_module

    # exp_dict = run_exp(
    #     net_module + "/silence_preprocessed",
    #     train_args_one_hot_encoder,
    #     dataset=training_datasets_silence_preprocessed,
    #     num_epochs=100,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )

    # train_args_nar_taco_encoder["net_args"]["n_blocks_dec"] = 16
    # exp_dict = run_exp(
    #     net_module + "_16blocks",
    #     train_args_nar_taco_encoder,
    #     dataset=training_datasets,
    #     num_epochs=200,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )
    # exp_dict = run_exp(
    #     net_module + "_16_blocks_silence_preprocessed",
    #     train_args_nar_taco_encoder,
    #     dataset=training_datasets_silence_preprocessed,
    #     num_epochs=200,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )

    # ============== X-Vector speaker embeddings ====================#
    forward_dataset_xvector = build_training_dataset(
        settings=train_settings,
        librispeech_key="train-clean-100",
        silence_preprocessing=False,
        xvectors_file=x_vector_extractions["x_vector_cnn/1e-3_not_silence_preprocessed"]["hdf"],
    )
    train_args_x_vector = copy.deepcopy(train_args_alternative_lr)
    net_module = "glowTTS_x_vector"
    train_args_x_vector["network_module"] = net_module
    train_args_x_vector["net_args"]["gin_channels"] = 512  # Size of speaker embeddings from trained X-Vector
    x_vect_train_job = x_vector_exp["x_vector_cnn/1e-3_not_silence_preprocessed"]["train_job"]
    train_args_x_vector["config"]["preload_from_files"] = {
        "x_vector_model": {
            "filename": x_vect_train_job.out_checkpoints[x_vect_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "prefix": "x_vector.",
            "ignore_missing": True,
        }
    }
    exp_dict = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed",
        train_args_x_vector,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets_xvectors,
        forward_device="cpu",
    )

    net_module = "glowTTS_x_vector_eval"
    train_args_x_vector_eval = copy.deepcopy(train_args_x_vector)
    train_args_x_vector_eval["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed",
        train_args_x_vector_eval,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
    )

    net_module = "glowTTS_x_vector_v2"
    train_args_x_vector_v2 = copy.deepcopy(train_args_x_vector)
    train_args_x_vector_v2["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed",
        train_args_x_vector_v2,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
    )

    train_args_x_vector_v2["net_args"]["train_x_vector_epoch"] = 10
    exp_dict = run_exp(
        net_module + "/enc192/100ep_x_vector_ep10/not_silence_preprocessed/",
        train_args_x_vector_v2,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
    )

    net_module = "glowTTS_x_vector_v3"
    train_args_x_vector_v3 = copy.deepcopy(train_args_alternative_lr)  #
    train_args_x_vector_v3["net_args"]["gin_channels"] = 512  # Size of speaker embeddings from trained X-Vector
    train_args_x_vector_v3["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed",
        train_args_x_vector_v3,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
    )

    net_module = "glowTTS_x_vector_v3_norm_xvector"
    train_args_x_vector_v3_norm_xvector = copy.deepcopy(train_args_x_vector_v3)
    train_args_x_vector_v3_norm_xvector["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed",
        train_args_x_vector_v3_norm_xvector,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
    )

    train_args_x_vector_v3_norm_xvector_768fc = copy.deepcopy(train_args_x_vector_v3_norm_xvector)
    train_args_x_vector_v3_norm_xvector_768fc["net_args"]["filter_channels"] = 768
    exp_dict = run_exp(
        net_module + "/enc768/100ep/not_silence_preprocessed",
        train_args_x_vector_v3_norm_xvector_768fc,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
    )

    net_module = "glowTTS_x_vector_v3"
    train_args_x_vector_v3_768fc = copy.deepcopy(train_args_x_vector_v3_norm_xvector)
    train_args_x_vector_v3_768fc["net_args"]["filter_channels"] = 768
    train_args_x_vector_v3_768fc["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/enc768/100ep/not_silence_preprocessed",
        train_args_x_vector_v3_768fc,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
    )

    net_module = "glowTTS_x_vector"
    train_args_x_vector_768fc = copy.deepcopy(train_args_x_vector)
    train_args_x_vector_768fc["net_args"]["filter_channels"] = 768
    train_args_x_vector_768fc["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/enc768/100ep/not_silence_preprocessed",
        train_args_x_vector_768fc,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets_xvectors,
    )
    experiments[net_module + "/enc768/100ep/not_silence_preprocessed"] = exp_dict

    #  ============================== DDI ActNorm ==============================
    net_module = "glowTTS_ddi_actnorm"
    train_args_ddi_actnorm = copy.deepcopy(train_args_alternative_lr)
    train_args_ddi_actnorm["network_module"] = net_module
    exp_dict = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed/LR_scheduled",
        train_args_ddi_actnorm,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        nisqa_evaluation=True,
        tts_eval_datasets=tts_forward_datasets,
        forward_device="cpu",
    )
    train_args_ddi_actnorm_lin_lr = copy.deepcopy(train_args_ddi_actnorm)
    train_args_ddi_actnorm_lin_lr["config"]["learning_rate"] = 1e-4
    del train_args_ddi_actnorm_lin_lr["config"]["learning_rates"]
    exp_dict = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed/LR_constant",
        train_args_ddi_actnorm_lin_lr,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
    )

    #  ============================== Decoder Z Test ==============================
    train_args_decoder_z_test = copy.deepcopy(train_args_alternative_lr)
    train_args_decoder_z_test["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5 * 1e-4, 50), np.linspace(5 * 1e-4, 1e-5, 50)))
    )

    net_module = "glowTTS_decoder_test_simple_linear"
    train_args_decoder_z_test["network_module"] = net_module

    glowTTS_train_job = experiments["glowTTS/enc192/100ep/not_silence_preprocessed"]["train_job"]
    train_args_decoder_z_test["config"]["preload_from_files"] = {
        "glowTTS": {
            "filename": glowTTS_train_job.out_checkpoints[glowTTS_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu"
    )

    net_module = "glowTTS_decoder_test_multi_layer_ffn"
    train_args_decoder_z_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu"
    )

    net_module = "glowTTS_decoder_test_blstm"
    train_args_decoder_z_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu"
    )

    net_module = "glowTTS_encoder_sample_test_simple_linear"
    train_args_encoder_sample_test = copy.deepcopy(train_args_decoder_z_test)
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_multi_layer_ffn"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_blstm"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_simple_linear"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_multi_layer_ffn"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_blstm"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc192/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    glowTTS_train_job = experiments["glowTTS/enc768/mean_only/not_silence_preprocessed"]["train_job"]
    train_args_decoder_z_test["config"]["preload_from_files"] = {
        "glowTTS": {
            "filename": glowTTS_train_job.out_checkpoints[glowTTS_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    train_args_decoder_z_test["net_args"]["filter_channels"] = 768
    train_args_decoder_z_test["net_args"]["mean_only"] = True

    net_module = "glowTTS_decoder_test_simple_linear"
    train_args_decoder_z_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_decoder_test_multi_layer_ffn"
    train_args_decoder_z_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_decoder_test_blstm"
    train_args_decoder_z_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_simple_linear"
    train_args_encoder_sample_test = copy.deepcopy(train_args_decoder_z_test)
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_multi_layer_ffn"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_blstm"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_simple_linear"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_multi_layer_ffn"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_blstm"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/mean_only/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    glowTTS_train_job = experiments["glowTTS/enc768/with_sigma/not_silence_preprocessed"]["train_job"]
    train_args_decoder_z_test["config"]["preload_from_files"] = {
        "glowTTS": {
            "filename": glowTTS_train_job.out_checkpoints[glowTTS_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    train_args_decoder_z_test["net_args"]["mean_only"] = False

    net_module = "glowTTS_decoder_test_simple_linear"
    train_args_decoder_z_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_decoder_test_multi_layer_ffn"
    train_args_decoder_z_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_decoder_test_blstm"
    train_args_decoder_z_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_decoder_z_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_simple_linear"
    train_args_encoder_sample_test = copy.deepcopy(train_args_decoder_z_test)
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_multi_layer_ffn"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_blstm"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_simple_linear"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_multi_layer_ffn"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_encoder_sample_test_maxlike_alignment_blstm"
    train_args_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/enc768/with_sigma/" + net_module,
        train_args_encoder_sample_test,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_simple_encoder_test_maxlike_alignment_multi_layer_ffn"
    train_args_simple_encoder_sample_test = copy.deepcopy(train_args_simple_encoder)
    train_args_simple_encoder_sample_test["network_module"] = net_module
    glowTTS_train_job = experiments["glowTTS_simple_encoder/silence_preprocessed"]["train_job"]
    train_args_simple_encoder_sample_test["config"]["preload_from_files"] = {
        "glowTTS": {
            "filename": glowTTS_train_job.out_checkpoints[glowTTS_train_job.returnn_config.get("num_epochs", 100)],
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    exp_dict = run_exp(
        "decoder_test/simple_enc/" + net_module,
        train_args_simple_encoder_sample_test,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    net_module = "glowTTS_simple_encoder_test_maxlike_alignment_multi_layer_ffn_v2"
    train_args_simple_encoder_sample_test["network_module"] = net_module
    exp_dict = run_exp(
        "decoder_test/simple_enc/" + net_module,
        train_args_simple_encoder_sample_test,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        forward_device="cpu",
    )

    return experiments
