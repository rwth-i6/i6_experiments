import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import build_training_dataset, TrainingDatasetSettings
from .config import get_training_config, get_extract_durations_forward__config, get_forward_config
from .pipeline import glowTTS_training, glowTTS_forward

from ..default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT

from i6_experiments.users.rossenbach.experiments.alignment_analysis_tts.gl_vocoder.default_vocoder import (
    get_default_vocoder,
)

from ..pytorch_networks.glowTTS_nar_taco_encoder import NarEncoderConfig


def get_pytorch_glowTTS():
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    prefix = "experiments/librispeech/tts_architecture/glow_tts/raw_audio/"

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
        train_data_forward=False,
        keep_epochs=None,
    ):
        training_config = get_training_config(
            returnn_common_root=RETURNN_COMMON,
            training_datasets=dataset,
            **args,
            use_custom_engine=use_custom_engine,
            pytorch_mode=True,
            keep_epochs=keep_epochs,
        )  # implicit reconstruction loss

        forward_config = get_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=dataset,
            **args,
            forward_args=forward_args,
            pytorch_mode=True,
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

        train_job = glowTTS_training(
            config=training_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
            num_epochs=num_epochs,
        )

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

        if extra_evaluate_epoch is not None:
            if extra_evaluate_epoch < num_epochs:
                glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[extra_evaluate_epoch],
                    config=forward_config,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name,
                    extra_evaluation_epoch=extra_evaluate_epoch,
                )

        tts_hdf = glowTTS_forward(
            checkpoint=train_job.out_checkpoints[num_epochs],
            config=forward_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
        )

        if spectrogram_foward:
            glowTTS_forward(
                checkpoint=train_job.out_checkpoints[num_epochs],
                config=forward_config2,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name,
                target="spectrograms",
            )

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

                glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config_train_data,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name + "_train_data",
                    target="spectrograms",
                )

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

                glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config_durations,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name,
                    target="durations",
                )

            if latent_space_forward:
                forward_config_latent_space = get_forward_config(
                    returnn_common_root=RETURNN_COMMON,
                    forward_dataset=dataset,
                    **args,
                    forward_args=forward_args,
                    pytorch_mode=True,
                    target="latent_space",
                )

                glowTTS_forward(
                    checkpoint=train_job.out_checkpoints[num_epochs],
                    config=forward_config_latent_space,
                    returnn_exe=RETURNN_PYTORCH_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    prefix=prefix + name,
                    target="latent_space",
                )

        if further_training:
            tts_hdf = glowTTS_forward(
                checkpoint=train_job2.out_checkpoints[num_epochs],
                config=forward_config,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name + "_further_training",
            )

        return tts_hdf

    # def run_exp_2_steps(name, params, net_module, config, dataset, num_epochs=100, use_custom_engine=False, debug=False):
    #     training_config = get_training_config(
    #         returnn_common_root=RETURNN_COMMON,
    #         training_datasets=dataset,
    #         network_module=net_module,
    #         net_args=params,
    #         config=config,
    #         debug=debug,
    #         use_custom_engine=use_custom_engine,
    #         pytorch_mode=True
    #     )  # implicit reconstruction loss

    #     forward_config = get_forward_config(
    #         returnn_common_root=RETURNN_COMMON,
    #         forward_dataset=dataset,
    #         network_module=net_module,
    #         net_args=params,
    #         debug=debug,
    #         pytorch_mode=True
    #     )
    #     train_job = glowTTS_training(
    #         config=training_config,
    #         returnn_exe=RETURNN_PYTORCH_EXE,
    #         returnn_root=MINI_RETURNN_ROOT,
    #         prefix=prefix + name,
    #         num_epochs=num_epochs
    #     )

    #     net_module_further = net_module + "_further_training"
    #     # net_module_further = net_module + "_test"
    #     config_further = config.copy()
    #     config_further["import_model_train_epoch1"] = train_job.out_checkpoints[num_epochs].path
    #     further_training_config = get_training_config(
    #         returnn_common_root=RETURNN_COMMON,
    #         training_datasets=dataset,
    #         network_module=net_module_further,
    #         net_args=params,
    #         config=config_further,
    #         debug=debug,
    #         use_custom_engine=use_custom_engine,
    #         pytorch_mode=True
    #     )

    #     further_train_job = glowTTS_training(
    #         config=further_training_config,
    #         returnn_exe=RETURNN_PYTORCH_EXE,
    #         returnn_root=MINI_RETURNN_ROOT,
    #         prefix=prefix + name,
    #         num_epochs=35
    #     )

    #     tts_hdf = glowTTS_forward(
    #         checkpoint=train_job.out_checkpoints[num_epochs],
    #         config=forward_config,
    #         returnn_exe=RETURNN_PYTORCH_EXE,
    #         returnn_root=MINI_RETURNN_ROOT,
    #         prefix=prefix + name
    #     )

    #     tts_hdf = glowTTS_forward(
    #         checkpoint=further_train_job.out_checkpoints[35],
    #         config=forward_config,
    #         returnn_exe=RETURNN_PYTORCH_EXE,
    #         returnn_root=MINI_RETURNN_ROOT,
    #         prefix=prefix + name,
    #         alias_addition="2"
    #     )

    #     # name = "vocoderTest"
    #     # vocoder = get_default_vocoder(name)
    #     # forward_vocoded, vocoder_forward_job = vocoder.vocode(
    #     #     tts_hdf, iterations=30, cleanup=True, name=name
    #     # )

    #     # tk.register_output(name + "/forward_dev_corpus.xml.gz", forward_vocoded)

    #     return tts_hdf

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

    # tts_hdf = run_exp(net_module, params, net_module, config, dataset=training_datasets, debug=True)

    # net_module = "glowTTS_v2"

    # training_datasets2 = build_training_dataset(silence_preprocessed=True, durations_file="/work/asr4/rossenbach/sisyphus_work_folders/tts_asr_2021_work/i6_experiments/users/rossenbach/tts/duration_extraction/ViterbiAlignmentToDurationsJob.AyAO6JWXTnVc/output/durations.hdf", center=False)

    # tts_hdf = run_exp_2_steps(name=net_module + "injected_durations", params=params, net_module=net_module, config=config, dataset=training_datasets2, debug=True)
    # tts_hdf = run_exp(name=net_module + "further_training", params=params, net_module=net_module, config=config, dataset=training_datasets, debug=True)

    # train_args["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))
    # net_module = "glowTTS_vMF"

    # args_vMF = train_args.copy()
    # args_vMF["net_args"]["mean_only"] = True
    # tts_hdf = run_exp(name=net_module, params=params, net_module=net_module, config=config, dataset=training_datasets, debug=True)

    net_module = "glowTTS"
    # tts_hdf = run_exp(net_module + "_warmup", train_args, dataset=training_datasets)

    # config["learning_rates"] = [get_lr_scale(params["hidden_channels"], x, 80) for x in np.arange(0, 200)]
    train_args["net_args"]["mean_only"] = True
    # tts_hdf = run_exp(net_module + "_warmup_fc768", params, net_module, config, dataset=training_datasets, debug=True, num_epochs=200)

    tts_hdf = run_exp(
        net_module + "/enc768/mean_only/not_silence_preprocessed",
        train_args,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_silence = copy.deepcopy(train_args)
    train_args_silence["net_args"]["fe_config"] = asdict(fe_config_silence_preprocessed)
    tts_hdf = run_exp(
        net_module + "/enc768/mean_only/silence_preprocessed",
        train_args_silence,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
    )

    train_args["net_args"]["mean_only"] = False
    train_args_silence["net_args"]["mean_only"] = False

    tts_hdf = run_exp(
        net_module + "/enc768/with_sigma/not_silence_preprocessed",
        train_args,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        further_training=True,
        spectrogram_foward=True,
    )
    tts_hdf = run_exp(
        net_module + "/enc768/with_sigma/silence_preprocessed",
        train_args_silence,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args["config"]["gradient_clip_norm"] = 10
    tts_hdf = run_exp(
        net_module + "/enc768/with_sigma/grad_clip_norm_10",
        train_args,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
    )

    train_args_betas = copy.deepcopy(train_args)
    train_args_betas["config"]["optimizer"] = {"class": "adam", "epsilon": 1e-9, "betas": (0.9, 0.98)}
    tts_hdf = run_exp(
        net_module + "/enc768/with_sigma/beta2_0.98",
        train_args_betas,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
    )

    train_args["net_args"]["filter_channels"] = 256
    tts_hdf = run_exp(
        net_module + "/enc256/not_silence_preprocessed",
        train_args,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
    )
    tts_hdf = run_exp(
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

    tts_hdf = run_exp(
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
    tts_hdf = run_exp(
        net_module + "/enc192/100ep/not_silence_preprocessed",
        train_args_alternative_lr,
        dataset=training_datasets,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    tts_hdf = run_exp(
        net_module + "/enc192/100ep/silence_preprocessed",
        train_args_alternative_lr,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_long_cooldown = copy.deepcopy(train_args_alternative_lr)
    train_args_long_cooldown["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(1e-5, 5e-4, 50), np.linspace(5e-4, 1e-5, 50), np.linspace(1e-5, 1e-7, 100)))
    )
    train_args_long_cooldown["config"]["optimizer"]["epsilon"] = 1e-8
    tts_hdf = run_exp(
        net_module + "/enc192/200ep/long_cooldown/not_silence_preprocessed",
        train_args_long_cooldown,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        durations_forward=True,
        latent_space_forward=True,
        train_data_forward=True,
        keep_epochs={100},
    )

    tts_hdf = run_exp(
        net_module + "/enc192/200ep/long_cooldown/silence_preprocessed",
        train_args_long_cooldown,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        keep_epochs={100},
    )

    train_args_fs4 = copy.deepcopy(train_args_long_cooldown)
    train_args_fs4["net_args"]["n_sqz"] = 4

    tts_hdf = run_exp(
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
    train_args_radam["config"]["learning_rate"] = 1e-4
    tts_hdf = run_exp(
        net_module + "/enc192/200ep/RAdam/lr1e-4/silence_preprocessed",
        train_args_radam,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_radam["config"]["learning_rate"] = 1e-5
    tts_hdf = run_exp(
        net_module + "/enc192/200ep/RAdam/lr1e-5/silence_preprocessed",
        train_args_radam,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_radam["config"]["learning_rate"] = 1e-6
    tts_hdf = run_exp(
        net_module + "/enc192/200ep/RAdam/lr1e-6/silence_preprocessed",
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

    tts_hdf = run_exp(
        net_module + "/not_silence_preprocessed",
        train_args_nar_taco_encoder,
        dataset=training_datasets,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        extra_evaluate_epoch=107,  # was to late so 100 was already deleted
    )
    tts_hdf = run_exp(
        net_module + "/silence_preprocessed",
        train_args_nar_taco_encoder,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=200,
        forward_args=forward_args,
        spectrogram_foward=True,
        extra_evaluate_epoch=100,
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

    tts_hdf = run_exp(
        net_module + "/silence_preprocessed",
        train_args_nar_taco_encoder_no_blstm,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    train_args_nar_taco_encoder_no_blstm["net_args"]["n_blocks_dec"] = 16

    tts_hdf = run_exp(
        net_module + "/16blocks/silence_preprocessed",
        train_args_nar_taco_encoder_no_blstm,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        durations_forward=True,
    )

    train_args_nar_taco_encoder_no_blstm_no_warmup = copy.deepcopy(train_args_nar_taco_encoder_no_blstm)
    train_args_nar_taco_encoder_no_blstm_no_warmup["config"]["learning_rates"] = list(
        np.concatenate((np.linspace(5e-4, 5e-4, 50), np.linspace(5e-4, 1e-6, 50)))
    )

    tts_hdf = run_exp(
        net_module + "/16blocks/radam_no_warmup/silence_preprocessed",
        train_args_nar_taco_encoder_no_blstm_no_warmup,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
    )

    net_module = "glowTTS_simple_encoder"
    train_args_simple_encoder = copy.deepcopy(train_args_nar_taco_encoder_no_blstm)
    train_args_simple_encoder["network_module"] = net_module
    train_args_simple_encoder["net_args"]["n_blocks_dec"] = 20

    tts_hdf = run_exp(
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

    tts_hdf = run_exp(
        net_module + "/silence_preprocessed",
        train_args_simple_encoder,
        dataset=training_datasets_silence_preprocessed,
        num_epochs=100,
        forward_args=forward_args,
        spectrogram_foward=True,
        extra_evaluate_epoch=67,
    )

    # net_module = "glowTTS_one_hot_encoder_mean"
    # train_args_one_hot_encoder = copy.deepcopy(train_args_simple_encoder)
    # train_args_one_hot_encoder["network_module"] = net_module

    # tts_hdf = run_exp(
    #     net_module + "/silence_preprocessed",
    #     train_args_one_hot_encoder,
    #     dataset=training_datasets_silence_preprocessed,
    #     num_epochs=100,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )

    # net_module = "glowTTS_one_hot_encoder_std"
    # train_args_one_hot_encoder["network_module"] = net_module

    # tts_hdf = run_exp(
    #     net_module + "/silence_preprocessed",
    #     train_args_one_hot_encoder,
    #     dataset=training_datasets_silence_preprocessed,
    #     num_epochs=100,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )

    # train_args_nar_taco_encoder["net_args"]["n_blocks_dec"] = 16
    # tts_hdf = run_exp(
    #     net_module + "_16blocks",
    #     train_args_nar_taco_encoder,
    #     dataset=training_datasets,
    #     num_epochs=200,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )
    # tts_hdf = run_exp(
    #     net_module + "_16_blocks_silence_preprocessed",
    #     train_args_nar_taco_encoder,
    #     dataset=training_datasets_silence_preprocessed,
    #     num_epochs=200,
    #     forward_args=forward_args,
    #     spectrogram_foward=True,
    # )

    return tts_hdf
