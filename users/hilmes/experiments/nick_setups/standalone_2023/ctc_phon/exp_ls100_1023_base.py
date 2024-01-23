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

from ..config import get_training_config, get_search_config, get_prior_config


def eow_phon_ls100_1023_base():
    prefix_name = "experiments/librispeech/standalone_2023/ls100_ctc_eow_phon/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000",
        preemphasis=0.97,
        peak_normalization=True,  # TODO: this is wrong compared to old setupsa and rescale, better test if it degrades
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        librispeech_key="train-clean-100",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev", "test"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            preemphasis=train_settings.preemphasis,
            peak_normalization=train_settings.peak_normalization,
        )

    arpa_4gram_lm = get_4gram_binary_lm()

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp(
        ft_name,
        datasets,
        train_args,
        search_args=None,
        with_prior=False,
        num_epochs=250,
        decoder="ctc.decoder.flashlight_phoneme_ctc",
    ):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if with_prior:
            returnn_config = get_prior_config(training_datasets=datasets, **train_args)
            prior_file = compute_prior(
                ft_name,
                returnn_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(training_name + "/prior.txt", prior_file)
            search_args["prior_file"] = prior_file

        returnn_search_config = get_search_config(**train_args, decoder_args=search_args, decoder=decoder)

        _, _, search_jobs = search(
            ft_name + "/last_%i" % num_epochs,
            returnn_search_config,
            train_job.out_checkpoints[num_epochs],
            test_dataset_tuples,
            RETURNN_EXE,
            MINI_RETURNN_ROOT,
        )

        return train_job, search_jobs

    from ..pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
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
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
            + list(np.linspace(7e-4, 7e-5, 110))
            + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
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
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name
                + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_gc1/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args_gc1,
                search_args=search_args,
                with_prior=True,
            )

    train_args_decay1e_2 = copy.deepcopy(train_args)
    train_args_decay1e_2["config"]["optimizer"]["weight_decay"] = 1e-2
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name
                + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_decay1e-2/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args_decay1e_2,
                search_args=search_args,
                with_prior=True,
            )

    search_args = {
        **default_search_args,
        "lm_weight": 3.5,
        "prior_scale": 0.3,
        "sil_score": -1000.0,
    }
    run_exp(
        prefix_name
        + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_decay1e-2/lm_test1_bs1024_th14",
        datasets=train_data,
        train_args=train_args_decay1e_2,
        search_args=search_args,
        with_prior=True,
        decoder="ctc.decoder.flashlight_phoneme_ctc_v2",
    )

    search_args = {
        "lexicon": get_text_lexicon(librispeech_key="train-clean-100"),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 1024,
        "arpa_lm": arpa_4gram_lm,
        "beam_threshold": 16,
        "lm_weight": 3.5,
        "prior_scale": 0.3,
    }
    run_exp(
        prefix_name
        + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_JJLR_peaknorm_decay1e-2/lm_test2_bs1024_th16",
        datasets=train_data,
        train_args=train_args_decay1e_2,
        search_args=search_args,
        with_prior=True,
        decoder="ctc.decoder.flashlight_phoneme_ctc",
    )

    ###### trying to reproduce 14.5% result from librispeech/librispeech_100_phon_ctc #########

    train_args_adamw_02 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }
    model_config_small_ff = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=384,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=1,
    )
    train_args = {
        **copy.deepcopy(train_args_adamw_02),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config_small_ff)},
    }
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            # TODO: add num_epochs 300
            run_exp(
                prefix_name
                + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_legacy_peaknorm_decay1e-2_FF384_accum1/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
            )

    train_args = {
        **copy.deepcopy(train_args_adamw_02),
        "network_module": "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6",
        "net_args": {"model_config_dict": asdict(model_config_small_ff)},
    }
    train_args["config"]["accum_grad_multiple_step"] = 2
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.0, 0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            train_job, _ = run_exp(
                prefix_name
                + "conformer_1023/i6modelsV1_VGG4LayerActFrontendV1_v6_legacy_peaknorm_decay1e-2_FF384_accum2/lm%.1f_prior%.2f_bs1024_th14"
                % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                num_epochs=300,
            )
