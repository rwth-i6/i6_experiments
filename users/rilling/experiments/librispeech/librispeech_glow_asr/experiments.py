from sisyphus import tk

from dataclasses import asdict

import numpy as np

import copy
from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset, get_text_lexicon, get_arpa_lm
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from .pipeline import training, search

from .config import get_training_config, get_search_config

def glowASR():
    prefix_name = "experiments/librispeech/librispeech_glow_asr/pytorch/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=3, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets("train-clean-100", settings=train_settings)

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )

        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
    label_datastream = cast(LabelDatastream, train_data.datastreams["phon_labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    from .data import get_tts_log_mel_datastream

    log_mel_datastream = get_tts_log_mel_datastream(silence_preprocessing=False)

    from .pytorch_networks.shared.configs import DbMelFeatureExtractionConfig
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import DBMelFilterbankOptions

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

    config = {}

    def run_exp(ft_name, datasets, train_args, search_args=None, num_epochs=100):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        returnn_search_config = get_search_config(**train_args, search_args=search_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        # averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        # best_checkpoint = get_best_checkpoint(train_job)

        search(
            ft_name + "/default_250",
            returnn_search_config,
            train_job.out_checkpoints[num_epochs],
            test_dataset_tuples,
            RETURNN_EXE,
            MINI_RETURNN_ROOT,
        )
        # search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        # search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    train_args = {
        "net_args": {
            "n_vocab": vocab_size_without_blank,
            "gin_channels": 256,
            "fe_config": asdict(fe_config)
        },
        "network_module": "basic_glowASR_linear",
        "debug": True,
        "config": {
            "preload_from_files": {
                "existing-model": {
                    "filename": "/u/lukas.rilling/experiments/glow_tts_asr_v2/alias/experiments/librispeech/tts_architecture/glow_tts/pytorch/glowTTS_warmup/training/output/models/epoch.100.pt",
                    "init_for_train": True,
                    "ignore_params_prefixes": ["encoder"],
                    "ignore_missing": True 
                }
            }
        },
    }

    default_search_args = {
        "lexicon": get_text_lexicon(),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 64,
        "arpa_lm": get_arpa_lm(),
        "lm_weight": 5,
        "beam_threshold": 50
    }
    # run_exp(prefix_name + "test", datasets=train_data, train_args=train_args)

    train_args2 = copy.deepcopy(train_args)
    train_args2["net_args"]["final_n_layers"] = 5
    train_args2["net_args"]["final_hidden_channels"] = 512

    # run_exp(prefix_name + "test2", datasets=train_data, train_args=train_args2, search_args=default_search_args)

    # train_args2["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))
    # run_exp(prefix_name + "test3", datasets=train_data, train_args=train_args2)

    train_args_conv = copy.deepcopy(train_args)
    train_args_conv["network_module"] = "basic_glowASR_conv"
    train_args_conv["net_args"]["final_n_layers"] = 4
    train_args_conv["net_args"]["final_hidden_channels"] = 512
    
    # run_exp(prefix_name + "test_conv", datasets=train_data, train_args=train_args_conv, search_args=default_search_args)

    train_args_conv["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))

    # run_exp(prefix_name + "test_conv_warmup", datasets=train_data, train_args=train_args_conv, search_args=default_search_args)

    # train_args_conv2 = copy.deepcopy(train_args)
    # train_args_conv2["network_module"] = "basic_glowASR_conv_v2"
    # train_args_conv2["net_args"]["final_n_layers"] = 4
    
    # run_exp(prefix_name + "test_conv2", datasets=train_data, train_args=train_args_conv2)

    train_args_blstm = copy.deepcopy(train_args)
    train_args_blstm["network_module"] = "basic_glowASR_blstm"
    train_args_blstm["net_args"]["final_n_layers"] = 4
    
    # run_exp(prefix_name + "test_blstm", datasets=train_data, train_args=train_args_blstm, search_args=default_search_args)

    train_args_blstm2 = copy.deepcopy(train_args_blstm)
    train_args_blstm2["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))
    # run_exp(prefix_name + "test_blstm_warmup", datasets=train_data, train_args=train_args_blstm2, search_args=default_search_args)

    train_args_blstm3 = copy.deepcopy(train_args_blstm)
    train_args_blstm3["net_args"]["final_hidden_channels"] = 512
    # run_exp(prefix_name + "test_blstm512", datasets=train_data, train_args=train_args_blstm3)

    train_args_blstm4 = copy.deepcopy(train_args_blstm3)
    train_args_blstm4["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))

    # train_args_blstm["config"]["learning_rates"] = list(np.concatenate((np.linspace(1e-5, 5*1e-4, 50), np.linspace(5*1e-4, 1e-5, 50))))
    run_exp(prefix_name + "test_blstm512_warmup", datasets=train_data, train_args=train_args_blstm4, search_args=default_search_args)





