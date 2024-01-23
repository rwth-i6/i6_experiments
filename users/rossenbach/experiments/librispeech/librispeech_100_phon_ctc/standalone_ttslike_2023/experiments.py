from sisyphus import tk

from dataclasses import asdict
import numpy as np

import copy
from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from .pipeline import training, search

from .config import get_training_config, get_search_config

def conformer_baseline():
    prefix_name = "experiments/librispeech/librispeech_100_phon_ctc/standalone_ttslike_2023"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        settings=train_settings
    )

    # build testing datasets
    test_dataset_tuples = {}
    #for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )


        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function


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
        norm=norm
    )
    
    config = {
    }


    from .data import get_lexicon
    lexicon = get_lexicon(with_blank=True, with_g2p=False)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon = BlissLexiconToWordLexicon(lexicon).out_lexicon

    def run_exp(ft_name, datasets, train_args, search_args=None):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        returnn_search_config = get_search_config(**train_args, search_args=search_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=250)

        #averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        #best_checkpoint = get_best_checkpoint(train_job)

        search(ft_name + "/default_250", returnn_search_config, train_job.out_checkpoints[250], test_dataset_tuples, RETURNN_EXE, MINI_RETURNN_ROOT)
        #search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    train_args = {
        "net_args":{"fe_config": asdict(fe_config)},
        "network_module": "small_conformer_static_searchtest_08_23",
        "debug": True,
        "config": {
            "learning_rates": [0.0001],
         },
    }

    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
    search_args = {
        "lexicon": word_lexicon,
        "arpa_lm": get_arpa_lm_dict()["4gram"]
    }

    run_exp(prefix_name + "/test_small", datasets=train_data, train_args=train_args, search_args=search_args)


    train_args = {
        "net_args":{"fe_config": asdict(fe_config)},
        "network_module": "basic_conformer_static_searchtest_08_23",
        "debug": True,
        "config": {},
    }


    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
    search_args = {
        "lexicon": word_lexicon,
        "arpa_lm": get_arpa_lm_dict()["4gram"],
        "lm_weight": 5,
        "beam_size": 50,
    }
    # failed
    # run_exp(prefix_name + "/test_medium_base1", datasets=train_data, train_args=train_args, search_args=search_args)
    for beam_size in [60, 100, 200, 500, 1000]:
    #for beam_size in [60, 100, 200, 500, 1000, 2000]:  # 2000 failed  # 2000 failed
        search_args_ = copy.deepcopy(search_args)
        search_args_["beam_size"] = beam_size
        run_exp(prefix_name + "/test_medium_lm5_bs%i" % beam_size, datasets=train_data, train_args=train_args, search_args=search_args_)

    for lm_weight in [1, 2, 3, 4, 6, 7, 8]:
        search_args_ = copy.deepcopy(search_args)
        search_args_["beam_size"] = 500
        search_args_["lm_weight"] = lm_weight
        run_exp(prefix_name + "/test_medium_lm%i_bs500" % lm_weight, datasets=train_data, train_args=train_args, search_args=search_args_)

    train_args = {
        "net_args":{"fe_config": asdict(fe_config)},
        "network_module": "large_conformer_static_searchtest_08_23",
        "debug": True,
        "config": {},
    }
    search_args = {
        "lexicon": word_lexicon,
        "arpa_lm": get_arpa_lm_dict()["4gram"],
        "lm_weight": 5,
        "beam_size": 50,
    }
    # Failed
    # run_exp(prefix_name + "/test_large_base1", datasets=train_data, train_args=train_args, search_args=search_args)
    
    
    
    from .data import get_lexicon
    lexicon_no_silence = get_lexicon(with_blank=True, with_g2p=False, add_silence=False)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon_no_silence = BlissLexiconToWordLexicon(lexicon_no_silence).out_lexicon

    from typing import cast
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
    label_datastream = cast(LabelDatastream, train_data.datastreams["phon_labels"])
    
    train_args = {
        "net_args":{"fe_config": asdict(fe_config)},
        "network_module": "i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "config": {
            "accum_grad_multiple_step": 2,
            "learning_rates": list(np.linspace(1e-6, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            "batch_size": 300 * 16000,
        }
    }
    search_args = {
        "lexicon": word_lexicon_no_silence,
        "returnn_vocab": label_datastream.vocab,
        "arpa_lm": get_arpa_lm_dict()["4gram"],
        "lm_weight": 5,
        "beam_size": 1024,
        "beam_threshold": 16,
    }
    run_exp(prefix_name + "/i6modelsV1_VGG4LayerActFrontendV1", datasets=train_data, train_args=train_args, search_args=search_args)


def conformer_baseline_no_spp():
    prefix_name = "experiments/librispeech/librispeech_100_phon_ctc/standalone_ttslike_2023_nospp"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        settings=train_settings
    )

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
        norm=norm
    )

    config = {
    }

    from .data import get_lexicon
    lexicon = get_lexicon(with_blank=True, with_g2p=False)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon = BlissLexiconToWordLexicon(lexicon).out_lexicon

    def run_exp(ft_name, datasets, train_args, search_args=None):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        returnn_search_config = get_search_config(**train_args, search_args=search_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=250)

        # averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        # best_checkpoint = get_best_checkpoint(train_job)

        search(ft_name + "/default_250", returnn_search_config, train_job.out_checkpoints[250], test_dataset_tuples,
               RETURNN_EXE, MINI_RETURNN_ROOT)
        # search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        # search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    train_args = {
        "net_args": {"fe_config": asdict(fe_config)},
        "network_module": "small_conformer_static_searchtest_08_23",
        "debug": True,
        "config": {
            "learning_rates": [0.0001],
        },
    }

    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
    search_args = {
        "lexicon": word_lexicon,
        "arpa_lm": get_arpa_lm_dict()["4gram"]
    }

    # run_exp(prefix_name + "/test_small", datasets=train_data, train_args=train_args, search_args=search_args)

    train_args = {
        "net_args": {"fe_config": asdict(fe_config)},
        "network_module": "basic_conformer_static_searchtest_08_23",
        "debug": True,
        "config": {},
    }

    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
    search_args = {
        "lexicon": word_lexicon,
        "arpa_lm": get_arpa_lm_dict()["4gram"],
        "lm_weight": 5,
        "beam_size": 50,
    }
    # run_exp(prefix_name + "/test_medium_base1", datasets=train_data, train_args=train_args, search_args=search_args)
    for beam_size in [60, 100, 200, 500, 1000]:
        search_args_ = copy.deepcopy(search_args)
        search_args_["beam_size"] = beam_size
        run_exp(prefix_name + "/test_medium_lm5_bs%i" % beam_size, datasets=train_data, train_args=train_args,
                search_args=search_args_)

    for lm_weight in [1, 2, 3, 4, 6, 7, 8]:
        search_args_ = copy.deepcopy(search_args)
        search_args_["beam_size"] = 500
        search_args_["lm_weight"] = lm_weight
        run_exp(prefix_name + "/test_medium_lm%i_bs500" % lm_weight, datasets=train_data, train_args=train_args,
                search_args=search_args_)

    train_args = {
        "net_args": {"fe_config": asdict(fe_config)},
        "network_module": "large_conformer_static_searchtest_08_23",
        "debug": True,
        "config": {},
    }
    search_args = {
        "lexicon": word_lexicon,
        "arpa_lm": get_arpa_lm_dict()["4gram"],
        "lm_weight": 5,
        "beam_size": 50,
    }
    run_exp(prefix_name + "/test_large_base1", datasets=train_data, train_args=train_args, search_args=search_args)
