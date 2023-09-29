from sisyphus import tk

import copy
import numpy as np
from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from .pipeline import training, search

from .config import get_training_config, get_search_config

def conformer_baseline():
    BPE_SIZE = 2000
    prefix_name = "experiments/librispeech/librispeech_100_attention/standalone_pt_2023"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[(1, 5, 1000)],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        bpe_size=BPE_SIZE,
        preemphasis=None,
        settings=train_settings
    )

    # build testing datasets
    test_dataset_tuples = {}
    #for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
            bpe_size=BPE_SIZE,
            preemphasis=None
        )


        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function


    config = {

    }

    def run_exp(ft_name, datasets, train_args, search_args=None, do_search=False):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=250)

        #averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        #best_checkpoint = get_best_checkpoint(train_job)

        if do_search:
            returnn_search_config = get_search_config(**train_args, search_args=search_args)
            search(ft_name + "/default_250", returnn_search_config, train_job.out_checkpoints[250], test_dataset_tuples, RETURNN_EXE, MINI_RETURNN_ROOT)
        #search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    train_args = {
        "net_args":{},
        "network_module": "attention_basic_newinit_convfirst",
        "debug": True,
        "config": {},
    }
    run_exp(prefix_name + "/test", datasets=train_data, train_args=train_args)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_basic_newinit_convfirst",
        "debug": True,
        "config": {
            "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
            "batch_size": 30000 * 160,
        },

    }
    run_exp(prefix_name + "/test2", datasets=train_data, train_args=train_args)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_basic_newinit_convfirst_specaug_lesszone",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
            "batch_size": 30000 * 160,
        },

    }
    run_exp(prefix_name + "/test3", datasets=train_data, train_args=train_args)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_basic_newinit_convfirst_specaug",
        "debug": True,
        "config": {
            "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
            "batch_size": 30000 * 160,
        },

    }
    run_exp(prefix_name + "/test4", datasets=train_data, train_args=train_args, do_search=True)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_basic_fairseqinit_correctconvfirst_specaug_lesszone",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
            "batch_size": 30000 * 160,
        },

    }
    run_exp(prefix_name + "/test5", datasets=train_data, train_args=train_args)

    train_args = {
        "net_args":{},
        "network_module": "attention_basic_newinit_correctconvfirst_specaug_lesszone",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
            "batch_size": 30000 * 160,
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/test6", datasets=train_data, train_args=train_args)
    
    
    train_args = {
        "net_args":{},
        "network_module": "attention_basic_newinit_correctconvfirst_specaug_lesszone_speedtest",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
            "batch_size": 15000 * 160,
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/speed_test", datasets=train_data, train_args=train_args)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_basic_newinit_correctconvfirst_specaug_lesszone_speedtest",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 1e-4, 30)) + list(np.linspace(1e-4, 1e-6, 220)),
            "batch_size": 30000 * 160,
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/speed_test_v2", datasets=train_data, train_args=train_args)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_basic_newinit_correctconvfirst_specaug_lesszone_speedtest",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 8e-4, 50)) + list(np.linspace(8e-4, 1e-6, 200)),
            "batch_size": 30000 * 160,
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/speed_test_v3", datasets=train_data, train_args=train_args)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_somewhat_tflike_pretraintest",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 8e-4, 50)) + list(np.linspace(8e-4, 1e-6, 200)),
            "batch_size": 30000 * 160,
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/pretrain_test", datasets=train_data, train_args=train_args)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_somewhat_tflike_pretraintest",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-4, 8e-4, 50)) + list(np.linspace(8e-4, 1e-6, 200)),
            "batch_size": 30000 * 160,
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/pretrain_test_v2", datasets=train_data, train_args=train_args)
    
    train_args = {
        "net_args":{},
        "network_module": "attention_somewhat_tflike_pretraintest",
        "debug": False,
        "config": {
            "learning_rates": ([1e-4] * 15) + list(np.linspace(1e-4, 8e-4, 35)) + list(np.linspace(8e-4, 1e-6, 200)),
            "batch_size": 30000 * 160,
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/pretrain_test_v3", datasets=train_data, train_args=train_args)


    train_args = {
        "net_args":{},
        "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1",
        "debug": False,
        "config": {
            "learning_rates": ([1e-4] * 50) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
            "batch_size": 30000 * 160,
            "max_seq_length": {"audio_features": 3000 * 160},
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1", datasets=train_data, train_args=train_args)

    train_args = {
        "net_args":{},
        "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 1e-4, 50)) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
            "batch_size": 25000 * 160,
            "max_seq_length": {"audio_features": 3000 * 160},
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_morewarmup", datasets=train_data, train_args=train_args)

    train_args = {
        "net_args":{},
        "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1_pretrain",
        "debug": False,
        "config": {
            "learning_rates": ([1e-4] * 50) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
            "batch_size": 30000 * 160,
            "max_seq_length": {"audio_features": 3000 * 160},
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_pretrain", datasets=train_data, train_args=train_args)
    
    
    train_args = {
        "net_args":{},
        "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1_nozoneout",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 1e-4, 50)) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
            "batch_size": 25000 * 160,
            "max_seq_length": {"audio_features": 3000 * 160},
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_nozoneout_morewarmup", datasets=train_data, train_args=train_args)


    train_args = {
        "net_args":{},
        "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1_noattdrop",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 1e-4, 50)) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
            "batch_size": 25000 * 160,
            "max_seq_length": {"audio_features": 3000 * 160},
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_noattdrop_morewarmup", datasets=train_data, train_args=train_args)
    
    
    train_args = {
        "net_args":{},
        "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1_nozoneout",
        "debug": False,
        "config": {
            "learning_rates": list(np.linspace(1e-5, 1e-4, 50)) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
            "batch_size": 25000 * 160,
            "max_seq_length": {"audio_features": 3000 * 160},
            "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-6},
        },
        "use_v3": True,
    }
    run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_nozoneout_morewarmup_adamw", datasets=train_data, train_args=train_args)