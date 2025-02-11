from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.bpe_lm import build_lm_training_datasets, LMDatasetSettings
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pipeline import training


def bpe_kazuki_lstm():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/kazuki_lstm"

    train_settings = LMDatasetSettings(
        train_partition_epoch=4,
        train_seq_ordering="laplace:.100",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe10k = build_lm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=10000,
        settings=train_settings,
    )
    
    train_settings_part100 = LMDatasetSettings(
        train_partition_epoch=100,
        train_seq_ordering="laplace:.100",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe10k_part100 = build_lm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=10000,
        settings=train_settings_part100,
    )
    
    label_datastream_bpe5000 = cast(LabelDatastream, train_data_bpe10k.datastreams["data"])
    vocab_size_without_blank = label_datastream_bpe5000.vocab_size


    # Extra version to debug LM dataset behavior
    MINI_RETURNN_ROOT = CloneGitRepositoryJob(
        "https://github.com/JackTemaki/MiniReturnn", commit="ac8df606d62a3474f5c8a1fb0ff11adb54bb75c6"
    ).out_repository.copy()
    MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

    lm_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v1_cfg import ModelConfig

    default_init_args = {
        'init_args_w': {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}},
        'init_args_b': {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}}
    }

    lstm_base_config = ModelConfig(
        vocab_dim=vocab_size_without_blank,
        embed_dim=512,
        hidden_dim=2048,
        n_lstm_layers=2,
        use_bottle_neck=False,
        dropout=0.2,
        init_args=default_init_args,
    )
    
    lstm_default_init_config = ModelConfig(
        vocab_dim=vocab_size_without_blank,
        embed_dim=512,
        hidden_dim=2048,
        n_lstm_layers=2,
        use_bottle_neck=False,
        dropout=0.2,
        init_args=None,
    )

    train_config_legacy = {
        "optimizer": {"class": "SGD"},
        #############
        "batch_size": 1280,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "learning_rate": 1.0,
        "decay": 0.8,
        "multi_num_epochs": train_settings.train_partition_epoch,
        "relative_error_threshold": 0,
        "multi_update_interval": 1,
        "error_measure": "dev_ce",
    }

    network_module = "lm.lstm.kazuki_lstm_zijian_variant_v1"
    train_args = {
        "config": train_config_legacy,
        "post_config": {"num_workers_per_gpu": 1},
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(lstm_base_config)},
        "debug": True,
        "add_cache_manager": True,
    }

    # Worked, but bad without reduction
    # training_name = prefix_name + "/" + network_module + ".2x2024_legacy_schedule"
    # train_job = training(training_name, train_data_bpe10k, train_args, num_epochs=30, **lm_returnn)
    # train_job.rqmt["gpu_mem"] = 11


    # What was this idea? Of course super bad...
    # train_config_legacy_const = {
    #     "optimizer": {"class": "SGD"},
    #     #############
    #     "batch_size": 4000,  # BPE tokens
    #     "accum_grad_multiple_step": 1,
    #     "learning_rates": [1.0],
    # }
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_legacy_const
    # training_name = prefix_name + "/" + network_module + ".2x2024_legacy_schedule_const"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)

    # Same here, bad idea
    # train_config_legacy_const_gradnorm = {
    #     "optimizer": {"class": "SGD"},
    #     #############
    #     "batch_size": 4000,  # BPE tokens
    #     "accum_grad_multiple_step": 1,
    #     "gradient_clip_norm": 2.0,
    #     "learning_rates": [1.0],
    # }
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_legacy_const_gradnorm
    # training_name = prefix_name + "/" + network_module + ".2x2024_legacy_schedule_const_gradnorm"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)

    # Also here
    # train_config_legacy_const_gradnorm = {
    #     "optimizer": {"class": "SGD"},
    #     #############
    #     "batch_size": 4000,  # BPE tokens
    #     "accum_grad_multiple_step": 1,
    #     "gradient_clip_norm": 0.002,
    #     "learning_rates": [1.0],
    # }
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_legacy_const_gradnorm
    # training_name = prefix_name + "/" + network_module + ".2x2024_legacy_schedule_const_gradnorm_strong"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)


    # Bad
    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 4000,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "learning_rates": [1e-4],
    }
    #train_args_modern_v1 = copy.deepcopy(train_args)
    #train_args_modern_v1["config"] = train_config_modern_v1
    #training_name = prefix_name + "/" + network_module + ".2x2024_RAdam_1e-4"
    #train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)


    # Bad
    # train_config_modern_v1 = {
    #     "optimizer": {"class": "RAdam"},
    #     #############
    #     "batch_size": 4000,  # BPE tokens
    #     "accum_grad_multiple_step": 1,
    #     "learning_rates": [1e-3],
    # }
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_modern_v1
    # training_name = prefix_name + "/" + network_module + ".2x2024_4k_RAdam_1e-3"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)

    # Bad
    # train_config_modern_v1 = {
    #     "optimizer": {"class": "RAdam", "epsilon": 1e-12},
    #     #############
    #     "batch_size": 4000,  # BPE tokens
    #     "accum_grad_multiple_step": 1,
    #     "learning_rates": [1e-3],
    # }
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_modern_v1
    # training_name = prefix_name + "/" + network_module + ".2x2024_4k_RAdam_1e-3_eps-12"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)
    
    
    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 1280,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "learning_rates": [1e-3],
    }

    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_modern_v1
    # training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)
    
    # With default init -> bad
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_modern_v1
    # train_args_modern_v1["net_args"] = {"model_config_dict": asdict(lstm_default_init_config)}
    # training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3_default_init"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)
    
    
    # with LR reduction and gradient clipping test
    # This one is the best so far with PPL, 57.5 use it as reference for longer training
    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 1280,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "learning_rates": ([1e-3] * 50) + list(np.linspace(1e-3, 1e-5, 50))
    }
    # Inf / Nan loss in 199, skip update fix was not applied yet
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_modern_v1
    # training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3_1ep_reduce_gcn1.0"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=100, **lm_returnn)

    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 1280,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "learning_rates": ([1e-3] * 50) + list(np.linspace(1e-3, 1e-5, 150))
    }
    train_args_modern_v1 = copy.deepcopy(train_args)
    train_args_modern_v1["config"] = train_config_modern_v1
    training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3_2ep_reduce_gcn1.0"
    train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=200, **lm_returnn)#

    from ...storage import add_lm
    from ...pipeline import NeuralLM
    add_lm("2x2024_kazuki_lstmlm_2ep", lm_model=NeuralLM(
        checkpoint = train_job.out_checkpoints[200],
        net_args = train_args_modern_v1["net_args"],
        network_module = network_module,
        prefix_name = training_name
    ))


    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 1280,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "learning_rates": ([1e-3] * 50) + list(np.linspace(1e-3, 1e-5, 250))
    }
    train_args_modern_v1 = copy.deepcopy(train_args)
    train_args_modern_v1["config"] = train_config_modern_v1
    training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3_3ep_reduce_gcn1.0"

    # never completed, not relevant so remove
    #train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=300, **lm_returnn)

    # with LR reduction and gradient clipping test
    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 1280,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 0.5,
        "learning_rates": ([1e-3] * 50) + list(np.linspace(1e-3, 1e-5, 50))
    }
    train_args_modern_v1 = copy.deepcopy(train_args)
    train_args_modern_v1["config"] = train_config_modern_v1
    training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3_1ep_reduce_gcn0.5"
    train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=100, **lm_returnn)


    # with larger batch and LR
    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 2560,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 0.5,
        "learning_rates": ([1.5e-3] * 50) + list(np.linspace(1.5e-3, 1e-5, 50))
    }
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_modern_v1
    # training_name = prefix_name + "/" + network_module + ".2x2024_2k_RAdam_1.5e-3_1ep_reduce_gcn0.5"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=100, **lm_returnn)


    # No relevant change
    # train_config_modern_v1 = {
    #     "optimizer": {"class": "RAdam", "epsilon": 1e-12},
    #     #############
    #     "batch_size": 1280,  # BPE tokens
    #     "accum_grad_multiple_step": 1,
    #     "learning_rates": [1e-3],
    # }
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_modern_v1
    # training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3_eps-12"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=500, **lm_returnn)

    # constant with 2e-3 had inf/nan loss

    # Not stable, constant in the beginning better
    # train_config_modern_v1 = {
    #     "optimizer": {"class": "RAdam"},
    #     #############
    #     "batch_size": 1280,  # BPE tokens
    #     "accum_grad_multiple_step": 1,
    #     "learning_rates": [1e-3] + list(np.linspace(2e-3,1e-5, 199))
    # }
    # train_args_modern_v1 = copy.deepcopy(train_args)
    # train_args_modern_v1["config"] = train_config_modern_v1
    # training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_reduce_short"
    # train_job = training(training_name, train_data_bpe10k_part100, train_args_modern_v1, num_epochs=200, **lm_returnn)
