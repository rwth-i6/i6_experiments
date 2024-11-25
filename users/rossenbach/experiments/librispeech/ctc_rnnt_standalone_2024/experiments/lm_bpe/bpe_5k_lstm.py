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


def bpe5k_kazuki_lstm():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/bpe_5k_lstm"

    train_settings_part100 = LMDatasetSettings(
        train_partition_epoch=100,
        train_seq_ordering="laplace:.100",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe5k_part100 = build_lm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=5000,
        settings=train_settings_part100,
    )
    
    label_datastream_bpe5000 = cast(LabelDatastream, train_data_bpe5k_part100.datastreams["data"])
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
    
    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 1280,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "learning_rates": ([1e-3] * 50) + list(np.linspace(1e-3, 1e-5, 150))
    }

    network_module = "lm.lstm.kazuki_lstm_zijian_variant_v1"
    train_args = {
        "config": train_config_modern_v1,
        "post_config": {"num_workers_per_gpu": 1},
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(lstm_base_config)},
        "debug": True,
        "add_cache_manager": True,
    }

    training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3_2ep_reduce_gcn1.0"
    train_job = training(training_name, train_data_bpe5k_part100, train_args, num_epochs=200, **lm_returnn)#

    from ...storage import add_lm
    from ...pipeline import NeuralLM
    add_lm("bpe5k_2x2024_kazuki_lstmlm_2ep", lm_model=NeuralLM(
        checkpoint=train_job.out_checkpoints[200],
        net_args=train_args["net_args"],
        network_module=network_module,
        prefix_name=training_name
    ))

    train_config_modern_v1 = {
        "optimizer": {"class": "RAdam"},
        #############
        "batch_size": 1280,  # BPE tokens
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "learning_rates": ([1e-3] * 100) + list(np.linspace(1e-3, 1e-5, 200))
    }

    network_module = "lm.lstm.kazuki_lstm_zijian_variant_v1"
    train_args = {
        "config": train_config_modern_v1,
        "post_config": {"num_workers_per_gpu": 1},
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(lstm_base_config)},
        "debug": True,
        "add_cache_manager": True,
    }

    training_name = prefix_name + "/" + network_module + ".2x2024_1k_RAdam_1e-3_3ep_reduce_gcn1.0"
    train_job = training(training_name, train_data_bpe5k_part100, train_args, num_epochs=300, **lm_returnn)#

    from ...storage import add_lm
    from ...pipeline import NeuralLM
    add_lm("bpe5k_2x2024_kazuki_lstmlm_3ep", lm_model=NeuralLM(
        checkpoint=train_job.out_checkpoints[300],
        net_args=train_args["net_args"],
        network_module=network_module,
        prefix_name=training_name,
        bpe_vocab=label_datastream_bpe5000.vocab,
        bpe_codes=label_datastream_bpe5000.codes,
    ))