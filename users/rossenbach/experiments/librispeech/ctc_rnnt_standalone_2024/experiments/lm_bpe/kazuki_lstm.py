from sisyphus import tk

from dataclasses import asdict
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.bpe_lm import build_lm_training_datasets, LMDatasetSettings
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pipeline import training


def bpe_kazuki_lstm():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/kazuki_lstm/"

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
    label_datastream_bpe5000 = cast(LabelDatastream, train_data_bpe10k.datastreams["data"])
    vocab_size_without_blank = label_datastream_bpe5000.vocab_size

    default_returnn = {
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

    train_config_24gbgpu = {
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
        "config": train_config_24gbgpu,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(lstm_base_config)},
        "debug": False,
        "add_cache_manager": True,
    }

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps"
    train_job = training(training_name, train_data_bpe10k, train_args, num_epochs=30, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
