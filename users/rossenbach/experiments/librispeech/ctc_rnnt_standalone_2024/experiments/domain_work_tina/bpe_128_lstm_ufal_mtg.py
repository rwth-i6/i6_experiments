from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.bpe_lm import build_lm_training_datasets, build_lm_custom_training_datasets, LMDatasetSettings
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pipeline import training


def ufal_mtg_lstm_lm():

    for BPE_SIZE in [128, 512, 5000]:
        prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/domain_test/bpe_%i_lstm" % BPE_SIZE

        ufal_medical_version_1_text = tk.Path(
            "/work/asr4/rossenbach/domain_data/UFAL_medical_shuffled/clean_version_1/all_en_medical_uniq_sorted_final.txt.gz",
            hash_overwrite="UFAL_medical_shuffled/clean_version_1/all_en_medical_uniq_sorted_final.txt.gz"
        )
        # this is for cv
        wmt22_medline_v1 = tk.Path(
            "/work/asr4/rossenbach/domain_data/wmt_medline_test_data/wmt22_medline_v1.txt",
            hash_overwrite="wmt22_medline_v1.txt"
        )

        MTG_trial4_train = tk.Path(
            "/work/asr4/rossenbach/domain_data/MTG/MTG_trial4_train.txt",
            hash_overwrite="MTG/MTG_trial4_train.txt"
        )
        # this is for cv
        MTG_trial4_dev = tk.Path(
            "/work/asr4/rossenbach/domain_data/MTG/MTG_trial4_dev.txt",
            hash_overwrite="MTG/MTG_trial4_dev.txt"
        )

        train_settings_part100 = LMDatasetSettings(
            train_partition_epoch=25,
            train_seq_ordering="laplace:.100",
        )
        train_settings_part25 = LMDatasetSettings(
            train_partition_epoch=25,
            train_seq_ordering="laplace:.100",
        )
        train_settings_part1 = LMDatasetSettings(
            train_partition_epoch=1,
            train_seq_ordering="laplace:.100",
        )
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_ls_part100 = build_lm_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings_part100,
        )

        train_data_ufal_part25 = build_lm_custom_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            train_text=ufal_medical_version_1_text,
            dev_text=wmt22_medline_v1,
            settings=train_settings_part25,
        )

        train_data_MTG_part1 = build_lm_custom_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            train_text=MTG_trial4_train,
            dev_text=MTG_trial4_dev,
            settings=train_settings_part1,
        )

        label_datastream = cast(LabelDatastream, train_data_ls_part100.datastreams["data"])
        vocab_size_without_blank = label_datastream.vocab_size


        # Extra version to debug LM dataset behavior
        MINI_RETURNN_ROOT = CloneGitRepositoryJob(
            "https://github.com/JackTemaki/MiniReturnn", commit="ac8df606d62a3474f5c8a1fb0ff11adb54bb75c6"
        ).out_repository.copy()
        MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

        lm_returnn = {
            "returnn_exe": RETURNN_EXE,
            "returnn_root": MINI_RETURNN_ROOT,
        }

        from ...pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v3_cfg import ModelConfig

        default_init_args = {
            'init_args_w': {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}},
            'init_args_b': {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}}
        }

        lstm_base_config = ModelConfig(
            vocab_dim=vocab_size_without_blank,
            embed_dim=128,
            hidden_dim=1024,
            n_lstm_layers=2,
            use_bottle_neck=False,
            lstm_dropout=0.05,
            output_dropout=0.2,
            init_args=default_init_args,
        )


        train_config_modern_v1 = {
            "optimizer": {"class": "RAdam"},
            #############
            "batch_size": 1700,  # BPE tokens
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "learning_rates": ([1e-3] * 50) + list(np.linspace(1e-3, 1e-5, 250))
        }

        network_module = "lm.lstm.kazuki_lstm_zijian_variant_v3"
        train_args = {
            "config": train_config_modern_v1,
            "post_config": {"num_workers_per_gpu": 1},
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(lstm_base_config)},
            "debug": True,
            "add_cache_manager": True,
        }

        training_name = prefix_name + "/" + network_module + ".2x1024_RAdam_1e-3_3p_reduce_gcn1.0"
        train_job = training(training_name, train_data_ls_part100, train_args, num_epochs=300, **lm_returnn)

        from ...storage import add_lm
        from ...pipeline import NeuralLM
        add_lm("bpe%i_2x1024_kazuki_lstmlm_3ep" % BPE_SIZE, lm_model=NeuralLM(
            checkpoint=train_job.out_checkpoints[300],
            net_args=train_args["net_args"],
            network_module=network_module,
            prefix_name=training_name
        ))

        training_name = prefix_name + "/ufal_" + network_module + ".2x1024_RAdam_1e-3_12ep_reduce_gcn1.0"
        train_job = training(training_name, train_data_ufal_part25, train_args, num_epochs=300, **lm_returnn)

        add_lm("ufal_bpe%i_2x1024_kazuki_lstmlm_12ep" % BPE_SIZE, lm_model=NeuralLM(
            checkpoint=train_job.out_checkpoints[300],
            net_args=train_args["net_args"],
            network_module=network_module,
            prefix_name=training_name
        ))

        # MTG
        lstm_high_dropout_config = ModelConfig(
            vocab_dim=vocab_size_without_blank,
            embed_dim=128,
            hidden_dim=1024,
            n_lstm_layers=2,
            use_bottle_neck=False,
            lstm_dropout=0.5,
            output_dropout=0.5,
            init_args=default_init_args,
        )

        train_config_modern_v1_100ep = {
            "optimizer": {"class": "RAdam"},
            #############
            "batch_size": 1700,  # BPE tokens
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "learning_rates": ([1e-3] * 50) + list(np.linspace(1e-3, 1e-5, 50))
        }

        train_args = {
            "config": train_config_modern_v1_100ep,
            "post_config": {"num_workers_per_gpu": 1},
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(lstm_high_dropout_config)},
            "debug": True,
            "add_cache_manager": True,
        }
        training_name = prefix_name + "/mtg_" + network_module + ".2x1024_RAdam_1e-3_100ep_reduce_gcn1.0"
        train_job = training(training_name, train_data_MTG_part1, train_args, num_epochs=100, **lm_returnn)

        add_lm("mtg_bpe%i_2x1024_kazuki_lstmlm_100ep" % BPE_SIZE, lm_model=NeuralLM(
            checkpoint=train_job.out_checkpoints[100],
            net_args=train_args["net_args"],
            network_module=network_module,
            prefix_name=training_name
        ))
