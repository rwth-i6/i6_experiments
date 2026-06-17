from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.bpe_lm import build_lm_training_datasets, LMDatasetSettings, build_lm_perplexity_dataset
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...pipeline import training, calculate_perplexity


def small_bpe_kazuki_lstm():


    def perplexity_helper(training_name, bpe_size, train_args, checkpoint):
        for dev_set in ["dev.short", "dev.commonvoice", "dev.librispeech", "dev.voxpopuli", "dev.yodas"]:
            dev_ls, word_count_file, bpe_count_file = build_lm_perplexity_dataset(prefix_name,
                                                                                  loquacious_key="train.small",
                                                                                  dev_dataset_key=dev_set,
                                                                                  bpe_size=bpe_size)
            calculate_perplexity(
                prefix_name=training_name + "/" + dev_set,
                dataset=dev_ls,
                train_args=train_args,
                config={},
                perplexity_module="lm.lstm.perplexity_calculator",
                perplexity_args={"config": {"word_count_file": word_count_file, "bpe_count_file": bpe_count_file}},
                checkpoint=checkpoint,
                **lm_returnn
            )


    for BPE_SIZE in [128, 1000]:
        prefix_name = "experiments/loquacious/standalone_2025/lm_bpe/train_small_bpe_%i_lstm" % BPE_SIZE

        train_settings_part100 = LMDatasetSettings(
            train_partition_epoch=100,
            train_seq_ordering="laplace:.100",
        )

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe_part100 = build_lm_training_datasets(
            prefix=prefix_name,
            loquacious_key="train.small",
            bpe_size=BPE_SIZE,
            settings=train_settings_part100,
        )

        label_datastream_bpe128 = cast(LabelDatastream, train_data_bpe_part100.datastreams["data"])
        vocab_size_without_blank = label_datastream_bpe128.vocab_size

        lm_returnn = {
            "returnn_exe": RETURNN_EXE,
            "returnn_root": MINI_RETURNN_ROOT,
        }

        from ...pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v2_cfg import ModelConfig
        from ...pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v3_cfg import ModelConfig as ModelConfigV3

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

        lstm_ls_config = ModelConfigV3(
            vocab_dim=vocab_size_without_blank,
            embed_dim=512,
            hidden_dim=2048,
            n_lstm_layers=2,
            use_bottle_neck=False,
            dropout=0.2,
            init_args=default_init_args,
            label_smoothing=1e-4,
        )

        train_config_modern_v1 = {
            "optimizer": {"class": "RAdam"},
            #############
            "batch_size": 1280,  # BPE tokens
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "learning_rates": ([1e-3] * 100) + list(np.linspace(1e-3, 1e-5, 400)),
            "num_workers_per_gpu": 1
        }

        network_module = "lm.lstm.kazuki_lstm_zijian_variant_v2"
        train_args = {
            "config": train_config_modern_v1,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(lstm_base_config)},
            "debug": True,
            "add_cache_manager": True,
        }

        training_name = prefix_name + "/" + network_module + ".2x2048_1k_RAdam_1e-3_5ep_reduce_gcn1.0"
        train_job = training(training_name, train_data_bpe_part100, train_args, num_epochs=500, **lm_returnn)

        from ...storage import add_lm
        from ...pipeline import NeuralLM
        add_lm("small_bpe%i_2x2048_kazuki_lstmlm_5ep" % BPE_SIZE, lm_model=NeuralLM(
            checkpoint=train_job.out_checkpoints[500],
            net_args=train_args["net_args"],
            network_module=network_module,
            prefix_name=training_name
        ))

        perplexity_helper(
            training_name,
            bpe_size=BPE_SIZE,
            train_args=train_args,
            checkpoint=train_job.out_checkpoints[500]
        )

        # test label smoothing
        train_args_labelsmooth = copy.deepcopy(train_args)
        train_args_labelsmooth["network_module"] = "lm.lstm.kazuki_lstm_zijian_variant_v3"
        train_args_labelsmooth["net_args"] = {"model_config_dict": asdict(lstm_ls_config)}
        training_name = prefix_name + "/" + network_module + ".2x2048_1k_RAdam_1e-3_5ep_reduce_gcn1.0_labelsmooth_1e-4"
        train_job = training(training_name, train_data_bpe_part100, train_args_labelsmooth, num_epochs=500, **lm_returnn)

        perplexity_helper(
            training_name,
            bpe_size=BPE_SIZE,
            train_args=train_args_labelsmooth,
            checkpoint=train_job.out_checkpoints[500]
        )


