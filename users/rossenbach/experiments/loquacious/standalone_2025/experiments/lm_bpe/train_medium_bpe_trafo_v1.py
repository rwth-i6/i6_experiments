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
from ...storage import add_lm, NeuralLM

from ...pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2_cfg import TransformerLMConfig, TransformerMHSAConfig, \
    TransformerBlockConfig, TransformerLinearConfig


def generate_transformer_block_config(input_dim, ff_dim, output_dim, num_heads, dropout=0.0, batch_first=True):
    linear_config = TransformerLinearConfig(
        input_dim=input_dim,
        ff_dim=ff_dim,
        output_dim=output_dim,
        dropout=0.0,
        batch_first=batch_first
    )
    mhsa_config = TransformerMHSAConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=batch_first
    )
    block_config = TransformerBlockConfig(
        linear_config=linear_config,
        mhsa_config=mhsa_config
    )
    return block_config

def train_medium_bpe_kazuki_trafo():

    for BPE_SIZE in [128, 256, 512, 1000, 2000, 10000]:
        prefix_name = "experiments/loquacious/standalone_2025/lm_bpe/train_medium_bpe_%i_trafo" % BPE_SIZE

        train_settings_part100 = LMDatasetSettings(
            train_partition_epoch=100,
            train_seq_ordering="laplace:.100",
        )
        train_settings_part25 = LMDatasetSettings(
            train_partition_epoch=25,
            train_seq_ordering="laplace:.100",
        )

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe128_part100 = build_lm_training_datasets(
            prefix=prefix_name,
            loquacious_key="train.medium",
            bpe_size=BPE_SIZE,
            settings=train_settings_part100,
        )
        train_data_bpe128_part25 = build_lm_training_datasets(
            prefix=prefix_name,
            loquacious_key="train.medium",
            bpe_size=BPE_SIZE,
            settings=train_settings_part25,
        )

        label_datastream_bpe128 = cast(LabelDatastream, train_data_bpe128_part100.datastreams["data"])
        vocab_size_without_blank = label_datastream_bpe128.vocab_size


        # Extra version to debug LM dataset behavior
        MINI_RETURNN_ROOT = CloneGitRepositoryJob(
            "https://github.com/JackTemaki/MiniReturnn", commit="e37396f8838343ba5e2c5053103244c9271f916a"
        ).out_repository.copy()
        MINI_RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_ROOT"

        lm_returnn = {
            "returnn_exe": RETURNN_EXE,
            "returnn_root": MINI_RETURNN_ROOT,
        }


        # 24 layer
        hidden_dim = 768
        trafo_block_config = generate_transformer_block_config(
            input_dim=hidden_dim,
            ff_dim= 4096,
            output_dim=hidden_dim,
            num_heads=8,
            dropout=0.0,
        )
        trafo_base_config = TransformerLMConfig(
            embed_dim=128,
            hidden_dim=hidden_dim,
            vocab_dim=vocab_size_without_blank,
            num_layers=24,
            block_config=trafo_block_config,
            batch_first=True,  # very important, state management in decoder does not work otherwise
            dropout=0.0,
        )

        trafo_block_config_drop = generate_transformer_block_config(
            input_dim=hidden_dim,
            ff_dim=4096,
            output_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
        )
        trafo_drop_config = TransformerLMConfig(
            embed_dim=128,
            hidden_dim=hidden_dim,
            vocab_dim=vocab_size_without_blank,
            num_layers=24,
            block_config=trafo_block_config_drop,
            batch_first=True,  # very important, state management in decoder does not work otherwise
            dropout=0.1,
        )

        network_module = "lm.trafo.kazuki_trafo_zijian_variant_v2"

        # 24 layer longer training
        train_config_modern_v1 = {
            "optimizer": {"class": "RAdam", "decoupled_weight_decay": True, "weight_decay": 0.005},
            #############
            "batch_size": 3000,  # BPE tokens
            "accum_grad_multiple_step": 2,
            "gradient_clip_norm": 2.0,
            "learning_rates": list(np.linspace(5e-6, 3e-4, 100)) + list(np.linspace(3e-4, 1e-6, 400)),  # determined by OCLR test
            "torch_amp_options": {"dtype": "bfloat16"},
            "max_seq_length": 400,
            "num_workers_per_gpu": 1
        }

        train_args = {
            "config": train_config_modern_v1,
            "post_config": {},
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(trafo_base_config)},
            "debug": True,
            "add_cache_manager": True,
        }

        training_name = prefix_name + "/" + network_module + ".24x768_2x3k_RAdam_3e-4_5ep_reduce_gcn2.0"
        train_job = training(training_name, train_data_bpe128_part100, train_args, num_epochs=500, **lm_returnn)
        train_job.rqmt["gpu_mem"] = 24

        add_lm("bpe%i_trafo24x768_5ep_medium" % BPE_SIZE, lm_model=NeuralLM(
            checkpoint=train_job.out_checkpoints[500],
            net_args=train_args["net_args"],
            network_module=network_module,
            prefix_name=training_name
        ))

        # if BPE_SIZE in [128, 1000, 2000, 10000]:
        #     # long training
        #     training_name = prefix_name + "/" + network_module + ".24x768_2x3k_RAdam_3e-4_5ep_reduce_gcn2.0_drop01_long"
        #     train_args_long = copy.deepcopy(train_args)
        #     train_args_long["net_args"] = {"model_config_dict": asdict(trafo_drop_config)}
        #     train_job = training(training_name, train_data_bpe128_part25, train_args_long, num_epochs=500, **lm_returnn)
        #     train_job.rqmt["gpu_mem"] = 24
        #     train_job.hold()
        #     train_job.move_to_hpc = True
