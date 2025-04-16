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

from ...pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v1_cfg import TransformerLMConfig, TransformerMHSAConfig, \
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

def bpe128_kazuki_trafo():

    for BPE_SIZE in [128, 512, 5000]:
        prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/lm_bpe/bpe_%i_trafo" % BPE_SIZE

        train_settings_part100 = LMDatasetSettings(
            train_partition_epoch=100,
            train_seq_ordering="laplace:.100",
        )

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe128_part100 = build_lm_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings_part100,
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

        hidden_dim = 768
        trafo_block_config = generate_transformer_block_config(
            input_dim=hidden_dim,
            ff_dim= 4096,
            output_dim=hidden_dim,
            num_heads=4,
            dropout=0.0,
        )
        trafo_base_config = TransformerLMConfig(
            embed_dim=128,
            hidden_dim=hidden_dim,
            vocab_dim=vocab_size_without_blank,
            num_layers=12,
            block_config=trafo_block_config,
            batch_first=True,  # very important, state management in decoder does not work otherwise
            dropout=0.0,
        )

        train_config_modern_v1 = {
            "optimizer": {"class": "RAdam"},
            #############
            "batch_size": 2000,  # BPE tokens (take more than kazuki because this is small)
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "learning_rates": ([1e-4] * 50) + list(np.linspace(1e-4, 1e-6, 150)),
            "torch_amp_options": {"dtype": "bfloat16"},
        }

        network_module = "lm.trafo.kazuki_trafo_zijian_variant_v1"
        train_args = {
            "config": train_config_modern_v1,
            "post_config": {"num_workers_per_gpu": 1},
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(trafo_base_config)},
            "debug": True,
            "add_cache_manager": True,
        }

        training_name = prefix_name + "/" + network_module + ".12x768_2k_RAdam_1e-3_2ep_reduce_gcn1.0"
        train_job = training(training_name, train_data_bpe128_part100, train_args, num_epochs=200, **lm_returnn)
        train_job.rqmt["gpu_mem"] = 24

        from ...storage import add_lm
        from ...pipeline import NeuralLM
        add_lm("bpe%i_trafo12x768_2ep" % BPE_SIZE, lm_model=NeuralLM(
            checkpoint=train_job.out_checkpoints[200],
            net_args=train_args["net_args"],
            network_module=network_module,
            prefix_name=training_name
        ))





        hidden_dim = 1024
        trafo_block_config = generate_transformer_block_config(
            input_dim=hidden_dim,
            ff_dim= 4096,
            output_dim=hidden_dim,
            num_heads=4,
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

        train_config_modern_v1 = {
            "optimizer": {"class": "RAdam"},
            #############
            "batch_size": 3000,  # BPE tokens
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "learning_rates": list(np.linspace(5e-6, 4e-4, 100)) + list(np.linspace(4e-4, 1e-6, 150)),  # determined by OCLR test
            "torch_amp_options": {"dtype": "bfloat16"},
        }

        train_args = {
            "config": train_config_modern_v1,
            "post_config": {"num_workers_per_gpu": 1},
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(trafo_base_config)},
            "debug": True,
            "add_cache_manager": True,
        }

        training_name = prefix_name + "/" + network_module + ".24x1024_3k_RAdam_1e-3_3ep_reduce_gcn1.0"
        train_job = training(training_name, train_data_bpe128_part100, train_args, num_epochs=300, **lm_returnn)
        train_job.rqmt["gpu_mem"] = 24
        train_job.hold()
        train_job.move_to_hpc = True

        add_lm("bpe%i_trafo24x1024_3ep" % BPE_SIZE, lm_model=NeuralLM(
            checkpoint=train_job.out_checkpoints[300],
            net_args=train_args["net_args"],
            network_module=network_module,
            prefix_name=training_name
        ))
