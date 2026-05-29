"""
Phoneme Transformer LM training on the LibriSpeech LM corpus + LS-960
transcripts, using the EOW phoneme inventory of the pHMM AM extended with
`<s>` / `</s>`.

Follows the standard RETURNN training-config conventions used by the pHMM AM
baseline in `experiments/phmm_phon/baseline.py` (behavior_version, explicit
extern_data, torch_amp, torch_dataloader_opts, ...).
"""

from dataclasses import asdict
from typing import cast

import numpy as np

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.phon_lm import LMDatasetSettings, build_phon_lm_training_datasets
from ...default_tools import RETURNN_EXE, RETURNN_ROOT
from ...pipeline import NeuralLM, training
from ...pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v1_cfg import (
    TransformerBlockConfig,
    TransformerLinearConfig,
    TransformerLMConfig,
    TransformerMHSAConfig,
)
from ...storage import add_lm


def _block_config(input_dim: int, ff_dim: int, num_heads: int, dropout: float, batch_first: bool = True):
    return TransformerBlockConfig(
        linear_config=TransformerLinearConfig(
            input_dim=input_dim, ff_dim=ff_dim, output_dim=input_dim, dropout=0.0, batch_first=batch_first
        ),
        mhsa_config=TransformerMHSAConfig(
            input_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=batch_first
        ),
    )


def phon_trafo_12x512_baseline():
    prefix_name = "example_setups/librispeech/posterior_hmm/lm_phon/trafo_12x512"

    train_settings = LMDatasetSettings(
        train_partition_epoch=100,
        train_seq_ordering="laplace:.100",
    )
    train_data = build_phon_lm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )

    label_datastream = cast(LabelDatastream, train_data.datastreams["data"])
    vocab_dim = label_datastream.vocab_size

    lm_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": RETURNN_ROOT,
    }

    hidden_dim = 512
    block_config = _block_config(
        input_dim=hidden_dim,
        ff_dim=2048,
        num_heads=4,
        dropout=0.0,
    )
    model_config = TransformerLMConfig(
        embed_dim=128,
        hidden_dim=hidden_dim,
        vocab_dim=vocab_dim,
        num_layers=12,
        block_config=block_config,
        batch_first=True,
        dropout=0.0,
    )

    num_epochs = 300

    train_config = {
        "behavior_version": 21,
        "extern_data": {
            "data": label_datastream.as_returnn_extern_data_opts(available_for_inference=True),
            "delayed": label_datastream.as_returnn_extern_data_opts(available_for_inference=True),
        },
        "optimizer": {
            "class": "radam",
            "decoupled_weight_decay": True,
            "weight_decay": 1e-2,
        },
        "learning_rates": list(np.linspace(5e-6, 3e-4, 100)) + list(np.linspace(3e-4, 1e-6, 200)),
        "batch_size": 4000,
        "max_seqs": 64,
        "max_seq_length": {"data": 2000},
        "accum_grad_multiple_step": 2,
        "gradient_clip_global_norm": 1.0,
        "torch_amp": {"dtype": "bfloat16"},
        "torch_dataloader_opts": {"num_workers": 1},
        "log_grad_norm": True,
    }

    network_module = "lm.trafo.kazuki_trafo_zijian_variant_v1"
    train_args = {
        "config": train_config,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "add_cache_manager": True,
    }

    training_name = prefix_name + "/" + network_module + ".12x512_4k_RAdam_3e-4_3ep_grad_clip1.0"
    train_job = training(training_name, train_data, train_args, num_epochs=num_epochs, **lm_returnn)
    train_job.rqmt["gpu_mem"] = 24

    add_lm(
        "phon_trafo12x512_3ep",
        lm_model=NeuralLM(
            checkpoint=train_job.out_checkpoints[num_epochs],
            net_args=train_args["net_args"],
            network_module=network_module,
            prefix_name=training_name,
            phon_vocab=label_datastream.vocab,
        ),
    )

    return train_job
