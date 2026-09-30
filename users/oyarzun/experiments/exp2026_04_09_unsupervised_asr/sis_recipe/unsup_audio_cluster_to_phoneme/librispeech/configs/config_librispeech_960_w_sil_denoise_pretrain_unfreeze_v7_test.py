"""
Test configuration for Version 7 (Cross-Modal Frozen Embeddings):
- 1 epoch of denoising pretraining
- 100 epochs of backtranslation fine-tuning (preloaded from pretraining checkpoint)
- Configured for CPU execution (interactive session)
- All evaluation, recognition, visualization, and analysis jobs removed
"""

import copy
import os
from typing import List, Optional

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.common.setups.serialization import PartialImport
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common_hubert import (
    build_training_datasets_hubert,
    build_test_datasets_hubert,
    get_audio_embeddings_path,
    get_text_embeddings_path,
)
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, alternate_batching

from sisyphus import tk

# -------------------------------------------------------------------------
# Global Dataset & Pipeline Settings
# -------------------------------------------------------------------------
settings = DatasetSettings(
    train_partition_epoch=500,
    train_seq_ordering="laplace:.1000",
)

# Build training dataset using the HuBERT clustering scheme
train_data = build_training_datasets_hubert(
    sil_prob=0.25,
    surround_w_sil=True,
    settings=settings,
)
test_data_dict = build_test_datasets_hubert()

# -------------------------------------------------------------------------
# Base Configuration (Targeting CPU execution)
# -------------------------------------------------------------------------
base_config = copy.deepcopy(base_config)
base_config["random_seed"] = 42
base_config["__network_module"] = "definitions.conformer_aed_cross_modal_shared_v1.Model"
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_denoise_v6.train_step"
base_config["save_interval"] = 10

# CPU Execution Settings
base_config["training"]["batch_size"] = 30000
base_config["training"]["max_seqs"] = 1000
base_config["training"]["accum_grad_multiple_step"] = 1
base_config["training"]["grad_scaler"] = None

base_config["train_rqmt"] = {
    "device": "cpu",
    "cpu_rqmt": 12,
    "mem_rqmt": 64,
    "time_rqmt": 1,
}

# Base model dimensions and frozen cross-modal embedding setup
base_config["model_args"].update({
    "num_enc_layers": 6,
    "num_text_dec_layers": 6,
    "num_audio_dec_layers": 6,
    "num_heads": 8,
    "model_dim": 512,
    "share_decoder": True,
    "text_out_dim": train_data.datastreams["target"].vocab_size,
    "audio_out_dim": train_data.datastreams["data"].vocab_size,
    "discriminator_type": None,
    "codebook_opts": {"codebook_prob": 0.0},
    "audio_embeddings_path": get_audio_embeddings_path(),
    "text_embeddings_path": get_text_embeddings_path(),
    "cross_modal_emb_dim": 300,
    "freeze_cross_modal_embeddings": True,
})


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"
    layers = 6

    # =========================================================================
    # 1. PRETRAINING JOB (1 Epoch Denoising Only)
    # =========================================================================
    p_name = f"test_pretrain_cm_enc-{layers}_dec-{layers}_ep-1_cpu"

    p_config = copy.deepcopy(base_config)
    p_config["model_args"].update({
        "num_enc_layers": layers,
        "num_text_dec_layers": layers,
        "num_audio_dec_layers": layers,
        "discriminator_type": None,
        "codebook_opts": {"codebook_prob": 0.0},
        "audio_embeddings_path": get_audio_embeddings_path(),
        "text_embeddings_path": get_text_embeddings_path(),
        "cross_modal_emb_dim": 300,
        "freeze_cross_modal_embeddings": True,
    })
    p_config["train_args"].update({
        "codebook_diversity_loss_scale": 0.0,
        "denoise_pretrain_epochs": 1,
        "pretrain_codebook_prob": 0.0,
        "pretrain_codebook_diversity_loss_scale": 0.0,
        "adv_loss_scale": 0.0,
        "pretrain_adv_loss_scale": 0.0,
        "text_masking_opts": {"mask_prob": 0.3, "min_span": 2, "max_span": 10, "expand": True, "insert_prob": 0.1},
        "audio_masking_opts": {"mask_prob": 0.3, "min_span": 4, "max_span": 20, "expand": True, "insert_prob": 0.1},
    })
    p_config["training"]["__num_epochs"] = 1
    p_config["training"]["__lr_opts"] = {
        "type": "dyn_lr_piecewise_linear",
        "piecewise_epochs": [0, 0.5, 1.0],
        "piecewise_values": [1e-5, 1e-3, 1e-4],
    }

    # Run pretraining job without any evaluations or analysis
    pretrain_job = run_experiment(
        training_name=f"{prefix_name}/{p_name}",
        config=p_config,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=[1],
        skip_eval=True,
        rasr_recog_opts=None,
        vis_epochs=None,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )

    # =========================================================================
    # 2. FINE-TUNING JOB (100 Epochs Backtranslation, Preloaded from Pretrain)
    # =========================================================================
    ft_epochs = 100
    ft_name = f"test_cm_enc-{layers}_dec-{layers}_pretrain-1_ft_ep-100_cpu"

    ft_config = copy.deepcopy(base_config)
    ft_config["model_args"].update({
        "num_enc_layers": layers,
        "num_text_dec_layers": layers,
        "num_audio_dec_layers": layers,
        "discriminator_type": None,
        "codebook_opts": {"codebook_prob": 0.0},
        "audio_embeddings_path": get_audio_embeddings_path(),
        "text_embeddings_path": get_text_embeddings_path(),
        "cross_modal_emb_dim": 300,
        "freeze_cross_modal_embeddings": True,
    })
    ft_config["train_args"].update({
        "codebook_diversity_loss_scale": 0.0,
        "denoise_pretrain_epochs": 0,  # Run backtranslation immediately
        "pretrain_codebook_prob": 0.0,
        "pretrain_codebook_diversity_loss_scale": 0.0,
        "adv_loss_scale": 0.0,
        "pretrain_adv_loss_scale": 0.0,
        "text_masking_opts": {"mask_prob": 0.3, "min_span": 2, "max_span": 10, "expand": True, "insert_prob": 0.1},
        "audio_masking_opts": {"mask_prob": 0.3, "min_span": 4, "max_span": 20, "expand": True, "insert_prob": 0.1},
        "gradual_unfreeze": True,
        "freeze_encoder": False,
        "gradual_unfreeze_proportion": 0.8,
        "gradual_unfreeze_start_iter": int(40_000 * 0.5),
        "gradual_unfreeze_end_iter": int(40_000 * 0.9),
        "bt_buffer_size_steps": 10,
        "bt_train_iterations": 50,
    })
    ft_config["training"]["__num_epochs"] = ft_epochs
    ft_config["training"]["preload_from_files"] = {
        "": {"filename": pretrain_job.out_checkpoints[1].path}
    }
    ft_config["training"]["__lr_opts"] = {
        "type": "dyn_lr_piecewise_linear",
        "piecewise_epochs": [0, 45.0, 90.0, 100.0],
        "piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    }

    # Run fine-tuning job without any evaluations or analysis
    run_experiment(
        training_name=f"{prefix_name}/{ft_name}",
        config=ft_config,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=[100],
        skip_eval=True,
        rasr_recog_opts=None,
        vis_epochs=None,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )
