"""
Version 8.1 Configuration:
- End-to-end ablation for the average encoder logits layer objective (avg-layers-1-6) trained from scratch.
- Denoising audio sequence lengthening removed: audio uses in-place 1-to-1 span masking (`expand=False, keep_length=True`),
  preserving exact length T and 1-to-1 alignment with unmasked targets.
- Text sequence lengthening preserved and expanded: text uses `expand=True, insert_prob=0.1` with K in [1, 15] (`max_k=15`).
- No sequence order alterations for either text or audio.
- Phase 1: 200 epochs pretraining from scratch (no discriminator, no codebooks), constant lr 1e-4, warm/learnable encoder,
  4 eval checkpoints: [50, 100, 150, 200].
- Phase 2: 500 epochs denoising + backtranslation fine-tuning from pretrain epoch 200 checkpoint, constant lr 1e-4,
  warm/learnable encoder, 4 eval checkpoints: [125, 250, 375, 500].
"""

import copy
import os
from typing import List

from i6_core.returnn.config import ReturnnConfig
from i6_core.serialization import Collection
from sisyphus import tk

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets, build_contrastive_pca_test_datasets
from ....data.common import DatasetSettings
from ... import __setup_base_name__
from .config_librispeech_960_v1 import base_config, alternate_batching


settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering="laplace:.1000",
)
train_data = build_training_datasets(
    sil_prob=0.25,
    surround_w_sil=True,
    settings=settings,
)
test_data_dict = build_test_datasets()
contrastive_pca_test_data_dict = build_contrastive_pca_test_datasets()

base_config = copy.deepcopy(base_config)
base_config["random_seed"] = 42
base_config["__network_module"] = "definitions.conformer_aed_discrete_shared_aux_rec_v1.Model"
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_aux_rec_v1.train_step"
base_config["save_interval"] = 25
base_config["training"]["__num_gpus"] = 1
base_config["training"]["batch_size"] = 15000
base_config["training"]["max_seqs"] = 1000
base_config["train_rqmt"]["cpu_rqmt"] = 12
base_config["train_rqmt"]["gpu_mem"] = 80
base_config["train_rqmt"]["mem_rqmt"] = 64
base_config["recog"] = {
    "batch_size": 20000,
}

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
    "audio_aux_rec_layers": list(range(1, 7)),  # avg-layers-1-6
})


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    layers = 6
    aux_rec_layers = list(range(1, layers + 1))
    aux_rec_scale = 0.5
    lr_val = 1e-4
    lr_str = "1e-4"

    # Masking settings:
    # Text: expanded with contiguous mask chunks, K in [1, 15] to increase length
    text_masking_opts = {
        "mask_prob": 0.3,
        "min_span": 2,
        "max_span": 10,
        "expand": True,
        "insert_prob": 0.1,
        "min_k": 1,
        "max_k": 15,
    }
    # Audio: strictly in-place 1-to-1 span masking (no length expansion or random insertions)
    audio_masking_opts = {
        "mask_prob": 0.3,
        "min_span": 4,
        "max_span": 20,
        "expand": False,
        "keep_length": True,
    }

    # =========================================================================
    # PHASE 1: Pretraining for 200 Epochs from Scratch
    # =========================================================================
    pretrain_epochs = 200
    pretrain_eval_epochs = [50, 100, 150, 200]

    pretrain_train_name = (
        f"pretrain_disc-neither_enc-{layers}_dec-{layers}_avg-layers-1-6_"
        f"constlr-{lr_str}_k15_ep-{pretrain_epochs}_v8.1"
    )

    pretrain_cfg = copy.deepcopy(base_config)
    pretrain_cfg["train_args"].update({
        "codebook_diversity_loss_scale": 0.0,
        "denoise_pretrain_epochs": pretrain_epochs,
        "pretrain_codebook_prob": 0.0,
        "pretrain_codebook_diversity_loss_scale": 0.0,
        "adv_loss_scale": 0.0,
        "pretrain_adv_loss_scale": 0.0,
        "audio_aux_rec_loss_scale": aux_rec_scale,
        "gradual_unfreeze": False,
        "freeze_encoder": False,
        "bt_buffer_size_steps": 10,
        "bt_train_iterations": 50,
        "text_masking_opts": text_masking_opts,
        "audio_masking_opts": audio_masking_opts,
    })

    pretrain_cfg["training"].update({
        "batch_size": 15000,
        "grad_scaler": None,
        "accum_grad_multiple_step": 2,
        "__num_epochs": pretrain_epochs,
        "__lr_opts": {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": [0, pretrain_epochs],
            "piecewise_values": [lr_val, lr_val],
        },
    })
    pretrain_cfg["recog_rqmt"] = {
        "time": 4,
        "mem": 32,
        "cpu": 6,
        "gpu": 1,
        "device": "gpu",
        "gpu_mem": 80,
    }
    pretrain_cfg.setdefault("train_rqmt", {})["mem_rqmt"] = 64

    pretrain_job = run_experiment(
        training_name=f"{prefix_name}/{pretrain_train_name}",
        config=pretrain_cfg,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=pretrain_eval_epochs,
        skip_eval=False,
        rasr_recog_opts=None,
        vis_epochs=pretrain_eval_epochs,
        vis_kwargs={"cosine_similarity_summary": True},
        run_contrastive_pca=True,
        contrastive_pca_test_data_dict=contrastive_pca_test_data_dict,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )

    pretrain_configs_meta = {
        pretrain_train_name: {
            "base_checkpoint": "scratch",
            "layers": layers,
            "disc_type": "None",
            "codebook_prob": 0.0,
            "unfreeze_mode": "warm (fully unfrozen)",
            "learning_rate": lr_str,
            "audio_aux_rec_loss_scale": aux_rec_scale,
            "audio_aux_rec_layers": str(aux_rec_layers),
            "text_mask_k_max": 15,
            "audio_masking": "in-place 1-to-1 (no expansion)",
            "pretrain_epochs": pretrain_epochs,
        }
    }

    # =========================================================================
    # PHASE 2: Denoising + Backtranslation Training for 500 Epochs
    # =========================================================================
    ft_epochs = 500
    eval_epochs = [125, 250, 375, 500]

    ft_train_name = (
        f"from-pretrain-ep200_disc-neither_enc-{layers}_dec-{layers}_unfreeze-warm_"
        f"constlr-{lr_str}_auxrec-{aux_rec_scale}_avg-layers-1-6_k15_ft_ep-{ft_epochs}_v8.1"
    )

    ft_cfg = copy.deepcopy(base_config)
    ft_cfg["train_args"].update({
        "codebook_diversity_loss_scale": 0.0,
        "denoise_pretrain_epochs": 0,
        "pretrain_codebook_prob": 0.0,
        "pretrain_codebook_diversity_loss_scale": 0.0,
        "adv_loss_scale": 0.0,
        "pretrain_adv_loss_scale": 0.0,
        "audio_aux_rec_loss_scale": aux_rec_scale,
        "gradual_unfreeze": False,
        "freeze_encoder": False,
        "bt_buffer_size_steps": 10,
        "bt_train_iterations": 50,
        "text_masking_opts": text_masking_opts,
        "audio_masking_opts": audio_masking_opts,
    })

    ft_cfg["training"].update({
        "batch_size": 15000,
        "grad_scaler": None,
        "accum_grad_multiple_step": 2,
        "__num_epochs": ft_epochs,
        "preload_from_files": {
            "pretrained_model": {
                "filename": pretrain_job.out_checkpoints[pretrain_epochs].path,
                "init_for_train": True,
                "checkpoint_key": "model",
                "ignore_missing": True,
                "ignore_params_prefixes": ["quantizer.", "discriminator."],
            }
        },
        "__lr_opts": {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": [0, ft_epochs],
            "piecewise_values": [lr_val, lr_val],
        },
    })

    ft_cfg["recog_rqmt"] = {
        "time": 4,
        "mem": 32,
        "cpu": 6,
        "gpu": 1,
        "device": "gpu",
        "gpu_mem": 80,
    }
    ft_cfg.setdefault("train_rqmt", {})["mem_rqmt"] = 64

    ft_train_job = run_experiment(
        training_name=f"{prefix_name}/{ft_train_name}",
        config=ft_cfg,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=eval_epochs,
        skip_eval=False,
        rasr_recog_opts=None,
        vis_epochs=eval_epochs,
        vis_kwargs={"cosine_similarity_summary": True},
        run_contrastive_pca=True,
        contrastive_pca_test_data_dict=contrastive_pca_test_data_dict,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )

    configs_meta = {
        ft_train_name: {
            "base_checkpoint": f"{pretrain_train_name}@epoch_{pretrain_epochs}",
            "layers": layers,
            "disc_type": "None",
            "codebook_prob": 0.0,
            "unfreeze_mode": "warm (fully unfrozen)",
            "learning_rate": lr_str,
            "audio_aux_rec_loss_scale": aux_rec_scale,
            "audio_aux_rec_layers": str(aux_rec_layers),
            "text_mask_k_max": 15,
            "audio_masking": "in-place 1-to-1 (no expansion)",
            "ft_epochs": ft_epochs,
            "cumulative_epochs": pretrain_epochs + ft_epochs,
        }
    }

    from ....hpo_summary import HpoResultsExcelJob

    # Summary report for Phase 2 (Finetuning)
    excel_job = HpoResultsExcelJob(
        configs_meta=configs_meta,
        train_jobs={ft_train_name: ft_train_job},
        eval_epochs=eval_epochs,
        target_num_epochs=ft_epochs,
        prefix_name=prefix_name,
        report_name="unsup_v8_1_aux_rec_summary",
    )
    tk.register_output(f"{prefix_name}/unsup_v8_1_aux_rec_summary.xlsx", excel_job.out_excel)
    tk.register_output(f"{prefix_name}/unsup_v8_1_aux_rec_summary.csv", excel_job.out_csv)

    # Summary report for Phase 1 (Pretraining)
    pretrain_excel_job = HpoResultsExcelJob(
        configs_meta=pretrain_configs_meta,
        train_jobs={pretrain_train_name: pretrain_job},
        eval_epochs=pretrain_eval_epochs,
        target_num_epochs=pretrain_epochs,
        prefix_name=f"{prefix_name}/pretrain",
        report_name="unsup_v8_1_pretrain_summary",
    )
    tk.register_output(f"{prefix_name}/pretrain/unsup_v8_1_pretrain_summary.xlsx", pretrain_excel_job.out_excel)
    tk.register_output(f"{prefix_name}/pretrain/unsup_v8_1_pretrain_summary.csv", pretrain_excel_job.out_csv)

    return {
        "pretrain_job": pretrain_job,
        "finetune_job": ft_train_job,
    }

