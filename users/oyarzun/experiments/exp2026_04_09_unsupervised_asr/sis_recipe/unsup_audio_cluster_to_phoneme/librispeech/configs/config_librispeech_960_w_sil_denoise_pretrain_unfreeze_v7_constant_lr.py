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
    DEFAULT_AUDIO_CROSS_MODAL_EMBEDDINGS_PATH,
    DEFAULT_TEXT_CROSS_MODAL_EMBEDDINGS_PATH,
    DEFAULT_HUBERT_CLUSTERS_960_HDFS,
    DEFAULT_HUBERT_CLUSTERS_DEV_OTHER_HDFS,
    DEFAULT_HUBERT_CLUSTERS_DEV_CLEAN_HDFS,
)
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, alternate_batching
from .config_librispeech_960_w_sil_denoise_pretrain_unfreeze_v7_cross_modal import (
    py as get_v7_cross_modal_jobs,
)

from sisyphus import tk

# -------------------------------------------------------------------------
# Global Dataset & Pipeline Settings
# -------------------------------------------------------------------------
settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering="laplace:.1000",
)

# Build training and evaluation datasets using the HuBERT clustering scheme
train_data = build_training_datasets_hubert(
    sil_prob=0.25,
    surround_w_sil=True,
    settings=settings,
)
test_data_dict = build_test_datasets_hubert()

# Base config targeting the cross-modal model and dedicated v6 train step
base_config = copy.deepcopy(base_config)
base_config["random_seed"] = 42
base_config["__network_module"] = "definitions.conformer_aed_cross_modal_shared_v1.Model"
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_denoise_v6.train_step"
base_config["save_interval"] = 25
base_config["training"]["__num_gpus"] = 1
base_config["training"]["batch_size"] = 30000
base_config["training"]["max_seqs"] = 1000
base_config["train_rqmt"]["cpu_rqmt"] = 12
base_config["train_rqmt"]["gpu_mem"] = 80
base_config["train_rqmt"]["mem_rqmt"] = 64

# Base model dimensions and cross-modal embedding setup
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

    # =========================================================================
    # ABLATION SPECIFICATION: Version 7 Constant Learning Rate Fine-Tuning
    #
    # Part 1: Resuming from epoch 375 of the top-performing v7 configurations:
    # 1. cm_clus-4P_enc-6_dec-6_pretrain-0_unfreeze-progressive_ft_ep-500_v7
    # 2. cm_clus-2P_enc-6_dec-6_pretrain-0_unfreeze-progressive_ft_ep-500_v7
    # 3. cm_clus-1P_enc-6_dec-6_pretrain-100_unfreeze-progressive_ft_ep-500_v7
    # - Duration: 625 additional epochs (total 1000 cumulative epochs)
    # - Evaluation Checkpoints: [125, 375, 625] (cumulative epochs 500, 750, 1000)
    #
    # Part 2: Resuming from epoch 500 of the top-performing v7 configurations:
    # 4. cm_clus-2P_enc-6_dec-6_pretrain-0_unfreeze-warm_ft_ep-500_v7 (reached 79.87% PER at Ep 500)
    # 5. cm_clus-4P_enc-6_dec-6_pretrain-0_unfreeze-progressive_ft_ep-500_v7 (reached 91.01% PER at Ep 500)
    # - Duration: 500 additional epochs (total 1000 cumulative epochs)
    # - Evaluation Checkpoints: [250, 500] (cumulative epochs 750, 1000)
    #
    # Common Training Parameters:
    # - Architecture: Fully unfrozen encoder (gradual_unfreeze=False, freeze_encoder=False)
    # - Learning Rate Choices:
    #     * 1e-4: Lower, conservative fine-tuning rate
    #     * 3e-4: Continuity rate matching epoch 375 (~3.44e-4) sweet spot
    #     * 5e-4: Higher rate (~1.5x) for aggressive fitting to prevent underfitting
    # =========================================================================

    layers = 6

    constant_lrs = [
        (1e-4, "1e-4"),
        (3e-4, "3e-4"),
        (5e-4, "5e-4"),
    ]

    base_checkpoints = [
        # --- Resuming from Epoch 375 (625 additional epochs; evals after 125, 375, 625 epochs) ---
        {
            "mult": 4,
            "pretrain_ep": 0,
            "base_name": f"cm_clus-4P_enc-{layers}_dec-{layers}_pretrain-0_unfreeze-progressive_ft_ep-500_v7",
            "resume_epoch": 375,
            "ft_epochs": 625,
            "eval_epochs": [125, 375, 625],
        },
        {
            "mult": 2,
            "pretrain_ep": 0,
            "base_name": f"cm_clus-2P_enc-{layers}_dec-{layers}_pretrain-0_unfreeze-progressive_ft_ep-500_v7",
            "resume_epoch": 375,
            "ft_epochs": 625,
            "eval_epochs": [125, 375, 625],
        },
        {
            "mult": 1,
            "pretrain_ep": 100,
            "base_name": f"cm_clus-1P_enc-{layers}_dec-{layers}_pretrain-100_unfreeze-progressive_ft_ep-500_v7",
            "resume_epoch": 375,
            "ft_epochs": 625,
            "eval_epochs": [125, 375, 625],
        },
        # --- Resuming from Epoch 500 (500 additional epochs; evals after 250, 500 epochs) ---
        {
            "mult": 2,
            "pretrain_ep": 0,
            "base_name": f"cm_clus-2P_enc-{layers}_dec-{layers}_pretrain-0_unfreeze-warm_ft_ep-500_v7",
            "resume_epoch": 500,
            "ft_epochs": 500,
            "eval_epochs": [250, 500],
        },
        {
            "mult": 4,
            "pretrain_ep": 0,
            "base_name": f"cm_clus-4P_enc-{layers}_dec-{layers}_pretrain-0_unfreeze-progressive_ft_ep-500_v7",
            "resume_epoch": 500,
            "ft_epochs": 500,
            "eval_epochs": [250, 500],
        },
    ]

    # Retrieve base v7 cross-modal fine-tuning jobs to link the DAG
    v7_outputs = get_v7_cross_modal_jobs()
    v7_ft_jobs = v7_outputs.get("finetune_jobs", {}) if isinstance(v7_outputs, dict) else {}

    finetune_jobs = {}
    configs_meta = {}

    for base in base_checkpoints:
        mult = base["mult"]
        pretrain_ep = base["pretrain_ep"]
        base_name = base["base_name"]
        resume_epoch = base["resume_epoch"]
        ft_epochs = base["ft_epochs"]
        eval_epochs = base["eval_epochs"]
        clus_str = f"{mult}P"
        k_clusters = mult * 41

        audio_emb_path = get_audio_embeddings_path(cluster_mult=mult, init_method="freq")
        text_emb_path = get_text_embeddings_path()
        audio_out_dim = max(train_data.datastreams["data"].vocab_size, k_clusters)

        # Resolve checkpoint path: through Sisyphus job output if present, or alias path
        if base_name in v7_ft_jobs and resume_epoch in v7_ft_jobs[base_name].out_checkpoints:
            ckpt_path = v7_ft_jobs[base_name].out_checkpoints[resume_epoch].path
        else:
            ckpt_path = tk.Path(
                f"/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/alias/unsup_audio_cluster_to_phoneme/librispeech/config_librispeech_960_w_sil_denoise_pretrain_unfreeze_v7_cross_modal/{base_name}/training/output/models/epoch.{resume_epoch:03d}.pt"
            )

        for lr_val, lr_str in constant_lrs:
            train_name = f"cm_clus-{clus_str}_enc-{layers}_dec-{layers}_pretrain-{pretrain_ep}_unfreeze-warm_constlr-{lr_str}_from-ep{resume_epoch}_ft_ep-{ft_epochs}_v7"

            config = copy.deepcopy(base_config)
            config["model_args"].update({
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "discriminator_type": None,
                "codebook_opts": {"codebook_prob": 0.0},
                "audio_embeddings_path": audio_emb_path,
                "text_embeddings_path": text_emb_path,
                "audio_out_dim": audio_out_dim,
                "cross_modal_emb_dim": 300,
                "freeze_cross_modal_embeddings": True,
            })

            config["train_args"].update({
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": 0,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.0,
                "pretrain_adv_loss_scale": 0.0,

                # Fully unfrozen architecture
                "gradual_unfreeze": False,
                "freeze_encoder": False,

                "bt_buffer_size_steps": 10,
                "bt_train_iterations": 50,
                "text_masking_opts": {"mask_prob": 0.3, "min_span": 2, "max_span": 10, "expand": True, "insert_prob": 0.1},
                "audio_masking_opts": {"mask_prob": 0.3, "min_span": 4, "max_span": 20, "expand": True, "insert_prob": 0.1},
            })

            config["training"].update({
                "batch_size": 15000,
                "grad_scaler": None,
                "accum_grad_multiple_step": 2,
                "__num_epochs": ft_epochs,
                "preload_from_files": {"": {"filename": ckpt_path}},
            })

            # Constant learning rate schedule across all ft_epochs
            config["training"]["__lr_opts"] = {
                "type": "dyn_lr_piecewise_linear",
                "piecewise_epochs": [0, ft_epochs],
                "piecewise_values": [lr_val, lr_val],
            }

            config["recog_rqmt"] = {"time": 48, "mem": 64, "cpu": 12, "device": "cpu"}
            config.setdefault("train_rqmt", {})["mem_rqmt"] = 64

            ft_train_job = run_experiment(
                training_name=f"{prefix_name}/{train_name}",
                config=config,
                train_data=train_data,
                test_data_dict=test_data_dict,
                keep_epochs=eval_epochs,
                skip_eval=False,
                rasr_recog_opts=None,
                vis_epochs=eval_epochs,
                vis_kwargs={"cosine_similarity_summary": True},
                additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            )
            finetune_jobs[train_name] = ft_train_job

            configs_meta[train_name] = {
                "base_checkpoint": f"{base_name}@epoch_{resume_epoch}",
                "cluster_mult": clus_str,
                "cluster_k": k_clusters,
                "init_method": "frequency",
                "layers": layers,
                "pretrain_ep": pretrain_ep,
                "unfreeze_mode": "warm (fully unfrozen)",
                "learning_rate": lr_str,
                "ft_epochs": ft_epochs,
                "cumulative_epochs": resume_epoch + ft_epochs,
                "frozen_embeddings": True,
                "emb_dim": 300,
                "disc_type": "None",
                "codebook_prob": 0.0,
            }

    # -------------------------------------------------------------------------
    # Results Summary (Excel & CSV Aggregator)
    # -------------------------------------------------------------------------
    from ....hpo_summary import HpoResultsExcelJob

    all_summary_eval_epochs = sorted(list({ep for b in base_checkpoints for ep in b["eval_epochs"]}))
    max_ft_epochs = max(b["ft_epochs"] for b in base_checkpoints)

    excel_job = HpoResultsExcelJob(
        configs_meta=configs_meta,
        train_jobs=finetune_jobs,
        eval_epochs=all_summary_eval_epochs,
        target_num_epochs=max_ft_epochs,
        prefix_name=prefix_name,
        report_name="unsup_v7_constant_lr_summary",
    )

    return {
        "excel_job": excel_job,
        "finetune_jobs": finetune_jobs,
    }

