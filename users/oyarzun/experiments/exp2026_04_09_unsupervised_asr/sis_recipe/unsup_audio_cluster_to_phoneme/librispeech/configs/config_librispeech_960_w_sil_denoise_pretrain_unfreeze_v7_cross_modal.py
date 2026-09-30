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
base_config["save_interval"] = 50
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
    # ABLATION SPECIFICATION (Version 7: Cross-Modal Frozen Embeddings)
    #
    # 1. Phoneme inventory cluster multipliers: 1P (41), 2P (82), 4P (164)
    #    with frequency-based cross-modal embeddings (init_method='freq')
    # 2. Architecture: 6 layers (enc-6, dec-6)
    # 3. Pretraining: 0 (no pretrain), 100, and 500 epochs denoising only
    # 4. Regularization: No codebooks nor discriminator (pure denoising + BT)
    # 5. Post-pretraining: 500 epochs backtranslation
    # 6. Encoder unfreeze mode:
    #      - Progressive unfreezing (gradual_unfreeze=True)
    #      - Encoder fully warm (gradual_unfreeze=False, freeze_encoder=False)
    # =========================================================================

    layers = 6
    ft_epochs = 500
    cluster_multipliers = [1, 2, 4]

    ablations = []
    for mult in cluster_multipliers:
        k_clusters = mult * 41
        clus_str = f"{mult}P"
        audio_emb_path = get_audio_embeddings_path(cluster_mult=mult, init_method="freq")
        text_emb_path = get_text_embeddings_path()
        audio_out_dim = max(train_data.datastreams["data"].vocab_size, k_clusters)

        for pretrain_epochs in [0, 100, 500]:
            for unfreeze in [True, False]:
                unfreeze_str = "progressive" if unfreeze else "warm"
                train_name = f"cm_clus-{clus_str}_enc-{layers}_dec-{layers}_pretrain-{pretrain_epochs}_unfreeze-{unfreeze_str}_ft_ep-{ft_epochs}_v7"

                model_args = {
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
                }

                train_args = {
                    "codebook_diversity_loss_scale": 0.0,
                    "denoise_pretrain_epochs": pretrain_epochs,
                    "pretrain_codebook_prob": 0.0,
                    "pretrain_codebook_diversity_loss_scale": 0.0,
                    "adv_loss_scale": 0.0,
                    "pretrain_adv_loss_scale": 0.0,

                    "gradual_unfreeze": unfreeze,
                    "freeze_encoder": False,
                    "gradual_unfreeze_proportion": 0.8,
                    "gradual_unfreeze_start_iter": int(200_000 * 0.5),
                    "gradual_unfreeze_end_iter": int(200_000 * 0.9),

                    "bt_buffer_size_steps": 10,
                    "bt_train_iterations": 50,
                }

                ablations.append((train_name, model_args, train_args, pretrain_epochs, ft_epochs, mult))

    # -------------------------------------------------------------------------
    # PHASE 1: Pretraining Jobs (Denoising Only, 100 and 500 Epochs per Cluster Size)
    # -------------------------------------------------------------------------
    pretrain_jobs = {}
    pretrain_durations = sorted(list({p_ep for _, _, _, p_ep, _, _ in ablations if p_ep > 0}))

    for mult in cluster_multipliers:
        k_clusters = mult * 41
        clus_str = f"{mult}P"
        audio_emb_path = get_audio_embeddings_path(cluster_mult=mult, init_method="freq")
        text_emb_path = get_text_embeddings_path()
        audio_out_dim = max(train_data.datastreams["data"].vocab_size, k_clusters)

        for pretrain_ep in pretrain_durations:
            p_name = f"pretrain_cm_clus-{clus_str}_enc-{layers}_dec-{layers}_ep-{pretrain_ep}_v7"

            p_config = copy.deepcopy(base_config)
            p_config["model_args"].update({
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
            p_config["train_args"].update({
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": pretrain_ep,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.0,
                "pretrain_adv_loss_scale": 0.0,

                # Standard masking parameters for denoising autoencoding
                "text_masking_opts": {"mask_prob": 0.3, "min_span": 2, "max_span": 10, "expand": True, "insert_prob": 0.1},
                "audio_masking_opts": {"mask_prob": 0.3, "min_span": 4, "max_span": 20, "expand": True, "insert_prob": 0.1},
            })

            p_config["training"].update({"batch_size": 30000, "grad_scaler": None, "accum_grad_multiple_step": 1})

            p_piecewise_epochs = [
                0,
                0.45 * pretrain_ep,
                0.9 * pretrain_ep,
                pretrain_ep,
            ]
            p_piecewise_values = [1e-5, 1e-3, 1e-5, 1e-6]
            p_config["training"]["__lr_opts"] = {
                "type": "dyn_lr_piecewise_linear",
                "piecewise_epochs": p_piecewise_epochs,
                "piecewise_values": p_piecewise_values,
            }
            p_config["training"]["__num_epochs"] = pretrain_ep
            p_config["recog_rqmt"] = {"time": 48, "mem": 64, "cpu": 12, "device": "cpu"}
            p_config.setdefault("train_rqmt", {})["mem_rqmt"] = 64

            if pretrain_ep == 100:
                p_keep_eps = [100]
                p_vis_eps = [50, 100]
            else:
                p_keep_eps = [125, 250, 375, 500]
                p_vis_eps = [125, 250, 375, 500]

            p_train_job = run_experiment(
                training_name=f"{prefix_name}/{p_name}",
                config=p_config,
                train_data=train_data,
                test_data_dict=test_data_dict,
                keep_epochs=p_keep_eps,
                skip_eval=False,
                rasr_recog_opts=None,
                vis_epochs=p_vis_eps,
                vis_kwargs={"cosine_similarity_summary": True},
                additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            )
            pretrain_jobs[(mult, pretrain_ep)] = p_train_job

    # -------------------------------------------------------------------------
    # PHASE 2: Backtranslation Finetuning Jobs (500 Epochs)
    # -------------------------------------------------------------------------
    finetune_jobs = {}
    all_eval_epochs = [125, 250, 375, 500]

    for train_name, model_args, train_args, pretrain_ep, ft_epochs, mult in ablations:
        config = copy.deepcopy(base_config)
        config["model_args"].update(model_args)
        config["train_args"].update(train_args)

        # In Phase 2, denoise_pretrain_epochs is 0 so backtranslation runs immediately
        config["train_args"]["denoise_pretrain_epochs"] = 0

        config["train_args"].update({
            "text_masking_opts": {"mask_prob": 0.3, "min_span": 2, "max_span": 10, "expand": True, "insert_prob": 0.1},
            "audio_masking_opts": {"mask_prob": 0.3, "min_span": 4, "max_span": 20, "expand": True, "insert_prob": 0.1},
        })
        config["training"].update({"batch_size": 15000, "grad_scaler": None, "accum_grad_multiple_step": 2})
        config["recog_rqmt"] = {"time": 48, "mem": 64, "cpu": 12, "device": "cpu"}
        config.setdefault("train_rqmt", {})["mem_rqmt"] = 64
        config["training"]["__num_epochs"] = ft_epochs

        # Preload pretraining checkpoint if pretraining was performed
        if pretrain_ep > 0 and (mult, pretrain_ep) in pretrain_jobs:
            p_job = pretrain_jobs[(mult, pretrain_ep)]
            config["training"]["preload_from_files"] = {"": {"filename": p_job.out_checkpoints[pretrain_ep].path}}

        piecewise_epochs = [
            0,
            0.45 * ft_epochs,
            0.9 * ft_epochs,
            ft_epochs,
        ]
        piecewise_values = [1e-5, 1e-3, 1e-5, 1e-6]
        config["training"]["__lr_opts"] = {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": piecewise_epochs,
            "piecewise_values": piecewise_values,
        }

        ft_train_job = run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=all_eval_epochs,
            skip_eval=False,
            rasr_recog_opts=None,
            vis_epochs=all_eval_epochs,
            vis_kwargs={"cosine_similarity_summary": True},
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )
        finetune_jobs[train_name] = ft_train_job

    # -------------------------------------------------------------------------
    # PHASE 3: Joined Results Summary (Excel & CSV Aggregator)
    # -------------------------------------------------------------------------
    from ....hpo_summary import HpoResultsExcelJob

    configs_meta = {}
    for train_name, model_args, train_args, pretrain_ep, ft_epochs, mult in ablations:
        configs_meta[train_name] = {
            "cluster_mult": f"{mult}P",
            "cluster_k": mult * 41,
            "init_method": "frequency",
            "layers": model_args.get("num_enc_layers", 6),
            "pretrain_ep": pretrain_ep,
            "unfreeze_mode": "progressive" if train_args.get("gradual_unfreeze", False) else "warm",
            "ft_epochs": ft_epochs,
            "frozen_embeddings": model_args.get("freeze_cross_modal_embeddings", True),
            "emb_dim": model_args.get("cross_modal_emb_dim", 300),
            "disc_type": "None",
            "codebook_prob": 0.0,
        }

    excel_job = HpoResultsExcelJob(
        configs_meta=configs_meta,
        train_jobs=finetune_jobs,
        eval_epochs=all_eval_epochs,
        prefix_name=prefix_name,
        report_name="unsup_v7_cross_modal_summary",
    )
    
    return {
        "excel_job": excel_job,
        "finetune_jobs": finetune_jobs,
        "pretrain_jobs": pretrain_jobs,
    }
