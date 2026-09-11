import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ....train_exp import run_experiment
from ..data.common import build_training_datasets_clean100, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, test_data_dict

from sisyphus import tk

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering=None,
)

# LibriSpeech 100-clean dataset (No LM data needed since no discriminator is used)
train_data_clean100 = build_training_datasets_clean100(
    sil_prob=0.25,
    surround_w_sil=True,
    settings=settings,
    include_lm_data=False,
)

# Extend the base config to use the dedicated v6 denoising training step module with perplexity logging
base_config = copy.deepcopy(base_config)
base_config["random_seed"] = 42
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_sup_asr_v6.train_step"


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    pretrain_num_epochs = 100
    asr_num_epochs = 200

    # -------------------------------------------------------------------------
    # 30 HPO Configurations for Codebook and Denoising Pretraining -> ASR
    # -------------------------------------------------------------------------
    raw_configs = [
        # --- Group A: Codebook Probability Variations (prob: 0.25, 0.5, 0.75, 1.0) on 3-layer ---
        ("cb_prob-0.25_v320_g2_div-0.1_l3", 3, {"codebook_prob": 0.25, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v320_g2_div-0.1_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.75_v320_g2_div-0.1_l3", 3, {"codebook_prob": 0.75, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-1.00_v320_g2_div-0.1_l3", 3, {"codebook_prob": 1.00, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),

        # --- Group B: Codebook Architecture Variations (V x G) on 3-layer ---
        ("cb_prob-0.50_v160_g2_div-0.1_l3", 3, {"codebook_prob": 0.50, "latent_vars": 160, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v640_g1_div-0.1_l3", 3, {"codebook_prob": 0.50, "latent_vars": 640, "latent_groups": 1, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v1024_g1_div-0.1_l3", 3, {"codebook_prob": 0.50, "latent_vars": 1024, "latent_groups": 1, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v512_g2_div-0.1_l3", 3, {"codebook_prob": 0.50, "latent_vars": 512, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-1.00_v640_g1_div-0.1_l3", 3, {"codebook_prob": 1.00, "latent_vars": 640, "latent_groups": 1, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-1.00_v1024_g1_div-0.1_l3", 3, {"codebook_prob": 1.00, "latent_vars": 1024, "latent_groups": 1, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),

        # --- Group C: Diversity Loss Weight Variations (0.0, 0.01, 0.05, 0.2) on 3-layer ---
        ("cb_prob-0.50_v320_g2_div-0.0_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.0, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v320_g2_div-0.01_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.01, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v320_g2_div-0.05_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.05, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v320_g2_div-0.2_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.2, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-1.00_v320_g2_div-0.05_l3", 3, {"codebook_prob": 1.00, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.05, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-1.00_v320_g2_div-0.2_l3", 3, {"codebook_prob": 1.00, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.2, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),

        # --- Group D: Quantizer Projection Depth & Factor on 3-layer ---
        ("cb_prob-0.50_v320_g2_depth-1_fac-1_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 1}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v320_g2_depth-2_fac-3_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 2, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-1.00_v320_g2_depth-2_fac-3_l3", 3, {"codebook_prob": 1.00, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 2, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),

        # --- Group E: Denoising Masking Strength during Pretraining on 3-layer ---
        ("cb_prob-0.50_v320_g2_mask-p0.2_s5_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.2, "min_span": 1, "max_span": 5}),
        ("cb_prob-0.50_v320_g2_mask-p0.3_s10_l3", 3, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.3, "min_span": 2, "max_span": 10}),
        ("cb_prob-1.00_v320_g2_mask-p0.3_s10_l3", 3, {"codebook_prob": 1.00, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.3, "min_span": 2, "max_span": 10}),

        # --- Group F: 6-Layer Architectures (Scaling Capacity) ---
        ("cb_prob-0.25_v320_g2_div-0.1_l6", 6, {"codebook_prob": 0.25, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v320_g2_div-0.1_l6", 6, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.75_v320_g2_div-0.1_l6", 6, {"codebook_prob": 0.75, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-1.00_v320_g2_div-0.1_l6", 6, {"codebook_prob": 1.00, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v640_g1_div-0.1_l6", 6, {"codebook_prob": 0.50, "latent_vars": 640, "latent_groups": 1, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v1024_g1_div-0.1_l6", 6, {"codebook_prob": 0.50, "latent_vars": 1024, "latent_groups": 1, "quantizer_depth": 1, "quantizer_factor": 3}, 0.1, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v320_g2_div-0.05_l6", 6, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.05, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
        ("cb_prob-0.50_v320_g2_div-0.2_l6", 6, {"codebook_prob": 0.50, "latent_vars": 320, "latent_groups": 2, "quantizer_depth": 1, "quantizer_factor": 3}, 0.2, {"mask_prob": 0.1, "min_span": 1, "max_span": 1}),
    ]

    assert len(raw_configs) == 30, f"Expected 30 configurations, got {len(raw_configs)}"

    # Build ablations list
    ablations = []
    for tag, layers, cb_opts, div_scale, mask_opts in raw_configs:
        name = f"v6_codebook_opt_{tag}"
        model_args = {
            "num_enc_layers": layers,
            "num_text_dec_layers": layers,
            "num_audio_dec_layers": layers,
            "codebook_opts": cb_opts,
        }
        train_args = {
            "codebook_diversity_loss_scale": div_scale,
            "denoise_pretrain_epochs": pretrain_num_epochs,
            "pretrain_codebook_prob": cb_opts.get("codebook_prob", 0.5),
            "pretrain_codebook_diversity_loss_scale": div_scale,
            "adv_loss_scale": 0.0,
            "pretrain_adv_loss_scale": 0.0,
            "use_lm_for_asr_adv": False,
            "pretrain_mask_opts": mask_opts,
        }
        training_args = {
            "batch_size": 4000,
        }
        ablations.append((name, model_args, train_args, training_args))

    # --- PHASE 1: Pretraining Jobs (100 Epochs, Denoising + Codebooks, NO Discriminator) ---
    pretrain_jobs = {}
    pretrain_vis_eps = [25, 50, 75, 100]
    pretrain_keep_eps = [25, 50, 75, 100]

    for ablation in ablations:
        train_name, model_args, train_args, training_args = ablation
        layers = model_args["num_enc_layers"]
        cb_opts = model_args["codebook_opts"]
        div_scale = train_args["pretrain_codebook_diversity_loss_scale"]
        mask_opts = train_args.get("pretrain_mask_opts", {"mask_prob": 0.1, "min_span": 1, "max_span": 1})

        config = copy.deepcopy(base_config)
        config["model_args"].update({
            "num_enc_layers": layers,
            "num_text_dec_layers": layers,
            "num_audio_dec_layers": layers,
            "codebook_opts": cb_opts,
        })
        config["train_args"].update({
            "codebook_diversity_loss_scale": 0.0,
            "denoise_pretrain_epochs": pretrain_num_epochs,
            "pretrain_codebook_prob": cb_opts.get("codebook_prob", 0.5),
            "pretrain_codebook_diversity_loss_scale": div_scale,
            "adv_loss_scale": 0.0,
            "pretrain_adv_loss_scale": 0.0,
            "use_lm_for_asr_adv": False,

            "pseudo_audio_text_ce_loss_scale": 0.0,
            "pseudo_text_audio_ce_loss_scale": 0.0,
            "supervised_asr_ce_loss_scale": 0.0,
            "asr_loss_warmup_steps": 0,

            "text_masking_opts": mask_opts,
            "audio_masking_opts": mask_opts,
        })
        config["training"].update({"batch_size": training_args.get("batch_size", 4000), "grad_scaler": None})

        piecewise_epochs = [
            0,
            0.45 * pretrain_num_epochs,
            0.9 * pretrain_num_epochs,
            pretrain_num_epochs,
        ]
        piecewise_values = [1e-5, 1e-3, 1e-5, 1e-6]
        config["training"]["__lr_opts"] = {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": piecewise_epochs,
            "piecewise_values": piecewise_values,
        }
        config["training"]["__num_epochs"] = pretrain_num_epochs
        config["training"]["save_interval"] = 10
        config["recog_rqmt"] = {"time": 24, "mem": 24, "cpu": 8, "gpu_mem": 11, "gpu": 1}

        p_name = f"pretrain_{train_name}_ep-{pretrain_num_epochs}"

        # Visualizations (PCA, cross-attention, variance analysis) during pretraining
        p_job = run_experiment(
            training_name=f"{prefix_name}/{p_name}",
            config=config,
            train_data=train_data_clean100,
            test_data_dict=test_data_dict,
            keep_epochs=pretrain_keep_eps,
            skip_eval=True,
            rasr_recog_opts=None,
            vis_epochs=pretrain_vis_eps,
            vis_kwargs={"cosine_similarity_summary": True},
        )
        pretrain_jobs[train_name] = p_job

    # --- PHASE 2: Supervised ASR Training (200 Epochs, Codebooks Active, NO Discriminator) ---
    asr_keep_eps = [50, 100, 150, 200]
    asr_vis_eps = [50, 100, 150, 200]
    asr_jobs = {}

    for ablation in ablations:
        train_name, model_args, train_args, training_args = ablation

        asr_train_name = f"asr_{train_name}_ep-{asr_num_epochs}"
        config = copy.deepcopy(base_config)
        config["model_args"].update(model_args)
        config["train_args"].update({
            "pseudo_audio_text_ce_loss_scale": 0.0,
            "pseudo_text_audio_ce_loss_scale": 0.0,
            "supervised_asr_ce_loss_scale": 1.0,
            "asr_loss_warmup_steps": 2000,
            "adv_loss_scale": 0.0,
            "pretrain_adv_loss_scale": 0.0,
            "use_lm_for_asr_adv": False,
        })
        config["train_args"].update(train_args)
        config["train_args"]["denoise_pretrain_epochs"] = 0  # Pretrain phase is finished

        # Masking for fine-tuning
        config["train_args"].update({
            "text_masking_opts": {
                "mask_prob": 0.3,
                "min_span": 2,
                "max_span": 10,
            },
            "audio_masking_opts": {
                "mask_prob": 0.3,
                "min_span": 4,
                "max_span": 20,
            },
        })
        config["training"].update(training_args)
        config["training"]["grad_scaler"] = None

        config["recog_rqmt"] = {"time": 24, "mem": 24, "cpu": 8}
        config["training"]["__num_epochs"] = asr_num_epochs
        config["training"]["save_interval"] = 10

        # Preload from the 100-epoch pretraining checkpoint
        p_job = pretrain_jobs[train_name]
        config["training"]["preload_from_files"] = {"": p_job.out_checkpoints[pretrain_num_epochs].path}

        piecewise_epochs = [
            0,
            0.45 * asr_num_epochs,
            0.9 * asr_num_epochs,
            asr_num_epochs,
        ]
        piecewise_values = [1e-5, 1e-3, 1e-5, 1e-6]
        config["training"]["__lr_opts"] = {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": piecewise_epochs,
            "piecewise_values": piecewise_values,
        }

        # Standalone PER evaluation and visualizations
        asr_job = run_experiment(
            training_name=f"{prefix_name}/{asr_train_name}",
            config=config,
            train_data=train_data_clean100,
            test_data_dict=test_data_dict,
            keep_epochs=asr_keep_eps,
            skip_eval=False,
            rasr_recog_opts=None,
            vis_epochs=asr_vis_eps,
            vis_kwargs={"cosine_similarity_summary": True},
        )
        asr_jobs[asr_train_name] = asr_job

    # --- PHASE 3: Joined Results Excel & CSV Aggregator Job ---
    from ...hpo_summary import HpoResultsExcelJob

    configs_meta = {}
    for ablation in ablations:
        train_name, model_args, train_args, _ = ablation
        asr_train_name = f"asr_{train_name}_ep-{asr_num_epochs}"
        cb_opts = model_args["codebook_opts"]
        configs_meta[asr_train_name] = {
            "layers": model_args["num_enc_layers"],
            "cb_prob": cb_opts.get("codebook_prob", 0.5),
            "cb_vars": cb_opts.get("latent_vars", 320),
            "cb_groups": cb_opts.get("latent_groups", 2),
            "div_loss": train_args.get("codebook_diversity_loss_scale", 0.0),
        }

    HpoResultsExcelJob(
        configs_meta=configs_meta,
        train_jobs=asr_jobs,
        eval_epochs=asr_keep_eps,
        target_num_epochs=asr_num_epochs,
        prefix_name=prefix_name,
    )
