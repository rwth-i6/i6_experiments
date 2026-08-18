import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, get_keep_epochs, test_data_dict, base_num_epochs
# import the baseline

from sisyphus import tk

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering=None,
)

#: ablation study, sil prob = 0 (remember meta gan paper, silence insertion part) 
# surrounding silence definition as in the paper
# Create two versions of the dataset: one without LM data for faster iteration/pretraining,
# and one with LM data for adversarial ASR.
train_data_no_lm = build_training_datasets(sil_prob=0.25, surround_w_sil=True, settings=settings, include_lm_data=False)
train_data_lm = build_training_datasets(sil_prob=0.25, surround_w_sil=True, settings=settings, include_lm_data=True)

# Extend the base config to use the new denoising training step module
base_config = copy.deepcopy(base_config)
base_config["random_seed"] = 42
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_sup_asr.train_step"


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    ablations = [
        # Debug ablation: 3 layers, disc in all stages, 5 pretrain epochs
        (
            "baseline_disc_enc-3_dec-3_denoise_ep-5_v5.1_lmdata_adv_asr_DEBUG",
            {
                "num_enc_layers": 3,
                "num_text_dec_layers": 3,
                "num_audio_dec_layers": 3,
                "discriminator_type": "lstm",
                "codebook_opts": {"codebook_prob": 0.0},
            },
            {
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": 5,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.1,
                "pretrain_adv_loss_scale": 0.1,
                "use_lm_for_asr_adv": True,
            },
            {
                "batch_size": 4000,
            },
            True  # is_debug
        )
    ] + [
        # no discriminator for both pretraining and ASR; only pretrain - asr
        (
            f"baseline_NO_disc_enc-{layers}_dec-{layers}_denoise_epoch-{pretrain_epochs}_v5.1",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "codebook_opts": {"codebook_prob": 0.0},
            },
            {
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": pretrain_epochs,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.0,
                "pretrain_adv_loss_scale": 0.0,
            },
            {
                "batch_size": 4000,
            }
        ) for layers in [3, 6] for pretrain_epochs in [100, 200, 500]
    ] + [
        # With discriminator in both pretraining and ASR
        (
            f"baseline_disc_enc-{layers}_dec-{layers}_denoise_epoch-{pretrain_epochs}_v5.1",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "discriminator_type": "lstm",
                "codebook_opts": {"codebook_prob": 0.0},
            },
            {
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": pretrain_epochs,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.1,
                "pretrain_adv_loss_scale": 0.1,
                "use_lm_for_asr_adv": True,
            },
            {
                "batch_size": 4000,
            }
        ) for layers in [3, 6] for pretrain_epochs in [100, 200, 500]
    ] + [
        # With discriminator in pretraining only, not in ASR
        (
            f"baseline_disc_enc-{layers}_dec-{layers}_denoise_epoch-{pretrain_epochs}_nodiscasr_v5.1",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "discriminator_type": "lstm",
                "codebook_opts": {"codebook_prob": 0.0},
            },
            {
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": pretrain_epochs,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.0,
                "pretrain_adv_loss_scale": 0.1,
            },
            {
                "batch_size": 4000,
            }
        ) for layers in [3, 6] for pretrain_epochs in [100, 200, 500]
    ] + [
        # With discriminator in both pretraining and ASR, WITH CODEBOOK
        (
            f"baseline_disc_enc-{layers}_dec-{layers}_denoise_epoch-{pretrain_epochs}_codebook_v5.1",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "discriminator_type": "lstm",
                "codebook_opts": {"codebook_prob": 0.5},
            },
            {
                "codebook_diversity_loss_scale": 0.1,
                "denoise_pretrain_epochs": pretrain_epochs,
                "pretrain_codebook_prob": 0.5,
                "pretrain_codebook_diversity_loss_scale": 0.1,
                "adv_loss_scale": 0.1,
                "pretrain_adv_loss_scale": 0.1,
                "use_lm_for_asr_adv": True,
            },
            {
                "batch_size": 4000,
            }
        ) for layers in [3, 6] for pretrain_epochs in [100, 200, 500]
    ] + [
        # Fully supervised ASR, no pretraining, no discriminator
        (
            f"baseline_sup_asr_-nopretrain-enc_{layers}_dec-{layers}_v5.1",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "codebook_opts": {"codebook_prob": 0.0},
            },
            {
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": 0,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.0,
                "pretrain_adv_loss_scale": 0.0,
                "asr_loss_warmup_steps": 0,
            },
            {
                "batch_size": 4000,
            }, 
        ) for layers in [3, 6]
    ]


    # --- PHASE 1: Pretraining Jobs ---
    unique_pretrains = set()
    for ablation in ablations:
        train_args = ablation[2]
        pretrain_epochs = train_args.get("denoise_pretrain_epochs", 0)
        layers = ablation[1]["num_enc_layers"]
        if pretrain_epochs > 0:
            unique_pretrains.add((layers, pretrain_epochs))
            
    pretrain_jobs = {}
    for layers, pretrain_epochs in unique_pretrains:
        config = copy.deepcopy(base_config)
        config["model_args"].update({
            "num_enc_layers": layers,
            "num_text_dec_layers": layers,
            "num_audio_dec_layers": layers,
            "discriminator_type": "lstm",
            "codebook_opts": {"codebook_prob": 0.0},
        })
        config["train_args"].update({
            "codebook_diversity_loss_scale": 0.0,
            "denoise_pretrain_epochs": pretrain_epochs,
            "pretrain_codebook_prob": 0.0,
            "pretrain_codebook_diversity_loss_scale": 0.0,
            "adv_loss_scale": 0.0,
            "pretrain_adv_loss_scale": 0.1,
            "use_lm_for_asr_adv": False,
            
            "pseudo_audio_text_ce_loss_scale": 0.0,
            "pseudo_text_audio_ce_loss_scale": 0.0,
            "supervised_asr_ce_loss_scale": 0.0,
            "asr_loss_warmup_steps": 0,
            
            "text_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
            "audio_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
        })
        config["training"].update({"batch_size": 4000, "grad_scaler": None})
        
        piecewise_epochs = [
            0,
            0.45 * pretrain_epochs,
            0.9 * pretrain_epochs,
            pretrain_epochs
        ]
        piecewise_values = [1e-5, 1e-3, 1e-5, 1e-6]
        config["training"]["__lr_opts"] = {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": piecewise_epochs,
            "piecewise_values": piecewise_values,
        }
        config["training"]["__num_epochs"] = pretrain_epochs
        config["recog_rqmt"] = {"time": 48, "mem": 24, "cpu": 8, "gpu_mem": 11, "gpu": 1}
        config.setdefault("train_rqmt", {})["mem_rqmt"] = 24
        
        train_name = f"pretrain_enc-{layers}_dec-{layers}_ep-{pretrain_epochs}_v5.1"
        
        train_job = run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data_no_lm,
            test_data_dict=test_data_dict,
            keep_epochs=[pretrain_epochs],
            skip_eval=True,
            rasr_recog_opts=None,
            vis_epochs=[],
        )
        pretrain_jobs[(layers, pretrain_epochs)] = train_job

    # --- PHASE 2: ASR Finetuning Jobs ---

    for ablation in ablations:
        train_name = ablation[0]
        model_args = ablation[1]
        train_args = ablation[2]
        training_args = ablation[3]
        is_debug = ablation[4] if (len(ablation) > 4 and ablation[4]) else False

        config = copy.deepcopy(base_config)
        config["model_args"].update(model_args)
        config["train_args"].update({
            "pseudo_audio_text_ce_loss_scale": 0.0,
            "pseudo_text_audio_ce_loss_scale": 0.0,
            "supervised_asr_ce_loss_scale": 1.0,
            "asr_loss_warmup_steps": 2000,
        })
        config["train_args"].update(train_args)
        config["train_args"].update({
            "text_masking_opts": {
                "mask_prob": 0.1,
                "min_span": 1,
                "max_span": 1,
            },
            "audio_masking_opts": {
                "mask_prob": 0.1,
                "min_span": 1,
                "max_span": 1,
            },
        })
        config["training"].update(training_args)
        config["training"]["grad_scaler"] = None

        if is_debug:
            config["training"]["__num_gpus"] = 1
        
        use_lm = train_args.get("use_lm_for_asr_adv", False)
        current_train_data = train_data_lm if use_lm else train_data_no_lm

        config["recog_rqmt"] = {"time": 48, "mem": 24, "cpu": 8}
        config.setdefault("train_rqmt", {})["mem_rqmt"] = 24
        
        pretrain_ep = train_args.get("denoise_pretrain_epochs", 0)
        
        config["training"]["__num_epochs"] = base_num_epochs
        if pretrain_ep > 0:
            layers = model_args["num_enc_layers"]
            p_job = pretrain_jobs[(layers, pretrain_ep)]
            config["training"]["preload_from_files"] = {"": {"filename": p_job.out_checkpoints[pretrain_ep].path}}

        piecewise_epochs = [
            0,
            0.45 * base_num_epochs,
            0.9 * base_num_epochs,
            base_num_epochs
        ]
        piecewise_values = [1e-5, 1e-3, 1e-5, 1e-6]
        config["training"]["__lr_opts"] = {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": piecewise_epochs,
            "piecewise_values": piecewise_values,
        }

        keep_eps = get_keep_epochs(base_num_epochs)
        if keep_eps is None: keep_eps = []
        vis_eps = [250, 500, 750, 1000]
            
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=current_train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_eps,
            skip_eval=False,
            rasr_recog_opts=None,
            vis_epochs=vis_eps,
            vis_kwargs={"cosine_similarity_summary": True},
        )

    # --- New Ablations with longer masking spans ---
    for ablation in ablations:
        train_name = ablation[0]
        model_args = ablation[1]
        train_args = ablation[2]
        training_args = ablation[3]
        is_debug = ablation[4] if (len(ablation) > 4 and ablation[4]) else False

        new_train_name = train_name + "_longer_spans"
        config = copy.deepcopy(base_config)
        config["model_args"].update(model_args)
        config["train_args"].update({
            "pseudo_audio_text_ce_loss_scale": 0.0,
            "pseudo_text_audio_ce_loss_scale": 0.0,
            "supervised_asr_ce_loss_scale": 1.0,
            "asr_loss_warmup_steps": 2000,
        })
        config["train_args"].update(train_args)
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

        if is_debug:
            config["training"]["__num_gpus"] = 1

        use_lm = train_args.get("use_lm_for_asr_adv", False)
        current_train_data = train_data_lm if use_lm else train_data_no_lm
        
        config["recog_rqmt"] = {"time": 48, "mem": 24, "cpu": 8}
        config.setdefault("train_rqmt", {})["mem_rqmt"] = 24
        
        pretrain_ep = train_args.get("denoise_pretrain_epochs", 0)
        
        config["training"]["__num_epochs"] = base_num_epochs
        if pretrain_ep > 0:
            layers = model_args["num_enc_layers"]
            p_job = pretrain_jobs[(layers, pretrain_ep)]
            config["training"]["preload_from_files"] = {"": {"filename": p_job.out_checkpoints[pretrain_ep].path}}

        piecewise_epochs = [
            0,
            0.45 * base_num_epochs,
            0.9 * base_num_epochs,
            base_num_epochs
        ]
        piecewise_values = [1e-5, 1e-3, 1e-5, 1e-6]
        config["training"]["__lr_opts"] = {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": piecewise_epochs,
            "piecewise_values": piecewise_values,
        }

        keep_eps = get_keep_epochs(base_num_epochs)
        if keep_eps is None: keep_eps = []
        vis_eps = [250, 500, 750, 1000]
            
        run_experiment(
            training_name=f"{prefix_name}/{new_train_name}",
            config=config,
            train_data=current_train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_eps,
            skip_eval=False,
            rasr_recog_opts=None,
            vis_epochs=vis_eps,
            vis_kwargs={"cosine_similarity_summary": True},
        )
