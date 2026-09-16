import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.common.setups.serialization import PartialImport
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, get_keep_epochs, test_data_dict, base_num_epochs, alternate_batching

from sisyphus import tk

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering="laplace:.1000",
)

#: ablation study, sil prob = 0.25 (remember meta gan paper, silence insertion part) 
train_data = build_training_datasets(sil_prob=0.25, surround_w_sil=True, settings=settings)

# Extend the base config to use the v6 training step module
base_config = copy.deepcopy(base_config)
base_config["random_seed"] = 42
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_denoise_v6.train_step"
base_config["save_interval"] = 50
base_config["training"]["__num_gpus"] = 1
base_config["training"]["batch_size"] = 30000
base_config["training"]["max_seqs"] = 1000
base_config["train_rqmt"]["cpu_rqmt"] = 12
base_config["train_rqmt"]["gpu_mem"] = 80
base_config["train_rqmt"]["mem_rqmt"] = 64


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"
    
    ablations = []
    
    # 500-epoch codebook orthogonality and diversity loss weight ablations (6 layers, from scratch)
    for cb_div in [0.01, 0.001]:
        for cb_orth in [0.01, 0.001]:
            for unfreeze in [True, False]:
                train_name = f"disc-codebooks_enc-6_dec-6_ep-500_div-{cb_div}_orth-{cb_orth}_unfreeze-{unfreeze}_ft_ep-500_v6.1"

                model_args = {
                    "num_enc_layers": 6,
                    "num_text_dec_layers": 6,
                    "num_audio_dec_layers": 6,
                    "discriminator_type": None,
                    "codebook_opts": {"codebook_prob": 0.5},
                }

                train_args = {
                    "codebook_diversity_loss_scale": cb_div,
                    "codebook_orth_loss_scale": cb_orth,
                    "denoise_pretrain_epochs": 500,
                    "pretrain_codebook_prob": 0.5,
                    "pretrain_codebook_diversity_loss_scale": cb_div,
                    "pretrain_codebook_orth_loss_scale": cb_orth,
                    "adv_loss_scale": 0.0,
                    "pretrain_adv_loss_scale": 0.0,

                    "gradual_unfreeze": unfreeze,
                    "gradual_unfreeze_proportion": 0.8,
                    "gradual_unfreeze_start_iter": int(200_000 * 0.5),
                    "gradual_unfreeze_end_iter": int(200_000 * 0.9),

                    "bt_buffer_size_steps": 10,
                    "bt_train_iterations": 50,
                }

                ablations.append((train_name, model_args, train_args, 500))

    # --- PHASE 1: Pretraining Jobs ---
    unique_pretrains = set()
    for train_name, model_args, train_args, ft_epochs in ablations:
        layers = model_args["num_enc_layers"]
        ep = train_args["denoise_pretrain_epochs"]
        disc_type = model_args["discriminator_type"]
        cb_prob = model_args["codebook_opts"]["codebook_prob"]
        cb_div = train_args.get("pretrain_codebook_diversity_loss_scale", 0.0)
        cb_orth = train_args.get("pretrain_codebook_orth_loss_scale", 0.0)
        unique_pretrains.add((layers, ep, disc_type, cb_prob, cb_div, cb_orth))
        
    pretrain_jobs = {}
    for layers, pretrain_epochs, disc_type, cb_prob, cb_div, cb_orth in unique_pretrains:
        config = copy.deepcopy(base_config)
        config["model_args"].update({
            "num_enc_layers": layers,
            "num_text_dec_layers": layers,
            "num_audio_dec_layers": layers,
            "discriminator_type": disc_type,
            "codebook_opts": {"codebook_prob": cb_prob},
        })
        
        pre_adv_scale = 0.1 if disc_type is not None else 0.0
        
        config["train_args"].update({
            "codebook_diversity_loss_scale": cb_div,
            "codebook_orth_loss_scale": cb_orth,
            "denoise_pretrain_epochs": pretrain_epochs,
            "pretrain_codebook_prob": cb_prob,
            "pretrain_codebook_diversity_loss_scale": cb_div,
            "pretrain_codebook_orth_loss_scale": cb_orth,
            "adv_loss_scale": 0.0,
            "pretrain_adv_loss_scale": pre_adv_scale,
            
            # Masking ops for pretraining (expanded)
            "text_masking_opts": {"mask_prob": 0.3, "min_span": 2, "max_span": 10, "expand": True, "insert_prob": 0.1},
            "audio_masking_opts": {"mask_prob": 0.3, "min_span": 4, "max_span": 20, "expand": True, "insert_prob": 0.1},
        })
        
        # Batch size 30000 for optimal GPU memory and tensor core utilization on 80GB H100 GPUs
        config["training"].update({"batch_size": 30000, "grad_scaler": None, "accum_grad_multiple_step": 1})
        
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
        config["recog_rqmt"] = {"time": 48, "mem": 64, "cpu": 12, "device": "cpu"}
        config.setdefault("train_rqmt", {})["mem_rqmt"] = 64
        
        train_name = f"pretrain_disc-codebooks_enc-{layers}_dec-{layers}_ep-{pretrain_epochs}_div-{cb_div}_orth-{cb_orth}_v6.1"
        
        # 4 checkpoints for pretraining visualization (epochs 125, 250, 375, 500)
        keep_eps = [125, 250, 375, 500] if pretrain_epochs == 500 else [int(pretrain_epochs * p) for p in [0.25, 0.5, 0.75, 1.0]]
        vis_eps = list(keep_eps)
        
        train_job = run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_eps,
            skip_eval=False,
            rasr_recog_opts=None,
            vis_epochs=vis_eps,
            vis_kwargs={"cosine_similarity_summary": True},
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )
        pretrain_jobs[(layers, pretrain_epochs, disc_type, cb_prob, cb_div, cb_orth)] = train_job


    # --- PHASE 2: Backtranslation Finetuning Jobs ---
    finetune_jobs = {}
    all_eval_epochs = set()
    for train_name, model_args, train_args, ft_epochs in ablations:
        config = copy.deepcopy(base_config)
        config["model_args"].update(model_args)
        
        config["train_args"].update(train_args)
        
        # Set denoise_pretrain_epochs to 0 for phase 2 so it exclusively backtranslates
        config["train_args"]["denoise_pretrain_epochs"] = 0
        
        config["train_args"].update({
            # Masking ops for backtranslation (expanded)
            "text_masking_opts": {"mask_prob": 0.3, "min_span": 2, "max_span": 10, "expand": True, "insert_prob": 0.1},
            "audio_masking_opts": {"mask_prob": 0.3, "min_span": 4, "max_span": 20, "expand": True, "insert_prob": 0.1},
        })
        config["training"].update({"batch_size": 30000, "grad_scaler": None, "accum_grad_multiple_step": 1})
        
        config["recog_rqmt"] = {"time": 48, "mem": 64, "cpu": 12, "device": "cpu"}
        config.setdefault("train_rqmt", {})["mem_rqmt"] = 64
        
        config["training"]["__num_epochs"] = ft_epochs
        
        layers = model_args["num_enc_layers"]
        ep = train_args["denoise_pretrain_epochs"]
        disc_type = model_args["discriminator_type"]
        cb_prob = model_args["codebook_opts"]["codebook_prob"]
        cb_div = train_args.get("pretrain_codebook_diversity_loss_scale", 0.0)
        cb_orth = train_args.get("pretrain_codebook_orth_loss_scale", 0.0)
        
        p_job = pretrain_jobs[(layers, ep, disc_type, cb_prob, cb_div, cb_orth)]
        config["training"]["preload_from_files"] = {"": {"filename": p_job.out_checkpoints[ep].path}}

        piecewise_epochs = [
            0,
            0.45 * ft_epochs,
            0.9 * ft_epochs,
            ft_epochs
        ]
        piecewise_values = [1e-5, 1e-3, 1e-5, 1e-6]
        config["training"]["__lr_opts"] = {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": piecewise_epochs,
            "piecewise_values": piecewise_values,
        }

        if ft_epochs == 1000:
            keep_eps = [250, 500, 750, 1000]
            vis_eps = [250, 500, 750, 1000]
        elif ft_epochs == 500:
            keep_eps = [125, 250, 375, 500]
            vis_eps = [125, 250, 375, 500]
        else:
            keep_eps = [int(ft_epochs * p) for p in [0.25, 0.5, 0.75, 1.0]]
            vis_eps = list(keep_eps)

        for e in keep_eps:
            all_eval_epochs.add(e)
            
        train_job = run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_eps,
            skip_eval=False,
            rasr_recog_opts=None,
            vis_epochs=vis_eps,
            vis_kwargs={"cosine_similarity_summary": True},
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )
        finetune_jobs[train_name] = train_job

    # --- PHASE 3: Joined Results Excel & CSV Aggregator Job ---
    from ....hpo_summary import HpoResultsExcelJob

    configs_meta = {}
    for train_name, model_args, train_args, ft_epochs in ablations:
        cb_opts = model_args.get("codebook_opts", {})
        configs_meta[train_name] = {
            "layers": model_args.get("num_enc_layers", ""),
            "disc_type": str(model_args.get("discriminator_type", "None")),
            "adv_scale": train_args.get("adv_loss_scale", 0.0),
            "cb_prob": cb_opts.get("codebook_prob", 0.0),
            "cb_div": train_args.get("codebook_diversity_loss_scale", 0.0),
            "cb_orth": train_args.get("codebook_orth_loss_scale", 0.0),
            "unfreeze": train_args.get("gradual_unfreeze", False),
            "frozen_enc": train_args.get("freeze_encoder", False),
            "pretrain_ep": train_args.get("denoise_pretrain_epochs", 0),
            "num_epochs": ft_epochs,
        }

    HpoResultsExcelJob(
        configs_meta=configs_meta,
        train_jobs=finetune_jobs,
        eval_epochs=sorted(list(all_eval_epochs)),
        prefix_name=prefix_name,
        report_name="unsup_v6_codebooks_summary",
    )
