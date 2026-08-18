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

# Extend the base config to use the new v6 training step module
base_config = copy.deepcopy(base_config)
base_config["random_seed"] = 42
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_denoise_v6.train_step"


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"
    
    # We will build up our list of ablations
    # discriminator strategies: "neither", "lstm", "codebooks"
    # layers: 3, 6
    # pretrain_epochs: 100, 500
    # gradual_unfreeze: True, False
    
    ablations = []
    
    # We want 3x2x2x2 = 24 ablations
    for disc_strat in ["neither", "lstm", "codebooks"]:
        for layers in [3, 6]:
            for pretrain_epochs in [100, 500]:
                for unfreeze in [True, False]:
                    
                    if disc_strat == "neither":
                        disc_type = None
                        codebook_prob = 0.0
                        pretrain_codebook_prob = 0.0
                        codebook_div = 0.0
                        pretrain_codebook_div = 0.0
                        adv_scale = 0.0
                        pretrain_adv_scale = 0.0
                    elif disc_strat == "lstm":
                        disc_type = "lstm"
                        codebook_prob = 0.0
                        pretrain_codebook_prob = 0.0
                        codebook_div = 0.0
                        pretrain_codebook_div = 0.0
                        adv_scale = 0.1
                        pretrain_adv_scale = 0.1
                    elif disc_strat == "codebooks":
                        disc_type = None
                        codebook_prob = 0.5
                        pretrain_codebook_prob = 0.5
                        codebook_div = 0.1
                        pretrain_codebook_div = 0.1
                        adv_scale = 0.1  # Codebook uses adv scale in discrete train step
                        pretrain_adv_scale = 0.1
                        
                    train_name = f"disc-{disc_strat}_enc-{layers}_dec-{layers}_ep-{pretrain_epochs}_unfreeze-{unfreeze}_v6.1"
                    
                    model_args = {
                        "num_enc_layers": layers,
                        "num_text_dec_layers": layers,
                        "num_audio_dec_layers": layers,
                        "discriminator_type": disc_type,
                        "codebook_opts": {"codebook_prob": codebook_prob},
                    }
                    
                    train_args = {
                        "codebook_diversity_loss_scale": codebook_div,
                        "denoise_pretrain_epochs": pretrain_epochs,
                        "pretrain_codebook_prob": pretrain_codebook_prob,
                        "pretrain_codebook_diversity_loss_scale": pretrain_codebook_div,
                        "adv_loss_scale": adv_scale,
                        "pretrain_adv_loss_scale": pretrain_adv_scale,
                        
                        "gradual_unfreeze": unfreeze,
                        "gradual_unfreeze_proportion": 0.8,
                        "gradual_unfreeze_start_iter": int(400_000 * 0.5),
                        "gradual_unfreeze_end_iter": int(400_000 * 0.9),
                        
                        "bt_buffer_size_steps": 10,
                        "bt_train_iterations": 50,
                    }
                    
                    ablations.append((train_name, model_args, train_args))


    # --- PHASE 1: Pretraining Jobs ---
    unique_pretrains = set()
    for train_name, model_args, train_args in ablations:
        layers = model_args["num_enc_layers"]
        ep = train_args["denoise_pretrain_epochs"]
        disc_type = model_args["discriminator_type"]
        cb_prob = model_args["codebook_opts"]["codebook_prob"]
        # Include disc setup in pretraining key, because pretraining uses disc
        unique_pretrains.add((layers, ep, disc_type, cb_prob))
        
    pretrain_jobs = {}
    for layers, pretrain_epochs, disc_type, cb_prob in unique_pretrains:
        config = copy.deepcopy(base_config)
        config["model_args"].update({
            "num_enc_layers": layers,
            "num_text_dec_layers": layers,
            "num_audio_dec_layers": layers,
            "discriminator_type": disc_type,
            "codebook_opts": {"codebook_prob": cb_prob},
        })
        
        # Deduce scales for pretraining from cb_prob and disc_type
        pre_adv_scale = 0.1 if (disc_type == "lstm" or cb_prob > 0) else 0.0
        pre_cb_div = 0.1 if cb_prob > 0 else 0.0
        
        config["train_args"].update({
            "codebook_diversity_loss_scale": pre_cb_div,
            "denoise_pretrain_epochs": pretrain_epochs,
            "pretrain_codebook_prob": cb_prob,
            "pretrain_codebook_diversity_loss_scale": pre_cb_div,
            "adv_loss_scale": 0.0,
            "pretrain_adv_loss_scale": pre_adv_scale,
            
            # Masking ops for pretraining (expanded)
            "text_masking_opts": {"mask_prob": 0.3, "min_span": 2, "max_span": 10, "expand": True, "insert_prob": 0.1},
            "audio_masking_opts": {"mask_prob": 0.3, "min_span": 4, "max_span": 20, "expand": True, "insert_prob": 0.1},
        })
        
        # Use batch_size 4000 to match v5 supervised
        config["training"].update({"batch_size": 4000, "grad_scaler": None, "accum_grad_multiple_step": 1})
        
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
        
        strat_name = "neither"
        if disc_type == "lstm": strat_name = "lstm"
        elif cb_prob > 0: strat_name = "codebooks"
        
        train_name = f"pretrain_disc-{strat_name}_enc-{layers}_dec-{layers}_ep-{pretrain_epochs}_v6.1"
        
        train_job = run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=[pretrain_epochs],
            skip_eval=True,
            rasr_recog_opts=None,
            vis_epochs=[],
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )
        pretrain_jobs[(layers, pretrain_epochs, disc_type, cb_prob)] = train_job


    # --- PHASE 2: Backtranslation Finetuning Jobs ---

    for train_name, model_args, train_args in ablations:
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
        config["training"].update({"batch_size": 4000, "grad_scaler": None, "accum_grad_multiple_step": 1})
        
        config["recog_rqmt"] = {"time": 48, "mem": 24, "cpu": 8, "gpu_mem": 11, "gpu": 1}
        config.setdefault("train_rqmt", {})["mem_rqmt"] = 24
        
        config["training"]["__num_epochs"] = base_num_epochs
        
        layers = model_args["num_enc_layers"]
        ep = train_args["denoise_pretrain_epochs"]
        disc_type = model_args["discriminator_type"]
        cb_prob = model_args["codebook_opts"]["codebook_prob"]
        
        p_job = pretrain_jobs[(layers, ep, disc_type, cb_prob)]
        config["training"]["preload_from_files"] = {"": {"filename": p_job.out_checkpoints[ep].path}}

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
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_eps,
            skip_eval=False,
            rasr_recog_opts=None,
            vis_epochs=vis_eps,
            vis_kwargs={"cosine_similarity_summary": True},
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )
