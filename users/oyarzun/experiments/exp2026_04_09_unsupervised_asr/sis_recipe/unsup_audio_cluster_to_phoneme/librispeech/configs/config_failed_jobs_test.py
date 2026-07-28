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

#: ablation study, sil prob = 0 (remember meta gan paper, silence insertion part) 
# surrounding silence definition as in the paper
train_data = build_training_datasets(sil_prob=0.25, surround_w_sil=True, settings=settings)

def py_mlm_v4_failed():
    base_config_mlm = copy.deepcopy(base_config)
    base_config_mlm["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_mlm_v3.train_step"
    
    prefix_name = f"{__setup_base_name__}/librispeech/config_librispeech_960_w_sil_MLM_pretrain_unfreeze_v4"

    # Job: unfreeze_p-60_start-50_end-90_predisc-True_btdisc-True_iter-100000_v4
    bt_total_steps = 400_000
    iter_val = 100_000
    prop = 0.6
    start = 0.5
    end = 0.9
    pre_disc = True
    bt_disc = True

    train_name = f"unfreeze_p-{int(prop*100)}_start-{int(start*100)}_end-{int(end*100)}_predisc-{pre_disc}_btdisc-{bt_disc}_iter-{iter_val}_v4_test"
    model_args = {
        "num_enc_layers": 6,
        "num_text_dec_layers": 6,
        "num_audio_dec_layers": 6,
        "discriminator_type": "lstm",
        "codebook_opts": {"codebook_prob": 0.0}
    }
    train_args = {
        "codebook_diversity_loss_scale": 0.0,
        "mlm_pretrain_steps": iter_val,
        "pretrain_codebook_prob": 0.0,
        "pretrain_codebook_diversity_loss_scale": 0.0,
        "adv_loss_scale": 0.1 if bt_disc else 0.0,
        "pretrain_adv_loss_scale": 0.1 if pre_disc else 0.0,
        "gradual_unfreeze": True,
        "gradual_unfreeze_proportion": prop,
        "gradual_unfreeze_start_iter": int(bt_total_steps * start),
        "gradual_unfreeze_end_iter": int(bt_total_steps * end),
        "bt_buffer_size_steps": 10,
        "bt_train_iterations": 50,
        "text_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
        "audio_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
    }
    training_args = {
        "batch_size": 4000,
        "__num_gpus": 1,
        "grad_scaler": None
    }
    
    config = copy.deepcopy(base_config_mlm)
    config["model_args"].update(model_args)
    config["train_args"].update(train_args)
    config["training"].update(training_args)
    
    run_experiment(
        training_name=f"{prefix_name}/{train_name}",
        config=config,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )


def py_denoise_v4_failed():
    base_config_denoise = copy.deepcopy(base_config)
    base_config_denoise["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_denoise_v3.train_step"
    
    prefix_name = f"{__setup_base_name__}/librispeech/config_librispeech_960_w_sil_denoise_pretrain_unfreeze_v4"

    # Job: unfreeze_p-60_start-50_end-90_predisc-True_btdisc-False_iter-100000_v4
    bt_total_steps = 400_000
    iter_val = 100_000
    prop = 0.6
    start = 0.5
    end = 0.9
    pre_disc = True
    bt_disc = False

    train_name = f"unfreeze_p-{int(prop*100)}_start-{int(start*100)}_end-{int(end*100)}_predisc-{pre_disc}_btdisc-{bt_disc}_iter-{iter_val}_v4_test"
    model_args = {
        "num_enc_layers": 6,
        "num_text_dec_layers": 6,
        "num_audio_dec_layers": 6,
        "discriminator_type": "lstm",
        "codebook_opts": {"codebook_prob": 0.0}
    }
    train_args = {
        "codebook_diversity_loss_scale": 0.0,
        "denoise_pretrain_steps": iter_val,
        "pretrain_codebook_prob": 0.0,
        "pretrain_codebook_diversity_loss_scale": 0.0,
        "adv_loss_scale": 0.1 if bt_disc else 0.0,
        "pretrain_adv_loss_scale": 0.1 if pre_disc else 0.0,
        "gradual_unfreeze": True,
        "gradual_unfreeze_proportion": prop,
        "gradual_unfreeze_start_iter": int(bt_total_steps * start),
        "gradual_unfreeze_end_iter": int(bt_total_steps * end),
        "bt_buffer_size_steps": 10,
        "bt_train_iterations": 50,
        "text_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
        "audio_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
    }
    training_args = {
        "batch_size": 4000,
        "__num_gpus": 1,
        "grad_scaler": None
    }
    
    config = copy.deepcopy(base_config_denoise)
    config["model_args"].update(model_args)
    config["train_args"].update(train_args)
    config["training"].update(training_args)
    
    run_experiment(
        training_name=f"{prefix_name}/{train_name}",
        config=config,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )


def py():
    #py_mlm_v4_failed()
    #py_denoise_v4_failed()
    py_denoise_wo_sil_v4_oom_test()

def py_denoise_wo_sil_v4_oom_test():
    # Use wo_sil config instead of w_sil
    from .config_librispeech_960_wo_sil_v1 import base_config as base_config_wo_sil
    base_config_denoise = copy.deepcopy(base_config_wo_sil)
    base_config_denoise["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_denoise_v3.train_step"
    
    prefix_name = f"{__setup_base_name__}/librispeech/config_librispeech_960_wo_sil_denoise_pretrain_unfreeze_v4"

    # Job: unfreeze_p-80_start-50_end-90_predisc-True_btdisc-True_v4
    bt_total_steps = 400_000
    iter_val = 100_000 # Assuming this defaults to 100000 based on standard setup
    prop = 0.8
    start = 0.5
    end = 0.9
    pre_disc = True
    bt_disc = True

    train_name = f"unfreeze_p-{int(prop*100)}_start-{int(start*100)}_end-{int(end*100)}_predisc-{pre_disc}_btdisc-{bt_disc}_v4_oom_test"
    model_args = {
        "num_enc_layers": 6,
        "num_text_dec_layers": 6,
        "num_audio_dec_layers": 6,
        "discriminator_type": "lstm",
        "codebook_opts": {"codebook_prob": 0.0}
    }
    train_args = {
        "codebook_diversity_loss_scale": 0.0,
        "denoise_pretrain_steps": iter_val,
        "pretrain_codebook_prob": 0.0,
        "pretrain_codebook_diversity_loss_scale": 0.0,
        "adv_loss_scale": 0.1 if bt_disc else 0.0,
        "pretrain_adv_loss_scale": 0.1 if pre_disc else 0.0,
        "gradual_unfreeze": True,
        "gradual_unfreeze_proportion": prop,
        "gradual_unfreeze_start_iter": int(bt_total_steps * start),
        "gradual_unfreeze_end_iter": int(bt_total_steps * end),
        "bt_buffer_size_steps": 10,
        "bt_train_iterations": 50,
        "text_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
        "audio_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
    }
    
    # REDUCED BATCH SIZE AND INCREASED ACCUMULATION FOR 11GB GPU
    training_args = {
        "batch_size": 1000,
        "accum_grad_multiple_step": 8,
        "__num_gpus": 1,
        "grad_scaler": None
    }
    
    config = copy.deepcopy(base_config_denoise)
    config["model_args"].update(model_args)
    config["train_args"].update(train_args)
    config["training"].update(training_args)
    
    run_experiment(
        training_name=f"{prefix_name}/{train_name}",
        config=config,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )
