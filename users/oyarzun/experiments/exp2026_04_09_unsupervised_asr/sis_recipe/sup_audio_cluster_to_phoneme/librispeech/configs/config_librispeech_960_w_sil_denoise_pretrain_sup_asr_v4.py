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
            "baseline_disc_enc-3_dec-3_denoise_ep-5_v4.1_lmdata_adv_asr_DEBUG",
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
        # With discriminator in both pretraining and ASR, using LM text for ASR phase
        (
            f"baseline_disc_enc-{layers}_dec-{layers}_denoise_ep-{pretrain_epochs}_v4.1_lmdata_adv_asr",
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
        ) for layers, pretrain_epochs in [
            (3, 10),
            (3, 100),
            (6, 100), 
        ]
    ] \
    + [
        # no discriminator for both pretraining and ASR; only pretrian - asr
        (
            f"baseline_NO_disc_enc-{layers}_dec-{layers}_denoise_ep-{pretrain_epochs}_v4.1",
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
        ) for layers, pretrain_epochs in [
            (3, 10),
            (3, 100),
            (6, 100), 
        ]
    ] \
    + [
        # With discriminator in pretraining only, not in ASR
        (
            f"baseline_disc_enc-{layers}_dec-{layers}_denoise_ep-{pretrain_epochs}_nodiscasr_v4.1",
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
            }, 
        ) for layers, pretrain_epochs in [
            (3, 10),
            (3, 100),
            (6, 10), 
            (6, 100), 
        ]
    ] + [
        # Fully supervised ASR, no pretraining, no discriminator
        (
            f"baseline_sup_asr_-nopretrain-enc_{layers}_dec-{layers}_v4.1",
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

        pretrain_epochs = train_args.get("denoise_pretrain_epochs", 0)
        asr_epochs = base_num_epochs - pretrain_epochs

        if pretrain_epochs > 0:
            piecewise_epochs = [
                0,
                0.45 * pretrain_epochs,
                0.9 * pretrain_epochs,
                pretrain_epochs,
                pretrain_epochs + 1e-5,
                pretrain_epochs + 0.45 * asr_epochs,
                pretrain_epochs + 0.9 * asr_epochs,
                base_num_epochs
            ]
            piecewise_values = [
                1e-5, 1e-3, 1e-5, 1e-6,
                1e-5, 1e-3, 1e-5, 1e-6
            ]
            config["training"]["__lr_opts"] = {
                "type": "dyn_lr_piecewise_linear",
                "piecewise_epochs": piecewise_epochs,
                "piecewise_values": piecewise_values,
            }

        use_lm = train_args.get("use_lm_for_asr_adv", False)
        current_train_data = train_data_lm if use_lm else train_data_no_lm

        config["recog_rqmt"] = {"time": 48, "mem": 24, "cpu": 8}
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=current_train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=False,
            rasr_recog_opts={"line_based_lexicon_file": current_train_data.add_opts["line_based_lexicon_file"]},
            vis_epochs=[250, 500, 750, 1000],
            vis_kwargs={"cosine_similarity_summary": True},
        )

    # --- New Ablations with longer masking spans ---
    for ablation in ablations:
        train_name = ablation[0]
        model_args = ablation[1]
        train_args = ablation[2]
        training_args = ablation[3]
        

        #: basically the fifth position in each ablation tuple (if exists, and set to True) is
        #: considered as an ablation being debugged --> use 1 gpu.
        is_debug = ablation[4] if len(ablation) > 4 else False

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

        pretrain_epochs = train_args.get("denoise_pretrain_epochs", 0)
        asr_epochs = base_num_epochs - pretrain_epochs

        if pretrain_epochs > 0:
            piecewise_epochs = [
                0,
                0.45 * pretrain_epochs,
                0.9 * pretrain_epochs,
                pretrain_epochs,
                pretrain_epochs + 1e-5,
                pretrain_epochs + 0.45 * asr_epochs,
                pretrain_epochs + 0.9 * asr_epochs,
                base_num_epochs
            ]
            piecewise_values = [
                1e-5, 1e-3, 1e-5, 1e-6,
                1e-5, 1e-3, 1e-5, 1e-6
            ]
            config["training"]["__lr_opts"] = {
                "type": "dyn_lr_piecewise_linear",
                "piecewise_epochs": piecewise_epochs,
                "piecewise_values": piecewise_values,
            }

        use_lm = train_args.get("use_lm_for_asr_adv", False)
        current_train_data = train_data_lm if use_lm else train_data_no_lm

        config["recog_rqmt"] = {"time": 48, "mem": 24, "cpu": 8, "gpu_mem": 11, "gpu": 1}
        run_experiment(
            training_name=f"{prefix_name}/{new_train_name}",
            config=config,
            train_data=current_train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=False,
            rasr_recog_opts={"line_based_lexicon_file": current_train_data.add_opts["line_based_lexicon_file"]},
            vis_epochs=[250, 500, 750, 1000],
            vis_kwargs={"cosine_similarity_summary": True},
        )
