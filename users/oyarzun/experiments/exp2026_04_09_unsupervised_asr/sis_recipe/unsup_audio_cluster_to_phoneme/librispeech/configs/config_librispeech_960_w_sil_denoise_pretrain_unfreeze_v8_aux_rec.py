import copy
import os
from typing import List

from i6_core.returnn.config import ReturnnConfig
from i6_core.serialization import Collection
from sisyphus import tk

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
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
})


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    layers = 6
    ft_epochs = 500
    eval_epochs = [125, 250, 375, 500]
    lr_val = 1e-4
    lr_str = "1e-4"
    aux_rec_scale = 0.5  # Default auxiliary self-supervised loss weight from wav2vec-U 2.0

    v6_alias_root = (
        "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/"
        "alias/unsup_audio_cluster_to_phoneme/librispeech/"
        "config_librispeech_960_w_sil_denoise_pretrain_unfreeze_v6"
    )

    base_checkpoints = [
        (
            "disc-codebooks",
            "pretrain_disc-codebooks_enc-6_dec-6_ep-500_v6.1",
            tk.Path(f"{v6_alias_root}/pretrain_disc-codebooks_enc-6_dec-6_ep-500_v6.1/training/output/models/epoch.500.pt"),
        ),
        (
            "disc-neither",
            "pretrain_disc-neither_enc-6_dec-6_ep-500_v6.1",
            tk.Path(f"{v6_alias_root}/pretrain_disc-neither_enc-6_dec-6_ep-500_v6.1/training/output/models/epoch.500.pt"),
        ),
    ]

    # Ablations over which encoder layer(s) receive the auxiliary self-supervised reconstruction loss:
    # - Individual layers 1..6
    # - Averaged logits across all layers 1..6 before computing the softmax
    aux_layer_ablations = [
        *(
            (f"layer-{l_idx}", [l_idx], f"single_layer_{l_idx}")
            for l_idx in range(1, layers + 1)
        ),
        ("avg-layers-1-6", list(range(1, layers + 1)), "avg_logits_layers_1_to_6"),
    ]

    finetune_jobs = {}
    configs_meta = {}

    for tag, pretrain_name, ckpt_path in base_checkpoints:
        for aux_layer_tag, aux_rec_layers, aux_mode_desc in aux_layer_ablations:
            train_name = (
                f"from-{tag}_enc-{layers}_dec-{layers}_unfreeze-warm_"
                f"constlr-{lr_str}_auxrec-{aux_rec_scale}_{aux_layer_tag}_ft_ep-{ft_epochs}_v8"
            )

            config = copy.deepcopy(base_config)
            config["model_args"].update({
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "discriminator_type": None,
                "codebook_opts": {"codebook_prob": 0.0},
                "audio_aux_rec_layers": aux_rec_layers,
            })

            config["train_args"].update({
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": 0,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.0,
                "pretrain_adv_loss_scale": 0.0,

                # Auxiliary self-supervised reconstruction loss weight (wav2vec-U 2.0 default: 0.5)
                "audio_aux_rec_loss_scale": aux_rec_scale,

                # Warm encoder (fully unfrozen)
                "gradual_unfreeze": False,
                "freeze_encoder": False,

                "bt_buffer_size_steps": 10,
                "bt_train_iterations": 50,

                "text_masking_opts": {
                    "mask_prob": 0.3,
                    "min_span": 2,
                    "max_span": 10,
                    "expand": True,
                    "insert_prob": 0.1,
                },
                # Alternative B1: 1-to-1 in-place span masking (`expand=False, keep_length=True`)
                # so masked input and unmasked targets share exact sequence length T and 1-to-1 alignment.
                "audio_masking_opts": {
                    "mask_prob": 0.3,
                    "min_span": 4,
                    "max_span": 20,
                    "expand": False,
                    "keep_length": True,
                },
            })

            config["training"].update({
                "batch_size": 15000,
                "grad_scaler": None,
                "accum_grad_multiple_step": 2,
                "__num_epochs": ft_epochs,
                "preload_from_files": {
                    "pretrained_model": {
                        "filename": ckpt_path,
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

            config["recog_rqmt"] = {
                "time": 4,
                "mem": 32,
                "cpu": 6,
                "gpu": 1,
                "device": "gpu",
                "gpu_mem": 80,
            }
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
                "base_checkpoint": f"{pretrain_name}@epoch_500",
                "layers": layers,
                "disc_type": "None",
                "codebook_prob": 0.0,
                "unfreeze_mode": "warm (fully unfrozen)",
                "learning_rate": lr_str,
                "audio_aux_rec_loss_scale": aux_rec_scale,
                "audio_aux_rec_layers": str(aux_rec_layers),
                "audio_aux_rec_mode": aux_mode_desc,
                "ft_epochs": ft_epochs,
                "cumulative_epochs": 500 + ft_epochs,
            }

    from ....hpo_summary import HpoResultsExcelJob

    excel_job = HpoResultsExcelJob(
        configs_meta=configs_meta,
        train_jobs=finetune_jobs,
        eval_epochs=eval_epochs,
        target_num_epochs=ft_epochs,
        prefix_name=prefix_name,
        report_name="unsup_v8_aux_rec_summary",
    )
    tk.register_output(f"{prefix_name}/unsup_v8_aux_rec_summary.xlsx", excel_job.out_excel)
    tk.register_output(f"{prefix_name}/unsup_v8_aux_rec_summary.csv", excel_job.out_csv)

    return {
        "finetune_jobs": finetune_jobs,
    }
