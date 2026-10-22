"""
Debug configuration for Version 8 (Auxiliary Self-Supervised Encoder Reconstruction on Real Audio):
- Fine-tunes `pretrain_disc-codebooks_enc-6_dec-6_ep-500_v6.1` (`epoch.500.pt`)
- Uses CPU (`device="cpu"`) and a small subset of data (20 sequences/epoch via partition_epoch on devtrain/dev subsets)
- Runs 2 epochs with `bt_buffer_size_steps=2`, `bt_train_iterations=4` to exercise both ACCUMULATE and TRAIN phases
  as well as evaluation in an interactive CPU session (`devel` partition).
"""

import copy

from i6_core.returnn.config import ReturnnConfig
from i6_core.serialization import Collection
from sisyphus import tk

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from ... import __setup_base_name__
from .config_librispeech_960_v1 import base_config, alternate_batching


settings = DatasetSettings(
    train_partition_epoch=150,
    train_seq_ordering="laplace:.100",
)
train_data = copy.deepcopy(
    build_training_datasets(
        sil_prob=0.25,
        surround_w_sil=True,
        settings=settings,
    )
)
test_data_dict = build_test_datasets()

# Subset training and eval datasets using existing devtrain segment file (3000 lines)
from ....data.common import TrainingDatasets as TDClass
devtrain_seg_file = train_data.eval_datasets["devtrain"].datasets["feature_clusters"].segment_file
train_data.train.datasets["feature_clusters"].segment_file = devtrain_seg_file
train_data.train.datasets["phon_indices"].segment_file = devtrain_seg_file
train_data.train.datasets["feature_clusters"].partition_epoch = 150
train_data.train.datasets["phon_indices"].partition_epoch = 150

devtrain_ds = train_data.eval_datasets["devtrain"]
devtrain_ds.datasets["feature_clusters"].partition_epoch = 300
devtrain_ds.datasets["phon_indices"].partition_epoch = 300

train_data = TDClass(
    train=train_data.train,
    eval_datasets={"devtrain": devtrain_ds},
    datastreams=train_data.datastreams,
    add_opts=train_data.add_opts,
)

base_config = copy.deepcopy(base_config)
base_config["random_seed"] = 42
base_config["__network_module"] = "definitions.conformer_aed_discrete_shared_aux_rec_v1.Model"
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_aux_rec_v1.train_step"
base_config["save_interval"] = 1

# CPU execution settings
base_config["training"]["__num_gpus"] = 1
base_config["training"]["batch_size"] = 1500
base_config["training"]["max_seqs"] = 4
base_config["training"]["accum_grad_multiple_step"] = 2
base_config["training"]["grad_scaler"] = None

base_config["train_rqmt"] = {
    "device": "cpu",
    "cpu_rqmt": 4,
    "mem_rqmt": 16,
    "time_rqmt": 1,
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
    ft_epochs = 2
    lr_val = 1e-4
    lr_str = "1e-4"
    aux_rec_scale = 0.5
    aux_rec_layers = [6]

    v6_alias_root = (
        "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/"
        "alias/unsup_audio_cluster_to_phoneme/librispeech/"
        "config_librispeech_960_w_sil_denoise_pretrain_unfreeze_v6"
    )
    ckpt_path = tk.Path(
        f"{v6_alias_root}/pretrain_disc-codebooks_enc-6_dec-6_ep-500_v6.1/training/output/models/epoch.500.pt"
    )

    train_name = (
        f"debug_from-disc-codebooks_enc-{layers}_dec-{layers}_unfreeze-warm_"
        f"constlr-{lr_str}_auxrec-{aux_rec_scale}_layer-6_ft_ep-{ft_epochs}_cpu"
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
        "audio_aux_rec_loss_scale": aux_rec_scale,
        "gradual_unfreeze": False,
        "freeze_encoder": False,
        "bt_buffer_size_steps": 2,
        "bt_train_iterations": 4,
        "text_masking_opts": {
            "mask_prob": 0.3,
            "min_span": 2,
            "max_span": 10,
            "expand": True,
            "insert_prob": 0.1,
        },
        "audio_masking_opts": {
            "mask_prob": 0.3,
            "min_span": 4,
            "max_span": 20,
            "expand": False,
            "keep_length": True,
        },
    })

    config["training"].update({
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

    run_experiment(
        training_name=f"{prefix_name}/{train_name}",
        config=config,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=[ft_epochs],
        skip_eval=True,
        rasr_recog_opts=None,
        vis_epochs=None,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )
