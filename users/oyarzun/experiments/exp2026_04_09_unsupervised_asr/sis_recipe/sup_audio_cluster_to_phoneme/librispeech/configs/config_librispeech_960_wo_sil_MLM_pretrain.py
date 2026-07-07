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
train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)

# Extend the base config to use the new MLM training step module
base_config = copy.deepcopy(base_config)
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_mlm.train_step"


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"




    ablations = [
        # With codebook (encoder quantization)
        (
            f"baseline_codebook_enc-{layers}_dec-{layers}_mlm_iter-{iter}_v1",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "codebook_opts": {},
            },
            {
                "codebook_diversity_loss_scale": 0.11,
                "mlm_pretrain_steps": iter,
            },
            {
                "batch_size": 4000,
            }
        ) for layers, iter in [
            (3, 10000),
            (3, 100000),
            (6, 10000), 
            (6, 100000), 
        ]
    ] + [
        # Without codebook
        (
            f"baseline_enc-{layers}_dec-{layers}_mlm_iter-{iter}_v1",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
            },
            {
                "codebook_diversity_loss_scale": 0.0,
                "mlm_pretrain_steps": iter,
            },
            {
                "batch_size": 4000,
            }
        ) for layers, iter in [
            (3, 10000),
            (3, 100000),
            (6, 100000), 
            (6, 100000), 
        ]
    ] + [
        # Without codebook in pretraining, With codebook in backtranslation
        (
            f"baseline_codebook_enc-{layers}_dec-{layers}_mlm_iter-{iter}_nocbpretrain_v1",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "codebook_opts": {},
            },
            {
                "codebook_diversity_loss_scale": 0.11,
                "mlm_pretrain_steps": iter,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
            },
            {
                "batch_size": 4000,
            }
        ) for layers, iter in [
            (3, 10000),
            (3, 100000),
            (6, 10000), 
            (6, 100000), 
        ]
    ]



    for train_name, model_args, train_args, training_args in ablations:
        config = copy.deepcopy(base_config)
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
        )
