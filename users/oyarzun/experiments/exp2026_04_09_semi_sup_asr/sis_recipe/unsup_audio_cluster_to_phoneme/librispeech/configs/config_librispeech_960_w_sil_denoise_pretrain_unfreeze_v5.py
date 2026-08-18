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

# Extend the base config to use the new denoising training step module
base_config = copy.deepcopy(base_config)
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_denoise_v3.train_step"


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    # Assuming 400k iterations in total for backtranslation, adjust if necessary
    bt_total_steps = 400_000


    ablations = [
        (
            f"unfreeze_enc-{layers}_dec-{layers}_p-{int(prop*100)}_start-{int(start*100)}_end-{int(end*100)}_predisc-{pre_disc}_btdisc-{bt_disc}_epoch-{pretrain_epochs}_v5",
            {
                "num_enc_layers": layers,
                "num_text_dec_layers": layers,
                "num_audio_dec_layers": layers,
                "discriminator_type": "lstm",
                "codebook_opts": {"codebook_prob": 0.0}
            },
            {
                "codebook_diversity_loss_scale": 0.0,
                "denoise_pretrain_epochs": pretrain_epochs,
                "pretrain_codebook_prob": 0.0,
                "pretrain_codebook_diversity_loss_scale": 0.0,
                "adv_loss_scale": 0.1 if bt_disc else 0.0,
                "pretrain_adv_loss_scale": 0.1 if pre_disc else 0.0,
                "gradual_unfreeze": True,
                "gradual_unfreeze_proportion": prop,
                "gradual_unfreeze_start_iter": int(bt_total_steps * start),
                "gradual_unfreeze_end_iter": int(bt_total_steps * end),
                "bt_buffer_size_steps": 10, #: number of iterations over which we accumulate data and pregenerate pairs. 
                "bt_train_iterations": 50, #: number of iterations where we train.
            },
            {
                "batch_size": 1000,
                "accum_grad_multiple_step": 8,
            }
        )
        for layers in [3, 6]
        for prop in [0.8, 0.6]
        for pretrain_epochs in [100, 200, 500]
        for start, end in [(0.5, 0.9)]
        for pre_disc in [True] 
        for bt_disc in [True, False] 
    ]

    for train_name, model_args, train_args, training_args in ablations:
        config = copy.deepcopy(base_config)
        config["model_args"].update(model_args)
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

        #config["training"]["__num_gpus"] = 1  # CHANGE TO 1 GPU for debugging.

        pretrain_ep = train_args["denoise_pretrain_epochs"]
        checkpoints = [pretrain_ep, pretrain_ep + 10, pretrain_ep + 100]
        
        keep_eps = get_keep_epochs(base_num_epochs)
        if keep_eps is None:
            keep_eps = []
        for chk in checkpoints:
            if chk not in keep_eps:
                keep_eps = sorted(keep_eps + [chk])
        
        vis_eps = [250, 500, 750, 1000]
        for chk in checkpoints:
            if chk not in vis_eps:
                vis_eps = sorted(vis_eps + [chk])

        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=keep_eps,
            skip_eval=False,
            rasr_recog_opts={"line_based_lexicon_file": train_data.add_opts["line_based_lexicon_file"]},
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            vis_epochs=vis_eps,
        )
