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

# Extend the base config to use the new MLM training step module
base_config = copy.deepcopy(base_config)
base_config["__train_step_module"] = "train_steps.aed_denoising_discrete_shared_backtranslation_mlm_v3.train_step"

def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    # Assuming 400k iterations in total for backtranslation, adjust if necessary
    bt_total_steps = 400_000

    prop = 0.8
    start = 0.5
    end = 0.9
    pre_cb = True
    bt_cb = False

    train_name = f"unfreeze_p-{int(prop*100)}_start-{int(start*100)}_end-{int(end*100)}_pre_cb-{pre_cb}_bt_cb-{bt_cb}_v2"
    model_args = {
        "num_enc_layers": 6,
        "num_text_dec_layers": 6,
        "num_audio_dec_layers": 6,
        "add_mlm_head": True,
    }
    if pre_cb or bt_cb:
        model_args["codebook_opts"] = {"codebook_prob": 1.0 if bt_cb else 0.0}

    train_args = {
        "codebook_diversity_loss_scale": 0.11 if bt_cb else 0.0,
        "mlm_pretrain_steps": 10_000,
        "pretrain_codebook_prob": 1.0 if pre_cb else 0.0,
        "pretrain_codebook_diversity_loss_scale": 0.11 if pre_cb else 0.0,
        "gradual_unfreeze": True,
        "gradual_unfreeze_proportion": prop,
        "gradual_unfreeze_start_iter": int(bt_total_steps * start),
        "gradual_unfreeze_end_iter": int(bt_total_steps * end),
        "bt_buffer_size_steps": 10, #: number of iterations over which we accumulate data and pregenerate pairs. 
        "bt_train_iterations": 50, #: number of iterations where we train.
    }

    training_args = {
        "batch_size": 1000,
        "accum_grad_multiple_step": 8,
    }

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
    
    # User requested a single GPU for this test config
    config["training"]["__num_gpus"] = 1

    run_experiment(
        training_name=f"{prefix_name}/{train_name}",
        config=config,
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )
