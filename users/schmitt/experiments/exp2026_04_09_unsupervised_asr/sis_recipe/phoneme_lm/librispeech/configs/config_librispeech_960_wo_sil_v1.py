import copy

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from ....sup_audio_cluster_to_phoneme.librispeech.configs.config_librispeech_960_v1 import (
    get_keep_epochs,
    base_num_epochs,
)

num_gpus = 2

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering="laplace:.1000",
)
train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)
test_data_dict_wo_sil = build_test_datasets(sil_prob=0.0, surround_w_sil=False)
test_data_dict = build_test_datasets()

base_config = {
    "__network_module": "definitions.transformer_decoder_lm_v1.Model",
    "__train_step_module": "train_steps.lm_ce.train_step",
    "__baseline_alias": "v1",
    # forward step / callback for the perplexity (CE/PPL) forward job (models.scoring.ppl)
    "__forward_step_module": "scoring.ppl.forward_step.forward_step",
    "__callback_module": "scoring.ppl.callback.PplScoresCallback",
    "train_rqmt": {
        "cpu_rqmt": 6,
    },
    "general": {
        "torch_dataloader_opts": {"num_workers": 1},
        "behavior_version": 25,
        # the phoneme LM only has text data; it is both the (only) input and the scoring target
        "default_data_key": "data",
        "default_target_key": "data",
    },
    "training": {
        "__num_gpus": num_gpus,
        "__num_epochs": base_num_epochs,
        "__lr_opts": {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": [
                0,
                0.45 * base_num_epochs,
                0.9 * base_num_epochs,
                base_num_epochs,
            ],
            "piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
        },
        "grad_scaler": None,
        "batch_size": 15_000,
        **optimizer_configs.v1,
        "max_seqs": 200,
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": 5.0,
    },
    "recog": {
        "batch_size": 15_000,
    },
    "model_args": {
        "num_layers": 6,
        "num_heads": 8,
        "model_dim": 512,
        "out_dim": train_data.datastreams["data"].vocab_size,
    },
    "train_args": {
        "ce_loss_scale": 1.0,
    },
}


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
        train_data=train_data,
        test_data_dict=test_data_dict_wo_sil,
        keep_epochs=get_keep_epochs(base_num_epochs),
        # plain LM: no ASR recognition, but compute the phoneme perplexity per kept checkpoint.
        skip_eval=True,
        ppl_opts={
            "checkpoints": get_keep_epochs(base_num_epochs),
        },
    )
