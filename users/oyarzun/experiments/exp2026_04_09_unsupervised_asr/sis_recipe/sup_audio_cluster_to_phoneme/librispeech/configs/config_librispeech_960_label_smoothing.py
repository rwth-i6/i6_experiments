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


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    """
    run_experiment(
        training_name=f"{prefix_name}/baseline", #use ablation study data
        config=copy.deepcopy(base_config),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
    )
    """

    ablations = (
        (0.05, 100), 
        (0.05, 500), 
        (0.075, 100), 
        (0.075, 500), 
        (0.15, 100), 
        (0.15, 500), 
        (0.2, 100), 
        (0.2, 500), 
    )

    layers = 5

    for exp_idx, (config, train_name) in enumerate(
        [
            *[
                (
                    dict_update_deep(
                        copy.deepcopy(base_config),
                        {
                            "model_args": {
                                "num_enc_layers": layers,
                                "num_text_dec_layers": layers,
                                "num_audio_dec_layers": layers,
                            },
                            "training.batch_size": 10_000,
                            "train_args": {
                                "aux_loss_scales": (),
                                "ce_loss_scale": 1.0,
                                "label_smoothing": smoothing_param, 
                                "label_smoothing_start_epoch": start_epoch, 
                                "masked_ce_loss_scale": 0.0,
                                "masking_opts": {
                                    "mask_prob": 0.0,
                                    "min_span": 0,  # 1
                                    "max_span": 0,  # 3
                                },
                            }
                        },
                    ),
                    f"label_smoothing-{smoothing_param}_start_epoch-{start_epoch}-layers-{layers}",
                )
                for smoothing_param, start_epoch in ablations  # each tuple is a different test config ; basically an ablation
            ]
        ]
    ):
        print("new")
        num_epochs = config["training"]["__num_epochs"]
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(num_epochs),
            skip_eval=False,
        )
