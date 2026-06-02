import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, get_keep_epochs, test_data_dict, base_num_epochs

from sisyphus import tk

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering=None,
)
train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
    )

    for exp_idx, (config, train_name) in enumerate(
        [
            *[
                (
                    dict_update_deep(
                        copy.deepcopy(base_config),
                        {
                            "model_args": {
                                "num_enc_layers": num_enc_layers,
                                "num_text_dec_layers": num_dec_layers,
                                "num_audio_dec_layers": num_dec_layers,
                            },
                            "training.batch_size": batch_size,
                        },
                    ),
                    f"baseline_enc-{num_enc_layers}_dec-{num_dec_layers}_bs-{batch_size}",
                )
                for num_enc_layers, num_dec_layers, batch_size in ((6, 6, 10_000),)
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
