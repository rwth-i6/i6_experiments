import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.models.recognition.discrete_audio_aed.rasr.forward_step import (
    DecoderConfigV1 as RasrDecoderConfigV1,
)

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, get_keep_epochs, base_num_epochs

from sisyphus import tk

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering=None,
)
train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)
test_data_dict_wo_sil = build_test_datasets(sil_prob=0.0, surround_w_sil=False)
test_data_dict = build_test_datasets()


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=False,
        # conditional (audio->phoneme) perplexity of the AED model on the last checkpoint, scored on
        # the wo-silence reference (matching this wo-sil model) via a separate PPL dataset; recognition
        # keeps the with-silence test_data_dict. Both expose audio + text (paired MetaDataset).
        ppl_opts={
            "checkpoints": [base_num_epochs],
            "input_modality": "audio",
            "test_data_dict": test_data_dict_wo_sil,
        },
    )

    # run_experiment(
    #     training_name=f"{prefix_name}/baseline",
    #     config=copy.deepcopy(base_config),
    #     train_data=train_data,
    #     test_data_dict=test_data_dict,
    #     keep_epochs=[base_num_epochs],
    #     skip_eval=False,
    #     rasr_recog_opts={},
    # )

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
        num_epochs = config["training"]["__num_epochs"]
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(num_epochs),
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
                                "num_enc_layers": 6,
                                "num_text_dec_layers": 6,
                                "num_audio_dec_layers": 6,
                            },
                            "training.batch_size": 10_000,
                            "train_args.label_smoothing": label_smooting,
                            # the deeper enc-6 model occasionally yields a single non-finite
                            # train score early on (e.g. ep1 step66) which otherwise aborts the
                            # whole training; skip such steps instead. Kept in the (hashed)
                            # training config since it changes training behavior.
                            "training.stop_on_nonfinite_train_score": False,
                        },
                    ),
                    f"baseline_enc-6_dec-6_bs-{10}k_ls-{label_smooting}",
                )
                for label_smooting in (0.1, 0.2)
            ]
        ]
    ):
        num_epochs = config["training"]["__num_epochs"]
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(num_epochs),
            skip_eval=False,
        )
