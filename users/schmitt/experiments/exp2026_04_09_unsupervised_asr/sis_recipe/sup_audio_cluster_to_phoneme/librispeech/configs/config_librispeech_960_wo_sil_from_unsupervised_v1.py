import copy
from typing import List, Dict

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


def py(checkpoints: Dict):
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=dict_update_deep(
            copy.deepcopy(base_config),
            {
                "training.preload_from_files": {
                    "pretrained_model": {
                        "init_for_train": True,
                        "checkpoint_key": "model",
                        "filename": checkpoints[
                            "unsup_denoising_audio_cluster_and_phoneme/librispeech/config_librispeech_960_w_sil_in_input_v1/baseline_gan-adv-0.1_disc-lstm_mask-p-0.1-span-1-1_max-num-sil-7_max-surround-1"
                        ][500],
                    }
                },
                "model_args.freeze_params_list": [rf"encoder\."],
            },
        ),
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
