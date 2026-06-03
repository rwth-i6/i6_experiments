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

from ....sup_audio_cluster_to_phoneme.librispeech.configs.config_librispeech_960_v1 import (
    base_config as base_config_,
    get_keep_epochs,
    test_data_dict,
    base_num_epochs,
)

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering="laplace:.1000",
)
train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)


base_config = dict_update_deep(
    base_config_,
    {
        "__train_step_module": "train_steps.aed_denoising_discrete_shared.train_step",
        "training": {
            "torch_batching": CodeWrapper("alternate_batching"),
            "accum_grad_multiple_step": 2,  # alternate batching
        },
        "train_post_config": {
            "tensorboard_opts": {
                # uneven so that both text and audio losses get logged (alternated batching)
                "log_every_n_train_steps": 51,
            },
        },
        "general.default_target_key": "phon_indices",
        "model_args": {
            "text_out_dim": train_data.datastreams["phon_indices"].vocab_size,
            "audio_out_dim": train_data.datastreams["data"].vocab_size,
        },
        "train_args": {
            "aux_loss_scales": (),
            "text_ce_loss_scale": 0.0,
            "text_masked_ce_loss_scale": 1.0,
            "audio_ce_loss_scale": 0.0,
            "audio_masked_ce_loss_scale": 1.0,
            "text_masking_opts": {
                "mask_prob": 0.3,
                "min_span": 2,  # 1
                "max_span": 10,  # 3
            },
            "audio_masking_opts": {
                "mask_prob": 0.3,
                "min_span": 4,  # 1
                "max_span": 20,  # 3
            },
        },
    },
    [
        "train_args.masking_opts",
        "train_args.ce_loss_scale",
        "train_args.masked_ce_loss_scale",
    ],
)


alternate_batching = PartialImport(
    code_object_path="i6_experiments.users.schmitt.returnn.alternate_batching.alternate_batching",
    import_as="alternate_batching",
    hashed_arguments={},
    unhashed_arguments={},
    unhashed_package_root=None,
)


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=True,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    )
