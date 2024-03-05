"""
Experiments for azDesktop2020 (local computer).
"""

from __future__ import annotations
from .conformer_import_moh_att_2023_06_30 import (
    config_11gb,
    post_config,
    from_scratch_model_def,
    from_scratch_training,
    model_recog,
)


def sis_run_with_prefix(prefix_name: str = None):
    """run the exp"""
    from i6_experiments.users.zeyer import tools_paths

    tools_paths.monkey_patch_i6_core()

    from .sis_setup import get_prefix_for_config
    from .train import train
    from i6_experiments.users.zeyer.recog import recog_training_exp
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_bpe10k_raw

    if not prefix_name:
        prefix_name = get_prefix_for_config(__file__)

    task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)

    config_ = config_11gb.copy()
    config_.update(
        {
            "torch_amp": "bfloat16",
            "batch_size": 30_000 * 160,
        }
    )

    model_with_checkpoint = train(
        prefix_name + "/base",
        task=task,
        config=config_,
        post_config=post_config,
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
        num_epochs=2000,
    )
    recog_training_exp(prefix_name + "/base", task, model_with_checkpoint, recog_def=model_recog)


py = sis_run_with_prefix  # if run directly via `sis m ...`
