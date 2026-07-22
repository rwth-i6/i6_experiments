"""
Text-only (single-task) denoising models -- a dedicated config for testing how well the shared
model denoises/reconstructs *text alone* (no audio task, no alternate batching). Serves as the
single-task reference for the multi-task text+audio setups in ``config_librispeech_960_wo_sil_v1``.

To keep the trained checkpoints / results (job hashes are content-based, not path-based), the model
config, dataset and recon helpers are imported verbatim from ``config_librispeech_960_wo_sil_v1``;
only the experiment location (alias/output path) differs.
"""

import copy

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ....train_exp import run_experiment
from ..data.common import build_text_only_training_datasets, build_test_datasets
from ... import __setup_base_name__

from .config_librispeech_960_wo_sil_v1 import (
    base_config,
    settings,
    get_keep_epochs,
    base_num_epochs,
    _text_recon_variant,
    _text_recon_sweep,
)

text_only_train_data = build_text_only_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)
test_data_dict = build_test_datasets()

# text-only variant of the multi-task base_config: text denoising train step, phoneme in/out keys,
# and drop alternate batching + all audio-task settings.
text_only_config = dict_update_deep(
    copy.deepcopy(base_config),
    {
        "__train_step_module": "train_steps.aed_denoising_discrete_text_only.train_step",
        "general.default_data_key": "phon_indices",
        "general.default_target_key": "phon_indices",
        "training.accum_grad_multiple_step": 1,  # no alternate batching -> no grad accumulation
    },
    [
        # drop alternate batching and all audio-task settings
        "training.torch_batching",
        "train_args.audio_ce_loss_scale",
        "train_args.audio_masked_ce_loss_scale",
        "train_args.audio_masking_opts",
    ],
)


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline_text-only",
        config=copy.deepcopy(text_only_config),
        train_data=text_only_train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=True,  # audio->text ASR is not meaningful for a text-only model
        recog_variants=[
            _text_recon_variant(text_only_config, base_num_epochs),
            # fixed-masking sweep to characterize the text denoiser (copy ceiling + curve)
            *_text_recon_sweep(base_num_epochs),
        ],
    )
