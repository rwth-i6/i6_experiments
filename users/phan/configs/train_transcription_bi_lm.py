"""
Train bidirectional masked LM on LBS transcription using text-only dataset
"""

from __future__ import annotations

from typing import Optional
import copy
from sisyphus import tk

from i6_experiments.users.phan.rf_models.default_model_configs import default_bidirectional_ilm_config
from i6_experiments.users.phan.rf_models.bilstm_lm import get_model, train_step
from i6_experiments.users.yang.torch.luca_ctc.configs import *


config_11gb = copy.deepcopy(config_11gb)
config_11gb.pop("dynamic_learning_rate", None)
config_11gb.pop("learning_rate_piecewise_steps", None)
config_11gb.pop("learning_rate_piecewise_values", None)
config_11gb.pop("learning_rate_invsqrt_norm", None)
config_11gb.pop("learning_rate_warmup_steps", None)


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix()
    from i6_experiments.users.phan.lbs_transcription_bpe10k.train_config import train_lbs_bpe10k_transcription_bidirectional_lm

    train_job = train_lbs_bpe10k_transcription_bidirectional_lm(
        get_model,
        train_step,
        hashed_config={
            "lm_cfg": default_bidirectional_ilm_config,
            "target_masking_rate": 0.2,
        },
        num_epochs=20,
    )
    train_job.add_alias(_sis_prefix + "/train")
    tk.register_output(_sis_prefix + "/bpe10k_transcription_bi_lm/learning_rates", train_job.out_learning_rates)


_sis_prefix: Optional[str] = None

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name

py = sis_run_with_prefix  # if run directly via `sis m ...`
