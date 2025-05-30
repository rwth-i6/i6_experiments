"""
Compute features mean and steddev on LBS using Albert's implementation
"""

from __future__ import annotations

from typing import Optional
import copy
from sisyphus import tk
from i6_experiments.users.yang.torch.luca_ctc.configs import *


config_11gb = copy.deepcopy(config_11gb)
config_11gb.pop("dynamic_learning_rate", None)
config_11gb.pop("learning_rate_piecewise_steps", None)
config_11gb.pop("learning_rate_piecewise_values", None)
config_11gb.pop("learning_rate_invsqrt_norm", None)
config_11gb.pop("learning_rate_warmup_steps", None)


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """Compute feature statistics for librispeech"""
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_log_mel_stats
    stats = get_librispeech_log_mel_stats(80)
    tk.register_output("lbs_feature_stats/mean", stats.mean)
    tk.register_output("lbs_feature_stats/std_dev", stats.std_dev)
    tk.register_output("lbs_feature_stats/info", stats.info)

py = sis_run_with_prefix  # if run directly via `sis m ...`
