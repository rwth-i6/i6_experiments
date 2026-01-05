from i6_core.returnn.config import ReturnnConfig

from .data.common import TrainingDatasets, SOURCE_DATASTREAM_KEY, TARGET_DATASTREAN_KEY

# changing these does not change the hash
post_config = {
    'use_tensorflow': True,
    'tf_log_memory_usage': True,
    'cleanup_old_models': True,
    'log_batch_size': True,
    'debug_mode': False,
    'cache_size': '0',
}

CF_CODE = """
import os
from subprocess import check_output

_cf_cache = {}

def cf(filename):
    if filename in _cf_cache:
        return _cf_cache[filename]
    cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn
"""

STACK_CODE = """
# https://github.com/rwth-i6/returnn/issues/957
# https://stackoverflow.com/a/16248113/133374
import resource
import sys
try:
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
sys.setrecursionlimit(10 ** 6)
"""


def get_training_config(datasets: TrainingDatasets, network):
    config = {
        "calculate_exp_loss": True,
        #############
        "train": datasets.train.as_returnn_opts(),
        "dev": datasets.cv.as_returnn_opts(),
        "target": "data",
        "extern_data": datasets.extern_data,
        #############
        "network": network
    }
    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        python_prolog=[CF_CODE, STACK_CODE],
        python_prolog_hash="",
    )

    return returnn_config



