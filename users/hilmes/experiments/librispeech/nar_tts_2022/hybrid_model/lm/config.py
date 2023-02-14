from i6_core.returnn.config import ReturnnConfig

from .data import TrainingDatasets, SOURCE_DATASTREAM_KEY, TARGET_DATASTREAM_KEY

# changing these does not change the hash
post_config = {
    'use_tensorflow': True,
    'tf_log_memory_usage': True,
    'cleanup_old_models': True,
    'log_batch_size': True,
    'debug_print_layer_output_template': True,
    'debug_mode': False,
    'batching': 'random'
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


def get_training_config(datasets: TrainingDatasets):
    config = {
        "batch_size": 900,
        "max_seq_length": 602,
        "max_seqs": 32,
        "chunking": "0",
        "calculate_exp_loss": True,
        "gradient_clip_global_norm": 2.0,
        "gradient_noise": 0.,
        "learning_rate": 1.,
        "learning_rate_control": "newbob_rel",
        "learning_rate_control_relative_error_relative_lr": False,
        "newbob_learning_rate_decay": 0.8,
        "newbob_relative_error_threshold": 0,
        "learning_rate_control_error_measure": "dev_score_output:exp",
        #############
        "train": datasets.train.as_returnn_opts(),
        "dev": datasets.cv.as_returnn_opts(),
    }

    config["network"] = {
        "input": {"class": "linear", "n_out": 128, "activation": "identity",
                  "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "from": [f"data:{SOURCE_DATASTREAM_KEY}"]},
        "lstm0": {"class": "rec", "unit": "lstm",
                  "forward_weights_init" : "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "n_out": 4096, "dropout": 0.2, "L2": 0.0, "direction": 1, "from": ["input"]},
        "lstm1": {"class": "rec", "unit": "lstm",
                  "forward_weights_init" : "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "recurrent_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                  "n_out": 4096, "dropout": 0.2, "L2": 0.0, "direction": 1, "from": ["lstm0"]},
        "output": {"class": "softmax", "dropout": 0.2, "use_transposed_weights": True,
                   "loss_opts": {'num_sampled': 16384, 'use_full_softmax': True, 'nce_loss': False},
                   "forward_weights_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                   "bias_init": "random_normal_initializer(mean=0.0, stddev=0.1)",
                   "loss": "sampling_loss", "target": f"data:{TARGET_DATASTREAM_KEY}", "from": ["lstm1"]}
    }

    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        python_prolog=[CF_CODE, STACK_CODE],
        python_prolog_hash="",
    )

    return returnn_config
