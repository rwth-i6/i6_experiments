from i6_core.returnn.config import ReturnnConfig

# from .zeineldeen_helpers.models.lm.transformer_lm import TransformerLM
from i6_experiments.users.zeineldeen.models.lm.transformer_lm import TransformerLM
from .data import TrainingDatasets, SOURCE_DATASTREAM_KEY, TARGET_DATASTREAN_KEY

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
        "batch_size": 1350,
        "max_seq_length": 1350,
        "max_seqs": 32,
        "chunking": "0",
        "calculate_exp_loss": True,
        "gradient_clip_global_norm": 1.0,
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

    trafo_lm = TransformerLM(
        source="data:" + SOURCE_DATASTREAM_KEY,
        target="data:" + TARGET_DATASTREAN_KEY,
        num_layers=24,
        ff_dim=4096,
        att_num_heads=8,
        vocab_size=datasets.extern_data[SOURCE_DATASTREAM_KEY]["dim"]
    )
    trafo_ext_lm = TransformerLM(
        source="data:" + SOURCE_DATASTREAM_KEY,
        target="data:" + TARGET_DATASTREAN_KEY,
        num_layers=24,
        ff_dim=4096,
        att_num_heads=8,
        vocab_size=datasets.extern_data[SOURCE_DATASTREAM_KEY]["dim"],
        use_as_ext_lm=True
    )
    trafo_lm.create_network()
    trafo_ext_lm.create_network()
    config["network"] = trafo_lm.network.get_net()

    # the current LM helpers have a bug for training, thus remove the invalid max_seq_len entry
    config["network"]["output"].pop("max_seq_len")

    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        python_prolog=[CF_CODE, STACK_CODE],
        python_prolog_hash="",
    )

    return returnn_config, trafo_ext_lm.network.get_net()



