#!rnn.py

"""
Directly run this config with RETURNN rnn.py, to test some model.
"""

task = "initialize_model"
target = "classes"
use_tensorflow = True
behavior_version = 12
default_input = "data"
batching = "sorted"
batch_size = 20000
max_seqs = 200
log = ["/tmp/dummy-returnn-model.log"]
log_verbosity = 3
log_batch_size = True
tf_log_memory_usage = True
tf_session_opts = {"gpu_options": {"allow_growth": True}}
model = "/tmp/dummy-returnn-model"

import os
import sys

_base_dir, _ = os.path.abspath(__file__).rsplit("/recipe/i6_experiments/", 2)
sys.path.insert(0, f"{_base_dir}/recipe")
sys.path.insert(1, f"{_base_dir}/ext/sisyphus")
sys.path.insert(2, f"{_base_dir}/tools/sisyphus")

from returnn_common import nn

from returnn.tf.util.data import (
    Dim,
    batch_dim,
    single_step_dim,
    SpatialDim,
    FeatureDim,
    ImplicitDynSizeDim,
    ImplicitSparseDim,
)

time_dim = SpatialDim("time")
audio_dim = FeatureDim("audio", 80)
out_spatial_dim = SpatialDim("out-spatial")
phones_dim = FeatureDim("phones", 61)

extern_data = {
    "data": {"dim_tags": [batch_dim, time_dim, audio_dim]},
    "classes": {
        "dim_tags": [batch_dim, out_spatial_dim],
        "sparse_dim": phones_dim,
        "vocab": {
            "class": "Vocabulary",
            "labels": [
                "pau",
                "aa",
                "ae",
                "ah",
                "ao",
                "aw",
                "ax",
                "ax-h",
                "axr",
                "ay",
                "b",
                "bcl",
                "ch",
                "d",
                "dcl",
                "dh",
                "dx",
                "eh",
                "el",
                "em",
                "en",
                "eng",
                "epi",
                "er",
                "ey",
                "f",
                "g",
                "gcl",
                "h#",
                "hh",
                "hv",
                "ih",
                "ix",
                "iy",
                "jh",
                "k",
                "kcl",
                "l",
                "m",
                "n",
                "ng",
                "nx",
                "ow",
                "oy",
                "p",
                "pcl",
                "q",
                "r",
                "s",
                "sh",
                "t",
                "tcl",
                "th",
                "uh",
                "uw",
                "ux",
                "v",
                "w",
                "y",
                "z",
                "zh",
            ],
            "vocab_file": None,
            "unknown_label": None,
            "user_defined_symbols": {"<sil>": 0},
        },
    },
}

from i6_experiments.users.zeyer.experiments.exp2022_07_21_transducer.pipeline_swb_2020 import (
    from_scratch_model_def as _model_def,
)
from i6_experiments.users.zeyer.experiments.exp2022_07_21_transducer.pipeline_swb_2020 import (
    model_recog as _recog_def,
)
from i6_experiments.users.zeyer.recog import _returnn_get_network as get_network

# https://github.com/rwth-i6/returnn/issues/957
# https://stackoverflow.com/a/16248113/133374
import resource
import sys

try:
    resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
sys.setrecursionlimit(10**6)
_cf_cache = {}


def cf(filename):
    "Cache manager"
    from subprocess import check_output, CalledProcessError

    if filename in _cf_cache:
        return _cf_cache[filename]
    if int(os.environ.get("RETURNN_DEBUG", "0")):
        print("use local file: %s" % filename)
        return filename  # for debugging
    try:
        cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    except CalledProcessError:
        print("Cache manager: Error occured, using local file")
        return filename
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn

# -*- mode: python; tab-width: 4 -*-
