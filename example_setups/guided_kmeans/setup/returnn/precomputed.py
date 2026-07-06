#!rnn.py

import os
import resource
import sys
from returnn.tensor import Dim, batch_dim

sys.path.insert(0, "/u/mann/src/sisyphus")


allow_random_model_init = True
backend = "torch"
batch_size = 2400000
default_input = "data"
device = "gpu"

max_seqs = 200
torch_log_memory_usage = False

config = {}


Dim._SimpleEquality = True
time_dim = Dim(description="time", dimension=None, kind=Dim.Types.Spatial)
pca_dim = Dim(description="audio", dimension=512, kind=Dim.Types.Feature)

extern_data = {"data": {"dim_tags": [batch_dim, time_dim, pca_dim]}}

model_outputs = {
    "output": {
    "dims": [ batch_dim, time_dim, pca_dim ],
        "dtype": "float32",
    }
}

from i6_experiments.example_setups.guided_kmeans.lib.models.returnn_iface import (  # noqa: E402
    get_dummy_model as get_model,  # noqa: F401
)
from i6_experiments.example_setups.guided_kmeans.lib.models.returnn_iface import ( # noqa: F401
    forward_step_passthrough as forward_step
)

# https://github.com/rwth-i6/returnn/issues/957
# https://stackoverflow.com/a/16248113/133374
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
        print("Cache manager: Error occurred, using local file")
        return filename
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn
