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
    _cf_cache[cached_fn] = cached_fn  # prevent re-caching if called again with the cached path
    return cached_fn


# Patch RETURNN's own cf() (used by HDFDataset.use_cache_manager) with the same
# identity-entry trick, so re-initialization of HDFDataset across MultiEpochDataset
# sub-epochs doesn't trigger redundant system cf calls on already-cached paths.
import returnn.util.basic as _returnn_util

_orig_returnn_cf = _returnn_util.cf


def _cf_dedup(filename):
    result = _orig_returnn_cf(filename)
    _returnn_util._cf_cache[result] = result
    return result


_returnn_util.cf = _cf_dedup


# Patch HDFDataset.__reduce__ to suppress use_cache_manager when pickling.
# Dataset.__reduce__ reconstructs via __init__ using the current self.files, which by then
# hold the already-cached paths. Without this patch, unpickling in a worker
# process calls cf() on those cached paths again, producing doubled destinations.
# Setting use_cache_manager=False in the pickle kwargs lets the worker open
# the already-local files directly.
from returnn.datasets.hdf import HDFDataset as _HDFDataset

_orig_hdf_reduce = _HDFDataset.__reduce__


def _patched_hdf_reduce(self):
    creator_func, (dataset_cls, kwargs, state) = _orig_hdf_reduce(self)
    kwargs = dict(kwargs)
    kwargs["use_cache_manager"] = False
    return creator_func, (dataset_cls, kwargs, state)


_HDFDataset.__reduce__ = _patched_hdf_reduce
