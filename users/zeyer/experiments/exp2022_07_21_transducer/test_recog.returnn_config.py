#!rnn.py

from typing import Tuple

task = "initialize_model"
target = "classes"
use_tensorflow = True
behavior_version = 12
default_input = "data"
batching = "sorted"
batch_size = 20000
max_seqs = 200
# log = ["./test-recog.returnn.log"]
log_verbosity = 3
log_batch_size = True
tf_log_memory_usage = True
tf_session_opts = {"gpu_options": {"allow_growth": True}}
model = "dummy-model-init"
config = {}

locals().update(**config)

import os
import sys

sys.path.insert(0, "/Users/az/i6/setups/2022-03-19--sis-i6-exp/recipe")
sys.path.insert(1, "/Users/az/i6/setups/2022-03-19--sis-i6-exp/ext/sisyphus")
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
                "pau", "aa", "ae", "ah", "ao", "aw", "ax", "ax-h", "axr", "ay", "b", "bcl", "ch", "d", "dcl",
                "dh", "dx", "eh", "el", "em", "en", "eng", "epi", "er", "ey", "f", "g", "gcl", "h#", "hh", "hv",
                "ih", "ix", "iy", "jh", "k", "kcl", "l", "m", "n", "ng", "nx", "ow", "oy", "p", "pcl", "q", "r",
                "s", "sh", "t", "tcl", "th", "uh", "uw", "ux", "v", "w", "y", "z", "zh"],
            "vocab_file": None,
            "unknown_label": None,
            "user_defined_symbols": {"<sil>": 0},
        },
    },
}


from i6_experiments.users.zeyer.experiments.exp2022_07_21_transducer.pipeline_swb_2020 import Model, _get_bos_idx
from i6_experiments.users.zeyer.experiments.exp2022_07_21_transducer.recog import IDecoder, beam_search


def _model_def(*, epoch: int, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    return Model(
        num_enc_layers=min((epoch - 1) // 2 + 2, 6) if epoch <= 10 else 6,
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
    )


def _recog_def(*,
               model: Model,
               data: nn.Tensor, data_spatial_dim: nn.Dim,
               targets_dim: nn.Dim,  # noqa
               ) -> nn.Tensor:
    """
    Function is run within RETURNN.

    :return: recog results including beam
    """
    batch_dims = data.batch_dims_ordered(data_spatial_dim)
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    class _Decoder(IDecoder):
        target_spatial_dim = enc_spatial_dim  # time-sync transducer
        include_eos = True

        def max_seq_len(self) -> nn.Tensor:
            """max seq len"""
            return nn.dim_value(enc_spatial_dim) * 2

        def initial_state(self) -> nn.LayerState:
            """initial state"""
            return model.decoder_default_initial_state(batch_dims=batch_dims)

        def bos_label(self) -> nn.Tensor:
            """BOS"""
            return nn.constant(model.blank_idx, shape=batch_dims, sparse_dim=model.wb_target_dim)

        def __call__(self, prev_target: nn.Tensor, *, state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
            enc = model.encoder_unstack(enc_args)
            probs, state = model.decode(
                **enc,
                enc_spatial_dim=nn.single_step_dim,
                wb_target_spatial_dim=nn.single_step_dim,
                prev_wb_target=prev_target,
                state=state)
            return probs.get_wb_label_log_probs(), state

    return beam_search(_Decoder())


from i6_experiments.users.zeyer.experiments.exp2022_07_21_transducer.recog import (
    _returnn_get_network as get_network,
)

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
