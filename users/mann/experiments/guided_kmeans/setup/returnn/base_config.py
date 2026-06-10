#!rnn.py

import os
import resource
import sys
from returnn.tensor import Dim, batch_dim

sys.path.insert(0, "/u/mann/src/sisyphus")
# sys.path.insert(0, "/u/mann/experiments/clones/2025-05-30--marten-unsupervised/recipe")
sys.path.insert(0, "/work/asr3/michel/mann/tools/rasr/librasr_recog/arch/linux-x86_64-standard")


allow_random_model_init = True
aux_loss_layers = [4, 8]
backend = "torch"
batch_size = 2400000
behavior_version = 21
default_input = "data"
device = "gpu"
enc_conformer_layer = {
    "class": "returnn.frontend.encoder.conformer.ConformerEncoderLayer",
    "ff_activation": {"class": "rf.relu_square"},
    "num_heads": 8,
}

log = ["./returnn.log"]
log_verbosity = 3
max_seqs = 200
newbob_multi_num_epochs = 20

preload_from_files = {
    "wav2vec2_base": {
        "checkpoint_key": None,
        "filename": "/u/mann/experiments/clones/2025-05-30--marten-unsupervised/work/i6_core/tools/download/DownloadJob.6tpoB1gucOXM/output/wav2vec2_large_60kh_no_finetune.bin",
        "ignore_missing": True,
        "init_for_train": True,
    }
}
rf_att_dropout_broadcast = False
target = "classes"
task = "forward"
torch_log_memory_usage = False

use_last_best_model = None
use_w2v_model = True
w2v_opts = {
    "config_file": "/u/mann/experiments/clones/2025-05-30--marten-unsupervised/work/i6_core/tools/download/DownloadJob.wycWPzGMhrAV/output/wav2vec2_large_60kh_no_finetune_config.json",
    "freeze_encoder_first_n_steps": 2500,
    "num_enc_layers": 15,
    "enc_logits_n_layers": 0,
}
config = {}

locals().update(**config)

Dim._SimpleEquality = True
time_dim = Dim(description="time", dimension=None, kind=Dim.Types.Spatial)
audio_dim = Dim(description="audio", dimension=1, kind=Dim.Types.Feature)
out_spatial_dim = Dim(description="out-spatial", dimension=None, kind=Dim.Types.Spatial)
vocab_dim = Dim(description="vocab", dimension=184, kind=Dim.Types.Feature)

extern_data = {"data": {"dim_tags": [batch_dim, time_dim, audio_dim]}}

model_outputs = {
    "output": {
        "dims": [
            batch_dim,
            Dim(description="wav2vec_seq", dimension=None, kind=Dim.Types.Spatial),
            Dim(description="enc", dimension=1024, kind=Dim.Types.Feature)
        ],
        "dtype": "float32",
    }
}

from i6_experiments.users.mann.lib.models.w2v.returnn_iface import (  # noqa: E402
    w2v_model_def as model_def,  # noqa: F401
)
from i6_experiments.users.mann.lib.models.returnn_iface import (  # noqa: E402
    encoder_forward as forward_def,  # noqa: F401
)
from i6_experiments.users.mann.lib.models.returnn_iface import (  # noqa: E402
    get_model_generic as get_model,  # noqa: F401
)
from i6_experiments.users.mann.lib.models.returnn_iface import forward_step  # noqa: F401

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
