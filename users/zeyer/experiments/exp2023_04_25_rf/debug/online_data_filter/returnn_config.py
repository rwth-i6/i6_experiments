#!rnn.py

"""
Run this directly:

Prepare PYTHONPATH such that it can import i6_core, i6_experiments, sisyphus and returnn.
(Or maybe `pip install returnn`, so then only the others.)

python3 .../returnn/rnn.py .../returnn_config.py
"""

from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear


accum_grad_multiple_step = 1
aux_loss_layers = [4, 8]
backend = "torch"
batch_size = 2400000
# batching = "laplace:.1000"
behavior_version = 20
cleanup_old_models = {"keep_last_n": 5}
default_input = "data"
device = "gpu"
dynamic_learning_rate = dyn_lr_piecewise_linear

# train = ...
# eval_datasets = ...
dry_run = True
use_dummy_datasets = True
train = 1
eval_datasets = {"dev": 1, "devtrain": 1}

grad_scaler = None
gradient_clip_global_norm = 5.0
learning_rate = 1.0
learning_rate_control_error_measure = "dev_score_ce"
learning_rate_file = "learning_rates"
learning_rate_invsqrt_norm = 20000
learning_rate_piecewise_steps = [295000, 590000, 652000]
learning_rate_piecewise_values = [1e-05, 0.001, 1e-05, 1e-06]
learning_rate_warmup_steps = 20000
log = ["./returnn.log"]
log_batch_size = True
log_verbosity = 5
max_seq_length = {"classes": 75}
max_seqs = 200
# model = "/u/zeyer/setups/combined/2021-05-31/work/i6_core/returnn/training/ReturnnTrainingJob.ImJqqRZe5GaO/output/models/epoch"
newbob_multi_num_epochs = 20
num_epochs = 500
optimizer = {
    "class": "adamw",
    "epsilon": 1e-16,
    "weight_decay": 0.01,
    "weight_decay_modules_blacklist": [
        "rf.Embedding",
        "rf.LearnedRelativePositionalEncoding",
    ],
}
pos_emb_dropout = 0.1
rf_att_dropout_broadcast = False
save_interval = 1
specaugment_steps = (5000, 15000, 25000)
speed_pert_discrete_values = [0.7, 0.8, 0.9, 1.0, 1.1]
target = "classes"
task = "train"
torch_dataloader_opts = {"num_workers": 1}
# torch_distributed = {"param_sync_step": 100, "reduce_type": "param"}
torch_log_memory_usage = True
# use_horovod = True
use_last_best_model = None
use_lovely_tensors = True
# use_train_proc_manager = True
watch_memory = True

from returnn.tensor import Dim, batch_dim

time_dim = Dim(description="time", dimension=None, kind=Dim.Types.Spatial)
audio_dim = Dim(description="audio", dimension=1, kind=Dim.Types.Feature)
out_spatial_dim = Dim(description="out-spatial", dimension=None, kind=Dim.Types.Spatial)
vocab_dim = Dim(description="vocab", dimension=10025, kind=Dim.Types.Spatial)


def _make_dummy_labels():
    return [f"<label-{i}>" for i in range(vocab_dim.dimension)]


extern_data = {
    "data": {"dim_tags": [batch_dim, time_dim, audio_dim]},
    "classes": {
        "dim_tags": [batch_dim, out_spatial_dim],
        "sparse_dim": vocab_dim,
        "vocab": {
            "vocab_file": None,
            "labels": _make_dummy_labels,
            "unknown_label": None,
            "bos_label": 0,
            "eos_label": 0,
        },
    },
}

from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.aed_online_data_filter import (
    from_scratch_model_def as _model_def,
)
from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.train import (
    _returnn_v2_get_model as get_model,
)
from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.aed_online_data_filter import (
    from_scratch_training as _train_def,
)
from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.train import (
    _returnn_v2_train_step as train_step,
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


# -*- mode: python; tab-width: 4 -*-
