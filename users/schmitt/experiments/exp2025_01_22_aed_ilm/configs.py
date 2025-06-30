"""
Configurations, i.e. RETURNN settings,
shared across several setups here in this directory.
"""

from __future__ import annotations
from typing import Any, Dict

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from .model.decoder import GlobalAttDecoder
from .train import _returnn_v2_train_step, from_scratch_training

import returnn.frontend as rf

_batch_size_factor = 160

config_24gb_num_epochs = 2000
config_24gb_v1 = dict(
  model_opts=dict(
    decoder_opts=rf.build_dict(
      GlobalAttDecoder,
      use_weight_feedback=True,
      use_att_ctx_in_state=True,
    )
  ),
  train_opts=dict(
    train_step_func=_returnn_v2_train_step,
    train_def=from_scratch_training,
    max_seq_length={"data": 19.5 * 16_000},  # 19.5 seconds
    rf_att_dropout_broadcast=False,
    specaugment_steps=(5_000, 15_000, 25_000),
    accum_grad_multiple_step=2,
    batch_size=35_000,
    batching="laplace:.1000",
    grad_scaler=None,
    gradient_clip_global_norm=5.0,
    optimizer={
      "class": "adamw",
      "epsilon": 1e-8,
      "weight_decay": 1e-6,
      "weight_decay_modules_blacklist": [
        "rf.Embedding",
        "rf.LearnedRelativePositionalEncoding",
      ],
    },
    num_epochs=config_24gb_num_epochs,
    lr_opts=dict(
      type="dyn_lr_piecewise_linear_epoch-wise",
      peak_lr=1e-3,
      init_lr=1e-3 * 1e-2,
      num_epochs=config_24gb_num_epochs,
    ),
    dataset_opts=dict(
      use_speed_pert=True,
      epoch_wise_filter={(1, 5): {"max_mean_len": 1000}},
    ),
    aux_loss_layers=(),
    cleanup_old_models=dict(
      keep=[config_24gb_num_epochs],
      keep_best_n=4,
      keep_last_n=1,
    )
  )
)

config_11gpu_mgpu_num_epochs = 500
config_11gb_mgpu_v1 = dict_update_deep(
  config_24gb_v1,
  {
    "train_opts.accum_grad_multiple_step": 4,
    "train_opts.batch_size": 15_000,
    "train_opts.num_epochs": config_11gpu_mgpu_num_epochs,
    "train_opts.lr_opts.num_epochs": config_11gpu_mgpu_num_epochs,
    "train_opts.cleanup_old_models.keep": [config_11gpu_mgpu_num_epochs],
    "train_opts.torch_distributed": {"reduce_type": "param", "param_sync_step": 100},
  }
)

mini_lstm_num_epochs = 10
mini_lstm_config_v1 = dict(
  model_opts=dict(
    decoder_opts=rf.build_dict(
      GlobalAttDecoder,
      use_weight_feedback=True,
      use_att_ctx_in_state=True,
      use_mini_att=True,
    )
  ),
  train_opts=dict(
    train_step_func=_returnn_v2_train_step,
    train_def=from_scratch_training,
    max_seq_length={"data": 19.5 * 16_000},  # 19.5 seconds
    rf_att_dropout_broadcast=False,
    accum_grad_multiple_step=4,
    batch_size=15_000,
    batching="random",
    optimizer={
      "class": "adamw",
      "epsilon": 1e-8,
      "weight_decay": 1e-6,
      "weight_decay_modules_blacklist": [
        "rf.Embedding",
        "rf.LearnedRelativePositionalEncoding",
      ],
    },
    num_epochs=mini_lstm_num_epochs,
    lr_opts={
      "type": "const_then_linear",
      "const_lr": 1e-4,
      "const_frac": 1 / 3,
      "final_lr": 1e-6,
      "num_epochs": mini_lstm_num_epochs
    },
    aux_loss_layers=(),
    cleanup_old_models=dict(
      keep=[mini_lstm_num_epochs],
      keep_best_n=1,
      keep_last_n=1,
    )
  )
)
