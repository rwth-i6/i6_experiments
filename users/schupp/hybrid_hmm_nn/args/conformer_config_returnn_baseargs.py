from ast import Or
from typing import OrderedDict


config_baseline_00 = {
      'task': "train",
      'use_tensorflow': True,
      'multiprocessing': True,
      'update_on_device': True,
      'stop_on_nonfinite_train_score': False,
      'log_batch_size': True,
      'debug_print_layer_output_template': True,
      'tf_log_memory_usage': True,
      'start_epoch': "auto",
      'start_batch': "auto",
      'batching': "sort_bin_shuffle:.64",  # f"laplace:{num_seqs//1000}"
      'batch_size': 6144,
      'chunking': "200:100",
      'truncation': -1,
      'cache_size': "0",
      'window': 1,
      'num_inputs': 50,
      'num_outputs': {
        'data': [50, 2],
        'classes': [12001, 1]
      },
      'target': 'classes',
      'optimizer' : {"class" : "nadam"},
      'optimizer_epsilon': 1e-8,
      'gradient_noise': 0.0,  # 0.1
      'learning_rate_control': "constant",
      'learning_rate_file': "learning_rates",
}

search_job_dispatcher_defaults = {
  'epochs' : [10, 40, 100, 140, 160, 180, 190, 195, 200]
}

# --------------- Conformer overall args -----------------

conformer_default_args_00 = OrderedDict(
  num_blocks = 2
)

# -------------- Sampling args --------------

sampling_default_args_00 = OrderedDict(
  time_reduction=2
)

# -------------- Feed forward -----------------

ff_default_args_00 = OrderedDict(
    ff_dim = 2048,
    ff_activation = "swish",
    ff_activation_dropout = 0.1, # TODO: check
    ff_post_dropout = 0.1, # TODO: check
    ff_half_ratio = 0.5,
)

# ------------ Self attention ------------------

sa_default_args_00 = OrderedDict(
    num_heads = 4,
    key_dim = 256, # TODO: check
    value_dim = 256, # TODO: check
    attention_left_only = True,
    sa_dropout = 0.1, # TODO: check
    linear_mapping_bias = False,
    sa_post_dropout = 0.1, # TODO: check
)

# -------------- Conv mod -----------------------

conv_default_args_00 = OrderedDict(
    kernel_size = 32,
    conv_act = "swish",
    conv_post_dropout = 0.1, # TODO: check
)

# ---------------- Shared -----------------------

# Args are shared with layers
shared_network_args_00 = OrderedDict(
  model_dim = 256, # TODO: check
  initialization = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
)