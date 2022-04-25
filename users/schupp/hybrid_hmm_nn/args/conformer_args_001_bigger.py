from ast import Or
from typing import OrderedDict

import math
import numpy

def lr1(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
       min_lr_ratio=1/50, decay_factor=0.99):
  num_lr = int(math.log(min_lr_ratio, decay_factor))
  return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                    [start] * constant_subepoch + \
                    list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                    [min_lr_ratio * start]

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
      'batch_size': 7244,
      'chunking': "400:200", # Bigger chunk
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
      'learning_rates' : lr1(warmup_subepoch=2, constant_subepoch=18, decay_factor=0.99) # TODO: handle this differently
}

returnn_rasr_args_defaults_00 = OrderedDict(
    feature_name = 'gammatone',
    alignment_name = 'align_hmm',
    num_classes = 12001,
    num_epochs = 200,
    partition_epochs = {'train': 20, 'dev': 1},
)

returnn_train_post_config_00 = OrderedDict(
  cleanup_old_models = {
    'keep': [10, 40, 100, 140, 160, 180, 190, 195, 200],
    'keep_best_n': 3, 
    'keep_last_n': 3}
)

# --------------- Conformer overall args -----------------

conformer_default_args_00 = OrderedDict(
  num_blocks = 12
)

# -------------- Sampling args --------------

sampling_default_args_00 = OrderedDict(
  time_reduction=2,
  embed_l2 = 1e-7,
  embed_dropout = 0.05,
)

# -------------- Feed forward -----------------

ff_default_args_00 = OrderedDict(
    ff_dim = 1024, # 2048 before
    ff_activation = "swish",
    ff_activation_dropout = 0.1, # TODO: check
    ff_post_dropout = 0.1, # TODO: check
    ff_half_ratio = 0.5,
)

# ------------ Self attention ------------------

sa_default_args_00 = OrderedDict(
    num_heads = 8,
    key_dim = 512, # TODO: check
    value_dim = 512, # TODO: check
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
  model_dim = 512, # Bigger
  initialization = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
)