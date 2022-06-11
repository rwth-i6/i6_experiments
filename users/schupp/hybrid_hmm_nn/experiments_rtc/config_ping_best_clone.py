# TODO: package, make imports smaller
from typing import OrderedDict
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import setup_god as god
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_config_returnn_baseargs as experiment_config_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_returnn_dict_network_generator

from sisyphus import gs
import copy
import numpy
import math

import inspect

OUTPUT_PATH = "conformer/ping_clone/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH
NAME = "ping-clone"

# This config is supposed to be a clone of pings best conformer:
# /work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10/work/crnn/sprint_training/CRNNSprintTrainingJob.R6Ivh9zimxO3/output/crnn.config
# At line 410 + from : `/work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10/config/conformer_half_specaug.py`

ORIGINAL_CONFIG = '/work/asr3/luescher/hiwis/pzheng/librispeech/transformer_conformer_21_11_10/work/crnn/sprint_training/CRNNSprintTrainingJob.R6Ivh9zimxO3/output/crnn.config'

def lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
       min_lr_ratio=1/50, decay_factor=0.99):

  num_lr = int(math.log(min_lr_ratio, decay_factor))
  return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                   [start] * constant_subepoch + \
                   list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                   [min_lr_ratio * start]

EP_SPLIT = 40

specaug_args = OrderedDict(
      max_len_feature = 15,
      max_len_time = 20,
      max_reps_feature = 1,
      max_reps_time = 20,
      min_learning_rate = 1e-05,
      min_reps_feature = 0,
      min_reps_time = 0,
)

config_args =  {
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
      'batching': "sort_bin_shuffle:.64",
      'batch_size': 6144,
      'chunking': "400:200",
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
      'behavior_version' : 12, # DIFFERENCE ( this is supposed to be the only difference here )
      'learning_rate_control': "constant",
      'learning_rate_file': "learning_rates",
      'learning_rates' : lr(),#
      **specaug_args
}

returnn_rasr_args_defaults = OrderedDict(
    feature_name = 'gammatone',
    alignment_name = 'align_hmm',
    num_classes = 12001,
    num_epochs = 600,
    partition_epochs = {'train': EP_SPLIT, 'dev': 1},
    shuffle_data = True, # Adds some etra ars to the sprint train call
)

returnn_train_post_config = OrderedDict(
  cleanup_old_models =  {'keep': [50, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600], 'keep_best_n': 3, 'keep_last_n': 3}
)

# --------------- Conformer overall args -----------------

conformer_defaults = OrderedDict(
  num_blocks = 12
)

# -------------- Sampling args --------------

sampling_default_args = OrderedDict(
  time_reduction=1,
  unsampling_strides = 3,
  embed_l2 = 0.0,
  embed_dropout = 0.0,
  stacking_stride = 3,
  window_size = 3,
  window_left = 2,
  window_right = 0,
)

# -------------- Feed forward -----------------

ff_default_args = OrderedDict(
    ff_dim = 2048,
    ff_activation = "swish",
    ff_activation_dropout = 0.1,
    ff_post_dropout = 0.1,
    ff_half_ratio = 0.5,
)

# ------------ Self attention ------------------

sa_default_args = OrderedDict(
    num_heads = 8,
    key_dim = 512,
    value_dim = 512,
    attention_left_only = False,
    sa_dropout = 0.1,
    linear_mapping_bias = False,
    sa_post_dropout = 0.1,
    fixed_pos = False,
    clipping = 400,
)

# -------------- Conv mod -----------------------

conv_default_args = OrderedDict(
    kernel_size = 32,
    conv_act = "swish",
    conv_post_dropout = 0.1,
)

# ---------------- Shared -----------------------

# Args are shared with layers
shared_network_args = OrderedDict(
  model_dim = 512,
  initialization = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)",
)

auxilary_loss_args = OrderedDict(
  aux_dim = 256,
  aux_strides = 3
)

experiment_data = god.create_experiment_world_001(
  name=NAME,
  output_path=OUTPUT_PATH,
  config_base_args=config_args,
  conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_03_feature_stacking_auxilary_loss,
  conformer_func_args=OrderedDict(
    # sampling args
    sampling_func_args = sampling_default_args,

    # Feed forward args, both the same by default
    ff1_func_args = ff_default_args,
    ff2_func_args = ff_default_args,

    # Self attention args
    sa_func_args = sa_default_args,

    # Conv mod args
    conv_func_args = conv_default_args,

    # Shared model args
    shared_model_args = shared_network_args,

    auxilary_at_layer = [6],
    auxilary_loss_args = auxilary_loss_args,

    # Conformer args
    **conformer_defaults ),
    returnn_train_post_config=returnn_train_post_config,
    returnn_rasr_args_defaults=returnn_rasr_args_defaults,

    test_construction=True,
    final_recog=False,
    new_final_recog=True,
    #write_dummpy_config="./test.config"
)

# ---------------------------------------- Sanity check config comparison --------------------------------------

## Ok to prove we replicated the config 
# we can just import the original config as python file
# much hacky this is 

import importlib.util
spec = importlib.util.spec_from_loader('ping_config', loader=None)
ping_config = importlib.util.module_from_spec(spec)

data = None
with open(ORIGINAL_CONFIG, 'r') as file:
    data = file.read()

exec(data, ping_config.__dict__)
# Now we got the whole config loaded as 'ping_config'

# check 1 all config args:
dont_check = [ # Thinks that cant match because of the behavior version
  "optimizer", # New syntax ping just uses nadam = True, bhv12 needs optimizer = {...}
  "behavior_version", # Pings config doesn't use this
  "learning_rates", # We could check assert close here ...
  "cleanup_old_models" # I have to extra epochs here ( 50, others match... )
]

for x in config_args:
  if x in dont_check:
    continue
  p = getattr(ping_config, x)
  t = config_args[x]
  assert p == t, f"not matching arg ({x}): \n {p} \nvs\n {t}"

# Check 2 is the network dict the same???
# First we do this using sorted jsons, because that make several thins easier:

import json
net_p = ping_config.network
net_t = experiment_data["network"]
net_p_keys = list(net_p.keys())
net_t_keys = list(net_t.keys())

# First do they all have the same keys?
import numpy as np
assert sorted(net_p_keys) == sorted(net_t_keys), f" not same keys:\nin ping but not t: {str(np.setdiff1d(net_p_keys,net_t_keys))}\n in t but not in p: {str(np.setdiff1d(net_t_keys,net_p_keys))}"

def diff_check(expected, actual):
  import difflib
  expected=expected.splitlines(1)
  actual=actual.splitlines(1)

  diff=difflib.unified_diff(expected, actual)

  return ''.join(diff)

pretty_nets = [json.dumps(x, indent=1, sort_keys=True) for x in [net_p, net_t]]

if False: # Write to local files, makes comparison easier
  for x, n in zip(pretty_nets, ["ping", "tim"]):
    with open(f"./{n}.json.conf", "w") as file:
      file.write(x)

# This check can never be 100% because we have all the in_spatial_dims crap from bhv=12
# But if we check the main args we can assert some equality

# Uncommon keys:
uncommon_keys = ["in_spatial_dims", "axes", "axis"]

# But we can delte all uncommon keys, and wrap all 'str' froms in [] then they shoule be 100% the same!
for net in [net_t, net_p]:
  for x in net:
    for k in uncommon_keys:
      if k in net[x]:
        del net[x][k]
    if "from" in net[x]:
      if isinstance(net[x]["from"], str):
        net[x]["from"] = [net[x]["from"]]

pretty_nets2 = [json.dumps(x, indent=1, sort_keys=True) for x in [net_p, net_t]]
diff = diff_check(*pretty_nets2) # Now the only difference is: ( also cause of bhv=12 )
#-  "eval": "self.network.get_config().typed_value('transform')(source(0), network=self.network)"
#+  "eval": "self.network.get_config().typed_value('transform')(source(0), network=self.network)",
#+  "from": [
#+   "data"
#+  ]
diff_l = diff.split("\n")
print(diff_l)
print(len(diff_l))
assert len(diff_l) < 20, "Nope that aint making sense" # Has to be good enough check, I mean look at the configs they are the same :P


# TODO: there is one last thing that I din't manage to replicate jet:
# I'm missing these rasr sprint arguments:
# --*.corpus.segment-order-shuffle=true ' '--*.segment-order-sort-by-time-length=true --*.segment-order-sort-by-time-length-chunk-size=-1
# (tried adding them via train_crp in recipe/i6_experiments/users/schupp/hybrid_hmm_nn/args/conformer_rasr_config_maker.py din't seem to work )