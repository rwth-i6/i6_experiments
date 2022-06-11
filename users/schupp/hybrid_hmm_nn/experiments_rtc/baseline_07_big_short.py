# This is 'big-short' V 07
# Train for only 3 full epochs i.e.: 120 sub epochs 40 split
# Model as about 86 M params

from atexit import register
from typing import OrderedDict
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import setup_god as god
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_config_returnn_baseargs as experiment_config_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conformer_args_004_bigger_baseline_short_train import original_args_big_baseline_00
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_returnn_dict_network_generator

from sisyphus import gs
import copy
import numpy
import math

import inspect

OUTPUT_PATH = "conformer/baseline_07_big_short/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

BASE = "baseline_07_big_short"

def make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
        min_lr_ratio=1/50, decay_factor=0.99):

    num_lr = int(math.log(min_lr_ratio, decay_factor))
    return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                    [start] * constant_subepoch + \
                    list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                    [min_lr_ratio * start]

def get_deafults_baseline_04(): # Old defaults from previous baseline
  args = original_args_big_baseline_00()
  args.config_args["extra_tag_tim_setup"] = 'baseline-big-short-03'
  del args.returnn_rasr_args_defaults["shuffle_data"] # Not needed cause set by default now
  args.returnn_train_post_config["cleanup_old_models"]["keep"] = [40, 80, 100, 120]

  return args

def get_defaults_baseline_04_shuffle_only():
  args = get_deafults_baseline_04()
  #args.config_args["extra_tag_tim_setup"] = 'baseline-big-short-02' already there from previous

  params = OrderedDict(
          segment_order_shuffle = True,
          segment_order_sort_by_time_length = False, # Already false per default, but lets explicity overwrite it
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  return args

def get_defaults():
  args = original_args_big_baseline_00()
  args.config_args["extra_tag_tim_setup"] = 'baseline-big-short-03'

  del args.returnn_rasr_args_defaults["shuffle_data"] # Not needed cause set by default now
  args.returnn_train_post_config["cleanup_old_models"]["keep"] = [40, 80, 100, 120]

  params = OrderedDict(
          segment_order_shuffle = True,
          segment_order_sort_by_time_length = False, # Already false per default, but lets explicity overwrite it
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
  args.config_args["learning_rates"] = learning_rates

  # and shorter learning rate
  return args


# + allowes to use se blocks for any module
# Note where se is applied differes per block
# ff_mod: After first linear layer
# conv_mod: After depthwise convolution
# att_mod: After self attention
def make_experiment_07_se_block(
  args, 
  NAME,
  aux_loss_layers = [6],
  se_block_for_module = [],
  dummy_config = None,
  test_construct = False,
  devtrain_recog = False
  ):

  # Theses are the versions that use se blocks
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_006_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_005_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.ff_mod_versions import make_ff_mod_004_se_block

  args_map = {
    "ff_mod" : args.ff_default_args,
    "att_mod" : args.sa_default_args,
    "conv_mod" : args.conv_default_args,
  }

  for k in se_block_for_module:
    args_map[k]["use_se_block"] = True # This flag is False per default for all conformer blocks ( i.e.: default is regular module version )

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conformer_returnn_dict_network_generator import make_conformer_04_stoch_depth_dynamic

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_04_stoch_depth_dynamic,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default

      conformer_ff1_func=make_ff_mod_004_se_block,
      conformer_ff2_func=make_ff_mod_004_se_block,

      ff1_func_args = args.ff_default_args,
      ff2_func_args = args.ff_default_args,

      # Self attention args
      conformer_self_att_func=make_self_att_mod_005_se_block,
      sa_func_args = args.sa_default_args,

      # Conv mod args
      conformer_self_conv_func=make_conv_mod_006_se_block,
      conv_func_args = args.conv_default_args,

      # Shared model args
      shared_model_args = args.shared_network_args,

      auxilary_at_layer = aux_loss_layers,
      auxilary_loss_args = args.auxilary_loss_args,

      # Conformer args
      **args.conformer_defaults ),
      returnn_train_post_config=args.returnn_train_post_config,
      returnn_rasr_args_defaults=args.returnn_rasr_args_defaults,

      write_dummpy_config=dummy_config,
      test_construction=test_construct,
     extra_recog_devtrain = devtrain_recog
  )

  return experiment_data


def make_experiment_07_se_block_v2(
  args, 
  NAME,
  aux_loss_layers = [6],
  se_block_for_module = [],
  dummy_config = None,
  test_construct = False
  ):

  # Theses are the versions that use se blocks
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_006_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_005_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.ff_mod_versions import make_ff_mod_004_se_block, make_ff_mod_004_se_block_v2

  args_map = {
    "ff_mod" : args.ff_default_args,
    "att_mod" : args.sa_default_args,
    "conv_mod" : args.conv_default_args,
  }

  for k in se_block_for_module:
    args_map[k]["use_se_block"] = True # This flag is False per default for all conformer blocks ( i.e.: default is regular module version )

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conformer_returnn_dict_network_generator import make_conformer_04_stoch_depth_dynamic

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_04_stoch_depth_dynamic,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default

      conformer_ff1_func=make_ff_mod_004_se_block_v2,
      conformer_ff2_func=make_ff_mod_004_se_block_v2,

      ff1_func_args = args.ff_default_args,
      ff2_func_args = args.ff_default_args,

      # Self attention args
      conformer_self_att_func=make_self_att_mod_005_se_block,
      sa_func_args = args.sa_default_args,

      # Conv mod args
      conformer_self_conv_func=make_conv_mod_006_se_block,
      conv_func_args = args.conv_default_args,

      # Shared model args
      shared_model_args = args.shared_network_args,

      auxilary_at_layer = aux_loss_layers,
      auxilary_loss_args = args.auxilary_loss_args,

      # Conformer args
      **args.conformer_defaults ),
      returnn_train_post_config=args.returnn_train_post_config,
      returnn_rasr_args_defaults=args.returnn_rasr_args_defaults,

      write_dummpy_config=dummy_config,
      test_construction=test_construct,
  )

  return experiment_data


# Allowes for custom dimensions for all modules...
def make_experiment_08_se_constom_dims( # TODO: finish
  args, 
  NAME,
  aux_loss_layers = [6],
  se_block_for_module = [],
  dummy_config = None,
  test_construct = False,
  devtrain_recog = False,
  ):

  # Theses are the versions that use se blocks
  # Only need to modifly conv mod actualy, rest already has fitting dimensions
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_011_se_block_and_pointwise_dim
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_005_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.ff_mod_versions import make_ff_mod_004_se_block

  args_map = {
    "ff_mod" : args.ff_default_args,
    "att_mod" : args.sa_default_args,
    "conv_mod" : args.conv_default_args,
  }

  for k in se_block_for_module:
    args_map[k]["use_se_block"] = True # This flag is False per default for all conformer blocks ( i.e.: default is regular module version )

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conformer_returnn_dict_network_generator import make_conformer_04_stoch_depth_dynamic

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_04_stoch_depth_dynamic,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default

      conformer_ff1_func=make_ff_mod_004_se_block,
      conformer_ff2_func=make_ff_mod_004_se_block,

      ff1_func_args = args.ff_default_args,
      ff2_func_args = args.ff_default_args,

      # Self attention args
      conformer_self_att_func=make_self_att_mod_005_se_block,
      sa_func_args = args.sa_default_args,

      # Conv mod args
      conformer_self_conv_func=make_conv_mod_011_se_block_and_pointwise_dim,
      conv_func_args = args.conv_default_args,

      # Shared model args
      shared_model_args = args.shared_network_args,

      auxilary_at_layer = aux_loss_layers,
      auxilary_loss_args = args.auxilary_loss_args,

      # Conformer args
      **args.conformer_defaults ),
      returnn_train_post_config=args.returnn_train_post_config,
      returnn_rasr_args_defaults=args.returnn_rasr_args_defaults,

      write_dummpy_config=dummy_config,
      test_construction=test_construct,
      extra_recog_devtrain=devtrain_recog
  )

  return experiment_data


def make_experiment_09_se_and_l2(
  args, 
  NAME,
  aux_loss_layers = [6],
  se_block_for_module = [],
  dummy_config = None,
  test_construct = False
  ):

  # Theses are the versions that use se blocks
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_006_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_005_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.ff_mod_versions import make_ff_mod_004_se_block

  args_map = {
    "ff_mod" : args.ff_default_args,
    "att_mod" : args.sa_default_args,
    "conv_mod" : args.conv_default_args,
  }

  for k in se_block_for_module:
    args_map[k]["use_se_block"] = True # This flag is False per default for all conformer blocks ( i.e.: default is regular module version )

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conformer_returnn_dict_network_generator import make_conformer_06_sd_se_l2_dynamic

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_06_sd_se_l2_dynamic,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default

      conformer_ff1_func=make_ff_mod_004_se_block,
      conformer_ff2_func=make_ff_mod_004_se_block,

      ff1_func_args = args.ff_default_args,
      ff2_func_args = args.ff_default_args,

      # Self attention args
      conformer_self_att_func=make_self_att_mod_005_se_block,
      sa_func_args = args.sa_default_args,

      # Conv mod args
      conformer_self_conv_func=make_conv_mod_006_se_block,
      conv_func_args = args.conv_default_args,

      # Shared model args
      shared_model_args = args.shared_network_args,

      auxilary_at_layer = aux_loss_layers,
      auxilary_loss_args = args.auxilary_loss_args,

      # Conformer args
      **args.conformer_defaults ),
      returnn_train_post_config=args.returnn_train_post_config,
      returnn_rasr_args_defaults=args.returnn_rasr_args_defaults,

      write_dummpy_config=dummy_config,
      test_construction=test_construct,
  )

  return experiment_data


# ------------------------- baseline: 'big-short-03' -----------------------
# baseline_04_big_short
# + se-block conv mod

def baseline():
  args = get_defaults()
  NAME = f"{BASE}"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    se_block_for_module = ["conv_mod"], #"ff_mod"],
    devtrain_recog = True
  )



def other_ff_se_version():
  args = get_defaults()
  NAME = f"{BASE}+se-block-v2-ff-mod"

  data = make_experiment_07_se_block_v2(
    args, 
    NAME,
    se_block_for_module = ["ff_mod"],
  )


def se_block_conv_module():
  args = get_defaults()
  NAME = f"{BASE}+se-block-v1.0-conv-mod"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

def num_blocks():
  for n in [10, 11, 13, 14, 15, 16, 17]:
    args = get_defaults()
    NAME = f"{BASE}+num-blocks-{n}-aux-at-half"
    args.conformer_defaults['num_blocks'] = n

    data = make_experiment_07_se_block(
      args, 
      NAME,
      aux_loss_layers = [n//2],
      se_block_for_module = ["ff_mod"],
      devtrain_recog = True

    )

def conv_dim():
  for d in [1024, 1538, 2048]:
    args = get_defaults()
    args.conv_default_args["pointwise_dim"] = d
    NAME = f"{BASE}+conv-dim-{d}"

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
    )

def att_dim():
  for d in [1024, 2048]:
    args = get_defaults()
    args.sa_default_args["key_dim"] = d
    NAME = f"{BASE}+att-dim-{d}"

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
      devtrain_recog = True
    )

def att_value_dim():
  for d in [1024, 2048]:
    args = get_defaults()
    args.sa_default_args["value_dim"] = d
    NAME = f"{BASE}+att-dim-value-dim={d}"

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
      devtrain_recog = True
    )

def ff_dim():
  for d in [1024, 3072, 4096]:
    args = get_defaults()
    args.ff_default_args["ff_dim"] = d
    NAME = f"{BASE}+ff-dim-{d}"

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
      devtrain_recog = True
    )

def frame_stacking_time_down_factor():
  for sample in [1, 2, 4]: # 3 is baseline
    args = get_defaults()
    NAME = f"{BASE}+donw-fact={sample}"

    args.sampling_default_args["stacking_stride"] = sample
    args.sampling_default_args["unsampling_strides"] = sample

    if sample == 1:
      args.config_args["batch_size"] = 3500

    args.auxilary_loss_args["aux_strides"] = sample # We want to keep auxilary loss layer

    data = make_experiment_07_se_block(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
      devtrain_recog = True
      #test_construct = True
    )

def gradient_noise_tests():
  for gradn in [0.1, 0.05]:
    args = get_defaults()
    args.config_args["gradient_noise"] = gradn
    NAME = f"{BASE}+grad-noise-{gradn}"

    data = make_experiment_07_se_block(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
    )

def huge_conformer(): # Baslicly Xl conformer with se block

  args = get_defaults()
  NAME = f'{BASE}+XL'

  args.shared_network_args['model_dim'] = 1024
  args.sa_default_args['key_dim'] = 512
  args.sa_default_args['value_dim'] = 512

  data = make_experiment_07_se_block(
    args, 
    NAME,
    se_block_for_module = ["ff_mod"],
    devtrain_recog = True
  )

def huge_conformer_normlization(): # TODO; set good normalizations here
  pass

def no_chunking_seq_len():
  stats = [
    {"max_seq_len" : 1800},
    {"batch_size" : 4600},
    {"max_seq_len" : 4000, "batch_size" : 5400},
  ]

  for s in stats:
    # Experiment w/o chunking, but set max_seq len
    args = get_defaults()
    NAME = f"{BASE}+no-chunking-" + "-".join([f"{k.replace('_', '-')}={v}" for k, v in s.items()])

    args.config_args.pop("chunking")
    for x in s:
      args.config_args[x] = s[x]
    #args.config_args["max_seq"] = 200 #TODO: is that any good

    data = make_experiment_07_se_block(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
    )
  
def maybe_optimal_l2_drop():
  # From intial experiments ( very old baseline, the best was L2 = 0.0001, dropout = 0.03)
  args = get_defaults()
  NAME = f"{BASE}+reduced-l2=0.0001-drop=0.03"
  L2 = 0.0001
  drop = 0.03

  args.sa_default_args["sa_dropout"] = drop
  args.sa_default_args["sa_post_dropout"] = drop

  args.conv_default_args["conv_post_dropout"] = drop

  args.ff_default_args["ff_activation_dropout"] = drop
  args.ff_default_args["ff_post_dropout"] = drop

  data = make_experiment_08_se_constom_dims(
    args, 
    NAME,
    se_block_for_module = ["ff_mod"],
    devtrain_recog = True
  )


def conv_kernel_size():

  for x in [7, 17, 32, 65]:
    args = get_defaults()
    NAME = f"{BASE}+conv-kernel-size-{x}"
    args.conv_default_args["kernel_size"] = x

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
      devtrain_recog = True
    )

def attention_heads():
  for a in [4, 16, 32]:
    args = get_defaults()
    NAME = f"{BASE}+att-heads={a}"

    args.sa_default_args["num_heads"] = a

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
      devtrain_recog = True
    )


def half_ration():

  for x in [0.4, 0.6]:
    args = get_defaults()
    NAME = f"{BASE}+other-ff-ration-{x}"
    args.ff_default_args["ff_half_ratio"] = x

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
      devtrain_recog = True
    )

def stacking_window_size():
  size_win_map = [
    {
      "window_right" : 1,
      "window_left" : 2,
      "window_size" : 4
    },
    {
      "window_right" : 2,
      "window_left" : 2,
      "window_size" : 5
    },
    {
      "window_right" : 1,
      "window_left" : 3,
      "window_size" : 5
    },
    {
      "window_right" : 2,
      "window_left" : 3,
      "window_size" : 6
    }
  ]
  for _map in size_win_map:
    args = get_defaults()
    args.sampling_default_args
    for x in _map:
      args.sampling_default_args[x] = _map[x]

    NAME = f"{BASE}+stacking-window={_map['window_size']}-right={_map['window_right']}-left={_map['window_left']}"

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
    )



def get_l2_init(inital=0):
  return OrderedDict(
    ff_mod1 = {
      "_conv1" : inital, 
      "_conv2" : inital,
      "_out" : inital,
    },
    self_att = {
      "_att_att" : inital,
      "_att_lin" : inital,
      "_out" : inital
    },
    conv_mod = {
      "_conv_pointwise1" : inital,
      "_conv_depthwise" : inital,
      "_conv_pointwise2" : inital,
      "_conv_output" : inital
    },
    ff_mod2 = {
        "_conv1" : inital, 
        "_conv2" : inital,
        "_out" : inital
      })

def with_l2_TEST():
  # TODO experiment using L2, right now we're not using it at all
  # We apply weight decay [35] with a value of 0.01 to the transposed convolution layers ...
  # Here make some thest with weight decay, if that suceeds use it...

  # Positions to apply:
  # - ff mod conv1 (upscale), ff mod conv2, ff mod out
  # - attmod attention, attmod linear, att mod out
  # - conv mod pointwise 2, conv mod depthwise, conv mod pointwise 2, conv mod out 


  for x in [0.0, 0.0001]:
    args = get_defaults()
    NAME = f"{BASE}+l2-everywhere={x}-TEST"
    layer_l2 = get_l2_init(x)

    args.conformer_defaults["per_layer_l2"] = [layer_l2] * args.conformer_defaults["num_blocks"]

    make_experiment_09_se_and_l2(
      args,
      NAME,
      se_block_for_module = ["ff_mod"],
      #dummy_config = "l2.test.config",
      #test_construct = True
      )

def learning_rate_optimal_decay():
  for decay in [0.90, 0.95]:
    args = get_defaults()
    NAME = f"{BASE}+lr-decay-fact={decay}"

    learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=decay)
    args.config_args["learning_rates"] = learning_rates

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
    )

def learning_rate_optimal_warmups():
  for wup in [5, 15]:
    args = get_defaults()
    NAME = f"{BASE}+warmups={wup}"

    learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=wup, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
    args.config_args["learning_rates"] = learning_rates

    data = make_experiment_08_se_constom_dims(
      args, 
      NAME,
      se_block_for_module = ["ff_mod"],
    )


def l2_on_wide_layers():
  for l2 in [0.01, 0.02, 0.003, 0.001]:
    args = get_defaults()
    NAME = f"{BASE}+l2-only-wide-layers={l2}"
    layer_l2 = OrderedDict(
      ff_mod1 = {"_conv1" : l2},
      conv_mod = {"_depthwise": l2},
      self_att = {"_att_lin" : l2},
      ff_mod2 = {"_conv1" : l2}
    )

    args.conformer_defaults["per_layer_l2"] = [layer_l2] * args.conformer_defaults["num_blocks"]

    make_experiment_09_se_and_l2(
      args,
      NAME,
      se_block_for_module = ["ff_mod"],
      )

def l2_on_outputs():
  for l2 in [0.01, 0.02, 0.003, 0.001]:
    args = get_defaults()
    NAME = f"{BASE}+l2-only-outputs={l2}"
    layer_l2 = OrderedDict(
      ff_mod1 = {"_out" : l2},
      conv_mod = {"_output": l2},
      self_att = {"_out" : l2},
      ff_mod2 = {"_out" : l2}
    )

    args.conformer_defaults["per_layer_l2"] = [layer_l2] * args.conformer_defaults["num_blocks"]

    make_experiment_09_se_and_l2(
      args,
      NAME,
      se_block_for_module = ["ff_mod"],
      )


def devtrain_recog():
  num_blocks()
  att_dim()
  ff_dim()
  baseline()

def main():
  other_ff_se_version()
  se_block_conv_module()

  #conv_dim() TODO: some broken
  att_value_dim()

  maybe_optimal_l2_drop()

  no_chunking_seq_len()

  gradient_noise_tests()

  huge_conformer()

  with_l2_TEST()
  l2_on_outputs()
  l2_on_wide_layers()

  frame_stacking_time_down_factor()

  #stacking_window_size() TODO: broken
  half_ration()
  learning_rate_optimal_decay()
  learning_rate_optimal_warmups()

  conv_kernel_size()
  attention_heads()
