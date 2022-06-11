# This is 'big-short' V 08
# + 16 blocks
# + se conv mod
# + reduced drop=0.03

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

OUTPUT_PATH = "conformer/baseline_08_big_short/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

BASE = "baseline_08_big_short"

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

  args.conformer_defaults["num_blocks"] = 16

  # Reduced dropout
  drop = 0.03
  args.sa_default_args["sa_dropout"] = drop
  args.sa_default_args["sa_post_dropout"] = drop
  args.conv_default_args["conv_post_dropout"] = drop
  args.ff_default_args["ff_activation_dropout"] = drop
  args.ff_default_args["ff_post_dropout"] = drop

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
  test_construct = False
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


def make_experiment_10_se_l2_skip(
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
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_07_sd_se_l2_skip,
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


def make_experiment_11_se_l2_sk_sd_v3(
  args, 
  NAME,
  aux_loss_layers = [6],
  se_block_for_module = [],
  dummy_config = None,
  extra_config_create_args=None,
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
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_08_sd_se_l2_sk_sd_v3,
    extra_returnn_net_creation_args=extra_config_create_args,
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


# ------------------------- baseline: 'big-short-08' -----------------------


def sd_ff_depth_scale_multiple_v2():
  for prob in [0.0, 0.1, 0.5, 1.0]:
  
    args = get_defaults()
    NAME = f'{BASE}+stoch-depth-v4.0-ff-mod+depth-scale-survival-prob-v1-p={prob}'
    extra_config_create_args = None

    if prob == 0.5:
      extra_config_create_args = {"recoursion_depth" : 9000}

    import numpy
    space = numpy.linspace(1.0, 0.5, num=24)

    def surv_prob_by_layer(l, L=12.0, p=0.2):
      return 1.0 - ((l/L) * (1.0 - p))

    sd_args = {
      i : {
        "ff_mod1" : surv_prob_by_layer(i, p=prob),
        "ff_mod2" : surv_prob_by_layer(i, p=prob),
      } for i in range(1, 12 + 1)
    }

    args.conformer_defaults.update(OrderedDict(
      apply_stochastic_depth = sd_args,
      multipy_by_surivial_prob_ineval = True
    ))

    make_experiment_11_se_l2_sk_sd_v3(
      args,
      NAME,
      extra_config_create_args = extra_config_create_args,
      aux_loss_layers = [8],
      se_block_for_module = ["ff_mod", "conv_mod"],
      #test_construct = True,
    )


def sd_conv_scale_multiple_v2():
  for prob in [0.0, 0.1, 0.5, 1.0]:
  
    args = get_defaults()
    NAME = f'{BASE}+stoch-depth-v4.0-ff-mod+depth-scale-survival-prob-v1-p={prob}'

    import numpy
    space = numpy.linspace(1.0, 0.5, num=12)

    def surv_prob_by_layer(l, L=12.0, p=0.2):
      return 1.0 - ((l/L) * (1.0 - p))

    sd_args = {
      i : {
        "conv_mod" : surv_prob_by_layer(i, p=prob),
      } for i in range(1, 12 + 1)
    }

    args.conformer_defaults.update(OrderedDict(
      apply_stochastic_depth = sd_args,
      multipy_by_surivial_prob_ineval = True
    ))

    make_experiment_11_se_l2_sk_sd_v3(
      args,
      NAME,
      aux_loss_layers = [8],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )


def baseline():
  args = get_defaults()
  NAME = f"{BASE}"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    aux_loss_layers = [8],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

def extra_aux():
  args = get_defaults()
  NAME = f"{BASE}+aux-6-12"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    aux_loss_layers = [6,12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

def extra_aux2():
  args = get_defaults()
  NAME = f"{BASE}+aux-4-8-12"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

def embed_l2_drop():
  l2s = [0.0001, 0.0, 0.0, 0.0]
  ds =  [0.003   , 0.1, 0.2, 0.05]
  for l2, d in zip(l2s, ds):
    args = get_defaults()
    NAME = f"{BASE}-embed-l2={l2}-drop={d}"
    args.sampling_default_args["embed_l2"] = l2
    args.sampling_default_args["embed_dropout"] = d

    data = make_experiment_07_se_block(
      args, 
      NAME,
      aux_loss_layers = [8],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )

def huge_conformer():
  args = get_defaults()
  NAME = f"{BASE}+XL"

  args.config_args["batch_size"] = 5000

  args.shared_network_args['model_dim'] = 1024
  args.sa_default_args['key_dim'] = 512
  args.sa_default_args['value_dim'] = 512

  data = make_experiment_07_se_block(
    args, 
    NAME,
    aux_loss_layers = [8],
    se_block_for_module = ["ff_mod", "conv_mod"],
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

def decoupled_weight_decay():
  for fact in [None,2e-05, 9e-06]:
    args = get_defaults()
    l2 = 0.002 # Default L2, just to see if we can get an improvement with decoupled...
    layer_l2 = OrderedDict(
      ff_mod1 = {"_out" : l2},
      conv_mod = {"_output": l2},
      self_att = {"_out" : l2},
      ff_mod2 = {"_out" : l2}
    )

    NAME = f"{BASE}-l2={l2}-decoupled={fact}"
    args.conformer_defaults["per_layer_l2"] = [layer_l2] * args.conformer_defaults["num_blocks"]
    if fact: # Base case is completely without decoupled
      args.config_args["decouple_constraints"] = True
      args.config_args["decouple_constraints_factor"] = fact

    make_experiment_09_se_and_l2(
      args,
      NAME,
        aux_loss_layers = [8],
        se_block_for_module = ["ff_mod", "conv_mod"],
      )

def even_space_skip(step, blocks): # Start with 1
  i = 1
  l = []
  while step*i <= blocks:
    l.append(step*i)
    i +=1
  return l

def skip_connections():

  def even_space_skip(step, blocks): # Start with 1
    i = 1
    l = []
    while step*i <= blocks:
      l.append(step*i)
      i +=1
    return l

  args = get_defaults()
  skips = [
    even_space_skip(1, args.conformer_defaults["num_blocks"]),
    even_space_skip(2, args.conformer_defaults["num_blocks"]),
    even_space_skip(3, args.conformer_defaults["num_blocks"]),
    even_space_skip(4, args.conformer_defaults["num_blocks"]),
    even_space_skip(5, args.conformer_defaults["num_blocks"])
  ]

  for skip in skips:
    args = get_defaults()
    NAME = f"{BASE}+skip-at-" + "-".join([str(s) for s in skip])

    args.conformer_defaults['skip_con_after_layer'] = skip

    make_experiment_10_se_l2_skip(
      args,
      NAME,
        aux_loss_layers = [8],
        se_block_for_module = ["ff_mod", "conv_mod"],
      )

def se_dim():
  for d in [26, 32, 48, 64, 128]:
    args = get_defaults()
    NAME = f"{BASE}+se-dim={d}"

    args.conv_default_args["se_version"] = 1
    args.conv_default_args["se_dim"] = d

    make_experiment_10_se_l2_skip(
      args,
      NAME,
        aux_loss_layers = [8],
        se_block_for_module = ["ff_mod", "conv_mod"],
      )

def se_act():
  for d in ["gelu", "relu"]:
    args = get_defaults()
    NAME = f"{BASE}+se-act={d}"

    args.conv_default_args["se_version"] = 1
    args.conv_default_args["se_act"] = d

    make_experiment_10_se_l2_skip(
      args,
      NAME,
        aux_loss_layers = [8],
        se_block_for_module = ["ff_mod", "conv_mod"],
      )


def main():
  baseline()
  embed_l2_drop()

  huge_conformer()
  decoupled_weight_decay()
  skip_connections()

  extra_aux()
  extra_aux2()

  se_dim()
  se_act()

  sd_ff_depth_scale_multiple_v2()
  sd_conv_scale_multiple_v2()