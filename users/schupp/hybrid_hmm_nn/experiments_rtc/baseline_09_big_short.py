# This is 'big-short' V 09
# + aux as 4 6 8

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

OUTPUT_PATH = "conformer/baseline_09_big_short/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

BASE = "baseline_09_big_short"

def make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
        min_lr_ratio=1/50, decay_factor=0.99):

    num_lr = int(math.log(min_lr_ratio, decay_factor))
    return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                    [start] * constant_subepoch + \
                    list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                    [min_lr_ratio * start]

def get_deafults_baseline_04(): # Old defaults from previous baseline
  args = original_args_big_baseline_00()
  args.config_args["extra_tag_tim_setup"] = 'baseline-big-short-09'
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
  args.config_args["extra_tag_tim_setup"] = 'baseline-big-short-09'

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


def make_experiment_12_se_l2_sk_sd_v3_GN(
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
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_09_sd_se_l2_sk_sd_v3_gn,
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


def make_experiment_12_se_l2_sk_sd_v3_GN_ff_gating(
  args, 
  NAME,
  aux_loss_layers = [6],
  se_block_for_module = [],
  use_ff_gating = False,
  dummy_config = None,
  test_construct = False,
  ):

  # Theses are the versions that use se blocks
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_006_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_005_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.ff_mod_versions import make_ff_mod_004_se_block, make_ff_mod_004_se_block_gating

  args_map = {
    "ff_mod" : args.ff_default_args,
    "att_mod" : args.sa_default_args,
    "conv_mod" : args.conv_default_args,
  }

  for k in se_block_for_module:
    args_map[k]["use_se_block"] = True # This flag is False per default for all conformer blocks ( i.e.: default is regular module version )

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conformer_returnn_dict_network_generator import make_conformer_06_sd_se_l2_dynamic

  ff_func = make_ff_mod_004_se_block
  if use_ff_gating:
    ff_func = make_ff_mod_004_se_block_gating

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_09_sd_se_l2_sk_sd_v3_gn,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default

      conformer_ff1_func=ff_func,
      conformer_ff2_func=ff_func,

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


def make_experiment_12_se_l2_sk_sd_v3_GN_ffg_sample_act(
  args, 
  NAME,
  aux_loss_layers = [6],
  se_block_for_module = [],
  use_ff_gating = False,
  dummy_config = None,
  test_construct = False,
  ):

  # Theses are the versions that use se blocks
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_006_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_005_se_block
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.ff_mod_versions import make_ff_mod_004_se_block, make_ff_mod_004_se_block_gating
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.subsampling_versions import make_subsampling_005_fstack_dyn_act

  args_map = {
    "ff_mod" : args.ff_default_args,
    "att_mod" : args.sa_default_args,
    "conv_mod" : args.conv_default_args,
  }

  for k in se_block_for_module:
    args_map[k]["use_se_block"] = True # This flag is False per default for all conformer blocks ( i.e.: default is regular module version )

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conformer_returnn_dict_network_generator import make_conformer_06_sd_se_l2_dynamic

  ff_func = make_ff_mod_004_se_block
  if use_ff_gating:
    ff_func = make_ff_mod_004_se_block_gating

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_09_sd_se_l2_sk_sd_v3_gn,
    conformer_func_args=OrderedDict(
      # sampling args

      subsampling_func=make_subsampling_005_fstack_dyn_act, # This allowes to overwrite `sampling_activation`
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default

      conformer_ff1_func=ff_func,
      conformer_ff2_func=ff_func,

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

# ------------------------- baseline: 'big-short-09' -----------------------

def baseline():
  # .... this is  just aux-4-6-8 for baseline-08
  args = get_defaults()
  NAME = f"{BASE}"

  data = make_experiment_12_se_l2_sk_sd_v3_GN(
    args, 
    NAME,
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
    test_construct = True,
  )


def groupnorm_groups():
  for g in [16, 32, 64, 128, 256]:
    args = get_defaults()
    NAME = f"{BASE}+gn-everywhere-g={g}"

    args.conformer_defaults["apply_groupnorm"] = {
      i : { mod : {
          "groups" : g,
          "epsilon" : 1e-5
        } for mod in ["conv_mod", "ff_mod1", "ff_mod2", "self_att"]
      } for i in range(1, 16 + 1)
    }
    print(f'TBS: applying: {args.conformer_defaults["apply_groupnorm"]}')

    data = make_experiment_12_se_l2_sk_sd_v3_GN(
      args, 
      NAME,
      #dummy_config = "gn.test.config",
    #  test_construct = True,
      aux_loss_layers = [4, 8, 12],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )

def groupnorm_only_conv():
  for g in [64]:
    args = get_defaults()
    NAME = f"{BASE}+gn-only-conv-g={g}"

    args.conformer_defaults["apply_groupnorm"] = {
      i : { mod : {
          "groups" : g,
          "epsilon" : 1e-5
        } for mod in ["conv_mod"]
      } for i in range(1, 16 + 1)
    }
    print(f'TBS: applying: {args.conformer_defaults["apply_groupnorm"]}')

    data = make_experiment_12_se_l2_sk_sd_v3_GN(
      args, 
      NAME,
      #dummy_config = "gn.test.config",
      aux_loss_layers = [4, 8, 12],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )

def groupnorm_only_ff():
  for g in [64]:
    args = get_defaults()
    NAME = f"{BASE}+gn-only-ff-g={g}"

    args.conformer_defaults["apply_groupnorm"] = {
      i : { mod : {
          "groups" : g,
          "epsilon" : 1e-5
        } for mod in ["ff_mod1", "ff_mod2"]
      } for i in range(1, 16 + 1)
    }
    print(f'TBS: applying: {args.conformer_defaults["apply_groupnorm"]}')

    data = make_experiment_12_se_l2_sk_sd_v3_GN(
      args, 
      NAME,
      #dummy_config = "gn.test.config",
      aux_loss_layers = [4, 8, 12],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )


def groupnorm_only_att():
  for g in [64]:
    args = get_defaults()
    NAME = f"{BASE}+gn-only-att-g={g}"

    args.conformer_defaults["apply_groupnorm"] = {
      i : { mod : {
          "groups" : g,
          "epsilon" : 1e-5
        } for mod in ["self_att"]
      } for i in range(1, 16 + 1)
    }
    print(f'TBS: applying: {args.conformer_defaults["apply_groupnorm"]}')

    data = make_experiment_12_se_l2_sk_sd_v3_GN(
      args, 
      NAME,
      #dummy_config = "gn.test.config",
      aux_loss_layers = [4, 8, 12],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )

def groupnorm_in_conv_mod():
  g = 64
  args = get_defaults()
  NAME = f"{BASE}+conv-mod-gn={g}"

  args.conformer_defaults["convolution_groupnorm"] = {
      i: { "groups" : g,
        "epsilon" : 1e-5
      } for i in range(1, 16 + 1)
  }

  data = make_experiment_12_se_l2_sk_sd_v3_GN(
    args, 
    NAME,
    #dummy_config = "gn.test.config",
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

def decrease_conv_kernel():
  args = get_defaults()
  NAME = f"{BASE}+conv-kernel=17"

  args.conv_default_args["kernel_size"] = 17

  data = make_experiment_12_se_l2_sk_sd_v3_GN(
    args, 
    NAME,
    #dummy_config = "gn.test.config",
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )


def ff_gating():
  args = get_defaults()
  NAME = f"{BASE}+ff-gate"
  # TODO ...
  make_experiment_12_se_l2_sk_sd_v3_GN_ff_gating(
    args,
    NAME,
    use_ff_gating = True,
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

def ff_act():
  for act in ["gelu", "relu"]:
    args = get_defaults()
    NAME = f"{BASE}+ff-act={act}"

    args.ff_default_args['ff_activation'] = act

    make_experiment_12_se_l2_sk_sd_v3_GN(
      args,
      NAME,
      aux_loss_layers = [4, 8, 12],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )

def subsampling_act():
  for act in ["gelu", "swish"]:
    args = get_defaults()
    
    NAME = f"{BASE}+subsampling-act={act}"
    args.sampling_default_args["sampling_activation"] = act

    make_experiment_12_se_l2_sk_sd_v3_GN_ffg_sample_act(
      args,
      NAME,
      aux_loss_layers = [4, 8, 12],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )

def huge_conformer():
  args = get_defaults()

  NAME = f'{BASE}+XL'

  args.config_args['batch_size'] = 4600
  args.shared_network_args['model_dim'] = 1024
  args.sa_default_args['key_dim'] = 512
  args.sa_default_args['value_dim'] = 512

  make_experiment_12_se_l2_sk_sd_v3_GN(
    args,
    NAME,
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

def chunks():
  for chunk in ["600:300", "200:100"]:
    args = get_defaults()
    NAME = f"{BASE}+chunk-{chunk}"

    args.config_args['chunking'] = chunk

    make_experiment_12_se_l2_sk_sd_v3_GN(
      args,
      NAME,
      aux_loss_layers = [4, 8, 12],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )
  
def no_specaug():
  args = get_defaults()
  NAME = f"{BASE}+no-specaug"
  args.conformer_defaults["no_specaug"] = True

  make_experiment_12_se_l2_sk_sd_v3_GN_ffg_sample_act(
    args,
    NAME,
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

def aux_and_skip():
  args = get_defaults()
  NAME = f"{BASE}+aux-and-skip"

  def even_space_skip(step, blocks): # Start with 1
    i = 1
    l = []
    while step*i <= blocks:
      l.append(step*i)
      i +=1
    return l

  skip = even_space_skip(2, args.conformer_defaults["num_blocks"])
  args.conformer_defaults['skip_con_after_layer'] = skip
  make_experiment_12_se_l2_sk_sd_v3_GN_ffg_sample_act(
    args,
    NAME,
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )


def main():
  groupnorm_groups()
  groupnorm_only_ff()
  groupnorm_only_conv()
  groupnorm_only_att()

  groupnorm_in_conv_mod()

  decrease_conv_kernel()
  huge_conformer()

  ff_gating()
  ff_act()
  subsampling_act()

  baseline()
  chunks()

  aux_and_skip()

  no_specaug()