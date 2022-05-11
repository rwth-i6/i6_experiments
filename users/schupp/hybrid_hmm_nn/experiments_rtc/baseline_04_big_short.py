# This is 'big-short'
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

OUTPUT_PATH = "conformer/baseline_03_big/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

BASE = "baseline_03_big_short"

def get_defaults():
  args = copy.deepcopy(original_args_big_baseline_00)
  return args

def make_experiment_03_rqmt(
  args, 
  NAME,
  aux_loss_layers = [6],
  test_construct = False
  ):

  experiment_data = god.create_experiment_world_003( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_03_feature_stacking_auxilary_loss,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default
      ff1_func_args = args.ff_default_args,
      ff2_func_args = args.ff_default_args,

      # Self attention args
      sa_func_args = args.sa_default_args,

      # Conv mod args
      conv_func_args = args.conv_default_args,

      # Shared model args
      shared_model_args = args.shared_network_args,

      auxilary_at_layer = aux_loss_layers,
      auxilary_loss_args = args.auxilary_loss_args,

      # Conformer args
      **args.conformer_defaults ),
      returnn_train_post_config=args.returnn_train_post_config,
      returnn_rasr_args_defaults=args.returnn_rasr_args_defaults,

      test_construction=test_construct,
  )


def make_experiment_04_seq_orders(
  args, 
  NAME,
  aux_loss_layers = [6],
  test_construct = False
  ):

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_03_feature_stacking_auxilary_loss,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default
      ff1_func_args = args.ff_default_args,
      ff2_func_args = args.ff_default_args,

      # Self attention args
      sa_func_args = args.sa_default_args,

      # Conv mod args
      conv_func_args = args.conv_default_args,

      # Shared model args
      shared_model_args = args.shared_network_args,

      auxilary_at_layer = aux_loss_layers,
      auxilary_loss_args = args.auxilary_loss_args,

      # Conformer args
      **args.conformer_defaults ),
      returnn_train_post_config=args.returnn_train_post_config,
      returnn_rasr_args_defaults=args.returnn_rasr_args_defaults,

      test_construction=test_construct,
  )


def make_experiment_05_batchnorm(
  args, 
  NAME,
  aux_loss_layers = [6],
  use_old_bn_defaults = False,
  overwrite_bn_settings = None,
  test_construct = False
  ):

  if not use_old_bn_defaults and not overwrite_bn_settings:
    args.conv_default_args["batch_norm_settings"] = { # If this is not set it would use old defaults
      "masked_time" : True # This has to be set because of behavior_version = 12
    } 

  if overwrite_bn_settings:
    args.conv_default_args["batch_norm_settings"] = overwrite_bn_settings
  # Old defaults
  #momentum = 0.1,
  #update_sample_only_in_training = False,
  #delay_sample_update = False,
  #param_version = 0,
  #masked_time = True,
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_004_old_defaults_batchnorm_or_dynamic
  conv_func = make_conv_mod_004_old_defaults_batchnorm_or_dynamic


  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_03_feature_stacking_auxilary_loss,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      # Feed forward args, both the same by default
      ff1_func_args = args.ff_default_args,
      ff2_func_args = args.ff_default_args,

      # Self attention args
      sa_func_args = args.sa_default_args,

      # Conv mod args
      conformer_self_conv_func = conv_func, # This is used to change batch norm settings
      conv_func_args = args.conv_default_args,

      # Shared model args
      shared_model_args = args.shared_network_args,

      auxilary_at_layer = aux_loss_layers,
      auxilary_loss_args = args.auxilary_loss_args,

      # Conformer args
      **args.conformer_defaults ),
      returnn_train_post_config=args.returnn_train_post_config,
      returnn_rasr_args_defaults=args.returnn_rasr_args_defaults,

      test_construction=test_construct,
  )

# ------------------------- baseline: 'big-short' -----------------------

def baseline_big_short():
  args = get_defaults()
  NAME = "baseline_03_big_short" # BASE

  make_experiment_03_rqmt(args, NAME)


# ---------------------------- ablation of best model elements -----------------------

def no_aux_loss():
  args = get_defaults()
  NAME = f"{BASE}+no-aux"

  make_experiment_03_rqmt(args, NAME, aux_loss_layers=[])

def no_frame_stacking(): #TODO:
  NAME = f"{BASE}+no-frame-stacking"
  pass

# ------------------------------ learning rate ------------------------------

def lr_shorter(): # TODO:
  pass

def lr_newbob(): # TODO:
  pass

# -------------------------------- batch norm -------------------------------

# + batchnorm instead of layer-norm
# This uses the new returnn defaults
def batchnorm_no_ln():
  args = get_defaults()
  NAME = f"{BASE}+batchnorm"

  make_experiment_05_batchnorm(args, NAME) 


# + old batchnorm defaults from behavior_version = 0
def batchnorm_old_defaults():
  args = get_defaults()
  NAME = f"{BASE}+batchnorm-old-defaults"

  # momentum = 0.1,
  # update_sample_only_in_training = False,
  # delay_sample_update = False,
  # param_version = 0,
  # masked_time = True,

  make_experiment_05_batchnorm(args, NAME, use_old_bn_defaults=True )


# ----------------------------- chunk/sequnce order ---------------------------

def seq_no_order_no_shuffle():
  args = get_defaults()
  NAME = f"{BASE}+no-seq-order+no-shuffle"

  params = OrderedDict(
          segment_order_shuffle = False,
          segment_order_sort_by_time_length = False,
          segment_order_sort_by_time_length_chunk_size = -1,
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  make_experiment_04_seq_orders(args, NAME)

def seq_only_shuffle():
  args = get_defaults()
  NAME = f"{BASE}+only-shuffle"

  params = OrderedDict(
          segment_order_shuffle = True,
          segment_order_sort_by_time_length = False, # Already false per default, but lets explicity overwrite it
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  make_experiment_04_seq_orders(args, NAME)

def seq_order_chunk_1000():
  args = get_defaults()
  NAME = f"{BASE}+shuffle+order-chunks=1000"

  params = OrderedDict(
          segment_order_shuffle = True,
          segment_order_sort_by_time_length = True,
          segment_order_sort_by_time_length_chunk_size = 1000,
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  make_experiment_04_seq_orders(args, NAME)

# -------------------------- stochastic depth -------------------------------


def main():
  # Baseline
  baseline_big_short()

  # Ablation
  no_aux_loss()

  # chunk/sequnce order
  seq_no_order_no_shuffle()
  seq_only_shuffle()
  seq_order_chunk_1000()