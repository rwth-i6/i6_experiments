# TODO: package, make imports smaller
from atexit import register
from typing import OrderedDict
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import setup_god as god
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_config_returnn_baseargs as experiment_config_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conformer_args_002_smaller_baseline import original_args_small_baseline_00
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_returnn_dict_network_generator

from sisyphus import gs
import copy
import numpy
import math

import inspect

OUTPUT_PATH = "conformer/baseline_02_small/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

def get_defaults():
  args = copy.deepcopy(original_args_small_baseline_00)
  return args

def make_experiment(
  args, 
  NAME,
  aux_loss_layers = [6]
  ):
  experiment_data = god.create_experiment_world_002( 
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

      extra_recog_epochs=[5], # Basicly early test epoch

      test_construction=False,
  )


# + stoch_depth for full conformer modules
def make_experiment_02_stoch_depth( # TODO (WIP)
  args, 
  NAME,
  aux_loss_layers = [6],
  sd_args = None
  ):

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.ff_mod_versions import make_ff_mod_003_sd02, make_ff_mod_001
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_004_stoch_depth, make_self_att_mod_003_rel_pos

  ffmod_func1 = make_ff_mod_001
  ffmod_func2 = make_ff_mod_001

  ff1_args = copy.deepcopy(args.ff_default_args)
  ff2_args = copy.deepcopy(args.ff_default_args)

  if "ffmod1" in sd_args:
    ffmod_func1 = make_ff_mod_003_sd02
    ff1_args.update(sd_args["ffmod1"])

  if "ffmod2" in sd_args:
    ffmod_func2 = make_ff_mod_003_sd02
    ff2_args.update(sd_args["ffmod2"])

  att_func = make_self_att_mod_003_rel_pos
  att_args = copy.deepcopy(args.sa_default_args)

  if "self_att" in sd_args:
    att_func = make_self_att_mod_004_stoch_depth
    att_args.update(sd_args["self_att"])

  experiment_data = god.create_experiment_world_002( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_03_feature_stacking_auxilary_loss,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = args.sampling_default_args,

      conformer_ff1_func = ffmod_func1,
      conformer_ff2_func = ffmod_func2,
      # Feed forward args, both the same by default
      ff1_func_args = ff1_args,
      ff2_func_args = ff2_args,

      # Self attention args
      conformer_self_att_func=att_func,
      sa_func_args = att_args, 

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

      extra_recog_epochs=[5], # Basicly early test epoch

      test_construction=False,
  )

def small_baseline():
  # Has all that bigger conformer has, exect all smaller dimension

  # World 02 extras:
  # + devtrain
  # + shuffle data, then order time len data

  args = get_defaults()
  NAME = "baseline_02_small"

  make_experiment(args, NAME)

# + data not ordered by time
def no_order_dataset():

  args = get_defaults()
  NAME = "baseline_02_small+no-dateset-order"
  print(NAME)

  args.returnn_rasr_args_defaults["shuffle_data"] = False
  args.config_args["extra_name"] = "not-shuffeled"

  make_experiment(args, NAME)

# + no auxilary loss
def no_aux_loss():

  args = get_defaults()
  NAME = "baseline_02_small+no-aux-loss"

  make_experiment(args, NAME, aux_loss_layers=[])

def ff_stoch_depth():
  args = get_defaults()
  NAME = "baseline_02_small+ff-mod-sd"

  make_experiment_02_stoch_depth(
    args,
    NAME,
    sd_args = {
      "ffmod1": {"survival_prob" : 0.5},
      "ffmod2": {"survival_prob" : 0.5},
    }
  )

def att_stoch_depth():
  args = get_defaults()
  NAME = "baseline_02_small+att-mod-sd"
  make_experiment_02_stoch_depth(
    args,
    NAME,
    sd_args = {
      "self_att": {"survival_prob" : 0.5},
    }
  )

# TODO
def conv_mod_defaults(): # Overwrites the defaults that canged since behavior_version =12
  pass

# + wo frame stacking (-> i.e.: No time downsampling )
def no_frame_stacking(): # TODO
  pass

# + wo positional encoding
def no_pos_encoding(): # TODO
  pass


def main():
  small_baseline()

  no_order_dataset()
  no_aux_loss()

  ff_stoch_depth()
  att_stoch_depth()