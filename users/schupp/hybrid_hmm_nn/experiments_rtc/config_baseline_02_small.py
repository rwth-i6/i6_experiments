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
      write_dummpy_config="./test.config"
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