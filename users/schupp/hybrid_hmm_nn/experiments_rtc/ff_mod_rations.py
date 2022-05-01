# TODO: package, make imports smaller
from typing import OrderedDict
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import setup_god as god
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_config_returnn_baseargs as experiment_config_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_returnn_dict_network_generator

from sisyphus import gs
import copy

import inspect

OUTPUT_PATH = "conformer/ff_mod_rations/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

def main():
  config_args = copy.deepcopy(experiment_config_args.config_baseline_00)
  NAME = "baseline+ffmod-rations-ffmods-only-first-4l_then-only-fist-ffmod-4l_then-no-ffmod-4l"

  god.create_experiment_world_001(
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_01, # Version that allowes to dynamicly swith ffmod's off
    conformer_func_args=OrderedDict(
      ff_modules = [
        (True, True),
        (True, True),
        (True, True),
        (True, True),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (False, False),
        (False, False),
        (False, False),
        (False, False),
      ],
      # sampling args
      sampling_func_args = experiment_config_args.sampling_default_args_00,

      # Feed forward args, both the same by default
      ff1_func_args = experiment_config_args.ff_default_args_00,
      ff2_func_args = experiment_config_args.ff_default_args_00,

      # Self attention args
      sa_func_args = experiment_config_args.sa_default_args_00,

      # Conv mod args
      conv_func_args = experiment_config_args.conv_default_args_00,

      # Shared model args
      shared_model_args = experiment_config_args.shared_network_args_00,

      # Conformer args
      **experiment_config_args.conformer_default_args_00 ),
      returnn_train_post_config=experiment_config_args.returnn_train_post_config_00,
      returnn_rasr_args_defaults=experiment_config_args.returnn_rasr_args_defaults_00
  )

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_002

  config_args = copy.deepcopy(experiment_config_args.config_baseline_00)
  NAME = "baseline+ffmod-rations-ffmods-only-first-4l_then-only-fist-ffmod-4l_then-no-ffmod-4l++pos-enc"


  sa_args = copy.deepcopy(experiment_config_args.sa_default_args_00)
  sa_args.update(OrderedDict(
    fixed_pos = False,
    clipping = 400,
  ))

  god.create_experiment_world_001(
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_01, # Version that allowes to dynamicly swith ffmod's off
    conformer_func_args=OrderedDict(
      ff_modules = [
        (True, True),
        (True, True),
        (True, True),
        (True, True),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (False, False),
        (False, False),
        (False, False),
        (False, False),
      ],
      # sampling args
      sampling_func_args = experiment_config_args.sampling_default_args_00,

      # Feed forward args, both the same by default
      ff1_func_args = experiment_config_args.ff_default_args_00,
      ff2_func_args = experiment_config_args.ff_default_args_00,

      # Self attention args
      conformer_self_att_func=make_self_att_mod_002,
      sa_func_args = sa_args,

      # Conv mod args
      conv_func_args = experiment_config_args.conv_default_args_00,

      # Shared model args
      shared_model_args = experiment_config_args.shared_network_args_00,

      # Conformer args
      **experiment_config_args.conformer_default_args_00 ),
      returnn_train_post_config=experiment_config_args.returnn_train_post_config_00,
      returnn_rasr_args_defaults=experiment_config_args.returnn_rasr_args_defaults_00
  )