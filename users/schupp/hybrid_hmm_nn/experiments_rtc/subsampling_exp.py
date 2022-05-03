# TODO: package, make imports smaller
from typing import OrderedDict
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import setup_god as god
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_args_001_bigger
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_config_returnn_baseargs as experiment_config_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_baseline_00
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args import conformer_returnn_dict_network_generator

from sisyphus import gs
import copy

import inspect

OUTPUT_PATH = "conformer/subsampling/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

def default():
  config_args = copy.deepcopy(experiment_config_args.config_baseline_00)
  NAME = "sub-conformer-baseline-proposal+pos-enc"

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_002

  sa_args = copy.deepcopy(experiment_config_args.sa_default_args_00)
  sa_args.update(OrderedDict( # Args for positional encoding
    fixed_pos = False,
    clipping = 400,
  ))

  su_args = copy.deepcopy(experiment_config_args.sampling_default_args_00)

  god.create_experiment_world_001(
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_00,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = su_args,

      # Feed forward args, both the same by default
      ff1_func_args = experiment_config_args.ff_default_args_00,
      ff2_func_args = experiment_config_args.ff_default_args_00,

      # Self attention args
      conformer_self_att_func=make_self_att_mod_002, # Version with positional encoding
      sa_func_args = sa_args,

      # Conv mod args
      conv_func_args = experiment_config_args.conv_default_args_00,

      # Shared model args
      shared_model_args = experiment_config_args.shared_network_args_00,

      # Conformer args
      **experiment_config_args.conformer_default_args_00 ),
      returnn_train_post_config=experiment_config_args.returnn_train_post_config_00,
      returnn_rasr_args_defaults=experiment_config_args.returnn_rasr_args_defaults_00,

      #test_construction=True

  )

def default_and_se():
  config_args = copy.deepcopy(experiment_config_args.config_baseline_00)
  NAME = "sub-conformer-baseline-proposal+pos-enc+se_block-on-embedding"

  config_args["extra_name"] = NAME

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_002
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.subsampling_versions import make_subsampling_003, make_unsampling_003

  sa_args = copy.deepcopy(experiment_config_args.sa_default_args_00)
  sa_args.update(OrderedDict( # Args for positional encoding
    fixed_pos = False,
    clipping = 400,
  ))

  su_args = copy.deepcopy(experiment_config_args.sampling_default_args_00)

  god.create_experiment_world_001(
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_00,
    conformer_func_args=OrderedDict(
      # sampling args
      sampling_func_args = su_args,
      subsampling_func=make_subsampling_003,
      unsampling_func=make_unsampling_003,

      # Feed forward args, both the same by default
      ff1_func_args = experiment_config_args.ff_default_args_00,
      ff2_func_args = experiment_config_args.ff_default_args_00,

      # Self attention args
      conformer_self_att_func=make_self_att_mod_002, # Version with positional encoding
      sa_func_args = sa_args,

      # Conv mod args
      conv_func_args = experiment_config_args.conv_default_args_00,

      # Shared model args
      shared_model_args = experiment_config_args.shared_network_args_00,

      # Conformer args
      **experiment_config_args.conformer_default_args_00 ),
      returnn_train_post_config=experiment_config_args.returnn_train_post_config_00,
      returnn_rasr_args_defaults=experiment_config_args.returnn_rasr_args_defaults_00,

      #test_construction=True
      write_dummpy_config="./test.config.se"
  )


def conv2():
  config_args = copy.deepcopy(experiment_config_args.config_baseline_00)
  NAME = "conformer-baseline-proposal+pos-enc+subsampling-2conv"

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.sa_mod_versions import make_self_att_mod_002
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.subsampling_versions import make_subsampling_002, make_unsampling_002

  sa_args = copy.deepcopy(experiment_config_args.sa_default_args_00)
  sa_args.update(OrderedDict( # Args for positional encoding
    fixed_pos = False,
    clipping = 400,
  ))

  su_args = copy.deepcopy(experiment_config_args.sampling_default_args_00)

  su_args.update(OrderedDict(
    conv0p_pools = [1, 1], # here lets 'only' pool over the features
    conv1p_pools = [2, 2], # Also pool over time i.e.: time-down=2
    time_reduction = 2 # For the 'unsampling' layer
  ))

  god.create_experiment_world_001(
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_00,
    conformer_func_args=OrderedDict(
      # sampling args
      subsampling_func=make_subsampling_002,
      unsampling_func=make_unsampling_002,
      sampling_func_args = su_args,

      # Feed forward args, both the same by default
      ff1_func_args = experiment_config_args.ff_default_args_00,
      ff2_func_args = experiment_config_args.ff_default_args_00,

      # Self attention args
      conformer_self_att_func=make_self_att_mod_002, # Version with positional encoding
      sa_func_args = sa_args,

      # Conv mod args
      conv_func_args = experiment_config_args.conv_default_args_00,

      # Shared model args
      shared_model_args = experiment_config_args.shared_network_args_00,

      # Conformer args
      **experiment_config_args.conformer_default_args_00 ),
      returnn_train_post_config=experiment_config_args.returnn_train_post_config_00,
      returnn_rasr_args_defaults=experiment_config_args.returnn_rasr_args_defaults_00,

      #test_construction=True Works :)
  )

  
def main():
  default()
  default_and_se()
  conv2()
