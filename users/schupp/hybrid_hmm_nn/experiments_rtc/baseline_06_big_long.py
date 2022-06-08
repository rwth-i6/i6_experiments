# This is 'big-long'
# Train for only 15 full epochs i.e.: 600 sub epochs 40 split
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

OUTPUT_PATH = "conformer/baseline_06_big_long/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

BASE = "baseline_06_big_long"

def make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
        min_lr_ratio=1/50, decay_factor=0.99):

    num_lr = int(math.log(min_lr_ratio, decay_factor))
    return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                    [start] * constant_subepoch + \
                    list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                    [min_lr_ratio * start]

def get_deafults_baseline_04(): # Old defaults from previous baseline
  args = original_args_big_baseline_00()

  args.returnn_rasr_args_defaults["num_epochs"] = 600
  args.config_args["extra_tag_tim_setup"] = 'baseline-big-long'
  del args.returnn_rasr_args_defaults["shuffle_data"] # Not needed cause set by default now
  args.returnn_train_post_config["cleanup_old_models"]["keep"] = [50, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600]

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
  args.config_args["extra_tag_tim_setup"] = 'baseline-big-long'

  args.returnn_rasr_args_defaults["num_epochs"] = 600
  del args.returnn_rasr_args_defaults["shuffle_data"] # Not needed cause set by default now
  args.returnn_train_post_config["cleanup_old_models"]["keep"] = [50, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600]

  params = OrderedDict(
          segment_order_shuffle = True,
          segment_order_sort_by_time_length = False, # Already false per default, but lets explicity overwrite it
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
  args.config_args["learning_rates"] = learning_rates

  # and shorter learning rate
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

  experiment_data = god.create_experiment_world_004( # New world that allowes adapting sequence order
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



# + allowes to use: apply_stochastic_depth
# e.g.:
# apply_stochastic_depth = {1 : { "ff_mod1" : 0.5 }} 
# -> would add stochastic depth to only the first block and only the frist ff module
def make_experiment_06_stoch_depth(
  args, 
  NAME,
  aux_loss_layers = [6],
  dummy_config = None,
  test_construct = False
  ):


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

      write_dummpy_config=dummy_config,
      test_construction=test_construct,
  )

  return experiment_data


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


def make_experiment_08_groupnorm(
  args, 
  NAME,
  aux_loss_layers = [6],
  overwrite_bn_settings = None,
  test_construct = False
  ):

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_007_group_norm
  conv_func = make_conv_mod_007_group_norm

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


def make_experiment_09_sampling_act(
  args, 
  NAME,
  aux_loss_layers = [6],
  test_construct = False
  ):

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.subsampling_versions import make_subsampling_005_fstack_dyn_act

  experiment_data = god.create_experiment_world_004( # New world that allowes adapting sequence order
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_03_feature_stacking_auxilary_loss,
    conformer_func_args=OrderedDict(

      subsampling_func=make_subsampling_005_fstack_dyn_act, # This allowes to overwrite `sampling_activation`
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


# + uses strided convolution for downsapling !not frame stacking!
def make_experiment_10_no_feature_stacking(
  args, 
  NAME,
  aux_loss_layers = [6],
  test_construct = False
  ):

  args.sampling_default_args["time_reduction"] = 3 # !! required, as feature statcking also did time down 3
  # The following are not needed any more: ( only required for frame stacking)
  del args.sampling_default_args["unsampling_strides"]
  del args.sampling_default_args["stacking_stride"]
  del args.sampling_default_args["window_size"]
  del args.sampling_default_args["window_right"]
  del args.sampling_default_args["window_left"]
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.subsampling_versions import make_subsampling_001, make_unsampling_001

  experiment_data = god.create_experiment_world_004( # New world that allowes adapting sequence order
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_03_feature_stacking_auxilary_loss,
    conformer_func_args=OrderedDict(

      subsampling_func=make_subsampling_001, # This allowes to overwrite `sampling_activation`
      unsampling_func=make_unsampling_001,
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

def make_experiment_11_switch_conv_and_att(
  args, 
  NAME,
  aux_loss_layers = [6],
  test_construct = False
  ):

  experiment_data = god.create_experiment_world_004( # New world that allowes adapting sequence order
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_05_sd_dyn_switch_conv_att,
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


def make_experiment_12_groupnorm_v2(
  args, 
  NAME,
  aux_loss_layers = [6],
  overwrite_bn_settings = None,
  use_tfa_implementation = False,
  dummy_config = None,
  test_construct = False
  ):

  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.conv_mod_versions import make_conv_mod_008_group_norm_custom, make_conv_mod_009_group_norm_custom_tf
  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.network_additions import tf_group_norm
  extras = None
  if use_tfa_implementation:
    import inspect
    conv_func = make_conv_mod_009_group_norm_custom_tf
    extras = {
      "extra_code_string" : "\n" + inspect.getsource(tf_group_norm) # Required for using the tfa group norm implementation
    }
  else:
    conv_func = make_conv_mod_008_group_norm_custom

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_03_feature_stacking_auxilary_loss,
    extra_returnn_net_creation_args=extras,
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

      write_dummpy_config=dummy_config,
      test_construction=test_construct,
  )


# ------------------------- baseline: 'big-short-02' -----------------------

# Basicly:  baseline_03_big_short +only-shuffle and +short-lr
def baseline():
  args = get_defaults()
  NAME = f"{BASE}"

  make_experiment_04_seq_orders(args, NAME)


def se_ffmod():
  args = get_defaults()
  NAME = f"{BASE}+se-block-v1.0-ff-mod"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    se_block_for_module = ["ff_mod"],
  )

def huge_conformer():

  args = get_defaults()
  NAME = f'{BASE}+XL'
  args.shared_network_args['model_dim'] = 1024
  args.sa_default_args['key_dim'] = 512
  args.sa_default_args['value_dim'] = 512

  make_experiment_04_seq_orders(
    args,
    NAME,
  )

def huge_conformer_se_ffmod():
  args = get_defaults()
  NAME = f'{BASE}+XL+stoch-depth-ff-mod'

  args.shared_network_args['model_dim'] = 1024
  args.sa_default_args['key_dim'] = 512
  args.sa_default_args['value_dim'] = 512

  import numpy
  space = numpy.linspace(1.0, 0.5, num=24)

  sd_args = {
    i : {
      "ff_mod1" : space[2*(i-1)],
      "ff_mod2"  : space[2*(i-1) + 1],
    } for i in range(1, 12 + 1)
  }

  args.conformer_defaults.update(OrderedDict(
    apply_stochastic_depth = sd_args
  ))

  make_experiment_06_stoch_depth(
    args,
    NAME,
  )

def lr_reset():
  args = get_defaults()
  NAME = f'{BASE}+lr-reset'
  
  # Shor lr but reset 3 times
  l1 = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
  l2 = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
  l3 = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)

  args.config_args["learning_rates"] = l1 + l2 + l3

  make_experiment_04_seq_orders(
    args,
    NAME,
  )



def old_learning_rate():
  args = get_defaults()
  NAME = f'{BASE}+long-lr'

  learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90, min_lr_ratio=1/50, decay_factor=0.99)

  args.config_args["learning_rates"] = learning_rates

  make_experiment_04_seq_orders(
    args,
    NAME,
  )


def sd_ff_depth_scale():
  args = get_defaults()
  NAME = f'{BASE}+stoch-depth-v2.0-ff-mod+depth-scale-survival-prob-v1-p=0.2'

  import numpy
  space = numpy.linspace(1.0, 0.5, num=24)

  def surv_prob_by_layer(l, L=12.0, p=0.2):
    return 1.0 - ((l/L) * (1.0 - p))

  sd_args = {
    i : {
      "ff_mod1" : surv_prob_by_layer(i),
      "ff_mod2" : surv_prob_by_layer(i),
    } for i in range(1, 12 + 1)
  }

  args.conformer_defaults.update(OrderedDict(
    apply_stochastic_depth = sd_args
  ))

  make_experiment_06_stoch_depth( args, NAME )


def get_defaults_base_08():
  args = original_args_big_baseline_00()
  args.config_args["extra_tag_tim_setup"] = 'baseline-big-long-08'

  del args.returnn_rasr_args_defaults["shuffle_data"] # Not needed cause set by default now

  args.returnn_rasr_args_defaults["num_epochs"] = 600
  args.returnn_train_post_config["cleanup_old_models"]["keep"] = [50, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600]

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

def learning_rate_adjusted():
  def lr_02(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
        min_lr_ratio=1/50, decay_factor=0.99):

    num_lr = int(math.log(min_lr_ratio, decay_factor))
    return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                    [start] * constant_subepoch + \
                    list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                    [min_lr_ratio * start]

  rates = [
    lr_02(warmup_start=0.0002, start=0.0005, warmup_subepoch=100, constant_subepoch=100, min_lr_ratio=1/50, decay_factor=0.99),
    lr_02(warmup_start=0.0002, start=0.0006, warmup_subepoch=50, constant_subepoch=40, min_lr_ratio=1/80, decay_factor=0.99),
    lr_02(warmup_start=0.0002, start=0.0005, warmup_subepoch=100, constant_subepoch=80, min_lr_ratio=1/40, decay_factor=0.98)
  ]

  names = [
    "warmup_start=0.0002-start=0.0005-warmup_subepoch=100-constant_subepoch=100-min_lr_ratio=1/50-decay_factor=0.99",
    "warmup_start=0.0002start=0.0006warmup_subepoch=50constant_subepoch=40min_lr_ratio=1/80decay_factor=0.99",
    "warmup_start=0.0002-start=0.0005-warmup_subepoch=100-constant_subepoch=80-min_lr_ratio=1/40-decay_factor=0.98"
  ]
  for r, n in zip(rates, names):
    args = get_defaults_base_08()
    NAME = f"{BASE}+" + n

    ds = 0.1
    args.config_args["learning_rates"] = r
    args.sampling_default_args["embed_dropout"] = ds

    data = make_experiment_07_se_block(
      args, 
      NAME,
      aux_loss_layers = [8],
      se_block_for_module = ["ff_mod", "conv_mod"],
    )

# + auxilary loss
# + se block conv-mod (ff)
def more_aux():
  args = get_defaults_base_08()
  NAME = f"{BASE}+aux-4-8-12+long-lr"

  learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90, min_lr_ratio=1/50, decay_factor=0.99)
  args.config_args["learning_rates"] = learning_rates

  data = make_experiment_07_se_block(
    args, 
    NAME,
    aux_loss_layers = [4, 8, 12],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )

  


def embed_dropout():
  ds = 0.1
  args = get_defaults_base_08()
  NAME = f"{BASE}-embed-drop={ds}"
  args.sampling_default_args["embed_dropout"] = ds



  data = make_experiment_07_se_block(
    args, 
    NAME,
    aux_loss_layers = [8],
    se_block_for_module = ["ff_mod", "conv_mod"],
  )


def main():
  baseline()
  se_ffmod()

  huge_conformer()
  huge_conformer_se_ffmod()
  sd_ff_depth_scale()

  lr_reset()
  old_learning_rate()

  embed_dropout()
  learning_rate_adjusted()

  more_aux()