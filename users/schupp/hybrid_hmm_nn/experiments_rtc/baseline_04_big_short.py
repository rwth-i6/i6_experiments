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

OUTPUT_PATH = "conformer/baseline_04_big_short/"
gs.ALIAS_AND_OUTPUT_SUBDIR = OUTPUT_PATH

BASE = "baseline_03_big_short" # TODO: the name was supposed to be '_04_'

def get_defaults():
  # Instance, then copy, sould ensure no share params
  args = original_args_big_baseline_00()
  return args

# Use this for all make_experiment_<num> with num > 03
def get_defaults_02():
  args = original_args_big_baseline_00() # This doenst share mutables right? TODO
  del args.returnn_rasr_args_defaults["shuffle_data"] # Not needed cause set by default now
  return args

# Keeping ep 50 but not 40 acutally doesn't really make sense
def get_defaults_03():
  args = original_args_big_baseline_00()
  del args.returnn_rasr_args_defaults["shuffle_data"] # Not needed cause set by default now
  args.returnn_train_post_config["cleanup_old_models"]["keep"] = [40, 60, 80, 100, 110, 120]
  return args

#learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
def make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
        min_lr_ratio=1/50, decay_factor=0.99):

    num_lr = int(math.log(min_lr_ratio, decay_factor))
    return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                    [start] * constant_subepoch + \
                    list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                    [min_lr_ratio * start]

def make_experiment_03_rqmt(
  args, 
  NAME,
  aux_loss_layers = [6],
  extra_recog_devtrain=False,
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

      extra_recog_devtrain=extra_recog_devtrain,

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


# ------------------------- baseline: 'big-short' -----------------------

def baseline_big_short():
  # time per sep: 0.384
  # 05.11-20:50:16:  GeForce GTX 1080 Ti
  # params (m): 86.8285
  # wers: {50: '9.89%', 80: '9.39%', 110: '8.5%', 120: '8.65%'}
  args = get_defaults()
  NAME = "baseline_03_big_short" # BASE

  make_experiment_03_rqmt(
    args, 
    NAME,
    extra_recog_devtrain = True
  )


# ---------------------------- ablation of best model elements -----------------------

def no_aux_loss():
  # time per sep: 0.341
  # 05.11-20:50:15:  GeForce GTX 1080 Ti
  # params (m): 82.7601
  # wers: {50: '9.94%', 80: '9.44%', 120: '8.87%'} (WORSE)
  args = get_defaults()
  NAME = f"{BASE}+no-aux"

  make_experiment_03_rqmt(args, NAME, aux_loss_layers=[])

def no_frame_stacking():
  args = get_defaults_03()
  NAME = f"{BASE}+no-frame-stacking"

  make_experiment_10_no_feature_stacking(
    args,
    NAME,
    #test_construct = True
  )

def switch_conv_att_mod():
  args = get_defaults_03()

  NAME = f"{BASE}+switch-att-conv-mod"

  make_experiment_11_switch_conv_and_att(
    args,
    NAME,
  )


# ------------------------------ learning rate ------------------------------

# The learning rate of this model was expected to be trained 15 epochs so wee try a faster decaying variant here:
def lr_shorter():
  NAME = f"{BASE}+shorter-lr-const=10-warmup-10-decay=0.98"

  args = get_defaults_02() # TODO: verify that dataset is still shuffled for this

  learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
  args.config_args["learning_rates"] = learning_rates

  make_experiment_03_rqmt(args, NAME)

# Uses 'newbob_multi_epoch'
def lr_newbob():
  NAME = f"{BASE}+newbob-multi-epoch"

  args = get_defaults_02()
  del args.config_args["learning_rates"] # Not needed when newbob
  # TODO: not sure if there setting conform well to how learning rate was handled before

  args.config_args.update({
    'learning_rate_control': "newbob_multi_epoch",
    'learning_rate_control_relative_error_relative_lr': True,
    'learning_rate_control_min_num_epochs_per_new_lr': 3,
    'newbob_learning_rate_decay': 0.9,
    'newbob_multi_update_interval': 1,
    'newbob_multi_num_epochs': args.EP_SPLIT
  })


  make_experiment_03_rqmt(args, NAME)

# --------------------------------- activations -----------------------------

def conv_mod_relu():
  args = get_defaults_03()
  NAME = f"{BASE}+conv-act=relu"

  args.conv_default_args["conv_act"] = "relu"

  make_experiment_03_rqmt(args, NAME)

def conv_mod_gelu():
  args = get_defaults_03()
  NAME = f"{BASE}+conv-act=gelu"

  args.conv_default_args["conv_act"] = "gelu"

  make_experiment_03_rqmt(args, NAME)

def subsampling_swish():
  args = get_defaults_03()
  NAME = f"{BASE}+subsampling-act=swish"

  args.sampling_default_args["sampling_activation"] = "swish"

  make_experiment_09_sampling_act(args, NAME)

def subsampling_gelu():
  args = get_defaults_03()
  NAME = f"{BASE}+subsampling-act=gelu"

  args.sampling_default_args["sampling_activation"] = "gelu"

  make_experiment_09_sampling_act(args, NAME)

# -------------------------------- batch norm -------------------------------

# + batchnorm instead of layer-norm
# This uses the new returnn defaults
def batchnorm_no_ln():
  # time per sep: 0.387
  # 05.11-22:06:27:  GeForce GTX 1080 Ti
  # params (m): 86.8407
  # wers: {50: '10.3%', 80: '9.34%', 120: '9.08%'}

  args = get_defaults_02()
  NAME = f"{BASE}+batchnorm"

  make_experiment_05_batchnorm(args, NAME) 

def groupnorm_noln():
  # time per sep: 0.383
  # 05.11-22:07:31:  GeForce GTX 1080 Ti
  # params (m): 86.8285
  # wers: {50: '9.78%', 80: '9.31%', 110: '8.56%', 120: '8.6%'}
  args = get_defaults_02()
  NAME = f"{BASE}+groupnorm"

  make_experiment_08_groupnorm(args, NAME)


def groupnorm_v2():
  args = get_defaults_03()
  NAME = f"{BASE}+groupnorm-v2-g=32"

  args.conv_default_args["groups"] = 32
  args.conv_default_args["epsilon"] = 1e-5

  make_experiment_12_groupnorm_v2(
    args, 
    NAME,
    dummy_config = "test.gn.config",
    test_construct = True
  )

def groupnorm_v2_tfa():
  args = get_defaults_03()
  NAME = f"{BASE}+groupnorm-tfa-g=32"

  args.conv_default_args["groups"] = 32
  args.conv_default_args["epsilon"] = 1e-5

  make_experiment_12_groupnorm_v2(
    args, 
    NAME,
    use_tfa_implementation=True
    #dummy_config = "test.gn.config",
    #test_construct = True
  )



# + old batchnorm defaults from behavior_version = 0
def batchnorm_old_defaults():
  args = get_defaults_02()
  NAME = f"{BASE}+batchnorm-old-defaults"

  # momentum = 0.1,
  # update_sample_only_in_training = False,
  # delay_sample_update = False,
  # param_version = 0,
  # masked_time = True,

  make_experiment_05_batchnorm(args, NAME, use_old_bn_defaults=True )


# ----------------------------- chunk/sequnce order ---------------------------
def seq_no_order_no_shuffle():
  # time per sep: 0.389
  # 05.11-20:51:00:  GeForce GTX 1080 Ti
  # params (m): 86.8285
  # wers: {50: '10.04%', 80: '9.45%', 110: '8.8%', 120: '8.47%'}

  args = get_defaults_02()
  NAME = f"{BASE}+no-seq-order+no-shuffle"

  params = OrderedDict(
          segment_order_shuffle = False,
          segment_order_sort_by_time_length = False,
          segment_order_sort_by_time_length_chunk_size = -1,
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  make_experiment_04_seq_orders(args, NAME)

def seq_only_shuffle():
  # time per sep: 0.382
  # 05.11-20:51:00:  GeForce GTX 1080 Ti
  # params (m): 86.8285
  # wers: {50: '10.03%', 80: '9.1%', 110: '8.59%', 120: '8.31%'}

  args = get_defaults_02()
  NAME = f"{BASE}+only-shuffle"

  params = OrderedDict(
          segment_order_shuffle = True,
          segment_order_sort_by_time_length = False, # Already false per default, but lets explicity overwrite it
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  make_experiment_04_seq_orders(args, NAME)

def seq_order_chunk_1000():
  args = get_defaults_02()
  NAME = f"{BASE}+shuffle+order-chunks=1000"

  params = OrderedDict(
          segment_order_shuffle = True,
          segment_order_sort_by_time_length = True,
          segment_order_sort_by_time_length_chunk_size = 1000,
  )

  args.returnn_rasr_args_defaults["overwrite_orders"] = {data: params for data in ["train", "dev", "devtrain"]}

  make_experiment_04_seq_orders(args, NAME)

# -------------------------- stochastic depth -------------------------------

# write test always they said, test good for your health they said
# runns some sanity check for the implementation
# This is also sort of a unit test: `sis/sis m config/baseline_04_big_short.test_sd`
def test_sd():
  args = get_defaults_02()
  NAME = f"{BASE}+stoch-depth-TEST"

  args.conformer_defaults.update(OrderedDict(
    # Should only add stocastic depth to the verry first module, and very last module
    apply_stochastic_depth = {1 : { "ff_mod1" : 0.5 }, 12 : { "ff_mod1" : 0.5 }} 
  ))

  data = make_experiment_06_stoch_depth(
    args, 
    NAME,
    dummy_config = "test.sd.config",
    test_construct = True
  )

  net = data['network']
  assert "enc_001_ff1_ff1_cond_train" in net
  assert "enc_012_ff1_ff1_cond_train" in net
  # more sanity checks?
  # We really only proved that the layer exists and output residual is added

# + 0.5 stoch_depth on *all* ffmods
def sd_ffmod(v=0):
  args = get_defaults_02()
  NAME = f"{BASE}+stoch-depth-v2.{v}-ff-mod"
  if v == 1:
    args.conformer_defaults['multipy_by_surivial_prob_ineval'] = False

  sd_args = {
    i : {
      "ff_mod1" : 0.5,
      "ff_mod2" : 0.5
    } for i in range(1, 12 + 1) # .. for all 12 blocks, yeah i know indexed from 1
  }

  args.conformer_defaults.update(OrderedDict(
    apply_stochastic_depth = sd_args
  ))

  make_experiment_06_stoch_depth( args, NAME )

# + also sd on ff mod, but don't multipy by survival prob in eval:w
def sd_ffmod_v2():
  sd_ffmod(v=1)

def sd_attmod_debug():
  args = get_defaults_02()
  NAME = f"{BASE}+stoch-depth-att-mod-debug"

  sd_args = {
    i : {
      "self_att" : 0.5,
    } for i in [1] #range(1, 12 + 1)
  }

  args.conformer_defaults.update(OrderedDict(
    apply_stochastic_depth = sd_args
  ))


  from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.subsampling_versions import make_subsampling_000_old_bhv, make_unsampling_001

  if False:
    args.sampling_default_args["time_reduction"] = 3
    del args.sampling_default_args["unsampling_strides"]
    del args.sampling_default_args["stacking_stride"]
    del args.sampling_default_args["window_size"]
    del args.sampling_default_args["window_right"]
    del args.sampling_default_args["window_left"]

  experiment_data = god.create_experiment_world_004( 
    name=NAME,
    output_path=OUTPUT_PATH,
    config_base_args=args.config_args,
    conformer_create_func=conformer_returnn_dict_network_generator.make_conformer_04_stoch_depth_dynamic,
    conformer_func_args=OrderedDict(
      # sampling args
      #subsampling_func=make_subsampling_000_old_bhv,
      #unsampling_func=make_unsampling_001,
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

      auxilary_at_layer = [],
      auxilary_loss_args = args.auxilary_loss_args,

      # Conformer args
      **args.conformer_defaults ),
      returnn_train_post_config=args.returnn_train_post_config,
      returnn_rasr_args_defaults=args.returnn_rasr_args_defaults,

      write_dummpy_config=None,
      test_construction=True,
  )


# + 0.5 stoch_depth on *all* self attention mods
def sd_attmod(v=0):
  args = get_defaults_02()
  NAME = f"{BASE}+stoch-depth-v2.{v}-att-mod"
  if v == 1:
    args.conformer_defaults['multipy_by_surivial_prob_ineval'] = False

  sd_args = {
    i : {
      "self_att" : 0.5,
    } for i in [1] #range(1, 12 + 1)
  }

  args.conformer_defaults.update(OrderedDict(
    apply_stochastic_depth = sd_args
  ))

  make_experiment_06_stoch_depth( 
    args, 
    NAME, 
    dummy_config = "test.sd.attmod.config",
    test_construct = True
  )

def sd_sanity_check():
  args = get_defaults_02()
  NAME = f"{BASE}+stoch-depth-p=1.0-TEST"

  sd_args = {
    i : {
      "ff_mod1" : 1.0,
      "ff_mod2" : 1.0
    } for i in range(1, 12 + 1)
  }

  args.conformer_defaults.update(OrderedDict(
    apply_stochastic_depth = sd_args
  ))

  make_experiment_06_stoch_depth( args, NAME )

def sd_attmod_v2():
  sd_attmod_v2(v=1)

# + 0.5 stoch_depth on *all* convolution modules
def sd_conv_mod(v=0):
  args = get_defaults_02()
  NAME = f"{BASE}+stoch-depth-v2.{v}-conv-mod"
  if v == 1:
    args.conformer_defaults['multipy_by_surivial_prob_ineval'] = False

  sd_args = {
    i : {
      "conv_mod" : 0.5,
    } for i in range(1, 12 + 1)
  }

  args.conformer_defaults.update(OrderedDict(
    apply_stochastic_depth = sd_args
  ))

  make_experiment_06_stoch_depth( args, NAME )

def sd_conv_mod_v2():
  sd_conv_mod(v=1)

# + linear scale the survival prob form 0.1 to 0.6 for all ffmodules
def sd_ff_linear_scale():
  args = get_defaults_02()
  NAME = f"{BASE}+stoch-depth-v2.0-ff-mod+linear-scale-survival-1.0-0.5"
  import numpy
  space = numpy.linspace(1.0, 0.5, num=24)

  sd_args = {
    i : {
      "ff_mod1" : space[2*(i-1)],
      "ff_mod2" : space[2*(i-1) + 1],
    } for i in range(1, 12 + 1)
  }

  args.conformer_defaults.update(OrderedDict(
    apply_stochastic_depth = sd_args
  ))

  make_experiment_06_stoch_depth( args, NAME )


def sd_ff_depth_scale():
  args = get_defaults_03()
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



# -------------------------------- Squeeze and exitation --------------------------------

# Test for se block construction
# sis/sis m config/baseline_04_big_short.se_test
def se_test():
  args = get_defaults_02()
  NAME = f"{BASE}+se-block-TEST"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    se_block_for_module = ["ff_mod", "att_mod", "conv_mod"],
    dummy_config = "se.test.config",
    test_construct = True
  )

  # Checks
  net = data["network"]
  for i in range(1, 12 + 1):
    assert f"enc_{i:03d}_ff1_ff1_SE_act1" in net
    assert f"enc_{i:03d}_sa_SE_act1" in net
    assert f"enc_{i:03d}_conv_SE_act1" in net
    # Truely this only check that the layers are present
    # Should be engough to, if the construction did not fail

def se_ffmod():

  args = get_defaults_02()
  NAME = f"{BASE}+se-block-v1.0-ff-mod"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    se_block_for_module = ["ff_mod"],
  )

def se_attmod():

  args = get_defaults_02()
  NAME = f"{BASE}+se-block-v1.0-att-mod"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    se_block_for_module = ["att_mod"],
  )

def se_convmod():
  # time per sep: 0.383
  # 05.11-20:53:11:  GeForce GTX 1080 Ti
  # params (m): 87.2282
  # wers: {50: '9.97%', 80: '9.25%', 110: '8.5%', 120: '8.51%'}

  args = get_defaults_02()
  NAME = f"{BASE}+se-block-v1.0-conv-mod"

  data = make_experiment_07_se_block(
    args, 
    NAME,
    se_block_for_module = ["conv_mod"],
  )

# -------------------------------- XL, huge conformer ----------------------------------------

def huge_conformer():

  args = get_defaults_02()
  NAME = f'{BASE}+XL'
  args.shared_network_args['model_dim'] = 1024
  args.sa_default_args['key_dim'] = 512
  args.sa_default_args['value_dim'] = 512

  make_experiment_03_rqmt(
    args,
    NAME,
    #test_construct = True
  )


def huge_conformer_shorter_lr():

  args = get_defaults_02()
  NAME = f'{BASE}+XL+short-lr'
  args.shared_network_args['model_dim'] = 1024
  args.sa_default_args['key_dim'] = 512
  args.sa_default_args['value_dim'] = 512

  learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
  args.config_args["learning_rates"] = learning_rates

  make_experiment_03_rqmt(
    args,
    NAME,
    #test_construct = True
  )


def huge_conformer_shorter_lr_stoch_depth():

  args = get_defaults_02()
  NAME = f'{BASE}+XL+short-lr+stoch-depth-ff-mod'
  args.shared_network_args['model_dim'] = 1024
  args.sa_default_args['key_dim'] = 512
  args.sa_default_args['value_dim'] = 512

  learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
  args.config_args["learning_rates"] = learning_rates


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
    #test_construct = True
  )

def conformer_16_blocks():
  args = get_defaults_02()
  NAME = f'{BASE}+16blocks+2aux-6-12'

  args.conformer_defaults['num_blocks'] = 16

  make_experiment_03_rqmt(
    args,
    NAME,
    aux_loss_layers = [6, 12],
    #test_construct = True
  )

def conformer_16_blocks_shorter_lr():
  args = get_defaults_02()
  NAME = f'{BASE}+16blocks+2aux-6-12+short-lr'

  args.conformer_defaults['num_blocks'] = 16

  learning_rates = make_log_lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=10, min_lr_ratio=1/40, decay_factor=0.98)
  args.config_args["learning_rates"] = learning_rates

  make_experiment_03_rqmt(
    args,
    NAME,
    aux_loss_layers = [6, 12],
    #test_construct = True
  )


# --------------------------- addint the experiments to computation graph -----------------------

def all_experiments_stochastic_depth():
  # TODO Start v2 experiments only if other where good

  #test_sd() # Test ...

  sd_ffmod()
  sd_ffmod_v2()

  #sd_attmod() This is broken TODO fix
  #sd_attmod_v2()

  sd_conv_mod()
  #sd_conv_mod_v2()

  sd_ff_linear_scale()

  sd_ff_depth_scale()

# ...

def all_experiments_se_block():

  #se_test() # Test...

  se_attmod()
  se_ffmod()
  se_convmod()

# ---

def all_experiments_seq():
  seq_no_order_no_shuffle()
  seq_only_shuffle()
  seq_order_chunk_1000()

def all_activation_experiments():

  conv_mod_relu()
  conv_mod_gelu()
  subsampling_swish()
  subsampling_gelu()

# ------------------------------- full computation graph ------------------------------------------

# We split the experiments,
# Have 3 managers running in paralel...

def main():
  # Baseline
  # baseline_big_short() This is finised and I wan't to text 'extra recog devtrain = True, with it'

  # Ablation
  no_aux_loss()

  # Norm experiments: 
  batchnorm_no_ln()
  groupnorm_noln()
  #groupnorm_v2()
  #batchnorm_old_defaults()

  huge_conformer()
  huge_conformer_shorter_lr()
  huge_conformer_shorter_lr_stoch_depth()
  conformer_16_blocks()


  lr_newbob()
  lr_shorter()
  conformer_16_blocks_shorter_lr()
  

def main2():
  all_experiments_seq()

def main3():
  all_experiments_stochastic_depth()

def main4():
  all_experiments_se_block()

def main5():
  all_activation_experiments()

  no_frame_stacking()
  switch_conv_att_mod()
  #groupnorm_v2_tfa()


def all():
  main()
  main2()
  main3()
  main4()
  main5()
  sd_sanity_check()