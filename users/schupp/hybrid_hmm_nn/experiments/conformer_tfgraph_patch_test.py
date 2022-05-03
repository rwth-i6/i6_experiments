# emformer implementation (masking based training)

import math
import numpy
import copy

from copy import deepcopy

from i6_core.returnn.config import CodeWrapper
import recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline.librispeech_hybrid_system as librispeech_hybrid_system
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.get_network_args import enc_half_args, get_network_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.network_builders.transformer_network_bhv12 import attention_for_hybrid


from sisyphus import *
gs.ALIAS_AND_OUTPUT_SUBDIR = "conformer/baseline_test_X"

type = 'conformer'
prefix = 'conf_test'

# -----------------------------------
#returnn_root = gs.RETURNN_ROOT
returnn_root = '/u/schupp/setups/ping_setup_refactor_tf23/returnn_tfgraph_patch'
gs.RETURNN_ROOT = returnn_root
# = returnn_root

gs.ALIAS_AND_OUTPUT_SUBDIR = "conformer/tests/"

# ------------------------------------
num_encoder_layers = 12
num_classes = 12001
num_inputs = 50
target = 'classes'

feature_name = 'gammatone_unnormalized'

# --------------------------------------------------
system = librispeech_hybrid_system.LibrispeechHybridSystem()
# ------------------------------------------------------------
def lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
       min_lr_ratio=1/50, decay_factor=0.99):

  num_lr = int(math.log(min_lr_ratio, decay_factor))
  return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                   [start] * constant_subepoch + \
                   list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                   [min_lr_ratio * start]

def batchnorm2layernorm(network, keep_batchnorm = False, axes = 'BF'):

  network_layernorm = copy.deepcopy(network)

  for name, layer in network_layernorm.items():

    if layer['class'] == 'rec':
      for name_rec, layer_rec in layer['unit'].items():
        if network_layernorm[name]['unit'][name_rec]['class'] == 'batch_norm':
          if not keep_batchnorm:
            network_layernorm[name]['unit'][name_rec]['class'] = 'layer_norm'
          else:
            network_layernorm[name]['unit'][name_rec]['class'] = 'norm'
            network_layernorm[name]['unit'][name_rec]['axes'] = axes

    if network_layernorm[name]['class'] == 'batch_norm':
      if not keep_batchnorm:
        network_layernorm[name]['class'] = 'layer_norm'
      else:
        network_layernorm[name]['class'] = 'norm'
        network_layernorm[name]['axes'] = axes

  return network_layernorm

# --------------------------------------------------------------------
def lr1(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
       min_lr_ratio=1/50, decay_factor=0.99):

  num_lr = int(math.log(min_lr_ratio, decay_factor))
  return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                   [start] * constant_subepoch + \
                   list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                   [min_lr_ratio * start]

# ------------------------------------
target = 'classes'
num_classes = 12001
num_encoder_layers = 12


additional_network_args = {} #dict_additional_network_args['two_vggs'] we aint want that for now
nn_train_args = {'mem_rqmt': 32, # This also seems a little low
                 'crnn_config': {'learning_rates': lr1(),
                                 'learning_rate': 0.0005 * 1/50,
                                 'min_learning_rate': 0.0005 * 1/50,
                                 'batch_size': 8256} 
                 }


# ---------------------------------------------------------------
system = librispeech_hybrid_system.LibrispeechHybridSystem()
# ---------------------------------------------------------------
def lr(warmup_start=0.0002, start=0.0005, warmup_subepoch=10, constant_subepoch=90,
       min_lr_ratio=1/50, decay_factor=0.99):

  num_lr = int(math.log(min_lr_ratio, decay_factor))
  return list(numpy.linspace(warmup_start, start, num=warmup_subepoch)) + \
                   [start] * constant_subepoch + \
                   list(start * numpy.logspace(1, num_lr, num=num_lr, base=decay_factor)) + \
                   [min_lr_ratio * start]

def batchnorm2layernorm(network, keep_batchnorm = False, axes='BF'):

  network_layernorm = copy.deepcopy(network)

  for name, layer in network_layernorm.items():

    if layer['class'] == 'rec':
      for name_rec, layer_rec in layer['unit'].items():
        if network_layernorm[name]['unit'][name_rec]['class'] == 'batch_norm':
          if not keep_batchnorm:
            network_layernorm[name]['unit'][name_rec]['class'] = 'layer_norm'
          else:
            network_layernorm[name]['unit'][name_rec]['class'] = 'norm'
            network_layernorm[name]['unit'][name_rec]['axes'] = 'BF'

    if network_layernorm[name]['class'] == 'batch_norm':
      if not keep_batchnorm:
        network_layernorm[name]['class'] = 'layer_norm'
      else:
        network_layernorm[name]['class'] = 'norm'
        network_layernorm[name]['axes'] = axes

  return network_layernorm

# --------------------------
SA_separated_additional_network_args = {
  'inspection_idx': list(range(1, 13)),
  'att_weights_inspection': True,
  'add_blstm_block': False,
  'add_conv_block': True
}
# -----------------------------------

def run(bhv=1):
  L2 = 0.00007
  drop = 0.05
  bs = 4004

  name = f"baseline_chunk200.100_bs-{bs}_l2-0.00007_drop-0.05-v{bhv}"
  # Lets stack with the new version of transformer!!!

  _nn_train_args = deepcopy(nn_train_args)

  #"wups2-const18": 
  _nn_train_args["crnn_config"].update({'learning_rates' :lr1(warmup_subepoch=2, constant_subepoch=18, decay_factor=0.99)})

  #_nn_train_args["crnn_config"]["log_verbosity"] = 5 # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)
  if bhv > 0:
    _nn_train_args["crnn_config"]["behavior_version"] = 12 #12 # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)

  _nn_train_args["crnn_config"]["optimizer"] = {"class": "nadam"}

  if bhv > 2:
    _nn_train_args["crnn_config"]["chunking"] = "200:100" # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)
  _nn_train_args["crnn_config"]["batch_size"] = bs #200 # TODO: just for testting this low # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)


  _enc_half_args = deepcopy(enc_half_args)
  _enc_half_args["ff_dim"] = 1538 # Best from previous

  #_enc_half_args["tbs_version"] = "half_ratio"
  #_enc_half_args["half_ratio"] = 0.5 # leave default for now
  # TODO: actually we want to use 0.4 now

  network_args = get_network_args(num_enc_layers = 12,
                                  type = type, enc_args = _enc_half_args,
                                  target = target, num_classes = num_classes,
                                  **additional_network_args)

  network = attention_for_hybrid(**network_args).get_network()


  for layer_name, layer_dict in network.items():
    if layer_dict.get('class', '') == 'linear' or layer_dict.get('class', '') == 'conv':
      network[layer_name].update({"L2" : L2})

  for layer_name, layer_dict in network.items():
    if layer_dict.get('dropout', 'none') != "none":
      network[layer_name]['dropout'] = drop

  for layer_name, layer_dict in network.items():
    if layer_dict.get('class', '') == 'batch_norm':
      network[layer_name]["masked_time"] = True

  feature_name = 'gammatone'

  #tmp_T_20_20_nn_train_args['crnn_config']['python_prolog'] = python_prolog

  system.nn_train(prefix + name, 400, network,
                  train_corpus_key = 'train-other-960',
                  feature_name = feature_name, align_name = 'align_hmm',
                  epoch_split = 400,
                  **_nn_train_args)

  system.nn_recog(name=prefix + name + "_recog",
                  train_corpus_key='train-other-960',
                  recog_corpus_key='dev-other', feature_name=feature_name,
                  #feature_scorer='tf_graph',
                  train_job_name=prefix + name,
                  epochs=[1,8,10,200, 280, 320, 360, 400],
                  csp_args={'flf_tool_exe': librispeech_hybrid_system.RASR_FLF_TOOL})

run(3)
