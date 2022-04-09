# emformer implementation (masking based training)

import math
import numpy
import copy

from copy import deepcopy

from collections import OrderedDict

from i6_core.returnn.config import CodeWrapper

import recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline.librispeech_hybrid_system as librispeech_hybrid_system
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.get_network_args import enc_half_args, get_network_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.network_builders.transformer_network_bhv12 import attention_for_hybrid

from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.experiments import baseline_experiment as ex_class
ex = ex_class()

from sisyphus import *
gs.ALIAS_AND_OUTPUT_SUBDIR = f"conformer/{ex.name}"

type = 'conformer' # important to set for transformer_network.py
prefix = 'conf_'

returnn_root = '/u/schupp/setups/ping_setup_refactor_tf23/returnn_tfgraph_patch'
gs.RETURNN_ROOT = returnn_root

# ------------------------------------
num_encoder_layers = 12
num_classes = 12001
num_inputs = 50
target = 'classes'


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

def run_exp():
  # Baseline defaults:

  L2 = 0.00007
  drop = 0.05
  bs = 7254

  run_experiment(L2=L2, drop=drop, bs=bs, w_timesample2=False)
  run_experiment(L2=L2, drop=drop, bs=6144, w_timesample2=False) # We need this cause other experiments are with bs=6144
  run_experiment(L2=L2, drop=drop, bs=10000, w_timesample2=True)
  run_experiment(L2=L2, drop=drop, bs=bs, w_timesample2=True, ff_dim=2048)

def run_experiment(*args, **kwargs)

  name , _nn_train_args, _enc_half_args = init_experiment_defaults(*args, **kwargs)
  
  
  # Lets stack with the new version of transformer!!!

  _nn_train_args = deepcopy(nn_train_args)

  #"wups2-const18": 
  _nn_train_args["crnn_config"].update({'learning_rates' :lr1(warmup_subepoch=2, constant_subepoch=18, decay_factor=0.99)})

  #_nn_train_args["crnn_config"]["log_verbosity"] = 5 # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)
  _nn_train_args["crnn_config"]["behavior_version"] = 12 #12 # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)

  _nn_train_args["crnn_config"]["optimizer"] = {"class": "nadam"}

  _nn_train_args["crnn_config"]["chunking"] = "200:100" # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)
  _nn_train_args["crnn_config"]["batch_size"] = bs #200 # TODO: just for testting this low # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)


  _enc_half_args = deepcopy(enc_half_args)
  _enc_half_args["ff_dim"] = ff_dim # Best from previous

  #_enc_half_args["tbs_version"] = "half_ratio"
  #_enc_half_args["half_ratio"] = 0.5 # leave default for now
  # TODO: actually we want to use 0.4 now

  if w_timesample2:
    time_sample = 2
    additional_network_args.update({
        'reduction_factor': [1, time_sample], # Sets the downsampling
        'transposed_conv': True
    })
    name += "_timedsample-2"

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

  rec_epochs = [20, 40, 80, 120] + [160, 180, 190, 200] # Extendet epochs for extendent train

  system.nn_train(prefix + "_" + name, 200, network,
                  train_corpus_key = 'train-other-960',
                  feature_name = feature_name, align_name = 'align_hmm',
                  epoch_split = 20,
                  **_nn_train_args)

  _inputs = OrderedDict(
    name=prefix + '_'  + name,
                train_corpus_key='train-other-960',
                recog_corpus_key='dev-other', feature_name=feature_name, # TODO: here also try different recogs
                train_job_name=prefix + '_'  + name, epochs=rec_epochs,
                csp_args={'flf_tool_exe': librispeech_hybrid_system.RASR_FLF_TOOL})


  recog(_inputs, only_dev_other=True)#, only_eps=[180, 200])

run_exp()
