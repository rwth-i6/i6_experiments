# change batchnorm parameter

from copy import deepcopy
import math
from re import L
import numpy

from collections import OrderedDict

from i6_core.returnn.config import CodeWrapper
import recipe.i6_experiments.users.schupp.hybrid_hmm_nn.pipeline.librispeech_hybrid_system as librispeech_hybrid_system
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.args.get_network_args import enc_half_args, get_network_args
from recipe.i6_experiments.users.schupp.hybrid_hmm_nn.network_builders.transformer_network_bhv12 import attention_for_hybrid

from sisyphus import *
# ------------------------------------------
gs.ALIAS_AND_OUTPUT_SUBDIR = "conformer/ff_dim/"
type = 'conformer'
prefix = 'conformer_'

system = librispeech_hybrid_system.LibrispeechHybridSystem()
# ------------------------------------------------------------
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

# We want to also test other batch_sizes
def all_sets_recog(inputs):
  orig_name = inputs["name"]
  for _set in ["dev-other", "dev-clean", "test-other", "test-clean"]:
    recog_only_s(orig_name, _set, inputs)

def recog_only_s(name, _set, inputs):
  inputs["name"] = name + "_" + _set
  inputs["recog_corpus_key"] = _set
  system.nn_recog(**inputs)

def recog(inputs ,only_dev_other=True):
  orig_name = inputs["name"]
  if only_dev_other:
    recog_only_s(orig_name, "dev-other", inputs)
  else:
    all_sets_recog(inputs)

def run_exp(): #  'swish' -> is same as baseline
  #train_ff_gate()
  train_ff_dim(512)
  train_ff_dim(1024)
  train_ff_dim(1534)
  train_ff_dim(2048)

def train_ff_dim(dim=256):
  name = f"conf__ff-dim-{dim}"
  # Lets stack with the new version of transformer!!!

  L2 = 0.00007
  drop = 0.05

  _nn_train_args = deepcopy(nn_train_args)

  #"wups2-const18": 
  _nn_train_args["crnn_config"].update({'learning_rates' :lr1(warmup_subepoch=2, constant_subepoch=18, decay_factor=0.99)})

  _nn_train_args["crnn_config"]["behavior_version"] = 12 #12 # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)
  _nn_train_args["crnn_config"]["optimizer"] = {"class": "nadam"}

  _nn_train_args["crnn_config"]["chunking"] = "200:100" # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)
  _nn_train_args["crnn_config"]["batch_size"] = 6144 # All configs we wanna reduce batch size ( exept the 8x blocks one, that should stay the same!)


  _enc_half_args = deepcopy(enc_half_args)
  _enc_half_args["ff_dim"] = dim # Best from previous

  _enc_half_args["tbs_version"] = "ff_gating"
  #_enc_half_args["value_dim"] = dim

  # These are the regular defaults
  _enc_half_args["ff_activation"] = "relu"
  _enc_half_args["conv_activation"] = "swish"

  _enc_half_args["half_ratio"] = 0.5 # leave default for now
  # TODO: actually we want to use 0.4 now

  _enc_half_args["with_se"] = False
  _enc_half_args["with_stochastic_depth"] = False
  _enc_half_args["use_ff_gating"] = False

  time_sample = 2
  additional_network_args.update({
      'reduction_factor': [1, time_sample], # Sets the downsampling
      'transposed_conv': True
  })


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

  for layer_name, layer_dict in network.items(): # This is always required for batch_norm with behavior_version=12
    if layer_dict.get('class', '') == 'batch_norm':
      network[layer_name]["masked_time"] = True



  system.nn_train(prefix + '_' + name, 200, network, # 200 is the 'long' train
                train_corpus_key = 'train-other-960', feature_name = 'gammatone', align_name = 'align_hmm',
                epoch_split = 20, add_stoch_depth=True,
                **_nn_train_args)
  # Havin epoch_split = 20 means  200/20 = 10 full epochs
  # whilte Ping had only 800/100 = 8 full epochs

  rec_epochs = [10, 20, 40, 100, 120] + [160, 180, 190, 200] # Extendet epochs for extendent train
                
  _inputs = OrderedDict(
    name=prefix + '_'  + name,
                train_corpus_key='train-other-960',
                recog_corpus_key='dev-other', # TODO: here also try different recogs
                train_job_name=prefix + '_'  + name, epochs=rec_epochs,
                csp_args={'flf_tool_exe': librispeech_hybrid_system.RASR_FLF_TOOL})

  recog(_inputs)

run_exp()
