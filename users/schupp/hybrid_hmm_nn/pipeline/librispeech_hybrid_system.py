"""
Originally lueschers 'librispeech_hybrid_system.py' as hybrid hmm/dnn pipeline

Modified by schupp
Adapted to use returnn 'beahavior_version=12', tf2.3, and more...
"""
import os
import copy
import numpy

import i6_core.corpus as corpus_recipes
import i6_core.am as am
import i6_core.mm as mm
import i6_core.features as features
import i6_core.util as util
import i6_core.meta as meta
import i6_core.rasr as rasr
import i6_core.discriminative_training.lattice_generation as lg

# Important corpus definitons from luescher:
from i6_private.users.pzheng.librispeech_luescher import CORPUS_PATH, DURATIONS, CONCURRENT

from i6_core.returnn import ReturnnRasrTrainingJob
from i6_core.returnn import ReturnnConfig, ReturnnModel
from i6_core.returnn import CompileTFGraphJob



### from recipe.experimental.michel.sequence_training import add_accuracy_output
from i6_experiments.users.schupp.hybrid_hmm_nn.helpers.specaugment_new import *
from i6_experiments.users.schupp.hybrid_hmm_nn.helpers.helper_functions import *
from i6_experiments.users.schupp.hybrid_hmm_nn.helpers.get_lm_config import *

from sisyphus import *
Path = tk.Path

def stoch_depth(x, y, surival_prop=0.5):
  import tensorflow as tf
  import tensorflow_addons as tfa
  # x in always kept,
  # y is the residual randomly dropped
  return tfa.layers.StochasticDepth(survival_probability=surival_prop)([x, y])

def stoch_depth_v2(x, y, surival_prop=0.5, network=None):
  import tensorflow as tf
  import tensorflow_addons as tfa
  # x in always kept,
  # y is the residual randomly dropped
  train = tfa.layers.StochasticDepth(survival_probability=surival_prop)([x, y])
  no_train = x + surival_prop * y
  out = network.cond_on_train(lambda : train, lambda : no_train)
  return out

def cyclepy(x):
    """
    Implementation tweaked from PyTorch
    implementation, to be used as functions & in Lambda layers
    """
    import tensorflow as tf
    import math as m
    pi = tf.constant(m.pi)
    term1 = tf.math.tanh(pi * x)
    term2 = tf.math.tanh(pi * tf.square(x) - 0.95) ** 2
    term3 = (
        tf.math.cos(tf.clip_by_value(x, clip_value_min=-3.0, clip_value_max=3.0)) ** 2
    )
    return term1 * term2 * term3

NATIVE_LSTM_PATH = "/u/schupp/lib/NativeLstm2.so"
#NATIVE_LSTM_PATH = gs.NATIVE_LSTM_PATH #"/u/zhou/libs/nativelstm2/tf1.12/NativeLstm2.so"
CRNN_PYTHON_EXE = gs.CRNN_PYTHON_EXE #'/u/zhou/softwares/python/3.6.1/bin/python3.6'

# no site
PYTHON_HOME = gs.CRNN_PYTHON_EXE #'/u/beck/programs/python/2.7.10'
PYTHON_PROGRAM_HOME = gs.CRNN_PYTHON_HOME #'/u/beck/programs/python/2.7.10/bin/python2.7'

CRNN_ROOT = None # Ok we'll set that in settings
#'/work/asr3/luescher/hiwis/pzheng_tim_adapted/librispeech/from-scratch/returnn_0802/returnn'

#RASR_FLF_TOOL = f'{gs.SPRINT_ROOT}/flf-tool.linux-x86_64-standard'
# TODO: there also other versions for this one
RASR_FLF_TOOL = f'{gs.SPRINT_ROOT}arch/{gs.SPRINT_ARCH}/flf-tool.linux-x86_64-standard'
RASR_AM_TRAINER = f'{gs.SPRINT_ROOT}arch/{gs.SPRINT_ARCH}/acoustic-model-trainer.linux-x86_64-standard'
#RASR_AM_TRAINER = f'{gs.SPRINT_ROOT}/acoustic-model-trainer.linux-x86_64-standard'


# chris original models, for training alignments
chris_960h_best_model_path = "/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/"
chris_100h_best_model_path = "/work/asr3/luescher/setups-data/librispeech/best-model/100h_2019-12-02/"
chris_100h_feature_path = "/work/asr3/luescher/setups-data/librispeech/features/extraction"


work_path_features = gs.FEATURE_EXTRACTION_PATH
origin_feature_path = work_path_features + "/features/extraction"

lexicon_path = Path("/u/corpora/speech/LibriSpeech/lexicon/original.lexicon.golik.xml.gz", cached=True)
am_path = Path(os.path.join(chris_960h_best_model_path,
                                   "EstimateMixturesJob.accumulate.6pfu7tgzPX3p/output/am.mix"), cached=True)

librispeech_corpora_keys = [
  "train-other-960",
  "train-clean-100",
  "dev-clean",
  "dev-other",
  "test-clean",
  "test-other"
]

# concurrent dict for corpus segmentation
concurrent = {key: CONCURRENT[key] for key in librispeech_corpora_keys}
concurrent["train-other-960"] = 100

# duration dict
durations = {key: DURATIONS[key] for key in librispeech_corpora_keys}

# librispeech corpora dict
librispeech_corpora = {}

for key in librispeech_corpora_keys:
  librispeech_corpora[key] = tk.Object()

  if key == "train-other-960":
    librispeech_corpora[key].corpus_file = Path(os.path.join(chris_960h_best_model_path,
                                                             "train-merged-960.corpus.gz"), cached=True)
  elif key == "train-other-100":
    librispeech_corpora[key].corpus_file = Path(os.path.join(chris_100h_best_model_path,
                                                             "train-clean-100.subcorpus.gz"), cached=True)
  else:
    librispeech_corpora[key].corpus_file = Path(os.path.join(chris_960h_best_model_path, "%s.corpus.gz" % key),
                                                cached=True)

  librispeech_corpora[key].audio_dir = os.path.join(CORPUS_PATH, key)
  librispeech_corpora[key].audio_format = 'wav'
  librispeech_corpora[key].duration = durations[key]


rasr.flow.FlowNetwork.default_flags = {'cache_mode': 'bundle'}

class LibrispeechHybridSystem(meta.System):

  def __init__(self):
    super().__init__()

    # init corpus: train|dev|test #
    self.corpus_keys = librispeech_corpora_keys
    self.corpora = librispeech_corpora
    self.concurrent = concurrent
    self.lexicon = lexicon_path

    # adapt to maximum one-day job #
    self.rtfs = {}
    for ck in self.corpus_keys:
      self.rtfs[ck] = int(24 / (self.corpora[ck].duration / self.concurrent[ck]))

    ## normalize pronunciation
    self.setup_corpora_and_lexica_config(lexicon_path, True)
    self.init_default_args()
    self.register_default_features_and_alignments_for_train_and_dev_copora()

  def setup_corpora_and_lexica_config(self, lex_file_path=None, norm_pron=False):

    for ck in self.corpus_keys:
      j = corpus_recipes.SegmentCorpusJob(self.corpora[ck].corpus_file, self.concurrent[ck])
      self.set_corpus(ck, self.corpora[ck], self.concurrent[ck], j.out_segment_path)

      # train-other-960 or train-clean-100
      if ck.startswith('train'):

        all_train_segments = corpus_recipes.SegmentCorpusJob(self.corpora[ck].corpus_file, 1)
        dev_size = 0.002

        split_segments = corpus_recipes.ShuffleAndSplitSegmentsJob(
          segment_file=all_train_segments.out_single_segment_files[1],
          split={'train': 1.0 - dev_size, 'dev': dev_size}
        )

        self.crp[ck + "_train"] = rasr.CommonRasrParameters(base=self.crp[ck])
        self.crp[ck + "_train"].concurrent = 1
        self.crp[ck + "_train"].segment_path = split_segments.out_segments['train']

        self.crp[ck + "_dev"] = rasr.CommonRasrParameters(base=self.crp[ck])
        self.crp[ck + "_dev"].concurrent = 1
        self.crp[ck + "_dev"].segment_path = split_segments.out_segments['dev']

      # lexicon #
      if lex_file_path is not None:
        lexicon_config = rasr.RasrConfig()
        lexicon_config.file = lex_file_path
        lexicon_config.normalize_pronunciation = norm_pron
        self.crp[ck].lexicon_config = lexicon_config
    print("TBS:")
    print(self.crp)

  # create_or_update_not_reset is True:
  #   create if new_am_config_or_new_am_args is None
  #   otherwise:
  #     update the am_args if update_not_replace
  #     otherwise: replace the acoustic config with the new acoustic config
  # otherwise: reset the acoustic config
  def setup_am_config(self, corpus_key,
                      new_am_config_or_new_am_args = None, new_am_post_config = None,
                      create_or_update_not_reset = True,
                      update_not_replace = True):

    if create_or_update_not_reset:

      am_args = copy.deepcopy(self.am_args)

      if update_not_replace: # update the default am_args
        if new_am_config_or_new_am_args:
          am_args.update(new_am_config_or_new_am_args)
      else: # replace the whole acoustic_model_config and acoustic_model_post_config
        if new_am_config_or_new_am_args:
          assert isinstance(new_am_config_or_new_am_args, rasr.RasrConfig)
          self.crp[corpus_key].acoustic_model_config = new_am_config_or_new_am_args
        if new_am_post_config:
          assert isinstance(new_am_post_config, rasr.RasrConfig)
          self.crp[corpus_key].acoustic_model_post_config = new_am_post_config

        # done
        return

      # set am config
      if "allophone_file" in am_args:
        allophone_file = am_args['allophone_file']
        del am_args['allophone_file']
        assert isinstance(allophone_file, (Path, str))
      else:
        allophone_file = False

      if "state_tying_file" in am_args:
        state_tying_file = am_args['state_tying_file']
        del am_args['state_tying_file']
        assert isinstance(state_tying_file, (Path, str))
        assert am_args['state_tying'] == "cart"

      if "allow_zero_weights" in am_args:
        allow_zero_weights = am_args['allow_zero_weights']
        del am_args['allow_zero_weights']
      else:
        allow_zero_weights = False

      # create or update
      self.crp[corpus_key].acoustic_model_config = am.acoustic_model_config(**am_args)
      if self.crp[corpus_key].acoustic_model_post_config is None:
        self.crp[corpus_key].acoustic_model_post_config = rasr.RasrConfig()
      self.crp[corpus_key].acoustic_model_post_config.mixture_set.allow_zero_weights = allow_zero_weights
      if allophone_file:
        self.crp[corpus_key].acoustic_model_post_config.allophones.add_from_file = allophone_file
      if am_args['state_tying'] == "cart":
        self.crp[corpus_key].acoustic_model_config.state_tying.file = state_tying_file

    else: # reset
      self.crp[corpus_key].acoustic_model_config = None
      self.crp[corpus_key].acoustic_model_post_config = None

  # create or replace or reset
  def setup_lm_config(self, corpus_key, new_lm_config = None,
                      create_or_replace_not_reset = True):

    if create_or_replace_not_reset:
      if not new_lm_config: # create language_model_config using the default 4-gram
        self.crp[corpus_key].language_model_config = get_standard_4gram(name='', scale=1.0)
      else:
        assert isinstance(new_lm_config, rasr.RasrConfig)
        self.crp[corpus_key].language_model_config = new_lm_config
    else:
      self.crp[corpus_key].language_model_config = None

  def reset_rasr_exe(self, corpus_key):

    assert corpus_key in self.corpora

    crp = self.crp[corpus_key]

    exe_names = ['acoustic_model_trainer_exe',
                 'allophone_tool_exe',
                 'costa_exe',
                 'feature_extraction_exe',
                 'feature_statistics_exe',
                 'flf_tool_exe',
                 'kws_tool_exe',
                 'lattice_processor_exe',
                 'lm_util_exe',
                 'nn_trainer_exe',
                 'speech_recognizer_exe'
    ]
    for name in exe_names:
      try:
        delattr(crp, name)
      except AttributeError:
        pass


  def init_default_args(self):

    self.am_args = {
      'state_tying': 'cart',  # get overwritten by state_tying_type
      'states_per_phone': 3,
      'state_repetitions': 1,
      'across_word_model': True,
      'early_recombination': False,
      'tdp_scale': 1.0,
      'tdp_transition': (3., 0., 30., 0.),  # loop, forward, skip, exit
      'tdp_silence': (0., 3., 'infinity', 20.),
      'tying_type': 'global',
      'nonword_phones': '',
      'tdp_nonword': (0., 3., 'infinity', 6.),
      'allophone_file': Path(os.path.join(
        chris_960h_best_model_path, "StoreAllophones.34VPSakJyy0U/output/allophones")),
      'state_tying_file': Path(os.path.join(
        chris_960h_best_model_path, "EstimateCartJob.tuIY1yeG7XGc/output/cart.tree.xml.gz")),
      'allow_zero_weights': True
    }

    # 16kHz Gammatone features
    self.gt_args = {
      'minfreq': 100,
      'maxfreq': 7500,
      'channels': 50,  # 68
      'warp_freqbreak': None,  # 3700
      'tempint_type': 'hanning',
      'tempint_shift': .01,
      'tempint_length': .025,
      'flush_before_gap': True,
      'do_specint': False,  # True
      'specint_type': 'hanning',
      'specint_shift': 4,
      'specint_length': 9,
      'normalize': True,
      'preemphasis': True,
      'legacy_scaling': False,
      'without_samples': False,
      'samples_options': {'audio_format': "wav",
                          'dc_detection': True},
      'normalization_options': {},
    }

    self.crnn_feature_scorer_args = {
      'feature_dimension': 50,
      'output_dimension': 12001,
      'mixture_scale': 1.0, # am_scale
      'prior_mixtures': am_path,
      'prior_scale': 0.3,
      'prior_file': None
    }

    self.tf_feature_flow_args = {
      'native_lstm_path': NATIVE_LSTM_PATH,
      'crnn_python_exe': CRNN_PYTHON_EXE,
      'crnn_root': CRNN_ROOT
    }

    self.tf_graph_feature_scorer_args = {
      'scale': 1.0, # am_scale
      'prior_mixtures': am_path,
      'priori_scale': 0.3,
      'prior_file': None
    }

    self.nn_recog_args = {'pronunciation_scale': 6.15,
                          'lm_scale': 13.5,
                          'parallelize_conversion': True,
                          'lattice_to_ctm_kwargs': {'fill_empty_segments': True, 'best_path_algo': "bellman-ford"},
                          'search_parameters': {'beam-pruning': 16.0,
                                               'beam-pruning-limit': 100000,
                                               'word-end-pruning': 0.5,
                                               'word-end-pruning-limit': 10000},
                          'lm_lookahead': True,
                          'lookahead_options': None,
                          'create_lattice': True,
                          'eval_single_best': True,
                          'eval_best_in_lattice': True,

                          'mem': 12,
                          'use_gpu': False,
                          'rtf': 15}

    self.nn_train_args = {

      'num_classes': 12001,
      'partition_epochs': {'train': 100, 'dev': 1},
      'num_epochs': 500,
      #'keep_epochs': [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],

      'save_interval': 1,
      'log_verbosity': 5,

      'device': 'gpu',
      'time_rqmt': 168,  # maximum one week
      'mem_rqmt': 12,
      'cpu_rqmt': 3,

      'use_python_control': False,
    }


    # lattice opts for sMBR training
    self.default_lattice_opts = {'concurrent': 500,
                                 'short_pauses_lemmata': ['[SILENCE]'],

                                  'numerator_options': {'mem': 16, 'rtf': 10},
                                  'raw-denominator_options': {'mem': 16, 'rtf': 10},
                                  'denominator_options': {'mem': 8, 'rtf': 10},
                                  'accuracy_options': {'mem': 16, 'rtf': 10}
                                 }

  # nn default args
  @classmethod
  def get_nn_training_args(cls, network, target='classes', num_inputs=50, num_classes=12001, epoch_split=100,
                           newbob=False,
                           use_spec_augment=True, summary_function=summary, mask_function=mask,
                           random_mask_function=random_mask, transform_function=transform,
                           use_dynamic_lr=False, dynamic_lr_function=custom_dynamic_learning_rate,
                           use_was=False, was_function=att_weight_suppression,
                           use_pretrain=False, pretrain_function=custom_construction_algo, num_repetitions=5,
                           use_pe_transformer_xl=False,
                           add_cyclemoid=False,
                           add_stoch_depth=False,
                           add_stoch_depth_v2=False
                           ):

    python_prolog = {}
    python_prolog['functions'] = []

    batch_size = 4096
    chunking = '200:100'
    learning_rate = 0.0005
    min_learning_rate = 1e-5

    if use_dynamic_lr:
      newbob = False

    # crnn config
    crnn_config = {
      'task': "train",
      'use_tensorflow': True,
      'multiprocessing': True,
      'update_on_device': True,

      #'model': "net-models",
      'stop_on_nonfinite_train_score': False,

      #'log': "crnn.train.log",
      'log_batch_size': True,
      'debug_print_layer_output_template': True,
      'tf_log_memory_usage': True,

      'start_epoch': "auto",
      'start_batch': "auto",
      'batching': "sort_bin_shuffle:.64",  # f"laplace:{num_seqs//1000}"

      'batch_size': batch_size,
      'chunking': chunking,

      'truncation': -1,
      'cache_size': "0",
      'window': 1,

      'num_inputs': num_inputs,
      'num_outputs': {
        'data': [num_inputs, 2],
        'classes': [num_classes, 1]
      },
      'target': target,

      'nadam': True,
      'optimizer_epsilon': 1e-8,

      'gradient_noise': 0.0,  # 0.1

      ## learning rate contral
      'learning_rate_control': "constant",

      'learning_rate_file': "learning_rates",
      'learning_rate': learning_rate,
      'min_learning_rate': min_learning_rate,

      'cleanup_old_models': {'keep_best_n': 3, 'keep_last_n': 3,
        'keep': [10, 20, 40, 80, 90, 100, 110, 116, 120, 140, 160, 180, 190, 200] },
                             #'keep': list(numpy.arange(2 * epoch_split, 12 * epoch_split + 1, epoch_split))},
                             #'keep': [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]},

      'network': network
    }

    if newbob:
      crnn_config.update({
        'learning_rate_control': "newbob_multi_epoch",
        'learning_rate_control_relative_error_relative_lr': True,
        'learning_rate_control_min_num_epochs_per_new_lr': 3,
        'newbob_learning_rate_decay': 0.9,
        'newbob_multi_update_interval': 1,
        'newbob_multi_num_epochs': epoch_split})



    if use_pe_transformer_xl:
      python_prolog['functions'] += [rel_shift, tile_weight_matrix]

    if use_spec_augment:
      python_prolog['functions'] += [summary_function, mask_function,
                        random_mask_function, transform_function]

    if add_cyclemoid:
      python_prolog['functions'] += [cyclepy]

    if add_stoch_depth:
      python_prolog['functions'] += [stoch_depth]

    if add_stoch_depth_v2:
      python_prolog['functions'] += [stoch_depth_v2]

    ## insert functions
    if use_pretrain:
      crnn_config['pretrain'] = {"repetitions": num_repetitions, "construction_algo": pretrain_function.__name__}
      python_prolog['functions'] += [pretrain_function]
    # for pretraining must edit the config.py file in recipe_old.crnn to pass the "repetitions" parameter

    if use_was:
      python_prolog['functions'] += [was_function]

    if use_dynamic_lr:
      crnn_config['dynamic_learning_rate'] = dynamic_lr_function.__name__
      python_prolog['functions'] += [dynamic_lr_function]

    return ReturnnConfig(crnn_config, python_prolog=python_prolog)


  def register_default_features_and_alignments_for_train_and_dev_copora(self):

    ## train corpora
    # 960h train
    self.register_feature('train-other-960', 'gammatone', os.path.join(
      chris_960h_best_model_path, "FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle"))
    self.register_feature('train-other-960', 'gammatone_unnormalized', os.path.join(
      origin_feature_path, "FeatureExtraction.Gammatone.Pwkx0rfszmwj/output/gt.cache.bundle"))
    self.register_alignment('train-other-960', 'align_hmm',
      os.path.join(chris_960h_best_model_path, "AlignmentJob.uPtxlMbFI4lx/output/alignment.cache.bundle"))

    # 100h train
    # self.register_feature('train-clean-100', 'gammatone', os.path.join(
    #   chris_100h_feature_path, "FeatureExtraction.Gammatone.jLxFUwRNFk0u/output/gt.cache.bundle"))
    self.register_alignment('train-clean-100', 'align_hmm',
                            os.path.join(chris_100h_best_model_path, "alignments/alignment.cache.bundle"))

    ## dev and test corpora
    # dev-clean
    self.register_feature('dev-clean', 'gammatone', os.path.join(
      origin_feature_path, "FeatureExtraction.Gammatone.lBf84CoDwVh8/output/gt.cache.bundle"))
    self.stm_files['dev-clean'] = Path(os.path.join(
      work_path_features, "corpus/CorpusToStm.VeLsMXIxIEz2/output/corpus.stm"), cached=True)

    # dev-other
    self.register_feature('dev-other', 'gammatone', os.path.join(
      origin_feature_path, "FeatureExtraction.Gammatone.rnNGEml7b5J7/output/gt.cache.bundle"))
    self.register_feature('dev-other', 'gammatone_unnormalized', os.path.join(
      origin_feature_path, "FeatureExtraction.Gammatone.QdxT0R4Uys5l/output/gt.cache.bundle"))
    self.stm_files['dev-other'] = Path(os.path.join(
      work_path_features, "corpus/CorpusToStm.gYvbBylM5ZfH/output/corpus.stm"), cached=True)

    # test-clean
    self.register_feature('test-clean', 'gammatone', os.path.join(
      origin_feature_path, "FeatureExtraction.Gammatone.a448OeqArSCt/output/gt.cache.bundle"))
    self.stm_files['test-clean'] = Path(os.path.join(
      work_path_features, "corpus/CorpusToStm.LtrzAaW0Mjgh/output/corpus.stm"), cached=True)
    # test-other
    self.register_feature('test-other', 'gammatone', os.path.join(
      origin_feature_path, "FeatureExtraction.Gammatone.iWMN66pUrOPh/output/gt.cache.bundle"))
    self.stm_files['test-other'] = Path(os.path.join(
      work_path_features, "corpus/CorpusToStm.8rQxILYiTlb6/output/corpus.stm"), cached=True)


  def nn_train_and_recog(self, nn_train_args=None, nn_recog_args=None, smbr=False):
    if not smbr:
      self.nn_train(**nn_train_args)
    else:
      self.smbr_nn_train(**nn_train_args)
    self.nn_recog(**nn_recog_args)

  # store models?
  # needs to set training parameters, resources and the network architecture up
  def nn_train(self, name, num_epochs, network,
               train_corpus_key = 'train-other-960',
               num_inputs = 50, num_classes = 12001, target = 'classes',
               epoch_split = 100, feature_name = 'gammatone', align_name = 'align_hmm', **kwargs
               ):

    train_feature_flow = self.feature_flows[train_corpus_key][feature_name]
    train_alignment = self.alignments[train_corpus_key][align_name]

    ## set the am config up (create)
    self.setup_am_config(train_corpus_key, create_or_update_not_reset=False)
    self.setup_am_config(train_corpus_key, create_or_update_not_reset=True,
                         new_am_config_or_new_am_args=None)


    nn_train_args = copy.deepcopy(self.nn_train_args)

    if num_classes:
      nn_train_args['num_classes'] = num_classes
    if epoch_split:
      # updata epoch_split and keep_epoch parameters
      nn_train_args['partition_epochs']['train'] = epoch_split

    nn_train_args['num_epochs'] = num_epochs

    csp_args = kwargs.pop('csp_args', None)
    if csp_args:
      nn_train_args['train_crp'] = copy.deepcopy(self.crp[train_corpus_key + '_train'])
      nn_train_args['dev_crp'] = copy.deepcopy(self.crp[train_corpus_key + '_dev'])

      for key, value in csp_args.items():
        setattr(nn_train_args['train_crp'], key, csp_args[key])
        setattr(nn_train_args['dev_crp'], key, csp_args[key])
    else:
      nn_train_args['train_crp'] = self.crp[train_corpus_key + '_train']
      nn_train_args['dev_crp'] = self.crp[train_corpus_key + '_dev']

    nn_train_args['feature_flow'] = train_feature_flow
    nn_train_args['alignment'] = train_alignment

    returnn_config = self.get_nn_training_args(network, target=target, num_inputs=num_inputs,
                                                             num_classes=num_classes, epoch_split=epoch_split,
                                                             newbob=kwargs.pop('newbob', False),
                                                             use_spec_augment=kwargs.pop('use_spec_augment', True),
                                                             summary_function=kwargs.pop('summary_function', summary),
                                                             mask_function=kwargs.pop('mask_function', mask),
                                                             random_mask_function=kwargs.pop('random_mask_function', random_mask),
                                                             transform_function=kwargs.pop('transform_function', transform),
                                                             use_dynamic_lr=kwargs.pop('use_dynamic_lr', False),
                                                             dynamic_lr_function=kwargs.pop('dynamic_lr_function',
                                                                                            custom_dynamic_learning_rate),
                                                             use_was=kwargs.pop('use_was', False),
                                                             was_function=kwargs.pop('was_function', att_weight_suppression),
                                                             use_pretrain=kwargs.pop('use_pretrain', False),
                                                             pretrain_function=kwargs.pop('pretrain_function',
                                                                                          custom_construction_algo),
                                                             num_repetitions=kwargs.pop('num_repetitions', 5),
                                                             use_pe_transformer_xl=kwargs.pop('use_pe_transformer_xl', False),
                                                             add_cyclemoid=kwargs.pop('add_cyclemoid', False),
                                                             add_stoch_depth=kwargs.pop('add_stoch_depth', False),
                                                             add_stoch_depth_v2=kwargs.pop('add_stoch_depth_v2', False)
                                                             )

    # update other parameters
    new_crnn_config = kwargs.pop('crnn_config', None)

    if new_crnn_config:
      for key in new_crnn_config.get('del', []):
        if key in returnn_config.config.keys():
          del returnn_config.config[key]
      new_crnn_config.pop('del', None)

      if kwargs.pop('replace_crnn_config', False):
        returnn_config.config = new_crnn_config
      else:
        if 'python_prolog' in new_crnn_config:
          returnn_config.python_prolog.update(new_crnn_config.pop('python_prolog'))

        returnn_config.config.update(new_crnn_config)

    for key in kwargs.pop('del', []):
      if key in nn_train_args.keys():
        del nn_train_args[key]
    kwargs.pop('del', None)

    nn_train_args.update(kwargs)
    j = ReturnnRasrTrainingJob(returnn_config = returnn_config, **nn_train_args)



    j.add_alias(f"{name}/train.job")
    self.jobs[train_corpus_key]['train_nn_%s' % name] = j

    self.nn_models[train_corpus_key][name] = j.out_models
    self.nn_configs[train_corpus_key][name] = j.out_returnn_config_file

    tk.register_output(f"{name}/returnn.config", j.out_returnn_config_file)
    tk.register_output(f"{name}/score_and_error.png", j.out_plot_se)
    tk.register_output(f"{name}/learning_rate.png", j.out_plot_lr)

  # lm_scale will overwrite the scale in the lm_config
  def nn_align(self, name, corpus_key, feature_name='gammatone',

               crnn_config=None, nn_model=None,
               train_corpus_key=None, train_job_name=None, epoch=None,

               am_args=None, lm_config=None, lm_scale=None,
               new_tf_feature_flow_args=None, feature_scorer=None,

               register_alignment=True,
               use_gpu=False, alignment_rqmt={'cpu': 3, 'mem': 12, 'time': 100.0},
               **kwargs):

    assert corpus_key in self.corpora
    if not crnn_config or not nn_model:
      assert train_corpus_key is not None and train_job_name is not None and epoch is not None
      assert train_corpus_key in self.corpora
      crnn_config = self.jobs[train_corpus_key]['train_nn_%s' % train_job_name].crnn_config
      nn_model = self.nn_models[train_corpus_key][train_job_name][epoch]

    tf_feature_flow_args = copy.deepcopy(self.tf_feature_flow_args)
    if new_tf_feature_flow_args:
      tf_feature_flow_args.update(new_tf_feature_flow_args)


    ## first reset
    self.setup_am_config(train_corpus_key, create_or_update_not_reset=False)
    self.setup_lm_config(corpus_key, create_or_replace_not_reset=False)

    ## create the am and lm config with the given parameters or default parameters
    self.setup_am_config(corpus_key, create_or_update_not_reset=True, new_am_config_or_new_am_args=am_args)
    self.setup_lm_config(corpus_key, create_or_replace_not_reset=True, new_lm_config=lm_config)

    if lm_scale:
      self.crp[corpus_key].language_model_config.scale = lm_scale


    csp_args = kwargs.get('csp_args', None)
    if csp_args:
      target_csp = copy.deepcopy(self.crp[corpus_key])
      for key, value in csp_args.items():
        setattr(target_csp, key, csp_args[key])
    else:
      target_csp = self.crp[corpus_key]

    crnn_config_for_flow = copy.deepcopy(crnn_config)
    crnn_config_for_flow = self.adapt_crnn_config_for_recog(crnn_config_for_flow)


    flow = self.get_full_tf_feature_flow(
      base_flow=self.feature_flows[corpus_key][feature_name],
      crnn_config=crnn_config_for_flow,
      nn_model=nn_model,
      **tf_feature_flow_args
    )

    if not feature_scorer:
      feature_scorer = self.feature_scorers[corpus_key].get('tf_graph', None)
      assert feature_scorer

    align_job = mm.AlignmentJob(
      crp=target_csp,
      feature_flow=flow,
      feature_scorer=feature_scorer,
      use_gpu=use_gpu, **kwargs
    )
    align_job.rqmt.update(alignment_rqmt)

    align_job.add_alias("alignment_%s" % name)
    tk.register_output("alignment_%s.bundle" % name, align_job.alignment_bundle)

    if register_alignment:
      self.register_alignment(corpus_key, "alignment_%s" % name, align_job.alignment_bundle.get_path())


  @classmethod
  def adapt_crnn_config_for_recog(cls, crnn_config, pop_num_epochs=False):

    assert isinstance(crnn_config, ReturnnConfig)

    if pop_num_epochs:
      crnn_config.post_config.pop('num_epochs', None)

    for name, layer in crnn_config.config['network'].items():
      if isinstance(layer.get('unit', ''), str) and layer.get('unit', '').startswith('lstm'):
        layer['unit'] = 'nativelstm2'
      # auxiliary loss
      if layer.get('target', None) and name != 'output':
        layer.pop('target')
        layer.pop('loss', None)
        layer.pop('loss_scale', None)
        layer.pop('loss_opts', None)
      # recurrent unit
      if layer.get('class', '') == 'rec' and \
        isinstance(layer.get('unit', ''), dict) and 'output' in layer['unit'].keys():
        if layer['unit']['output'].get('class', None) == 'softmax':
          layer['unit']['output']['class'] = 'linear'
          layer['unit']['output']['activation'] = 'log_softmax'
          layer['unit']['output'].pop('target', None)


    if crnn_config.config['network']['output'].get('class', None) == 'softmax':
      # set output to log-softmax
      crnn_config.config['network']['output']['class'] = 'linear'
      crnn_config.config['network']['output']['activation'] = 'log_softmax'
      crnn_config.config['network']['output'].pop('target', None)
    return crnn_config

  @classmethod
  def get_feature_flow(cls, base_flow, tf_flow):
    feature_flow = rasr.FlowNetwork()
    base_mapping = feature_flow.add_net(base_flow)
    tf_mapping = feature_flow.add_net(tf_flow)
    feature_flow.interconnect_inputs(base_flow, base_mapping)
    feature_flow.interconnect(base_flow, base_mapping, tf_flow, tf_mapping, {'features': 'input-features'})
    feature_flow.interconnect_outputs(tf_flow, tf_mapping)

    return feature_flow

  @classmethod
  def get_tf_flow(cls, nn_model, tf_graph, native_lstm_path):

    model_base = nn_model.model.get_path().replace('.meta','')
    model_path = nn_model.model

    tf_flow = rasr.FlowNetwork()
    ## this is an intermediate connection point
    tf_flow.add_input('input-features')

    tf_flow.add_output('features')
    tf_flow.add_param('id')
    tf_fwd = tf_flow.add_node('tensorflow-forward', 'tf-fwd', {'id': '$(id)'})


    tf_flow.link('network:input-features', tf_fwd + ':features')
    tf_flow.link(tf_fwd + ':log-posteriors', 'network:features')

    tf_flow.config = rasr.RasrConfig()
    tf_flow.config[tf_fwd].input_map.info_0.param_name = 'features'
    tf_flow.config[tf_fwd].input_map.info_0.tensor_name = 'extern_data/placeholders/data/data'
    tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name = 'extern_data/placeholders/data/data_dim0_size'

    tf_flow.config[tf_fwd].output_map.info_0.param_name = 'log-posteriors'
    tf_flow.config[tf_fwd].output_map.info_0.tensor_name = 'output/output_batch_major'

    tf_flow.config[tf_fwd].loader.type = 'meta'
    tf_flow.config[tf_fwd].loader.meta_graph_file = tf_graph
    tf_flow.config[tf_fwd].loader.saved_model_file = rasr.StringWrapper(model_base, model_path)

    tf_flow.config[tf_fwd].loader.required_libraries = native_lstm_path

    return tf_flow

  def get_full_tf_feature_flow(self, base_flow, crnn_config, nn_model,
                               native_lstm_path, crnn_python_exe=None,
                               crnn_root=None):


    if not crnn_python_exe:
      crnn_python_exe = gs.RETURNN_PYTHON_EXE

    if not crnn_root:
      crnn_root = gs.RETURNN_ROOT

    compile_graph_job = CompileTFGraphJob(crnn_config, returnn_python_exe=crnn_python_exe, returnn_root=crnn_root)

    return self.get_feature_flow(base_flow,
                            self.get_tf_flow(nn_model, compile_graph_job.out_graph, native_lstm_path))

  # set feature scorer for a corpus, require a training job
  def get_crnn_feature_scorer(self, name, train_corpus_key, train_job_name, epoch=None, model=None,
                              reuse=False, reestimated_prior=False, **kwargs):

    # epoch and not model
    if epoch is not None and model is None:
      model = self.nn_models[train_corpus_key][train_job_name][epoch]

    assert model is not None

    if reestimated_prior:
      assert 'corpus' in kwargs
      assert 'feature_name' in kwargs

      prior_corpus_key = kwargs.pop('prior_corpus_key')
      feature_name = kwargs.pop('feature_name')

    feature_scorer = rasr.ReturnnScorer(model=model, **kwargs)

    if reestimated_prior:
      s = copy.deepcopy(feature_scorer)
      s.config.priori_scale = 0.0

      feature_flow = self.feature_flows[prior_corpus_key][feature_name]
      csp = self.crp[prior_corpus_key]

      score_features_job = am.ScoreFeaturesJob(crp=csp,
                                           feature_flow=feature_flow,
                                           feature_scorer=s,
                                           normalize=True,
                                           plot_prior=True,
                                           rtf=5.0)

      score_features_job.set_rqmt('run', {'mem': 8})

      kwargs.pop('prior_file', None)
      feature_scorer = rasr.CRNNScorer(
        model=model, prior_file=score_features_job.prior, **kwargs)

    # stored only when the prior is not corpus specific
    if reuse and not reestimated_prior:
      feature_scorer_name = 'crnn'
      if name:
        feature_scorer_name = '_'.join([name, feature_scorer_name])
      self.feature_scorers[train_corpus_key][feature_scorer_name] = feature_scorer

    return feature_scorer

  # return the feature flow
  def get_precomputed_hybrid_feature_scorer(self, name, train_corpus_key, reuse=False, **kwargs):
    feature_scorer = rasr.PrecomputedHybridFeatureScorer(**kwargs)
    if reuse:
      self.feature_scorers[train_corpus_key]['_'.join([name, 'tf_graph']) if name else 'tf_graph'] = feature_scorer

    return feature_scorer


  def nn_recog(self, name, recog_corpus_key,
               train_job_name, epochs,
               train_corpus_key='train-other-960',

               feature_name='gammatone',
               feature_scorer='tf_graph',
               model_path=None,
               am_args=None, lm_config=None,

               new_feature_scorer_args=None,
               new_tf_feature_flow_args=None,
               new_nn_recog_args=None,

               lm_scale_optimize=True,
               rerecog=False,
               am_scale_optimize=False, use_gpu=False, csp_args=None,
               sort_files=True
               ):

    # name: used for the recog. jobs and scorers
    assert train_corpus_key in self.corpora and recog_corpus_key in self.corpora

    if isinstance(epochs, int):
      epochs = [epochs]

    if epochs is None:
      assert model_path

      models = {0: ReturnnModel(self.jobs[train_corpus_key]['train_nn_%s' % train_job_name].crnn_config_file, Path(model_path), 1)}
      epochs = [0]
    else:
      models = self.nn_models[train_corpus_key][train_job_name]

    self.reset_rasr_exe(recog_corpus_key)

    ## first reset
    self.setup_am_config(recog_corpus_key, create_or_update_not_reset=False)
    self.setup_lm_config(recog_corpus_key, create_or_replace_not_reset=False)

    ## set the am and lm config
    self.setup_am_config(recog_corpus_key, create_or_update_not_reset=True, update_not_replace=True,
                         new_am_config_or_new_am_args=am_args)
    self.setup_lm_config(recog_corpus_key, create_or_replace_not_reset=True,
                         new_lm_config=lm_config)

    self.set_sclite_scorer(recog_corpus_key, **{'sort_files': sort_files})

    nn_recog_args = copy.deepcopy(self.nn_recog_args)
    if new_nn_recog_args:
      nn_recog_args.update(new_nn_recog_args)
    nn_recog_args['corpus'] = recog_corpus_key
    for epoch in epochs:
      assert epoch in list(models.keys()), "epoch %d not saved in %s" % (epoch, train_job_name)

      # crnn feature scorer
      if feature_scorer == 'crnn':
        crnn_feature_scorer_args = copy.deepcopy(self.crnn_feature_scorer_args)
        if new_feature_scorer_args:
          crnn_feature_scorer_args.update(new_feature_scorer_args)

        nn_recog_args['feature_scorer'] = self.get_crnn_feature_scorer(
          '',
          train_corpus_key,
          train_job_name, epoch=None if epoch == 0 else epoch,
          model=None if epoch != 0 else models[epoch],
          **crnn_feature_scorer_args)

        nn_recog_args['flow'] = self.feature_flows[recog_corpus_key][feature_name]

        extra_post_config = rasr.RasrConfig()
        extra_post_config['*'].python_home = PYTHON_HOME
        extra_post_config['*'].python_program_name = PYTHON_PROGRAM_HOME
        nn_recog_args['extra_post_config'] = extra_post_config

      elif feature_scorer == 'tf_graph':
        crnn_config = copy.deepcopy(self.jobs[train_corpus_key]['train_nn_%s' % train_job_name].returnn_config)
        crnn_config = self.adapt_crnn_config_for_recog(crnn_config, pop_num_epochs=True)

        tf_feature_flow_args = copy.deepcopy(self.tf_feature_flow_args)

        if new_tf_feature_flow_args:
          tf_feature_flow_args.update(new_tf_feature_flow_args)

        tf_graph_feature_scorer_args = copy.deepcopy(self.tf_graph_feature_scorer_args)
        if new_feature_scorer_args:
          tf_graph_feature_scorer_args.update(new_feature_scorer_args)

        nn_recog_args['flow'] = self.get_full_tf_feature_flow(
          base_flow=self.feature_flows[recog_corpus_key][feature_name],
          crnn_config=crnn_config,
          nn_model=models[epoch],#self.nn_models[train_corpus_key][train_job_name][epoch],
          **tf_feature_flow_args
        )
        nn_recog_args['feature_scorer'] = self.get_precomputed_hybrid_feature_scorer(
          '', recog_corpus_key, **tf_graph_feature_scorer_args)
      else:
        raise NotImplementedError

      if csp_args:
        for key, value in csp_args.items():
          setattr(self.crp[recog_corpus_key], key, csp_args[key])

      nn_recog_args['name'] = f"{name}/{epoch:03}"

      recog_job_name = 'recog_%s' % nn_recog_args['name']
      opt_job_name = 'optimize_%s' % nn_recog_args['name']

      if rerecog:
        if lm_scale_optimize:
          if not am_scale_optimize:
            self.recog_and_optimize(**nn_recog_args)
          else:
            self.recog(**nn_recog_args)
            self.optimize_am_lm(recog_job_name, recog_corpus_key, nn_recog_args['pronunciation_scale'],
                                  nn_recog_args['lm_scale'], '', opt_only_lm_scale=False)
            new_lm_scale = self.jobs[recog_corpus_key][opt_job_name].best_lm_score
            new_am_scale = self.jobs[recog_corpus_key][opt_job_name].best_am_score
            new_recog_name = nn_recog_args['name'] + '-opt'

            nn_recog_args['name'] = new_recog_name
            nn_recog_args['pronunciation_scale'] = new_am_scale
            nn_recog_args['lm_scale'] = new_lm_scale
            self.recog(**nn_recog_args)
      else:
        self.recog(**nn_recog_args)
        if lm_scale_optimize:
          if not am_scale_optimize:
            self.optimize_am_lm(recog_job_name, recog_corpus_key, nn_recog_args['pronunciation_scale'],
                                  nn_recog_args['lm_scale'], '', opt_only_lm_scale=True)
          else:
            self.optimize_am_lm(recog_job_name, recog_corpus_key, nn_recog_args['pronunciation_scale'],
                                  nn_recog_args['lm_scale'], '', opt_only_lm_scale=False)

      if use_gpu:
        # rerecog
        # lm_gc_job: AdvancedTreeSearchLmImageAndGlobalCacheJob
        j = self.jobs[recog_corpus_key][recog_job_name]
        j.rqmt.update({'gpu': 1, 'time': 5, 'mem': 32, 'qsub_args': '-l qname=*1080*'})
        j.lm_gc_job.rqmt.update({'mem': 32})

        j = self.jobs[recog_corpus_key].get(recog_job_name + '-optlm', None)
        if j:
          j.rqmt.update({'gpu': 1, 'time': 5, 'mem': 32, 'qsub_args': '-l qname=*1080*'})
          j.lm_gc_job.rqmt.update({'mem': 32})

        j = self.jobs[recog_corpus_key].get(recog_job_name + '-opt', None)
        if j:
          j.rqmt.update({'gpu': 1, 'time': 5, 'mem': 32, 'qsub_args': '-l qname=*1080*'})
          j.lm_gc_job.rqmt.update({'mem': 32})


  def generate_lattices(self, name, lat_gen_corpus_key, feature_name,
                        feature_scorer, new_lattice_options=None):

    assert lat_gen_corpus_key in self.corpora

    lattice_options = copy.deepcopy(self.default_lattice_opts)
    if new_lattice_options:
      lattice_options.update(new_lattice_options)

    csp_for_lat_gen = copy.deepcopy(self.crp[lat_gen_corpus_key])
    feature_flow = self.feature_flows[lat_gen_corpus_key][feature_name]

    if lattice_options['concurrent'] != csp_for_lat_gen.concurrent:

      train_segments = corpus_recipes.SegmentCorpusJob(csp_for_lat_gen.corpus_file, lattice_options['concurrent'])
      csp_for_lat_gen.concurrent = lattice_options['concurrent']
      lattice_options.segment_path = train_segments.segment_path

    if 'lm' in lattice_options:
      if isinstance(lattice_options['lm'], rasr.RasrConfig):
        csp_for_lat_gen.language_model_config = lattice_options['lm']
      else:
        csp_for_lat_gen.language_model_config = rasr.RasrConfig()
        for key, val in lattice_options['lm'].items():
          csp_for_lat_gen.language_model_config[key] = val

    if 'lexicon' in lattice_options:
      csp_for_lat_gen.lexicon_config = rasr.RasrConfig()
      for key, val in lattice_options['lexicon'].items():
        csp_for_lat_gen.lexicon_config[key] = val

    # Numerator
    num = lg.NumeratorLatticeJob(crp=csp_for_lat_gen,
                                 feature_flow=feature_flow,
                                 feature_scorer=feature_scorer,
                                 **lattice_options['numerator_options'])

    self.jobs[lat_gen_corpus_key]['numerator_%s' % name] = num

    # Raw Denominator
    rawden = lg.RawDenominatorLatticeJob(crp=csp_for_lat_gen,
                                         feature_flow=feature_flow,
                                         feature_scorer=feature_scorer,
                                         **lattice_options['raw-denominator_options'])

    self.jobs[lat_gen_corpus_key]['raw-denominator_%s' % name] = rawden

    # Denominator
    den = lg.DenominatorLatticeJob(crp=csp_for_lat_gen,
                                   raw_denominator_path=rawden.lattice_path,
                                   numerator_path=num.lattice_path,
                                   **lattice_options['denominator_options'])

    self.jobs[lat_gen_corpus_key]['denominator_%s' % name] = den

    # State Accuracy
    stat = lg.StateAccuracyJob(crp=csp_for_lat_gen,
                               feature_flow=feature_flow,
                               feature_scorer=feature_scorer,
                               denominator_path=den.lattice_path,
                               **lattice_options['accuracy_options'])

    self.jobs[lat_gen_corpus_key]['state-accuracy_%s' % name] = stat

    # Phone Accuracy
    phon = lg.PhoneAccuracyJob(crp=csp_for_lat_gen,
                               feature_flow=feature_flow,
                               feature_scorer=feature_scorer,
                               denominator_path=den.lattice_path,
                               short_pauses=lattice_options['short_pauses_lemmata'],
                               **lattice_options['accuracy_options'])

    self.jobs[lat_gen_corpus_key]['phone-accuracy_%s' % name] = phon


  ## needs to access models
  ## lattice generation parameters
  # ArchiveJob not found
  def smbr_nn_train(self, train_job_name, lattice_job_name,
                    model_train_job_name, model_epoch=500, corpus_key='train-other-960',
                    num_inputs=50, num_classes=12001, target='classes',
                    num_epochs=200,
                    epoch_split=100, feature_name='gammatone', align_name='align_hmm',
                    new_feature_scorer_args=None, **kwargs):
    pass
    
    crnn_feature_scorer_args = copy.deepcopy(self.crnn_feature_scorer_args)
    if new_feature_scorer_args:
      crnn_feature_scorer_args.update(new_feature_scorer_args)

    feature_scorer = self.get_crnn_feature_scorer(
      '',
      corpus_key,
      model_train_job_name, model_epoch,
      **crnn_feature_scorer_args)

    self.generate_lattices(lattice_job_name, corpus_key, feature_name, feature_scorer)


    lattice_archive = rasr.util.ArchiverJob(mode='combine',
                                              path=self.jobs[corpus_key][
                                                'state-accuracy_%s' % lattice_job_name].lattice_bundle).out_archive
    segment_wise_alignment_archive = rasr.util.ArchiverJob(mode='combine',
                                                             path=self.jobs[corpus_key]['state-accuracy_%s' % lattice_job_name]. \
                                                             segmentwise_alignment_bundle).out_archive

    if kwargs.get('csp_args', None):
      csp_for_accuracy = copy.deepcopy(self.crp[corpus_key])
      setattr(csp_for_accuracy, key, kwargs['csp_args'][key])
    else:
      csp_for_accuracy = self.crp[corpus_key]

    # ----------------------------------------------------------------
    crnn_config = copy.deepcopy(self.jobs[corpus_key][model_train_job_name].returnn_config)
    # Adapt config for sequence training
    crnn_config, additional_rasr_config_files, additional_rasr_post_config_files = add_accuracy_output(
      csp=csp_for_accuracy, crnn_config=crnn_config,
      accuracy_lattices=lattice_archive,
      segment_wise_alignment=segment_wise_alignment_archive,
      feature_scorer=feature_scorer, feature_flow=self.feature_flows[corpus_key][feature_name],
      import_model=self.jobs[model_train_job_name].models[model_epoch].model)
    crnn_config["batch_size"] = crnn_config["batch_size"] // 2

    # crnn_config can still be updated through kwargs['crnn_config']
    crnn_config.update(kwargs.pop('crnn_config',
                                  {'learning_rate': 1e-5,
                                   'max_seq_length': 1600
                                   }))

    # ---------------------- smbr training job nn_train_args
    new_nn_train_args = {'replace_crnn_config': True,
                         'mem_rqmt': 16}
    new_nn_train_args.update(kwargs)

    new_nn_train_args['returnn_config'] = ReturnnConfig(crnn_config)

    new_nn_train_args['additional_rasr_config_files'] = additional_rasr_config_files
    new_nn_train_args['additional_rasr_post_config_files'] = additional_rasr_post_config_files


    # network is already setup in the new_nn_train_args
    self.nn_train(name=train_job_name, num_epochs=num_epochs,
                  network=None, train_corpus_key=corpus_key,
                  num_inputs=num_inputs, num_classes=num_classes, target=target,
                  epoch_split=epoch_split, feature_name=feature_name, align_name=align_name,
                  **new_nn_train_args)

  # generate gammatone_features
  def gammatone_feature(self, corpus_key, name, **kwargs):

    self.jobs[corpus_key]['gt_features_' + name] = f = features.GammatoneJob(self.crp[corpus_key], **kwargs)
    if "gt_options" in kwargs and "channels" in kwargs.get("gt_options"):
      f.add_alias('%s_gt_%i_features_%s' % (corpus_key, kwargs.get("gt_options").get("channels"), name))
    else:
      f.add_alias('%s_gt_features_%s' % (corpus_key, name))

    self.feature_caches[corpus_key]['gt_features_' + name] = f.feature_path['gt']
    self.feature_bundles[corpus_key]['gt_features_' + name] = f.feature_bundle['gt']

    feature_path = rasr.FlagDependentFlowAttribute(
      'cache_mode', {
      #'task_dependent': self.feature_caches[corpus_key]['gt_features_' + name],
      'bundle': self.feature_bundles[corpus_key]['gt_features_' + name]})

    self.feature_flows[corpus_key]['gt_features_' + name] = features.basic_cache_flow(feature_path)
    #self.feature_flows[corpus_key]['uncached_gt_features_' + name] = f.feature_flow

  # from zhou
  def bundle_to_cache(self, bundle_path, pattern='cache'):
    with open(bundle_path, 'r') as f:
      cacheL = f.read().split()
    cacheD = {}
    for idx in range(len(cacheL)):
      cacheD[idx+1] = Path(cacheL[idx], cached=True)
    temp = cacheL[0].replace('%s.1' %pattern,'%s.$(TASK)' %pattern)
    return util.MultiPath(temp, cacheD, cached=True)

  # Todo: modification for 'task_dependent'
  def register_feature(self, corpus_key, fname, bundle_path):

    assert corpus_key in self.corpora
    assert fname not in self.feature_bundles[corpus_key], 'overwrite existing feature!'
    #
    # args = {#{'task_dependent' : self.bundle_to_cache(bundle_path),
    #        'bundle': Path(bundle_path, cached=True)}

    feature_path = rasr.FlagDependentFlowAttribute('cache_mode',
                                                     {'bundle': Path(bundle_path, cached=True)})

    self.feature_flows[corpus_key][fname] = features.basic_cache_flow(feature_path)


  def register_alignment(self, corpus_key, align_name, bundle_path):
    assert corpus_key in self.corpora
    assert align_name not in self.alignments[corpus_key].keys(), 'overwrite existing alignment !'

    args = {#{'task_dependent' : self.bundle_to_cache(bundle_path),
            'bundle': Path(bundle_path, cached=True)}
    self.alignments[corpus_key][align_name] = rasr.FlagDependentFlowAttribute('cache_mode', args)

    #print(self.alignments[corpus_key][align_name].alternatives)


