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
import i6_core.text as text

# Important corpus definitons from luescher:
from i6_experiments.users.schupp.hybrid_hmm_nn.helpers.librispeech_luescher import CORPUS_PATH, DURATIONS, CONCURRENT

from i6_core.returnn import ReturnnRasrTrainingJob
from i6_core.returnn import ReturnnConfig, ReturnnModel
from i6_core.returnn import CompileTFGraphJob



### from recipe.experimental.michel.sequence_training import add_accuracy_output
from i6_experiments.users.schupp.hybrid_hmm_nn.helpers.specaugment_new import *
from i6_experiments.users.schupp.hybrid_hmm_nn.helpers.helper_functions import *
from i6_experiments.users.schupp.hybrid_hmm_nn.helpers.get_lm_config import *


from recipe.i6_core.corpus.filter import FilterCorpusBySegmentsJob
from recipe.i6_core.corpus.convert import CorpusToStmJob

from sisyphus import *
Path = tk.Path

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
  "train-clean-100", # For some reason this aint there anymore TODO: check why...
  "dev-clean",
  "dev-other",
  "test-clean",
  "test-other",
  "devtrain2000" # Added later hopefully this aint breaking anything
]

# concurrent dict for corpus segmentation
CONCURRENT["devtrain2000"] = 10 # Increased from 1
concurrent = {key: CONCURRENT[key] for key in librispeech_corpora_keys}
concurrent["train-other-960"] = 100

percent_estimated_2000_segs = 0.0072
# duration dict
DURATIONS["devtrain2000"] = DURATIONS["train-other-960"] * percent_estimated_2000_segs
durations = {key: DURATIONS[key] for key in librispeech_corpora_keys}

# librispeech corpora dict
librispeech_corpora = {}


train_corpi_path = Path(os.path.join(chris_960h_best_model_path,"train-merged-960.corpus.gz"), cached=True)
# "/work/asr3/luescher/hiwis/pzheng/librispeech/from-scratch/work/corpus/librispeech/CreateMergedBlissCorpus.g2W17wOMnwgW/output/corpus.xml.gz"
bliss_merged = train_corpi_path
uncompressed_segment_list = Path("/u/schupp/setups/i6_exp_conformer_rtc/devtrain_segments")
filter_segs_corpus = FilterCorpusBySegmentsJob(bliss_merged, uncompressed_segment_list, True).out_corpus

for key in librispeech_corpora_keys:
  librispeech_corpora[key] = tk.Object()

  if key == "train-other-960": # TODO: not sure if this makes sense
    librispeech_corpora[key].corpus_file = Path(os.path.join(chris_960h_best_model_path,
                                                             "train-merged-960.corpus.gz"), cached=True)
  elif key == "train-other-100":
    librispeech_corpora[key].corpus_file = Path(os.path.join(chris_100h_best_model_path,
                                                             "train-clean-100.subcorpus.gz"), cached=True)
  elif key == "devtrain2000":
    librispeech_corpora[key].corpus_file = filter_segs_corpus
  else:
    librispeech_corpora[key].corpus_file = Path(os.path.join(chris_960h_best_model_path, "%s.corpus.gz" % key),
                                                cached=True)

  librispeech_corpora[key].audio_dir = os.path.join(CORPUS_PATH, key)
  librispeech_corpora[key].audio_format = 'wav'
  librispeech_corpora[key].duration = durations[key]


rasr.flow.FlowNetwork.default_flags = {'cache_mode': 'bundle'}

class LibrispeechHybridSystemTim(meta.System):

  def __init__(self):
    super().__init__()

    # init corpus: train|dev|test #
    self.corpus_keys = librispeech_corpora_keys
    self.corpora = librispeech_corpora
    self.concurrent = concurrent
    self.lexicon = lexicon_path

    self.RASR_FLF_TOOL = RASR_FLF_TOOL # TODO: move elsewhere

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

        # Train cross validataion
        self.crp[ck + "_dev"] = rasr.CommonRasrParameters(base=self.crp[ck])
        self.crp[ck + "_dev"].concurrent = 1
        self.crp[ck + "_dev"].segment_path = split_segments.out_segments['dev']

        # devtrain/train-cv -> CE, FER, WER


      # lexicon #
      if lex_file_path is not None:
        lexicon_config = rasr.RasrConfig()
        lexicon_config.file = lex_file_path
        lexicon_config.normalize_pronunciation = norm_pron
        self.crp[ck].lexicon_config = lexicon_config
    
    # Add devtrain

    percent_estimated_2000_segs = 0.0072
    total_train_num_segments = 281241 # 0.0072 * 281241 ~ 2000

    # TODO: we could to the this, but then we would have to also consider 'dev'
    #split_devtrain = corpus_recipes.ShuffleAndSplitSegmentsJob(
    #  segment_file=all_train_segments.out_single_segment_files[1],
    #  split={"devtrain2000" : percent_estimated_2000_segs}
    #)

    devtrain_segments = text.TailJob(
        self.crp["train-other-960_train"].segment_path, 
        num_lines=2000, zip_output=False
    ).out
    
    self.crp["devtrain2000"] = rasr.CommonRasrParameters(base=self.crp["train-other-960_train"])
    self.crp["devtrain2000"].concurrent = 1
    self.crp["devtrain2000"].segment_path = devtrain_segments


    ck = "devtrain2000"
    self.set_corpus(ck, self.corpora[ck], self.concurrent[ck], devtrain_segments)
    if lex_file_path is not None:
      lexicon_config = rasr.RasrConfig()
      lexicon_config.file = lex_file_path
      lexicon_config.normalize_pronunciation = norm_pron
      self.crp[ck].lexicon_config = lexicon_config

    # TODO: finish

  rasr_am_config_is_created = False
  def create_rasr_am_config(self, train_corpus_key):
    self.setup_am_config(train_corpus_key, create_or_update_not_reset=False)
    self.setup_am_config(train_corpus_key, create_or_update_not_reset=True,
                         new_am_config_or_new_am_args=None)
    self.rasr_am_config_is_created = True

  def init_rasr_am_lm_config_recog( self, recog_corpus_key = None):
    am_args = None
    lm_config = None
    sort_files = True

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
  
  def get_precomputed_hybrid_feature_scorer(self, name, train_corpus_key, reuse=False, **kwargs):
    feature_scorer = rasr.PrecomputedHybridFeatureScorer(**kwargs)
    if reuse:
      self.feature_scorers[train_corpus_key]['_'.join([name, 'tf_graph']) if name else 'tf_graph'] = feature_scorer

    return feature_scorer


  def get_feature_flow(cls, base_flow, tf_flow):
    feature_flow = rasr.FlowNetwork()
    base_mapping = feature_flow.add_net(base_flow)
    tf_mapping = feature_flow.add_net(tf_flow)
    feature_flow.interconnect_inputs(base_flow, base_mapping)
    feature_flow.interconnect(base_flow, base_mapping, tf_flow, tf_mapping, {'features': 'input-features'})
    feature_flow.interconnect_outputs(tf_flow, tf_mapping)

    return feature_flow

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


  def register_default_features_and_alignments_for_train_and_dev_copora(self):

    ## train corpora
    # 960h train
    self.register_feature('train-other-960', 'gammatone', os.path.join(
      chris_960h_best_model_path, "FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle"))
    self.register_feature('train-other-960', 'gammatone_unnormalized', os.path.join(
      origin_feature_path, "FeatureExtraction.Gammatone.Pwkx0rfszmwj/output/gt.cache.bundle"))
    self.register_alignment('train-other-960', 'align_hmm',
      os.path.join(chris_960h_best_model_path, "AlignmentJob.uPtxlMbFI4lx/output/alignment.cache.bundle"))

    # devtrain2000 very usure about this one TODO this should realy be only a subset
    self.register_feature('devtrain2000', 'gammatone', os.path.join( chris_960h_best_model_path, "FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle"))
    # TODO I found the stm for train-other-960 here: /work/asr3/luescher/hiwis/pzheng/librispeech/from-scratch/output/create_corpora/stm lets use it

    # For this we need to filter the corpus:

    # FilterCorpusBySegmentsJob
    devtrain_corp = librispeech_corpora['devtrain2000'].corpus_file
    to_stm = CorpusToStmJob(devtrain_corp)
    self.stm_files['devtrain2000'] = to_stm.out_stm_path

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


