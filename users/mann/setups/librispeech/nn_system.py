from sisyphus import *
Path = setup_path(__package__)

import copy

# -------------------- Recipes --------------------

from i6_experiments.users.mann.setups.nn_system.base_system import NNSystem

# -------------------- System --------------------

class LibriNNSystem(NNSystem):
  def __init__(self, language='en', **kwargs):
    default_kwargs = {
      'epochs': [12, 24, 32, 80, 160],
      'num_input': 50,
      **kwargs
    }
    kwargs = {**default_kwargs, **kwargs}
    super().__init__(**kwargs)

    self.csp['base'].python_home                   = None
    self.csp['base'].python_program_name           = None

    # self.corpora           = {}
    # self.corpus_recipe     = librispeech
    # self.corpus_recipe.corpora = copy.deepcopy(librispeech.LibriSpeechCorpora().corpora)
    # self.corpus_recipe.scorers = {c: recog.scoring.Sclite for c in librispeech.stm_path.keys()}
    # self.corpus_recipe.lexicon = copy.deepcopy(librispeech.lexica)
    # self.corpus_recipe.scorer_args = {c: {'ref': librispeech.stm_path[c]} for c in librispeech.stm_path.keys()}
    # self.glm_files             = {}
    # self.lexica                = {}
    # self.cart_trees        = {}
    # self.lda_matrices      = {}

    self._corpus_file = "train-clean-100"
    self.subcorpus_mapping = {'train': self._corpus_file, 
                              'dev': 'dev-clean', 
                              'test-clean': 'test-clean', 
                              'dev-other': 'dev-other', 
                              'test-other': 'test-other'}
    # self.concurrent        = librispeech.concurrent
    self.features          = {'gt'}

    self.gt_options = { 'maxfreq'          : 7500,
                        'channels'         :   50,
                        'do_specint'       : False}

    self.legacy_am_args     = { 'tdp_transition' : (3.0, 0.0,       30.0,  0.0),
                                'tdp_silence'    : (0.0, 3.0, 'infinity', 20.0)  }
    self.am_args  = { 'tdp_transition' : (0.69, 0.69,       30.0,  0.0),
                      'tdp_silence'    : (0.69, 0.69, 'infinity', 20.0)  }
    # self.lm_args     = {'lm_path': librispeech.lms['4gram']} # perplexität: 150
    self.lexica_args = {}
    self.costa_args  = { 'eval_recordings': True, 'eval_lm': False }

    # self.cart_args = {'phonemes': self.corpus_recipe.cart_phonemes,
    #                   'steps': self.corpus_recipe.cart_steps,
    #                   'max_leaves': 4501,
    #                   'hmm_states': 3}
      
    self.cart_lda_args = { 'name'                       : 'mono',
                           'corpus'                     : 'train',
                           'initial_flow'               : 'gt15',
                           'context_flow'               : 'gt',
                           'context_size'               :  15,
                           'alignment'                  : 'train_mono',
                           'num_dim'                    : 40,
                           'num_iter'                   :  1,
                           'eigenvalue_args'            : {},
                           'generalized_eigenvalue_args': {'all' : {'verification_tolerance': 1e14} } }

    self.init_nn_args = {'name': 'crnn',
                         'corpus': 'train',
                         'dev_size': 0.001,
                         'bad_segments': None,
                         'dump': True}

    self.default_nn_training_args = {'feature_corpus': 'train',
                                     'alignment': ('train', 'init_align', -1),
                                     'num_classes': lambda s: {
                                       'monophone':   211,
                                       'cart'     : 12001
                                     }[s.csp['base'].acoustic_model_config.state_tying.type],
                                     'num_epochs': 160,
                                     'train_corpus': 'crnn_train',
                                     'dev_corpus'  : 'crnn_dev',
                                     'partition_epochs': {'train': 8, 'dev' : 1},
                                     'save_interval': 4,
                                     'time_rqmt': 120,
                                     'mem_rqmt' : 24,
                                     'use_python_control': True,
                                     'feature_flow': 'gt'}

    self.default_nn_dump_args =     {'feature_corpus': 'train',
                                     'alignment': ('train', 'init_align', -1),
                                     'num_classes': 12001,
                                     'num_epochs': 1,
                                     'train_corpus': 'crnn_train_dump',
                                     'dev_corpus'  : 'crnn_dev',
                                     'partition_epochs': {'train': 1, 'dev' : 1},
                                     'save_interval': 4,
                                     'time_rqmt': 1,
                                     'mem_rqmt' : 4,
                                     'use_python_control': True,
                                     'feature_flow': 'gt'}

    self.default_scorer_args = {'prior_mixtures': ('train', 'init_mixture'),
                                'prior_scale': 0.70,
                                'feature_dimension': 50}

    self.default_recognition_args = {'corpus': 'dev',
                                     'flow': 'gt',
                                     'pronunciation_scale': 3.0,
                                     'lm_scale': 5.0,
                                     'search_parameters': {'beam-pruning': 18.0, # TODO: 15
                                                           'beam-pruning-limit': 100000,
                                                           'word-end-pruning': 0.5,
                                                           'word-end-pruning-limit': 10000},
                                     'lattice_to_ctm_kwargs' : { 'fill_empty_segments' : True,
                                                                 'best_path_algo': 'bellman-ford' },
                                     'rtf': 50}
    
    self.default_compile_tf_bins = dict(
      crnn_python_exe  = '/u/michel/py2-theano/bin/python2.7',
      crnn_root        = '/u/michel/git_projects/returnn'
    )
    self.default_compile_tf_bins = {}

    # import paths
    self.default_reduced_segment_path = '/work/asr3/michel/mann/misc/tmp/reduced.train.segments'
    self.PREFIX_PATH1K = "/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/"
    self.PREFIX_PATH = "/work/asr3/luescher/setups-data/librispeech/best-model/100h_2019-04-10/"
    self.default_allophones_file = self.PREFIX_PATH + "StoreAllophones.34VPSakJyy0U/output/allophones"
    self.default_alignment_file = Path(self.PREFIX_PATH + "AlignmentJob.Mg44tFDRPnuh/output/alignment.cache.bundle", cached=True)
    self.default_cart_file = Path(self.PREFIX_PATH + "EstimateCartJob.knhvHK9ONIOC/output/cart.tree.xml.gz", cached=True)
    # self.cart_trees['gmm'] = self.default_cart_file
    self.default_mixture_path = Path(self.PREFIX_PATH + "EstimateMixturesJob.accumulate.dctnjFBP8hos/output/am.mix", cached=True)
    self.default_mono_mixture_path = Path("/u/michel/setups/librispeech/work/mm/mixtures/EstimateMixturesJob.accumulate.rNKsxWShoABt/output/am.mix", cached=True)
    self.default_feature_paths = {self._corpus_file: self.PREFIX_PATH + "FeatureExtraction.Gammatone.tp4cEAa0YLIP/output/gt.cache.",
                            'dev-clean' : "/u/michel/setups/librispeech/work/features/extraction/FeatureExtraction.Gammatone.DA0TtL8MbCKI/output/gt.cache.",
                            "test-clean" : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.INa6z5A4JvZ5/output/gt.cache.",
                            "train-other-960": self.PREFIX_PATH1K + "FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.",
                            "dev-other" : self.PREFIX_PATH1K + "FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.",
                            "test-other" : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qqN3kYqQ6QHF/output/gt.cache.",
                          }

    PREFIX_PATH1K = "/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/"
    PREFIX_PATH = "/work/asr3/luescher/setups-data/librispeech/best-model/100h_2019-04-10/"
    
    self.initial_system = {
      "allophones" : PREFIX_PATH + "StoreAllophones.34VPSakJyy0U/output/allophones",
      "alignment"  : Path(PREFIX_PATH + "AlignmentJob.Mg44tFDRPnuh/output/alignment.cache.bundle", cached=True),
      "cart"       : Path(PREFIX_PATH + "EstimateCartJob.knhvHK9ONIOC/output/cart.tree.xml.gz", cached=True),
      "mixture"    : Path(PREFIX_PATH + "EstimateMixturesJob.accumulate.dctnjFBP8hos/output/am.mix", cached=True),
      "mono_mixture": Path("/u/michel/setups/librispeech/work/mm/mixtures/EstimateMixturesJob.accumulate.rNKsxWShoABt/output/am.mix", cached=True),
      "features" : {
        "train-clean-100": PREFIX_PATH + "FeatureExtraction.Gammatone.tp4cEAa0YLIP/output/gt.cache.",
        "dev-clean"      : "/u/michel/setups/librispeech/work/features/extraction/FeatureExtraction.Gammatone.DA0TtL8MbCKI/output/gt.cache.",
        "test-clean"     : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.INa6z5A4JvZ5/output/gt.cache.",
        "train-other-960": PREFIX_PATH1K + "FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.",
        "dev-other"      : PREFIX_PATH1K + "FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.",
        "test-other"     : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qqN3kYqQ6QHF/output/gt.cache.",
      }
    }

    # self.feature_mappings = {cv : {'caches': util.MultiPath(path + "$(TASK)",
    #                                 {i: path + "{}".format(i) for i in range(librispeech.concurrent[cv.split("-")[0]])}, cached=True),
    #                   'bundle': Path(path + "bundle", cached=True)}
    #                 for cv, path in self.default_feature_paths.items()
    #           }

    # self.feature_mappings = {cv : {'caches': util.MultiPath(path + "$(TASK)",
    #                                 {i: path + "{}".format(i) for i in range(librispeech.concurrent[cv.split("-")[0]])}, cached=True),
    #                   'bundle': Path(path + "bundle", cached=True)}
    #                 for cv, path in self.initial_system["features"].items()
    #           }
    # self.alignment = {self._corpus_file: self.default_alignment_file,
    #                   "dev-clean" : None,
    #                   "test-clean" : None,
    #                   "dev-other": None,
    #                   "test-other": None
    #                   }

    # self.corpus_recipe.segment_whitelist = {c: None for c in self.subcorpus_mapping.values()}
    # self.corpus_recipe.segment_blacklist = {c: None for c in self.subcorpus_mapping.values()}

  def run(self, steps='all'):
    if steps == 'all':
      steps = {'init', 'custom', 'nn_init'}

    if 'init' in steps:
      steps.remove('init')
      self.init_corpora()
      self.init_am(**self.am_args)
      self.init_lm(**self.lm_args)
      self.init_lexica(**self.lexica_args)
      self.init_cart_questions(**self.cart_args)
      self.set_initial_system()
      self.set_scorer()
      self.store_allophones('train')
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)
      self.extract_features()

    if 'custom' in steps:
      steps.remove('custom')
      self.init_corpora()
      self.init_am(**self.am_args)
      self.init_lm(**self.lm_args)
      self.init_lexica(**self.lexica_args)
      for c, v in self.subcorpus_mapping.items():
        self.set_initial_system(corpus=c, 
                feature=self.feature_mappings[v], 
                alignment=self.alignment[v],
                prior_mixture=self.initial_system["mixture"],
                cart=self.initial_system["cart"],
                allophones=self.initial_system["allophones"])
      self.set_scorer()
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)
    
    if 'custom_mono' in steps:
      steps.remove('custom_mono')
      self.init_corpora()
      self.init_am(**self.am_args)
      self.init_lm(**self.lm_args)
      self.init_lexica(**self.lexica_args)
      for c, v in self.subcorpus_mapping.items():
        self.set_initial_system(corpus=c, 
                feature=self.feature_mappings[v], 
                alignment=self.alignment[v],
                prior_mixture=self.initial_system["mono_mixture"],
                cart=self.initial_system["cart"],
                allophones=self.initial_system["allophones"],
                state_tying='mono')
      self.set_scorer()
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)

    if 'nn_init' in steps:
      steps.remove('nn_init')
      self.init_nn(**self.init_nn_args)
    
    super().run(steps)

class SelectionFunctor:
  def __init__(self, attr, **kwargs):
    self.data = kwargs
    self.attr = attr

  def __call__(self, system):
    return self.data[getattr(system, self.attr)]

class LibriOtherNNSystem(LibriNNSystem):

  def __init__(self, **kwargs):
    default_kwargs = dict(
      epochs = [120, 240, 320, 800, 1600],
      num_input = 50,
      **kwargs
    )
    super().__init__(**default_kwargs)
    subcorpus_mapping = {
      'train': 'train-other-960',
      'dev'  :       'dev-other',
      'test' :      'test-other',
    }

    PATH_ROOT = "/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/"
    PREFIX_PATH_100h = "/work/asr3/luescher/setups-data/librispeech/best-model/100h_2019-04-10/"
    self.initial_system = {
      "allophones"   : Path(PATH_ROOT + "StoreAllophones.34VPSakJyy0U/output/allophones", cached=True),
      "alignment"    : Path(PATH_ROOT + "AlignmentJob.uPtxlMbFI4lx/output/alignment.cache.bundle", cached=True),
      "cart"         : Path(PATH_ROOT + "EstimateCartJob.tuIY1yeG7XGc/output/cart.tree.xml.gz", cached=True),
      "prior_mixture": dict(
        monophone = Path("/u/michel/setups/librispeech/work/mm/mixtures/EstimateMixturesJob.accumulate.rNKsxWShoABt/output/am.mix", cached=True),
        cart      = Path(PREFIX_PATH_100h + "EstimateMixturesJob.accumulate.dctnjFBP8hos/output/am.mix", cached=True)
      ),
      # "prior_mixture": Path("/u/michel/setups/librispeech/work/mm/mixtures/EstimateMixturesJob.accumulate.rNKsxWShoABt/output/am.mix", cached=True),
    }
    self.default_recognition_args['lm_scale'] = 12.0
    self.subcorpus_mapping = subcorpus_mapping
    self.corpus_recipe.segment_whitelist = {c: None for c in self.subcorpus_mapping.values()}
    self.corpus_recipe.segment_blacklist = {c: None for c in self.subcorpus_mapping.values()}
    self.alignment['train-other-960'] = self.initial_system.pop('alignment')
    self.default_nn_training_args = {
      'feature_corpus': 'train',
      'alignment': ('train', 'init_align', -1),
      'num_classes': SelectionFunctor("state_tying_type", monophone=211, cart=12001),
      'num_epochs': 1600,
      'partition_epochs': {'train': 80, 'dev' : 1},
      'save_interval': 40,
      'time_rqmt': min(120, 24 * max(self.default_epochs)),
      'mem_rqmt' : 24,
      'use_python_control': True,
      'feature_flow': 'gt'
    }
    self.default_recognition_args['lm_scale'] = 5.0
    self.base_crnn_config.update(
      learning_rate_control_min_num_epochs_per_new_lr = 20,
      newbob_learning_rate_decay = 0.7,
      newbob_multi_num_epochs = 80,
      newbob_multi_update_interval = 40
    )
  
  def get_state_tying(self):
    return self.csp['base'].acoustic_model_config.state_tying.type
  
  def set_state_tying(self, value):
    assert value in {'monophone', 'cart'}, "No other state tying types supported yet"
    self.csp['base'].acoustic_model_config.state_tying.type = value
    for c, v in self.subcorpus_mapping.items():
      initial_system = copy.deepcopy(self.initial_system)
      self.set_initial_system(
        corpus=c,
        feature=self.feature_mappings[v], 
        alignment=self.alignment[v],
        state_tying=value, 
        prior_mixture=initial_system.pop('prior_mixture')[value],
        **initial_system
      )
      print("changed")
  
  state_tying_type = property(get_state_tying, set_state_tying)

  def run(self, steps={'mono'}):
    for state_tying in ['monophone', 'mono', 'cart']:
      # state_tying = 'monophone'
      if state_tying in steps:
        steps.remove(state_tying)
        if state_tying == 'mono': state_tying = 'monophone'
        self.init_corpora()
        self.init_am(**self.am_args)
        self.init_lm(**self.lm_args)
        self.init_lexica(**self.lexica_args)
        self.set_scorer()
        for c, v in self.subcorpus_mapping.items():
          initial_system = copy.deepcopy(self.initial_system)
          self.set_initial_system(
            corpus=c,
            feature=self.feature_mappings[v], 
            alignment=self.alignment[v],
            state_tying=state_tying, 
            prior_mixture=initial_system.pop('prior_mixture')[state_tying],
            **initial_system
          )
          self.costa(c, **self.costa_args)
    
    super().run(steps)
    

    

