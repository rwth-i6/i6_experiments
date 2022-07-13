from sisyphus import *
Path = setup_path(__package__)

import copy

# -------------------- Recipes --------------------

import i6_core
recipe = i6_core

# import i6_core.am           as am
# from i6_core import am
import i6_core.cart         as cart
import i6_core.corpus       as corpus_i6_cores
# import i6_core.crnn         as crnn
# import i6_core.features     as features
# import i6_core.lda          as lda
# import i6_core.meta         as meta
# import i6_core.mm           as mm
# import i6_core.sat          as sat
import i6_core.rasr       as sprint
# import i6_core.vtln         as vtln
import i6_core.recognition  as recog
import i6_core.util         as util
# import recipe.discriminative_training.lattice_generation as lg
# import recipe.experimental.mann.custom_jobs.custom_sprint_training as cst
# import recipe.experimental.mann.extractors as extr
from i6_core.meta.system import select_element
from i6_experiments.users.mann.setups.nn_system.base_system import NNSystem
# from recipe.setups.mann.util import Selector


# -------------------- System --------------------

class SWBNNSystem(NNSystem):
  def __init__(self, state_tying='cart', **kwargs):
    super().__init__(**kwargs)

    # self.csp['base'].python_home                   = gs.SPRINT_PYTHON_HOME
    # self.csp['base'].python_program_name           = gs.SPRINT_PYTHON_EXE

    self.corpora           = {}
    # self.corpus_recipe     = swb
    # self.corpus_recipe.lexicon = swb.lexica
    self.glm_files         = {}
    self.lexica            = {}
    self.lda_matrices      = {}
    self.cart_trees        = {}
    self.vtln_files        = {}


    self.subcorpus_mapping = { 'train': 'full', 'dev': 'dev_zoltan', 'eval': 'hub5-01'}
    self.concurrent        = { 'train':   200 , 'dev':           20, 'eval':        20}
    self.features          = { 'mfcc', 'gt' }

    """
    self.gt_options = { 'maxfreq'          : 7500,
                        'channels'         :   50,
                        'do_specint'       : False}
    """
    self.gt_options = { 'maxfreq'          : 3800,
                        'channels'         :   40,
                        'warp_freqbreak'   : 3700,
                        'do_specint'       : False}

    self.silence_params = {'absolute_silence_threshold': 250,
                          'discard_unsure_segments': True,
                          'min_surrounding_silence': 0.1,
                          'fill_up_silence': True,
                          'silence_ratio': 0.25,
                          'silence_threshold': 0.05}
                          
    self.am_args  = { 'tdp_transition' : (0.69, 0.69,       30.0,  0.0),
                      'tdp_silence'    : (0.69, 0.69, 'infinity', 20.0)  }
    self.lm_args     = {}
    self.lexica_args = {}
    self.costa_args  = { 'eval_recordings': True, 'eval_lm': False }
    
    # self.cart_args   = { 'phonemes'   : swb.cart_phonemes,
    #                      'steps'      : swb.cart_steps,
    #                      'max_leaves' : 9001,
    #                      'hmm_states' :    3 }

    self.cart_lda_args = { 'name'                       : 'mono',
                           'corpus'                     : 'train',
                           'initial_flow'               : 'gt',
                           'context_flow'               : 'gt',
                           'context_size'               :  15,
                           'alignment'                  : 'train_mono',
                           'num_dim'                    : 40,
                           'num_iter'                   :  2,
                           'eigenvalue_args'            : {},
                           'generalized_eigenvalue_args': {'all' : {'verification_tolerance': 1e14} } }
    self.default_alignment_logs = ['/work/asr3/michel/setups-data/SWB_sis/' + \
        'mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.log.{id}.gz' \
            .format(id=id) for id in range(1, 201)]

    self.init_nn_args = {
      'name': 'crnn',
      'corpus': 'train',
      'dev_size': 0.05,
      'alignment_logs': True,
    }
    
    # self.num_classes_map = {'cart': self.cart_args['max_leaves'],
    #                         'mono': 136}

    self.default_nn_training_args = {'feature_corpus': 'train',
                                     'alignment': ('train', 'init_align', -1),
                                    #  'num_classes': self.num_classes_map[state_tying],
                                     'num_epochs': 320,
                                     'partition_epochs': {'train': 24, 'dev' : 1},
                                     'save_interval': 4,
                                     'time_rqmt': 120,
                                     'mem_rqmt' : 12,
                                     'use_python_control': True,
                                     'feature_flow': 'gt'}

    self.default_scorer_args = {'prior_mixtures': ('train', 'init_mixture'),
                                'prior_scale': 0.70,
                                'feature_dimension': 40}

    self.default_recognition_args = {'corpus': 'dev',
                                     'flow': 'gt',
                                     'pronunciation_scale': 1.0,
                                     'lm_scale': 10.,
                                     'search_parameters': {'beam-pruning': 16.0,
                                                           'beam-pruning-limit': 100000,
                                                           'word-end-pruning': 0.5,
                                                           'word-end-pruning-limit': 10000},
                                     'lattice_to_ctm_kwargs' : { 'fill_empty_segments' : True,
                                                                 'best_path_algo': 'bellman-ford' },
                                     'rtf': 50}
    # import paths
    self.default_reduced_segment_path = '/u/mann/experiments/librispeech/recipe/setups/mann/librispeech/reduced.train.segments'
    PREFIX_PATH                       = "/work/asr3/michel/setups-data/SWB_sis/"
    PREFIX_PATH_asr4                  = "/work/asr4/michel/setups-data/SWB_sis/"
    self.default_allophones_file      = PREFIX_PATH + "allophones/StoreAllophones.wNiR4cF7cdOE/output/allophones"
    self.default_alignment_file       = Path('/work/asr3/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.cache.bundle', cached=True)
    self.extra_alignment_file         = Path('/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.cache.bundle', cached=True) # gmm
    self.default_alignment_logs = ['/work/asr3/michel/setups-data/SWB_sis/' + \
        'mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.log.{id}.gz' \
            .format(id=id) for id in range(1, 201)]
    self.extra_alignment_logs = [
        f'/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.log.{id}.gz'
        for id in range(1, 201)
    ]
    self.default_cart_file            = Path(PREFIX_PATH + "cart/estimate/EstimateCartJob.Wxfsr7efOgnu/output/cart.tree.xml.gz", cached=True)

    self.default_mixture_path  = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.Fb561bWZLwiJ/output/am.mix",cached=True)
    # self.default_mono_mixture_path = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.6GiPivpCTK2M/output/am.mix", cached=True)
    self.default_mono_mixture_path = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.m5wLIWW876pl/output/am.mix", cached=True)
    self.default_feature_paths = {
          'train': PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.Jlfrg2riiRX3/output/gt.cache.",
          'dev'  : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.dVkMNkHYPXb4/output/gt.cache.",
          'eval' : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.O4lUG0y7lrKt/output/gt.cache."
          }

    initial_system = {
      "allophones": PREFIX_PATH + "allophones/StoreAllophones.wNiR4cF7cdOE/output/allophones",
      "alignment" : {
        'full'      : Path('/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.cache.bundle', cached=True), # gmm
        'dev_zoltan': None,
        'hub5-01'   : None
      },
      "cart"      : Path(PREFIX_PATH + "cart/estimate/EstimateCartJob.Wxfsr7efOgnu/output/cart.tree.xml.gz", cached=True),
      "prior_mixture": {
        "cart"     : Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.Fb561bWZLwiJ/output/am.mix", cached=True),
        "monophone": Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.m5wLIWW876pl/output/am.mix", cached=True),
      },
      "feature": {
        self.subcorpus_mapping[cv] : {
          'caches': util.MultiPath(
            path_template = path + "$(TASK)",
            hidden_paths  = {i: path + "{}".format(i) for i in range(self.concurrent[cv])},
            cached        = True
          ),
          'bundle': Path(path + "bundle", cached=True)
        } if path else None for cv, path in self.default_feature_paths.items()
      }
    }

    # self.initial_system = Selector(**initial_system)

    self.feature_mappings = {
      self.subcorpus_mapping[cv] : {
        'caches': util.MultiPath(path + "$(TASK)",
            {i: path + "{}".format(i) for i in range(self.concurrent[cv])}, cached=True),
        'bundle': Path(path + "bundle", cached=True)}
      if path else None
      for cv, path in self.default_feature_paths.items()
      }

    self.alignment = {
      'full'      : self.default_alignment_file,
      'dev_zoltan': None,
      'hub5-01'   : None
    }

    # self.corpus_recipe.segment_whitelist = {c: None for c in self.subcorpus_mapping.values()}
    # self.corpus_recipe.segment_blacklist = {c: None for c in self.subcorpus_mapping.values()}

  def set_hub5_scorer(self, corpus):
    self.scorers       [corpus] = recog.Hub5Score
    self.scorer_args   [corpus] = { 'ref': self.stm_files[corpus], 'glm': self.glm_files[corpus] }
    self.scorer_hyp_arg[corpus] = 'hyp'

  @tk.block()
  def init_corpora(self):
    for c in self.subcorpus_mapping.keys():
      self.corpora[c] = self.corpus_recipe.corpora[c][self.subcorpus_mapping[c]]
      j = corpus_recipes.SegmentCorpus(self.corpora[c].corpus_file, self.concurrent[c])
      self.set_corpus(c, self.corpora[c], self.concurrent[c], j.segment_path)

      self.jobs[c]['segment_corpus'] = j
      tk.register_output('%s.corpus.gz' % c, self.corpora[c].corpus_file)

    for c in ['dev', 'eval']:
      self.stm_files[c] = swb.stm_path[self.subcorpus_mapping[c]]
      self.glm_files[c] = swb.glm_path[self.subcorpus_mapping[c]]
      tk.register_output('%s.stm' % c, self.stm_files[c])
      tk.register_output('%s.glm' % c, self.glm_files[c])
    self.stm_files['train'] = corpus_recipes.CorpusToStm(self.corpora['train'].corpus_file).stm_path
    tk.register_output('%s.stm' % 'train', self.stm_files['train'])
  
  def init_lm(self, lm_path=None):
    for c in self.subcorpus_mapping.keys():
      if c == 'train':
        continue
      self.csp[c].language_model_config = sprint.SprintConfig()
      self.csp[c].language_model_config.type  = 'ARPA'
      self.csp[c].language_model_config.file  = lm_path if lm_path is not None else self.corpus_recipe.lms['eval']
      self.csp[c].language_model_config.scale = 10.0

  @tk.block()
  def init_lexica(self, train_lexicon=None, eval_lexicon=None, normalize_pronunciation=True):
    if train_lexicon is None:
      train_lexicon = self.corpus_recipe.lexicon['train']
    if eval_lexicon is None:
      eval_lexicon  = self.corpus_recipe.lexicon['eval']

    for c in self.corpora:
      self.csp[c].lexicon_config                         = sprint.SprintConfig()
      if c == 'train':
        self.csp[c].lexicon_config.file                  = train_lexicon
      else:
        self.csp[c].lexicon_config.file                  = eval_lexicon
      self.csp[c].lexicon_config.normalize_pronunciation = normalize_pronunciation

    tk.register_output('train.lexicon.gz', train_lexicon)
    tk.register_output('eval.lexicon.gz',  eval_lexicon)

  @tk.block()
  def init_cart_questions(self, *args, **kwargs):
    self.cart_questions = cart.PythonCartQuestions(*args, **kwargs)

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
      self.set_hub5_scorer('dev')
      self.set_hub5_scorer('eval')
      self.store_allophones('train')
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)

    for state_tying in ['monophone', 'mono', 'cart']:
      # state_tying = 'monophone'
      if state_tying in steps:
        steps.remove(state_tying)
        if state_tying == 'mono': state_tying = 'monophone'
        for c, v in self.subcorpus_mapping.items():
          # initial_system = copy.deepcopy(self.initial_system)
          # print(self.initial_system[v])
          self.set_initial_system(
            corpus=c,
            state_tying=state_tying, 
            **self.initial_system.select(v, state_tying)
          )
    
    if 'custom_align' in steps:
      steps.remove('custom_align')
      self.alignments['train']['init_align'] = self.default_alignment_file
      # self.alignments['train'] = {'init_align': self.default_alignment_file

    if 'custom' in steps:
      steps.remove('custom')
      self.alignments['train']['init_align'] = self.default_alignment_file
      self.init_corpora()
      self.init_am(**self.am_args)
      self.init_lm(**self.lm_args)
      self.init_lexica(**self.lexica_args)
      for c, v in self.subcorpus_mapping.items():
        self.set_initial_system(corpus=c, 
                feature=self.feature_mappings[v], 
                alignment=self.alignment[v],
                prior_mixture=self.default_mixture_path,
                cart=self.default_cart_file,
                allophones=self.default_allophones_file)
      self.set_hub5_scorer('dev')
      self.set_hub5_scorer('eval')
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
                prior_mixture=self.default_mono_mixture_path,
                cart=self.default_cart_file,
                allophones=self.default_allophones_file,
                state_tying='mono')
      self.set_hub5_scorer('dev')
      self.set_hub5_scorer('eval')
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)

    if 'init_nn' in steps:
      steps.remove('init_nn')
      self.init_nn(**self.init_nn_args)
    
    super().run(steps)

