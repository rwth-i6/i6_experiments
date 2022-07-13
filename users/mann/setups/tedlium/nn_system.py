from sisyphus import *
Path = setup_path(__package__)

import copy

# -------------------- Recipes --------------------

import recipe.am           as am
import recipe.cart         as cart
import recipe.corpus       as corpus_recipes
import recipe.crnn         as crnn
import recipe.corpus.tedlium2 as tedlium
import recipe.features     as features
import recipe.lda          as lda
import recipe.meta         as meta
import recipe.mm           as mm
import recipe.sat          as sat
import recipe.sprint       as sprint
import recipe.vtln         as vtln
import recipe.recognition  as recog
import recipe.util         as util
import recipe.discriminative_training.lattice_generation as lg
import recipe.experimental.mann.custom_jobs.custom_sprint_training as cst
from recipe.meta.system import select_element
from recipe.setups.mann.nn_system.base_system import BaseSystem, NNSystem

# -------------------- System --------------------

class TedliumNNSystem(NNSystem):
  def __init__(self, state_tying, features='logmel', **kwargs):
    print(kwargs)
    super().__init__(**kwargs)

    self.csp['base'].python_home                   = gs.SPRINT_PYTHON_HOME
    self.csp['base'].python_program_name           = gs.SPRINT_PYTHON_EXE

    self.num_classes_map = {'mono': 118, 'carts': 9001}

    self.corpora           = {}
    # tedlium = copy.deepcopy(tedlium)
    self.corpus_recipe     = tedlium
    tedlium.corpora = copy.deepcopy(tedlium.corpora)
    self.corpus_recipe.corpora = {
            'train': {'train': tedlium.corpora['train']},
            'eval' : {
                'dev' : tedlium.corpora[ 'dev'], 
                'test': tedlium.corpora['test']
                }
            }
    self.corpus_recipe.scorers     = {c:          recog.scoring.Sclite for c in tedlium.stm_files.keys()}
    self.corpus_recipe.scorer_args = {c: {'ref': tedlium.stm_files[c]} for c in tedlium.stm_files.keys()}
    self.corpus_recipe.lexicon = tedlium.lexica
    self.glm_files         = {}
    self.lexica            = {}
    self.cart_trees        = {}
    self.lda_matrices      = {}

    self.subcorpus_mapping = {'train': 'train', 'dev': 'dev', 'test': 'test'}
    self.concurrent        = {'train': 50, 'dev': 10, 'test': 10, 'eval':10}
    self.features          = {'gt': 50, 'logmel': 80}
    assert features in self.features
    self.default_feature_flow = features 

    self.gt_options = { 'maxfreq'          : 7500,
                        'channels'         :   50,
                        'do_specint'       : False}

    self.legacy_am_args     = { 'tdp_transition' : (3.0, 0.0,       30.0,  0.0),
                                'tdp_silence'    : (0.0, 3.0, 'infinity', 20.0)  }
    self.am_args  = { 'tdp_transition' : (0.69, 0.69,       30.0,  0.0),
                      'tdp_silence'    : (0.69, 0.69, 'infinity', 20.0)  }
    self.lm_args     = {'lm_path': Path("/work/asr3/zhou/kaldi/egs/tedlium/s5_r2/data/local/local_lm/data/arpa/4gram_small.arpa.gz", cached=True)}
    self.lexica_args = {'eval_lexicon'   : self.corpus_recipe.lexicon['dev']}
    self.costa_args  = {'eval_recordings': True, 'eval_lm': False }
    self.cart_args = {'phonemes': self.corpus_recipe.cart_phonemes,
                      'steps': self.corpus_recipe.cart_steps,
                      'max_leaves': 9001,
                      'hmm_states': 3}

    self.cart_lda_args = { 'name'                       : 'mono',
                           'corpus'                     : 'train',
                           'initial_flow'               : features,
                           'context_flow'               : features,
                           'context_size'               :  15,
                           'alignment'                  : 'train_mono',
                           'num_dim'                    :  80, # 40, auch egal
                           'num_iter'                   :  1,
                           'eigenvalue_args'            : {},
                           'generalized_eigenvalue_args': {'all' : {'verification_tolerance': 1e6} } } # TODO: 1e12,10,6, dem willi doch egal

    self.init_nn_args = {'name'        : 'crnn',
                         'corpus'      : 'train',
                         'dev_size'    : 0.01,
                         'bad_segments': Path('/u/michel/setups/tedlium/dependencies/wei_align_bad.txt')}


    self.default_nn_training_args = {'feature_corpus': 'train',
                                     'alignment': ('train', 'init_align', -1),
                                     'num_classes': self.num_classes_map[state_tying],
                                     'num_epochs': 128,
                                     'partition_epochs': {'train': 8, 'dev' : 1},
                                     'save_interval': 4,
                                     'time_rqmt': 120,
                                     'mem_rqmt' : 24,
                                     'use_python_control': True,
                                     'feature_flow': features}

    self.default_scorer_args = {'prior_mixtures': ('train', 'init_mixture'),
                                'prior_scale': 0.70,
                                'feature_dimension': self.features[features]}

    self.default_recognition_args = {'corpus': 'dev',
                                     'flow': features,
                                     'pronunciation_scale': 3.0,
                                     'lm_scale': 12.,
                                     'search_parameters': {'beam-pruning': 16.0,
                                                           'beam-pruning-limit': 100000,
                                                           'word-end-pruning': 0.5,
                                                           'word-end-pruning-limit': 10000},
                                     'lattice_to_ctm_kwargs' : { 'fill_empty_segments' : True,
                                                                 'best_path_algo': 'bellman-ford' },
                                     'rtf': 50}

    self.default_lattice_options = {'concurrent'                : 200,
                                    'lm'                        : {'type' : 'ARPA', 'scale' : 10.},
                                    'short_pauses_lemmata'      : ['[SILENCE]','[NOISE]','[BREATH]','[LAUGH]','[HESITATION]','[MUSIC]','[RUSTLE]','[THROAT]','[MISPRONOUNCED]'],
                                    'numerator_options'         : {},
                                    'raw-denominator_options'   : {},
                                    'denominator_options'       : {}}

    # import paths
    self.default_reduced_segment_path = '/u/mann/experiments/librispeech/recipe/setups/mann/librispeech/reduced.train.segments'
    # PREFIX_PATH = "/u/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/"
    PREFIX_PATH = "/work/asr4/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/"
    self.default_allophones_file = PREFIX_PATH + "allophones/StoreAllophones.XJxWDielG2RP/output/allophones"
    self.default_alignment_file  = PREFIX_PATH + "mm/alignment/AlignmentJob.G3KckGxFzL3W/output/alignment.cache.bundle"
    self.default_alignment_logs  = [PREFIX_PATH + f"mm/alignment/AlignmentJob.G3KckGxFzL3W/output/alignment.log.{task}.gz" for task in range(1, 51)]
    self.default_cart_file       = PREFIX_PATH + "cart/estimate/EstimateCartJob.rG2TjSvJElRT/output/cart.tree.xml.gz"
    self.default_mixture_path    = '/u/michel/setups/tedlium/dependencies/am.mix'
    self.default_mono_mixture_path = Path("/u/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/mm/mixtures/EstimateMixturesJob.accumulate.IdnUZ7q492pN/output/am.mix", cached=True)
    self.default_feature_paths = {
      'logmel': {
        'train': PREFIX_PATH + "features/extraction/FeatureExtraction.LOGMEL.zX6arCkgNY8y/output/logmel.cache.",
        'dev'  : PREFIX_PATH + "features/extraction/FeatureExtraction.LOGMEL.QSJduoVpxd59/output/logmel.cache.",
        'test' : PREFIX_PATH + "features/extraction/FeatureExtraction.LOGMEL.YIk0t5RHKvYK/output/logmel.cache." },
      'gt': {
        'train': PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.DRxLlzsC7TZ6/output/gt.cache.",
        'dev'  : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.JbJpmSZIZRMc/output/gt.cache.",
        'test' : PREFIX_PATH + "",
      }
    }
    

    self.feature_mappings = {cv : {'caches': util.MultiPath(path + "$(TASK)",
                                    {i: path + "{}".format(i) for i in range(tedlium.concurrent[cv])}, cached=True),
                      'bundle': Path(path + "bundle", cached=True)}
                    for cv, path in self.default_feature_paths[features].items()
              }
    self.alignment = {
            'train': Path(self.default_alignment_file, cached=True),
            'dev'  : None,
            'test' : None
            }

    self.corpus_recipe.segment_whitelist = {c: None for c in self.subcorpus_mapping.values()}
    self.corpus_recipe.segment_blacklist = {c: None for c in self.subcorpus_mapping.values()}

  def run(self, steps='all'):
    if steps == 'all':
      steps = {'init', 'init_logmel', 'nn_init'}

    if 'init' in steps:
      print("Run init")
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

    if 'init_logmel' in steps:
      print("Run init_logmel")
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
                allophones=self.default_allophones_file,
                feature_flow='logmel')
      self.set_scorer()
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)

    if 'custom_mono' in steps:
      print("Run init_mono")
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
                feature_flow=self.default_feature_flow, state_tying='mono')
      self.set_scorer()
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)


    if 'nn_init' in steps:
      print("Run nn_init")
      self.init_nn(**self.init_nn_args)
    
    print(steps)
    super().run(steps=steps)
