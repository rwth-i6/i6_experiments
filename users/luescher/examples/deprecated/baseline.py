from sisyphus import *
Path = setup_path(__package__)

import copy

# -------------------- Recipes --------------------

import recipe.am           as am
import recipe.cart         as cart
import recipe.corpus       as corpus_recipes
import recipe.corpus.swb1  as swb
import recipe.crnn         as crnn
import recipe.features     as features
import recipe.lda          as lda
import recipe.meta         as meta
import recipe.mm           as mm
import recipe.sat          as sat
import recipe.sprint       as sprint
import recipe.vtln         as vtln
import recipe.recognition  as recog
import recipe.discriminative_training.lattice_generation as lg
from recipe.meta.system import select_element

# -------------------- System --------------------

class SWBSystem(meta.System):
  def __init__(self):
    super().__init__()

    self.csp['base'].python_home                   = gs.SPRINT_PYTHON_HOME
    self.csp['base'].python_program_name           = gs.SPRINT_PYTHON_EXE
    self.required_tf_libraries = '/u/beck/setups/swb1/dependencies/returnn_native_ops/NativeLstm2/9fa3cd7f72/NativeLstm2.so'

    self.corpora           = {}
    self.glm_files         = {}
    self.lexica            = {}
    self.lda_matrices      = {}
    self.cart_trees        = {}
    self.vtln_files        = {}

    self.subcorpus_mapping = { 'train': 'full', 'dev': 'dev_zoltan', 'eval': 'hub5-01'}
    self.concurrent        = { 'train':   200 , 'dev':           20, 'eval':        20}
    self.features          = { 'mfcc', 'gt' }

    self.gt_options = { 'maxfreq'          : 3800,
                        'channels'         :   40,
                        'warp_freqbreak'   : 3700,
                        'do_specint'       : False}

    self.am_args     = { 'tdp_transition' : (3.0, 0.0,       30.0,  0.0),
                         'tdp_silence'    : (0.0, 3.0, 'infinity', 20.0)  }
    self.lm_args     = {}
    self.lexica_args = {}
    self.cart_args   = { 'phonemes'   : swb.cart_phonemes,
                         'steps'      : swb.cart_steps,
                         'max_leaves' : 9001,
                         'hmm_states' :    3 }
    self.costa_args  = { 'eval_recordings': True, 'eval_lm': False }

    self.mono_train_args = { 'feature_energy_flow' : 'energy,mfcc+deriv+norm',
                             'feature_flow'        : 'mfcc+deriv+norm',
                             'align_iter'          : 20,
                             'splits'              : 10,
                             'accs_per_split'      :  2 }
    self.mono_rec_args   = { 'name'   : 'mono',
                             'iters'  : [8, 10],
                             'corpus' : 'dev',
                             'feature_flow'   : 'mfcc+deriv+norm',
                             'feature_scorer' : ('train', 'train_mono'),
                             'pronunciation_scale':  1.0,
                             'lm_scale'           : 10.0,
                             'search_parameters' :  { 'beam-pruning'           :   15.0,
                                                      'beam-pruning-limit'     : 100000,
                                                      'word-end-pruning'       :   0.5,
                                                      'word-end-pruning-limit' :  15000 },
                             'rtf' : 25,
                             'mem' :  4, }

    self.cart_lda_args = { 'name'                       : 'mono',
                           'corpus'                     : 'train',
                           'initial_flow'               : 'mfcc+deriv+norm',
                           'context_flow'               : 'mfcc',
                           'context_size'               :  9,
                           'alignment'                  : 'train_mono',
                           'num_dim'                    : 40,
                           'num_iter'                   :  2,
                           'eigenvalue_args'            : {},
                           'generalized_eigenvalue_args': {'all' : {'verification_tolerance': 1e16} } }

    self.tri_train_args = { 'feature_flow'      : 'mfcc+context+lda',
                            'initial_alignment' : 'train_mono',
                            'splits'            : 10,
                            'accs_per_split'    :  2 }
    self.tri_rec_args   = { 'name'   : 'tri',
                            'iters'  : [8, 10],
                            'corpus' : 'dev',
                            'feature_flow'   : 'mfcc+context+lda',
                            'feature_scorer' : ('train', 'train_tri'),
                            'pronunciation_scale':  1.0,
                            'lm_scale'           : 20.0,
                            'search_parameters' :  { 'beam-pruning'           :   15.0,
                                                     'beam-pruning-limit'     : 100000,
                                                     'word-end-pruning'       :   0.5,
                                                     'word-end-pruning-limit' :  15000 },
                            'parallelize_conversion' : True,
                            'rtf' : 50,
                            'mem' :  4, }
    self.optimize_triphone_am_lm = False

    self.sdm_args = { 'name'         : 'sdm',
                      'corpus'       : 'train',
                      'feature_flow' : 'mfcc+context+lda',
                      'alignment'    : ('train', 'train_tri', -1) }

    self.vtln_feature_flow_args = { 'corpora'      : ['train', 'dev', 'eval'],
                                    'name'         : 'uncached_mfcc+context+lda',
                                    'base_flow'    : 'uncached_mfcc',
                                    'context_size' : 9,
                                    'lda_matrix'   : 'mono' }
    self.vtln_warp_mix_args     = { 'name'           : 'tri',
                                    'corpus'         : 'train',
                                    'feature_flow'   : 'uncached_mfcc+context+lda',
                                    'feature_scorer' : 'estimate_mixtures_sdm',
                                    'alignment'      : 'train_tri',
                                    'splits'         : 8,
                                    'accs_per_split' : 2 }
    self.vtln_features_args     = { 'name'             : 'mfcc+context+lda',
                                    'train_corpus'     : 'train',
                                    'eval_corpus'      : 'dev',
                                    'raw_feature_flow' : 'uncached_mfcc+context+lda',
                                    'vtln_files'       : 'tri'}
    self.vtln_train_args        = { 'initial_alignment' : 'train_tri',
                                    'feature_flow'      : 'mfcc+context+lda+vtln',
                                    'splits'            : 10,
                                    'accs_per_split'    :  2 }
    self.vtln_recognition_args  = { 'name'   : 'vtln',
                                    'iters'  : [8, 10],
                                    'corpus' : 'dev',
                                    'feature_flow'   : 'uncached_mfcc+context+lda+vtln',
                                    'feature_scorer' : ('train', 'train_vtln'),
                                    'pronunciation_scale' :  1.0,
                                    'lm_scale'            : 20.0,
                                    'search_parameters' :  { 'beam-pruning'           :   15.0,
                                                             'beam-pruning-limit'     : 100000,
                                                             'word-end-pruning'       :    0.5,
                                                             'word-end-pruning-limit' :  15000 },
                                    'parallelize_conversion' : True,
                                    'rtf' : 50,
                                    'mem' :  4 }

    self.sat_train_args       = { 'name'           : 'sat',
                                  'corpus'         : 'train',
                                  'feature_cache'  : 'mfcc',
                                  'feature_flow'   : 'mfcc+context+lda',
                                  'cache_regex'    : '^mfcc.*$',
                                  'alignment'      : 'train_tri',
                                  'mixtures'       : 'estimate_mixtures_sdm',
                                  'splits'         : 10,
                                  'accs_per_split' :  2 }
    self.sat_recognition_args = { 'name'     : 'sat',
                                  'iters'    : [8, 10],
                                  'corpus'   : 'dev',
                                  'prev_ctm'       : 'recog_tri.iter-10',
                                  'feature_cache'  : 'mfcc',
                                  'cache_regex'    : '^mfcc.*$',
                                  'feature_flow'   : 'mfcc+context+lda',
                                  'cmllr_mixtures' : ('train', 'estimate_mixtures_sdm'),
                                  'recog_feature_scorer' : ('train', 'train_sat'),
                                  'pronunciation_scale'  :  1.0,
                                  'lm_scale'             : 20.0,
                                  'search_parameters' :  { 'beam-pruning'           :   15.0,
                                                           'beam-pruning-limit'     : 100000,
                                                           'word-end-pruning'       :   0.5,
                                                           'word-end-pruning-limit' :  15000 },
                                  'parallelize_conversion' : True,
                                  'rtf' : 50,
                                  'mem' :  4 }

    self.vtlnsdm_args             = { 'name'         : 'sdm.vtln',
                                      'corpus'       : 'train',
                                      'feature_flow' : 'mfcc+context+lda+vtln',
                                      'alignment'    : 'train_tri' }
    self.vtlnsat_train_args       = { 'name'           : 'vtln-sat',
                                      'corpus'         : 'train',
                                      'feature_cache'  : 'mfcc+context+lda+vtln',
                                      'feature_flow'   : 'mfcc+context+lda+vtln',
                                      'cache_regex'    : '^.*\\+vtln$',
                                      'alignment'      : 'train_tri',
                                      'mixtures'       : 'estimate_mixtures_sdm.vtln',
                                      'splits'         : 10,
                                      'accs_per_split' :  2 }
    self.vtlnsat_recognition_args = { 'name'     : 'vtln-sat',
                                      'iters'    : [8, 10],
                                      'corpus'   : 'dev',
                                      'prev_ctm'      : 'recog_vtln.iter-10',
                                      'feature_cache' : 'mfcc',
                                      'cache_regex'   : '^mfcc.*$',
                                      'feature_flow'  : 'uncached_mfcc+context+lda+vtln',
                                      'cmllr_mixtures' : ('train', 'estimate_mixtures_sdm.vtln'),
                                      'recog_feature_scorer' : ('train', 'train_vtln-sat'),
                                      'pronunciation_scale'  : 1.0,
                                      'lm_scale'             : 20.0,
                                      'search_parameters' :  { 'beam-pruning'           :   15.0,
                                                               'beam-pruning-limit'     : 100000,
                                                               'word-end-pruning'       :    0.5,
                                                               'word-end-pruning-limit' :  15000 },
                                      'parallelize_conversion' : True,
                                      'rtf' : 50,
                                      'mem' :  4 }
    self.optimize_vtlnsat_am_lm = True
    self.init_nn_args = {'name': 'crnn',
                         'corpus': 'train',
                         'dev_size': 0.05,
                         'window_size': 1}

    self.default_nn_training_args = {'feature_corpus': 'train',
                                     'alignment': ('train', 'train_vtln-sat', -1),
                                     'num_classes': self.cart_args['max_leaves'],
                                     'num_epochs': 256,
                                     'partition_epochs': {'train': 8, 'dev' : 1},
                                     'save_interval': 2,
                                     'time_rqmt': 48,
                                     'use_python_control': False,
                                     'feature_flow': 'gt'}

    self.default_scorer_args = {'prior_mixtures': ('train', 'estimate_mixtures_sdm'),
                                'prior_scale': 0.60,
                                'feature_dimension': 40}

    self.default_recognition_args = {'corpus': 'dev',
                                     'flow': 'gt',
                                     'pronunciation_scale': 1.0,
                                     'lm_scale': 10.,
                                     'search_parameters': {'beam-pruning': 16.0,
                                                           'beam-pruning-limit': 100000,
                                                           'word-end-pruning': 0.5,
                                                           'word-end-pruning-limit': 10000},
                                     'rtf': 50}

    self.default_lattice_options = {'concurrent'                : 200,
                                    'lm'                        : {'file' : '#TODO',
                                                                   'type' : 'ARPA', 'scale' : 10.0},
                                    'short_pauses_lemmata'      : ['[SILENCE]','[NOISE]','[VOCALIZED-NOISE]','[LAUGHTER]'],
                                    'numerator_options'         : {},
                                    'raw-denominator_options'   : {},
                                    'denominator_options'       : {}}


  def set_hub5_scorer(self, corpus):
    self.scorers       [corpus] = recog.Hub5Score
    self.scorer_args   [corpus] = { 'ref': self.stm_files[corpus], 'glm': self.glm_files[corpus] }
    self.scorer_hyp_arg[corpus] = 'hyp'

  @tk.block()
  def init_corpora(self):
    for c in ['train', 'dev', 'eval']:
      self.corpora[c] = swb.corpora[c][self.subcorpus_mapping[c]]
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

  def init_am(self, **kwargs):
    self.csp['base'].acoustic_model_config = am.acoustic_model_config(**kwargs)

  def init_lm(self, lm_path=None):
    for c in ['dev', 'eval']:
      self.csp[c].language_model_config = sprint.SprintConfig()
      self.csp[c].language_model_config.type  = 'ARPA'
      self.csp[c].language_model_config.file  = lm_path if lm_path is not None else swb.lms['eval']
      self.csp[c].language_model_config.scale = 10.0

  @tk.block()
  def init_lexica(self, train_lexicon=None, eval_lexicon=None, normalize_pronunciation=True):
    if train_lexicon is None:
      train_lexicon = swb.lexica['train']
    if eval_lexicon is None:
      eval_lexicon  = swb.lexica['eval']

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

  @tk.block()
  def extract_features(self):
    if 'mfcc' in self.features:
      for c in ['train', 'dev', 'eval']:
        self.mfcc_features(c)
        self.energy_features(c)
      self.normalize('train', 'mfcc+deriv', ['train', 'dev', 'eval'])
      self.add_energy_to_features('train', 'mfcc+deriv+norm')
      tk.register_output('train.mfcc.normalization.matrix', self.normalization_matrices['train']['mfcc+deriv+norm'])
    if 'gt' in self.features:
      for c in ['train', 'dev', 'eval']:
        self.gt_features(c, gt_options=self.gt_options)
        #self.energy_features(c)
      #self.add_energy_to_features('train', 'gt')

    unknown_features = set(self.features).difference({'mfcc', 'gt'})
    if len(unknown_features) > 0:
      raise ValueError('Invalid features: %s' % unknown_features)

  @tk.block()
  def monophone_training(self, feature_energy_flow, feature_flow, align_iter, splits, accs_per_split):
    self.linear_alignment('1', 'train', feature_energy_flow)

    action_sequence  = meta.align_and_accumulate_sequence(align_iter, 1, mark_accumulate=False, mark_align=False)
    action_sequence += meta.split_and_accumulate_sequence(splits, accs_per_split) + ['align!']

    self.train(name     = 'mono',
               corpus   = 'train',
               sequence = action_sequence,
               flow     = feature_flow,
               initial_mixtures = meta.select_element(self.mixtures, 'train', 'linear_alignment_1'),
               align_keep_values = { 'default': 5, 'selected': tk.gs.JOB_DEFAULT_KEEP_VALUE })

  def recognition(self, name, iters, corpus, feature_flow, feature_scorer, pronunciation_scale, lm_scale, search_parameters, rtf, mem, **kwargs):
    with tk.block('%s_recognition' % name):
      for it in iters:
        self.recog(name   = '%s.iter-%d' % (name, it),
                   corpus = corpus,
                   flow           = feature_flow,
                   feature_scorer = list(feature_scorer) + [it - 1],
                   pronunciation_scale = pronunciation_scale,
                   lm_scale            = lm_scale,
                   search_parameters = search_parameters,
                   rtf = rtf,
                   mem = mem,
                   **kwargs)
        self.jobs[corpus]['lat2ctm_%s.iter-%d' % (name, it)].rqmt = { 'time': max(self.csp[corpus].corpus_duration / (2. * self.concurrent[corpus]), .5), 'cpu': 1, 'mem': 5 }

  @tk.block()
  def cart_and_lda(self, name, corpus, initial_flow, context_flow, context_size, alignment, num_dim, num_iter, eigenvalue_args, generalized_eigenvalue_args):
    for f in self.feature_flows.values():
      f['%s+context' % context_flow] = lda.add_context_flow(feature_net = f[context_flow],
                                                            max_size    = context_size,
                                                            right       = int(context_size / 2.0))

    cart_lda = meta.CartAndLDA(original_csp                = self.csp[corpus],
                               initial_flow                = self.feature_flows[corpus][initial_flow],
                               context_flow                = self.feature_flows[corpus]['%s+context' % context_flow],
                               alignment                   = meta.select_element(self.alignments, corpus, alignment),
                               questions                   = self.cart_questions,
                               num_dim                     = num_dim,
                               num_iter                    = num_iter,
                               eigenvalue_args             = eigenvalue_args,
                               generalized_eigenvalue_args = generalized_eigenvalue_args)
    self.jobs[corpus]['cart_and_lda_%s' % name] = cart_lda
    self.lda_matrices[name] = cart_lda.last_lda_matrix
    self.cart_trees[name]   = cart_lda.last_cart_tree
    tk.register_output('cart_%s.tree.xml.gz' % name, cart_lda.last_cart_tree)

    for f in self.feature_flows.values():
      f['%s+context+lda' % context_flow] = features.add_linear_transform(f['%s+context' % context_flow], cart_lda.last_lda_matrix)

    for csp in self.csp.values():
      csp.acoustic_model_config.state_tying.type = 'cart'
      csp.acoustic_model_config.state_tying.file = cart_lda.last_cart_tree

  @tk.block()
  def triphone_training(self, feature_flow, initial_alignment, splits, accs_per_split):
    action_sequence =   ['accumulate'] \
                      + meta.align_then_split_and_accumulate_sequence(splits, accs_per_split, mark_align=False) \
                      + ['align!']

    self.train(name     = 'tri',
               corpus   = 'train',
               sequence = action_sequence,
               flow     = feature_flow,
               initial_alignment = meta.select_element(self.alignments, 'train', initial_alignment),
               align_keep_values = { 'default': 5, 'selected': tk.gs.JOB_DEFAULT_KEEP_VALUE })

  def single_density_mixtures(self, name, corpus, feature_flow, alignment):
    self.estimate_mixtures(name        = name,
                           corpus      = corpus,
                           flow        = feature_flow,
                           alignment   = meta.select_element(self.alignments, corpus, alignment),
                           split_first = False)

  def vtln_feature_flow(self, corpora, name, base_flow, context_size=None, lda_matrix=None):
    for corpus in corpora:
      flow = self.feature_flows[corpus][base_flow]
      if context_size is not None:
        flow = lda.add_context_flow(feature_net = flow,
                                    max_size    = context_size,
                                    right       = int(context_size / 2.0))
      if lda_matrix is not None:
        flow = features.add_linear_transform(flow, self.lda_matrices[lda_matrix])
      self.feature_flows[corpus][name] = flow

  @tk.block()
  def vtln_warping_mixtures(self, name, corpus, feature_flow, feature_scorer, alignment, splits, accs_per_split):
    feature_flow      = self.feature_flows[corpus][feature_flow]
    warp = vtln.ScoreFeaturesWithWarpingFactorsJob(csp            = self.csp[corpus],
                                                   feature_flow   = feature_flow,
                                                   feature_scorer = meta.select_element(self.feature_scorers, corpus, feature_scorer),
                                                   alignment      = meta.select_element(self.alignments,      corpus, alignment))
    self.jobs[corpus]['vtln_warping_map_%s' % name] = warp

    seq = meta.TrainWarpingFactorsSequence(self.csp[corpus], None, feature_flow, warp.warping_map, warp.alphas_file,
                                           ['accumulate'] + meta.split_and_accumulate_sequence(splits, accs_per_split))
    self.mixtures[corpus]['vtln_warping_mix_%s' % name] = seq.selected_mixtures
    self.vtln_files[name + '_alphas_file'] = warp.alphas_file
    self.vtln_files[name + '_warping_map'] = warp.warping_map
    self.vtln_files[name + '_mixtures']    = seq.selected_mixtures

  @tk.block()
  def extract_vtln_features(self, name, train_corpus, eval_corpus, raw_feature_flow, vtln_files, **kwargs):
      self.vtln_features(name             = name,
                         corpus           = train_corpus,
                         raw_feature_flow = self.feature_flows[eval_corpus][raw_feature_flow],
                         warping_map      = self.vtln_files[vtln_files + '_warping_map'],
                         **kwargs)
      self.feature_flows[eval_corpus][raw_feature_flow + '+vtln'] = vtln.recognized_warping_factor_flow(self.feature_flows[eval_corpus][raw_feature_flow],
                                                                                                        self.vtln_files[vtln_files + '_alphas_file'],
                                                                                                        self.vtln_files[vtln_files + '_mixtures'][-1])

  @tk.block()
  def vtln_training(self, initial_alignment, feature_flow, splits, accs_per_split):
    action_sequence =   ['accumulate']\
                      + meta.align_then_split_and_accumulate_sequence(splits, accs_per_split) + ['align!']

    self.train(name     = 'vtln',
               corpus   = 'train',
               sequence = action_sequence,
               flow     = feature_flow,
               initial_alignment = self.alignments['train'][initial_alignment][-1],
               align_keep_values = { 'default': 5, 'selected': tk.gs.JOB_DEFAULT_KEEP_VALUE })

  def estimate_cmllr(self, name, corpus, feature_cache, feature_flow, cache_regex, alignment, mixtures, overlay=None):
    speaker_seg      = corpus_recipes.SegmentCorpusBySpeaker(self.corpora[corpus].corpus_file)
    old_segment_path = self.csp[corpus].segment_path.hidden_paths
    mapped_alignment = sprint.MapSegmentsWithBundles(old_segments = old_segment_path,
                                                     cluster_map  = speaker_seg.cluster_map_file,
                                                     files        = alignment.hidden_paths,
                                                     filename     = 'cluster.$(TASK).bundle')
    mapped_features  = sprint.MapSegmentsWithBundles(old_segments = old_segment_path,
                                                     cluster_map  = speaker_seg.cluster_map_file,
                                                     files        = feature_cache.hidden_paths,
                                                     filename     = 'cluster.$(TASK).bundle')
    new_segments     = sprint.ClusterMapToSegmentList(speaker_seg.cluster_map_file)

    overlay = '%s_cmllr_%s' % (corpus, name) if overlay is None else overlay
    self.add_overlay(corpus, overlay)
    self.csp[overlay].segment_path = new_segments.segment_path
    self.replace_named_flow_attr(overlay, cache_regex, 'cache', mapped_features.bundle_path)

    cmllr = sat.EstimateCMLLRJob(csp          = self.csp[overlay],
                                 feature_flow = self.feature_flows[overlay][feature_flow],
                                 mixtures     = mixtures,
                                 alignment    = mapped_alignment.bundle_path,
                                 cluster_map  = speaker_seg.cluster_map_file,
                                 num_clusters = speaker_seg.num_speakers)
    cmllr.rqmt = { 'time': max(self.csp[overlay].corpus_duration / (50 * self.csp[overlay].concurrent), 1.), 'cpu': 1, 'mem': 4 }
    self.feature_flows[corpus]['%s+cmllr' % feature_flow] = sat.add_cmllr_transform(self.feature_flows[corpus][feature_flow],
                                                                                    speaker_seg.cluster_map_file, cmllr.transforms)

    self.jobs[corpus]['segment_corpus_by_speaker'] = speaker_seg
    self.jobs[overlay]['mapped_alignment'] = mapped_alignment
    self.jobs[overlay]['mapped_features']  = mapped_features
    self.jobs[overlay]['new_segments']     = new_segments
    self.jobs[overlay]['cmllr']            = cmllr

  @tk.block()
  def sat_training(self, name, corpus, feature_cache, feature_flow, cache_regex, alignment, mixtures, splits, accs_per_split):
    self.estimate_cmllr(name          = name,
                        corpus        = corpus,
                        feature_cache = meta.select_element(self.feature_caches, corpus, feature_cache),
                        feature_flow  = feature_flow,
                        cache_regex   = cache_regex,
                        alignment     = meta.select_element(self.alignments, corpus, alignment),
                        mixtures      = meta.select_element(self.mixtures  , corpus, mixtures))

    action_sequence = ['accumulate'] + meta.align_then_split_and_accumulate_sequence(splits, accs_per_split, mark_align=False) + ['align!']
    self.train(name     = name,
               corpus   = corpus,
               sequence = action_sequence,
               flow     = '%s+cmllr' % feature_flow,
               initial_alignment = meta.select_element(self.alignments, corpus, alignment),
               align_keep_values = { 'default': 5, 'selected': tk.gs.JOB_DEFAULT_KEEP_VALUE })

  @tk.block()
  def sat_recognition(self, name, iters, corpus, prev_ctm, feature_cache, cache_regex, feature_flow, cmllr_mixtures, recog_feature_scorer, **kwargs):
    recognized_corpus = corpus_recipes.ReplaceTranscriptionFromCtm(self.corpora[corpus].corpus_file,
                                                                   self.ctm_files[corpus][prev_ctm])

    speaker_seg      = corpus_recipes.SegmentCorpusBySpeaker(self.corpora[corpus].corpus_file)

    overlay = '%s_sat' % corpus
    self.add_overlay(corpus, overlay)
    self.csp[overlay].corpus_config      = copy.deepcopy(self.csp[corpus].corpus_config)
    self.csp[overlay].corpus_config.file = recognized_corpus.output_corpus_path
    self.csp[overlay].segment_path       = copy.deepcopy(self.csp[corpus].segment_path)

    self.corpora[overlay]                = copy.deepcopy(self.corpora[corpus])
    self.corpora[overlay].corpus_file    = recognized_corpus.output_corpus_path

    alignment = mm.AlignmentJob(csp            = self.csp[overlay],
                                feature_flow   = self.feature_flows[overlay][feature_flow],
                                feature_scorer = self.default_mixture_scorer(meta.select_element(self.mixtures, corpus, cmllr_mixtures)))

    self.estimate_cmllr(name          = name,
                        corpus        = overlay,
                        feature_cache = meta.select_element(self.feature_caches, corpus, feature_cache),
                        feature_flow  = feature_flow,
                        cache_regex   = cache_regex,
                        alignment     = alignment.alignment_path,
                        mixtures      = meta.select_element(self.mixtures, corpus, cmllr_mixtures),
                        overlay       = overlay)

    self.feature_flows[corpus]['%s+cmllr' % feature_flow] = sat.add_cmllr_transform(self.feature_flows[corpus][feature_flow],
                                                                                    speaker_seg.cluster_map_file,
                                                                                    self.jobs[overlay]['cmllr'].transforms)

    self.recognition(name           = name,
                     iters          = iters,
                     corpus         = corpus,
                     feature_flow   = '%s+cmllr' % feature_flow,
                     feature_scorer = recog_feature_scorer,
                     **kwargs)

  def init_nn(self, name, corpus, dev_size, window_size):
    all_segments = corpus_recipes.SegmentCorpus(self.corpora[corpus].corpus_file, 1)
    new_segments = corpus_recipes.ShuffleAndSplitSegments(segment_file = all_segments.single_segment_files[1],
                                                          split        = { 'train': 1.0 - dev_size, 'dev': dev_size })

    overlay_name = '%s_train' % name
    self.add_overlay(corpus, overlay_name)
    self.csp[overlay_name].concurrent   = 1
    self.csp[overlay_name].segment_path = new_segments.new_segments['train']

    overlay_name = '%s_dev' % name
    self.add_overlay('train', overlay_name)
    self.csp[overlay_name].concurrent   = 1
    self.csp[overlay_name].segment_path = new_segments.new_segments['dev']

    self.jobs[corpus][              'all_segments_%s' % name] = all_segments
    self.jobs[corpus]['shuffle_and_split_segments_%s' % name] = new_segments

    for c in ['train', 'eval']:
      self.feature_flows[c]['mfcc+context%d' % window_size] = lda.add_context_flow(feature_net = self.feature_flows[c]['mfcc'],
                                                                                   max_size    = window_size,
                                                                                   right       = window_size // 2)

    if 'train_corpus' not in self.default_nn_training_args:
      self.default_nn_training_args['train_corpus'] = '%s_train' % self.init_nn_args['name']
    if 'dev_corpus'   not in self.default_nn_training_args:
      self.default_nn_training_args['dev_corpus']   = '%s_dev'   % self.init_nn_args['name']
    if 'feature_flow' not in self.default_nn_training_args:
      self.default_nn_training_args['feature_flow'] = 'mfcc+context%d' % self.init_nn_args['window_size']

    if 'feature_dimension' not in self.default_scorer_args:
      self.default_scorer_args['feature_dimension'] = self.init_nn_args['window_size'] * 16

    if 'flow' not in self.default_recognition_args:
      self.default_recognition_args['flow'] = 'mfcc+context%d' % self.init_nn_args['window_size']

  def get_tf_flow(self, feature_flow, crnn_config, model_path, epoch, output_tensor_name='output/output_batch_major' , append=False):
    config_dict = copy.deepcopy(crnn_config)
    # Replace lstm unit to nativelstm2
    for layer in config_dict['network'].values():
      if layer.get('unit',None) in {'lstmp'}:
        layer['unit']='nativelstm2'
    # set output to log-softmax
    config_dict['network']['output']['class'] = 'linear'
    config_dict['network']['output']['activation'] = 'log_softmax'
    config_dict['target'] = 'classes'
    config_dict['num_outputs']['classes'] = [self.default_nn_training_args['num_classes'], 1]

    model_path = sprint.StringWrapper(model_path+'.{e:03d}'.format(e=epoch), Path(model_path+'.{e:03d}.meta'.format(e=epoch)))
    compile_graph_job = crnn.CompileTFGraphJob(crnn.CRNNConfig(config_dict,{}))
    tf_graph = compile_graph_job.graph

    # Setup TF AM flow node
    tf_flow = sprint.FlowNetwork()
    tf_flow.add_input('input-features')
    tf_flow.add_output('features')
    tf_flow.add_param('id')
    tf_fwd = tf_flow.add_node('tensorflow-forward', 'tf-fwd', {'id': '$(id)'})
    tf_flow.link('network:input-features', tf_fwd + ':features')
    tf_flow.link(tf_fwd + ':log-posteriors', 'network:features')

    tf_flow.config = sprint.SprintConfig()

    tf_flow.config[tf_fwd].input_map.info_0.param_name              = 'features'
    tf_flow.config[tf_fwd].input_map.info_0.tensor_name             = 'extern_data/placeholders/data/data'
    tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name  = 'extern_data/placeholders/data/data_dim0_size'

    tf_flow.config[tf_fwd].output_map.info_0.param_name  = 'log-posteriors'
    tf_flow.config[tf_fwd].output_map.info_0.tensor_name = output_tensor_name

    tf_flow.config[tf_fwd].loader.type               = 'meta'
    tf_flow.config[tf_fwd].loader.meta_graph_file    = tf_graph
    tf_flow.config[tf_fwd].loader.saved_model_file   = model_path

    tf_flow.config[tf_fwd].loader.required_libraries = self.required_tf_libraries

    # interconnect flows
    out_flow = sprint.FlowNetwork()
    base_mapping = out_flow.add_net(feature_flow)
    tf_mapping   = out_flow.add_net(tf_flow)
    out_flow.interconnect_inputs(feature_flow, base_mapping)
    out_flow.interconnect(feature_flow, base_mapping, tf_flow, tf_mapping, {'features': 'input-features'})

    if append:
      concat = out_flow.add_node('generic-vector-f32-concat', 'concat', attr={'timestamp-port':'features'})
      out_flow.link(tf_mapping[tf_flow.get_output_links('features').pop()], concat + ':tf')
      out_flow.link(base_mapping[feature_flow.get_output_links('features').pop()], concat + ':features')
      out_flow.add_output('features')
      out_flow.link(concat, 'network:features')
    else:
      out_flow.interconnect_outputs(tf_flow, tf_mapping)

    return out_flow

  def nn(self, name, training_args, crnn_config, scorer_args, recognition_args, epochs=None, reestimate_prior=False, use_tf_flow=False):
    if epochs is None:
      epochs = [16, 32, 48, 64]

    train_args = dict(**self.default_nn_training_args)
    train_args.update(training_args)
    train_args['name']        = name
    train_args['crnn_config'] = crnn_config

    with tk.block('NN - %s' % name):
      self.train_nn(**train_args)
      self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].add_alias('nn_%s' % name)
      tk.register_output('plot_se_%s.png' % name, self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].plot_se)
      tk.register_output('plot_lr_%s.png' % name, self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].plot_lr)

      for epoch in epochs:
        scorer_name = name + ('-%d' % epoch)

        score_args = copy.deepcopy(self.default_scorer_args)
        score_args.update(scorer_args)
        if not use_tf_flow:
          score_args['name']             = scorer_name
          score_args['corpus']           = train_args['feature_corpus']
          score_args['output_dimension'] = train_args['num_classes']
          score_args['model']            = (train_args['feature_corpus'], name, epoch)
          self.create_nn_feature_scorer(**score_args)


        if reestimate_prior == 'CRNN':
          assert False, "reestimating the priors is pending review, please use reestimate_prior=True instead"
#          score_features = crnn.sprint_training.CRNNSprintComputePriorJob(train_csp        = self.csp[train_args['train_corpus']],
#                                                                          dev_csp          = self.csp[train_args['dev_corpus']],
#                                                                          feature_flow     = self.feature_flows[train_args['feature_corpus']][train_args['feature_flow']],
#                                                                          model            = self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].models[epoch])
#          self.jobs[train_args['feature_corpus']]["compute_prior_%s" % scorer_name] = score_features
#          scorer_name += '-prior'
#          score_args['name'] = scorer_name
#          score_args['prior_file'] = score_features.prior
#          self.create_nn_feature_scorer(**score_args)
        elif reestimate_prior == True:
          feature_corpus = train_args['feature_corpus']
          feature_flow   = train_args['feature_flow']
          feature_scorer = meta.select_element(self.feature_scorers, feature_corpus, scorer_name)
          feature_scorer = copy.deepcopy(feature_scorer)
          feature_scorer.config.priori_scale = 0.0
          score_features = am.ScoreFeaturesJob(csp            = self.csp[feature_corpus],
                                               feature_flow   = self.feature_flows[feature_corpus][feature_flow],
                                               feature_scorer = feature_scorer,
                                               normalize      = True,
                                               plot_prior     = True,
                                               rtf            = 5.0)
          tk.register_output("plot_%s_prior.png"%scorer_name, score_features.prior_plot)
          scorer_name += '-prior'
          score_args['name'] = scorer_name
          score_args['prior_file'] = score_features.prior
          self.create_nn_feature_scorer(**score_args)

        recog_args = copy.deepcopy(self.default_recognition_args)
        if 'search_parameters' in recognition_args:
          recog_args['search_parameters'].update(recognition_args['search_parameters'])
          remaining_args = copy.copy(recognition_args)
          del remaining_args['search_parameters']
          recog_args.update(remaining_args)
        else:
          recog_args.update(recognition_args)
        recog_args['name']           = 'crnn-%s-%d%s' % (name, epoch, '-prior' if reestimate_prior else '')
        if use_tf_flow:
          feature_flow_net = meta.system.select_element(self.feature_flows, recog_args['corpus'], recog_args['flow'])
          model_path = self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].crnn_config.post_config['model']
          tf_flow = self.get_tf_flow(feature_flow_net, crnn_config, model_path, epoch)
          recog_args['flow'] = tf_flow
          recog_args['feature_scorer'] = scorer = sprint.PrecomputedHybridFeatureScorer(prior_mixtures=select_element(self.mixtures, train_args['feature_corpus'], score_args['prior_mixtures']),
                                                                               priori_scale=score_args.get('prior_scale',0.),
                                                                               prior_file=score_features.prior if reestimate_prior else None)
          self.feature_scorers[train_args['feature_corpus']][scorer_name] = scorer
        else:
          recog_args['feature_scorer'] = (train_args['feature_corpus'], scorer_name)
        with tk.block('recog-%d' % epoch):
          self.recog_and_optimize(**recog_args)

  @tk.block()
  def generate_lattices(self, name, corpus, feature_scorer, feature_flow, lattice_options={}):

    lattice_opt = dict(**self.default_lattice_options)
    lattice_opt.update(lattice_options)

    # New overlay with more segments per corpus and new lm
    self.add_overlay(corpus,'%s_lattice' % corpus)

    if lattice_opt['concurrent'] != self.csp[corpus].concurrent:
      lattice_segments = corpus_recipes.SegmentCorpus(self.corpora[corpus].corpus_file, lattice_opt['concurrent'])
      self.jobs[corpus]['segment_%s_lattices' % corpus] = lattice_segments
      self.csp['%s_lattice' % corpus].concurrent = lattice_opt['concurrent']
      self.csp['%s_lattice' % corpus].segment_path = lattice_segments.segment_path
      flow = copy.deepcopy(feature_flow)
      flow.flags['cache_mode'] = 'bundle'
    else:
      flow = feature_flow

    if 'lm'in lattice_opt:
      if isinstance(lattice_opt['lm'],sprint.SprintConfig):
          self.csp['%s_lattice' % corpus].language_model_config = lattice_opt['lm']
      else:
        self.csp['%s_lattice' % corpus].language_model_config = sprint.SprintConfig()
        for key, val in lattice_opt['lm'].items():
            self.csp['%s_lattice' % corpus].language_model_config[key]  = val

    if 'lexicon' in lattice_opt:
      self.csp['%s_lattice' % corpus].lexicon_config = sprint.SprintConfig()
      for key, val in lattice_opt['lexicon'].items():
        self.csp['%s_lattice' % corpus].lexicon_config[key]  = val



    # Numerator
    num = lg.NumeratorLatticeJob(csp               = self.csp['%s_lattice' % corpus],
                                 feature_flow      = flow,
                                 feature_scorer    = meta.select_element(self.feature_scorers, corpus, feature_scorer),
                                 **lattice_opt['numerator_options'])

    self.jobs           [corpus]['numerator_%s' % name] = num
    self.lattice_bundles[corpus]['numerator_%s' % name] = num.lattice_bundle
    self.lattice_caches [corpus]['numerator_%s' % name] = num.single_lattice_caches

    # Raw Denominator
    rawden = lg.RawDenominatorLatticeJob(csp                    = self.csp['%s_lattice' % corpus],
                                         feature_flow             = flow,
                                         feature_scorer           = meta.select_element(self.feature_scorers, corpus, feature_scorer),
                                         **lattice_opt['raw-denominator_options'])

    self.jobs           [corpus]['raw-denominator_%s' % name] = rawden
    self.lattice_bundles[corpus]['raw-denominator_%s' % name] = rawden.lattice_bundle
    self.lattice_caches [corpus]['raw-denominator_%s' % name] = rawden.single_lattice_caches

    # Denominator
    den = lg.DenominatorLatticeJob(csp = self.csp['%s_lattice' % corpus],
                                   raw_denominator_path=rawden.lattice_path,
                                   numerator_path=num.lattice_path,
                                   **lattice_opt['denominator_options'])
    self.jobs           [corpus]['denominator_%s' % name] = den
    self.lattice_bundles[corpus]['denominator_%s' % name] = den.lattice_bundle
    self.lattice_caches [corpus]['denominator_%s' % name] = den.single_lattice_caches

    # State Accuracy
    stat = lg.StateAccuracyJob(csp = self.csp['%s_lattice' % corpus],
                             feature_flow=flow,
                             feature_scorer=meta.select_element(self.feature_scorers, corpus, feature_scorer),
                             denominator_path=den.lattice_path)

    self.jobs           [corpus]['state-accuracy_%s' % name] = stat
    self.alignments     [corpus]['state-accuracy_%s' % name] = stat.segmentwise_alignment_bundle
    self.lattice_bundles[corpus]['state-accuracy_%s' % name] = stat.lattice_bundle
    self.lattice_caches [corpus]['state-accuracy_%s' % name] = stat.single_lattice_caches

    # Phone Accuracy
    phon = lg.PhoneAccuracyJob(csp = self.csp['%s_lattice' % corpus],
                             feature_flow=flow,
                             feature_scorer=meta.select_element(self.feature_scorers, corpus, feature_scorer),
                             denominator_path=den.lattice_path,
                             short_pauses= lattice_opt['short_pauses_lemmata'])

    self.jobs           [corpus]['phone-accuracy_%s' % name] = phon
    self.alignments     [corpus]['phone-accuracy_%s' % name] = phon.segmentwise_alignment_bundle
    self.lattice_bundles[corpus]['phone-accuracy_%s' % name] = phon.lattice_bundle
    self.lattice_caches [corpus]['phone-accuracy_%s' % name] = phon.single_lattice_caches

  def generate_seqtrain_config(self, name, corpus, feature_scorer, feature_flow, num_classes, returnn_config,
                               import_model=None,
                               learning_rate=0.000005, lattice_options={}, reestimate_prior=False, prior_scale=0.0,
                               prior_file=None, lm_file=None, lm_scale=10., am_scale=1., arc_scale=None,
                               criterion='smbr', ce_smoothing=0.1, num_sprint_instances=1,
                               silence_weight=1.0, discard_silence=False, efficient=True):

    # create overlay
    self.add_overlay(corpus, '%s_seq' % corpus)
    lm_config = sprint.SprintConfig()
    if lm_file is None:
      lm_config.file = self.corpus_recipe.lm['train']
    else:
      lm_config.file = lm_file
    lm_config.type = 'ARPA'
    lm_config.scale = lm_scale
    self.csp['%s_seq' % corpus].language_model_config = lm_config

    # generate lattices for sequence training
    if not criterion.startswith('lf_'):
      lattice_opts = {'lm': lm_config}
      lattice_opts.update(lattice_options)
      self.generate_lattices(name, corpus, feature_scorer, feature_flow, lattice_opts)

    # add prior file (this is bad for SWB but maybe helpful for Quaero)
    if reestimate_prior:
      feature_scorer = copy.deepcopy(feature_scorer)
      feature_scorer.config.priori_scale = 0.0
      score_features = am.ScoreFeaturesJob(csp=self.csp[corpus],
                                           feature_flow=feature_flow,
                                           feature_scorer=feature_scorer,
                                           normalize=True,
                                           plot_prior=True,
                                           rtf=20.0)

      feature_scorer.config.priori_scale = prior_scale
      feature_scorer.config.prior_file = score_features.prior
      tk.register_output('%s.prior' % name, score_features.prior)
    elif prior_file is not None:
      feature_scorer.config.priori_scale = prior_scale
      feature_scorer.config.prior_file = prior_file

    if criterion in ['smbr', 'mmi', 'frameMMI']:
      lattices = self.lattice_bundles[corpus]['state-accuracy_%s' % name]
      alignments = self.alignments[corpus]['state-accuracy_%s' % name]
    elif criterion == 'mpe':
      lattices = self.lattice_bundles[corpus]['phone-accuracy_%s' % name]
      alignments = self.alignments[corpus]['phone-accuracy_%s' % name]
    elif criterion in ['lf_smbr', 'lf_mmi']:
      pass
    else:
      assert False, "only (lf_)mmi, (lf_)smbr, mpe and frameMMI training supportet"

    returnn_config_seq = copy.deepcopy(returnn_config)
    returnn_config_seq["learning_rate"] = learning_rate
    from recipe.experimental.michel.sequence_training import add_accuracy_output, add_mmi_output, add_lfmmi_output

    if criterion == 'mmi':
      returnn_config_seq, add_sprint_conf, add_print_post_conf = add_mmi_output(self.csp['%s_seq' % corpus],
                                                                                returnn_config_seq, lattices, alignments,
                                                                                feature_scorer, feature_flow, num_classes,
                                                                                ce_smoothing,
                                                                                arc_scale=arc_scale,
                                                                                import_model=import_model,
                                                                                num_sprint_instances=num_sprint_instances)
    elif criterion in ['smbr', 'mpe', 'frameMMI']:
      returnn_config_seq, add_sprint_conf, add_print_post_conf = add_accuracy_output(self.csp['%s_seq' % corpus],
                                                                                     returnn_config_seq,
                                                                                     lattices, alignments,
                                                                                     feature_scorer, feature_flow,
                                                                                     num_classes,
                                                                                     criterion=criterion,
                                                                                     import_model=import_model,
                                                                                     ce_smoothing=ce_smoothing,
                                                                                     arc_scale=arc_scale,
                                                                                     num_sprint_instances=num_sprint_instances)
    elif criterion in ['lf_mmi', 'lf_smbr']:
      # TODO: implement
      searchgraph = lm_file
      returnn_config_seq, add_sprint_conf, add_print_post_conf = add_lfmmi_output(self.csp['%s_seq' % corpus],
                                                                                  returnn_config_seq, searchgraph,
                                                                                  criterion, am_scale,
                                                                                  ce_smoothing, import_model=import_model,
                                                                                  prior_file=tk.uncached_path(
                                                                                    score_features.prior) if reestimate_prior else prior_file,
                                                                                  prior_scale=prior_scale,
                                                                                  silence_weight=silence_weight,
                                                                                  discard_silence=discard_silence,
                                                                                  efficient=efficient)
      if criterion == 'lf_mmi':
        returnn_config_seq['network']['output_lfmmi']['loss_opts']['frame_rejection_threshold'] = 1e-6
        returnn_config_seq['network']['output_lfmmi']['loss_scale'] *= -1

    return returnn_config_seq, add_sprint_conf, add_print_post_conf


  def run(self, steps='all'):
    if steps == 'all':
      steps = {'init', 'monophone', 'triphone', 'vtln', 'sat', 'vtln+sat', 'nn'}

    if 'init' in steps:
      self.init_corpora()
      self.init_am(**self.am_args)
      self.init_lm(**self.lm_args)
      self.init_lexica(**self.lexica_args)
      self.init_cart_questions(**self.cart_args)
      self.set_hub5_scorer('dev')
      self.set_hub5_scorer('eval')
      self.store_allophones('train')
      for c in ['train', 'dev', 'eval']:
        self.costa(c, **self.costa_args)
      self.extract_features()

    if 'monophone' in steps:
      self.monophone_training(**self.mono_train_args)
      self.recognition(**self.mono_rec_args)

    if 'cart' in steps:
      self.cart_and_lda(**self.cart_lda_args)

    if 'triphone' in steps:
      self.cart_and_lda(**self.cart_lda_args)
      self.triphone_training(**self.tri_train_args)
      self.recognition(**self.tri_rec_args)
      if self.optimize_triphone_am_lm:
        self.optimize_am_lm(name             = 'recog_tri.iter-10',
                            corpus           = 'dev',
                            initial_am_scale = self.tri_rec_args['pronunciation_scale'],
                            initial_lm_scale = self.tri_rec_args['lm_scale'])

    if 'vtln' in steps or 'sat' in steps or 'sdm' in steps:
      self.single_density_mixtures(**self.sdm_args)

    if 'vtln' in steps:
      self.vtln_feature_flow(**self.vtln_feature_flow_args)
      self.vtln_warping_mixtures(**self.vtln_warp_mix_args)
      self.extract_vtln_features(**self.vtln_features_args)
      self.vtln_training(**self.vtln_train_args)
      self.recognition(**self.vtln_recognition_args)

    if 'sat' in steps:
      self.sat_training(**self.sat_train_args)
      self.sat_recognition(**self.sat_recognition_args)

    if 'vtln+sat' in steps:
      self.single_density_mixtures(**self.vtlnsdm_args)
      self.sat_training(**self.vtlnsat_train_args)
      self.sat_recognition(**self.vtlnsat_recognition_args)
      if self.optimize_vtlnsat_am_lm:
        self.optimize_am_lm(name             = 'recog_vtln-sat.iter-10',
                            corpus           = 'dev',
                            initial_am_scale = self.vtlnsat_recognition_args['pronunciation_scale'],
                            initial_lm_scale = self.vtlnsat_recognition_args['lm_scale'])
        self.jobs['dev']['optimize_recog_vtln-sat.iter-10'].rqmt = { 'time': 48, 'cpu': 1, 'mem': 5 }

    if 'nn' in steps:
      self.init_nn(**self.init_nn_args)
