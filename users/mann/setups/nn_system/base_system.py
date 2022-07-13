from sisyphus import *
Path = setup_path(__package__)

import copy
import warnings
import math
import os
from collections import ChainMap

# -------------------- Recipes --------------------
import i6_core
recipe = i6_core

import i6_experiments.users.mann.experimental as experimental

import i6_core.am           as am
import i6_core.cart         as cart
import i6_core.corpus       as corpus_recipes
import i6_core.returnn      as crnn
import i6_core.features     as features
import i6_core.lda          as lda
import i6_core.meta         as meta
import i6_core.rasr         as sprint
import i6_core.util         as util
import i6_experiments.users.mann.experimental.helpers as helpers
from i6_experiments.users.mann.experimental.extractors import ExtractStateTyingStats
from i6_core.lexicon.allophones import DumpStateTyingJob

from i6_core.meta.system import select_element
# from i6_core.crnn.multi_sprint_training import PickleSegments

NotSpecified = object()
Default = object()

from collections import namedtuple
trans = namedtuple('Transition', ['forward', 'loop'])

nlog = lambda x: -math.log(x)

def complete_tuple_normal(a, b):
  assert (a is None) != (b is None)
  if a is None:
    assert 0. <= b <= 1.
    return 1-b, b
  assert 0. <= a <= 1.
  return a, 1-a

def probability_to_penalty(prob_tuple):
  return tuple(nlog(x) for x in prob_tuple)

def feature_path(stub):
  path = stub
  return {
    'caches': util.MultiPath(path + "$(TASK)",
        {i: path + "{}".format(i) for i in range(self.concurrent[cv])}, cached=True),
    'bundle': Path(path + "bundle", cached=True)
  }

# -------------------- System --------------------

from i6_experiments.common.setups.rasr import RasrSystem

from dataclasses import dataclass
from typing import Union

@dataclass
class ExpConfig:
  crnn_config: Union[dict, crnn.ReturnnConfig]
  training_args: dict
  plugin_args: dict
  compile_crnn_config: Union[dict, crnn.ReturnnConfig]

class BaseSystem(RasrSystem):
  def __init__(self):
    super().__init__()
    # self.csp['base'].python_home                   = gs.SPRINT_PYTHON_HOME
    # self.csp['base'].python_program_name           = gs.SPRINT_PYTHON_EXE

  def set_scorer(self):
    for c,v in self.subcorpus_mapping.items():
      if c == 'train':
        continue
      self.scorers[c]        = self.corpus_recipe.scorers[v]
      self.scorer_args[c]    = self.corpus_recipe.scorer_args[v]
      self.scorer_hyp_arg[c] = 'hyp'

  def feature_path(self, stub, corpus):
    path = stub
    return {
      'caches': util.MultiPath(path + "$(TASK)",
          {i: path + "{}".format(i) for i in range(self.concurrent[corpus])}, cached=True),
      'bundle': Path(path + "bundle", cached=True)
    }
  
  @property
  def csp(self):
    return self.crp
  
  @csp.setter
  def csp(self, crp):
    self.crp = crp

  @tk.block()
  def init_corpora(self):
    for c in self.subcorpus_mapping.keys():
      if c.split('-')[0] in self.corpus_recipe.corpora:
        corpus_set = c.split('-')[0]
      else: # assume is eval set
        corpus_set = 'eval'
      self.corpora[c] = self.corpus_recipe.corpora[corpus_set][self.subcorpus_mapping[c]]
      j = corpus_recipes.SegmentCorpus(self.corpora[c].corpus_file, self.concurrent[corpus_set])
      self.set_corpus(c, self.corpora[c], self.concurrent[corpus_set], j.segment_path)

      self.jobs[c]['segment_corpus'] = j
      tk.register_output('%s.corpus.gz' % c, self.corpora[c].corpus_file)

  def init_am(self, **kwargs):
    self.csp['base'].acoustic_model_config = am.acoustic_model_config(**kwargs)
  
  def set_transition_probabilities(
      self,
      spf=None, spl=None, sif=None, sil=None,
      speech_transitions=None,
      silence_transitions=None,
      silence_exit=20.0,
      corpus='base'):
    assert (spf or spl) and (sif or sil)
    speech_transition  = probability_to_penalty(complete_tuple_normal(spl, spf)) + ('infinity', 0.0)
    silence_transition = probability_to_penalty(complete_tuple_normal(sil, sif)) + ('infinity', silence_exit)
    self.set_tdps(corpus, speech_transition, silence_transition)

  def set_tdps(self, corpus='train',
               tdp_transition=(3.0, 0.0, 30.0,  0.0),
               tdp_silence=(0.0, 3.0, 'infinity', 20.0)):
    # TODO: mach base zu dev
    if corpus != 'base':
      self.csp[corpus].acoustic_model_config = \
        copy.deepcopy(self.csp['base'].acoustic_model_config)

    config = self.csp[corpus].acoustic_model_config
    config.tdp['*'].loop        = tdp_transition[0]
    config.tdp['*'].forward     = tdp_transition[1]
    config.tdp['*'].skip        = tdp_transition[2]
    config.tdp['*'].exit        = tdp_transition[3]

    config.tdp.silence.loop     = tdp_silence[0]
    config.tdp.silence.forward  = tdp_silence[1]
    config.tdp.silence.skip     = tdp_silence[2]
    config.tdp.silence.exit     = tdp_silence[3]
  
  def get_tdps(self, corpus='train', log=True):
    import math
    f = float
    if log:
      f = lambda x: math.exp(-float(x))
    config = self.csp[corpus].acoustic_model_config
    speech_transitions = (
      config.tdp['*'].loop,
      config.tdp['*'].forward,
      config.tdp['*'].skip,
      config.tdp['*'].exit)
    silence_transitions = (
      config.tdp.silence.loop,
      config.tdp.silence.forward,
      config.tdp.silence.skip,
      config.tdp.silence.exit)
    return tuple(map(f, speech_transitions)), tuple(map(f, silence_transitions))

  def init_lm(self, lm_path=None):
    assert lm_path or any(hasattr(self.corpus_recipe, attr) for attr in ("lm", "lms")), \
      "Could not find language model"
    if lm_path: lm = lm_path
    else:
      lm = self.corpus_recipe.lm if hasattr(self.corpus_recipe, "lm") else self.corpus_recipe.lms
    for c in self.subcorpus_mapping.keys():
      if c == 'train':
        continue
      self.csp[c].language_model_config = sprint.SprintConfig()
      self.csp[c].language_model_config.type  = 'ARPA'
      self.csp[c].language_model_config.file  = lm
      self.csp[c].language_model_config.scale = 12.0

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

  @tk.block()
  def extract_features(self):
    if 'mfcc' in self.features:
      for c in self.subcorpus_mapping.keys():
        self.mfcc_features(c)
        self.energy_features(c)
      self.normalize('train', 'mfcc+deriv', self.subcorpus_mapping.keys())
      self.add_energy_to_features('train', 'mfcc+deriv+norm')
      tk.register_output('train.mfcc.normalization.matrix', self.normalization_matrices['train']['mfcc+deriv+norm'])
    if 'mfcc40' in self.features:
      num_features = 40
      filter_width = 268.258 * 16 / num_features
      mfcc_options = {'filter_width'    : filter_width,
                      'cepstrum_options': {'outputs': num_features}}
      for c in self.subcorpus_mapping.keys():
        self.mfcc_features(c,mfcc_options=mfcc_options,name='mfcc40')
    if 'gt' in self.features:
      for c in self.subcorpus_mapping.keys():
        self.gt_features(c, gt_options=self.gt_options)
    unknown_features = set(self.features).difference({'mfcc', 'mfcc40', 'gt'})
    if len(unknown_features) > 0:
      raise ValueError('Invalid features: %s' % unknown_features)

  @tk.block()
  def extract_nn_features(self, name, corpus, nn_flow, nn_opts=None):
    port_name_mapping = {'features': name}
    job_name = "%s_features" % name
    opts = {'rtf':5., 'mem': 5}
    if nn_opts is not None:
      opts.update(nn_opts)

    self.jobs[corpus]['%s_features' % name] = f = features.FeatureExtraction(self.csp[corpus], nn_flow, port_name_mapping=port_name_mapping, job_name=job_name, **opts)
    f.add_alias('%s_%s_features' % (corpus,name))
    self.feature_caches [corpus][name] = f.feature_path[name]
    self.feature_bundles[corpus][name] = f.feature_bundle[name]

    feature_path = sprint.FlagDependentFlowAttribute('cache_mode', { 'task_dependent' : self.feature_caches [corpus][name],
                                                                     'bundle'         : self.feature_bundles[corpus][name] })
    self.feature_flows[corpus][name] = features.basic_cache_flow(feature_path)
    self.feature_flows[corpus]['uncached_%s' % name] = nn_flow

  def concat_features(self, corpus, names):
    if not type(names) == list:
      names = [names]
    fullname = '+'.join(names)
    flows = [self.feature_flows[corpus][name] for name in names]

    net = sprint.FlowNetwork()
    net.add_param('id')
    net.add_output('features')
    concat = net.add_node('generic-vector-f32-concat', 'concat')
    for i,flow in enumerate(flows):
      mapping = net.add_net(flow)
      net.interconnect_inputs(flow, mapping)
      net.link(mapping[flow.get_output_links('features').pop()], concat+':in%d' % i)

    net.link(concat, 'network:features')

    self.feature_flows[corpus][fullname] = net

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
        # alignment=self.alignment[v],
        state_tying=value, 
        prior_mixture=initial_system.pop('prior_mixture').get(value, None),
        **initial_system
      )
      print("changed")
    
  def num_classes(self):
    state_tying = DumpStateTyingJob(self.csp["train"]).out_state_tying
    tk.register_output("state-tying_mono", state_tying)
    num_states = ExtractStateTyingStats(state_tying).num_states
    return num_states
  
  def set_decoder(self, decoder, **kwargs):
    assert hasattr(decoder, "decode"), "Decoder object must provide 'decode' method"
    self.decoder = decoder
    self.decoder.set_system(self, **kwargs)
  
  def set_trainer(self, trainer, **kwargs):
    assert hasattr(trainer, "train"), "Trainer object must provide 'train' method"
    self.trainer = trainer
    self.trainer.set_system(self, **kwargs)

  def set_initial_system(self, corpus='train', feature=None, alignment=None, 
                         prior_mixture=None, cart=None, allophones=None, 
                         segment_whitelist=None, segment_blacklist=None,
                         feature_flow='gt', state_tying='cart'):
    if feature is None:
      feature = self.corpus_recipe.feature[self.subcorpus_mapping[corpus]]
    #if alignment is None:
    #  alignment = self.corpus_recipe.alignment[self.subcorpus_mapping[corpus]]
    if prior_mixture is None:
      prior_mixture = self.corpus_recipe.mixture[self.subcorpus_mapping[corpus]]
    if cart is None:
      cart = self.corpus_recipe.cart[self.subcorpus_mapping[corpus]]
    if allophones is None:
      allophones = self.corpus_recipe.allophones[self.subcorpus_mapping[corpus]]
    if segment_blacklist is None:
      segment_blacklist = self.corpus_recipe.segment_blacklist.get(self.subcorpus_mapping[corpus], None)
    if segment_whitelist is None:
      segment_whitelist = self.corpus_recipe.segment_whitelist.get(self.subcorpus_mapping[corpus], None)

    if feature is not None:
      #Train Features
      self.feature_caches [corpus][feature_flow] = feature['caches']
      self.feature_bundles[corpus][feature_flow] = feature['bundle']
      feature_path = sprint.FlagDependentFlowAttribute('cache_mode', { 'task_dependent' : self.feature_caches [corpus][feature_flow],
                                                                       'bundle'         : self.feature_bundles[corpus][feature_flow] })

      self.feature_flows[corpus][feature_flow] = features.basic_cache_flow(feature_path)

    self.alignments[corpus]['init_align'] = sprint.NamedFlowAttribute('alignment', alignment)
    self.cart_trees['init_cart'] = cart
    self.mixtures[corpus]['init_mixture'] = prior_mixture

    if segment_blacklist is not None:
      j = corpus_recipes.FilterCorpusBySegments(self.csp[corpus].corpus_config.file,segment_blacklist,invert_match=True,compressed=True)
      j.add_alias("filter_{}_corpus_black".format(corpus))
      self.csp[corpus].corpus_config.file = j.out_corpus
    if segment_whitelist is not None:
      j = corpus_recipes.FilterCorpusBySegments(self.csp[corpus].corpus_config.file,segment_whitelist,invert_match=False,compressed=True)
      j.add_alias("filter_{}_corpus_white".format(corpus))
      self.csp[corpus].corpus_config.file = j.out_corpus

    if state_tying == 'cart':
      for csp in self.csp.values():
        csp.acoustic_model_config.state_tying.type = 'cart'
        csp.acoustic_model_config.state_tying.file = self.cart_trees['init_cart']
        csp.acoustic_model_config.allophones.add_all = False
        csp.acoustic_model_config.allophones.add_from_file = allophones if allophones is not None else ""
        csp.acoustic_model_config.allophones.add_from_lexicon = True

    if state_tying in {'mono', 'monophone'}:
      for csp in self.csp.values():
        csp.acoustic_model_config.state_tying.type = 'monophone'
        csp.acoustic_model_config.allophones.add_all = False
        csp.acoustic_model_config.allophones.add_from_file = allophones if allophones is not None else ""
        csp.acoustic_model_config.allophones.add_from_lexicon = True
  
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
  def triphone_training(self, name, feature_flow, initial_alignment, splits, accs_per_split, **kwargs):
    action_sequence =   ['accumulate'] \
                      + meta.align_then_split_and_accumulate_sequence(splits, accs_per_split, mark_align=False) \
                      + ['align!']

    self.train(name     = name,
               corpus   = 'train',
               sequence = action_sequence,
               flow     = feature_flow,
               initial_alignment = meta.select_element(self.alignments, 'train', initial_alignment),
               align_keep_values = { 'default': 5, 'selected': tk.gs.JOB_DEFAULT_KEEP_VALUE },
               **kwargs)

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
  
  def init_nn(self, name, corpus, dev_size, bad_segments=None, dump=False, alignment_logs=None):
    all_segments = corpus_recipes.SegmentCorpusJob(self.corpora[corpus].corpus_file, 1)
    if alignment_logs is True:
      import glob
      log_path_pattern = os.path.join(
        os.path.dirname(
          select_element(self.alignments, corpus, 'init_align').value.get_path(),
        ),
        "alignment.log.*.gz"
      )
      alignment_logs = glob.glob(log_path_pattern)
    if alignment_logs is not None:
      import i6_experiments.users.mann.experimental.extractors as extr
      all_segments = extr.FilterSegmentsByAlignmentFailures({1: all_segments.single_segment_files[1]}, alignment_logs)
    new_segments = corpus_recipes.ShuffleAndSplitSegmentsJob(segment_file = all_segments.out_single_segment_files[1],
                                                          split        = { 'train': 1.0 - dev_size, 'dev': dev_size })

    overlay_name = '%s_train' % name
    self.segments = {}
    self.add_overlay(corpus, overlay_name)
    self.csp[overlay_name].concurrent   = 1
    self.csp[overlay_name].segment_path = new_segments.out_segments['train']

    overlay_name = '%s_dev' % name
    self.add_overlay('train', overlay_name)
    self.csp[overlay_name].concurrent   = 1
    self.csp[overlay_name].segment_path = new_segments.out_segments['dev']

    self.jobs[corpus][              'all_segments_%s' % name] = all_segments
    self.jobs[corpus]['shuffle_and_split_segments_%s' % name] = new_segments

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

    if dump:
      overlay_name = 'crnn_train_dump'
      self.add_overlay('crnn_train', overlay_name)
      self.csp[overlay_name].concurrent   = 1
      self.csp[overlay_name].segment_path = self.default_reduced_segment_path
  
  @staticmethod
  def adapt_returnn_config_for_recog(config: crnn.ReturnnConfig):
    config_dict = config.config
    unnecessary_params = [
      "chunking", "max_seqs", "batching", "cache_size", "gradient", "learning_rate", "nadam", "newbob"
    ]
    from itertools import product
    for param, key in product(unnecessary_params, list(config_dict)):
      if param in key:
        del config_dict[key]
    # Replace lstm unit to nativelstm2
    for layer in config_dict['network'].values():
      unit = layer.get('unit', None)
      if isinstance(unit, dict):
        continue
      if unit in {'lstmp'}:
        layer['unit']='nativelstm2'
    # set output to log-softmax
    config_dict['network'][adjust_output_layer].update({
      'class': 'linear',
      'activation': 'log_softmax'
    })
    config_dict['target'] = 'classes'
    config_dict['num_outputs']['classes'] = [self.functor_value(self.default_nn_training_args['num_classes']), 1]

  def get_tf_flow(
      self, feature_flow, crnn_config, model,  output_tensor_name=None , 
      append=False, drop_layers=None, adjust_output_layer="output",
      req_libraries=NotSpecified, alias=None, add_tf_flow=True,
      **compile_args
    ):
    if hasattr(crnn_config, "build_compile_config"):
    # if custom_compile:
      config_dict = crnn_config.build_compile_config()
    else:
      config_dict = copy.deepcopy(crnn_config)
    if output_tensor_name is None:
      output_tensor_name = 'output/output_batch_major'
    # Replace lstm unit to nativelstm2
    for layer in config_dict['network'].values():
      unit = layer.get('unit', None)
      if isinstance(unit, dict):
        continue
      if unit in {'lstmp'}:
        layer['unit']='nativelstm2'
    # set output to log-softmax
    if adjust_output_layer \
        and config_dict['network'][adjust_output_layer]['class'] != 'rec':
      config_dict['network'][adjust_output_layer].update({
        'class': 'linear',
        'activation': 'log_softmax'
      })
      # config_dict['network']['output']['class'] = 'linear'
      # config_dict['network']['output']['activation'] = 'log_softmax'
      config_dict['target'] = 'classes'
    config_dict['num_outputs']['classes'] = [self.functor_value(self.default_nn_training_args['num_classes']), 1]
    window = config_dict.get('window', 1)
    if window > 1:
      feature_flow = lda.add_context_flow(feature_flow, max_size=window, right=window // 2)

    model_path = tk.uncached_path(model.model)[:-5]
    if not isinstance(config_dict, crnn.CRNNConfig):
      # print("Not of instance CRNNConfig")
      config_dict = crnn.CRNNConfig(config_dict, {})
    compile_graph_job = crnn.CompileTFGraphJob(
      config_dict, **compile_args
    )
    # print("Prolog: ", getattr(compile_graph_job.crnn_config, "python_prolog", "--"))
    if alias is not None:
      alias = f"compile_crnn-{alias}"
      # print("Compile alias: ", alias)
      compile_graph_job.add_alias(alias)
      self.jobs["dev"][alias] = compile_graph_job
      # print("Alias happened")
    
    if not add_tf_flow:
      return feature_flow

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
    tf_flow.config[tf_fwd].loader.saved_model_file   = sprint.StringWrapper(model_path, model)

    # tf_flow.config[tf_fwd].loader.required_libraries = '/u/beck/setups/swb1/dependencies/returnn_native_ops/NativeLstm2/9fa3cd7f72/NativeLstm2.so'
    if req_libraries is NotSpecified:
      tf_flow.config[tf_fwd].loader.required_libraries = gs.TF_NATIVE_OPS
    elif req_libraries is not None:
      if isinstance(req_libraries, list):
        req_libraries = ':'.join(req_libraries)
      tf_flow.config[tf_fwd].loader.required_libraries = req_libraries

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

  def nn(self, name, training_args, crnn_config, epochs=None):
    if epochs is None:
      epochs = self.default_epochs
      # epochs = [16, 32, 48, 64]

    train_args = dict(**self.default_nn_training_args)
    train_args.update(training_args)
    train_args['name']        = name
    train_args['crnn_config'] = crnn_config

    with tk.block('NN - %s' % name):
      self.train_nn(**train_args)
      self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].add_alias('nn_%s' % name)
      tk.register_output('plot_se_%s.png' % name, self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].plot_se)
      tk.register_output('plot_lr_%s.png' % name, self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].plot_lr)


  def nn_dump(self, name, training_args, crnn_config, checkpoint, debug_config=False, reduced_segments=None, extra_dumps=None, dumps=Default):
    crnn_config = copy.deepcopy(crnn_config)
    if dumps is Default:
      dumps = {'bw': 'fast_bw', 'ce': 'data:classes'}
    elif dumps is None:
      dumps = {}
    # crnn_config['network']['dump_bw'] = {'class': 'hdf_dump', 'filename': 'bw_dumps', 'from': ['fast_bw'], 'is_output_layer': True}
    # crnn_config['network']['dump_ce'] = {'class': 'hdf_dump', 'filename': 'ce_dumps', 'from': ['data:classes'], 'is_output_layer': True}
    if isinstance(extra_dumps, dict):
      dumps.update(extra_dumps)
    for key, layer in dumps.items():
      if layer != 'data:classes' and layer not in crnn_config['network']:
        print(f"Warning BaseSystem.nn_dump: skip layer {layer} for dumping.")
        continue
      crnn_config['network'][f'dump_{key}'] = {'class': 'hdf_dump', 'filename': f'{key}_dumps', 'from': [layer], 'is_output_layer': True}
    crnn_config['learning_rate'] = 0
    train_args = dict(**self.default_nn_training_args)
    train_args.update(training_args)
    train_args.pop('extra_python', None)
    train_args['num_epochs']  = 1
    train_args['partition_epochs'] = {'train': 1, 'dev': 1}
    train_args['name']        = alias = f"dump_{name}"
    train_args['crnn_config'] = crnn_config
    if debug_config:
      crnn_config["max_seqs"] = crnn_config["start_epoch"] = 1
    if reduced_segments: 
      overlay_name = 'crnn_train_dump'
      if not overlay_name in self.csp:
        self.add_overlay('crnn_train', overlay_name)
        self.csp[overlay_name].concurrent   = 1
        self.csp[overlay_name].segment_path = self.default_reduced_segment_path
      if isinstance(reduced_segments, (str, tk.Path)):
        self.csp[overlay_name].segment_path = reduced_segments
      train_args['train_corpus'] = 'crnn_train_dump'
    if checkpoint is not None:
      ts, cs = helpers.get_continued_training(
          system=self,
          training_args=train_args,
          base_config=crnn_config,
          lr_continuation=0,
          base_training=checkpoint,
          copy_mode='preload'
      )
      train_args.update(**ts)
      train_args['crnn_config'] = cs
    self.train_nn(**train_args)
    j = self.jobs[train_args['feature_corpus']]['train_nn_%s' % alias]
    j.add_alias('nn_%s' % alias)
    tk.register_output('dump_nn_{}_learning_rates'.format(name), j.learning_rates)


  def nn_and_recog(self, name, training_args, crnn_config, scorer_args,  
                   recognition_args, epochs=None, reestimate_prior=False, compile_args=None,
                   optimize=True, use_tf_flow=False, label_sync_decoding=False, 
                  #  delayed_build=False,
                   alt_training=False,
                   compile_crnn_config=None, alt_decoding=NotSpecified):
    if compile_args is None:
      compile_args = {}
    if epochs is None:
      epochs = self.default_epochs
    train_args = dict(**self.default_nn_training_args)
    train_args.update(training_args)
    train_args['name']        = name
    train_args['returnn_config'] = crnn_config # if not delayed_build else crnn_config.build()

    with tk.block('NN - %s' % name):
      if alt_training:
        self.trainer.train(**train_args)
      else:
        self.train_nn(**train_args)
      self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].add_alias('nn_%s' % name)
      tk.register_output('plot_se_%s.png' % name, self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].out_plot_se)
      tk.register_output('plot_lr_%s.png' % name, self.jobs[train_args['feature_corpus']]['train_nn_%s' % name].out_plot_lr)

      for epoch in epochs:
        kwargs = locals().copy()
        del kwargs["self"], kwargs["label_sync_decoding"]
        kwargs["training_args"] = kwargs.pop("train_args")
        if alt_decoding is not NotSpecified:
          kwargs["decoding_args"] = kwargs.pop("alt_decoding")
          self.decoder.decode(**kwargs)
          continue
        else:
          assert use_tf_flow, "Otherwise not supported"
          self.decode(_adjust_train_args=False, **kwargs)
          continue

  def decode(
      self, name, epoch,
      crnn_config,
      training_args, scorer_args,
      recognition_args,
      extra_suffix=None,
      recog_name=None,
      compile_args=None,
      compile_crnn_config=None,
      reestimate_prior=False, optimize=True,
      _adjust_train_args=True,
      **_ignored
    ):
    if _adjust_train_args:
      if compile_args is None:
        compile_args = {}
      training_args = {**self.default_nn_training_args, **training_args}
  
    scorer_name = name + ('-%d' % epoch)
    if recog_name is None:
      recog_name = scorer_name
    else:
      recog_name += "-" + str(epoch)
    if extra_suffix is not None:
      recog_name = "-".join([recog_name, extra_suffix])

    score_args = dict(**self.default_scorer_args)
    score_args.update(scorer_args)
    if reestimate_prior=='CRNN':
      num_classes  = self.functor_value(training_args["num_classes"])
      score_features = crnn.sprint_training.CRNNSprintComputePriorJob(
        train_csp    = self.csp[training_args['train_corpus']],
        dev_csp      = self.csp[training_args['dev_corpus']],
        feature_flow = self.feature_flows[training_args['feature_corpus']][training_args['feature_flow']],
        num_classes  = self.functor_value(training_args["num_classes"]),
        alignment    = select_element(self.alignments, training_args["feature_corpus"], training_args["alignment"]),
        model        = self.jobs[training_args['feature_corpus']]['train_nn_%s' % name].models[epoch],
        crnn_python_exe = training_args.get("crnn_python_exe", None),
        crnn_root       = training_args.get("crnn_root", None)
      )
      self.jobs[training_args["feature_corpus"]]["crnn_compute_prior_%s" % scorer_name] = score_features
      scorer_name += '-prior'
      score_args['name'] = scorer_name
      score_args['prior_file'] = score_features.prior
    elif reestimate_prior == True:
      assert False, "reestimating prior using sprint is not yet possible, you should implement it or use 'CRNN'"

    recog_args = copy.deepcopy(self.default_recognition_args)
    # copy feature flow used in training
    recog_args['flow'] = training_args['feature_flow']
    if 'search_parameters' in recognition_args:
      recog_args['search_parameters'].update(recognition_args['search_parameters'])
      remaining_args = copy.copy(recognition_args)
      del remaining_args['search_parameters']
      recog_args.update(remaining_args)
    else:
      recog_args.update(recognition_args)
    recog_args['name']           = 'crnn-%s%s' % (recog_name, '-prior' if reestimate_prior else '')
    feature_flow_net = meta.system.select_element(self.feature_flows, recog_args['corpus'], recog_args['flow'])
    model = self.jobs[training_args['feature_corpus']]['train_nn_%s' % name].models[epoch]
    compile_args = copy.deepcopy(compile_args)
    if compile_crnn_config is None:
      compile_crnn_config = crnn_config
    compile_network_extra_config = compile_args.pop('compile_network_extra_config', {})
    compile_crnn_config['network'].update(compile_network_extra_config)
    if "alias" in compile_args and not isinstance(compile_args['alias'], str):
      compile_args['alias'] = name
    recog_args['flow'] = tf_flow = self.get_tf_flow(
      feature_flow_net, compile_crnn_config, model, 
      recog_args.pop('output_tensor_name', None), 
      drop_layers=recog_args.pop('drop_layers', None),
      req_libraries=recog_args.pop('req_libraries', NotSpecified),
      **compile_args,
    )
    recog_args['feature_scorer'] = scorer = sprint.PrecomputedHybridFeatureScorer(
      prior_mixtures=select_element(self.mixtures, training_args['feature_corpus'], score_args['prior_mixtures']),
      priori_scale=score_args.get('prior_scale', 0.),
      # prior_file=score_features.prior if reestimate_prior else None,
      prior_file=score_args.get("prior_file", None),
      scale=score_args.get('mixture_scale', 1.0)
    )
    self.feature_scorers[training_args['feature_corpus']][scorer_name] = scorer

    # reset tdps for recognition
    with tk.block('recog-V%d' % epoch):
      if optimize:
        self.recog_and_optimize(**recog_args)
      else:
        self.recog(**recog_args)


  def train_nn_multi_alignments(self, name, feature_corpus, train_corpus, dev_corpus, feature_flow, alignments, crnn_config, num_classes, **kwargs):
    dataset_mapping = { 
      key: dict(train_csp    = self.csp[train_corpus],
                dev_csp      = self.csp[dev_corpus],
                feature_flow = self.feature_flows[feature_corpus][feature_flow],
                alignment    = select_element(self.alignments, feature_corpus, alignment)
      )
      for key, alignment in alignments.items()
    }

    j = crnn.CRNNMultiSprintTrainingJob(
      dataset_mapping = dataset_mapping,
      crnn_config     = crnn_config,
      num_classes     = self.functor_value(num_classes),
      **kwargs)
    self.jobs[feature_corpus]['train_nn_%s' % name] = j
    self.nn_models[feature_corpus][name] = j.models
    self.nn_checkpoints[feature_corpus][name] = j.checkpoints
    self.nn_configs[feature_corpus][name] = j.crnn_config_file


  def nn_align(self, nn_name, crnn_config, epoch, scorer_suffix='', use_tf=True, mem_rqmt=8,
    compile_crnn_config=None, name=None, feature_flow='gt', dump=False, **kwargs
    ):
    # get custom tf 
    if compile_crnn_config is None:
      compile_crnn_config = crnn_config
    if use_tf:
      feature_flow_net = meta.system.select_element(self.feature_flows, 'train', feature_flow)
      model = self.jobs['train']['train_nn_%s' % nn_name].models[epoch]
      flow = self.get_tf_flow(feature_flow_net, compile_crnn_config, model)
    else:
      raise NotImplementedError('Without tf not implemented yet.')

    # get alignment
    model_name = '%s-%s' % (nn_name, epoch)
    name = name if name else model_name
    alignment_name = 'alignment_%s' % name
    self.align(
      name=name if not name else name, corpus='train', 
      flow=flow, feature_scorer=self.feature_scorers['train'][model_name + scorer_suffix],
      **kwargs,
    )
    self.jobs['train'][alignment_name].rqmt['mem'] = mem_rqmt
    if dump:
      from recipe.mm import DumpAlignmentJob
      j = DumpAlignmentJob(self.csp['train'], self.feature_flows['train'][feature_flow], self.alignments['train'][name])
      tk.register_output(alignment_name, j.alignment_bundle)
      return j.alignment_bundle
    return None

  def run(self, steps='all'):
    if steps == 'all':
      steps = {'init', 'custom', 'nn_init'}

    if 'init' in steps:
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
      self.set_scorer()
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)
    
    if 'custom_mono' in steps:
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
                state_tying='mono',
                feature_flow=self.default_feature_flow)
      self.set_scorer()
      for c in self.subcorpus_mapping.keys():
        self.costa(c, **self.costa_args)

    if 'nn_init' in steps:
      self.init_nn(**self.init_nn_args)

import i6_experiments.users.mann.nn as chelp
# from recipe.crnn.helpers.mann import tcnn_network
from i6_experiments.users.mann.experimental.sequence_training import add_bw_layer, add_bw_output, add_fastbw_configs
from i6_experiments.users.mann.experimental.util import safe_csp

class Arguments:
  def __init__(self, signature, args, kwargs):
    self.signature = signature
    self.parameters = signature.parameters
    self.args   = list(args)
    self.kwargs = dict(kwargs)
  
  def __getitem__(self, key, default=None):
    if not key in self.parameters:
      return default
    if key in self.kwargs:
      return self.kwargs[key]
    parameter_index = list(self.parameters.keys()).index(key)
    return self.args[parameter_index]
  
  def __setitem__(self, key, value):
    if not key in self.parameters or key in self.kwargs:
      self.kwargs[key] = value
      return
    parameter_index = list(self.parameters.keys()).index(key)
    self.args[parameter_index] = value
    
  def transform(self, key, transformation):
    value = self[key]
    self[key] = transformation(value)

from types import MappingProxyType
Auto = object()

class NNSystem(BaseSystem):
  def __init__(self, num_input=None, epochs=None):
    super().__init__()

    assert num_input and epochs
    self.default_epochs = epochs

    self.nn_config_dicts = {'train': {}}
    self.compile_configs = {}

    self.base_crnn_config = {
      "num_input" : num_input,
      "l2" : 0.01,
      "lr" : 0.00025,
      "dropout" : 0.1,
      "batch_size" : 5000,
      "max_seqs" : 200,
      "nadam" : True,
      "gradient_clip" : 0,
      "learning_rate_control" : "newbob_multi_epoch",
      "learning_rate_control_error_measure" : 'dev_score_output_bw',
      "min_learning_rate" : 1e-6,
      "update_on_device" : True,
      "cache_size" : "0",
      "batching" : "random",
      "chunking" : "50:25",
      "truncation" : -1,
      "gradient_noise" : 0.1,
      "learning_rate_control_relative_error_relative_lr" : True,
      "learning_rate_control_min_num_epochs_per_new_lr" : 4,
      "newbob_learning_rate_decay" : 0.9,
      "newbob_multi_num_epochs" : 8,
      "newbob_multi_update_interval" : 4,
      "optimizer_epsilon" : 1e-08,
      "use_tensorflow" : True,
      "multiprocessing" : True,
      "cleanup_old_models" : {'keep': epochs}
    }

    self.base_lstm_config = {
      "layers": 6 * [512],
    }

    self.base_ffnn_config = {
      "layers": 6 * [1700],
      "num_input": num_input * 15,
      "window": 15
    }

    self.base_viterbi_args = {
      "lr": 0.0008,
      "learning_rates": [ 
        0.0003,
        0.0003555555555555555,
        0.0004111111111111111,
        0.00046666666666666666,
        0.0005222222222222222,
        0.0005777777777777778,
        0.0006333333333333334,
        0.0006888888888888888,
        0.0007444444444444445,
        0.0008]
    }

    self.base_tcnn_config = {
      "layers": 6 * [1700],
      "filters": 5 * [2] + 1 * [1],
      "dilation": [1, 2, 4, 8, 16, 1]
    }

    self.base_bw_args = {
      "am_scale" : 0.7,
      "tdp_scale" : 1.0,
      "prior_scale": 1.0
    }

    self.base_lstm_bw_args = {
      "am_scale" : 1.0,
      "tdp_scale" : 1.0,
      "prior_scale": 0.7, 
    }

    self.base_pretrain_args = {
      "initial_am" : 0.01,
      "final_am": 0.7,
      "final_epoch": 5,
    }

    from i6_experiments.users.mann.setups.tdps import CombinedModelApplicator
    self.plugins = {
      "tdps": CombinedModelApplicator(self),
    }

    from i6_experiments.users.mann.nn.lstm import viterbi_lstm
    from i6_experiments.users.mann.nn import prior, pretrain, bw
    import i6_experiments.users.mann.setups.prior as tp
    def make_bw_lstm(num_input, epochs, num_frames, numpy=False):
      viterbi_config = viterbi_lstm(num_input, epochs)
      bw_config = bw.ScaleConfig.copy_add_bw(
        viterbi_config, self.csp["train"],
        am_scale=0.7, prior_scale=0.035
      )
      priors_txt = tp.get_prior_from_transcription_job(
        self, total_frames=num_frames
      )
      prior.prepare_static_prior(bw_config, prob=True)
      prior.add_static_prior(bw_config, priors_txt)
      return pretrain.PretrainConfigHolder(
          bw_config, rel_prior_scale=0.05
      )

    def make_bw_ffnn(num_input, epochs, numpy=False):
      viterbi_config = viterbi_lstm(num_input, epochs)
      bw_config = bw.ScaleConfig.copy_add_bw(
        viterbi_config, self.csp["train"],
        am_scale=0.7, prior_scale=0.035
      )
      priors = tp.get_prior_from_transcription(self)
      prior.prepare_static_prior(bw_config, prob=True)
      prior.add_static_prior_from_var(bw_config, priors, numpy)
      return pretrain.PretrainConfigHolder(
          bw_config, rel_prior_scale=0.05
      )

    self.baselines = {
      "viterbi_lstm": lambda: viterbi_lstm(num_input, epochs),
      "bw_lstm_fixed_prior": lambda: make_bw_lstm(num_input, epochs),
      "bw_lstm_fixed_prior_new": lambda: make_bw_lstm(num_input, epochs, True),
      "bw_lstm_fixed_prior_job": lambda num_frames: make_bw_lstm(num_input, epochs, num_frames),
      "bw_ffnn_fixed_prior": lambda: {},
    }
  
  
  # @staticmethod
  def _config_handling(func):
    from inspect import signature, Parameter
    sig = signature(func)
    # print(sig)
    assert "config" in sig.parameters or "crnn_config" in sig.parameters
    key = "config" if "config" in sig.parameters else "crnn_config"
    # assert sig.parameters[key].kind == Parameter.POSITIONAL_OR_KEYWORD
    def wrapper(*args, **kwargs):
      arguments = Arguments(sig, args, kwargs)
      cls = arguments["self"]
      get_real_config = lambda config: cls.nn_config_dicts['train'][config]
      # print(arguments[key] is None)
      if isinstance(arguments[key], str):
        # print("yes")
        arguments.transform(key, get_real_config)
      # print(arguments[key])
      return func(*arguments.args, **arguments.kwargs)
    # def wrapper(self, config, *args, **kwargs):
    #   if isinstance(config, str):
    #     config = self.nn_config_dicts['train'][config]
    #   return func(self, config, *args, **kwargs)
    return wrapper
  
  def init_lstm(self, name, num_input=50, layers=None, l2=0.0, lr=1e-4, dropout=0.1, **kwargs):
    config = chelp.blstm_config(num_input, chelp.blstm_network(layers, dropout, l2), lr, **kwargs)
    del config['max_seq_length'], config['adam']
    config['network']['output']['loss_opts'] = {"focal_loss_factor": 2.0}
    config["network"]["output"]["loss"] = "ce"
    self.nn_config_dicts["train"][name] = config
    return copy.deepcopy(config)

  def init_ffnn(self, name, num_input=50, layers=None, activation='relu', l2=0.0, lr=1e-4, dropout=0.1, **kwargs):
    config = chelp.feed_forward_config(num_input, chelp.mlp_network(layers,activation, dropout,l2),lr,**kwargs)
    config['network']['output']['loss_opts'] = {"focal_loss_factor": 2.0}
    config["network"]["output"]["loss"] = "ce"
    del config['momentum']
    self.nn_config_dicts["train"][name] = config
    return config

  # def init_tcnn(self, name, num_input=50, )
  def init_tcnn(self, name, num_input=50, layers=None, filters=None, dilation=None, padding="same", 
               activation='relu', dropout=0.2, l2=0.0, batch_norm=True, lr=0.00025, **kwargs):
    config = chelp.blstm_config(num_input, chelp.tcnn_network(layers, filters, dilation, padding, activation, dropout, l2, batch_norm), lr, **kwargs)
    config['network']['output']['loss_opts'] = {"focal_loss_factor": 2.0}
    config["network"]["output"]["loss"] = "ce"
    self.nn_config_dicts["train"][name] = config
    return config
  
  @_config_handling
  def add_pretrain(self, config, initial_am, final_am, final_epoch, prior_scale=1.0, absolute_scale=1.0):
    # if isinstance(config, str):
    #   config = self.nn_config_dicts['train']['name']
    config['pretrain']             = 'default'
    config['pretrain_repetitions'] = {'default': 0} 
    scales = helpers.ScaleSetter(initial_am, final_am, final_epoch, prior_scale, absolute_scale)
    args = {}
    scales.set_config(args, config)
    config['extra_python'] = args['extra_python']
  
  def add_bw_layer(self, name, am_scale=1.0, ce_smoothing=0.0,
                   import_model=None, exp_average=0.001, 
                   prior_scale=1.0, tdp_scale=1.0):
    add_bw_layer(self.csp['train'], self.nn_config_dicts['train'][name], am_scale, ce_smoothing,
      import_model, exp_average, prior_scale, tdp_scale)
    
  def decode(self, name, compile_crnn_config=None, **kwargs):
    if isinstance(compile_crnn_config, str):
      compile_crnn_config = self.nn_config_dicts['train'][compile_crnn_config]
    self.compile_configs[name] = compile_crnn_config
    return super().decode(name=name, compile_crnn_config=compile_crnn_config, **kwargs)
  
  def run_exp(
    self,
    name: str,
    exp_config: ExpConfig,
    **kwargs
  ):
    kwargs = ChainMap(kwargs, exp_config.__dict__)
    self.nn_and_recog(
      name, **kwargs
    )
  
  def nn_and_recog(
      self,
      name,
      *,
      crnn_config = None, epochs=None, 
      scorer_args=None, recognition_args=None,
      training_args=None, compile_args=None,
      delayed_build=Auto,
      reestimate_prior=False, optimize=True, use_tf_flow=True,
      compile_crnn_config=None, plugin_args=None,
      fast_bw_args={},
      label_sync_decoding=False, **kwargs):
    # experimental
    # for var_name in locals():
    #   if not var_name.endswith("_args"):
    #     continue
    #   locals()[var_name] = locals()[var_name] or {}
    if epochs is None:
      epochs = self.default_epochs
    # get train config
    if isinstance(crnn_config, str):
      crnn_config = self.nn_config_dicts['train'][crnn_config]
    elif crnn_config is None:
      crnn_config = self.nn_config_dicts['train'][name]
    self.nn_config_dicts['train'][name] = crnn_config
    # get compile config
    if isinstance(compile_crnn_config, str):
      compile_crnn_config = self.nn_config_dicts['train'][compile_crnn_config]
      compile_crnn_config.config.pop("extra_python", None)
    self.compile_configs[name] = compile_crnn_config
    # elif compile_crnn_config is None:
    #   compile_crnn_config = self.nn_config_dicts['train'][name]
    training_args = copy.deepcopy(training_args) or {}
    crnn_config = copy.deepcopy(crnn_config)
    extra_config_args = {}
    if delayed_build is Auto:
      delayed_build = hasattr(crnn_config, "build")
    if delayed_build:
      extra_config_args["crnn_config_pre_build"] = crnn_config
      crnn_config = crnn_config.build()
    recognition_args = copy.deepcopy(recognition_args) or {}
    scorer_args = copy.deepcopy(scorer_args) or {}
    plugin_args = copy.deepcopy(plugin_args) or {}
    config_args = {
      "crnn_config": crnn_config,
      "scorer_args": scorer_args,
      "training_args": training_args,
      "recognition_args": recognition_args,
      **extra_config_args
    }
    if "extra_python" in crnn_config.config:
      training_args['extra_python'] = crnn_config.pop('extra_python')

    with safe_csp(self) as tcsp:
      if 'transition_probabilities' in recognition_args:
        self.set_transition_probabilities(**recognition_args.pop('transition_probabilities'), corpus='dev')
      if 'transition_probabilities' in training_args:
        self.tdps.set(training_args.pop('transition_probabilities'), corpus='train')
      for plugin, args in plugin_args.items():
        self.plugins[plugin].apply(**args, **config_args)
      if any(layer['class'] == 'fast_bw' for layer in crnn_config.config['network'].values()) \
        and 'additional_rasr_config_files' not in training_args:
        # print('update')
        amc = fast_bw_args.get("acoustic_model_extra_config", None)
        if amc is not None:
          type(amc)
          # from recipe.i6_core.rasr import RasrConfig
          assert isinstance(amc, sprint.RasrConfig)
        additional_sprint_config_files, additional_sprint_post_config_files \
          = add_fastbw_configs(self.csp['train'], **fast_bw_args) # TODO: corpus dependent and training args
        training_args = {
          **training_args,
          'additional_rasr_config_files':      additional_sprint_config_files,
          'additional_rasr_post_config_files': additional_sprint_post_config_files,
        }
      return super().nn_and_recog(
        name, training_args, crnn_config, scorer_args, recognition_args, epochs=epochs, 
        reestimate_prior=reestimate_prior, optimize=optimize, use_tf_flow=use_tf_flow, compile_args=compile_args,
        compile_crnn_config=compile_crnn_config,
        label_sync_decoding=label_sync_decoding, **kwargs)
  
  def nn_align(self, nn_name, epoch, crnn_config=None, compile_crnn_config=None, plugin_args=MappingProxyType({}), **kwargs):
    if crnn_config is None and compile_crnn_config is None:
      compile_crnn_config = self.compile_configs[nn_name]
    if crnn_config is None:
      crnn_config = self.nn_config_dicts['train'][nn_name]
    with safe_csp(self) as tcsp:
      for plugin, args in plugin_args.items():
        self.plugins[plugin].apply(**args)
      return super().nn_align(nn_name, crnn_config, epoch, compile_crnn_config=compile_crnn_config, **kwargs)

  @_config_handling
  def nn_dump(self, name, training_args, crnn_config, plugin_args=None, *args, **kwargs):
    plugin_args = plugin_args or {}
    if crnn_config is None:
      crnn_config = self.nn_config_dicts['train'][name]
    crnn_config = copy.deepcopy(crnn_config)
    if "extra_python" in crnn_config:
      training_args['extra_python'] = crnn_config.pop('extra_python')
    if "extra_python" in crnn_config:
      training_args['extra_python'] = crnn_config.pop('extra_python')
    with safe_csp(self) as tcsp:
      if 'transition_probabilities' in training_args:
        self.tdps.set(training_args.pop('transition_probabilities'), corpus='train')
      for plugin, pargs in plugin_args.items():
        self.plugins[plugin].apply(**pargs)
      if any(layer['class'] == 'fast_bw' for layer in crnn_config['network'].values()) \
        and 'additional_sprint_config_files' not in training_args:
        # print('update')
        additional_sprint_config_files, additional_sprint_post_config_files = add_fastbw_configs(self.csp['train']) # TODO: corpus dependent and training args
        training_args = {
          **training_args,
          'additional_sprint_config_files':      additional_sprint_config_files,
          'additional_sprint_post_config_files': additional_sprint_post_config_files,
        }
      return super().nn_dump(name, training_args, crnn_config, *args, **kwargs)

  
  def run(self, steps="all", init_only=False):
    super().run(steps)
    if "init_bw" in steps:
      print("Run init_bw")
      additional_sprint_config_files, additional_sprint_post_config_files = add_fastbw_configs(self.csp['train']) # TODO: corpus dependent and training args
      self.default_bw_training_args = {
        "additional_sprint_config_files": additional_sprint_config_files,
        "additional_sprint_post_config_files": additional_sprint_post_config_files
      }

    if "baseline_bw_tcnn" in steps:
      name = "baseline_bw_tcnn"
      print("Run " + name)
      kwargs = {**self.base_crnn_config, **self.base_tcnn_config}
      self.init_tcnn(name, **kwargs)
      self.add_bw_layer(name, **self.base_bw_args)
      self.add_pretrain(name, **self.base_pretrain_args)
      if not init_only:
        self.nn_and_recog(
          name=name,
          # training_args=self.default_nn_training_args,
          crnn_config=self.nn_config_dicts['train'][name],
          scorer_args={}, recognition_args={}, epochs=self.default_epochs,
        )
    
    if "baseline_bw_ffnn" in steps:
      name = "baseline_bw_ffnn"
      print("Run "+ name)
      kwargs = {**self.base_crnn_config, **self.base_ffnn_config}
      self.init_ffnn(name, **kwargs)
      self.add_bw_layer(name, **self.base_bw_args)
      self.add_pretrain(name, **self.base_pretrain_args)
      if not init_only:
        self.nn_and_recog(
          name=name,
          # training_args=self.default_bw_training_args,
          crnn_config=self.nn_config_dicts['train'][name],
          scorer_args={}, recognition_args={}, epochs=self.default_epochs,
        )
      
    if "baseline_viterbi_ffnn" in steps:
      name = "baseline_viterbi_ffnn"
      print("Run "+ name)
      kwargs = {
        **self.base_crnn_config,
        **self.base_ffnn_config,
        **self.base_viterbi_args}
      self.init_ffnn(name, **kwargs)
      print("Hi")
      if not init_only:
        self.nn_and_recog(
          name=name,
          # training_args=self.default_nn_training_args,
          crnn_config=self.nn_config_dicts['train'][name],
          scorer_args={}, recognition_args={}, epochs=self.default_epochs,
        )

    if "baseline_viterbi_lstm" in steps:
      name = "baseline_viterbi_lstm"
      print("Run "+ name)
      kwargs = {
        **self.base_crnn_config,
        **self.base_lstm_config,
        **self.base_viterbi_args}
      self.init_lstm(name, **kwargs)
      if not init_only:
        self.nn_and_recog(
          name=name,
          # training_args=self.default_nn_training_args,
          crnn_config=self.nn_config_dicts['train'][name],
          scorer_args={}, recognition_args={}, epochs=self.default_epochs,
        )
