__all__ = ["ExpConfig", "RecognitionConfig", "BaseSystem", "NNSystem"]
from sisyphus import *
Path = setup_path(__package__)

import copy
import warnings
import math
import os
from collections import ChainMap
from typing import Optional

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
from i6_core import rasr
import i6_core.util         as util
import i6_experiments.users.mann.experimental.helpers as helpers
from i6_experiments.users.mann.experimental.extractors import ExtractStateTyingStats
from i6_core.lexicon.allophones import DumpStateTyingJob, StoreAllophonesJob
from i6_experiments.users.mann.setups.tdps import CombinedModel

from i6_core.meta.system import select_element
# from i6_core.crnn.multi_sprint_training import PickleSegments

NotSpecified = object()
Default = object()

from collections import namedtuple, UserString
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
from i6_experiments.common.setups.rasr.nn_system import NnSystem as CommonNnSystem

# from dataclasses import dataclass
import dataclasses
from typing import Union, Optional

RASR_SCALE_MAP = {
	"am_scale": "mixture-set.scale",
	"prior_scale": "mixture-set.priori-scale",
	"tdp_scale": "tdp.scale"
}

class Key(UserString):
	def __init__(self, value):
		super().__init__(value)
	
	def __add__(self, other):
		return Key(".".join((self.data, other)))

class AbstractConfig:
	def extend(self, **extensions):
		changes = {key: {**getattr(self, key), **value} for key, value in extensions.items()}
		return dataclasses.replace(self, **changes)

	def replace(self, **kwargs):
		return dataclasses.replace(self, **kwargs)

@dataclasses.dataclass
class RecognitionConfig(AbstractConfig):
	lm_scale : float = NotSpecified
	am_scale : float = NotSpecified
	prior_scale : float = NotSpecified
	tdp_scale : float = NotSpecified
	tdps : CombinedModel = NotSpecified
	beam_pruning : float = NotSpecified
	beam_pruning_threshold : Union[int, float] = NotSpecified

	def replace(self, **kwargs):
		return dataclasses.replace(self, **kwargs)

	def to_dict(self) -> dict:
		out_dict = {}
		extra_config_keys = ["am_scale", "tdp_scale", "prior_scale"]
		if any(getattr(self, key) is not NotSpecified for key in extra_config_keys):
			rasr_am_key = "flf-lattice-tool.network.recognizer.acoustic-model."
			extra_rasr_config = rasr.RasrConfig()
			for key in extra_config_keys:
				if getattr(self, key) is not NotSpecified:
					extra_rasr_config[rasr_am_key + RASR_SCALE_MAP[key]] = getattr(self, key)
			out_dict["extra_config"] = extra_rasr_config
		if self.tdps is not NotSpecified:
			extra_rasr_config = out_dict.get("extra_config", rasr.RasrConfig())
			extra_rasr_config["flf-lattice-tool.network.recognizer.acoustic-model"] = self.tdps.to_acoustic_model_extra_config()
			out_dict["extra_config"] = extra_rasr_config
		if self.lm_scale is not NotSpecified:
			out_dict["lm_scale"] = self.lm_scale
		out_dict["search_parameters"] = {}
		if self.beam_pruning is not NotSpecified:
			out_dict["search_parameters"]["beam-pruning"] = self.beam_pruning
		if self.beam_pruning_threshold is not NotSpecified:
			out_dict["search_parameters"]["beam-pruning-threshold"] = self.beam_pruning_threshold
		return out_dict
		

@dataclasses.dataclass
class ExpConfig(AbstractConfig):
	crnn_config: Union[str, crnn.ReturnnConfig] = None
	training_args: dict = None
	plugin_args: dict = None
	compile_crnn_config: Union[str, crnn.ReturnnConfig] = None
	recognition_args: dict = None
	fast_bw_args: dict = None
	epochs: list = None
	scorer_args: dict = None
	reestimate_prior: str = None
	dump_epochs: list = None

	def extend(self, **extensions):
		changes = {key: {**getattr(self, key), **value} for key, value in extensions.items()}
		return dataclasses.replace(self, **changes)


class BaseSystem(RasrSystem):
	def __init__(
		self,
		rasr_binary_path,
		native_ops_path,
		returnn_root=None,
		returnn_python_exe=None,
		returnn_python_home=None,
		**kwargs
	):
		super().__init__(rasr_binary_path=rasr_binary_path, **kwargs)
		self.native_ops_path = native_ops_path
		self.returnn_root = returnn_root
		self.returnn_python_home = returnn_python_home
		self.returnn_python_exe = returnn_python_exe
		self._num_classes_dict = {}

		self.nn_checkpoints = {}
	
	def add_overlay(self, origin, name):
		super().add_overlay(origin, name)
		self.nn_checkpoints[name] = {}
	
	def set_corpus(self, name, *args, **kwargs):
		super().set_corpus(name, *args, **kwargs)
		self.nn_checkpoints[name] = {}

	@property
	def csp(self):
		return self.crp
	
	@csp.setter
	def csp(self, crp):
		self.crp = crp

	def get_state_tying(self):
		return self.csp['base'].acoustic_model_config.state_tying.type
	
	def set_state_tying(self, value, cart_file: Optional[tk.Path] = None, extra_args={}, **kwargs):
		assert value in {'monophone', 'cart', 'monophone-no-tying-dense', 'lut', 'lookup'}, "No other state tying types supported yet"
		if value == 'cart': assert cart_file is not None, "Cart file must be specified"
		for crp in self.crp.values():
			crp.acoustic_model_config.state_tying.type = value
			crp.acoustic_model_config.state_tying.file = cart_file
			for k, v in {**extra_args, **kwargs}.items():
				crp.acoustic_model_config.state_tying[k] = v
		
	def num_classes(self):
		if self.get_state_tying() in self._num_classes_dict:
			return self._num_classes_dict[self.get_state_tying()]
		state_tying = DumpStateTyingJob(self.csp["train"]).out_state_tying
		tk.register_output("state-tying_mono", state_tying)
		num_states = ExtractStateTyingStats(state_tying).out_num_states
		return num_states
	
	def get_allophone_file(self):
		return StoreAllophonesJob(self.csp["train"]).out_allophone_file
	
	def get_state_tying_file(self):
		return DumpStateTyingJob(self.csp["train"]).out_state_tying
	
	def silence_idx(self):
		state_tying = DumpStateTyingJob(self.csp["train"]).out_state_tying
		return ExtractStateTyingStats(state_tying).out_silence_idx
	
	def set_num_classes(self, state_tying, num_classes):
		self._num_classes_dict[state_tying] = num_classes
	
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
															questions                   = self.cart_questions[corpus],
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
	
	def filter_alignment(self, alignment):
		all_segments = corpus_recipes.SegmentCorpusJob(self.corpora[corpus].corpus_file, 1)
		import i6_experiments.users.mann.experimental.extractors as extr
		from i6_core.corpus.filter import FilterSegmentsByListJob
		alignment_logs = alignment
		filter_list = extr.ExtractAlignmentFailuresJob(alignment_logs).out_filter_list
		all_segments = FilterSegmentsByListJob({1: all_segments.out_single_segment_files[1]}, filter_list)

	def init_nn(self, name, corpus, dev_size, bad_segments=None, dump=False, alignment_logs=None):
		all_segments = corpus_recipes.SegmentCorpusJob(self.corpora[corpus].corpus_file, 1)
		if alignment_logs is True:
			import glob
			log_path_pattern = os.path.join(
				os.path.dirname(
					select_element(self.alignments, corpus, 'init_align').get_path(),
				),
				"alignment.log.*.gz"
			)
			alignment_logs = glob.glob(log_path_pattern)
		if alignment_logs is not None:
			import i6_experiments.users.mann.experimental.extractors as extr
			from i6_core.corpus.filter import FilterSegmentsByListJob
			filter_list = extr.ExtractAlignmentFailuresJob(alignment_logs).out_filter_list
			all_segments = FilterSegmentsByListJob({1: all_segments.out_single_segment_files[1]}, filter_list)
		new_segments = corpus_recipes.ShuffleAndSplitSegmentsJob(
			segment_file=all_segments.out_single_segment_files[1],
			split={ 'train': 1.0 - dev_size, 'dev': dev_size }
		)

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
			self.default_nn_training_args['train_corpus'] = '%s_train' % name
		if 'dev_corpus'   not in self.default_nn_training_args:
			self.default_nn_training_args['dev_corpus']   = '%s_dev'   % name
		if 'feature_flow' not in self.default_nn_training_args:
			self.default_nn_training_args['feature_flow'] = 'mfcc+context%d' % window_size

		if 'feature_dimension' not in self.default_scorer_args:
			self.default_scorer_args['feature_dimension'] = window_size * 16

		if 'flow' not in self.default_recognition_args:
			self.default_recognition_args['flow'] = 'mfcc+context%d' % window_size

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
			config = crnn_config.build_compile_config()
		if hasattr(crnn_config, "build"):
		# if custom_compile:
			config = crnn_config.build()
		else:
			config = copy.deepcopy(crnn_config)
		config_dict = config.config
		if output_tensor_name is None:
			output_tensor_name = 'output/output_batch_major'
		# Replace lstm unit to nativelstm2
		lstm_flag = False
		for layer in config_dict['network'].values():
			unit = layer.get('unit', None)
			if isinstance(unit, dict):
				continue
			if unit in {'lstmp'}:
				layer['unit']='nativelstm2'
				lstm_flag = True
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
		config_dict['num_outputs']['classes'] = [self.num_classes(), 1]
		window = config_dict.get('window', 1)
		if window > 1:
			feature_flow = lda.add_context_flow(feature_flow, max_size=window, right=window // 2)

		model_path = tk.uncached_path(model.model)[:-5]
		if not isinstance(config_dict, crnn.ReturnnConfig):
			config_dict = crnn.ReturnnConfig(config_dict, {})
		compile_graph_job = crnn.CompileTFGraphJob(
			config, **compile_args
		)
		if alias is not None:
			alias = f"compile_crnn-{alias}"
			compile_graph_job.add_alias(alias)
			self.jobs["dev"][alias] = compile_graph_job
		
		if not add_tf_flow:
			return feature_flow

		tf_graph = compile_graph_job.out_graph

		# Setup TF AM flow node
		tf_flow = sprint.FlowNetwork()
		tf_flow.add_input('input-features')
		tf_flow.add_output('features')
		tf_flow.add_param('id')
		tf_fwd = tf_flow.add_node('tensorflow-forward', 'tf-fwd', {'id': '$(id)'})
		tf_flow.link('network:input-features', tf_fwd + ':features')
		tf_flow.link(tf_fwd + ':log-posteriors', 'network:features')

		tf_flow.config = sprint.RasrConfig()

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
			if lstm_flag:
				tf_flow.config[tf_fwd].loader.required_libraries = self.native_ops_path
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
		if self.returnn_root is not None:
			train_args['returnn_root'] = self.returnn_root
		if self.returnn_python_exe is not None:
			train_args['returnn_python_exe'] = self.returnn_python_exe

		with tk.block('NN - %s' % name):
			self.train_nn(**train_args)
			train_job.add_alias('nn_%s' % name)
			tk.register_output('plot_se_%s.png' % name, train_job.plot_se)
			tk.register_output('plot_lr_%s.png' % name, train_job.plot_lr)


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


	def nn_and_recog(
		self, name, training_args, crnn_config, scorer_args,  
		recognition_args, epochs=None, reestimate_prior='CRNN', compile_args=None,
		optimize=True, use_tf_flow=False, label_sync_decoding=False, 
	#  delayed_build=False,
		alt_training=False, dump_epochs=None,
		compile_crnn_config=None, alt_decoding=NotSpecified
	):
		if compile_args is None:
			compile_args = {}
		if epochs is None:
			epochs = self.default_epochs
		train_args = dict(**self.default_nn_training_args)
		train_args.update(training_args)
		train_args['name']        = name
		train_args['returnn_config'] = crnn_config # if not delayed_build else crnn_config.build()
		if self.returnn_root is not None and 'returnn_root' not in train_args:
			train_args['returnn_root'] = self.returnn_root
		if self.returnn_python_exe is not None and 'returnn_python_exe' not in train_args:
			train_args['returnn_python_exe'] = self.returnn_python_exe

		with tk.block('NN - %s' % name):
			if alt_training:
				reestimate_prior = "alt"
				self.trainer.train(**train_args)
			else:
				self.train_nn(**train_args)
			train_job = self.jobs[train_args['feature_corpus']]['train_nn_%s' % name]
			train_job.add_alias('nn_%s' % name)
			self.nn_checkpoints[train_args['feature_corpus']][name] = train_job.out_checkpoints
			tk.register_output('plot_se_%s.png' % name, train_job.out_plot_se)
			tk.register_output('plot_lr_%s.png' % name, train_job.out_plot_lr)

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
	
	def extract_prior(self, name, crnn_config, training_args, epoch):
		# alignment = None
		# if score_args.pop("use_alignment", True):
		alignment = select_element(self.alignments, training_args["feature_corpus"], training_args["alignment"])
		num_classes  = self.functor_value(training_args["num_classes"])
		score_features = crnn.ReturnnRasrComputePriorJob(
			train_crp = self.csp[training_args['train_corpus']],
			dev_crp = self.csp[training_args['dev_corpus']],
			feature_flow = self.feature_flows[training_args['feature_corpus']][training_args['feature_flow']],
			num_classes = self.functor_value(training_args["num_classes"]),
			alignment = alignment,
			returnn_config = crnn_config,
			model_checkpoint = self.jobs[training_args['feature_corpus']]['train_nn_%s' % name].out_checkpoints[epoch],
			returnn_python_exe = training_args.get("returnn_python_exe", None),
			returnn_root = training_args.get("returnn_root", None)
		)
		return score_features
			
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
		if reestimate_prior == "alt":
			extract_prior = self.trainer.extract_prior
		else:
			extract_prior = self.extract_prior
		if reestimate_prior in {'CRNN', 'alt'}:
			self.jobs[training_args["feature_corpus"]]["returnn_compute_prior_%s" % scorer_name] \
				= score_features = extract_prior(name, crnn_config, training_args, epoch)
			scorer_name += '-prior'
			score_args['name'] = scorer_name
			score_args['prior_file'] = score_features.out_prior_xml_file
		elif reestimate_prior == 'transcription':
			from i6_experiments.users.mann.setups.prior import get_prior_from_transcription_job
			# prior = get_prior_from_transcription_job(self, self.num_frames["train"])
			# score_args['prior_file'] = prior["xml"]
			score_args['prior_file'] = self.prior_system.prior_xml_file
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
		model = self.jobs[training_args['feature_corpus']]['train_nn_%s' % name].out_models[epoch]
		compile_args = copy.deepcopy(compile_args)
		if compile_crnn_config is None:
			compile_crnn_config = crnn_config
		compile_network_extra_config = compile_args.pop('compile_network_extra_config', {})
		compile_crnn_config.config['network'].update(compile_network_extra_config)
		if "alias" in compile_args and not isinstance(compile_args['alias'], str):
			compile_args['alias'] = name
		recog_args['flow'] = tf_flow = self.get_tf_flow(
			feature_flow_net, compile_crnn_config, model, 
			recog_args.pop('output_tensor_name', None), 
			drop_layers=recog_args.pop('drop_layers', None),
			req_libraries=recog_args.pop('req_libraries', NotSpecified),
			**compile_args,
		)
		if score_args['prior_mixtures'] is not None:
			prior_mixtures = select_element(self.mixtures, training_args['feature_corpus'], score_args['prior_mixtures']) 
		else:
			from i6_core.mm import CreateDummyMixturesJob
			prior_mixtures = CreateDummyMixturesJob(self.num_classes(), self.num_input).out_mixtures
		recog_args['feature_scorer'] = scorer = sprint.PrecomputedHybridFeatureScorer(
			prior_mixtures=prior_mixtures,
			priori_scale=score_args.get('prior_scale', 0.),
			# prior_file=score_features.prior if reestimate_prior else None,
			prior_file=score_args.get("prior_file", None),
			scale=score_args.get('mixture_scale', 1.0)
		)
		self.feature_scorers[training_args['feature_corpus']][scorer_name] = scorer

		extra_rqmts = recog_args.pop('extra_rqmts', {})
		# reset tdps for recognition
		with tk.block('recog-V%d' % epoch):
			if optimize:
				self.recog_and_optimize(**recog_args)
				self.jobs[recog_args["corpus"]]['recog_%s' % recog_args["name"] + "-optlm"].rqmt.update(extra_rqmts)
			else:
				self.recog(**recog_args)
			self.jobs[recog_args["corpus"]]['recog_%s' % recog_args["name"]].rqmt.update(extra_rqmts)


	def nn_align(
		self, nn_name, crnn_config, epoch, scorer_suffix='', mem_rqmt=8,
		compile_crnn_config=None, name=None, feature_flow='gt', dump=False,
		compile_args=None, evaluate=False, feature_scorer=None, **kwargs
	):
		# get custom tf
		compile_args = compile_args or {}
		if compile_crnn_config is None:
			compile_crnn_config = crnn_config
		feature_flow_net = meta.system.select_element(self.feature_flows, 'train', feature_flow)
		model = self.jobs['train']['train_nn_%s' % nn_name].out_models[epoch]
		flow = self.get_tf_flow(
			feature_flow_net,
			compile_crnn_config,
			model,
			**compile_args
		)

		# get alignment
		model_name = '%s-%s' % (nn_name, epoch)
		name = name or model_name
		alignment_name = 'alignment_%s' % name
		feature_scorer = model_name + scorer_suffix if feature_scorer is None else feature_scorer
		self.align(
			name=name if not name else name, corpus='train', 
			flow=flow,
			feature_scorer=meta.select_element(self.feature_scorers, 'train', feature_scorer),
			**kwargs,
		)
		j = self.jobs['train'][alignment_name]
		j.rqmt['mem'] = mem_rqmt
		j.add_alias("align/%s" % name)
		if evaluate:
			self.evaluate_alignment(name, corpus='train')
		if dump:
			from recipe.mm import DumpAlignmentJob
			j = DumpAlignmentJob(self.csp['train'], self.feature_flows['train'][feature_flow], self.alignments['train'][name])
			tk.register_output(alignment_name, j.alignment_bundle)
			return j.alignment_bundle
		return None
  
	def evaluate_alignment(self, name, corpus, alignment=None, alignment_logs=None):
		from i6_core.corpus import FilterSegmentsByAlignmentConfidenceJob
		from i6_experiments.users.mann.experimental.statistics import (
			SilenceAtSegmentBoundaries,
			AlignmentStatisticsJob
		)
		if alignment is None:
			alignment = name
			alignment = select_element(
				self.alignments,
				corpus,
				name
			)
			try:
				alignment = alignment.alternatives["bundle"]
			except AttributeError:
				alignment = alignment.value
		if alignment_logs is None:
			alignment_logs = (
				alignment
				.creator
				.out_log_file
			)
		alignment_scores = FilterSegmentsByAlignmentConfidenceJob(
			alignment_logs, 10, self.csp[corpus]
		)
		tk.register_output("stats_align/scores/{}.png".format(name), alignment_scores.out_plot_avg)
		args = (
			alignment,
			# self.csp["train"].acoustic_model_config.allophones.add_from_file,
			self.get_allophone_file(),
			self.csp["train"].segment_path.hidden_paths,
			self.csp["train"].concurrent
		)
		asj = AlignmentStatisticsJob(*args)
		asj.add_alias(f"alignment_stats-{name}")
		stats = asj.counts
		tk.register_output("stats_align/counts/{}".format(name), stats)
		return stats

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

from i6_experiments.users.mann.experimental.sequence_training import add_bw_layer, add_bw_output, add_fastbw_configs
from i6_experiments.users.mann.experimental.util import safe_crp

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

# class Encoder(enum.Enum):
# 	LSTM = "lstm"
# 	FFNN = "ffnn"
# 	TDNN = "tdnn"

from i6_experiments.users.mann.nn.config import viterbi_ffnn
from i6_experiments.users.mann.nn.config import viterbi_lstm
from i6_experiments.users.mann.nn.config import make_baseline as make_tdnn_baseline
from i6_experiments.users.mann.nn.config.constants import BASE_BW_LRS

class ConfigBuilder:
	def __init__(self, system):
		self.system = system
		self.config_args = {}
		self.network_args = {}
		self.scales = {}
		self.base_constructor = None
		self.transforms = []

	def set_ffnn(self):
		self.encoder = viterbi_ffnn
		return self
	
	def set_lstm(self):
		self.encoder = viterbi_lstm
		return self
	
	def set_tdnn(self):
		self.encoder = make_tdnn_baseline
		return self
	
	def set_config_args(self, config_args):
		self.config_args = config_args
		return self
	
	def set_network_args(self, network_args):
		self.network_args = network_args
		return self
	
	def set_pretrain(self, **kwargs):
		raise NotImplementedError()
		return self
	
	def set_transcription_prior(self):
		self.transforms.append(self.system.prior_system.add_to_config)
		return self
		
	def set_tina_scales(self):
		self.scales = {
			"am_scale": 0.3,
			"prior_scale": 0.1,
			"tdp_scale": 0.1
		}
		return self
	
	def set_oclr(self):
		from i6_experiments.users.mann.nn import learning_rates
		raise NotImplementedError()
		return self
	
	def set_specaugment(self):
		from i6_experiments.users.mann.nn import specaugment
		self.transforms.append(specaugment.set_config)
		return self
	
	def build(self):
		from i6_experiments.users.mann.nn import BASE_BW_LRS
		from i6_experiments.users.mann.nn import prior, pretrain, bw, get_learning_rates
		kwargs = BASE_BW_LRS.copy()
		kwargs.update(self.config_args)
		# network_kwargs.update(self.network_args)
		# viterbi_config = viterbi_lstm(num_input, network_kwargs=network_kwargs, **kwargs)
		viterbi_config_dict = self.encoder(self.system.num_input, network_kwargs=self.network_args, **kwargs)

		bw_config = bw.ScaleConfig.copy_add_bw(
			viterbi_config_dict, self.system.csp["train"],
			num_classes=self.system.num_classes(),
			**self.scales
		)

		for transform in self.transforms:
			transform(bw_config)
		
		return bw_config

class NNSystem(BaseSystem):
	def __init__(self, num_input=None, epochs=None, rasr_binary_path=Default, **kwargs):
		if rasr_binary_path is Default:
			rasr_binary_path = tk.Path(gs.RASR_ROOT).join_right('arch/linux-x86_64-standard')
		super().__init__(rasr_binary_path, **kwargs)

		assert num_input and epochs
		self.num_input = num_input
		self.default_epochs = epochs

		self.nn_config_dicts = {'train': {}}
		self.compile_configs = {}

		from i6_experiments.users.mann.setups.tdps import CombinedModelApplicator
		self.plugins = {
			"tdps": CombinedModelApplicator(self),
		}

		import i6_experiments.users.mann.setups.prior as tp
		self.prior_system: tp.PriorSystem = None

		from i6_experiments.users.mann.nn.config import viterbi_lstm
		from i6_experiments.users.mann.nn import prior, pretrain, bw, get_learning_rates
		def make_bw_lstm(
			num_input, num_frames=None, transcription_prior=True,
			numpy=False, tina=False,
			config_args={}, network_args={}
		) -> pretrain.PretrainConfigHolder:
			from i6_experiments.users.mann.nn.config import BASE_BW_LRS
			kwargs = BASE_BW_LRS.copy()
			network_kwargs = {}
			# if tina:
			# 	from i6_experiments.users.mann.nn.constants import TINA_UPDATES_1K, TINA_NETWORK_CONFIG
			kwargs.update(config_args)
			network_kwargs.update(network_args)
			viterbi_config = viterbi_lstm(num_input, network_kwargs=network_kwargs, **kwargs)
			bw_config = bw.ScaleConfig.copy_add_bw(
				viterbi_config, self.csp["train"],
				am_scale=0.7, prior_scale=0.035,
				tdp_scale=0.5,
				num_classes=self.num_classes(),
			)
			# bw_config.config["learning_rates"] = get_learning_rates(increase=70, decay=70)
			del bw_config.config["learning_rate"]
			if transcription_prior:
				self.prior_system.add_to_config(bw_config)
			return pretrain.PretrainConfigHolder(
					bw_config, rel_prior_scale=0.05
			)

		def make_bw(
			num_input,
			transcription_prior=True,
			base_config=None,
			config_args={}, network_args={},
			scales=None,
		) -> pretrain.PretrainConfigHolder:
			from i6_experiments.users.mann.nn import BASE_BW_LRS
			kwargs = BASE_BW_LRS.copy()
			if not base_config:
				network_kwargs = {}
				kwargs.update(config_args)
				network_kwargs.update(network_args)
				viterbi_config = viterbi_lstm(num_input, network_kwargs=network_kwargs, **kwargs)
			else:
				viterbi_config = copy.deepcopy(base_config)
				viterbi_config.config.update(config_args)
			pretrain_scales = True
			if scales == "tina":
				scales = {
					"am_scale": 0.3,
					"prior_scale": 0.1,
					"tdp_scale": 0.1
				}
				pretrain_scales = False
			elif isinstance(scales, dict):
				scales = scales.copy()
			else:
				scales = {
					"am_scale": 0.7,
					"prior_scale": 0.035,
					"tdp_scale": 0.5,
				}
			bw_config = bw.ScaleConfig.copy_add_bw(
				viterbi_config, self.csp["train"],
				num_classes=self.num_classes(),
				**scales
			)
			# bw_config.config["learning_rates"] = get_learning_rates(increase=70, decay=70)
			del bw_config.config["learning_rate"]
			if transcription_prior:
				self.prior_system.add_to_config(bw_config)
			if not pretrain_scales:
				return bw_config
			return pretrain.PretrainConfigHolder(
					bw_config, rel_prior_scale=0.05
			)

		def make_bw_ffnn(num_input, numpy=False, transcription_prior=True):
			from i6_experiments.users.mann.nn.config import viterbi_ffnn, BASE_BW_LRS
			# from i6_experiments.users.mann.nn.constants import BASE_BW_LRS
			kwargs = BASE_BW_LRS.copy()
			viterbi_config = viterbi_ffnn(num_input, **kwargs)
			bw_config = bw.ScaleConfig.copy_add_bw(
				viterbi_config, self.csp["train"],
				am_scale=0.7,
				prior_scale=0.35,
				tdp_scale=0.5,
				num_classes=self.num_classes(),
			)
			bw_config.config["learning_rates"] = get_learning_rates(inc_min_ratio=0.25, increase=70, decay=70)
			if transcription_prior:
				self.prior_system.add_to_config(bw_config)
			return pretrain.PretrainConfigHolder(
					bw_config, rel_prior_scale=0.5
			)
		

		from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
		self.baselines = {
			"viterbi_lstm": lambda: viterbi_lstm(num_input),
			"bw_lstm_fixed_prior": lambda: make_bw_lstm(num_input),
			"bw_lstm_fixed_prior_new": lambda: make_bw_lstm(num_input, True),
			"bw_lstm_fixed_prior_job": lambda num_frames: make_bw_lstm(num_input, num_frames),
			"bw_lstm_povey_prior": lambda: make_bw_lstm(num_input, num_frames=None, transcription_prior=False),
			"bw_ffnn_fixed_prior": lambda: make_bw_ffnn(num_input),
			"bw_lstm_tina_1k": lambda: make_bw_lstm(num_input, tina=True, config_args=TINA_UPDATES_1K, network_args=TINA_NETWORK_CONFIG),
			"bw_lstm_tina_swb": lambda: make_bw_lstm(num_input, tina=True, config_args=TINA_UPDATES_SWB, network_args=TINA_NETWORK_CONFIG),
			"bw_tina_swb": lambda config: make_bw(num_input, base_config=config, config_args=TINA_UPDATES_SWB, scales="tina"),
			"bw_tina_swb_povey": lambda config: make_bw(num_input, base_config=config, config_args=TINA_UPDATES_SWB, scales="tina", transcription_prior=False),
		}
	
	def _config_handling(func):
		from inspect import signature, Parameter
		sig = signature(func)
		assert "config" in sig.parameters or "crnn_config" in sig.parameters
		key = "config" if "config" in sig.parameters else "crnn_config"
		def wrapper(*args, **kwargs):
			arguments = Arguments(sig, args, kwargs)
			cls = arguments["self"]
			get_real_config = lambda config: cls.nn_config_dicts['train'][config]
			if isinstance(arguments[key], str):
				arguments.transform(key, get_real_config)
			return func(*arguments.args, **arguments.kwargs)
		return wrapper
		
	def decode(self, name, compile_crnn_config=None, **kwargs):
		if isinstance(compile_crnn_config, str):
			compile_crnn_config = self.nn_config_dicts['train'][compile_crnn_config]
		self.compile_configs[name] = compile_crnn_config
		return super().decode(name=name, compile_crnn_config=compile_crnn_config, **kwargs)
	
	def init_dump_system(self, segments, **default_dump_args):
		from .. import dump
		self.dump_system = dump.HdfDumpster(self, segments, default_dump_args)
	
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
			reestimate_prior='CRNN', optimize=True, use_tf_flow=True,
			compile_crnn_config=None, plugin_args=None,
			fast_bw_args={},
			dump_epochs=None, dump_args=None,
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

		with safe_crp(self) as tcsp:
			for plugin, args in plugin_args.items():
				self.plugins[plugin].apply(**args, **config_args)
			if any(layer['class'] == 'fast_bw' for layer in crnn_config.config['network'].values()) \
				and 'additional_rasr_config_files' not in training_args:
				additional_sprint_config_files, additional_sprint_post_config_files \
					= add_fastbw_configs(self.csp['train'], **fast_bw_args) # TODO: corpus dependent and training args
				training_args = {
					**training_args,
					'additional_rasr_config_files':      additional_sprint_config_files,
					'additional_rasr_post_config_files': additional_sprint_post_config_files,
				}
			# del tdp from train config
			self.crp["train"].acoustic_model_config = self.crp["base"].acoustic_model_config._copy()
			del self.crp["train"].acoustic_model_config.tdp

			dump_config = copy.deepcopy(crnn_config)
			# call training
			j = super().nn_and_recog(
				name, training_args, crnn_config, scorer_args, recognition_args, epochs=epochs, 
				reestimate_prior=reestimate_prior, optimize=optimize, use_tf_flow=use_tf_flow, compile_args=compile_args,
				compile_crnn_config=compile_crnn_config,
				label_sync_decoding=label_sync_decoding, **kwargs)

			if dump_epochs is None:
				return j
			assert hasattr(self, "dump_system"), "Dump system not available"
			if hasattr(dump_config, "build"): # likely a pretrain config
				dump_config = dump_config.config
			for epoch in dump_epochs:
				self.dump_system.run(
					name=name,
					returnn_config=dump_config,
					epoch=epoch,
					training_args=training_args,
					fast_bw_args=fast_bw_args,
					plot_args=dump_args,
				)

	
	def nn_align(self, nn_name, epoch, crnn_config=None, compile_crnn_config=None, plugin_args=MappingProxyType({}), **kwargs):
		if crnn_config is None and compile_crnn_config is None:
			compile_crnn_config = self.compile_configs[nn_name]
		if crnn_config is None:
			crnn_config = self.nn_config_dicts['train'][nn_name]
		with safe_crp(self) as tcsp:
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
		with safe_crp(self) as tcsp:
			if 'transition_probabilities' in training_args:
				self.tdps.set(training_args.pop('transition_probabilities'), corpus='train')
			for plugin, pargs in plugin_args.items():
				self.plugins[plugin].apply(**pargs)
			if any(layer['class'] == 'fast_bw' for layer in crnn_config['network'].values()) \
				and 'additional_sprint_config_files' not in training_args:
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
