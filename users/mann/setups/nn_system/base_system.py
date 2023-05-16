__all__ = ["ExpConfig", "RecognitionConfig", "BaseSystem", "NNSystem", "ConfigBuilder"]
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
from i6_experiments.users.mann.nn import bw

from i6_core.meta.system import select_element
# from i6_core.crnn.multi_sprint_training import PickleSegments

NotSpecified = object()
Default = object()

from collections import namedtuple, UserString, defaultdict
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
from ..util import DelayedPhonemeIndex, DelayedPhonemeInventorySize

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
	
	def to_dict(self, **kwargs):
		return dataclasses.asdict(self, **kwargs)

@dataclasses.dataclass
class RecognitionConfig(AbstractConfig):
	lm_scale : float = NotSpecified
	am_scale : float = NotSpecified
	prior_scale : float = NotSpecified
	tdp_scale : float = NotSpecified
	tdps : CombinedModel = NotSpecified
	beam_pruning : float = NotSpecified
	beam_pruning_threshold : Union[int, float] = NotSpecified
	pronunciation_scale : float = NotSpecified
	altas : Optional[float] = NotSpecified
	use_gpu: Optional[bool] = False
	extra_args : Optional[dict] = dataclasses.field(default_factory=dict)
	extra_config : Optional[rasr.RasrConfig] = NotSpecified

	def replace(self, **kwargs):
		return dataclasses.replace(self, **kwargs)

	def to_dict(self, _full_tdp_config=False, **extra_args) -> dict:
		out_dict = {**extra_args, **self.extra_args}
		extra_config_keys = ["am_scale", "tdp_scale", "prior_scale"]
		if self.extra_config is not NotSpecified:
			extra_rasr_config = out_dict.get("extra_config", rasr.RasrConfig())
			extra_rasr_config._update(self.extra_config)
			out_dict["extra_config"] = extra_rasr_config
		if any(getattr(self, key) is not NotSpecified for key in extra_config_keys):
			rasr_am_key = "flf-lattice-tool.network.recognizer.acoustic-model."
			extra_rasr_config = out_dict.get("extra_config", rasr.RasrConfig())
			for key in extra_config_keys:
				if getattr(self, key) is not NotSpecified:
					extra_rasr_config[rasr_am_key + RASR_SCALE_MAP[key]] = getattr(self, key)
			out_dict["extra_config"] = extra_rasr_config
		if self.tdps is not NotSpecified:
			extra_rasr_config = out_dict.get("extra_config", rasr.RasrConfig())
			if _full_tdp_config:
				extra_rasr_config["flf-lattice-tool.network.recognizer.acoustic-model"] = self.tdps.to_acoustic_model_config()
			else:
				extra_rasr_config["flf-lattice-tool.network.recognizer.acoustic-model"] = self.tdps.to_acoustic_model_extra_config()
			out_dict["extra_config"] = extra_rasr_config
		if self.altas is not NotSpecified:
			extra_rasr_config = out_dict.get("extra_config", rasr.RasrConfig())
			extra_rasr_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = self.altas
			out_dict["extra_config"] = extra_rasr_config
		if self.lm_scale is not NotSpecified:
			out_dict["lm_scale"] = self.lm_scale
		out_dict["search_parameters"] = {}
		if self.beam_pruning is not NotSpecified:
			out_dict["search_parameters"]["beam-pruning"] = self.beam_pruning
		if self.beam_pruning_threshold is not NotSpecified:
			out_dict["search_parameters"]["beam-pruning-limit"] = self.beam_pruning_threshold
		if self.pronunciation_scale is not NotSpecified:
			out_dict["pronunciation_scale"] = self.pronunciation_scale
		out_dict["use_gpu"] = self.use_gpu
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
	alt_training: bool = False
	alt_decoding: Union[bool, dict] = False
	compile_args: dict = None

	def extend(self, **extensions):
		changes = {key: {**getattr(self, key), **value} for key, value in extensions.items()}
		return dataclasses.replace(self, **changes)


class AlignmentConfig:
	acoustic_model_key = [
		"acoustic-model-trainer",
		"aligning-feature-extractor",
		"feature-extraction",
		"alignment",
		"model-combination",
		"acoustic-model",
	]
	prior_scale_key = acoustic_model_key + [
		"mixture-set",
		"priori-scale"
	]
	tdp_scale_key = acoustic_model_key + ["tdp", "scale"]

	def __init__(self, tdp_scale=None, prior_scale=None, tdps=None, correct_fsa=True):
		self.config = rasr.RasrConfig()
		if tdps is not None:
			assert isinstance(tdps, CombinedModel)
			self.set_tdps(tdps)
		if tdp_scale is not None:
			self.set_tdp_scale(tdp_scale)
		if prior_scale is not None:
			self.set_prior_scale(prior_scale)
		self.apply_fixes()

	def apply_fixes(self):
		for key in [
			"allow-for-silence-repetitions",
			"fix-allophone-context-at-word-boundaries",
			"fix-tdp-leaving-epsilon-arc",
			"transducer-builder-filter-out-invalid-allophones",
		]:
			self.config[".".join(self.acoustic_model_key + ["*", key])] = True if "repetitions" not in key else False
	
	def set_tdps(self, tdps):
		self.config[".".join(self.acoustic_model_key)] = tdps.to_acoustic_model_config()

	def set_prior_scale(self, prior_scale):
		key = ".".join(self.prior_scale_key)
		self.config[key] = prior_scale

	def set_tdp_scale(self, tdp_scale):
		key = ".".join(self.tdp_scale_key)
		self.config[key] = tdp_scale


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
		self.nn_priors = {}
		self.compile_graphs = {}

		self.decoders = {
			"default": self,
		}

		from ..clean import LatticeCleaner
		self.lattice_cleaner = LatticeCleaner()

		self.state_tying_mode = None

		self.all_segments = {}
	
	def add_overlay(self, origin, name):
		super().add_overlay(origin, name)
		self.nn_checkpoints[name] = {}
		self.nn_priors[name] = defaultdict(dict)
		self.compile_graphs[name] = dict()
	
	def set_corpus(self, name, *args, **kwargs):
		super().set_corpus(name, *args, **kwargs)
		self.nn_checkpoints[name] = {}
		self.nn_priors[name] = defaultdict(dict)
		self.compile_graphs[name] = dict()

	@property
	def csp(self):
		return self.crp
	
	@csp.setter
	def csp(self, crp):
		self.crp = crp

	def get_state_tying(self):
		return self.csp['base'].acoustic_model_config.state_tying.type
	
	def set_state_tying(
		self,
		value,
		cart_file: Optional[tk.Path] = None,
		use_boundary_classes=False,
		use_word_end_classes=False,
		hmm_partition=3,
	):
		assert value in {'monophone', 'cart', 'monophone-no-tying-dense', 'lut', 'lookup'}, "No other state tying types supported yet"
		if value == 'cart': assert cart_file is not None, "Cart file must be specified"
		for crp in self.crp.values():
			del crp.acoustic_model_config.state_tying
			crp.acoustic_model_config.state_tying.type = value
			crp.acoustic_model_config.state_tying.file = cart_file
			if use_boundary_classes or use_word_end_classes:
				crp.acoustic_model_config.state_tying.use_boundary_classes = use_boundary_classes
				crp.acoustic_model_config.state_tying.use_word_end_classes = use_word_end_classes
			if hmm_partition != 3:
				crp.acoustic_model_config.hmm.states_per_phone = hmm_partition
		
	def num_classes(self):
		if self.get_state_tying() in self._num_classes_dict:
			return self._num_classes_dict[self.get_state_tying()]
		if self.state_tying_mode is None:
			state_tying = DumpStateTyingJob(self.crp["train"]).out_state_tying
			# tk.register_output("state-tying_mono", state_tying)
			num_states = ExtractStateTyingStats(state_tying).out_num_states
			return num_states
		elif self.state_tying_mode == "dense":
			assert "no-tying-dense" in self.get_state_tying()
			num_labels = DelayedPhonemeInventorySize(self.crp["train"].lexicon_config.file)
			we = self.crp["base"].acoustic_model_config.state_tying.use_word_end_classes
			hmm_partition = self.crp["base"].acoustic_model_config.hmm.states_per_phone
			return num_labels * hmm_partition * (2 if we else 1)
		else:
			raise AssertionError("Unknown state tying mode %s" % self.state_tying_mode)
	
	def get_allophone_file(self):
		return StoreAllophonesJob(self.crp["train"]).out_allophone_file
	
	def get_state_tying_file(self, corpus="train"):
		return DumpStateTyingJob(self.crp[corpus]).out_state_tying
	
	def silence_idx(self):
		state_tying = DumpStateTyingJob(self.csp["train"]).out_state_tying
		return ExtractStateTyingStats(state_tying).out_silence_idx
	
	def set_num_classes(self, state_tying, num_classes):
		self._num_classes_dict[state_tying] = num_classes
	
	def set_decoder(self, key, decoder, **kwargs):
		assert hasattr(decoder, "decode"), "Decoder object must provide 'decode' method"
		self.decoder = decoder
		self.decoders[key] = decoder
		self.decoder.set_system(self, **kwargs)
	
	def set_trainer(self, trainer, **kwargs):
		assert hasattr(trainer, "train"), "Trainer object must provide 'train' method"
		self.trainer = trainer
		self.trainer.set_system(self, **kwargs)
	
	def _get_scorer(self, name, epoch, extra_suffix=None, optlm=None, prior=None):
		name = "scorer_crnn-{}-{}".format(name, epoch)
		if extra_suffix is not None:
			name += extra_suffix
		if prior is None:
			prior = name + "-prior" in self.jobs["dev"]
		if prior: name += "-prior"
		if optlm is None:
			optlm = name + "-optlm" in self.jobs["dev"]
		if optlm: name += "-optlm"
		return self.jobs["dev"][name]
	
	def get_wer(self, name, epoch, precise=False, **kwargs):
		scorer_job = self._get_scorer(name, epoch, **kwargs)
		if precise is False:
			return scorer_job.out_wer
		assert precise is True
		return scorer_job.out_num_errors / scorer_job.out_ref_words * 100
	
	def get_last_wer(self, name, **kwargs):
		pass

	def init_report_system(self, fname):
		# import
		from ..reports import ReportSystem
		self.report_system = ReportSystem(fname)
	
	def report(self, name, data, **kwargs):
		self.report_system.print(name, data, **kwargs)

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
	
	def filter_segments(self, alignment, alignment_logs=None):
		corpus = "train"
		all_segments = corpus_recipes.SegmentCorpusJob(self.corpora[corpus].corpus_file, 1)
		if alignment_logs is not None:
			import i6_experiments.users.mann.experimental.extractors as extr
			from i6_core.corpus.filter import FilterSegmentsByListJob
			filter_list = extr.ExtractAlignmentFailuresJob(alignment_logs).out_filter_list
			all_segments = FilterSegmentsByListJob({1: all_segments.out_single_segment_files[1]}, filter_list)
		new_segments = corpus_recipes.ShuffleAndSplitSegmentsJob(
			segment_file=all_segments.out_single_segment_files[1],
			split={ 'train': 1.0 - self.dev_size, 'dev': self.dev_size }
		)
		self.csp["train"].segment_path = all_segments.out_single_segment_files[1]
		self.csp["crnn_train"].segment_path = new_segments.out_segments['train']
		self.csp["crnn_dev"].segment_path = new_segments.out_segments['dev']

	def init_nn(self, name, corpus, dev_size, bad_segments=None, dump=False, alignment_logs=None):
		self.dev_size = dev_size
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
		segments = all_segments
		if alignment_logs is not None:
			import i6_experiments.users.mann.experimental.extractors as extr
			from i6_core.corpus.filter import FilterSegmentsByListJob
			filter_list = extr.ExtractAlignmentFailuresJob(alignment_logs).out_filter_list
			segments = FilterSegmentsByListJob({1: all_segments.out_single_segment_files[1]}, filter_list)
		new_segments = corpus_recipes.ShuffleAndSplitSegmentsJob(
			segment_file=segments.out_single_segment_files[1],
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

		self.all_segments[corpus] = all_segments.out_single_segment_files[1]
		self.jobs[corpus][              'all_segments_%s' % name] = all_segments
		self.jobs[corpus]['all_segments'] = all_segments
		self.jobs[corpus]['shuffle_and_split_segments_%s' % name] = new_segments

		if 'train_corpus' not in self.default_nn_training_args:
			self.default_nn_training_args['train_corpus'] = '%s_train' % name
		if 'dev_corpus'   not in self.default_nn_training_args:
			self.default_nn_training_args['dev_corpus']   = '%s_dev'   % name
		if 'feature_flow' not in self.default_nn_training_args:
			self.default_nn_training_args['feature_flow'] = 'mfcc+context%d' % window_size

		if 'feature_dimension' not in self.default_scorer_args:
			self.default_scorer_args['feature_dimension'] = window_size * 16

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
	
	def compile_model(
		self,
		returnn_config,
		alias=None,
		adjust_output_layer="output",
		use_global_binaries=True,
		**compile_args,
	):
		if hasattr(returnn_config, "build"):
			config = returnn_config.build()
		else:
			config = copy.deepcopy(returnn_config)
		config_dict = config.config
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
			config_dict['target'] = 'classes'
		config_dict['num_outputs']['classes'] = [self.num_classes(), 1]
		if not use_global_binaries:
			compile_args = {
				"returnn_python_exe": self.returnn_python_exe,
				"returnn_root": self.returnn_root,
				**compile_args,
			}
		compile_graph_job = crnn.CompileTFGraphJob(
			config, **compile_args
		)
		if alias is not None:
			self.compile_graphs["train"][alias] = compile_graph_job.out_graph, lstm_flag
			alias = f"compile_returnn/{alias}"
			compile_graph_job.add_alias(alias)
			self.jobs["train"][alias.replace("/", "_")] = compile_graph_job
		return compile_graph_job.out_graph, lstm_flag

	def get_rasr_tf_flow(
			self, feature_flow, graph, model,
			output_tensor_name=None, 
			append=False,
			req_libraries=NotSpecified,
			add_tf_flow=True,
		):
		if output_tensor_name is None:
			output_tensor_name = 'output/output_batch_major'

		model_path = tk.uncached_path(model.model)[:-5]
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

		tf_flow.config[tf_fwd].input_map.info_0.param_name             = 'features'
		tf_flow.config[tf_fwd].input_map.info_0.tensor_name            = 'extern_data/placeholders/data/data'
		tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name = 'extern_data/placeholders/data/data_dim0_size'

		tf_flow.config[tf_fwd].output_map.info_0.param_name  = 'log-posteriors'
		tf_flow.config[tf_fwd].output_map.info_0.tensor_name = output_tensor_name

		tf_flow.config[tf_fwd].loader.type             = 'meta'
		tf_flow.config[tf_fwd].loader.meta_graph_file  = tf_graph
		tf_flow.config[tf_fwd].loader.saved_model_file = sprint.StringWrapper(model_path, model)

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

	def get_tf_flow(
			self, feature_flow, crnn_config, model, output_tensor_name=None , 
			append=False, drop_layers=None, adjust_output_layer="output",
			req_libraries=NotSpecified, alias=None, add_tf_flow=True,
			graph=None, 
			**compile_args
		):
		if graph is None:
			graph, lstm_flag = self.compile_model(
				crnn_config, alias=alias,
				adjust_output_layer=adjust_output_layer,
				**compile_args,
			)
		else:
			graph, lstm_flag = graph
		
		if output_tensor_name is None:
			output_tensor_name = 'output/output_batch_major'
		if not add_tf_flow:
			return feature_flow

		# tf_graph = compile_graph_job.out_graph
		model_path = tk.uncached_path(model.model)[:-5]
		tf_graph = graph

		# Setup TF AM flow node
		tf_flow = sprint.FlowNetwork()
		tf_flow.add_input('input-features')
		tf_flow.add_output('features')
		tf_flow.add_param('id')
		tf_fwd = tf_flow.add_node('tensorflow-forward', 'tf-fwd', {'id': '$(id)'})
		tf_flow.link('network:input-features', tf_fwd + ':features')
		tf_flow.link(tf_fwd + ':log-posteriors', 'network:features')

		tf_flow.config = sprint.RasrConfig()

		tf_flow.config[tf_fwd].input_map.info_0.param_name             = 'features'
		tf_flow.config[tf_fwd].input_map.info_0.tensor_name            = 'extern_data/placeholders/data/data'
		tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name = 'extern_data/placeholders/data/data_dim0_size'

		tf_flow.config[tf_fwd].output_map.info_0.param_name  = 'log-posteriors'
		tf_flow.config[tf_fwd].output_map.info_0.tensor_name = output_tensor_name

		tf_flow.config[tf_fwd].loader.type             = 'meta'
		tf_flow.config[tf_fwd].loader.meta_graph_file  = tf_graph
		tf_flow.config[tf_fwd].loader.saved_model_file = sprint.StringWrapper(model_path, model)

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
		# if self.returnn_root is not None:
		# 	train_args['returnn_root'] = self.returnn_root
		# if self.returnn_python_exe is not None:
		# 	train_args['returnn_python_exe'] = self.returnn_python_exe
		train_args.setdefault('returnn_root', self.returnn_root)
		train_args.setdefault('returnn_python_exe', self.returnn_python_exe)

		with tk.block('NN - %s' % name):
			self.train_nn(**train_args)
			train_job.add_alias('nn_%s' % name)
			tk.register_output('plot_se_%s.png' % name, train_job.plot_se)
			tk.register_output('plot_lr_%s.png' % name, train_job.plot_lr)

	def nn_and_recog(
		self, name, training_args, crnn_config, scorer_args,  
		recognition_args,
		epochs=None,
		reestimate_prior='CRNN',
		compile_args=None,
		optimize=True, use_tf_flow=False,
		alt_training=False, dump_epochs=None,
		compile_crnn_config=None,
		alt_decoding=False,
		train_prefix=None,
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
				reestimate_prior = {"CRNN": "alt"}.get(reestimate_prior, reestimate_prior)
				self.trainer.train(**train_args)
			else:
				self.train_nn(**train_args)
			train_job = self.jobs[train_args['feature_corpus']]['train_nn_%s' % name]
			alias_prefix = train_prefix or "train_nn"
			train_job.add_alias(os.path.join(alias_prefix, name))
			self.nn_checkpoints[train_args['feature_corpus']][name] = train_job.out_checkpoints
			tk.register_output('plot_se_%s.png' % name, train_job.out_plot_se)
			tk.register_output('plot_lr_%s.png' % name, train_job.out_plot_lr)

			tk.register_output("nn_configs/{}/returnn.config".format(name), train_job.out_returnn_config_file)

			# default decoding procedure
			for epoch in epochs:
				kwargs = locals().copy()
				del kwargs["self"]
				kwargs["training_args"] = kwargs.pop("train_args")
				assert use_tf_flow, "Otherwise not supported"
				self.decode(_adjust_train_args=False, **kwargs)

			# alternative decoding procedure
			if not alt_decoding:
				assert train_job
				return train_job
			if isinstance(alt_decoding, bool) and alt_decoding:
				alt_decoding = {}
			alt_decoding_epochs = alt_decoding.pop("epochs", epochs)
			for epoch in alt_decoding_epochs:
				kwargs = locals().copy()
				del kwargs["self"], kwargs["alt_decoding_epochs"]
				kwargs["training_args"] = kwargs.pop("train_args")
				kwargs["decoding_args"] = kwargs.pop("alt_decoding")
				self.decoder.decode(**kwargs)
			
			return train_job
	
	def extract_prior(self, name, crnn_config, training_args, epoch, bw=False):
		# alignment = None
		# if score_args.pop("use_alignment", True):
		if crnn_config is None:
			crnn_config = self.nn_config_dicts["train"][name]
		alignment = select_element(self.alignments, training_args["feature_corpus"], training_args["alignment"])
		num_classes  = self.functor_value(training_args["num_classes"])
		if bw:
			assert "fast_bw" in crnn_config.config["network"]
			crnn_config = copy.deepcopy(crnn_config)
			crnn_config.config["network"]["bw_prior"] = {
				"class": "eval", "from": "fast_bw", "eval": "source(0) + eps", "eval_locals": {"eps": 1e-10}
			}
			crnn_config.config["forward_output_layer"] = "bw_prior"
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
		self,
		name, epoch,
		crnn_config,
		training_args, scorer_args,
		recognition_args,
		extra_suffix=None,
		recog_name=None,
		compile_args=None,
		compile_crnn_config=None,
		reestimate_prior=False,
		prior_config=None,
		optimize=True,
		clean=True,
		_feature_scorer=None,
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
			# recog_name = "-".join([recog_name, extra_suffix])
			recog_name = recog_name + extra_suffix

		if ".tuned" in recog_name:
			print(recognition_args["search_parameters"])

		score_args = dict(**self.default_scorer_args)
		score_args.update(scorer_args)
		if isinstance(reestimate_prior, str) and reestimate_prior.startswith("alt"):
			extract_prior = self.trainer.extract_prior
		else:
			extract_prior = self.extract_prior
		if reestimate_prior in {'CRNN', 'alt', "bw", "alt-bw", "alt-CRNN"}:
			self.jobs[training_args["feature_corpus"]]["returnn_compute_prior_%s" % scorer_name] \
				= score_features = extract_prior(name, prior_config or crnn_config, training_args, epoch, bw=reestimate_prior.endswith("bw"))
			self.nn_priors[training_args["feature_corpus"]][name][epoch] = score_features.out_prior_xml_file
			scorer_name += '-prior'
			score_args['name'] = scorer_name
			score_args['prior_file'] = score_features.out_prior_xml_file
		elif reestimate_prior == 'bw':
			pass
		elif reestimate_prior == 'transcription':
			score_args['prior_file'] = self.prior_system.prior_xml_file
		elif reestimate_prior == True:
			assert False, "reestimating prior using sprint is not yet possible, you should implement it or use 'CRNN'"

		recog_args = copy.deepcopy(self.default_recognition_args)
		# copy feature flow used in training
		# recog_args['flow'] = training_args['feature_flow']
		if 'search_parameters' in recognition_args:
			# if ".tuned" in recog_name:
			# 	print(recognition_args["search_parameters"])
			recog_args['search_parameters'].update(recognition_args['search_parameters'])
			# if ".tuned" in recog_name:
			# 	print(recog_args["search_parameters"])
			remaining_args = copy.copy(recognition_args)
			del remaining_args['search_parameters']
			assert "search_parameters" not in remaining_args, "search_parameters should not be in remaining args"
			recog_args.update(remaining_args)
			# if ".tuned" in recog_name:
			# 	print(recog_args["search_parameters"])
		else:
			recog_args.update(recognition_args)
		recog_args['name']           = 'crnn-%s%s' % (recog_name, '-prior' if reestimate_prior else '')
		if 'flow' not in recog_args:
			feature_flow_net = meta.system.select_element(self.feature_flows, recog_args['corpus'], training_args['feature_flow'])
			model = self.jobs[training_args['feature_corpus']]['train_nn_%s' % name].out_models[epoch]
			compile_args = copy.deepcopy(compile_args)
			if compile_crnn_config is None:
				compile_crnn_config = crnn_config
			compile_network_extra_config = compile_args.pop('compile_network_extra_config', {})
			compile_crnn_config.config['network'].update(compile_network_extra_config)
			compile_args.setdefault("alias", name)

			graph = compile_args.pop("graph", None)
			if graph is not None:
				graph = self.compile_graphs[training_args['feature_corpus']][graph]

			recog_args['flow'] = tf_flow = self.get_tf_flow(
				feature_flow_net, compile_crnn_config, model, 
				output_tensor_name=recog_args.pop('output_tensor_name', None), 
				drop_layers=recog_args.pop('drop_layers', None),
				graph=graph,
				req_libraries=recog_args.pop('req_libraries', NotSpecified),
				**compile_args,
			)
		if score_args['prior_mixtures'] is not None:
			prior_mixtures = select_element(self.mixtures, training_args['feature_corpus'], score_args['prior_mixtures']) 
		else:
			from i6_core.mm import CreateDummyMixturesJob
			prior_mixtures = CreateDummyMixturesJob(self.num_classes(), self.num_input).out_mixtures
		if _feature_scorer is not None:
			assert callable(_feature_scorer)
			recog_args['feature_scorer'] = scorer = _feature_scorer(
				name=recog_name,
				prior_mixtures=prior_mixtures,
				priori_scale=score_args.get('prior_scale', 0.),
				prior_file=select_element(self.nn_priors, training_args['feature_corpus'], score_args.get("prior_file", None), epoch),
				scale=score_args.get('mixture_scale', 1.0),
			)
		else:
			if 'feature_scorer' not in recog_args:
				recog_args['feature_scorer'] = scorer = sprint.PrecomputedHybridFeatureScorer(
					prior_mixtures=prior_mixtures,
					priori_scale=score_args.get('prior_scale', 0.),
					# prior_file=score_features.prior if reestimate_prior else None,
					prior_file=select_element(self.nn_priors, training_args['feature_corpus'], score_args.get("prior_file", None), epoch),
					scale=score_args.get('mixture_scale', 1.0),
				)
			else:
				scorer = recog_args['feature_scorer']
		self.feature_scorers[training_args['feature_corpus']][scorer_name] = scorer

		extra_rqmts = recog_args.pop('extra_rqmts', {})
		# reset tdps for recognition
		with tk.block('recog-V%d' % epoch):
			js = []
			wer = None
			if optimize:

				if ".tuned" in recog_name:
					print(recog_args["search_parameters"])
				self.recog_and_optimize(**recog_args)
				opt_recog_job = self.jobs[recog_args["corpus"]]['recog_%s' % recog_args["name"] + "-optlm"]
				opt_recog_job.rqmt.update(extra_rqmts)
				wer_job = self.jobs[recog_args["corpus"]]['scorer_%s' % recog_args["name"] + "-optlm"]
				js.append(opt_recog_job)
			else:
				self.recog(**recog_args)
				wer_job = self.jobs[recog_args["corpus"]]['scorer_%s' % recog_args["name"]]
			recog_job = self.jobs[recog_args["corpus"]]['recog_%s' % recog_args["name"]]
			recog_job.rqmt.update(extra_rqmts)
			js.append(recog_job)
		
		if clean:
			assert wer_job
			for j in js:
				self.lattice_cleaner.clean(j, wer_job.out_wer)

	def nn_align(
		self, nn_name, crnn_config, epoch, scorer_suffix='', mem_rqmt=8,
		compile_crnn_config=None, name=None, feature_flow='gt', dump=False,
		graph=None, time_rqmt=4, flow=None, feature_corpus="train",
		compile_args=None, evaluate=False, feature_scorer=None, **kwargs
	):
		# get custom tf
		if flow is None:
			compile_args = compile_args or {}
			if compile_crnn_config is None:
				compile_crnn_config = crnn_config
			feature_flow_net = meta.system.select_element(self.feature_flows, 'train', feature_flow)
			model = self.jobs[feature_corpus]['train_nn_%s' % nn_name].out_models[epoch]
			if graph is not None:
				graph = self.compile_graphs['train'][graph]
			flow = self.get_tf_flow(
				feature_flow_net,
				compile_crnn_config,
				model,
				graph=graph,
				**compile_args
			)

		# get alignment
		model_name = '%s-%s' % (nn_name, epoch)
		name = name or model_name
		alignment_name = 'alignment_%s' % name
		feature_scorer = model_name + scorer_suffix if feature_scorer is None else feature_scorer
		self.align(
			name=name if not name else name,
			corpus='train', 
			flow=flow,
			feature_scorer=meta.select_element(self.feature_scorers, feature_corpus, feature_scorer),
			**kwargs,
		)
		j = self.jobs['train'][alignment_name]
		j.rqmt['mem'] = mem_rqmt
		if time_rqmt is not None:
			j.rqmt['time'] = time_rqmt
		j.add_alias("align/%s" % name)
		if evaluate:
			if not isinstance(evaluate, dict):
				evaluate = {}
			stats = self.evaluate_alignment(name, corpus='train', **evaluate)
			return stats
		if dump:
			from recipe.mm import DumpAlignmentJob
			j = DumpAlignmentJob(self.csp['train'], self.feature_flows['train'][feature_flow], self.alignments['train'][name])
			tk.register_output(alignment_name, j.alignment_bundle)
			return j.alignment_bundle
		return None
  
	def evaluate_alignment(self, name, corpus, alignment=None, alignment_logs=None, ref_alignment="init_align"):
		from i6_core.corpus import FilterSegmentsByAlignmentConfidenceJob
		from i6_experiments.users.mann.experimental.statistics.alignment import ComputeTseJob
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
				pass
			try:
				alignment = alignment.value
			except AttributeError:
				pass
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
			self.get_allophone_file(),
			self.csp["train"].segment_path.hidden_paths,
			self.csp["train"].concurrent
		)
		asj = AlignmentStatisticsJob(*args)
		asj.add_alias(f"alignment_stats-{name}")
		stats = asj.counts
		tk.register_output("stats_align/counts/{}".format(name), stats)

		# compute TSE
		tse = ComputeTseJob(
			alignment,
			meta.select_element(self.alignments, corpus, ref_alignment),
			self.get_allophone_file()
		)
		tse.add_alias(f"tse-{name}")
		tk.register_output("stats_align/tse/{}".format(name), tse.out_tse)
		return stats

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
		self.ce_args = {}
		self.scales = {}
		self.encoder = None
		self.loss = "bw"
		self.prior = "povey"
		self.transforms = []
		self.updates = {}
		self.deletions = []
	
	def register(self, name):
		self.system.builders[name] = self.copy()

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
	
	def set_ce_args(self, **ce_args):
		self.ce_args = ce_args
		return self
	
	def set_pretrain(self, **kwargs):
		raise NotImplementedError()
		return self
	
	def set_transcription_prior(self):
		# self.transforms.append(self.system.prior_system.add_to_config)
		self.prior = "transcription"
		return self
	
	def set_loss(self, loss="bw"):
		assert loss in ["bw", "viterbi"]
		self.loss = loss
		return self
	
	def set_povey_prior(self):
		# self.transforms = [
		# 	t for t in self.transforms if t != self.system.prior_system.add_to_config
		# ]
		self.prior = "povey"
		return self
	
	def set_no_prior(self):
		self.prior = None
		return self
	
	def set_scales(self, am=None, prior=None, tdp=None):
		self.scales.update({
			m + "_scale": v
			for m, v in locals().items()
			if m != "self" and v is not None
		})
		return self
		
	def set_tina_scales(self):
		self.scales = {
			"am_scale": 0.3,
			"prior_scale": 0.1,
			"tdp_scale": 0.1
		}
		return self
	
	def copy(self):
		new_instance = type(self)(self.system)
		new_instance.config_args = self.config_args.copy()
		new_instance.network_args = self.network_args.copy()
		new_instance.scales = self.scales.copy()
		new_instance.ce_args = self.ce_args.copy()
		new_instance.transforms = self.transforms.copy()
		new_instance.updates = self.updates.copy()
		new_instance.deletions = self.deletions.copy()
		new_instance.encoder = self.encoder
		new_instance.prior = self.prior
		new_instance.loss = self.loss
		return new_instance
	
	def set_oclr(self, dur=None, **kwargs):
		from i6_experiments.users.mann.nn.learning_rates import get_learning_rates
		dur = dur or int(0.8 * max(self.system.default_epochs))
		self.update({
			"learning_rates": get_learning_rates(
				increase=dur, decay=dur, **kwargs
			),
			"newbob_multi_num_epochs" : self.system.default_nn_training_args["partition_epochs"].get("train", 1),
			"newbob_multi_update_interval" : 1,
		})
		return self
	
	def set_specaugment(self):
		from i6_experiments.users.mann.nn import specaugment
		self.transforms.append(specaugment.set_config)
		return self
	
	def update(self, *args, **kwargs):
		self.updates.update(*args, **kwargs)
		return self
	
	def delete(self, *args):
		self.deletions += args
		return self
	
	def build(self):
		from i6_experiments.users.mann.nn import BASE_BW_LRS
		from i6_experiments.users.mann.nn import prior, pretrain, bw, get_learning_rates
		kwargs = BASE_BW_LRS.copy()
		kwargs.update(self.config_args)

		if self.encoder is viterbi_lstm:
			viterbi_config_dict = self.encoder(
				self.system.num_input,
				network_kwargs=self.network_args,
				ce_args=self.ce_args,
				**kwargs)
		else:
			viterbi_config_dict = self.encoder(
				self.system.num_input,
				network_kwargs=self.network_args,
				**kwargs)

		assert "chunking" in viterbi_config_dict.config

		assert self.prior in ["povey", "transcription", None], "Unknown prior: {}".format(self.prior)

		if self.loss == "bw":
			config = bw.ScaleConfig.copy_add_bw(
				viterbi_config_dict, self.system.csp["train"],
				num_classes=self.system.num_classes(),
				prior=self.prior,
				**self.scales,
			)
		elif self.loss == "viterbi":
			config = bw.ScaleConfig.from_config(viterbi_config_dict)
		else:
			raise ValueError("Unknown loss: {}".format(self.loss))

		if self.prior == "transcription":
			assert self.loss == "bw"
			self.system.prior_system.add_to_config(config)

		config.config.update(copy.deepcopy(self.updates))

		for key in self.deletions:
			del config.config[key]

		for transform in self.transforms:
			transform(config)
		
		return config
	
	def build_compile_config(self):
		viterbi_config = self.encoder(
			self.system.num_input,
			network_kwargs=self.network_args,
		)

		net = viterbi_config.config["network"]
		for key in ["loss", "loss_opts", "targets"]:
			net["output"].pop(key, None)
		net["output"]["n_out"] = self.system.num_classes()

		pruned_config_dict = {
			k: v for k, v in viterbi_config.config.items()
			if k in ["network", "num_outputs", "extern_data"]
		}

		return crnn.ReturnnConfig(pruned_config_dict)


class NNSystem(BaseSystem):
	def __init__(self, num_input=None, epochs=None, rasr_binary_path=Default, **kwargs):
		if rasr_binary_path is Default:
			p = gs.RASR_ROOT
			if isinstance(p, str):
				p = tk.Path(p)
			rasr_binary_path = p.join_right('arch/linux-x86_64-standard')
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

		from .. import dump
		self.dump_system: dump.HdfDumpster = None

		from i6_core.tools import CloneGitRepositoryJob
		self.cleaner_returnn_root = CloneGitRepositoryJob(
			"https://github.com/DanEnergetics/returnn.git",
			branch="mann-cleanup-mod"
		).out_repository

		from i6_experiments.users.mann.nn.config import viterbi_lstm
		self.baselines = {
			"viterbi_lstm": lambda: viterbi_lstm(num_input),
		}

		# config builders attached to the system
		self.default_builder = ConfigBuilder(self)
		self.builders = {}
	
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

	def init_prior_system(system, total_frames, **kwargs):
		from .. import prior
		system.prior_system = prior.PriorSystem(system, total_frames, **kwargs)
	
	def clean(self, training_name, epochs, exec_immediately=False, cleaner_args=None, feature_corpus="train"):
		from i6_core.returnn import WriteReturnnConfigJob
		from i6_experiments.users.mann.experimental.cleaner import ReturnnCleanupOldModelsJob, ReturnnCleanerConfig
		training_job = self.jobs[feature_corpus]["train_nn_%s" % training_name]

		cleaner_args = cleaner_args or {}
		cleaner_args.setdefault("returnn_root", self.cleaner_returnn_root)

		cleaner_config = ReturnnCleanerConfig.from_epochs(epochs)

		model=training_job.out_model_dir.join_right("epoch")
		if exec_immediately:
			model._available = lambda *args, **kwargs: True 
		print("Path available: ", model.creator.path_available(model))
		j = ReturnnCleanupOldModelsJob(
			cleaner_config,
			scores=training_job.out_learning_rates if not exec_immediately else None,
			model=model,
			**cleaner_args,
		)
		j.set_vis_name("Clean {}".format(training_name))
		tk.register_output(".cleaner/%s.log" % training_name, j.out_log_file)
	
	def run_exp(
		self,
		name: str,
		exp_config: ExpConfig,
		returnn_config: Optional[crnn.ReturnnConfig] = None,
		**kwargs
	):
		kwargs = ChainMap(kwargs, exp_config.__dict__)
		assert not (returnn_config and kwargs.get("crnn_config", None))
		if returnn_config:
			kwargs["crnn_config"] = returnn_config
		return self.nn_and_recog(
			name, **kwargs
		)
	
	def run_decode(
		self,
		name: str,
		exp_config: ExpConfig,
		type="default",
		decoding_args={},
		**kwargs
	):
		kwargs = ChainMap(kwargs, exp_config.__dict__)
		try:
			decoder = self.decoders[type]
		except KeyError:
			raise ValueError("Unknown decoder type {}".format(type))
		decoder.decode(
			name, decoding_args=decoding_args, **kwargs
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
		alt_training=False,
		dump_epochs=None, dump_args=None,
		dump_config=None,
		qsub_args=None,
		**kwargs
	):
		# experimental
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
		self.compile_configs[name] = compile_crnn_config
		training_args = copy.deepcopy(training_args) or {}
		crnn_config = copy.deepcopy(crnn_config)
		fast_bw_args = copy.deepcopy(fast_bw_args)
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
			if (
				isinstance(crnn_config, bw.ScaleConfig) and \
					any(layer['class'] == 'fast_bw' for layer in crnn_config.config['network'].values())
				) and 'additional_rasr_config_files' not in training_args:
				additional_sprint_config_files, additional_sprint_post_config_files \
					= add_fastbw_configs(self.csp[fast_bw_args.pop("corpus", "train")], **fast_bw_args) # TODO: corpus dependent and training args
				if alt_training:
					assert isinstance(crnn_config, bw.ScaleConfig)
					crnn_config = crnn_config.set_rasr_config(
						additional_sprint_config_files["fastbw"],
						additional_sprint_post_config_files["fastbw"],
						save_under=name
					)
				else:
					training_args = {
						**training_args,
						'additional_rasr_config_files':      additional_sprint_config_files,
						'additional_rasr_post_config_files': additional_sprint_post_config_files,
					}

				
			# del tdp from train config
			self.crp["train"].acoustic_model_config = self.crp["base"].acoustic_model_config._copy()
			del self.crp["train"].acoustic_model_config.tdp

			if not dump_config:
				dump_config = copy.deepcopy(crnn_config)
			# call training
			j = super().nn_and_recog(
				name, training_args, crnn_config, scorer_args, recognition_args, epochs=epochs, 
				reestimate_prior=reestimate_prior, optimize=optimize, use_tf_flow=use_tf_flow, compile_args=compile_args,
				compile_crnn_config=compile_crnn_config, alt_training=alt_training,
				**kwargs)

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
					**(dump_args or {}),
				)
			
			return j

	
	def nn_align(self, nn_name, epoch, crnn_config=None, compile_crnn_config=None, plugin_args=MappingProxyType({}), **kwargs):
		if crnn_config is None and compile_crnn_config is None:
			compile_crnn_config = self.compile_configs[nn_name]
		if crnn_config is None:
			crnn_config = self.nn_config_dicts['train'][nn_name]
		with safe_crp(self) as tcsp:
			for plugin, args in plugin_args.items():
				self.plugins[plugin].apply(**args)
			return super().nn_align(nn_name, crnn_config, epoch, compile_crnn_config=compile_crnn_config, **kwargs)
