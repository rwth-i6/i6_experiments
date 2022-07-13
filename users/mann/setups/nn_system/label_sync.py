from sisyphus import *

import copy

import recipe.meta as meta
import recipe.sprint as sprint
import recipe.recognition as recog

from collections import ChainMap
from recipe.setups.mann.nn_system.base_system import NNSystem
from contextlib import contextmanager

NonExistent = object()

_cuda_version = "9.1"
_cudnn_version = "7.1"
_tf_path = '/work/asr3/michel/rawiel/tensorflow/zhou2/tensorflow/'
_env = {}
_env.update(
    # CRNN_ROOT = '/u/zhou/RETURNN',
    # SPRINT_ARCH += ''
    SPRINT_ARCH = 'linux-x86_64-standard-label_sync_decoding',
    SPRINT_ROOT = '/u/zhou/rasr-dev/',
    CRNN_ROOT = '/u/mann/src/returnn_zhou',
    CRNN_PYTHON_HOME = '/u/zhou/softwares/python/3.6.1',
    CRNN_PYTHON_EXE = '/u/zhou/softwares/python/3.6.1/bin/python3.6',
    DEFAULT_ENVIRONMENT_SET = {
        'MKL_NUM_THREADS' : 1,
        'OMP_NUM_THREADS' : 1,
        'PATH'            : ':'.join(['/usr/local/cuda-{}/bin'.format(_cuda_version),
                                    '/usr/local/cuda-{}/lib64'.format(_cuda_version),
                                    '/rbi/sge/bin', '/rbi/sge/bin/lx-amd64',
                                    '/usr/local/sbin', '/usr/local/bin',
                                    '/usr/sbin', '/usr/bin',
                                    '/sbin', '/bin',
                                    '/usr/games', '/usr/local/games',
                                    '/snap/bin']),
        'LD_LIBRARY_PATH' : ':'.join(['/usr/local/acml-4.4.0/cblas_mp/lib',
                                    '/usr/local/acml-4.4.0/gfortran64_mp/lib/',
                                    '/usr/local/acml-4.4.0/gfortran64/lib',
                                    '/usr/local/cuda-{}/extras/CUPTI/lib64/'.format(_cuda_version),
                                    '/usr/local/cuda-{}/lib64'.format(_cuda_version),
                                    _tf_path,
                                    '/usr/local/cudnn-{}-v{}/lib64'.format(_cuda_version, _cudnn_version)]),
        'CUDA_PATH'       : '/usr/local/cuda-{}'.format(_cuda_version),
        'TMPDIR'          : '/var/tmp'}
)

def get_override(d, key, value):
  old_value = d.get(key, NonExistent)
  d[key] = value
  return old_value

def saved_update(d: dict, updict: dict):
  rev_dict = {}
  for key, value in updict.items():
    rev_dict[key] = get_override(d, key, value)
  return rev_dict

def revert_dict(d: dict, rev_dict: dict):
  for key, value in rev_dict.items():
    if value is NonExistent:
      d.pop(key)
      continue
    d[key] = value


@contextmanager
def label_sync_decoding_binaries():
  saved_env = saved_update(gs.__dict__, _env)
  yield
  revert_dict(gs.__dict__, saved_env)

class LabelSyncDecoder:
    
    default_lm_scale = 5.0
    default_pruning_factor = 18.0
    default_search_options = {
        'label-pruning'          : default_lm_scale * default_pruning_factor,
        'label-pruning-limit'    : 5000,
        'word-end-pruning'       : 0.5,
        'word-end-pruning-limit' : 500,
        'create-lattice'         : True,
        'optimize-lattice'       : False
    }
    default_label_lookahead_options = {
        'scale'             : 1.0,
        'history-limit'     : 1,
        'cache-size-low'    : 2000,
        'cache-size-high'   : 3000
    }
    default_label_scorer_args = {
        # 'num_classes': 211,
        'extra_args' : {
            'transform-output-negate' : True,
            'max-batch-size'          : 64
        }
    }

    def __init__(self, num_classes):
        self.default_label_scorer_args["num_classes"] = num_classes
    
    def set_system(self, system, pass_classes=False, pass_lm_scale=False, pass_pruning=False):
        if pass_classes:
            self.default_label_scorer_args["num_classes"] \
                = system.functor_value(system.default_nn_training_args["num_classes"])
        if pass_lm_scale:
            self.default_lm_scale = l = system.default_recognition_args["lm_scale"]
            self.default_search_options["label-pruning"]
        if pass_pruning:
            self.default_pruning_factor = p = system.default_recognition_args["search_options"]["beam-pruning"]
        # self.default_search_options["label-pruning"] = l * p
        self.system = system
    
    def recog_label_sync(self, name, corpus, flow, label_scorer, lm_scale,
            label_tree=None, pronunciation_scale=0.0, rtf=None,
            parallelize_conversion=False, lattice_to_ctm_kwargs=None,
            prefix="", lattice_to_cpronunciation_scale=0.0, skip_silence=False,
            search_parameters=None, lm_lookahead=True, lm_lookahead_options=None,
            separate_lookahead_lm=False, lookahead_lm_config=None,
            separate_recombination_lm=False, recombination_lm_config=None,
            allow_label_recombination=True, label_recombination_limit=0,
            allow_word_end_recombination=True, word_end_recombination_limit=None,
            allow_label_loop=True, allow_blank_label=False,
            include_blank_label_score=None, blank_label_probability_threshold=None,
            fixed_beam_search=False, eos_threshold=None, cleanup_interval=100,
            length_normalization=False, normalize_label_only=False, 
            step_re_normalization=False, full_sum_decoding=False, 
            additional_cn_decoding=False, fill_empty_cn=True,
            feature_scorer=None,
            use_gpu=False, gpu_dependent_job=False, kaldi_scoring=False, bpe_to_words=False,
            qsub_args=None, week_queue=True, day_queue=False, extra_mem=4, num_proc=3):
        import recipe.recognition as recog
        if lattice_to_ctm_kwargs is None:
            lattice_to_ctm_kwargs = {}
        # assert corpus_name in stmFiles.keys(), 'unknown corpus name %s' %corpus_name

        am_config = self.system.csp[corpus].acoustic_model_config
        lm_config = self.system.csp[corpus].language_model_config

        extra_config = sprint.SprintConfig()
        if pronunciation_scale > 0:
            extra_config.flf_lattice_tool.network.recognizer.pronunciation_scale = pronunciation_scale

        # allow no silence in the lexicon
        if skip_silence:
            extra_config['*'].skip_silence = True

        if separate_lookahead_lm and lookahead_lm_config is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.separate_lookahead_lm = True
            extra_config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm = lookahead_lm_config
        
        if separate_recombination_lm and recombination_lm_config is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.separate_recombination_lm = True
            extra_config.flf_lattice_tool.network.recognizer.recognizer.recombination_lm = recombination_lm_config

        if allow_label_recombination:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.allow_label_recombination = True
        if label_recombination_limit is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.label_recombination_limit = label_recombination_limit

        # default true is RASR
        if allow_word_end_recombination is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.allow_word_end_recombination = allow_word_end_recombination
        if word_end_recombination_limit is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.word_end_recombination_limit = word_end_recombination_limit

        if allow_label_loop:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.allow_label_loop = True

        if allow_blank_label:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.allow_blank_label = True
        if include_blank_label_score is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.include_blank_label_score = include_blank_label_score
        if blank_label_probability_threshold is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.blank_label_probability_threshold = blank_label_probability_threshold    

        if full_sum_decoding:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.full_sum_decoding = True
        

        if fixed_beam_search:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.fixed_beam_search = True
        if eos_threshold is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.eos_threshold = eos_threshold

        if cleanup_interval is not None:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.cleanup_interval = cleanup_interval

        # only one of the two
        if length_normalization:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.length_normalization = True
        elif step_re_normalization:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.step_re_normalization = True

        if normalize_label_only:
            extra_config.flf_lattice_tool.network.recognizer.recognizer.normalize_label_only = True

        # extra_config_builder = sprint.ConfigBuilder()
        # extra_config.flf_lattice_tool.network.recognizer.recognizer._update(extra_config_builder(**recognizer_args))

        # different job hash based on gpu option #
        if gpu_dependent_job:
            extra_config.gpu_job_flag.use_gpu = use_gpu

        extra_post_config = sprint.SprintConfig()
        extra_post_config.flf_lattice_tool.network.evaluator.algorithm = 'bellman-ford'

        if feature_scorer is not None:
            feature_scorer.apply_config('flf-lattice-tool.network.recognizer.acoustic-model.mixture-set', extra_config, extra_post_config)
            extra_config.flf_lattice_tool.network.recognizer.use_mixture = True

        if rtf is None:
            duration   = self.system.csp[corpus].corpus_duration
            concurrent = self.system.csp[corpus].concurrent
            if week_queue: # one week
                rtf = int( 168 / (duration / concurrent) )
            elif day_queue: # adapt to maximum 1-day job
                rtf = int( 24 / (duration / concurrent) )
            else: # decoding queue of maximum 5 hours
                rtf = int( 5 / (duration / concurrent) )

        mem_rqmt = 4.0
        if lm_config.type in ('combine', 'tfrnn'):
            mem_rqmt += 4.0
            mem_rqmt += extra_mem

        self.system.csp[corpus].language_model_config.scale = lm_scale

        if label_tree is None:
            label_tree = sprint.LabelTree(label_unit='hmm', use_transition_penalty=True)
        kwargs = {"search_options"       : search_parameters,
                "lm_lookahead"         : lm_lookahead,
                "lookahead_options"    : lm_lookahead_options,
                "use_gpu"              : use_gpu,
                "rtf"                  : rtf,
                "mem"                  : mem_rqmt,
                "extra_config"         : extra_config,
                "extra_post_config"    : extra_post_config 
                }

        rec = recog.LabelSyncSearchJob(
            csp           = self.system.csp[corpus],
            feature_flow  = meta.select_element(self.system.feature_flows, corpus, flow),
            label_scorer  = meta.select_element(self.system.feature_scorers, corpus, label_scorer),
            #  label_scorer  = select_element(self.feature_scorers, corpus, feature_scorer),
            label_tree    = label_tree,
            **kwargs
        )
        rec.set_vis_name('Recog %s%s' % (prefix, name))
        rec.add_alias('%srecog_%s' % (prefix, name))
        self.system.jobs[corpus]['recog_%s' % name] = rec

        self.system.jobs[corpus]['lat2ctm_%s' % name] \
            = lat2ctm \
            = recog.LatticeToCtmJob(
                csp           = self.system.csp[corpus],
                lattice_cache = rec.lattice_bundle,
                parallelize   = parallelize_conversion,
                **lattice_to_ctm_kwargs
            )
        self.system.ctm_files[corpus]['recog_%s' % name] = lat2ctm.ctm_file

#   # lattice to ctm #
#   lat2ctm_extra_config = sprint.SprintConfig()
#   lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = 'best'
#   lat2ctm = recog.LatticeToCtmJob(csp            = csp,
#                                   lattice_cache  = search.lattice_path,
#                                   parallelize    = True,
#                                   best_path_algo = 'bellman-ford',
#                                   fill_empty_segments = fill_empty_cn,
#                                   extra_config   = lat2ctm_extra_config)

        if kaldi_scoring:
            scorer = recog.KaldiScorer( 
                corpus_path = self.system.csp[corpus].corpus_config.file,
                ctm = lat2ctm.ctm_file,
                map = {}
            )
        else:
            kwargs = copy.deepcopy(self.system.scorer_args[corpus])
            kwargs[self.system.scorer_hyp_arg[corpus]] = lat2ctm.ctm_file
            scorer = self.system.scorers[corpus](**kwargs)
            # kwargs = copy.deepcopy(self.scorer_args[corpus])
            # kwargs[self.scorer_hyp_arg[corpus]] = lat2ctm.ctm_file
            # scorer = self.scorers[corpus](**kwargs)

        self.system.jobs[corpus]['scorer_%s'  % name] = scorer
        tk.register_output('%srecog_%s.reports' % (prefix, name), scorer.report_dir)
    
    def decode(self, name, crnn_config, compile_crnn_config, recognition_args, epoch, scorer_args=None, decoding_args=None, **extra_args):
        kwargs = locals()
        del kwargs["self"], kwargs["extra_args"]
        kwargs["label_sync_args"] = kwargs.pop("decoding_args")
        compile_config = kwargs.pop("compile_crnn_config")
        if compile_config is not None:
            kwargs["crnn_config"] = compile_config
        with label_sync_decoding_binaries():
            self.label_sync_decode(**kwargs)

    def label_sync_decode(self, name, crnn_config, recognition_args, epoch, scorer_args=None, label_sync_args=None):
        scorer_args = scorer_args or {}
        label_sync_args = label_sync_args or {}
        label_sync_args = copy.deepcopy(label_sync_args)
        # make feature flow
        recog_args = ChainMap(self.system.default_recognition_args, recognition_args)
        feature_flow_net = meta.select_element(self.system.feature_flows, recog_args['corpus'], recog_args['flow'])
        model = self.system.jobs['train']['train_nn_%s' % name].models[epoch]
        feature_flow = self.system.get_tf_flow(
            feature_flow_net, crnn_config, model,
            rec_step_by_step='output',
            rec_json_info=True,
            alias=name, add_tf_flow=False
        )
        
        # label tree
        label_tree = sprint.LabelTree("hmm", use_transition_penalty=False)

        # label scorer
        label_scorer_args = {
            **self.default_label_scorer_args,
            **label_sync_args.pop("label_scorer_args", {})
        }
        scorerType = 'tf-ffnn-transducer'
        label_scorer = sprint.LabelScorer(scorerType, **label_scorer_args)

        model_graph = self.system.jobs["dev"][f"compile_crnn-{name}"].graph
        model_path = tk.uncached_path(model.model)[:-5]
        loader_config = sprint.SprintConfig()
        loader_config.type               = 'meta'
        loader_config.meta_graph_file    = model_graph
        loader_config.saved_model_file   = sprint.StringWrapper(model_path, model)
        # loader_config.required_libraries = "%s:%s:%s" %(default_nativeLSTM, default_lstm, default_kenlm)
        # if label_sync_args.pop("native_lstm", False):
        if any(layer.get("unit", None) == "lstmp" for layer in crnn_config["network"].values()):
            print("Found lstm layer in label_sync_decode")
            default_nativeLSTM = tk.Path('/u/zhou/libs/nativelstm2/tf1.12/NativeLstm2.so')
            # loader_config.required_libraries = gs.TF_NATIVE_OPS
            loader_config.required_libraries = default_nativeLSTM
        label_scorer.set_loader_config(loader_config)
        label_scorer.set_input_config()

        lm_scale       = label_sync_args.pop("lm_scale"      , self.default_lm_scale)
        pruning_factor = label_sync_args.pop("pruning_factor", self.default_pruning_factor)
        search_options          = {**self.default_search_options         , **label_sync_args.pop("search_options"   , {})}
        label_lookahead_options = {
            **self.default_label_lookahead_options,
            **{'scale': lm_scale},
            **label_sync_args.pop("lookahead_options", {})
        }
        search_options["label-pruning"] = lm_scale * pruning_factor

        # mixtures
        feature_scorer = sprint.PrecomputedHybridFeatureScorer(
            prior_mixtures=meta.select_element(self.system.mixtures, 'train', ('train', 'init_mixture')),
            priori_scale=scorer_args.get('prior_scale', 1.0),
            scale=scorer_args.get('mixture_scale', 1.0)
        )
        name = "-".join(["crnn", name, str(epoch)])
        self.recog_label_sync(
            name, "dev", feature_flow, label_scorer, lm_scale,
            label_tree=label_tree, lm_lookahead_options=label_lookahead_options,
            search_parameters=search_options, feature_scorer=feature_scorer,
            use_gpu=True, **label_sync_args
        )
