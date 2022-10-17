import copy
import itertools
import sys
from typing import Dict, Union, List, Tuple, Optional

from sisyphus import gs, tk
from sisyphus.delayed_ops import DelayedFormat

from i6_core import rasr
from i6_core import recognition as recog
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.compile import CompileTFGraphJob, CompileNativeOpJob
from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.extract_prior import ReturnnRasrComputePriorJob
from i6_core.returnn.training import Checkpoint
from i6_core.lexicon.allophones import DumpStateTyingJob

from i6_experiments.common.setups.rasr.nn_system import NnSystem
from i6_experiments.common.setups.rasr.util import RasrInitArgs, RasrDataInput
from i6_experiments.users.rossenbach.recognition.label_sync_search import (
    LabelSyncSearchJob,
)
from i6_experiments.users.rossenbach.mm.alignment import LabelAlignmentJob

from i6_experiments.users.rossenbach import rasr as rasr_experimental


class CtcRecognitionArgs:
    def __init__(
            self,
            eval_epochs,
            lm_scales,
            recog_args,
            search_parameters,
            compile_exec=None,
            blas_lib=None,
    ):
        """"
        :param eval_epochs: [7, 8, 9, 10] # iterations to evaluate corresponding to the "splits" iterations
        :param pronunciation_scales: [1.0] # scales the pronunciation props (which are simply 1.0 in most cases), only relevant then when using "normalize-pronunciations"
        :param lm_scales: [9.0, 9.25, 9.50, 9.75, 10.0, 10.25, 10.50] # obviously
        :param recog_args:
            {
                'feature_flow': dev_corpus_name,
                'pronunciation_scale': pronunciation_scale, # ???
                'lm_scale': lm_scale, # ???
                'lm_lookahead': True, # use lookahead, using the lm for pruning partial words
                'lookahead_options': None, # TODO:
                'create_lattice': True, # write lattice cache files
                'eval_single_best': True, # show the evaluation of the best path in lattice in the log (model score)
                'eval_best_in_lattice': True, # show the evaluation of the best path in lattice in the log (oracle)
                'best_path_algo': 'bellman-ford',  # options: bellman-ford, dijkstra
                'fill_empty_segments': False, # insert dummy when transcription output is empty
                'scorer': recog.Sclite,
                'scorer_args': {'ref': create_corpora.stm_files['dev-other']},
                'scorer_hyp_args': "hyp",
                'rtf': 30, # time estimation for jobs
                'mem': 8, # memory for jobs
                'use_gpu': False, # True makes no sense
            }
        :param search_parameters:
            {
                'beam_pruning': 14.0, # prob ratio of best path compared to pruned path
                'beam-pruning-limit': 100000, # maximum number of paths
                'word-end-pruning': 0.5, # pruning ratio at the end of completed words
                'word-end-pruning-limit': 15000 # maximum number of paths at completed words
            },
        ##################################################
        """
        self.eval_epochs = eval_epochs
        self.lm_scales = lm_scales
        self.recognition_args = recog_args
        self.search_parameters = search_parameters
        self.compile_exec = compile_exec
        self.blas_lib = blas_lib


class CtcSystem(NnSystem):
    """
    - 3 corpora types: train, dev and test
    - only train corpora will be aligned
    - dev corpora for tuning
    - test corpora for final eval

    to create beforehand:
    - corpora: name and i6_core.meta.system.Corpus
    - lexicon
    - lm

    settings needed:
    - am
    - lm
    - lexicon
    - feature extraction
    """

    def __init__(
        self, returnn_config, default_training_args, recognition_args,
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
        returnn_root: Optional[tk.Path] = None,
        returnn_python_home: Optional[tk.Path] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        blas_lib: Optional[tk.Path] = None,
    ):
        """

        :param ReturnnConfig returnn_config:
        :param dict default_training_args: are passed to `train_nn`
        :param CtcRecognitionArgs recognition_args:
        """
        super().__init__(
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
            blas_lib=blas_lib,
        )
        # self.crp["base"].python_home = rasr_python_home
        # self.crp["base"].python_program_name = rasr_python_exe

        self.returnn_config = returnn_config
        self.defalt_training_args = default_training_args
        self.recognition_args = recognition_args

        self.ctc_am_args = None

        self.state_tying = None

        self.default_align_keep_values = {
            "default": 5,
            "selected": gs.JOB_DEFAULT_KEEP_VALUE,
        }

        self.tf_checkpoints = {}  # type: Dict[str, Dict[int, Checkpoint]]

    # -------------------- Setup --------------------
    def init_system(
        self,
        rasr_init_args: RasrInitArgs,
        train_data: Dict[str, RasrDataInput],
        dev_data: Dict[str, RasrDataInput],
        test_data: Dict[str, RasrDataInput],
    ):
        self.rasr_init_args = rasr_init_args

        self._init_am(**self.rasr_init_args.am_args)

        self._assert_corpus_name_unique(train_data, dev_data, test_data)

        self.crp["base"].acoustic_model_config.allophones.add_all = True
        self.crp["base"].acoustic_model_config.allophones.add_from_lexicon = False

        self.crp["base"].acoustic_model_config.phonology.future_length = 0
        self.crp["base"].acoustic_model_config.phonology.history_length = 0

        for name, v in sorted(train_data.items()):
            add_lm = True if v.lm is not None else False
            self.add_corpus(name, data=v, add_lm=add_lm)
            self.train_corpora.append(name)

            break

        for name, v in sorted(dev_data.items()):
            self.add_corpus(name, data=v, add_lm=True)
            self.dev_corpora.append(name)

        for name, v in sorted(test_data.items()):
            self.add_corpus(name, data=v, add_lm=True)
            self.test_corpora.append(name)

    def create_full_sum_loss_config(
        self,
        num_classes,
        sprint_loss_config=None,
        sprint_loss_post_config=None,
        skip_segments=None,
        **kwargs,
    ):
        """

        :param num_classes:
        :param sprint_loss_config:
        :param sprint_loss_post_config:
        :param skip_segments:
        :param kwargs:
        :return:
        :rtype: (RasrConfig, RasrConfig)
        """
        crp = self.crp["loss"]
        mapping = {
            "acoustic_model": "*.model-combination.acoustic-model",
            "corpus": "*.corpus",
            "lexicon": "*.model-combination.lexicon",
        }

        config, post_config = rasr.build_config_from_mapping(
            crp, mapping, parallelize=(crp.concurrent == 1)
        )
        # concrete action in PythonControl called from RETURNN SprintErrorSignals.py derived from Loss/Layers
        config.neural_network_trainer.action = "python-control"
        config.neural_network_trainer.python_control_loop_type = "python-control-loop"
        config.neural_network_trainer.extract_features = False
        # allophone-state transducer
        config["*"].transducer_builder_filter_out_invalid_allophones = True
        config["*"].fix_allophone_context_at_word_boundaries = True
        # Automaton manipulation (RASR): default CTC topology
        config.neural_network_trainer.alignment_fsa_exporter.add_blank_transition = (
            kwargs.get("add_blank_transition", True)
        )
        config.neural_network_trainer.alignment_fsa_exporter.allow_label_loop = (
            kwargs.get("allow_label_loop", True)
        )
        # default blank replace silence
        if kwargs.get("blank_label_index", None) is not None:
            config.neural_network_trainer.alignment_fsa_exporter.blank_label_index = (
                kwargs.get("blank_label_index", None)
            )
            # maybe not needed
        config["*"].allow_for_silence_repetitions = False
        config["*"].number_of_classes = num_classes
        # config['*'].normalize_lemma_sequence_scores = True

        config._update(sprint_loss_config)
        post_config._update(sprint_loss_post_config)
        return config, post_config

    def create_rasr_loss_opts(cls, sprint_exe=None, custom_config=None, **kwargs):
        """

        :param sprint_exe:
        :param custom_config:
        :param kwargs:
        :return:
        """
        trainer_exe = rasr.RasrCommand.select_exe(sprint_exe, "nn-trainer")
        python_seg_order = False  # get automaton by segment name
        sprint_opts = {
            "sprintExecPath": trainer_exe,
            "minPythonControlVersion": 4,
            "numInstances": kwargs.get("num_sprint_instance", 2),
            "usePythonSegmentOrder": python_seg_order,
        }
        if custom_config:
            sprint_opts["sprintConfigStr"] = DelayedFormat("\"--config={} --*.LOGFILE=nn-trainer.loss.log --*.TASK=1\"", custom_config)
        else:
            sprint_opts["sprintConfigStr"] = "--config=rasr.loss.config --*.LOGFILE=nn-trainer.loss.log --*.TASK=1"
        return sprint_opts

    def make_loss_crp(
        self, ref_corpus_key, corpus_file=None, loss_am_config=None, **kwargs
    ):
        loss_crp = copy.deepcopy(self.crp[ref_corpus_key])
        if corpus_file is not None:
            crp_config = loss_crp.corpus_config
            crp_config.file = corpus_file
            loss_crp.corpus_config = crp_config
            all_segments = SegmentCorpusJob(corpus_file, 1)
            loss_crp.segment_path = all_segments.out_segment_path
        if loss_am_config is not None:
            loss_crp.acoustic_model_config = loss_am_config
        # if kwargs.get('sprint_loss_lm', None) is not None:
        #    lm_name = kwargs.pop('sprint_loss_lm', None)
        #    lm_scale = kwargs.pop('sprint_loss_lm_scale', 5.0)
        #    loss_crp.language_model_config = self.lm_setup.get_lm_config(name=lm_name, scale=lm_scale)
        if kwargs.get("sprint_loss_lexicon", None) is not None:
            # in case overwrite parent crp
            lexicon_config = copy.deepcopy(loss_crp.lexicon_config)
            lexicon_config.file = tk.Path(kwargs.get("sprint_loss_lexicon", None))
            loss_crp.lexicon_config = lexicon_config
        return loss_crp

    def train_nn(
        self,
        name,
        train_corpus_key,
        cv_corpus_key,
        feature_flow,
        returnn_config,
        num_classes,
        add_speaker_map=False,
        **kwargs,
    ):
        assert isinstance(
            returnn_config, ReturnnConfig
        ), "Passing returnn_config as dict to train_nn is no longer supported, please construct a ReturnnConfig object instead"


        self.crp["loss"] = rasr.CommonRasrParameters(base=self.crp[train_corpus_key])
        config, post_config = self.create_full_sum_loss_config(num_classes)
        custom_config = rasr.RasrConfig.WriteRasrConfigJob(config, post_config).out_config

        def add_rasr_loss(network, custom_config=None):
            network["rasr_loss"] = {
                "class": "copy",
                "from": "output",
                "loss_opts": {
                    'tdp_scale': 0.0,
                    "sprint_opts": self.create_rasr_loss_opts(custom_config=custom_config)
                },
                "loss": "fast_bw",
                "target": None,
            }

        if returnn_config.staged_network_dict:
            for net in returnn_config.staged_network_dict.values():
                add_rasr_loss(net, custom_config=custom_config)
        else:
            if returnn_config.config['network']['output'].get("loss", None) != "fast_bw":
                add_rasr_loss(returnn_config.config["network"], custom_config=custom_config)

        # if add_speaker_map:
        #     from i6_core.corpus.convert import CorpusToSpeakerMap
        #     speaker_map = CorpusToSpeakerMap(self.corpora[corpus_key].corpus_file).out_speaker_target_map
        #     if "dev" not in returnn_config.config:
        #         returnn_config.config["dev"] = {}
        #     if "train" not in returnn_config.config:
        #         returnn_config.config["train"] = {}
        #     returnn_config.config["train"]["target_maps"] = {'speaker_name': speaker_map}
        #     returnn_config.config["dev"]["target_maps"] = {'speaker_name': speaker_map}

        j = ReturnnTrainingJob(
            returnn_config=returnn_config,
            **kwargs,
        )

        j.add_alias("train_nn_%s_%s" % (train_corpus_key, name))
        self.jobs[train_corpus_key]["train_nn_%s" % name] = j
        self.tf_checkpoints[name] = j.out_checkpoints
        #self.nn_models[corpus_key][name] = j.out_models
        self.nn_configs[train_corpus_key][name] = j.out_returnn_config_file

        state_tying_job = DumpStateTyingJob(self.crp[train_corpus_key])
        tk.register_output(
            "{}_{}_state_tying".format(train_corpus_key, name),
            state_tying_job.out_state_tying,
        )
        self.state_tying = state_tying_job.out_state_tying


    # compile model graph from crnn config file
    def make_model_graph(
        self, returnn_config, labelSyncSearch=False, **kwargs
    ):
        """
        :param ReturnnConfig returnn_config:
        :param labelSyncSearch:
        :param kwargs:
        :return:
        """
        args = {
            "returnn_config": returnn_config,
            'returnn_python_exe': self.defalt_training_args['returnn_python_exe'],
            'returnn_root': self.defalt_training_args['returnn_root'],
        }
        if labelSyncSearch:
            args.update(
                {
                    "rec_step_by_step": kwargs.get("recName", "output"),
                    "rec_json_info": kwargs.get("recJsonInfo", True),
                }
            )
        compile_graph_job = CompileTFGraphJob(**args)
        tf_graph = compile_graph_job.out_graph

        return tf_graph

    def make_tf_feature_flow(
        self, feature_flow, tf_graph, tf_checkpoint
    ):
        """
        :param feature_flow:
        :param Path tf_graph
        :param Checkpoint tf_checkpoint:
        :return:
        """
        # TODO: HACK
        from i6_core.returnn.compile import CompileNativeOpJob
        from i6_experiments.users.rossenbach.returnn.flow import make_tf_feature_flow

        # DO NOT USE BLAS ON I6, THIS WILL SLOW DOWN RECOGNITION ON OPTERON MACHNIES BY FACTOR 4
        native_op = CompileNativeOpJob(
            "NativeLstm2",
            returnn_python_exe=self.recognition_args.compile_exec or self.defalt_training_args['returnn_python_exe'],
            returnn_root=self.defalt_training_args['returnn_root'],
            blas_lib=self.recognition_args.blas_lib,
            search_numpy_blas=False).out_op

        return make_tf_feature_flow(
            feature_flow=feature_flow,
            tf_graph=tf_graph,
            tf_checkpoint=tf_checkpoint,
            native_ops=[native_op]
        )


    # if gpu full, use cpu for more queue slots so that recognition exps can start earlier
    # def estimate_nn_prior(self, corpus_key, nn_name, feature_flow, epoch, **kwargs):
    def estimate_nn_prior(self, corpus_key, feature_flow, tf_checkpoint, **kwargs):
        #assert epoch in self.nn_models[corpus_key][nn_name].keys(), "epoch %d not saved in %s" %(epoch, nn_name)

        if kwargs.get('use_exist_prior', False):
            assert False
            #assert self.nn_priors[corpus_key][nn_name][epoch], 'No existing prior found'
            #return self.nn_priors[corpus_key][nn_name][epoch]
        else:
            args = {
                'train_crp': self.crp[corpus_key + "_train"],
                'dev_crp': self.crp[corpus_key + "_cv"],
                'feature_flow': self.feature_flows[corpus_key][feature_flow],
                'model_checkpoint': tf_checkpoint,
                'time_rqmt': 8,
                'mem_rqmt': 8,
                'cpu_rqmt': 2,
                'device': kwargs.get('nn_prior_device','gpu'),
                'returnn_config': self.get_specific_returnn_config(self.returnn_config),
                'returnn_python_exe': self.defalt_training_args['returnn_python_exe'],
                'returnn_root': self.defalt_training_args['returnn_root'],
            }
            prior_job = ReturnnRasrComputePriorJob(**args)
            return prior_job.out_prior_xml_file

    def recog(
            self,
            name,
            corpus_key,
            flow,
            tf_checkpoint,
            pronunciation_scale,
            lm_scale,
            lm_lookahead,
            lookahead_options=None,
            parallelize_conversion=False,
            lattice_to_ctm_kwargs=None,
            prefix="",
            **kwargs,
    ):
        """
        :param str name:
        :param str corpus_key:
        :param str|list[str]|tuple[str]|rasr.FlagDependentFlowAttribute flow:
        :param Checkpoint tf_checkpoint:
        :param float pronunciation_scale:
        :param float lm_scale:
        :param bool lm_lookahead:
        :param dict|None lookahead_options:
        :param bool parallelize_conversion:
        :param dict|None lattice_to_ctm_kwargs:
        :param str prefix:
        :param kwargs:
        :return:
        """
        if lattice_to_ctm_kwargs is None:
            lattice_to_ctm_kwargs = {}

        self.crp[corpus_key].language_model_config.scale = lm_scale
        self.crp[corpus_key].acoustic_model_config.tdp["*"].skip = 0
        self.crp[corpus_key].acoustic_model_config.tdp.silence.skip = 0

        model_combination_config = rasr.RasrConfig()
        model_combination_config.pronunciation_scale = pronunciation_scale

        # label tree #
        label_unit = kwargs.pop('label_unit', None)
        assert label_unit, 'label_unit not given'
        label_tree_args = kwargs.pop('label_tree_args',{})
        label_tree = rasr_experimental.LabelTree(label_unit, **label_tree_args)

        scorer_type = kwargs.pop('label_scorer_type', None)
        assert scorer_type, 'label_scorer_type not given'
        label_scorer_args = kwargs.pop('label_scorer_args',{})
        # add vocab file
        from i6_experiments.users.rossenbach.rasr.vocabulary import GenerateLabelFileFromStateTying
        label_scorer_args['labelFile'] = GenerateLabelFileFromStateTying(self.state_tying, add_eow=True).out_label_file
        label_scorer_args['priorFile'] = self.estimate_nn_prior(self.train_corpora[0], feature_flow=flow, tf_checkpoint=tf_checkpoint, **kwargs)
        am_scale = label_scorer_args.get('scale', 1.0)

        tf_graph = self.make_model_graph(
            self.returnn_config
        )

        feature_flow = self.make_tf_feature_flow(self.feature_flows[corpus_key][flow], tf_graph, tf_checkpoint, **kwargs)

        label_scorer = rasr_experimental.LabelScorer(scorer_type, **label_scorer_args)

        extra_config = rasr.RasrConfig()
        if pronunciation_scale > 0:
            extra_config.flf_lattice_tool.network.recognizer.pronunciation_scale = pronunciation_scale

        if lm_lookahead:
            assert lookahead_options is not None
            # we want to alter this now
            lookahead_options = copy.deepcopy(lookahead_options)
            if lookahead_options.get("scale", None) is None:
                lookahead_options["scale"] = lm_scale

        # Fixed CTC settings:
        extra_config.flf_lattice_tool.network.recognizer.recognizer.allow_label_loop = True
        extra_config.flf_lattice_tool.network.recognizer.recognizer.allow_blank_label = True

        extra_config.flf_lattice_tool.network.recognizer.recognizer.allow_label_recombination = True
        extra_config.flf_lattice_tool.network.recognizer.recognizer.allow_word_end_recombination = True

        rec = LabelSyncSearchJob(
            crp=self.crp[corpus_key],
            feature_flow=feature_flow,
            label_scorer=label_scorer,
            label_tree=label_tree,
            lm_lookahead=lm_lookahead,
            lookahead_options=lookahead_options,
            extra_config=extra_config,
            **kwargs,
        )
        rec.set_vis_name("Recog %s%s" % (prefix, name))
        rec.add_alias("%srecog_%s" % (prefix, name))
        self.jobs[corpus_key]["recog_%s" % name] = rec

        self.jobs[corpus_key]["lat2ctm_%s" % name] = lat2ctm = recog.LatticeToCtmJob(
            crp=self.crp[corpus_key],
            lattice_cache=rec.out_lattice_bundle,
            parallelize=parallelize_conversion,
            **lattice_to_ctm_kwargs,
        )
        self.ctm_files[corpus_key]["recog_%s" % name] = lat2ctm.out_ctm_file

        kwargs = copy.deepcopy(self.scorer_args[corpus_key])
        kwargs[self.scorer_hyp_arg[corpus_key]] = lat2ctm.out_ctm_file
        scorer = self.scorers[corpus_key](**kwargs)

        self.jobs[corpus_key]["scorer_%s" % name] = scorer
        tk.register_output("%srecog_%s.reports" % (prefix, name), scorer.out_report_dir)

    def nn_align(
            self,
            name,
            corpus_key,
            flow,
            tf_checkpoint,
            pronunciation_scale,
            alignment_options=None,
            parallelize_conversion=False,
            prefix="",
            **kwargs,
    ):
        """
        :param str name:
        :param str corpus_key:
        :param str|list[str]|tuple[str]|rasr.FlagDependentFlowAttribute flow:
        :param Checkpoint tf_checkpoint:
        :param float pronunciation_scale:
        :param float lm_scale:
        :param bool lm_lookahead:
        :param dict|None lookahead_options:
        :param bool parallelize_conversion:
        :param dict|None lattice_to_ctm_kwargs:
        :param str prefix:
        :param kwargs:
        :return:
        """

        # self.crp[corpus_key].language_model_config.scale = lm_scale
        #self.crp[corpus_key].acoustic_model_config.tdp["*"].skip = 0
        #self.crp[corpus_key].acoustic_model_config.tdp.silence.skip = 0

        model_combination_config = rasr.RasrConfig()
        model_combination_config.pronunciation_scale = pronunciation_scale

        # label tree #
        label_unit = kwargs.pop('label_unit', None)
        assert label_unit, 'label_unit not given'
        label_tree_args = kwargs.pop('label_tree_args',{})
        # label_tree = rasr_experimental.LabelTree(label_unit, **label_tree_args)

        scorer_type = kwargs.pop('label_scorer_type', None)
        assert scorer_type, 'label_scorer_type not given'
        label_scorer_args = kwargs.pop('label_scorer_args',{})
        # add vocab file
        from i6_experiments.users.rossenbach.rasr.vocabulary import GenerateLabelFileFromStateTying
        label_scorer_args['labelFile'] = GenerateLabelFileFromStateTying(self.state_tying, add_eow=True).out_label_file
        label_scorer_args['priorFile'] = self.estimate_nn_prior(self.train_corpora[0], feature_flow=flow, tf_checkpoint=tf_checkpoint, **kwargs)
        am_scale = label_scorer_args.get('scale', 1.0)

        tf_graph = self.make_model_graph(
            self.returnn_config
        )

        feature_flow = self.make_tf_feature_flow(self.feature_flows[corpus_key][flow], tf_graph, tf_checkpoint, **kwargs)

        label_scorer = rasr_experimental.LabelScorer(scorer_type, **label_scorer_args)

        extra_config = rasr.RasrConfig()
        if pronunciation_scale > 0:
            extra_config.flf_lattice_tool.network.recognizer.pronunciation_scale = pronunciation_scale

        # Fixed CTC settings:
        extra_config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.alignment.allow_label_loop = True

        if alignment_options is None:
            alignment_options = {
                'label-pruning'      : 10,
                'label-pruning-limit': 10000,
            }

        # label alignment
        align_args = {
            'crp'           : self.crp[corpus_key],
            'use_gpu'       : kwargs.get('use_gpu', True),
            'feature_flow'  : feature_flow,
            'label_scorer'  : label_scorer,
            'alignment_options' : alignment_options, # aligner search option,
            'extra_config': extra_config,
        }
        align_job = LabelAlignmentJob(**align_args)
        #align_job.rqmt.update(job_rqmt)
        alignment = rasr.FlagDependentFlowAttribute( 'cache_mode',
                                                       { 'task_dependent' : align_job.out_alignment_path,
                                                         'bundle'         : align_job.out_alignment_bundle
                                                         }
                                                       )
        self.alignments[corpus_key][name] = [alignment]
        if kwargs.get('register_output', False):
            tk.register_output('%s_%s' %(corpus_key, name), align_job.out_alignment_bundle)
        return name



    def recognition(
            self,
            name: str,
            training_name: str,
            iters: List[int],
            lm_scales: Union[float, List[float]],
            optimize_am_lm_scale: bool,
            # parameters just for passing through
            corpus_key: str,
            feature_flow: Union[
                str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute
            ],
            pronunciation_scales: Union[float, List[float]],
            search_parameters: dict,
            rtf: float,
            mem: float,
            parallelize_conversion: bool,
            lattice_to_ctm_kwargs: dict,
            **kwargs,
    ):
        """
        A small wrapper around the meta.System.recog function that will set a Sisyphus block and
        run over all specified model iterations and lm scales.

        :param name: name for the recognition, note that iteration and lm will be named by the function
        :param training_name: name of the already defined training
        :param iters: which training iterations to use for recognition
        :param lm_scales: all lm scales that should be used for recognition
        :param feature_scorer_key: (training_corpus_name, training_name)
        :param optimize_am_lm_scale: will optimize the lm-scale and re-run recognition with the optimal value
        :param kwargs: see meta.System.recog and meta.System.recog_and_optimize
        :param corpus_key: corpus to run recognition on
        :param feature_flow:
        :param pronunciation_scales:
        :param search_parameters:
        :param rtf:
        :param mem:
        :param parallelize_conversion:
        :param lattice_to_ctm_kwargs:
        :return:
        """
        assert (
                "lm_scale" not in kwargs
        ), "please use lm_scales for GmmSystem.recognition()"
        with tk.block(f"{name}_recognition"):
            recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

            pronunciation_scales = (
                [pronunciation_scales]
                if isinstance(pronunciation_scales, float)
                else pronunciation_scales
            )

            lm_scales = [lm_scales] if isinstance(lm_scales, float) else lm_scales

            for it, p, l in itertools.product(iters, pronunciation_scales, lm_scales):
                self.recog(
                    name=f"{name}-{corpus_key}-ps{p:02.2f}-lm{l:02.2f}-iter{it:02d}",
                    prefix=f"recognition/{name}/",
                    corpus_key=corpus_key,
                    flow=feature_flow,
                    tf_checkpoint=self.tf_checkpoints[training_name][it],
                    pronunciation_scale=p,
                    lm_scale=l,
                    search_parameters=search_parameters,
                    rtf=rtf,
                    mem=mem,
                    parallelize_conversion=parallelize_conversion,
                    lattice_to_ctm_kwargs=lattice_to_ctm_kwargs,
                    **kwargs,
                )


    def run(self, steps: Union[List, Tuple] = ("all",)):
        """
        run setup

        :param steps:
        :return:
        """
        assert len(steps) > 0
        if len(steps) == 1 and steps[0] == "all":
            steps = ["extract", "train"]

        if "init" in steps:
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        for all_c in self.train_corpora + self.dev_corpora + self.test_corpora:
            self.costa(all_c, prefix="costa/", **self.rasr_init_args.costa_args)

        for trn_c in self.train_corpora:
            self.store_allophones(trn_c)

        for eval_c in self.dev_corpora + self.test_corpora:
            self.create_stm_from_corpus(eval_c)
            self.set_sclite_scorer(eval_c)

        if "extract" in steps:
            self.extract_features(feat_args=self.rasr_init_args.feature_extraction_args)

        if "train" in steps:
            num_classes = 139  # fixed for now

            self.train_nn(
                "default",
                corpus_key=self.train_corpora[0],
                feature_flow="gt",
                returnn_config=self.returnn_config,
                num_classes=num_classes,
                alignment=None,
                **self.defalt_training_args,
            )
            #out_models = self.nn_models["train-clean-100"]["default"]
            #tk.register_output("tests/ctc_training", out_models[180].model)

        if "recog" in steps:
            trn_c = self.train_corpora[0]
            name = "default"
            for dev_c in self.dev_corpora:
                self.recognition(
                    name=f"{trn_c}-{dev_c}",
                    training_name="default",
                    iters=self.recognition_args.eval_epochs,
                    lm_scales=self.recognition_args.lm_scales,
                    corpus_key=dev_c,
                    search_parameters=self.recognition_args.search_parameters,
                    optimize_am_lm_scale=False,
                    pronunciation_scales=[1.0],
                    parallelize_conversion=False,
                    lattice_to_ctm_kwargs={},
                    **self.recognition_args.recognition_args,
                )

