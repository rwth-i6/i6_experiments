import copy
import sys
from typing import Dict, Union, List, Tuple

from sisyphus import gs, tk

from i6_core import rasr
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.rasr_training import ReturnnRasrTrainingJob

from i6_experiments.common.setups.hybrid.rasr_system import RasrSystem
from i6_experiments.common.setups.hybrid.util import RasrInitArgs, RasrDataInput


class CtcSystem(RasrSystem):
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

    def __init__(self,
                 returnn_config,
                 default_training_args,
                 rasr_python_home,
                 rasr_python_exe):
        """

        :param ReturnnConfig returnn_config:
        :param dict default_training_args:
        """
        super().__init__()
        self.crp["base"].python_home = rasr_python_home
        self.crp["base"].python_program_name = rasr_python_exe

        self.returnn_config = returnn_config
        self.defalt_training_args = default_training_args

        self.ctc_am_args = None

        self.default_align_keep_values = {
            "default": 5,
            "selected": gs.JOB_DEFAULT_KEEP_VALUE,
        }

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

        # make traindev
        # from i6_core.corpus.transform import MergeCorporaJob
        # bliss_corpora = []
        # lexica = []
        # for name, v in sorted(train_data.items()):
        #     bliss_corpora.append(v.corpus_object.corpus_file)
        #     lexica.append(v.corpus_object)
        # for name, v in sorted(dev_data.items()):
        #     bliss_corpora.append(v.corpus_object.corpus_file)

        # merged_bliss = MergeCorporaJob(bliss_corpora, "merged_train_dev")
        # RasrDataInput()
        # self.add_corpus("merged_train_dev", )

        # train_segments = {}
        # for name, v in sorted(train_data.items()):

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

    def create_full_sum_loss_config(self, num_classes,
                                  sprint_loss_config=None, sprint_loss_post_config=None, skip_segments=None,
                                   **kwargs):
        crp = self.crp['loss']
        mapping = { 'acoustic_model' : '*.model-combination.acoustic-model',
                    'corpus'         : '*.corpus',
                    'lexicon'        : '*.model-combination.lexicon'
                    }

        config, post_config = rasr.build_config_from_mapping(crp, mapping, parallelize=(crp.concurrent == 1))
        # concrete action in PythonControl called from RETURNN SprintErrorSignals.py derived from Loss/Layers
        config.neural_network_trainer.action                   = 'python-control'
        config.neural_network_trainer.python_control_loop_type = 'python-control-loop'
        config.neural_network_trainer.extract_features         = False
        # allophone-state transducer
        config['*'].transducer_builder_filter_out_invalid_allophones = True
        config['*'].fix_allophone_context_at_word_boundaries         = True
        # Automaton manipulation (RASR): default CTC topology
        config.neural_network_trainer.alignment_fsa_exporter.add_blank_transition = kwargs.get('add_blank_transition', True)
        config.neural_network_trainer.alignment_fsa_exporter.allow_label_loop     = kwargs.get('allow_label_loop', True)
        # default blank replace silence
        if kwargs.get('blank_label_index', None) is not None:
            config.neural_network_trainer.alignment_fsa_exporter.blank_label_index = kwargs.get('blank_label_index', None)
            # maybe not needed
        config['*'].allow_for_silence_repetitions = False
        config['*'].number_of_classes             = num_classes
        #config['*'].normalize_lemma_sequence_scores = True
        if skip_segments is not None:
            config.neural_network_trainer['*'].segments_to_skip = skip_segments


        config._update(sprint_loss_config)
        post_config._update(sprint_loss_post_config)
        return config, post_config

    def create_rasr_loss_opts(cls, sprint_exe=None, **kwargs):
        trainer_exe = rasr.RasrCommand.select_exe(sprint_exe, 'nn-trainer')
        python_seg_order = False # get automaton by segment name
        sprint_opts = { 'sprintExecPath'  : trainer_exe,
                        'sprintConfigStr' : '--config=rasr.loss.config --*.LOGFILE=nn-trainer.loss.log --*.TASK=1',
                        'minPythonControlVersion' : 4,
                        'numInstances'            : kwargs.get('num_sprint_instance', 2),
                        'usePythonSegmentOrder'   : python_seg_order
                        }
        return sprint_opts


    def make_loss_crp(self, ref_corpus_key, corpus_file=None, loss_am_config=None, **kwargs):
        loss_crp = copy.deepcopy(self.crp[ref_corpus_key])
        if corpus_file is not None:
            crp_config = loss_crp.corpus_config
            crp_config.file = corpus_file
            loss_crp.corpus_config = crp_config
            all_segments = SegmentCorpusJob(corpus_file, 1)
            loss_crp.segment_path = all_segments.out_segment_path
        if loss_am_config is not None:
            loss_crp.acoustic_model_config = loss_am_config
        #if kwargs.get('sprint_loss_lm', None) is not None:
        #    lm_name = kwargs.pop('sprint_loss_lm', None)
        #    lm_scale = kwargs.pop('sprint_loss_lm_scale', 5.0)
        #    loss_crp.language_model_config = self.lm_setup.get_lm_config(name=lm_name, scale=lm_scale)
        if kwargs.get('sprint_loss_lexicon', None) is not None:
            # in case overwrite parent crp
            lexicon_config = copy.deepcopy(loss_crp.lexicon_config)
            lexicon_config.file = tk.Path(kwargs.get('sprint_loss_lexicon', None))
            loss_crp.lexicon_config = lexicon_config
        return loss_crp

    def train_nn(
            self,
            name,
            corpus_key,
            feature_flow,
            returnn_config,
            num_classes,
            **kwargs,
    ):
        assert isinstance(
            returnn_config, ReturnnConfig
        ), "Passing returnn_config as dict to train_nn is no longer supported, please construct a ReturnnConfig object instead"

        corpus_key = self.train_corpora[0]
        train_corpus_key = corpus_key + "_train"
        cv_corpus_key = corpus_key + "_cv"
        cv_size = 0.005
        all_segments = SegmentCorpusJob(self.corpora[corpus_key].corpus_file, 1).out_single_segment_files[1]
        new_segments = ShuffleAndSplitSegmentsJob(
            segment_file=all_segments,
            split={'train': 1.0 - cv_size, 'cv': cv_size})
        train_segments = new_segments.out_segments['train']
        cv_segments = new_segments.out_segments['cv']

        self.add_overlay(corpus_key, train_corpus_key)
        self.crp[train_corpus_key].segment_path = train_segments
        self.crp[train_corpus_key].corpus_config.segment_order_shuffle = True
        self.crp[train_corpus_key].corpus_config.segment_order_sort_by_time_length = True
        self.crp[train_corpus_key].corpus_config.segment_order_sort_by_time_length_chunk_size = 384
        self.add_overlay(corpus_key, cv_corpus_key)
        self.crp[cv_corpus_key].segment_path = cv_segments

        self.crp['loss'] = rasr.CommonRasrParameters(base=self.crp[corpus_key])
        config, post_config = self.create_full_sum_loss_config(num_classes)

        def add_rasr_loss(network):

            network['rasr_loss'] = {
                'class': 'copy', 'from': 'output',
                'loss_opts': {
                    #'tdp_scale': 0.0,
                    'sprint_opts': self.create_rasr_loss_opts()
                },
                'loss': 'fast_bw',
                'target': None,
            }

        if returnn_config.staged_network_dict:
            for net in returnn_config.staged_network_dict.values():
                add_rasr_loss(net)
        else:
            add_rasr_loss(returnn_config.config['network'])

        j = ReturnnRasrTrainingJob(
            train_crp=self.crp[train_corpus_key],
            dev_crp=self.crp[cv_corpus_key],
            feature_flow=self.feature_flows[corpus_key][feature_flow],
            returnn_config=returnn_config,
            num_classes=self.functor_value(num_classes),
            additional_rasr_config_files={'rasr.loss': config},
            additional_rasr_post_config_files={'rasr.loss': post_config},
            **kwargs,
        )

        j.add_alias("train_nn_%s_%s" % (corpus_key, name))
        self.jobs[corpus_key]["train_nn_%s" % name] = j
        self.nn_models[corpus_key][name] = j.out_models
        self.nn_configs[corpus_key][name] = j.out_returnn_config_file

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
            self.extract_features(
                feat_args=self.rasr_init_args.feature_extraction_args
            )

        if "train" in steps:
            num_classes = 139 # fixed for now

            self.train_nn(
                "default",
                corpus_key=self.train_corpora[0],
                feature_flow="gt",
                returnn_config=self.returnn_config,
                num_classes=num_classes,
                alignment=None,
                **self.defalt_training_args
            )
            out_models = self.nn_models["train-clean-100"]["default"]
            tk.register_output("tests/ctc_training", out_models[180].model)