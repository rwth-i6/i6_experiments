__all__ = ["FactoredHybridSystem"]

import copy
import itertools
import sys

from dataclasses import asdict
from IPython import embed
from typing import Dict, List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

from sisyphus.delayed_ops import DelayedFormat

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.util import MultiPath, MultiOutputPath

from i6_experiments.common.setups.rasr.nn_system import NnSystem

from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
    ReturnnRasrDataInput,
    RasrSteps,
)

from i6_experiments.users.raissi.setups.common.helpers.pipeline_data import (
    ContextMapper,
    LabelInfo,
    PipelineStages
)
from i6_experiments.users.raissi.setups.common.helpers.train_helpers import (
    get_extra_config_segment_order,
)

from i6_experiments.users.raissi.setups.common.helpers.rasr import (
    SystemOutput,
)

from i6_experiments.users.raissi.setups.librispeech.util.pipeline_helpers import (
    get_label_info,
    get_alignment_keys,
    get_lexicon_args,
    get_tdp_values,
)




# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class FactoredHybridSystem(NnSystem):
    """
    #ToDo
    """
    def __init__(
            self,
            returnn_root: Optional[str] = None,
            returnn_python_home: Optional[str] = None,
            returnn_python_exe: Optional[str] = None, #tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
            hybrid_init_args: RasrInitArgs = None,
            train_data: Dict[str, RasrDataInput] = None,
            dev_data: Dict[str, RasrDataInput] = None,
            test_data: Dict[str, RasrDataInput] = None,
    ):
        super().__init__(
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
        )
        #arguments used for the initialization of the system, they should be set before running the system
        self.hybrid_init_args = hybrid_init_args
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        #useful paths
        self.dependency_path = '/work/asr4/raissi/setups/librispeech/960-ls/dependencies'

        #general modeling approach
        self.label_info = LabelInfo(**get_label_info())
        self.lexicon_args = get_lexicon_args()
        self.tdp_values = get_tdp_values()

        #transcription priors in pickle format
        self.priors = {'monostate':        ('/').join([self.dependency_path , 'priors', 'daniel', 'monostate', 'monostate.pickle']),
                       'monostate-EOW':    ('/').join([self.dependency_path , 'priors', 'daniel', 'monostate', 'monostate.we.pickle']),
                       'threepartite':     ('/').join([self.dependency_path , 'priors', 'daniel', 'threepartite', 'threepartite.pickle']),
                       'threepartite-EOW': ('/').join([self.dependency_path , 'priors', 'daniel', 'threepartite', 'threepartite.we.pickle'])}

        self.cv_info = {'pre_path': ('/').join([self.dependency_path, 'data', 'zhou-corpora']),
                                      'train_dev': 'train-dev.corpus.xml',
                                      'train_dev_segments': 'train-dev.segments',
                                      'dev_cv': 'dev-cv.corpus.xml',
                                      'dev_cv_segments': 'dev-cv.segments',
                                      'features_path': ('/').join(['FeatureExtraction.Gammatone.yly3ZlDOfaUm', 'output', 'gt.cache.bundle'])
                                      }

        #dataset infomration
        self.cv_corpora = []
        self.devtrain_corpora = []

        self.train_input_data = None
        self.cv_input_data = None
        self.devtrain_input_data = None
        self.dev_input_data = None
        self.test_input_data = None


        self.datasets = {}
        self.hdfs = {}
        self.basic_feature_flows = {}

        # train information
        self.initial_nn_args =   {'num_input': 50,
                                  'num_classes': self.label_info.n_state_classes,
                                  'partition_epochs': {'train': 20, 'dev': 1},
                                  'n_epochs': 25,
                                  #'log_verbosity': 5,
                                  'returnn_python_exe': returnn_python_exe}

        self.initial_train_args = {'time_rqmt': 168,
                                   'mem_rqmt': 32,
                                   'num_classes': self.label_info.n_state_classes,
                                   'num_epochs': self.initial_nn_args['n_epochs'] * self.initial_nn_args['partition_epochs']['train'],
                                   'partition_epochs': self.initial_nn_args['partition_epochs']}

        #pipeline info
        self.context_mapper = ContextMapper()
        self.stage = PipelineStages(get_alignment_keys())
        
        self.trainers    = None #ToDo external trainer class
        self.recognizers = {}
        self.aligners = {}
        self.returnn_configs = {}
        self.graphs = {}

        self.experiments = {}

        #inference related
        self.tf_map = {"triphone": "right", "diphone": "center", "context": "left"}        #Triphone forward model
        self.mixtures["dummy"] = mm.CreateDummyMixturesJob(self.label_info.get_number_of_dense_classes(),
                                                           self.initial_nn_args["num_input"]).out_mixtures #gammatones

        self.outputs = {}

    #----------- pipeline construction -----------------
    def set_experiment_dict(self, key, alignment, context):
        name = self.stage.get_name(alignment, context)
        self.experiments[key] = {
            "name": name,
            "train_job": None,
            "graph": {"train": None, "inference": None},
            "returnn_config": None,
            "align_job": None,
            "decode_job": {"runner": None, "args": None}
        }

    #--------------------- Init procedure -----------------
    def init_system(self):
        for param in [self.hybrid_init_args, self.train_data, self.dev_data, self.test_data]:
            assert param is not None

        self._init_am(**self.hybrid_init_args.am_args)
        self._assert_corpus_name_unique(self.train_data, self.dev_data, self.test_data)

        for corpus_key, rasr_data_input in sorted(self.train_data.items()):
            add_lm = True if rasr_data_input.lm is not None else False
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=add_lm)
            self.train_corpora.append(corpus_key)

        for corpus_key, rasr_data_input in sorted(self.dev_data.items()):
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=True)
            self.dev_corpora.append(corpus_key)

        for corpus_key, rasr_data_input in sorted(self.test_data.items()):
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=True)
            self.test_corpora.append(corpus_key)

    def _update_crp_am_setting(self, crp_key, tdp_type=None, label_info=None):
        #ToDo handle different tdp values: default, based on transcription, based on an alignment
        if label_info is None:
            label_info = copy.deepcopy(self.label_info)

        tdp_pattern = self.tdp_values['pattern']
        if tdp_type is None:
            tdp_values = self.tdp_values['default']

        elif isinstance(tdp_type, tuple):
            tdp_values = self.tdp_values[tdp_type[0]][tdp_type[1]]

        else:
            print("Not implemented")
            import sys
            sys.exit()

        crp = self.crp[crp_key]
        for ind, ele in enumerate(tdp_pattern):
            for type in ["*", "silence"]:
                crp.acoustic_model_config["tdp"][type][ele] = tdp_values[type][ind]

        crp.acoustic_model_config.state_tying.type = label_info.state_tying
        if label_info.use_word_end_class:
            crp.acoustic_model_config.state_tying.use_word_end_classes = label_info.use_word_end_class
        crp.acoustic_model_config.state_tying.use_boundary_classes = label_info.use_boundary_classes
        crp.acoustic_model_config.hmm.states_per_phone = label_info.n_states_per_phone

        crp.acoustic_model_config.allophones.add_all = self.lexicon_args['add_all_allophones']
        crp.acoustic_model_config.allophones.add_from_lexicon = not self.lexicon_args['add_all_allophones']
        crp.lexicon_config.normalize_pronunciation = self.lexicon_args['norm_pronunciation']


    def init_datasets(
        self,
        train_data: Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]],
        cv_data: Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]],
        devtrain_data: Optional[
            Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]]
        ] = None,
        dev_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        test_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        train_cv_pairing: Optional[
            List[Tuple[str, ...]]
        ] = None,  # List[Tuple[trn_c, cv_c, name, dvtr_c]]
    ):
        devtrain_data = devtrain_data if devtrain_data is not None else {}
        dev_data = dev_data if dev_data is not None else {}
        test_data = test_data if test_data is not None else {}

        self._assert_corpus_name_unique(
            train_data, cv_data, devtrain_data, dev_data, test_data
        )
        datasets = [train_data, cv_data, devtrain_data, dev_data, test_data]

        self.train_input_data = train_data
        self.cv_input_data = cv_data
        self.devtrain_input_data = devtrain_data
        self.dev_input_data = dev_data
        self.test_input_data = test_data

        self.train_corpora.extend(list(train_data.keys()))
        self.cv_corpora.extend(list(cv_data.keys()))
        self.devtrain_corpora.extend(list(devtrain_data.keys()))
        self.dev_corpora.extend(list(dev_data.keys()))
        self.test_corpora.extend(list(test_data.keys()))

        self._set_train_data(train_data)
        self._set_train_data(cv_data)
        self._set_train_data(devtrain_data)
        self._set_eval_data(dev_data)
        self._set_eval_data(test_data)

    def get_system_output(
        self,
        corpus_key: str,
        corpus_type: str,
        step_idx: int,
        steps: RasrSteps,
        extract_features: List[str],
    ):
        """
        :param corpus_key: corpus name identifier
        :param corpus_type: corpus used for: train, dev or test
        :param step_idx: select a specific step from the defined list of steps
        :param steps: all steps in pipeline
        :param extract_features: list of features to extract for later usage
        :return GmmOutput:
        """
        sys_out = SystemOutput()
        sys_out.crp = self.crp[corpus_key]
        sys_out.feature_flows = self.feature_flows[corpus_key]
        sys_out.features = self.feature_caches[corpus_key]

        for feat_name in extract_features:
            tk.register_output(
                f"features/{corpus_key}_{feat_name}_features.bundle",
                self.feature_bundles[corpus_key][feat_name],
            )


        return sys_out

    def run_output_step(self, step_args, step_idx, steps):
        for corpus_key, corpus_type in step_args.corpus_type_mapping.items():
            if (
                corpus_key
                not in self.train_corpora + self.dev_corpora + self.test_corpora
            ):
                continue
            if corpus_key not in self.outputs.keys():
                self.outputs[corpus_key] = {}
            self.outputs[corpus_key][step_args.name] = self.get_system_output(
                corpus_key,
                corpus_type,
                step_idx,
                steps,
                step_args.extract_features,
            )

    def _set_train_data(self, data_dict):
        for c_key, c_data in data_dict.items():
            self.crp[c_key] = c_data.get_crp() if c_data.crp is None else c_data.crp
            self.feature_flows[c_key] = c_data.feature_flow

    def _set_eval_data(self, data_dict):
        for c_key, c_data in data_dict.items():
            self.ctm_files[c_key] = {}
            self.crp[c_key] = c_data.get_crp() if c_data.crp is None else c_data.crp
            self.feature_flows[c_key] = c_data.feature_flow

    def prepare_data_for_bw_with_separate_cv(self, dev_key="dev_magic", train_key="train_magic",
                                             cv_key="cv_magic", bw_key="bw"):
        cv_corpus = ("/").join(
            [self.cross_validation_info['pre_path'], self.cross_validation_info['dev_cv']])
        cv_segment = ("/").join(
            [self.cross_validation_info['pre_path'], self.cross_validation_info['dev_cv.segments']])
        self.set_sprint_corpora(dev_key,
                                cv_key,
                                cv_segment,
                                corpus_file=cv_corpus)

        cv_feature_path = Path(("/").join(
            [self.cross_validation_info['pre_path'], self.cross_validation_info['features_path'], "output",
             "gt.cache.bundle"]))
        self.feature_flows[cv_key] = features.basic_cache_flow(cv_feature_path)

        self.csp[bw_key] = copy.deepcopy(self.csp[train_key])
        self.csp[bw_key].corpus_config.file = ("/").join(
            [self.cross_validation_info['pre_path'], self.cross_validation_info['merged_corpus_path']])
        self.csp[bw_key].corpus_config.segments.file = ("/").join(
            [self.cross_validation_info['pre_path'], self.cross_validation_info['merged_corpus_segment']])

    # -------------------- Helpers --------------------
    def _get_prior_info_dict(self):
        prior_dict = {}
        for k in ['left-context', 'center-state', 'right-context']:
            prior_dict[f'{k}-prior'] = {}
            prior_dict[f'{k}-prior']['scale'] = 0.0
            prior_dict[f'{k}-prior']['file'] = None
        return prior_dict

    def _add_output_alias_for_train_job(
        self,
        train_job: Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob],
        train_corpus_key: str,
        cv_corpus_key: str,
        name: str,
    ):
        train_job.add_alias(f"train_nn/{train_corpus_key}_{cv_corpus_key}/{name}_train")
        tk.register_output(
            f"train_nn/{train_corpus_key}_{cv_corpus_key}/{name}_learning_rate.png",
            train_job.out_plot_lr,
        )

    def _get_model_checkpoint(self, model_job, epoch):
        return model_job.checkpoints[epoch]

    def _get_model_path(self, model_job, epoch):
        return model_job.checkpoints[epoch].ckpt_path

    def _delete_multitask_components(self, returnn_config):
        config = copy.deepcopy(returnn_config)

        for o in ["left", "right"]:
            if 'linear' in config.config['network'][f'{o}-output']['from']:
                for i in [1, 2]:
                    for k in ['leftContext', 'triphone']:
                        del config.config['network'][f'linear{i}-{k}']
            del config.config['network'][f'{o}-output']

        return config

    def _delete_mlps(self, returnn_config, keys=None, source=None):
        #source is either a string or a tupe
        if source is None:
            source = ['encoder-output']
        elif isinstance(source, tuple):
            source = list(source)
        else: source = [source]

        returnn_cfg = copy.deepcopy(returnn_config)
        if keys is None:
            keys = ['left', 'center', 'right']
        for l in list(returnn_cfg.config['network'].keys()):
            if 'linear' in l:
                del returnn_cfg.config['network'][l]
        for o in keys:
            returnn_cfg.config['network'][f'{o}-output']['from'] = source
        return returnn_cfg



    # -------------------- Training --------------------

    def returnn_training(
        self,
        name,
        returnn_config,
        nn_train_args,
        train_corpus_key,
        cv_corpus_key,
        devtrain_corpus_key=None,
    ):
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        returnn_config.config["train"] = self.train_input_data[
            train_corpus_key
        ].get_data_dict()
        returnn_config.config["dev"] = self.cv_input_data[cv_corpus_key].get_data_dict()
        if devtrain_corpus_key is not None:
            returnn_config.config["eval_datasets"] = {
                "devtrain": self.devtrain_input_data[
                    devtrain_corpus_key
                ].get_data_dict()
            }

        train_job = returnn.ReturnnTrainingJob(
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
        )
        self._add_output_alias_for_train_job(
            train_job=train_job,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=cv_corpus_key,
            name=name,
        )

        return train_job

    def returnn_rasr_training(
        self,
        name,
        returnn_config,
        nn_train_args,
        train_corpus_key,
        cv_corpus_key,
    ):
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[cv_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        assert train_data.feature_flow == dev_data.feature_flow
        assert train_data.features == dev_data.features
        assert train_data.alignments == dev_data.alignments

        if train_data.feature_flow is not None:
            feature_flow = train_data.feature_flow
        else:
            if isinstance(train_data.features, rasr.FlagDependentFlowAttribute):
                feature_path = train_data.features
            elif isinstance(train_data.features, (MultiPath, MultiOutputPath)):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": train_data.features,
                    },
                )
            elif isinstance(train_data.features, tk.Path):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "bundle": train_data.features,
                    },
                )
            else:
                raise NotImplementedError

            feature_flow = features.basic_cache_flow(feature_path)
            if isinstance(train_data.features, tk.Path):
                feature_flow.flags = {"cache_mode": "bundle"}

        if isinstance(train_data.alignments, rasr.FlagDependentFlowAttribute):
            alignments = copy.deepcopy(train_data.alignments)
            net = rasr.FlowNetwork()
            net.flags = {"cache_mode": "bundle"}
            alignments = alignments.get(net)
        elif isinstance(train_data.alignments, (MultiPath, MultiOutputPath)):
            raise NotImplementedError
        elif isinstance(train_data.alignments, tk.Path):
            alignments = train_data.alignments
        else:
            raise NotImplementedError

        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = returnn.ReturnnRasrTrainingJob(
            train_crp=train_crp,
            dev_crp=dev_crp,
            feature_flow=feature_flow,
            alignment=alignments,
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
        )
        self._add_output_alias_for_train_job(
            train_job=train_job,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=cv_corpus_key,
            name=name,
        )

        return train_job


    def returnn_rasr_training_fullsum(
        self,
        experiment_key,
        train_corpus_key,
        dev_corpus_key,
        train_feature_corpus,
        dev_feature_corpus,
        nn_train_args,
    ):
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[dev_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()


        """
        if 'extra_rasr_config' not in nn_train_args:
            nn_train_args['extra_rasr_config'] = get_extra_config_segment_order(size=300)
        else:
            extra_rasr_config_additional =  n_train_args['extra_rasr_config']
            nn_train_args['extra_rasr_config'] = get_extra_config_segment_order(size=300, extra_config=extra_rasr_config_additional)
        """

        if 'returnn_config' not in nn_train_args:
            returnn_config = self.experiments[experiment_key]['returnn_config']
        else: returnn_config = nn_train_args.pop('returnn_config')
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = returnn.ReturnnRasrTrainingBWJob(
            train_crp=train_crp,
            dev_crp=dev_crp,
            feature_flows={"train": self.feature_flows[train_feature_corpus],
                           "dev": self.feature_flows[dev_feature_corpus]},
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
        )

        self._add_output_alias_for_train_job(
            train_job=train_job,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=dev_corpus_key,
            name=self.experiments[experiment_key]['name'],
        )

        self.experiments[experiment_key]["train_job"] = train_job

    # -------------------- Recognition --------------------




    # -------------------- run setup  --------------------

    def run(self, steps: RasrSteps):
        if "init" in steps.get_step_names_as_list():
            self.init_system()
        for eval_c in self.dev_corpora + self.test_corpora:
            stm_args = (
                self.hybrid_init_args.stm_args
                if self.hybrid_init_args.stm_args is not None
                else {}
            )
            self.create_stm_from_corpus(eval_c, **stm_args)
            self._set_scorer_for_corpus(eval_c)

        for step_idx, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                if step_args is None:
                    step_args = self.hybrid_init_args.feature_extraction_args
                for all_c in (
                    self.train_corpora
                    + self.cv_corpora
                    + self.devtrain_corpora
                    + self.dev_corpora
                    + self.test_corpora
                ):
                    self.feature_caches[all_c] = {}
                    self.feature_bundles[all_c] = {}
                    self.feature_flows[all_c] = {}
                self.extract_features(step_args)
                # ---------- Step Output ----------
            if step_name.startswith("output"):
                self.run_output_step(step_args, step_idx=step_idx, steps=steps)

