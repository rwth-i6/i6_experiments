__all__ = ["UkrainianGMMSystem",
           "UkrainianFactoredHybridSystem"]

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
import i6_core.corpus as corpus_recipe
import i6_core.features as features
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.returnn as returnn
import i6_core.text as text

from i6_core.util import MultiPath, MultiOutputPath


from i6_experiments.common.setups.rasr.nn_system import (
    NnSystem
)

from i6_experiments.common.setups.rasr.gmm_system import (
    GmmSystem
)

from i6_experiments.common.setups.rasr.util import (
    RasrInitArgs,
    RasrDataInput,
    ReturnnRasrDataInput,
    RasrSteps,
)

from i6_experiments.common.datasets.librispeech.constants import (
    num_segments
)


#get_recog_ctx_args*() functions are imported here
from i6_experiments.users.raissi.experiments.librispeech.search.recognition_args import *
from i6_experiments.users.raissi.setups.common.helpers.estimate_povey_like_prior_fh import *


from i6_experiments.users.raissi.returnn.rasr_returnn_bw import (
    ReturnnRasrTrainingBWJob
)

from i6_experiments.users.raissi.setups.common.helpers.pipeline_data import (
    ContextEnum,
    ContextMapper,
    LabelInfo,
    PipelineStages,
    SprintFeatureToHdf
)

from i6_experiments.users.raissi.setups.common.helpers.network_architectures import (
    get_graph_from_returnn_config
)

from i6_experiments.users.raissi.setups.common.helpers.train_helpers import (
    get_extra_config_segment_order,
)

from i6_experiments.users.raissi.setups.common.helpers.specaugment_returnn_epilog import (
    get_specaugment_epilog,
)

from i6_experiments.users.raissi.setups.common.helpers.returnn_epilog import (
    get_epilog_code_dense_label,
)

from i6_experiments.users.raissi.setups.common.util.rasr import (
    SystemInput,
)

from i6_experiments.users.raissi.setups.librispeech.util.pipeline_helpers import (
    get_label_info,
    get_alignment_keys,
    get_lexicon_args,
    get_tdp_values,
)

from i6_experiments.users.raissi.setups.librispeech.search.factored_hybrid_search import(
    FHDecoder
)
# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- GMM System --------------------
#do I need this?
class UkrainianGMMSystem(GmmSystem):
    def run(self, steps: Union[List, Tuple] = ("all",)):
        if "init" in steps:
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        for all_c in self.train_corpora + self.dev_corpora + self.test_corpora:
            costa_args = copy.deepcopy(self.rasr_init_args.costa_args)
            if self.crp[all_c].language_model_config is None:
                costa_args["eval_lm"] = False
            self.costa(all_c, prefix="costa/", **costa_args)
            if costa_args["eval_lm"]:
                self.jobs[all_c]["costa"].update_rqmt("run", {"mem": 24, "time": 24})

        for trn_c in self.train_corpora:
            self.store_allophones(trn_c)

        for eval_c in self.dev_corpora + self.test_corpora:
            self.create_stm_from_corpus(eval_c)
            self.set_sclite_scorer(eval_c)

        if "extract" in steps:
            self.extract_features(
                feat_args=self.rasr_init_args.feature_extraction_args
            )

        # ---------- Monophone ----------
        if "mono" in steps:
            for trn_c in self.train_corpora:
                self.monophone_training(
                    corpus=trn_c,
                    linear_alignment_args=self.monophone_args.linear_alignment_args,
                    **self.monophone_args.training_args,
                )

                for dev_c in self.dev_corpora:
                    feature_scorer = (
                        trn_c,
                        f"train_{self.monophone_args.training_args['name']}",
                    )

                    self.recognition(
                        f"{self.monophone_args.training_args['name']}-{trn_c}-{dev_c}",
                        corpus=dev_c,
                        feature_scorer=feature_scorer,
                        **self.monophone_args.recognition_args,
                    )

                for tst_c in self.test_corpora:
                    pass

        # ---------- CaRT ----------
        if "cart" in steps:
            for trn_c in self.train_corpora:
                self.cart_and_lda(
                    corpus=trn_c,
                    **self.triphone_args.cart_lda_args,
                )

        # ---------- Triphone ----------
        if "tri" in steps:
            for trn_c in self.train_corpora:
                self.triphone_training(
                    corpus=trn_c,
                    **self.triphone_args.training_args,
                )

                for dev_c in self.dev_corpora:
                    feature_scorer = (
                        trn_c,
                        f"train_{self.triphone_args.training_args['name']}",
                    )

                    self.recognition(
                        f"{self.triphone_args.training_args['name']}-{trn_c}-{dev_c}",
                        corpus=dev_c,
                        feature_scorer=feature_scorer,
                        **self.triphone_args.recognition_args,
                    )

                for c in self.test_corpora:
                    pass

                # ---------- SDM Tri ----------
                self.single_density_mixtures(
                    corpus=trn_c,
                    **self.triphone_args.sdm_args,
                )

        # ---------- VTLN ----------
        if "vtln" in steps:
            for trn_c in self.train_corpora:
                self.vtln_feature_flow(
                    train_corpora=trn_c,
                    corpora=[trn_c] + self.dev_corpora + self.test_corpora,
                    **self.vtln_args.training_args["feature_flow"],
                )

                self.vtln_warping_mixtures(
                    corpus=trn_c,
                    feature_flow=self.vtln_args.training_args["feature_flow"]["name"],
                    **self.vtln_args.training_args["warp_mix"],
                )

                self.extract_vtln_features(
                    name=self.triphone_args.training_args["feature_flow"],
                    train_corpus=trn_c,
                    eval_corpus=self.dev_corpora + self.test_corpora,
                    raw_feature_flow=self.vtln_args.training_args["feature_flow"][
                        "name"
                    ],
                    vtln_files=self.vtln_args.training_args["warp_mix"]["name"],
                )

                self.vtln_training(
                    corpus=trn_c,
                    **self.vtln_args.training_args["train"],
                )

                for dev_c in self.dev_corpora:
                    feature_scorer = (
                        trn_c,
                        f"train_{self.vtln_args.training_args['train']['name']}",
                    )

                    self.recognition(
                        f"{self.vtln_args.training_args['train']['name']}-{trn_c}-{dev_c}",
                        corpus=dev_c,
                        feature_scorer=feature_scorer,
                        **self.vtln_args.recognition_args,
                    )

                for tst_c in self.test_corpora:
                    pass

                # ---------- SDM VTLN ----------
                self.single_density_mixtures(
                    corpus=trn_c,
                    **self.vtln_args.sdm_args,
                )

        # ---------- SAT ----------
        if "sat" in steps:
            for trn_c in self.train_corpora:
                self.sat_training(
                    corpus=trn_c,
                    **self.sat_args.training_args,
                )

                for dev_c in self.dev_corpora:
                    pass

                for tst_c in self.test_corpora:
                    pass

                # ---------- SDM Sat ----------
                self.single_density_mixtures(
                    corpus=trn_c,
                    **self.sat_args.sdm_args,
                )

        # ---------- VTLN+SAT ----------
        if "vtln+sat" in steps:
            for trn_c in self.train_corpora:
                self.sat_training(
                    corpus=trn_c,
                    **self.vtln_sat_args.training_args,
                )

                for dev_c in self.dev_corpora:
                    pass

                for tst_c in self.test_corpora:
                    pass

                # ---------- SDM VTLN+SAT ----------
                self.single_density_mixtures(
                    corpus=trn_c,
                    **self.vtln_sat_args.sdm_args,
                )


# -------------------- Factored Hybrid System --------------------

class UkrainianFactoredHybridSystem(NnSystem):
    """
    self.crp_names are the corpora used during training and decoding: train, cvtrain, devtrain for train and all corpora for decoding
    """

    def __init__(
            self,
            returnn_root: Optional[str] = None,
            returnn_python_home: Optional[str] = None,
            returnn_python_exe: Optional[str] = None,
            # tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
            rasr_binary_path: Optional[str] = tk.Path(('/').join([gs.RASR_ROOT, 'arch', 'linux-x86_64-standard'])),
            rasr_init_args: RasrInitArgs = None,
            train_data: Dict[str, RasrDataInput] = None,
            dev_data: Dict[str, RasrDataInput] = None,
            test_data: Dict[str, RasrDataInput] = None,
    ):
        super().__init__(
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
        )
        # arguments used for the initialization of the system, they should be set before running the system
        self.rasr_init_args = rasr_init_args
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data


        # general modeling approach
        self.label_info = LabelInfo(**get_label_info())
        self.lexicon_args = get_lexicon_args(norm_pronunciation=False)
        self.tdp_values = get_tdp_values()

        # transcription priors in pickle format
        self.priors = None #these are for transcript priors

        # dataset infomration
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
        self.initial_nn_args = {'num_input': 40}

        self.initial_train_args = {'time_rqmt': 168,
                                   'mem_rqmt': 40,
                                   'log_verbosity': 3,
                                   }

        # pipeline info
        self.context_mapper = ContextMapper()
        self.stage = PipelineStages(get_alignment_keys())
        self.contexts = {'mono': ContextEnum(self.context_mapper.get_enum(1)),
                         'di'  : ContextEnum(self.context_mapper.get_enum(2)),
                         'tri' : ContextEnum(self.context_mapper.get_enum(4)),
                         }

        self.trainers = None  # ToDo external trainer class
        self.recognizers = {}
        self.aligners = {}
        self.returnn_configs = {}
        self.graphs = {}

        self.experiments = {}

        # inference related
        self.tf_map = {"triphone": "right", "diphone": "center", "context": "left"}  # Triphone forward model
        self.tf_library = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/binaries/recognition/NativeLstm2.so"

        self.inputs = {}
        self.train_key = None  # "train"

    # ----------- pipeline construction -----------------
    def set_experiment_dict(self, key, alignment, context, postfix_name=""):
        name = self.stage.get_name(alignment, context)
        self.experiments[key] = {
            "name": ('-').join([name, postfix_name]),
            "train_job": None,
            "graph": {"train": None, "inference": None},
            "returnn_config": None,
            "align_job": None,
            "decode_job": {"runner": None, "args": None},
            "extra_returnn_code": {"epilog": "", "prolog": ""}
        }

    def set_crp_pairings(self):
        keys = self.corpora.keys()
        train_key = [k for k in keys if 'train' in k][0]
        all_names = ['train', 'cvtrain', 'devtrain']
        all_names.extend([n for n in keys if n != train_key])
        self.crp_names = dict()
        for k in all_names:
            if 'train' in k:
                self.crp_names[k] = f'{train_key}.{k}'
            else:
                self.crp_names[k] = k

    def set_returnn_config_for_experiment(self, key, returnn_config):
        assert key in self.experiments.keys()
        self.experiments[key]["returnn_config"] = returnn_config
        self.experiments[key]["extra_returnn_code"]["prolog"] = returnn_config.python_prolog
        self.experiments[key]["extra_returnn_code"]["epilog"] = returnn_config.python_epilog

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
            train_job: Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob, ReturnnRasrTrainingBWJob],
            name: str,
    ):
        train_job.add_alias(f"train/nn_{name}")
        tk.register_output(
            f"train/{name}_learning_rate.png",
            train_job.out_plot_lr,
        )

    def _get_model_checkpoint(self, model_job, epoch):
        return model_job.out_checkpoints[epoch]

    def _get_model_path(self, model_job, epoch):
        return model_job.out_checkpoints[epoch].ckpt_path

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
        # source is either a string or a tupe
        if source is None:
            source = ['encoder-output']
        elif isinstance(source, tuple):
            source = list(source)
        else:
            source = [source]

        returnn_cfg = copy.deepcopy(returnn_config)
        if keys is None:
            keys = ['left', 'center', 'right']
        for l in list(returnn_cfg.config['network'].keys()):
            if 'linear' in l:
                del returnn_cfg.config['network'][l]
        for o in keys:
            returnn_cfg.config['network'][f'{o}-output']['from'] = source
        return returnn_cfg

    def set_local_flf_tool(self, path=None):
        if path is None:
            path = "/u/raissi/dev/rasr_tf14py38_private/src/Tools/Flf/flf-tool.linux-x86_64-standard"
        self.csp["base"].flf_tool_exe = path

    # --------------------- Init procedure -----------------
    def init_system(self):
        for param in [self.rasr_init_args, self.label_info, self.train_data, self.dev_data, self.test_data]:
            assert param is not None
        self._init_am(**self.rasr_init_args.am_args)
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

    def init_datasets(
            self,
            train_data: Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]],
            cv_data: Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]],
            devtrain_data: Optional[
                Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]]
            ] = None,
            dev_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
            test_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
    ):
        devtrain_data = devtrain_data if devtrain_data is not None else {}
        dev_data = dev_data if dev_data is not None else {}
        test_data = test_data if test_data is not None else {}

        self._assert_corpus_name_unique(
            train_data, cv_data, devtrain_data, dev_data, test_data
        )

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

    def get_system_input(
            self,
            corpus_key: str,
            extract_features: List[str],

    ):
        """
        :param corpus_key: corpus name identifier
        :param extract_features: list of features to extract for later usage
        :return SystemInput:
        """
        sys_in = SystemInput()
        sys_in.crp = self.crp[corpus_key]
        sys_in.feature_flows = self.feature_flows[corpus_key]
        sys_in.features = self.feature_caches[corpus_key]
        if corpus_key in self.alignments:
            sys_in.alignments = self.alignments[corpus_key]

        for feat_name in extract_features:
            tk.register_output(
                f"features/{corpus_key}_{feat_name}_features.bundle",
                self.feature_bundles[corpus_key][feat_name],
            )

        return sys_in

    def run_input_step(self, step_args):
        for corpus_key, corpus_type in step_args.corpus_type_mapping.items():
            if (
                    corpus_key
                    not in self.train_corpora + self.dev_corpora + self.test_corpora
            ):
                continue
            if 'train' in corpus_key:
                if self.train_key is None:
                    self.train_key = corpus_key
                else:
                    if self.train_key != corpus_key:
                        assert (False,
                                "You already set the train key to be {self.train_key}, you cannot have more than one train key")
            if corpus_key not in self.inputs.keys():
                self.inputs[corpus_key] = {}
            self.inputs[corpus_key][step_args.name] = self.get_system_input(
                corpus_key,
                step_args.extract_features,
            )

    def _set_train_data(self, data_dict):
        for c_key, c_data in data_dict.items():
            self.crp[c_key] = c_data.get_crp() if c_data.crp is None else c_data.crp
            self.feature_flows[c_key] = c_data.feature_flow
            if c_data.alignments:
                self.alignments[c_key] = c_data.alignments

    def _set_eval_data(self, data_dict):
        for c_key, c_data in data_dict.items():
            self.ctm_files[c_key] = {}
            self.crp[c_key] = c_data.get_crp() if c_data.crp is None else c_data.crp
            self.feature_flows[c_key] = c_data.feature_flow
            self.set_sclite_scorer(c_key)

    def _update_crp_am_setting(self, crp_key, tdp_type=None, add_base_allophones=False):
        # ToDo handle different tdp values: default, based on transcription, based on an alignment
        tdp_pattern = self.tdp_values['pattern']
        if tdp_type in ['default']:  # additional later, maybe enum or so
            tdp_values = self.tdp_values[tdp_type]

        elif isinstance(tdp_type, tuple):
            tdp_values = self.tdp_values[tdp_type[0]][tdp_type[1]]

        else:
            print("Not implemented tdp type")
            import sys
            sys.exit()

        crp = self.crp[crp_key]
        for ind, ele in enumerate(tdp_pattern):
            for type in ["*", "silence"]:
                crp.acoustic_model_config["tdp"][type][ele] = tdp_values[type][ind]


        if self.label_info.use_word_end_classes:
            crp.acoustic_model_config.state_tying.use_word_end_classes = self.label_info.use_word_end_classes
        crp.acoustic_model_config.state_tying.use_boundary_classes = self.label_info.use_boundary_classes
        crp.acoustic_model_config.hmm.states_per_phone = self.label_info.n_states_per_phone
        if 'train' in crp_key:
            crp.acoustic_model_config.state_tying.type = self.label_info.state_tying
            self.label_info.set_sil_ids(crp)
        else:
            crp.acoustic_model_config.state_tying.type = 'no-tying-dense'  # for correct tree of dependency

        crp.acoustic_model_config.allophones.add_all = self.lexicon_args['add_all_allophones']
        crp.acoustic_model_config.allophones.add_from_lexicon = not self.lexicon_args['add_all_allophones']
        if add_base_allophones:
            crp.acoustic_model_config.allophones.add_from_file = self.base_allophones

        crp.lexicon_config.normalize_pronunciation = self.lexicon_args['norm_pronunciation']


    def _update_am_setting_for_all_crps(self, train_tdp_type, eval_tdp_type, add_base_allophones=False):
        types = {'train': train_tdp_type, 'eval': eval_tdp_type}
        for t in types.keys():
            if types[t] == 'heuristic':
                if self.label_info.n_states_per_phone > 1:
                    types[t] = (types[t], 'threepartite')
                else:
                    types[t] = (types[t], 'monostate')

        for crp_k in self.crp_names.keys():
            if 'train' in crp_k:
                self._update_crp_am_setting(crp_key=self.crp_names[crp_k], tdp_type=types['train'],
                                            add_base_allophones=add_base_allophones)
            else:
                self._update_crp_am_setting(crp_key=self.crp_names[crp_k], tdp_type=types['eval'],
                                            add_base_allophones=add_base_allophones)

    # ----- data preparation for train-----------------------------------------------------
    def get_epilog_for_train(self, specaug_args=None):
        if specaug_args is not None:
            spec_augment_epilog = get_specaugment_epilog(**specaug_args)
        else:
            spec_augment_epilog = None
        return get_epilog_code_dense_label(n_input=self.initial_nn_args["num_input"],
                                           n_contexts=self.label_info.n_contexts,
                                           n_states=self.label_info.n_states_per_phone,
                                           specaugment=spec_augment_epilog)

    def prepare_train_data_with_cv_from_train(self, input_key, chunk_size=1152):
        train_corpus_path = self.corpora[self.train_key].corpus_file
        total_train_num_segments = num_segments[self.train_key]
        cv_size = 300 / total_train_num_segments

        all_segments = corpus_recipe.SegmentCorpusJob(
            train_corpus_path, 1
        ).out_single_segment_files[1]

        splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
            all_segments, {"train": 1 - cv_size, "cv": cv_size}
        )
        train_segments = splitted_segments_job.out_segments["train"]
        cv_segments = splitted_segments_job.out_segments["cv"]
        devtrain_segments = text.TailJob(
            train_segments, num_lines=1000, zip_output=False
        ).out

        # ******************** NN Init ********************
        nn_train_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(shuffle_data=True,
                                                                                          segment_order_sort_by_time_length=True,
                                                                                          chunk_size=chunk_size)
        nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
        nn_train_data_inputs = {self.crp_names['train']: nn_train_data}

        nn_cv_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input()
        nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
        nn_cv_data_inputs = {self.crp_names['cvtrain']: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input()
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names['devtrain']: nn_devtrain_data}

        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs


    def set_rasr_returnn_input_datas(self, input_key, chunk_size=1152):
        for k in self.corpora.keys():
            assert self.inputs[k] is not None
            assert self.inputs[k][input_key] is not None

        nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs = prepare_train_data_with_separate_cv(input_key, chunk_size)

        nn_dev_data_inputs = {
            self.crp_names['dev']: self.inputs["dev-clean"][input_key].as_returnn_rasr_data_input(),
        }
        nn_test_data_inputs = {
            self.crp_names['test']: self.inputs["test-clean"][input_key].as_returnn_rasr_data_input(),
        }

        self.init_datasets(
            train_data=nn_train_data_inputs,
            cv_data=nn_cv_data_inputs,
            devtrain_data=nn_devtrain_data_inputs,
            dev_data=nn_dev_data_inputs,
            test_data=nn_test_data_inputs)
        label_info_args = {
            'n_states_per_phone': self.label_info.n_states_per_phone,
            'n_contexts': self.label_info.n_contexts}
        self.initial_nn_args.update(label_info_args)

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
            name=name,
        )

        return train_job

    def returnn_rasr_training(
            self,
            experiment_key,
            train_corpus_key,
            dev_corpus_key,
            nn_train_args,
    ):
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[dev_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        if 'returnn_config' not in nn_train_args:
            returnn_config = self.experiments[experiment_key]['returnn_config']
        else:
            returnn_config = nn_train_args.pop('returnn_config')
        assert isinstance(returnn_config, returnn.ReturnnConfig)

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
            name=self.experiments[experiment_key]['name'],
        )
        self.experiments[experiment_key]["train_job"] = train_job
        self.set_graph_for_experiment(experiment_key)

    def returnn_rasr_training_fullsum(
            self,
            experiment_key,
            train_corpus_key,
            dev_corpus_key,
            nn_train_args,
    ):
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[dev_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        if 'returnn_config' not in nn_train_args:
            returnn_config = self.experiments[experiment_key]['returnn_config']
        else:
            returnn_config = nn_train_args.pop('returnn_config')
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = ReturnnRasrTrainingBWJob(
            train_crp=train_crp,
            dev_crp=dev_crp,
            feature_flows={"train": train_data.feature_flow,
                           "dev": dev_data.feature_flow},
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
        )

        self._add_output_alias_for_train_job(
            train_job=train_job,
            name=self.experiments[experiment_key]['name'],
        )

        self.experiments[experiment_key]["train_job"] = train_job
        self.set_graph_for_experiment(experiment_key)

    # ---------------------Prior Estimation--------------
    def get_hdf_path(self, hdf_key):
        if hdf_key is not None:
            assert hdf_key in self.hdfs.keys()
            return self.hdfs[hdf_key]

        if self.train_key not in self.hdfs.keys():
            self.create_hdf()

        return self.hdfs[self.train_key]

    def create_hdf(self):
        gammaton_features_paths = self.feature_caches[self.train_key]['gt'].hidden_paths
        feature_caches = [gammaton_features_paths[i].get_path() for i in
                          range(1, len(gammaton_features_paths.keys()) + 1)]
        hdfJob = SprintFeatureToHdf(feature_caches)
        self.hdfs[self.train_key] = hdfJob.hdf_files

        hdfJob.add_alias(f"hdf/{self.train_key}")
        tk.register_output(f"hdf/{self.train_key}.hdf.1", self.hdfs[self.train_key][0])

    def set_mono_priors(self, key, epoch, tf_library=None, tm=None, nStateClasses=None, hdf_key='960'):
        if nStateClasses is None:
            nStateClasses = self.label_info.get_n_state_classes()

        if tm is None:
            tm = self.tf_map

        if tf_library is None:
            tf_library = self.tf_library

        name = f"{self.experiments[key]['name']}-epoch-{epoch}"
        model_checkpoint = self._get_model_checkpoint(
            self.experiments[key]["train_job"], epoch
        )
        graph = self.experiments[key]["graph"]["inference"]

        hdf_paths = self.get_hdf_path(hdf_key)

        estimateJob = EstimateMonophonePriors_(graph=graph,
                                               model=model_checkpoint,
                                               dataPaths=hdf_paths,
                                               datasetIndices=list(range(len(hdf_paths) // 3)),
                                               libraryPath=tf_library,
                                               nStates=nStateClasses,
                                               tensorMap=tm,
                                               gpu=1)
        if name is not None:
            estimateJob.add_alias(f"priors/priors-{name}")
        xmlJob = DumpXmlForMonophone(estimateJob.priorFiles,
                                     estimateJob.numSegments,
                                     nStates=nStateClasses)
        priorFiles = [xmlJob.centerPhonemeXml]
        if name is not None:
            xmlName = f"priors/{name}-xmlpriors"
        else:
            xmlName = "mono-prior"
        tk.register_output(xmlName, priorFiles[0])
        self.experiments[key]["priors"] = priorFiles

    def set_diphone_priors(self, key, epoch, tf_library=None, nStateClasses=None, nContexts=None,
                           gpu=1, time=20, isSilMapped=True, hdf_key=None):
        assert (self.label_info.sil_id is not None)
        if nStateClasses is None:
            nStateClasses = self.label_info.get_n_state_classes()
        if nContexts is None:
            nContexts = self.label_info.n_contexts

        if tf_library is None:
            tf_library = self.tf_library

        name = f"{self.experiments[key]['name']}-epoch-{epoch}"
        model_checkpoint = self._get_model_checkpoint(
            self.experiments[key]["train_job"], epoch
        )
        graph = self.experiments[key]["graph"]["inference"]

        hdf_paths = self.get_hdf_path(hdf_key)

        estimateJob = EstimateRasrDiphoneAndContextPriors(graphPath=graph,
                                                          model=model_checkpoint,
                                                          dataPaths=hdf_paths,
                                                          datasetIndices=list(range(len(hdf_paths) // 3)),
                                                          libraryPath=tf_library,
                                                          nStates=nStateClasses,
                                                          tensorMap=self.tf_map,
                                                          nContexts=nContexts,
                                                          nStateClasses=nStateClasses,
                                                          gpu=gpu,
                                                          time=time)

        estimateJob.add_alias(f"priors-{name}")
        xmlJob = DumpXmlRasrForDiphone(estimateJob.diphoneFiles,
                                       estimateJob.contextFiles,
                                       estimateJob.numSegments,
                                       nContexts=nContexts,
                                       nStateClasses=nStateClasses,
                                       adjustSilence=isSilMapped,
                                       silBoundaryIndices=[0, self.label_info.sil_id])

        priorFiles = [xmlJob.diphoneXml, xmlJob.contextXml]
        if name is not None:
            xmlName = f"priors/{name}-xmlpriors"
        else:
            xmlName = "diphone-priors"
        tk.register_output(xmlName, priorFiles[0])
        self.experiments[key]["priors"] = priorFiles

    # -------------------- Decoding --------------------
    def set_graph_for_experiment(self, key):
        config = copy.deepcopy(self.experiments[key]["returnn_config"])
        name = self.experiments[key]["name"]
        python_prolog = self.experiments[key]["extra_returnn_code"]["prolog"]
        python_epilog = self.experiments[key]["extra_returnn_code"]["epilog"]

        t_graph = get_graph_from_returnn_config(config, python_prolog, python_epilog)
        if "source" in config.config["network"].keys():  # specaugment
            for k in ["fwd", "bwd"]:
                config.config["network"][f"{k}_1"]["from"] = "data"
            infer_graph = get_graph_from_returnn_config(config, python_prolog, python_epilog)
        else:
            infer_graph = t_graph

        self.experiments[key]["graph"]["inference"] = infer_graph
        tk.register_output(f'graphs/{name}-infer_graph', infer_graph)

    def get_recognizer_and_args(
            self,
            key,
            context_type,
            epoch,
            crp_corpus=None,
            gpu=True,
            is_min_duration=False,
            is_multi_encoder_output=False,
            tf_library=None,
            dummy_mixtures=None,
    ):

        name = ('-').join([self.experiments[key]["name"], crp_corpus, f'e{epoch}-'])
        if context_type.value in [self.context_mapper.get_enum(i) for i in range(6, 9)]:
            name = f'{self.experiments[key]["name"]}-delta-e{epoch}-'

        model_path = self._get_model_path(self.experiments[key]["train_job"], epoch)
        num_encoder_output = (
                self.experiments[key]["returnn_config"].config["network"]["fwd_1"]["n_out"]
                * 2
        )
        p_info = self._get_prior_info_dict()
        assert self.experiments[key]['priors'] is not None

        isSpecAug = (
            True
            if "source"
               in self.experiments[key]["returnn_config"].config["network"].keys()
            else False
        )
        if context_type.value in [
            self.context_mapper.get_enum(1),
            self.context_mapper.get_enum(7),
        ]:
            if isSpecAug:
                recog_args = get_recog_mono_specAug_args()
            else:
                recog_args = get_recog_mono_args()
            scales = recog_args["priorScales"]
            del recog_args["priorScales"]
            p_info['center-state-prior']['scale'] = scales['center-state']
            p_info['center-state-prior']['file'] = self.experiments[key]['priors'][0]
            recog_args["priorInfo"] = p_info

        elif context_type.value in [
            self.context_mapper.get_enum(2),
            self.context_mapper.get_enum(8)
        ]:
            recog_args = get_recog_diphone_fromGmm_specAug_args()
            scales = recog_args["shared_args"]["priorScales"]
            del recog_args["shared_args"]["priorScales"]
            p_info['center-state-prior']['scale'] = scales['center-state']
            p_info['left-context-prior']['scale'] = scales['left-context']
            p_info['center-state-prior']['file'] = self.experiments[key]['priors'][0]
            p_info['left-context-prior']['file'] = self.experiments[key]['priors'][1]
            recog_args["shared_args"]["priorInfo"] = p_info
        else:
            print("implement other contexts")
            assert (False)

        recog_args["use_word_end_classes"] = self.label_info.use_word_end_classes
        recog_args["n_states_per_phone"] = self.label_info.n_states_per_phone
        recog_args["n_contexts"] = self.label_info.n_contexts
        recog_args["is_min_duration"] = is_min_duration
        recog_args["num_encoder_output"] = num_encoder_output

        if context_type.value not in [
            self.context_mapper.get_enum(1),
            self.context_mapper.get_enum(7),
        ]:
            recog_args["4gram_args"].update(recog_args["shared_args"])
            recog_args["lstm_args"].update(recog_args["shared_args"])

        if dummy_mixtures is None:
            dummy_mixtures = mm.CreateDummyMixturesJob(self.label_info.get_n_of_dense_classes(),
                                                       self.initial_nn_args["num_input"]).out_mixtures  # gammatones

        assert (self.label_info.sil_id is not None)

        recognizer = FHDecoder(
            name=name,
            search_crp=self.crp[crp_corpus],
            context_type=context_type,
            context_mapper=self.context_mapper,
            feature_path=self.feature_flows[crp_corpus],
            model_path=model_path,
            graph=self.experiments[key]["graph"]["inference"],
            mixtures=dummy_mixtures,
            eval_files=self.scorer_args[crp_corpus],
            tf_library=tf_library,
            is_multi_encoder_output=is_multi_encoder_output,
            silence_id=self.label_info.sil_id,
            gpu=gpu,
        )

        return recognizer, recog_args

    # -------------------- run setup  --------------------

    def run(self, steps: RasrSteps):
        if "init" in steps.get_step_names_as_list():
            self.init_system()
        for eval_c in self.dev_corpora + self.test_corpora:
            stm_args = (
                self.rasr_init_args.stm_args
                if self.rasr_init_args.stm_args is not None
                else {}
            )
            self.create_stm_from_corpus(eval_c, **stm_args)
            self._set_scorer_for_corpus(eval_c)

        for step_idx, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                if step_args is None:
                    step_args = self.rasr_init_args.feature_extraction_args
                step_args['gt']['prefix'] = 'features/'
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
                # ---------- Step Input ----------
            if step_name.startswith("input"):
                self.run_input_step(step_args)

