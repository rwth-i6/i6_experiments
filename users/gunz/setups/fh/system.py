__all__ = ["FactoredHybridSystem"]

import copy
import itertools
import math
import random
import typing
from typing import Dict, List, Optional, Union

# -------------------- Sisyphus --------------------
from sisyphus import gs, tk
from sisyphus.delayed_ops import DelayedBase, DelayedFallback

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
import i6_core.features as features
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.returnn as returnn
import i6_core.text as text
from i6_core.util import MultiPath, MultiOutputPath


from i6_experiments.common.datasets.librispeech.constants import num_segments
from i6_experiments.common.setups.rasr.nn_system import NnSystem
from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
    ReturnnRasrDataInput,
    RasrSteps,
)


from i6_private.users.gunz.util.delayed import DelayedArgMin

from ..common.compile_graph import compile_tf_graph_from_returnn_config
from .decoder.config import PriorConfig, PriorInfo
from .decoder.search import FHDecoder, SearchParameters
from .factored import PhoneticContext, LabelInfo
from .priors import (
    DumpXmlForMonophoneJob,
    DumpXmlForDiphoneJob,
    DumpXmlForTriphoneJob,
    EstimateMonophonePriorsJob,
    EstimateDiphonePriorsJob,
    EstimateTriphoneForwardPriorsJob,
)
from .util.hdf import SprintFeatureToHdf
from .util.pipeline_helpers import get_lexicon_args, get_tdp_values
from .util.rasr import SystemInput

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------

PRIOR_RNG_SEED = 133769420


class ExtraReturnnCode(typing.TypedDict):
    epilog: str
    prolog: str


class Graphs(typing.TypedDict):
    train: typing.Optional[tk.Path]
    inference: typing.Optional[tk.Path]


class Experiment(typing.TypedDict, total=False):
    extra_returnn_code: ExtraReturnnCode
    name: str
    graph: Graphs
    priors: typing.Optional[PriorInfo]
    returnn_config: typing.Optional[returnn.ReturnnConfig]
    train_job: typing.Optional[returnn.ReturnnRasrTrainingJob]


class FactoredHybridSystem(NnSystem):
    """
    self.crp_names are the corpora used during training and decoding: train, cvtrain, devtrain for train and all corpora for decoding
    """

    def __init__(
        self,
        returnn_root: Optional[typing.Union[str, tk.Path]] = None,
        returnn_python_home: Optional[typing.Union[str, tk.Path]] = None,
        returnn_python_exe: Optional[
            typing.Union[str, tk.Path]
        ] = None,  # tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
        rasr_binary_path: Optional[typing.Union[str, tk.Path]] = tk.Path(
            ("/").join([gs.RASR_ROOT, "arch", "linux-x86_64-standard"])
        ),
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

        # useful paths
        self.dependency_path = (
            "/work/asr4/raissi/setups/librispeech/960-ls/dependencies"
        )

        # general modeling approach
        self.label_info = LabelInfo.default_ls()
        self.lexicon_args = get_lexicon_args()
        self.tdp_values = get_tdp_values()

        # transcription priors in pickle format
        dep_path = "/".join([self.dependency_path, "priors", "daniel"])
        self.priors = {
            "monostate": ("/").join([dep_path, "monostate", "monostate.pickle"]),
            "monostate-EOW": ("/").join([dep_path, "monostate", "monostate.we.pickle"]),
            "threepartite": ("/").join(
                [dep_path, "threepartite", "threepartite.pickle"]
            ),
            "threepartite-EOW": ("/").join(
                [dep_path, "threepartite", "threepartite.we.pickle"]
            ),
        }

        self.cv_info = {
            "pre_path": ("/").join([self.dependency_path, "data", "zhou-corpora"]),
            "train_segments": "train.segments",
            "train-dev_corpus": "train-dev.corpus.xml",
            "train-dev_segments": "train-dev.segments",
            "train-dev_lexicon_withunk": "oov.withUnkPhm.aligndev.xml.gz",
            "train-dev_lexicon_nounk": "oov.noUnkPhm.aligndev.xml.gz",
            "cv_corpus": "dev-cv.corpus.xml",
            "cv_segments": "dev-cv.segments",
            "features_postpath_cv": ("/").join(
                [
                    "FeatureExtraction.Gammatone.yly3ZlDOfaUm",
                    "output",
                    "gt.cache.bundle",
                ]
            ),
            "features_tkpath_train": Path(
                "/work/asr_archive/assis/luescher/best-models/librispeech/960h_2019-04-10/FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle"
            ),
        }
        self.base_allophones = "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.boJyUrd9Bd89/output/allophones"

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
        self.initial_nn_args = {"num_input": 50}

        self.initial_train_args = {
            "time_rqmt": 168,
            "mem_rqmt": 7,
            "log_verbosity": 5,
        }

        self.experiments: typing.Dict[str, Experiment] = {}

        # inference related
        self.tf_map = {
            "triphone": "right",
            "diphone": "center",
            "context": "left",
        }  # Triphone forward model

        compile_native_op_job = returnn.CompileNativeOpJob(
            "NativeLstm2",
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            blas_lib=None,
        )
        self.tf_library = compile_native_op_job.out_op

        self.inputs = {}
        self.train_key = None  # "train-other-960"

    # ----------- pipeline construction -----------------
    def set_experiment_dict(
        self, key: str, alignment: str, context: str, postfix_name=""
    ):
        name = f"{context}-from-{alignment}"
        self.experiments[key] = {
            "name": ("-").join([name, postfix_name]),
            "train_job": None,
            "graph": {"train": None, "inference": None},
            "returnn_config": None,
            "align_job": None,
            "decode_job": {"runner": None, "args": None},
            "extra_returnn_code": {"epilog": "", "prolog": ""},
        }

    def set_crp_pairings(self):
        keys = self.corpora.keys()
        train_key = [k for k in keys if "train" in k][0]
        all_names = ["train", "cvtrain", "devtrain"]
        all_names.extend([n for n in keys if n != train_key])
        self.crp_names = dict()
        for k in all_names:
            if "train" in k:
                self.crp_names[k] = f"{train_key}.{k}"
            else:
                self.crp_names[k] = k

    def set_returnn_config_for_experiment(self, key, returnn_config):
        assert key in self.experiments.keys()
        self.experiments[key]["returnn_config"] = returnn_config
        self.experiments[key]["extra_returnn_code"][
            "prolog"
        ] = returnn_config.python_prolog
        self.experiments[key]["extra_returnn_code"][
            "epilog"
        ] = returnn_config.python_epilog

    # -------------------- Helpers --------------------
    def _add_output_alias_for_train_job(
        self,
        train_job: Union[
            returnn.ReturnnTrainingJob,
            returnn.ReturnnRasrTrainingJob,
            "ReturnnRasrTrainingBWJob",
        ],
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
        return model_job.out_checkpoints[epoch]

    def _delete_multitask_components(self, returnn_config):
        config = copy.deepcopy(returnn_config)

        for o in ["left", "right"]:
            if "linear" in config.config["network"][f"{o}-output"]["from"]:
                for i in [1, 2]:
                    for k in ["leftContext", "triphone"]:
                        del config.config["network"][f"linear{i}-{k}"]
            del config.config["network"][f"{o}-output"]

        return config

    def _delete_mlps(self, returnn_config, keys=None, source=None):
        # source is either a string or a tupe
        if source is None:
            source = ["encoder-output"]
        elif isinstance(source, tuple):
            source = list(source)
        else:
            source = [source]

        returnn_cfg = copy.deepcopy(returnn_config)
        if keys is None:
            keys = ["left", "center", "right"]
        for l in list(returnn_cfg.config["network"].keys()):
            if "linear" in l:
                del returnn_cfg.config["network"][l]
        for o in keys:
            returnn_cfg.config["network"][f"{o}-output"]["from"] = source
        return returnn_cfg

    def set_local_flf_tool(self, path=None):
        if path is None:
            path = (
                "/u/raissi/dev/rasr-dense/src/Tools/Flf/flf-tool.linux-x86_64-standard"
            )
        self.csp["base"].flf_tool_exe = path

    # --------------------- Init procedure -----------------
    def init_system(self):
        for param in [
            self.rasr_init_args,
            self.train_data,
            self.dev_data,
            self.test_data,
        ]:
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
            if "train" in corpus_key:
                if self.train_key is None:
                    self.train_key = corpus_key
                else:
                    if self.train_key != corpus_key:
                        assert (
                            False,
                            "You already set the train key to be {self.train_key}, you cannot have more than one train key",
                        )
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
        tdp_pattern = self.tdp_values["pattern"]
        if tdp_type in ["default"]:  # additional later, maybe enum or so
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

        if "train" in crp_key:
            crp.acoustic_model_config.state_tying.type = self.label_info.state_tying
        else:
            crp.acoustic_model_config.state_tying.type = (
                "no-tying-dense"  # for correct tree of dependency
            )

        if self.label_info.use_word_end_classes:
            crp.acoustic_model_config.state_tying.use_word_end_classes = (
                self.label_info.use_word_end_classes
            )
        crp.acoustic_model_config.state_tying.use_boundary_classes = (
            self.label_info.use_boundary_classes
        )
        crp.acoustic_model_config.hmm.states_per_phone = (
            self.label_info.n_states_per_phone
        )

        crp.acoustic_model_config.allophones.add_all = self.lexicon_args[
            "add_all_allophones"
        ]
        crp.acoustic_model_config.allophones.add_from_lexicon = not self.lexicon_args[
            "add_all_allophones"
        ]
        if add_base_allophones:
            crp.acoustic_model_config.allophones.add_from_file = self.base_allophones

        crp.lexicon_config.normalize_pronunciation = self.lexicon_args[
            "norm_pronunciation"
        ]

    def _update_am_setting_for_all_crps(
        self, train_tdp_type, eval_tdp_type, add_base_allophones=False
    ):
        types = {"train": train_tdp_type, "eval": eval_tdp_type}
        for t in types.keys():
            if types[t] == "heuristic":
                if self.label_info.n_states_per_phone > 1:
                    types[t] = (types[t], "threepartite")
                else:
                    types[t] = (types[t], "monostate")

        for crp_k in self.crp_names.keys():
            if "train" in crp_k:
                self._update_crp_am_setting(
                    crp_key=self.crp_names[crp_k],
                    tdp_type=types["train"],
                    add_base_allophones=add_base_allophones,
                )
            else:
                self._update_crp_am_setting(
                    crp_key=self.crp_names[crp_k],
                    tdp_type=types["eval"],
                    add_base_allophones=add_base_allophones,
                )

    # ----- data preparation for train-----------------------------------------------------
    def prepare_train_data_with_cv_from_train(self, input_key, chunk_size=1152):
        train_corpus_path = self.corpora[self.train_key].corpus_file
        total_train_num_segments = num_segments[self.train_key]
        cv_size = 3000 / total_train_num_segments

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
        nn_train_data = self.inputs[self.train_key][
            input_key
        ].as_returnn_rasr_data_input(
            shuffle_data=True,
            segment_order_sort_by_time_length=True,
            chunk_size=chunk_size,
        )
        nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
        nn_train_data_inputs = {self.crp_names["train"]: nn_train_data}

        nn_cv_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input()
        nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
        nn_cv_data_inputs = {self.crp_names["cvtrain"]: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][
            input_key
        ].as_returnn_rasr_data_input()
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names["devtrain"]: nn_devtrain_data}

        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs

    def prepare_train_data_with_separate_cv(self, input_key, chunk_size=1152):
        # for now it is only possibel by hardcoding stuff.
        train_corpus = ("/").join(
            [self.cv_info["pre_path"], self.cv_info["train-dev_corpus"]]
        )
        train_segments = ("/").join(
            [self.cv_info["pre_path"], self.cv_info["train_segments"]]
        )
        train_feature_path = self.cv_info["features_tkpath_train"]
        train_feature_flow = features.basic_cache_flow(train_feature_path)

        cv_corpus = ("/").join([self.cv_info["pre_path"], self.cv_info["cv_corpus"]])
        cv_segments = ("/").join(
            [self.cv_info["pre_path"], self.cv_info["cv_segments"]]
        )
        cv_feature_path = Path(
            ("/").join([self.cv_info["pre_path"], self.cv_info["features_postpath_cv"]])
        )
        cv_feature_flow = features.basic_cache_flow(cv_feature_path)

        devtrain_segments = text.TailJob(
            train_segments, num_lines=1000, zip_output=False
        ).out

        nn_train_data = self.inputs[self.train_key][
            input_key
        ].as_returnn_rasr_data_input(
            shuffle_data=True,
            segment_order_sort_by_time_length=True,
            chunk_size=chunk_size,
        )

        nn_train_data.update_crp_with(
            corpus_file=train_corpus, segment_path=train_segments, concurrent=1
        )
        nn_train_data.feature_flow = train_feature_flow
        nn_train_data.features = train_feature_path
        nn_train_data_inputs = {self.crp_names["train"]: nn_train_data}

        nn_cv_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input()
        nn_cv_data.update_crp_with(
            corpus_file=cv_corpus, segment_path=cv_segments, concurrent=1
        )
        nn_cv_data.feature_flow = cv_feature_flow
        nn_cv_data.features = cv_feature_path
        nn_cv_data_inputs = {self.crp_names["cvtrain"]: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][
            input_key
        ].as_returnn_rasr_data_input()
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names["devtrain"]: nn_devtrain_data}

        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs

    def get_bw_crp_from_train_crp(self, train_crp_key):
        crp_bw = copy.deepcopy(self.crp[train_crp_key])

        train_dev_segments = ("/").join(
            [self.cv_info["pre_path"], self.cv_info[f"train-dev_segments"]]
        )
        lexicon_subname = "with" if self.label_info.add_unknown_phoneme else "no"
        train_dev_lexicon = ("/").join(
            [
                self.cv_info["pre_path"],
                self.cv_info[f"train-dev_lexicon_{lexicon_subname}unk"],
            ]
        )

        crp_bw.corpus_config.segment.file = train_dev_segments
        crp_bw.lexicon_config.file = train_dev_lexicon

        return crp_bw

    def set_rasr_returnn_input_datas(
        self, input_key, chunk_size=1152, is_cv_separate_from_train=False
    ):
        for k in self.corpora.keys():
            assert self.inputs[k] is not None
            assert self.inputs[k][input_key] is not None

        if is_cv_separate_from_train:
            f = self.prepare_train_data_with_separate_cv
        else:
            f = self.prepare_train_data_with_cv_from_train

        nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs = f(
            input_key, chunk_size
        )

        nn_dev_data_inputs = {
            self.crp_names["dev-clean"]: self.inputs["dev-clean"][
                input_key
            ].as_returnn_rasr_data_input(),
            self.crp_names["dev-other"]: self.inputs["dev-other"][
                input_key
            ].as_returnn_rasr_data_input(),
        }
        nn_test_data_inputs = {
            self.crp_names["test-clean"]: self.inputs["test-clean"][
                input_key
            ].as_returnn_rasr_data_input(),
            self.crp_names["test-other"]: self.inputs["test-other"][
                input_key
            ].as_returnn_rasr_data_input(),
        }

        self.init_datasets(
            train_data=nn_train_data_inputs,
            cv_data=nn_cv_data_inputs,
            devtrain_data=nn_devtrain_data_inputs,
            dev_data=nn_dev_data_inputs,
            test_data=nn_test_data_inputs,
        )
        label_info_args = {
            "n_states_per_phone": self.label_info.n_states_per_phone,
            "n_contexts": self.label_info.n_contexts,
        }
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

        if "returnn_config" not in nn_train_args:
            returnn_config = self.experiments[experiment_key]["returnn_config"]
        else:
            returnn_config = nn_train_args.pop("returnn_config")
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
            name=self.experiments[experiment_key]["name"],
        )
        self.experiments[experiment_key]["train_job"] = train_job

    def returnn_rasr_training_fullsum(
        self,
        experiment_key,
        train_corpus_key,
        dev_corpus_key,
        nn_train_args,
    ):
        raise NotImplementedError("what is BW training?")

        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[dev_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        if "returnn_config" not in nn_train_args:
            returnn_config = self.experiments[experiment_key]["returnn_config"]
        else:
            returnn_config = nn_train_args.pop("returnn_config")
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = ReturnnRasrTrainingBWJob(
            train_crp=train_crp,
            dev_crp=dev_crp,
            feature_flows={
                "train": train_data.feature_flow,
                "dev": dev_data.feature_flow,
            },
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
        )

        self._add_output_alias_for_train_job(
            train_job=train_job,
            name=self.experiments[experiment_key]["name"],
        )

        self.experiments[experiment_key]["train_job"] = train_job

    # ---------------------Prior Estimation--------------
    def get_hdf_path(self, hdf_key: typing.Optional[str]):
        if hdf_key is not None:
            assert hdf_key in self.hdfs.keys()
            return self.hdfs[hdf_key]

        if self.train_key not in self.hdfs.keys():
            self.create_hdf()

        return self.hdfs[self.train_key]

    def create_hdf(self):
        gammatone_features_paths: MultiPath = self.feature_caches[self.train_key]["gt"]
        hdf_job = SprintFeatureToHdf(
            feature_caches=gammatone_features_paths,
        )

        self.hdfs[self.train_key] = hdf_job.out_hdf_files

        hdf_job.add_alias(f"hdf/{self.train_key}")
        tk.register_output(f"hdf/{self.train_key}.hdf.1", self.hdfs[self.train_key][0])

    def set_mono_priors(
        self,
        key: str,
        epoch: int,
        tf_library="self",
        tm=None,
        n_state_classes: typing.Optional[int] = None,
        hdf_key="960",
        gpu=0,
    ):
        if n_state_classes is None:
            n_state_classes = self.label_info.get_n_state_classes()
        if tm is None:
            tm = self.tf_map
        if tf_library == "self":
            tf_library = self.tf_library

        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}-epoch-{epoch}"
        model_checkpoint = self._get_model_checkpoint(
            self.experiments[key]["train_job"], epoch
        )
        graph = self.experiments[key]["graph"]["inference"]

        hdf_paths = self.get_hdf_path(hdf_key)
        estimate_job = EstimateMonophonePriorsJob(
            graph=graph,
            model=model_checkpoint,
            data_paths=hdf_paths,
            dataset_indices=random.Random(PRIOR_RNG_SEED).sample(
                list(range(len(hdf_paths))), k=len(hdf_paths) // 3
            ),
            native_lstm_path=tf_library,
            num_states=n_state_classes,
            tensor_map=tm,
            gpu=gpu,
        )
        estimate_job.add_alias(f"priors/{name}")

        xml_job = DumpXmlForMonophoneJob(
            estimate_job.prior_files,
            estimate_job.num_segments,
            num_states=n_state_classes,
        )

        if name is not None:
            xml_name = f"priors/{name}/center-state.xml"
        else:
            xml_name = "mono-prior"

        tk.register_output(xml_name, xml_job.center_phoneme_xml)
        self.experiments[key]["priors"] = PriorInfo(
            center_state_prior=PriorConfig(file=xml_job.center_phoneme_xml, scale=0.0),
            left_context_prior=None,
            right_context_prior=None,
        )

    def set_diphone_priors(
        self,
        key,
        epoch,
        tf_library: typing.Optional[typing.Union[tk.Path, str]] = None,
        label_info: typing.Optional[LabelInfo] = None,
        gpu=0,
        is_sil_mapped=True,
        hdf_key=None,
    ):
        assert self.label_info.sil_id is not None

        if label_info is None:
            label_info = self.label_info
        if tf_library == "self":
            tf_library = self.tf_library

        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}-epoch-{epoch}"
        model_checkpoint = self._get_model_checkpoint(
            self.experiments[key]["train_job"], epoch
        )
        graph = self.experiments[key]["graph"]["inference"]

        hdf_paths = self.get_hdf_path(hdf_key)
        estimate_job = EstimateDiphonePriorsJob(
            graph=graph,
            model=model_checkpoint,
            data_paths=hdf_paths,
            dataset_indices=random.Random(PRIOR_RNG_SEED).sample(
                list(range(len(hdf_paths))), k=len(hdf_paths) // 3
            ),
            native_lstm_path=tf_library,
            label_info=label_info,
            tensor_map=self.tf_map,
            gpu=gpu,
        )
        estimate_job.add_alias(f"priors/{name}")

        xml_job = DumpXmlForDiphoneJob(
            estimate_job.diphone_files,
            estimate_job.context_files,
            estimate_job.num_segments,
            nContexts=label_info.n_contexts,
            nStateClasses=label_info.get_n_state_classes(),
            adjustSilence=is_sil_mapped,
            silBoundaryIndices=[0, self.label_info.sil_id],
        )

        results = [
            ("center-state", xml_job.diphoneXml),
            ("left-context", xml_job.contextXml),
        ]
        for context_name, file in results:
            xml_name = (
                f"priors/{name}/{context_name}.xml"
                if name is not None
                else f"priors/{context_name}.xml"
            )
            tk.register_output(xml_name, file)

        self.experiments[key]["priors"] = PriorInfo(
            center_state_prior=PriorConfig(file=xml_job.diphoneXml, scale=0.0),
            left_context_prior=PriorConfig(file=xml_job.contextXml, scale=0.0),
            right_context_prior=None,
        )

    def set_triphone_priors(
        self,
        key,
        epoch,
        tf_library: typing.Optional[typing.Union[tk.Path, str]] = None,
        label_info: typing.Optional[LabelInfo] = None,
        gpu=1,
        num_batch_splits=3,
        is_sil_mapped=True,
        hdf_key=None,
    ):
        assert self.label_info.sil_id is not None

        if label_info is None:
            label_info = self.label_info
        if tf_library == "self":
            tf_library = self.tf_library

        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}-epoch-{epoch}"
        model_checkpoint = self._get_model_checkpoint(
            self.experiments[key]["train_job"], epoch
        )
        graph = self.experiments[key]["graph"]["inference"]

        hdf_paths = self.get_hdf_path(hdf_key)
        estimate_job = EstimateTriphoneForwardPriorsJob(
            graph=graph,
            model=model_checkpoint,
            data_paths=hdf_paths,
            dataset_indices=random.Random(PRIOR_RNG_SEED).sample(
                list(range(len(hdf_paths))), k=len(hdf_paths) // 3
            ),
            native_lstm_path=tf_library,
            label_info=label_info,
            tensor_map=self.tf_map,
            gpu=gpu,
            num_splits=num_batch_splits,
        )
        estimate_job.add_alias(f"priors/{name}")

        xml_job = DumpXmlForTriphoneJob(
            estimate_job.right_context_files,
            estimate_job.center_state_files,
            estimate_job.left_context_files,
            estimate_job.num_segments,
            label_info=label_info,
            adjust_silence=is_sil_mapped,
            sil_boundary_indices=[0, self.label_info.sil_id],
        )

        results = [
            ("right-context", xml_job.out_right_context_xml),
            ("center-state", xml_job.out_center_state_xml),
            ("left-context", xml_job.out_left_context_xml),
        ]
        for context_name, file in results:
            xml_name = (
                f"priors/{name}/{context_name}.xml"
                if name is not None
                else f"priors/{context_name}.xml"
            )
            tk.register_output(xml_name, file)

        self.experiments[key]["priors"] = PriorInfo(
            center_state_prior=PriorConfig(
                file=xml_job.out_center_state_xml, scale=0.0
            ),
            left_context_prior=PriorConfig(
                file=xml_job.out_left_context_xml, scale=0.0
            ),
            right_context_prior=PriorConfig(
                file=xml_job.out_right_context_xml, scale=0.0
            ),
        )

    # -------------------- Decoding --------------------
    def set_graph_for_experiment(self, key):
        config = copy.deepcopy(self.experiments[key]["returnn_config"])
        name = self.experiments[key]["name"]
        python_prolog = self.experiments[key]["extra_returnn_code"]["prolog"]
        python_epilog = self.experiments[key]["extra_returnn_code"]["epilog"]

        if "source" in config.config["network"].keys():  # specaugment
            for v in config.config["network"].values():
                if v["from"] == "source":
                    v["from"] = "data"
                elif isinstance(v["from"], list):
                    v["from"] = [
                        "data" if val == "source" else val for val in v["from"]
                    ]
            del config.config["network"]["source"]

        config.config["extern_data"]["data"]["same_dim_tags_as"] = {
            "T": returnn.CodeWrapper("__time_tag__")
        }

        infer_graph = compile_tf_graph_from_returnn_config(
            config, python_prolog=python_prolog, python_epilog=python_epilog, returnn_root=self.returnn_root
        )

        self.experiments[key]["graph"]["inference"] = infer_graph
        tk.register_output(f"graphs/{name}-infer.pb", infer_graph)

    def get_recognizer_and_args(
        self,
        key: str,
        context_type: PhoneticContext,
        epoch: int,
        crp_corpus: str,
        gpu=True,
        is_multi_encoder_output=False,
        tf_library=None,
        dummy_mixtures=None,
        **decoder_kwargs,
    ):
        if context_type in [
            PhoneticContext.mono_state_transition,
            PhoneticContext.diphone_state_transition,
            PhoneticContext.tri_state_transition,
        ]:
            name = f'{self.experiments[key]["name"]}-delta-e{epoch}-'
        else:
            name = f"{self.experiments[key]['name']}-{crp_corpus}-e{epoch}"

        graph = self.experiments[key]["graph"].get("inference", None)
        assert graph is not None, "set graph first"

        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"

        recog_args = SearchParameters.default_for_ctx(context_type, priors=p_info)

        if dummy_mixtures is None:
            dummy_mixtures = mm.CreateDummyMixturesJob(
                self.label_info.get_n_of_dense_classes(),
                self.initial_nn_args["num_input"],
            ).out_mixtures  # gammatones

        assert self.label_info.sil_id is not None

        model_path = self._get_model_path(self.experiments[key]["train_job"], epoch)
        recognizer = FHDecoder(
            name=name,
            search_crp=self.crp[crp_corpus],
            context_type=context_type,
            feature_path=self.feature_flows[crp_corpus],
            model_path=model_path,
            graph=graph,
            mixtures=dummy_mixtures,
            eval_files=self.scorer_args[crp_corpus],
            tf_library=tf_library,
            is_multi_encoder_output=is_multi_encoder_output,
            silence_id=self.label_info.sil_id,
            gpu=gpu,
            **decoder_kwargs,
        )

        return recognizer, recog_args

    def recog_optimize_prior_tdp_scales(
        self,
        *,
        key: str,
        context_type: PhoneticContext,
        crp_corpus: str,
        epoch: int,
        num_encoder_output: int,
        prior_scales: typing.Union[
            typing.List[typing.Tuple[float]],  # center
            typing.List[typing.Tuple[float, float]],  # center, left
            typing.List[typing.Tuple[float, float, float]],  # center, left, right
        ],
        tdp_scales: typing.List[float],
        altas_value=14.0,
        altas_beam=14.0,
        altas_pron_scale: typing.Optional[float] = None,
        altas_lm_scale: typing.Optional[float] = None,
        gpu=False,
        is_min_duration=False,
        is_multi_encoder_output=False,
        tf_library=None,
        dummy_mixtures=None,
        **decoder_kwargs,
    ) -> DelayedBase:
        assert len(prior_scales) > 0
        assert len(tdp_scales) > 0

        recognizer, recog_args = self.get_recognizer_and_args(
            key=key,
            context_type=context_type,
            crp_corpus=crp_corpus,
            epoch=epoch,
            gpu=gpu,
            is_multi_encoder_output=is_multi_encoder_output,
            tf_library=tf_library,
            dummy_mixtures=dummy_mixtures,
            **decoder_kwargs,
        )

        recog_args.altas = altas_value
        recog_args.beam = altas_beam
        if altas_pron_scale is not None:
            recog_args.pron_scale = altas_pron_scale
        if altas_lm_scale is not None:
            recog_args.lm_scale = altas_lm_scale

        prior_scales = [tuple(round(p, 2) for p in priors) for priors in prior_scales]
        prior_scales = [
            (p, 0.0, 0.0)
            if isinstance(p, float)
            else (p[0], 0.0, 0.0)
            if len(p) == 1
            else (p[0], p[1], 0.0)
            if len(p) == 2
            else p
            for p in prior_scales
        ]
        tdp_scales = [round(s, 2) for s in tdp_scales]

        jobs = {
            ((c, l, r), tdp): recognizer.recognize_count_lm(
                label_info=self.label_info,
                num_encoder_output=num_encoder_output,
                is_min_duration=is_min_duration,
                search_parameters=recog_args.with_tdp_scale(tdp).with_prior_scale(
                    left=l, center=c, right=r
                ),
                name_override=f"{self.experiments[key]['name']}-pC{c}-pL{l}-pR{r}-tdp{tdp}",
                add_sis_alias_and_output=False,
                opt_lm_am=False,
            )
            for ((c, l, r), tdp) in itertools.product(prior_scales, tdp_scales)
        }

        for ((c, l, r), tdp), recog_jobs in jobs.items():
            pre_name = f"scales/{self.experiments[key]['name']}-e{epoch}/Lm{recog_args.lm_scale}-Pron{recog_args.pron_scale}-pC{c}-pL{l}-pR{r}-tdp{tdp}"

            recog_jobs.search.add_alias(pre_name)
            tk.register_output(f"{pre_name}.wer", recog_jobs.sclite.out_wer)

        jobs_with_fallback = {
            k: DelayedFallback(v.sclite.out_wer, math.inf) for k, v in jobs.items()
        }
        return DelayedArgMin(jobs_with_fallback)

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
                step_args["gt"]["prefix"] = "features/"
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
