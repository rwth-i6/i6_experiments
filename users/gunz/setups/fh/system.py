__all__ = ["FactoredHybridSystem"]

import copy
import dataclasses
import typing
from typing import Dict, List, Optional, Union

# -------------------- Sisyphus --------------------
from sisyphus import delayed_ops, tk

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
import i6_core.features as features
import i6_core.lexicon as lexicon
import i6_core.mm as mm
import i6_core.meta as meta
import i6_core.rasr as rasr
import i6_core.recognition as recognition
import i6_core.returnn as returnn
import i6_core.text as text
from i6_core.util import MultiPath, MultiOutputPath


from i6_experiments.common.datasets.librispeech.constants import durations, num_segments
from i6_experiments.common.setups.rasr.config.am_config import Tdp
from i6_experiments.common.setups.rasr.config.lm_config import TfRnnLmRasrConfig
from i6_experiments.common.setups.rasr.hybrid_decoder import HybridDecoder
from i6_experiments.common.setups.rasr.nn_system import NnSystem
from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
    ReturnnRasrDataInput,
    RasrSteps,
)
from i6_experiments.common.setups.rasr.util.decode import (
    AdvTreeSearchJobArgs,
    DevRecognitionParameters,
    Lattice2CtmArgs,
    OptimizeJobArgs,
    PriorPath,
)

from ..common.decoder.rtf import ExtractSearchStatisticsJob
from ..common.hdf import RasrAlignmentToHDF, RasrFeaturesToHdf
from ..common.nn.cache_epilog import hdf_dataset_cache_epilog, hdf_dataset_cache_epilog_v0
from ..common.nn.compile_graph import compile_tf_graph_from_returnn_config
from ..common.tdp import TDP
from .decoder.config import PriorConfig, PriorInfo, SearchParameters
from .decoder.search import FHDecoder
from .factored import PhoneticContext, LabelInfo
from .priors import (
    get_returnn_config_for_center_state_prior_estimation,
    get_returnn_config_for_left_context_prior_estimation,
    get_returnn_configs_for_right_context_prior_estimation,
    smoothen_priors,
    JoinRightContextPriorsJob,
    ReshapeCenterStatePriorsJob,
)
from .util.argmin import ComputeArgminJob
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
    alignment_job: typing.Optional[mm.AlignmentJob]
    name: str
    graph: Graphs
    priors: typing.Optional[PriorInfo]
    prior_job: typing.Optional[returnn.ReturnnRasrComputePriorJobV2]
    returnn_config: typing.Optional[returnn.ReturnnConfig]
    train_job: typing.Optional[returnn.ReturnnRasrTrainingJob]


@dataclasses.dataclass(frozen=True)
class TuningResult:
    best_config: delayed_ops.Delayed
    decoder: HybridDecoder


def to_tdp(tdp_tuple: typing.Tuple[TDP, TDP, TDP, TDP]) -> Tdp:
    return Tdp(loop=tdp_tuple[0], forward=tdp_tuple[1], skip=tdp_tuple[2], exit=tdp_tuple[3])


class FactoredHybridSystem(NnSystem):
    """
    self.crp_names are the corpora used during training and decoding: train, cvtrain, devtrain for train and all corpora for decoding
    """

    def __init__(
        self,
        returnn_root: typing.Union[str, tk.Path],
        returnn_python_exe: typing.Union[str, tk.Path],
        rasr_binary_path: typing.Union[str, tk.Path],
        returnn_python_home: typing.Optional[Union[str, tk.Path]] = None,
        rasr_init_args: RasrInitArgs = None,
        train_data: Dict[str, RasrDataInput] = None,
        dev_data: Dict[str, RasrDataInput] = None,
        test_data: Dict[str, RasrDataInput] = None,
        blas_lib: Optional[Path] = None,
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

        self.lm_gc_simple_hash = False

        self.filter_segments: typing.Union[Path, str, typing.List[str]] = []

        # useful paths
        self.dependency_path = "/work/asr4/raissi/setups/librispeech/960-ls/dependencies"

        # general modeling approach
        self.label_info = LabelInfo.default_ls()
        self.lexicon_args = get_lexicon_args()
        self.tdp_values = get_tdp_values()

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
                "/work/asr_archive/assis/luescher/best-models/librispeech/960h_2019-04-10/FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle",
                cached=True,
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
            "cpu_rqmt": 2,
            "time_rqmt": 168,
            "mem_rqmt": 7,
            "log_verbosity": 3,
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
            blas_lib=blas_lib,
        )
        compile_native_op_job.rqmt = {"cpu": 1, "mem": 4, "time": 0.5}

        self.native_lstm2_job = compile_native_op_job

        self.inputs = {}
        self.train_key = None  # "train-other-960"

        self.do_not_set_returnn_python_exe_for_graph_compiles = False
        self.cv_num_segments = 3000

    # ----------- pipeline construction -----------------
    def set_experiment_dict(self, key: str, alignment: str, context: str, postfix_name=""):
        name = f"{context}-from-{alignment}"
        self.experiments[key] = {
            "name": ("-").join([name, postfix_name]),
            "train_job": None,
            "graph": {"train": None, "inference": None},
            "returnn_config": None,
            "align_job": None,
            "decode_job": {"runner": None, "args": None},
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

    def set_returnn_config_for_experiment(self, key: str, returnn_config: returnn.ReturnnConfig):
        assert key in self.experiments.keys()
        self.experiments[key]["returnn_config"] = returnn_config

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
        tk.register_output(f"train/{name}_scores.png", train_job.out_plot_se)

    def _compute_returnn_rasr_priors(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: returnn.ReturnnConfig,
        share: float,
        time_rqmt: typing.Union[int, float] = 12,
        checkpoint: typing.Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        model_checkpoint = (
            checkpoint
            if checkpoint is not None
            else self._get_model_checkpoint(self.experiments[key]["train_job"], epoch)
        )

        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[dev_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        if share != 1.0:
            train_crp = copy.deepcopy(train_crp)
            segment_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
                segment_file=train_crp.segment_path,
                split={"priors": share, "rest": 1 - share},
                shuffle=True,
            )
            train_crp.segment_path = segment_job.out_segments["priors"]

        # assert train_data.feature_flow == dev_data.feature_flow
        # assert train_data.features == dev_data.features
        # assert train_data.alignments == dev_data.alignments

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

        assert isinstance(returnn_config, returnn.ReturnnConfig)

        prior_job = returnn.ReturnnRasrComputePriorJobV2(
            train_crp=train_crp,
            dev_crp=dev_crp,
            model_checkpoint=model_checkpoint,
            feature_flow=feature_flow,
            alignment=None,
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            mem_rqmt=6,
            time_rqmt=time_rqmt,
        )

        return prior_job

    def _compute_returnn_rasr_priors_via_hdf(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: returnn.ReturnnConfig,
        share: float,
        time_rqmt: typing.Union[int, float] = 12,
        checkpoint: typing.Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        model_checkpoint = (
            checkpoint
            if checkpoint is not None
            else self._get_model_checkpoint(self.experiments[key]["train_job"], epoch)
        )
        returnn_config = self.get_hdf_config_from_returnn_rasr_data(
            alignment_allophones=None,
            dev_corpus_key=dev_corpus_key,
            include_alignment=False,
            laplace_ordering=False,
            num_tied_classes=None,
            partition_epochs={"train": 1, "dev": 1},
            returnn_config=returnn_config,
            train_corpus_key=train_corpus_key,
        )

        if share != 1.0:
            segment_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
                segment_file=returnn_config.config["train"]["seq_list_file"],
                split={"priors": share, "rest": 1 - share},
                shuffle=True,
            )
            returnn_config.config["train"]["seq_list_file"] = segment_job.out_segments["priors"]

        prior_job = returnn.ReturnnComputePriorJobV2(
            model_checkpoint=model_checkpoint,
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            mem_rqmt=12,
            time_rqmt=time_rqmt,
        )

        return prior_job

    def _get_model_checkpoint(self, model_job, epoch):
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
            path = "/u/raissi/dev/rasr-dense/src/Tools/Flf/flf-tool.linux-x86_64-standard"
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
        devtrain_data: Optional[Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]]] = None,
        dev_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        test_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
    ):
        devtrain_data = devtrain_data if devtrain_data is not None else {}
        dev_data = dev_data if dev_data is not None else {}
        test_data = test_data if test_data is not None else {}

        self._assert_corpus_name_unique(train_data, cv_data, devtrain_data, dev_data, test_data)

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

        return sys_in

    def run_input_step(self, step_args):
        for corpus_key, corpus_type in step_args.corpus_type_mapping.items():
            if corpus_key not in self.train_corpora + self.dev_corpora + self.test_corpora:
                continue
            if "train" in corpus_key:
                if self.train_key is None:
                    self.train_key = corpus_key
                else:
                    if self.train_key != corpus_key:
                        assert (
                            False
                        ), "You already set the train key to be {self.train_key}, you cannot have more than one train key"
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
            assert False, f"unimplemented tdp type {tdp_type}"

        crp = self.crp[crp_key]
        for ind, ele in enumerate(tdp_pattern):
            for type in ["*", "silence"]:
                crp.acoustic_model_config["tdp"][type][ele] = tdp_values[type][ind]

        if "train" in crp_key:
            crp.acoustic_model_config.state_tying.type = self.label_info.state_tying
        else:
            # Previously set to: "no-tying-dense"  # for correct tree of dependency
            #
            # Now using str(self.label_info.state_tying) for hash compatibility, otherwise same behavior
            crp.acoustic_model_config.state_tying.type = str(self.label_info.state_tying)

        if self.label_info.phoneme_state_classes.use_word_end():
            crp.acoustic_model_config.state_tying.use_word_end_classes = (
                self.label_info.phoneme_state_classes.use_word_end()
            )
        crp.acoustic_model_config.state_tying.use_boundary_classes = (
            self.label_info.phoneme_state_classes.use_boundary()
        )
        crp.acoustic_model_config.hmm.states_per_phone = self.label_info.n_states_per_phone

        crp.acoustic_model_config.allophones.add_all = self.lexicon_args["add_all_allophones"]
        crp.acoustic_model_config.allophones.add_from_lexicon = not self.lexicon_args["add_all_allophones"]
        if add_base_allophones:
            crp.acoustic_model_config.allophones.add_from_file = self.base_allophones

        crp.lexicon_config.normalize_pronunciation = self.lexicon_args["norm_pronunciation"]

    def _update_am_setting_for_all_crps(self, train_tdp_type, eval_tdp_type, add_base_allophones=False):
        types = {"train": train_tdp_type, "eval": eval_tdp_type}
        for t in types.keys():
            if types[t].startswith("heuristic"):
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
    def prepare_train_data_with_cv_from_train(self, input_key):
        train_corpus_path = self.corpora[self.train_key].corpus_file
        total_train_num_segments = num_segments[self.train_key]
        cv_size = (self.cv_num_segments if self.cv_num_segments is not None else 3000) / total_train_num_segments

        segment_job = corpus_recipe.SegmentCorpusJob(train_corpus_path, 1)
        all_segments = segment_job.out_single_segment_files[1]

        if self.filter_segments:
            all_segments = corpus_recipe.FilterSegmentsByListJob(
                segment_job.out_single_segment_files, self.filter_segments
            ).out_single_segment_files[1]

        splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
            all_segments, {"train": 1 - cv_size, "cv": cv_size}
        )
        train_segments = splitted_segments_job.out_segments["train"]
        cv_segments = splitted_segments_job.out_segments["cv"]
        devtrain_segments = text.TailJob(train_segments, num_lines=1000, zip_output=False).out

        # ******************** NN Init ********************
        nn_train_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            shuffle_data=True,
            segment_order_sort_by_time_length=True,
        )
        nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
        nn_train_data_inputs = {self.crp_names["train"]: nn_train_data}

        nn_cv_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            shuffle_data=True,
            segment_order_sort_by_time_length=True,
        )
        nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
        nn_cv_data_inputs = {self.crp_names["cvtrain"]: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            shuffle_data=True,
            segment_order_sort_by_time_length=True,
        )
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names["devtrain"]: nn_devtrain_data}

        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs

    def prepare_train_data_with_separate_cv(self, input_key):
        # Example CV info
        #
        # self.cv_info = {
        #     "pre_path": ("/").join(
        #         [
        #             "/work/asr4/raissi/setups/librispeech/960-ls/dependencies",
        #             "data",
        #             "zhou-corpora",
        #         ]
        #     ),
        #     "train_segments": "train.segments",
        #     "train-dev_corpus": "train-dev.corpus.xml",
        #     "train-dev_segments": "train-dev.segments",
        #     "train-dev_lexicon_withunk": "oov.withUnkPhm.aligndev.xml.gz",
        #     "train-dev_lexicon_nounk": "oov.noUnkPhm.aligndev.xml.gz",
        #     "cv_corpus": "dev-cv.corpus.xml",
        #     "cv_segments": "dev-cv.segments",
        #     "features_postpath_cv": ("/").join(
        #         [
        #             "FeatureExtraction.Gammatone.yly3ZlDOfaUm",
        #             "output",
        #             "gt.cache.bundle",
        #         ]
        #     ),
        #     "features_tkpath_train": Path(
        #         "/work/asr_archive/assis/luescher/best-models/librispeech/960h_2019-04-10/FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.bundle"
        #     ),
        # }

        # for now it is only possible by hardcoding stuff.
        train_corpus = self.cv_info["train-dev_corpus"]
        train_segments = tk.Path(self.cv_info["train_segments"], cached=True)
        train_feature_path = tk.Path(self.cv_info["features_tkpath_train"], cached=True)
        train_feature_flow = features.basic_cache_flow(train_feature_path)

        cv_corpus = self.cv_info["cv_corpus"]
        cv_segments = self.cv_info["cv_segments"]
        cv_feature_path = Path(self.cv_info["features_postpath_cv"], cached=True)
        cv_feature_flow = features.basic_cache_flow(cv_feature_path)

        devtrain_segments = text.TailJob(train_segments, num_lines=1000, zip_output=False).out

        nn_train_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            shuffle_data=True,
            segment_order_sort_by_time_length=True,
        )

        nn_train_data.update_crp_with(corpus_file=train_corpus, segment_path=train_segments, concurrent=1)
        nn_train_data.feature_flow = train_feature_flow
        nn_train_data.features = train_feature_path
        nn_train_data_inputs = {self.crp_names["train"]: nn_train_data}

        nn_cv_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            shuffle_data=True,
            segment_order_sort_by_time_length=True,
        )
        nn_cv_data.update_crp_with(corpus_file=cv_corpus, segment_path=cv_segments, concurrent=1)
        nn_cv_data.feature_flow = cv_feature_flow
        nn_cv_data.features = cv_feature_path
        nn_cv_data_inputs = {self.crp_names["cvtrain"]: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            shuffle_data=True,
            segment_order_sort_by_time_length=True,
        )
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names["devtrain"]: nn_devtrain_data}

        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs

    def get_bw_crp_from_train_crp(self, train_crp_key):
        crp_bw = copy.deepcopy(self.crp[train_crp_key])

        train_dev_segments = ("/").join([self.cv_info["pre_path"], self.cv_info[f"train-dev_segments"]])
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

    def set_rasr_returnn_input_datas(self, input_key, is_cv_separate_from_train=False, **kwargs):
        for k in self.corpora.keys():
            assert self.inputs[k] is not None
            assert self.inputs[k][input_key] is not None

        if is_cv_separate_from_train:
            f = self.prepare_train_data_with_separate_cv
        else:
            f = self.prepare_train_data_with_cv_from_train

        nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs = f(input_key)

        nn_dev_data_inputs = {
            self.crp_names["dev-clean"]: self.inputs["dev-clean"][input_key].as_returnn_rasr_data_input(),
            self.crp_names["dev-other"]: self.inputs["dev-other"][input_key].as_returnn_rasr_data_input(),
        }
        nn_test_data_inputs = {
            self.crp_names["test-clean"]: self.inputs["test-clean"][input_key].as_returnn_rasr_data_input(),
            self.crp_names["test-other"]: self.inputs["test-other"][input_key].as_returnn_rasr_data_input(),
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

    def get_hdf_config_from_returnn_rasr_data(
        self,
        *,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: returnn.ReturnnConfig,
        partition_epochs: typing.Dict[str, int],
        alignment_allophones: typing.Optional[tk.Path] = None,
        num_tied_classes: typing.Optional[int] = None,
        include_alignment: bool = True,
        laplace_ordering: bool = True,
    ):
        returnn_config = copy.deepcopy(returnn_config)

        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[dev_corpus_key]
        train_crp = copy.deepcopy(train_data.get_crp())
        dev_crp = copy.deepcopy(dev_data.get_crp())

        train_crp.acoustic_model_config.allophones.add_all = True
        train_hdf_job = RasrFeaturesToHdf(train_data.features)

        dataset_cfg = {
            "class": "MetaDataset",
            "data_map": {"data": ("audio", "features")},
            "datasets": {
                "audio": {
                    "class": "NextGenHDFDataset",
                    "input_stream_name": "features",
                    "files": train_hdf_job.out_hdf_files,
                },
            },
            "seq_ordering": f"random:{PRIOR_RNG_SEED}",
        }

        if include_alignment:
            allophone_job = lexicon.StoreAllophonesJob(train_crp)
            tying_job = lexicon.DumpStateTyingJob(train_crp)
            alignment_hdf_job = RasrAlignmentToHDF(
                alignment_bundle=train_data.alignments,
                allophones=alignment_allophones
                if alignment_allophones is not None
                else allophone_job.out_allophone_file,
                num_tied_classes=self.label_info.get_n_of_dense_classes()
                if num_tied_classes is None
                else num_tied_classes,
                state_tying=tying_job.out_state_tying,
            )
            dataset_cfg["data_map"] = {**dataset_cfg["data_map"], "classes": ("alignment", "classes")}
            dataset_cfg["datasets"] = {
                **dataset_cfg["datasets"],
                "alignment": {
                    "class": "NextGenHDFDataset",
                    "input_stream_name": "classes",
                    "files": [alignment_hdf_job.out_hdf_file],
                },
            }

        if laplace_ordering:
            assert (
                include_alignment
            ), "can only order laplacian if training w/ an alignment, training goes OOM otherwise"

            dataset_cfg["seq_ordering"] = f"laplace:.384:{PRIOR_RNG_SEED}"
            dataset_cfg["seq_order_control_dataset"] = "alignment"

        dev_data = {
            **dataset_cfg,
            "partition_epoch": partition_epochs["dev"],
            "seq_list_file": dev_crp.segment_path,
        }
        train_data = {
            **dataset_cfg,
            "partition_epoch": partition_epochs["train"],
            "seq_list_file": train_crp.segment_path,
        }

        update_cfg = returnn.ReturnnConfig(
            config={"train": train_data, "dev": dev_data},
            python_epilog=hdf_dataset_cache_epilog,
        )
        returnn_config.update(update_cfg)

        return returnn_config

    def returnn_training(
        self,
        experiment_key: str,
        returnn_config: returnn.ReturnnConfig,
        nn_train_args: typing.Any,
        on_2080: bool = False,
    ):
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = returnn.ReturnnTrainingJob(
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **(nn_train_args or {}),
        )
        if on_2080:
            train_job.rqmt["qsub_args"] = "-l qname=*2080*"

        self._add_output_alias_for_train_job(
            train_job=train_job,
            name=self.experiments[experiment_key]["name"],
        )
        self.experiments[experiment_key]["train_job"] = train_job

        return train_job

    def returnn_training_from_hdf(
        self,
        experiment_key: str,
        returnn_config: returnn.ReturnnConfig,
        nn_train_args,
        train_hdfs: typing.List[tk.Path],
        dev_hdfs: typing.List[tk.Path],
        on_2080: bool = True,
        dev_data: typing.Optional[typing.Dict[str, typing.Any]] = None,
        train_data: typing.Optional[typing.Dict[str, typing.Any]] = None,
        use_old_cache_epilog: bool = False,
    ):
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        nn_train_args = copy.copy(nn_train_args)

        partition_epochs = nn_train_args.pop("partition_epochs")
        dev_data = {
            "class": "NextGenHDFDataset",
            "files": dev_hdfs,
            "input_stream_name": "features",
            "partition_epoch": partition_epochs["dev"],
            **(dev_data or {}),
        }
        train_data = {
            "class": "NextGenHDFDataset",
            "files": train_hdfs,
            "input_stream_name": "features",
            "partition_epoch": partition_epochs["train"],
            "seq_ordering": f"random:{PRIOR_RNG_SEED}",
            **(train_data or {}),
        }

        returnn_config = copy.deepcopy(returnn_config)
        update_config = returnn.ReturnnConfig(
            config={"dev": dev_data, "train": train_data},
            python_epilog=hdf_dataset_cache_epilog if not use_old_cache_epilog else hdf_dataset_cache_epilog_v0,
        )
        returnn_config.update(update_config)

        return self.returnn_training(
            experiment_key=experiment_key, returnn_config=returnn_config, on_2080=on_2080, nn_train_args=nn_train_args
        )

    def returnn_rasr_training(
        self,
        experiment_key,
        train_corpus_key,
        dev_corpus_key,
        nn_train_args,
        on_2080: bool = False,
        include_alignment: bool = True,
    ) -> returnn.ReturnnRasrTrainingJob:
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[dev_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        if "returnn_config" not in nn_train_args:
            returnn_config = self.experiments[experiment_key]["returnn_config"]
        else:
            returnn_config = nn_train_args.pop("returnn_config")
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        # These asserts are currently not relevant and wrong during from-scratch training
        # assert train_data.feature_flow == dev_data.feature_flow
        # assert train_data.features == dev_data.features
        # assert train_data.alignments == dev_data.alignments

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

        if not include_alignment:
            alignments = None
        elif isinstance(train_data.alignments, rasr.FlagDependentFlowAttribute):
            alignments = copy.deepcopy(train_data.alignments)
            net = rasr.FlowNetwork()
            net.flags = {"cache_mode": "bundle"}
            alignments = alignments.get(net)
        elif isinstance(train_data.alignments, (MultiPath, MultiOutputPath)):
            raise NotImplementedError
        elif isinstance(train_data.alignments, tk.Path):
            alignments = train_data.alignments
        else:
            raise NotImplementedError(f"cannot deal w/ alignments of type {type(train_data.alignments)}")

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

        if on_2080:
            train_job.rqmt["sbatch_args"] = ["--gres=gpu:rtx_2080"]

        self._add_output_alias_for_train_job(
            train_job=train_job,
            name=self.experiments[experiment_key]["name"],
        )
        self.experiments[experiment_key]["train_job"] = train_job

        return train_job

    def returnn_rasr_training_via_hdf(
        self,
        experiment_key: str,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: returnn.ReturnnConfig,
        partition_epochs: typing.Dict[str, int],
        alignment_allophones: typing.Optional[Path] = None,
        nn_train_args: typing.Optional[typing.Any] = None,
        num_tied_classes: typing.Optional[int] = None,
        include_alignment: bool = True,
        laplace_ordering: bool = True,
    ):
        returnn_config = self.get_hdf_config_from_returnn_rasr_data(
            train_corpus_key=train_corpus_key,
            dev_corpus_key=dev_corpus_key,
            returnn_config=returnn_config,
            partition_epochs=partition_epochs,
            alignment_allophones=alignment_allophones,
            num_tied_classes=num_tied_classes,
            include_alignment=include_alignment,
            laplace_ordering=laplace_ordering,
        )
        return self.returnn_training(
            experiment_key=experiment_key,
            returnn_config=returnn_config,
            nn_train_args={"mem_rqmt": 16, **nn_train_args},
        )

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
        hdf_job = RasrFeaturesToHdf(feature_caches=gammatone_features_paths)

        self.hdfs[self.train_key] = hdf_job.out_hdf_files

        hdf_job.add_alias(f"hdf/{self.train_key}")

        return hdf_job

    def set_mono_priors_returnn_rasr(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: typing.Optional[returnn.ReturnnConfig] = None,
        output_layer_name: str = "center-output",
        data_share: float = 1.0 / 3.0,
        smoothen: bool = False,
        via_hdf: bool = False,
        checkpoint: typing.Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}/e{epoch}"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        config = copy.deepcopy(returnn_config)
        config.config["forward_output_layer"] = output_layer_name
        config.config["network"][output_layer_name]["register_as_extern_data"] = output_layer_name

        job = (
            self._compute_returnn_rasr_priors_via_hdf(
                key=key,
                epoch=epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=config,
                share=data_share,
                checkpoint=checkpoint,
            )
            if via_hdf
            else self._compute_returnn_rasr_priors(
                key,
                epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=config,
                share=data_share,
                time_rqmt=4.9,
                checkpoint=checkpoint,
            )
        )
        job.add_alias(f"priors/{name}/c")

        p_info = PriorInfo(
            center_state_prior=PriorConfig(file=job.out_prior_xml_file, scale=0.0),
        )
        p_info = smoothen_priors(p_info) if smoothen else p_info
        self.experiments[key]["priors"] = p_info

        tk.register_output(f"priors/{name}/center-state.xml", p_info.center_state_prior.file)

        return job

    def set_diphone_priors_returnn_rasr(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: typing.Optional[returnn.ReturnnConfig] = None,
        left_context_output_layer_name: str = "left-output",
        center_state_output_layer_name: str = "center-output",
        data_share: float = 1.0 / 3.0,
        smoothen: bool = False,
        via_hdf: bool = False,
        checkpoint: typing.Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}/e{epoch}"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        left_config = get_returnn_config_for_left_context_prior_estimation(
            returnn_config,
            left_context_softmax_layer=left_context_output_layer_name,
        )
        center_config = get_returnn_config_for_center_state_prior_estimation(
            returnn_config,
            label_info=self.label_info,
            center_state_softmax_layer=center_state_output_layer_name,
        )

        prior_jobs = {
            ctx: self._compute_returnn_rasr_priors_via_hdf(
                key=key,
                epoch=epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=cfg,
                share=data_share,
                checkpoint=checkpoint,
            )
            if via_hdf
            else self._compute_returnn_rasr_priors(
                key,
                epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=cfg,
                share=data_share,
                checkpoint=checkpoint,
            )
            for (ctx, cfg) in [("l", left_config), ("c", center_config)]
        }

        for (ctx, job) in prior_jobs.items():
            job.add_alias(f"priors/{name}/{ctx}")

        center_priors = ReshapeCenterStatePriorsJob(prior_jobs["c"].out_prior_txt_file, label_info=self.label_info)
        center_priors_xml = center_priors.out_prior_xml

        p_info = PriorInfo(
            center_state_prior=PriorConfig(file=center_priors_xml, scale=0.0),
            left_context_prior=PriorConfig(file=prior_jobs["l"].out_prior_xml_file, scale=0.0),
            right_context_prior=None,
        )
        p_info = smoothen_priors(p_info) if smoothen else p_info

        results = [
            ("center-state", p_info.center_state_prior.file),
            ("left-context", p_info.left_context_prior.file),
        ]
        for context_name, file in results:
            xml_name = f"priors/{name}/{context_name}.xml" if name is not None else f"priors/{context_name}.xml"
            tk.register_output(xml_name, file)

        self.experiments[key]["priors"] = p_info

    def set_triphone_priors_returnn_rasr(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: typing.Optional[returnn.ReturnnConfig] = None,
        left_context_output_layer_name: str = "left-output",
        center_state_output_layer_name: str = "center-output",
        right_context_output_layer_name: str = "right-output",
        data_share: float = 1.0 / 3.0,
        smoothen: bool = False,
        via_hdf: bool = False,
        checkpoint: typing.Optional[returnn.Checkpoint] = None,
    ):
        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}/e{epoch}"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        left_config = get_returnn_config_for_left_context_prior_estimation(
            returnn_config,
            left_context_softmax_layer=left_context_output_layer_name,
        )
        center_config = get_returnn_config_for_center_state_prior_estimation(
            returnn_config,
            label_info=self.label_info,
            center_state_softmax_layer=center_state_output_layer_name,
        )
        right_configs = get_returnn_configs_for_right_context_prior_estimation(
            returnn_config,
            label_info=self.label_info,
            right_context_softmax_layer=right_context_output_layer_name,
        )

        prior_jobs = {
            ctx: self._compute_returnn_rasr_priors_via_hdf(
                key=key,
                epoch=epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=cfg,
                share=data_share,
                checkpoint=checkpoint,
            )
            if via_hdf
            else self._compute_returnn_rasr_priors(
                key,
                epoch,
                train_corpus_key=train_corpus_key,
                dev_corpus_key=dev_corpus_key,
                returnn_config=cfg,
                share=data_share,
                time_rqmt=8,
                checkpoint=checkpoint,
            )
            for (ctx, cfg) in (
                ("l", left_config),
                ("c", center_config),
                *((f"r{i}", cfg) for i, cfg in enumerate(right_configs)),
            )
        }

        for (ctx, job) in prior_jobs.items():
            job.add_alias(f"priors/{name}/{ctx}")

        center_priors = ReshapeCenterStatePriorsJob(prior_jobs["c"].out_prior_txt_file, label_info=self.label_info)
        center_priors_xml = center_priors.out_prior_xml

        right_priors = [prior_jobs[f"r{i}"].out_prior_txt_file for i in range(len(right_configs))]
        right_prior_xml = JoinRightContextPriorsJob(right_priors, label_info=self.label_info).out_prior_xml

        p_info = PriorInfo(
            center_state_prior=PriorConfig(file=center_priors_xml, scale=0.0),
            left_context_prior=PriorConfig(file=prior_jobs["l"].out_prior_xml_file, scale=0.0),
            right_context_prior=PriorConfig(file=right_prior_xml, scale=0.0),
        )
        p_info = smoothen_priors(p_info) if smoothen else p_info

        results = [
            ("center-state", p_info.center_state_prior.file),
            ("left-context", p_info.left_context_prior.file),
            ("right-context", p_info.right_context_prior.file),
        ]
        for context_name, file in results:
            xml_name = f"priors/{name}/{context_name}.xml" if name is not None else f"priors/{context_name}.xml"
            tk.register_output(xml_name, file)

        self.experiments[key]["priors"] = p_info

    # -------------------- Decoding --------------------
    def set_graph_for_experiment(self, key, override_cfg: typing.Optional[returnn.ReturnnConfig] = None):
        config = copy.deepcopy(override_cfg if override_cfg is not None else self.experiments[key]["returnn_config"])

        name = self.experiments[key]["name"]

        if "source" in config.config["network"].keys():  # specaugment
            for v in config.config["network"].values():
                if "from" not in v:
                    continue
                if v["from"] == "source":
                    v["from"] = "data"
                elif isinstance(v["from"], list):
                    v["from"] = ["data" if val == "source" else val for val in v["from"]]
            del config.config["network"]["source"]

        infer_graph = compile_tf_graph_from_returnn_config(
            config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe
            if not self.do_not_set_returnn_python_exe_for_graph_compiles
            else None,
            alias=f"graphs/{name}",
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
        tf_library: typing.Union[tk.Path, str, typing.List[tk.Path], typing.List[str], None] = None,
        dummy_mixtures: typing.Optional[tk.Path] = None,
        lm_gc_simple_hash: typing.Optional[bool] = None,
        crp: typing.Optional[rasr.RasrConfig] = None,
        **decoder_kwargs,
    ):
        if context_type in [
            PhoneticContext.mono_state_transition,
            PhoneticContext.diphone_state_transition,
            PhoneticContext.tri_state_transition,
        ]:
            name = f'{self.experiments[key]["name"]}-delta/e{epoch}/{crp_corpus}'
        else:
            name = f"{self.experiments[key]['name']}/e{epoch}/{crp_corpus}"

        graph = self.experiments[key]["graph"].get("inference", None)
        if graph is None:
            self.set_graph_for_experiment(key=key)
            graph = self.experiments[key]["graph"]["inference"]

        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"

        recog_args = SearchParameters.default_for_ctx(context_type, priors=p_info)

        if dummy_mixtures is None:
            dummy_mixtures = mm.CreateDummyMixturesJob(
                self.label_info.get_n_of_dense_classes(),
                self.initial_nn_args["num_input"],
            ).out_mixtures  # gammatones

        assert self.label_info.sil_id is not None

        model_path = self._get_model_checkpoint(self.experiments[key]["train_job"], epoch)
        crp_corpus_base = crp_corpus.split(".", 1)[0]
        recognizer = FHDecoder(
            name=name,
            search_crp=self.crp[crp_corpus] if crp is None else crp,
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
            corpus_duration=durations[crp_corpus_base],
            lm_gc_simple_hash=lm_gc_simple_hash
            if (lm_gc_simple_hash is not None and lm_gc_simple_hash) or self.lm_gc_simple_hash
            else None,
            **decoder_kwargs,
        )

        return recognizer, recog_args

    def get_cart_params(self, key: str):
        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"
        return SearchParameters.default_cart(priors=p_info)

    def recognize_optimize_scales_nn_pch(
        self,
        *,
        key: str,
        epoch: int,
        crp_corpus: str,
        n_out: int,
        cart_tree_or_tying_config: typing.Union[tk.Path, rasr.RasrConfig],
        params: SearchParameters,
        log_softmax_returnn_config: returnn.ReturnnConfig,
        prior_scales: List[float],
        tdp_scales: Optional[List[float]] = None,
        tdp_speech: Optional[List[typing.Tuple[TDP, TDP, TDP, TDP]]] = None,
        tdp_silence: Optional[List[typing.Tuple[TDP, TDP, TDP, TDP]]] = None,
        tune_altas: int = 12,
        tune_beam: int = 14,
        encoder_output_layer: str = "output",
        mem_rqmt: int = 4,
        cpu_rqmt: int = 2,
        alias_output_prefix: str = "",
        prior_epoch: typing.Union[int, str] = "",
    ) -> TuningResult:
        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"

        p_mixtures = mm.CreateDummyMixturesJob(n_out, self.initial_nn_args["num_input"]).out_mixtures

        crp = copy.deepcopy(self.crp[crp_corpus])

        if isinstance(cart_tree_or_tying_config, rasr.RasrConfig):
            crp.acoustic_model_config.state_tying = cart_tree_or_tying_config
        else:
            crp.acoustic_model_config.state_tying.file = cart_tree_or_tying_config
            crp.acoustic_model_config.state_tying.type = "cart"

        def SearchJob(*args, **kwargs):
            kwargs["lmgc_scorer"] = rasr.DiagonalMaximumScorer(p_mixtures)
            kwargs["separate_lm_image_gc_generation"] = True
            return recognition.AdvancedTreeSearchJob(*args, **kwargs)

        decoder = HybridDecoder(
            rasr_binary_path=self.rasr_binary_path,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            required_native_ops=None,
            search_job_class=SearchJob,
            alias_output_prefix=f"{alias_output_prefix}scales-nn-pch/",
        )
        decoder.set_crp("init", crp)

        corpus = meta.CorpusObject()
        corpus.corpus_file = crp.corpus_config.file
        corpus.audio_format = crp.audio_format
        corpus.duration = crp.corpus_duration

        decoder.init_eval_datasets(
            eval_datasets={crp_corpus: corpus},
            concurrency={crp_corpus: crp.concurrent},
            corpus_durations=durations,
            feature_flows=self.feature_flows,
            stm_paths={crp_corpus: self.scorer_args[crp_corpus]["ref"]},
        )

        @dataclasses.dataclass
        class RasrConfigWrapper:
            obj: rasr.RasrConfig

            def get(self) -> rasr.RasrConfig:
                return self.obj

        adv_search_extra_config = rasr.RasrConfig()
        adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
            tune_altas
        )

        lat2ctm_extra_config = rasr.RasrConfig()
        lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = "best"

        tdp_sil = tdp_silence if tdp_silence is not None else [params.tdp_silence]
        tdp_ssp = tdp_speech if tdp_speech is not None else [params.tdp_speech]
        tdp_sc = tdp_scales if tdp_scales is not None else [params.tdp_scale]

        decoder.recognition(
            name=self.experiments[key]["name"],
            checkpoints={epoch: self._get_model_checkpoint(self.experiments[key]["train_job"], epoch)},
            epochs=[epoch],
            forward_output_layer=encoder_output_layer,
            prior_paths={
                f"rp{prior_epoch}": PriorPath(
                    acoustic_mixture_path=p_mixtures,
                    prior_xml_path=p_info.center_state_prior.file,
                )
            },
            recognition_parameters={
                crp_corpus: [
                    DevRecognitionParameters(
                        altas=[tune_altas],
                        am_scales=[1],
                        lm_scales=[params.lm_scale],
                        prior_scales=prior_scales,
                        pronunciation_scales=[params.pron_scale],
                        speech_tdps=[to_tdp(tdp) for tdp in tdp_ssp],
                        silence_tdps=[to_tdp(tdp) for tdp in tdp_sil],
                        nonspeech_tdps=[to_tdp(tdp) for tdp in tdp_sil],
                        tdp_scales=tdp_sc,
                    )
                ]
            },
            returnn_config=log_softmax_returnn_config,
            lm_configs={"4gram": RasrConfigWrapper(obj=crp.language_model_config)},
            search_job_args=AdvTreeSearchJobArgs(
                search_parameters={
                    "beam-pruning": tune_beam,
                    "beam-pruning-limit": params.beam_limit,
                    "word-end-pruning": params.we_pruning,
                    "word-end-pruning-limit": params.we_pruning_limit,
                },
                use_gpu=False,
                mem=mem_rqmt,
                cpu=cpu_rqmt,
                lm_lookahead=True,
                lmgc_mem=12,
                lookahead_options=None,
                create_lattice=True,
                eval_best_in_lattice=True,
                eval_single_best=True,
                extra_config=adv_search_extra_config,
                extra_post_config=None,
                rtf=4,
            ),
            lat_2_ctm_args=Lattice2CtmArgs(
                parallelize=True,
                best_path_algo="bellman-ford",
                encoding="utf-8",
                extra_config=lat2ctm_extra_config,
                extra_post_config=None,
                fill_empty_segments=True,
            ),
            scorer_args=self.scorer_args[crp_corpus],
            optimize_parameters=OptimizeJobArgs(
                opt_only_lm_scale=True,
                maxiter=100,
                precision=2,
                extra_config=None,
                extra_post_config=None,
            ),
            optimize_pron_lm_scales=False,
        )

        n_errors = {
            (key, exp_name): job.out_num_errors
            for key, jobs in decoder.jobs.items()
            for exp_name, job in jobs["score"].items()
        }
        best_overall_n_err = ComputeArgminJob(n_errors)

        wer = {
            (key, exp_name): job.out_wer
            for key, jobs in decoder.jobs.items()
            for exp_name, job in jobs["score"].items()
        }
        best_overall_wer = ComputeArgminJob(wer)

        name = self.experiments[key]["name"]
        tk.register_output(f"scales-nn-pch/{name}/scales", best_overall_n_err.out_argmin)
        tk.register_output(f"scales-nn-pch/{name}/n_err", best_overall_n_err.out_min)
        tk.register_output(f"scales-nn-pch/{name}/wer", best_overall_wer.out_min)

        return TuningResult(best_config=best_overall_n_err.out_argmin, decoder=decoder)

    def recognize_cart(
        self,
        *,
        key: str,
        epoch: int,
        crp_corpus: str,
        n_cart_out: int,
        cart_tree_or_tying_config: typing.Union[tk.Path, rasr.RasrConfig],
        params: SearchParameters,
        log_softmax_returnn_config: returnn.ReturnnConfig,
        encoder_output_layer: str = "output",
        gpu: bool = False,
        mem_rqmt: int = 8,
        cpu_rqmt: int = 4,
        native_ops: typing.Optional[
            typing.List[str]
        ] = None,  # This is a list of native op names (like "NativeLstm2"), not compiled op paths
        calculate_statistics: bool = False,
        opt_lm_am_scale: bool = False,
        rtf: typing.Optional[float] = None,
        lm_gc_simple_hash: typing.Optional[bool] = None,
        parallel: typing.Optional[int] = None,
        adv_search_extra_config: typing.Optional[rasr.RasrConfig] = None,
        alias_output_prefix: str = "",
        create_lattice: bool = True,
        search_rqmt_update: Optional[dict] = None,
        crp_update: Optional[typing.Callable] = None,
        prior_epoch: typing.Union[int, str] = "",
        decode_trafo_lm: bool = False,
    ) -> recognition.AdvancedTreeSearchJob:
        p_info: PriorInfo = self.experiments[key].get("priors", None)
        assert p_info is not None, "set priors first"

        p_mixtures = mm.CreateDummyMixturesJob(n_cart_out, self.initial_nn_args["num_input"]).out_mixtures

        crp = copy.deepcopy(self.crp[crp_corpus])

        if isinstance(cart_tree_or_tying_config, rasr.RasrConfig):
            crp.acoustic_model_config.state_tying = cart_tree_or_tying_config
        else:
            crp.acoustic_model_config.state_tying.file = cart_tree_or_tying_config
            crp.acoustic_model_config.state_tying.type = "cart"

        if crp_update is not None:
            crp_update(crp)

        adv_tree_search_job: recognition.AdvancedTreeSearchJob

        def SearchJob(*args, **kwargs):
            nonlocal adv_tree_search_job

            if (lm_gc_simple_hash is not None and lm_gc_simple_hash) or self.lm_gc_simple_hash:
                kwargs["lmgc_scorer"] = rasr.DiagonalMaximumScorer(p_mixtures)
            if parallel is not None:
                kwargs["parallel"] = parallel
            kwargs["separate_lm_image_gc_generation"] = True

            adv_tree_search_job = recognition.AdvancedTreeSearchJob(*args, **kwargs)
            if search_rqmt_update is not None:
                adv_tree_search_job.rqmt.update(search_rqmt_update)
            return adv_tree_search_job

        decoder = HybridDecoder(
            rasr_binary_path=self.rasr_binary_path,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            required_native_ops=native_ops,
            search_job_class=SearchJob,
            alias_output_prefix=alias_output_prefix,
        )
        decoder.set_crp("init", crp)

        corpus = meta.CorpusObject()
        corpus.corpus_file = crp.corpus_config.file
        corpus.audio_format = crp.audio_format
        corpus.duration = crp.corpus_duration

        decoder.init_eval_datasets(
            eval_datasets={crp_corpus: corpus},
            concurrency={crp_corpus: crp.concurrent},
            corpus_durations=durations,
            feature_flows=self.feature_flows,
            stm_paths={crp_corpus: self.scorer_args[crp_corpus]["ref"]},
        )

        @dataclasses.dataclass
        class RasrConfigWrapper:
            obj: rasr.RasrConfig

            def get(self) -> rasr.RasrConfig:
                return self.obj

        if params.altas is not None:
            if adv_search_extra_config is None:
                adv_search_extra_config = rasr.RasrConfig()
            else:
                adv_search_extra_config = copy.deepcopy(adv_search_extra_config)

            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                params.altas
            )

        lat2ctm_extra_config = rasr.RasrConfig()
        lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = "best"

        lm_configs = {"4gram": RasrConfigWrapper(obj=crp.language_model_config)}

        if decode_trafo_lm:
            lm_cfg = TfRnnLmRasrConfig(
                common_prefix=True,
                meta_graph_path=Path(
                    "/work/asr3/raissi/shared_workspaces/gunz/dependencies/ls-eugen-trafo-lm/graph.meta"
                ),
                returnn_checkpoint=returnn.Checkpoint(
                    index_path=Path(
                        "/work/asr3/raissi/shared_workspaces/gunz/dependencies/ls-eugen-trafo-lm/epoch.030.index"
                    )
                ),
                scale=crp.language_model_config.scale + 2,
                softmax_adapter="quantized-blas-nce-16bit",
                state_manager="transformer-with-common-prefix-16bit",
                transform_output_log=False,
                vocab_path=Path("/work/asr3/raissi/shared_workspaces/gunz/dependencies/ls-eugen-trafo-lm/vocabulary"),
            )
            lm_configs["eugen-trafo"] = lm_cfg

        decoder.recognition(
            name=self.experiments[key]["name"],
            checkpoints={epoch: self._get_model_checkpoint(self.experiments[key]["train_job"], epoch)},
            epochs=[epoch],
            forward_output_layer=encoder_output_layer,
            prior_paths={
                f"rp{prior_epoch}": PriorPath(
                    acoustic_mixture_path=p_mixtures,
                    prior_xml_path=p_info.center_state_prior.file,
                )
            },
            recognition_parameters={
                crp_corpus: [
                    DevRecognitionParameters(
                        altas=[params.altas] if params.altas is not None else None,
                        am_scales=[1],
                        lm_scales=[params.lm_scale],
                        prior_scales=[params.prior_info.center_state_prior.scale],
                        pronunciation_scales=[params.pron_scale],
                        speech_tdps=[
                            Tdp(
                                loop=params.tdp_speech[0],
                                forward=params.tdp_speech[1],
                                skip=params.tdp_speech[2],
                                exit=params.tdp_speech[3],
                            )
                        ],
                        silence_tdps=[
                            Tdp(
                                loop=params.tdp_silence[0],
                                forward=params.tdp_silence[1],
                                skip=params.tdp_silence[2],
                                exit=params.tdp_silence[3],
                            )
                        ],
                        nonspeech_tdps=[
                            Tdp(
                                loop=params.tdp_non_word[0],
                                forward=params.tdp_non_word[1],
                                skip=params.tdp_non_word[2],
                                exit=params.tdp_non_word[3],
                            )
                        ],
                        tdp_scales=[params.tdp_scale],
                    )
                ]
            },
            returnn_config=log_softmax_returnn_config,
            lm_configs=lm_configs,
            search_job_args=AdvTreeSearchJobArgs(
                search_parameters={
                    "beam-pruning": params.beam,
                    "beam-pruning-limit": params.beam_limit,
                    "word-end-pruning": params.we_pruning,
                    "word-end-pruning-limit": params.we_pruning_limit,
                },
                use_gpu=gpu,
                mem=mem_rqmt,
                cpu=cpu_rqmt,
                lm_lookahead=True,
                lmgc_mem=12,
                lookahead_options=None,
                create_lattice=create_lattice,
                eval_best_in_lattice=True,
                eval_single_best=True,
                extra_config=adv_search_extra_config,
                extra_post_config=None,
                rtf=rtf if rtf is not None else 4,
            ),
            lat_2_ctm_args=Lattice2CtmArgs(
                parallelize=True,
                best_path_algo="bellman-ford",
                encoding="utf-8",
                extra_config=lat2ctm_extra_config,
                extra_post_config=None,
                fill_empty_segments=True,
            ),
            scorer_args=self.scorer_args[crp_corpus],
            optimize_parameters=OptimizeJobArgs(
                opt_only_lm_scale=True,
                maxiter=100,
                precision=2,
                extra_config=None,
                extra_post_config=None,
            ),
            optimize_pron_lm_scales=opt_lm_am_scale,
        )

        if calculate_statistics:

            assert adv_tree_search_job is not None
            stats_job = ExtractSearchStatisticsJob(
                search_logs=list(adv_tree_search_job.out_log_file.values()), corpus_duration_hours=durations[crp_corpus]
            )
            exp_str = decoder._get_scales_string(
                am_scale=params.pron_scale,
                lm_scale=params.lm_scale,
                prior_scale=params.prior_info.center_state_prior.scale,
                tdp_scale=params.tdp_scale,
                tdp_speech=to_tdp(params.tdp_speech),
                tdp_silence=to_tdp(params.tdp_silence),
                tdp_nonspeech=to_tdp(params.tdp_non_word),
                altas=params.altas,
            )
            stats_alias = f"{alias_output_prefix}statistics-nn-pch/{self.experiments[key]['name']}/ep{epoch}/rp{prior_epoch}/{exp_str}_beam{params.beam}_bl{params.beam_limit}"

            stats_job.add_alias(stats_alias)
            tk.register_output(f"{stats_alias}/avg_states", stats_job.avg_states)
            tk.register_output(f"{stats_alias}/avg_trees", stats_job.avg_trees)
            tk.register_output(f"{stats_alias}/rtf", stats_job.decoding_rtf)

        return adv_tree_search_job

    # -------------------- run setup  --------------------

    def run(self, steps: RasrSteps):
        if "init" in steps.get_step_names_as_list():
            self.init_system()
        for eval_c in self.dev_corpora + self.test_corpora:
            stm_args = self.rasr_init_args.stm_args if self.rasr_init_args.stm_args is not None else {}
            self.create_stm_from_corpus(eval_c, **stm_args)
            self._set_scorer_for_corpus(eval_c)

        for step_idx, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                if step_args is None:
                    step_args = self.rasr_init_args.feature_extraction_args
                step_args["gt"]["prefix"] = "features/"
                for all_c in (
                    self.train_corpora + self.cv_corpora + self.devtrain_corpora + self.dev_corpora + self.test_corpora
                ):
                    self.feature_caches[all_c] = {}
                    self.feature_bundles[all_c] = {}
                    self.feature_flows[all_c] = {}

                self.extract_features(step_args)
                # ---------- Step Input ----------
            if step_name.startswith("input"):
                self.run_input_step(step_args)
