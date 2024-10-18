__all__ = ["FactoredHybridSystem"]

import copy
import dataclasses
import itertools
import sys
from IPython import embed
from enum import Enum
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

from sisyphus.delayed_ops import DelayedFormat

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
import i6_core.features as features
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog
import i6_core.returnn as returnn
import i6_core.text as text

from i6_core.util import MultiPath, MultiOutputPath
from i6_core.lexicon.allophones import DumpStateTyingJob, StoreAllophonesJob

# common modules
from i6_experiments.common.setups.rasr.nn_system import NnSystem

from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
    RasrSteps,
    ReturnnRasrDataInput,
    ReturnnRasrTrainingArgs,
)

# user dependent
import i6_experiments.users.raissi.setups.common.encoder as encoder_archs
import i6_experiments.users.raissi.setups.common.helpers.network as net_helpers
import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers

from i6_experiments.users.raissi.setups.common.util.tdp import to_tdp

from i6_experiments.users.raissi.setups.common.util.rasr import (
    SystemInput,
)

from i6_experiments.users.raissi.setups.common.helpers.train.cache_epilog import hdf_dataset_cache_epilog

# user based modules
from i6_experiments.users.raissi.setups.common.data.backend import Backend, BackendInfo

from i6_experiments.users.raissi.args.system.am import (
    get_lexicon_args,
    get_tdp_values,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)

from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import (
    TrainingCriterion,
    SingleSoftmaxType,
    Experiment,
    InputKey,
)

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    BASEFactoredHybridDecoder,
    BASEFactoredHybridAligner,
    BASEFactoredHybridLatticeGenerator,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorInfo,
    PriorConfig,
    PosteriorScales,
    SearchParameters,
    AlignmentParameters,
)

from i6_experiments.users.raissi.setups.common.features.taxonomy import FeatureInfo

from i6_experiments.users.raissi.setups.common.helpers.network.frame_rate import FrameRateReductionRatioinfo
from i6_experiments.users.raissi.setups.common.util.hdf.dump import RasrFeaturesToHdf
from i6_experiments.users.raissi.costum.returnn.rasr_returnn_bw import ReturnnRasrTrainingBWJob
from i6_experiments.users.raissi.costum.returnn.rasr_returnn_vit import ReturnnRasrTrainingVITJob

from i6_experiments.users.raissi.costum.corpus.segments import SegmentCorpusNoPrefixJob

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- Systems --------------------
class BASEFactoredHybridSystem(NnSystem):
    """
    this class supports both cart and factored hybrid
    """

    def __init__(
        self,
        returnn_root: Optional[str] = None,
        returnn_python_home: Optional[str] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        rasr_binary_path: Optional[tk.Path] = None,
        rasr_init_args: RasrInitArgs = None,
        train_data: Dict[str, RasrDataInput] = None,
        dev_data: Dict[str, RasrDataInput] = None,
        test_data: Dict[str, RasrDataInput] = None,
        initial_nn_args: Dict = {},
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
        self.label_info = LabelInfo.default_ls()
        self.feature_info = FeatureInfo.default()
        self.staging_info = train_helpers.StagingInfo.default()
        self.frame_rate_reduction_ratio_info = FrameRateReductionRatioinfo.default()
        self.lexicon_args = get_lexicon_args(norm_pronunciation=False)
        self.tdp_values = get_tdp_values()
        self.segments_to_exclude = None

        # since ivectors are still from old systems, they are not concatenated to feature caches
        # If one plans to use creat_hdf function when using ivectors, then there is an assertion
        self.extract_ivectors = False

        # transcription priors in pickle format
        self.priors = None  # these are for transcript priors

        # dataset infomration
        # see if you want to have these or you want to handle this in the function that sets the cv
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

        # backend for train and decode
        self.backend_info = BackendInfo.default()

        # data and pipeline related
        self.inputs = {}  # nested dictionary first keys corpora, second level InputKeys
        self.experiments: Dict[str, Experiment] = {}

        # train information
        self.initial_train_args = {
            "cpu_rqmt": 4,
            "time_rqmt": 168,
            "mem_rqmt": 7,
            "log_verbosity": 3,
        }

        self.cart_state_tying_args = None  # just for simplicity

        # default to None to break it, set it in the config
        self.shuffling_params = {
            "shuffle_data": True,
            "segment_order_sort_by_time_length_chunk_size": 348,
        }
        self.fullsum_log_linear_scales = {"label_posterior_scale": 0.3, "transition_scale": 0.3}

        # extern classes and objects
        self.training_criterion: TrainingCriterion = TrainingCriterion.FULLSUM
        self.trainers = {
            "returnn": returnn.ReturnnTrainingJob,
            "rasr-returnn": returnn.ReturnnRasrTrainingJob,
            "rasr-returnn-costum-vit": ReturnnRasrTrainingVITJob,
            "rasr-returnn-costum-bw": ReturnnRasrTrainingBWJob,
        }
        self.recognizers = {"base": BASEFactoredHybridDecoder}
        self.aligners = {"base": BASEFactoredHybridAligner}
        self.lattice_generators = {"base": BASEFactoredHybridLatticeGenerator}
        self.returnn_configs = {}
        self.graphs = {}

        # inference related
        self.native_lstm2_path: Optional[tk.Path] = None

        # keys when you have different dev and test sets
        self.train_key = None
        self.cv_num_segments = 100
        self.cv_info = None

        self.set_initial_nn_args(initial_nn_args=initial_nn_args)

    def set_crp_pairings(self, dev_key, test_key):
        # have a dict of crp_names so that you can refer to your data as you want
        keys = self.corpora.keys()
        if self.train_key is None:
            self.train_key = [k for k in keys if "train" in k][0]
            print("WARNING: train key was None, it has been set to self.train_key")
        all_names = ["train", "cvtrain", "devtrain", "align"]
        all_names.extend([n for n in keys if n != self.train_key])
        self.crp_names = dict()
        for k in all_names:
            if "train" in k:
                crp_n = f"{self.train_key}.{k}"
                if "cv" in k:
                    if len(self.cv_corpora):
                        crp_n = self.cv_corpora[-1]
                self.crp_names[k] = crp_n
                self._add_feature_and_alignment_for_crp_with_existing_crp(
                    existing_crp_key=self.train_key, new_crp_key=crp_n
                )
            if "align" in k:
                self.crp_names["align.train"] = f"{self.train_key}.{k}"

        self.crp_names["dev"] = dev_key
        self.crp_names["test"] = test_key
        self.sort_returnn_config = True

    def set_experiment_dict(self, key, alignment, context, postfix_name=""):
        prefix_name = f"{context}-from-{alignment}"
        name = ("-").join([prefix_name, postfix_name]) if len(postfix_name) else prefix_name
        self.experiments[key] = {
            "name": name,
            "train_job": None,
            "graph": {"train": None, "inference": None},
            "returnn_config": None,
            "align_job": None,
            "decode_job": {"runner": None, "args": None},
            "extra_returnn_code": {"epilog": "", "prolog": ""},
        }

    # -------------------- Internal helpers --------------------
    def _init_datasets(
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

    def _add_feature_and_alignment_for_crp_with_existing_crp(self, existing_crp_key, new_crp_key):
        assert (
            self.alignments[existing_crp_key] is not None
        ), f"you need to set the alignment for {existing_crp_key} first"
        # assuming feature flows, caches and buundles are all set similarly
        assert (
            self.feature_flows[existing_crp_key][self.feature_info.feature_type.get()] is not None
        ), f"you need to set the features for {existing_crp_key} first"
        self.alignments[new_crp_key] = self.alignments[existing_crp_key]

        feature_name = self.feature_info.feature_type.get()

        if feature_name not in self.feature_caches[existing_crp_key]:
            self.feature_caches[existing_crp_key][feature_name] = {}

        self.feature_caches[new_crp_key] = {feature_name: self.feature_caches[existing_crp_key][feature_name]}

        self.feature_bundles[new_crp_key] = {feature_name: self.feature_bundles[existing_crp_key][feature_name]}
        self.feature_flows[new_crp_key] = {feature_name: self.feature_flows[existing_crp_key][feature_name]}

    def _get_system_input(
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

    def _get_crp_from_existing(
        self, existing_crp: rasr.CommonRasrParameters, corpus_path: Path, segment_path: Path, prefix_name: str = None
    ):
        new_crp = copy.deepcopy(existing_crp)
        new_crp.corpus_config.file = corpus_path
        new_crp.corpus_config.segments.file = segment_path
        new_crp.corpus_config.remove_corpus_name_prefix = f"{prefix_name}/"

        return new_crp

    def _run_input_step(self, step_args):
        for corpus_key, corpus_type in step_args.corpus_type_mapping.items():
            if "train" in corpus_key:
                self.train_key = corpus_key
            if corpus_key not in self.train_corpora + self.cv_corpora + self.dev_corpora + self.test_corpora:
                continue
            if "train" in corpus_key:
                if self.train_key is None:
                    self.train_key = corpus_key
                else:
                    assert (
                        self.train_key == corpus_key
                    ), f"You already set the train key to be {self.train_key}, you cannot have more than one train key"
            if corpus_key not in self.inputs.keys():
                self.inputs[corpus_key] = {}
            self.inputs[corpus_key][step_args.name] = self._get_system_input(
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
            self._set_scorer_for_corpus(c_key)

    def _update_crp_am_setting(
        self,
        crp_key: str,
        tdp_type: str = None,
        decode_state_tying: RasrStateTying = RasrStateTying.triphone,
        add_base_allophones=False,
    ):
        # ToDo handle different tdp values: default, based on transcription, based on an alignment
        tdp_pattern = self.tdp_values["pattern"]
        if tdp_type is None:
            tdp_type = "default"
        if tdp_type == "default":  # additional later, maybe enum or so
            tdp_values = self.tdp_values[tdp_type]
        elif isinstance(tdp_type, tuple):
            tdp_values = self.tdp_values[tdp_type[0]][tdp_type[1]]

        crp = self.crp[crp_key]
        for ind, ele in enumerate(tdp_pattern):
            for type in ["*", "silence"]:
                crp.acoustic_model_config["tdp"][type][ele] = tdp_values[type][ind]

        if self.label_info.state_tying == RasrStateTying.cart:
            crp.acoustic_model_config.state_tying.type = self.label_info.state_tying
            assert (
                self.cart_state_tying_args is not None
            ), "for cart state tying you need to set state tying file for label_info"
            crp.acoustic_model_config.state_tying.file = self.cart_state_tying_args["cart"]
        else:
            if self.label_info.phoneme_state_classes.use_word_end():
                crp.acoustic_model_config.state_tying.use_word_end_classes = (
                    self.label_info.phoneme_state_classes.use_word_end()
                )
            crp.acoustic_model_config.state_tying.use_boundary_classes = (
                self.label_info.phoneme_state_classes.use_boundary()
            )
            crp.acoustic_model_config.hmm.states_per_phone = self.label_info.n_states_per_phone
            if "train" in crp_key:
                crp.acoustic_model_config.state_tying.type = self.label_info.state_tying
            else:
                crp.acoustic_model_config.state_tying.type = decode_state_tying

        crp.acoustic_model_config.allophones.add_all = self.lexicon_args["add_all_allophones"]
        crp.acoustic_model_config.allophones.add_from_lexicon = not self.lexicon_args["add_all_allophones"]
        if add_base_allophones:
            crp.acoustic_model_config.allophones.add_from_file = self.base_allophones

        crp.lexicon_config.normalize_pronunciation = self.lexicon_args["norm_pronunciation"]

    def reset_state_tying(self, crp_list=None, state_tying: RasrStateTying = RasrStateTying.diphone):
        # the default decoding state tying is no-tying-dense, unless nn_precomputed feature scorer is used
        if crp_list is None:
            crp_list = list(self.crp_names.keys())
        self.label_info = dataclasses.replace(self.label_info, state_tying=state_tying)
        for crp_k in crp_list:
            self._update_crp_am_setting(self.crp_names[crp_k], decode_state_tying=state_tying)

    def _get_segment_file(self, corpus_path, remove_prefix=None):
        if remove_prefix is not None:
            j = SegmentCorpusNoPrefixJob(bliss_corpus=corpus_path, num_segments=1, remove_prefix=remove_prefix)
        else:
            j = corpus_recipe.SegmentCorpusJob(
                bliss_corpus=corpus_path,
                num_segments=1,
            )
        return j.out_single_segment_files[1]

    def _get_merged_corpus_for_corpora(
        self, corpora, name="loss-corpus", strategy=corpus_recipe.MergeStrategy.SUBCORPORA
    ):
        merged_corpus_job = corpus_recipe.MergeCorporaJob(corpora, name=name, merge_strategy=strategy)
        return merged_corpus_job.out_merged_corpus

    def _add_merged_cv_corpus(self, segment_list: Path):
        corpus_key = self.cv_no_train_key = ("_").join(self.cv_corpora)
        reference_crp = self.crp[self.cv_corpora[0]]
        cv_corpora_obj = [self.corpora[k].corpus_file for k in self.cv_corpora]
        durations = np.sum([self.corpora[k].duration for k in self.cv_corpora])
        merged_corpus_all = corpus_recipe.MergeCorporaJob(
            cv_corpora_obj, name=corpus_key, merge_strategy=corpus_recipe.MergeStrategy.FLAT
        ).out_merged_corpus
        merged_corpus = corpus_recipe.FilterCorpusBySegmentsJob(
            bliss_corpus=merged_corpus_all, segment_file=segment_list, delete_empty_recordings=True
        ).out_corpus

        lexicon = {
            "filename": reference_crp.lexicon_config.file,
            "normalize_pronunciation": False,
        }

        from i6_core.meta.system import CorpusObject

        corpus_object = CorpusObject()
        corpus_object.corpus_file = merged_corpus
        corpus_object.audio_format = "wav"
        corpus_object.audio_dir = None
        corpus_object.duration = durations

        cv_segments = self._get_segment_file(merged_corpus)

        rasr_data_input = RasrDataInput(
            corpus_object=corpus_object,
            concurrent=1,
            lexicon=lexicon,
        )
        self.add_corpus(corpus_key, data=rasr_data_input, add_lm=False)
        joined_crp = self._get_crp_from_existing(
            existing_crp=reference_crp,
            corpus_path=merged_corpus,
            segment_path=cv_segments,
            prefix_name="",
        )
        joined_crp.concurrent = 1
        joined_crp.segment_path = cv_segments

        self.crp[corpus_key] = joined_crp
        self.cv_corpora.append(corpus_key)

    def _get_prior_info_dict(self):
        prior_dict = {}
        for k in ["left-context", "center-state", "right-context"]:
            prior_dict[f"{k}-prior"] = {}
            prior_dict[f"{k}-prior"]["scale"] = 0.0
            prior_dict[f"{k}-prior"]["file"] = None
        return prior_dict

    def _add_output_alias_for_train_job(
        self,
        train_job: Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob],
        name: str,
    ):
        train_job.add_alias(f"train/nn_{name}")
        tk.register_output(
            f"train/{name}_learning_rate.png",
            train_job.out_plot_lr,
        )

    # -------------------- External helpers --------------------
    def get_model_checkpoint(self, model_job, epoch):
        return model_job.out_checkpoints[epoch]

    def get_model_path(self, model_job, epoch):
        return model_job.out_checkpoints[epoch].ckpt_path

    def delete_multitask_components(self, returnn_config, n_mlps=2):
        config = copy.deepcopy(returnn_config)

        for o in ["left", "right"]:
            if "linear" in config.config["network"][f"{o}-output"]["from"]:
                for i in range(n_mlps):
                    for k in ["leftContext", "triphone"]:
                        del config.config["network"][f"linear{i+1}-{k}"]
            del config.config["network"][f"{o}-output"]

        return config

    def _set_native_lstm_path(self, search_numpy_blas=True, blas_lib=None):
        compile_native_op_job = returnn.CompileNativeOpJob(
            "NativeLstm2",
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            blas_lib=blas_lib,
            search_numpy_blas=search_numpy_blas,
        )
        self.native_lstm2_path = compile_native_op_job.out_op

    def set_local_flf_tool_for_decoding(self, path):
        self.crp["base"].flf_tool_exe = path

    # --------------------- Init procedure -----------------
    def set_initial_nn_args(self, initial_nn_args):
        """
        self.initial_nn_args = {
            "num_input": None,
            "keep_epochs": None,
            "keep_best_n": None,
            "partition_epochs": {"train": 1, "dev": 1}
        }
        """
        assert "partition_epochs" in initial_nn_args, "You forgot to set the partition epochs in the initial_nn_args"
        self.partition_epochs = initial_nn_args.pop("partition_epochs")
        self.initial_nn_args = initial_nn_args

    def init_system(
        self,
        frr_additional_args=None,
        feature_info_additional_args=None,
        label_info_additional_args=None,
        native_lstm_compilation_args=None,
    ):

        if self.native_lstm2_path is None:
            if native_lstm_compilation_args is None:
                native_lstm_compilation_args = {}
            self._set_native_lstm_path(**native_lstm_compilation_args)

        if feature_info_additional_args is not None:
            self.feature_info = dataclasses.replace(self.feature_info, **feature_info_additional_args)

        if label_info_additional_args is not None:
            self.label_info = dataclasses.replace(self.label_info, **label_info_additional_args)

        if frr_additional_args is not None:
            self.frame_rate_reduction_ratio_info = dataclasses.replace(
                self.frame_rate_reduction_ratio_info, **frr_additional_args
            )

        for param in [self.rasr_init_args, self.label_info, self.train_data, self.dev_data, self.test_data]:
            assert param is not None

        self._init_am(**self.rasr_init_args.am_args)
        self._assert_corpus_name_unique(self.train_data, self.dev_data, self.test_data)

        for corpus_key, rasr_data_input in sorted(self.train_data.items()):
            add_lm = True if rasr_data_input.lm is not None else False
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=add_lm)
            self.train_corpora.append(corpus_key)

        for corpus_key, rasr_data_input in sorted(self.dev_data.items()):
            add_lm = True if rasr_data_input.lm is not None else False
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=add_lm)
            self.dev_corpora.append(corpus_key)

        for corpus_key, rasr_data_input in sorted(self.test_data.items()):
            add_lm = True if rasr_data_input.lm is not None else False
            self.add_corpus(corpus_key, data=rasr_data_input, add_lm=add_lm)
            self.test_corpora.append(corpus_key)

        assert len(self.train_corpora) < 2, "you can have only one corpus for training"

    def update_am_setting_for_all_crps(self, train_tdp_type, eval_tdp_type, add_base_allophones=False):
        types = {"train": train_tdp_type, "eval": eval_tdp_type}
        for t in types.keys():
            if types[t].startswith("heuristic"):
                if self.label_info.n_states_per_phone > 1:
                    types[t] = (types[t], "threepartite")
                else:
                    types[t] = (types[t], "monostate")

        for crp_k in self.crp_names.keys():
            if "train" in self.crp_names[crp_k]:
                self._update_crp_am_setting(
                    crp_key=self.crp_names[crp_k], tdp_type=types["train"], add_base_allophones=add_base_allophones
                )
            else:
                self._update_crp_am_setting(
                    crp_key=self.crp_names[crp_k], tdp_type=types["eval"], add_base_allophones=add_base_allophones
                )

    def set_rasr_returnn_input_datas(
        self, input_key: InputKey = InputKey.BASE, cv_corpus_key: str = None, is_cv_separate_from_train=False
    ):
        for k in self.corpora.keys():
            assert self.inputs[k] is not None
            assert self.inputs[k][input_key] is not None
        assert cv_corpus_key is not None or not is_cv_separate_from_train, "define a corpus key for separate cv data"

        feature_name = self.feature_info.feature_type.get()

        configure_automata = False
        if self.training_criterion in [TrainingCriterion.FULLSUM, TrainingCriterion.sMBR]:
            configure_automata = True
        # get the train data for alignment before you changed any setting
        nn_train_align_data = copy.deepcopy(
            self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(feature_flow_key=feature_name)
        )

        if is_cv_separate_from_train:
            (
                nn_train_data_inputs,
                nn_cv_data_inputs,
                nn_devtrain_data_inputs,
            ) = self.prepare_rasr_train_data_with_separate_cv(
                input_key, cv_corpus_key=cv_corpus_key, configure_rasr_automaton=configure_automata
            )
        else:
            (
                nn_train_data_inputs,
                nn_cv_data_inputs,
                nn_devtrain_data_inputs,
            ) = self.prepare_rasr_train_data_with_cv_from_train(input_key, cv_num_segments=self.cv_num_segments)

        nn_train_data_inputs[self.crp_names["align.train"]] = nn_train_align_data

        nn_dev_data_inputs = {
            self.crp_names["dev"]: self.inputs[self.crp_names["dev"]][input_key].as_returnn_rasr_data_input(
                feature_flow_key=feature_name
            ),
        }
        nn_test_data_inputs = {
            self.crp_names["test"]: self.inputs[self.crp_names["test"]][input_key].as_returnn_rasr_data_input(
                feature_flow_key=feature_name
            ),
        }
        self._init_datasets(
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

    # ----- data preparation for train-----------------------------------------------------
    def prepare_rasr_train_data_with_cv_from_train(
        self, input_key: InputKey = InputKey.BASE, cv_num_segments: int = 100
    ):

        assert self.train_key is not None, "You did not specify the train_key"
        train_corpus_path = self.corpora[self.train_key].corpus_file

        all_segments = self._get_segment_file(corpus_path=train_corpus_path)

        assert self.train_key in self.num_segments, "It seems that you set a wrong train key in inputs step"
        cv_size = cv_num_segments / self.num_segments[self.train_key]

        splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
            all_segments, {"train": 1 - cv_size, "cv": cv_size}
        )
        train_segments = splitted_segments_job.out_segments["train"]
        cv_segments = splitted_segments_job.out_segments["cv"]
        devtrain_segments = text.TailJob(train_segments, num_lines=100, zip_output=False).out

        # ******************** NN Init ********************
        feature_name = self.feature_info.feature_type.get()

        nn_train_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            feature_flow_key=feature_name, shuffling_parameters=self.shuffling_params
        )
        nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
        nn_train_data_inputs = {self.crp_names["train"]: nn_train_data}

        nn_cv_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(feature_flow_key=feature_name)
        nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
        nn_cv_data_inputs = {self.crp_names["cvtrain"]: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            feature_flow_key=feature_name
        )
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names["devtrain"]: nn_devtrain_data}

        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs

    def prepare_rasr_train_data_with_separate_cv(
        self, input_key: InputKey = InputKey.BASE, cv_corpus_key="dev-other", configure_rasr_automaton=False
    ):
        train_corpus_key = self.train_key

        train_corpus = self.corpora[train_corpus_key].corpus_file
        cv_corpus = self.corpora[cv_corpus_key].corpus_file

        merged_name = "loss-corpus"
        merged_corpus = self._get_merged_corpus_for_corpora(corpora=[train_corpus, cv_corpus], name=merged_name)
        # strategy=corpus_recipe.MergeStrategy.FLAT)

        # segments
        train_segments = self._get_segment_file(train_corpus)
        cv_segments = self._get_segment_file(cv_corpus)
        merged_segments = self._get_segment_file(merged_corpus, remove_prefix=f"{merged_name}/")

        devtrain_segments = text.TailJob(train_segments, num_lines=1000, zip_output=False).out

        if configure_rasr_automaton:
            self.crp_names["bw"] = f"{self.train_key}.bw"
            crp_bw = self._get_crp_from_existing(
                existing_crp=self.crp[self.train_key],
                corpus_path=merged_corpus,
                segment_path=merged_segments,
                prefix_name=merged_name,
            )
            self.crp[self.crp_names["bw"]] = crp_bw

        feature_name = self.feature_info.feature_type.get()

        nn_train_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            feature_flow_key=feature_name,
            shuffling_parameters=self.shuffling_params,
            returnn_rasr_training_args=ReturnnRasrTrainingArgs(partition_epochs=self.partition_epochs["train"]),
        )
        nn_train_data.update_crp_with(corpus_file=train_corpus, segment_path=train_segments, concurrent=1)
        nn_train_data_inputs = {self.crp_names["train"]: nn_train_data}

        nn_cv_data = self.inputs[cv_corpus_key][input_key].as_returnn_rasr_data_input(
            feature_flow_key=feature_name,
            returnn_rasr_training_args=ReturnnRasrTrainingArgs(partition_epochs=self.partition_epochs["dev"]),
        )
        nn_cv_data.update_crp_with(corpus_file=cv_corpus, segment_path=cv_segments, concurrent=1)
        nn_cv_data_inputs = {self.crp_names["cvtrain"]: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            feature_flow_key=feature_name
        )
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names["devtrain"]: nn_devtrain_data}

        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs

    # -------------------------------------------------------------------------

    def returnn_training(
        self,
        experiment_key: str,
        returnn_config: returnn.ReturnnConfig,
        nn_train_args: Any,
        on_2080: bool = False,
    ):
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = returnn.ReturnnTrainingJob(
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
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
        train_hdfs: List[tk.Path],
        dev_hdfs: List[tk.Path],
        on_2080: bool = True,
        dev_data: Optional[Dict[str, Any]] = None,
        train_data: Optional[Dict[str, Any]] = None,
    ):
        from textwrap import dedent

        assert isinstance(returnn_config, returnn.ReturnnConfig)

        # if user is setting partition_epochs in the train args and whether it is inconsistent with the system info
        assert self.partition_epochs is not None, "Set the partition_epochs dictionary"
        if "partition_epochs" in nn_train_args:
            for k in ["train", "dev"]:
                assert nn_train_args["partition_epochs"][k] == self.partition_epochs[k], "wrong partition_epochs"
        nn_train_args.pop("partition_epochs")

        dev_data = {
            "class": "NextGenHDFDataset",
            "files": dev_hdfs,
            "input_stream_name": "features",
            "partition_epoch": self.partition_epochs["dev"],
            **(dev_data or {}),
        }
        train_data = {
            "class": "NextGenHDFDataset",
            "files": train_hdfs,
            "input_stream_name": "features",
            "partition_epoch": self.partition_epochs["train"],
            "seq_ordering": f"random:{PRIOR_RNG_SEED}",
            **(train_data or {}),
        }

        returnn_config = copy.deepcopy(returnn_config)
        update_config = returnn.ReturnnConfig(
            config={"dev": dev_data, "train": train_data},
            python_epilog=train_helpers.cache_epilog.hdf_dataset_cache_epilog,
        )
        returnn_config.update(update_config)

        return self.returnn_training(
            experiment_key=experiment_key, returnn_config=returnn_config, on_2080=on_2080, nn_train_args=nn_train_args
        )

    def concat_features_with_ivec(self, feature_net, ivec_path):
        """
        copy pasted from i6_core.adaptation.ivector, it cannot import at the moment due to bob module
        Generate a new flow-network with i-vectors repeated and concatenated to original feature stream
        :param feature_net: original flow-network
        :param ivec_path: ivec_path from IVectorExtractionJob
        :return:
        """
        # copy original net
        net = rasr.FlowNetwork(name=feature_net.name)
        net.add_param(["id", "start-time", "end-time"])
        net.add_output("features")
        mapping = net.add_net(feature_net)
        net.interconnect_inputs(feature_net, mapping)

        # load ivec cache and repeat
        fc = net.add_node("generic-cache", "feature-cache-ivec", {"id": "$(id)", "path": ivec_path})
        sync = net.add_node("signal-repeating-frame-prediction", "sync")
        net.link(fc, sync)
        for node in feature_net.get_output_links("features"):
            net.link(node, "%s:%s" % (sync, "target"))

        # concat original feature output with repeated ivecs
        concat = net.add_node(
            "generic-vector-f32-concat",
            "concatenation",
            {"check-same-length": True, "timestamp-port": "feature-1"},
        )
        for node in feature_net.get_output_links("features"):
            net.link(node, "%s:%s" % (concat, "feature-1"))
        net.link(sync, "%s:%s" % (concat, "feature-2"))

        net.link(concat, "network:features")

        return net

    def concat_features_with_ivectors_for_feature_flow(self):
        pass

    def get_feature_and_alignment_flows_for_training(self, data):
        if data.feature_flow is not None:
            feature_flow = data.feature_flow
        else:
            if isinstance(data.features, rasr.FlagDependentFlowAttribute):
                feature_path = data.features
            elif isinstance(data.features, (MultiPath, MultiOutputPath)):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": data.features,
                    },
                )
            elif isinstance(data.features, tk.Path):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "bundle": data.features,
                    },
                )
            else:
                raise NotImplementedError

            feature_flow = features.basic_cache_flow(feature_path)
            if isinstance(data.features, tk.Path):
                feature_flow.flags = {"cache_mode": "bundle"}

        if isinstance(data.alignments, rasr.FlagDependentFlowAttribute):
            alignments = copy.deepcopy(data.alignments)
            net = rasr.FlowNetwork()
            net.flags = {"cache_mode": "bundle"}
            alignments = alignments.get(net)
        elif isinstance(data.alignments, (MultiPath, MultiOutputPath)):
            raise NotImplementedError
        elif isinstance(data.alignments, tk.Path):
            alignments = data.alignments
        else:
            raise NotImplementedError

        return feature_flow, alignments

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

        # if user is setting partition_epochs in the train args and whether it is inconsistent with the system info
        assert self.partition_epochs is not None, "Set the partition_epochs dictionary"
        if "partition_epochs" in nn_train_args:
            for k in ["train", "dev"]:
                assert nn_train_args["partition_epochs"][k] == self.partition_epochs[k], "wrong partition_epochs"
        else:
            nn_train_args["partition_epochs"] = self.partition_epochs

        if "returnn_config" not in nn_train_args:
            returnn_config = self.experiments[experiment_key]["returnn_config"]
        else:
            returnn_config = nn_train_args.pop("returnn_config")
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        if (
            train_data.feature_flow == dev_data.feature_flow
            and train_data.features == dev_data.features
            and train_data.alignments == dev_data.alignments
        ):
            trainer = self.trainers["rasr-returnn"]
            feature_flow, alignments = self.get_feature_and_alignment_flows_for_training(data=train_data)
        else:
            trainer = self.trainers["rasr-returnn-costum-vit"]
            feature_flow = {"train": None, "dev": None}
            alignments = {"train": None, "dev": None}
            feature_flow["train"], alignments["train"] = self.get_feature_and_alignment_flows_for_training(
                data=train_data
            )
            feature_flow["dev"], alignments["dev"] = self.get_feature_and_alignment_flows_for_training(data=dev_data)

        if self.segments_to_exclude is not None:
            extra_config = (
                rasr.RasrConfig() if "extra_rasr_config" not in nn_train_args else nn_train_args["extra_rasr_config"]
            )
            extra_config["*"].segments_to_skip = self.segments_to_exclude
            nn_train_args["extra_rasr_config"] = extra_config

        train_job = trainer(
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

        # if user is setting partition_epochs in the train args and whether it is inconsistent with the system info
        assert self.partition_epochs is not None, "Set the partition_epochs dictionary"
        if "partition_epochs" in nn_train_args:
            for k in ["train", "dev"]:
                assert nn_train_args["partition_epochs"][k] == self.partition_epochs[k], "wrong partition_epochs"
        else:
            nn_train_args["partition_epochs"] = self.partition_epochs

        if "returnn_config" not in nn_train_args:
            returnn_config = self.experiments[experiment_key]["returnn_config"]
        else:
            returnn_config = nn_train_args.pop("returnn_config")
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = self.trainers["rasr-returnn-costum-bw"](
            train_crp=train_crp,
            dev_crp=dev_crp,
            feature_flows={"train": train_data.feature_flow, "dev": dev_data.feature_flow},
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
    def get_hdf_path(self, hdf_key: Optional[str]):
        if hdf_key is not None:
            assert hdf_key in self.hdfs.keys()
            return self.hdfs[hdf_key]

        if self.train_key not in self.hdfs.keys():
            self.create_hdf()

        return self.hdfs[self.train_key]

    def create_hdf(self):
        assert not self.extract_ivectors, "your system does not prepare the feature caches with the ivectors yet"
        gammatone_features_paths: MultiPath = self.feature_caches[self.train_key][self.feature_info.feature_type.get()]
        hdf_job = RasrFeaturesToHdf(
            feature_caches=gammatone_features_paths,
        )

        self.hdfs[self.train_key] = hdf_job.out_hdf_files

        hdf_job.add_alias(f"hdf/{self.train_key}")
        tk.register_output("hdf/hdf.train.1", hdf_job.out_hdf_files[0])

        return hdf_job

        # -----------------------Decoding --------------------------

    def get_parameters_for_decoder(
        self, context_type: PhoneticContext, prior_info: PriorInfo
    ) -> Union[SearchParameters, AlignmentParameters]:
        parameters = SearchParameters.default_for_ctx(context_type, priors=prior_info)
        if self.frame_rate_reduction_ratio_info.factor > 2:
            sp_tdp = (10.0, 0.0, "infinity", 0.0)
            sil_tdp = (10.0, 0.0, "infinity", 20.0)
            parameters = parameters.with_tdp_speech(sp_tdp).with_tdp_silence(sil_tdp)

        return parameters

    def get_parameters_for_aligner(
        self, context_type: PhoneticContext, prior_info: PriorInfo
    ) -> Union[SearchParameters, AlignmentParameters]:
        parameters = AlignmentParameters.default_for_ctx(context_type, priors=prior_info)
        if self.frame_rate_reduction_ratio_info.factor > 2:
            sp_tdp = (0.0, 3.0, "infinity", 0.0)
            sil_tdp = (10.0, 0.0, "infinity", 0.0)
            parameters = parameters.with_tdp_speech(sp_tdp).with_tdp_silence(sil_tdp)

        return parameters

    # -------------------- run setup  --------------------

    def run(self, steps: RasrSteps):
        # init args are the label info args

        if "init" in steps.get_step_names_as_list():
            init_args = steps.get_args_via_idx(0)
            feature_info_args = init_args["feature_info"] if "feature_info" in init_args else None
            frame_rate_args = init_args["frr_info"] if "frr_info" in init_args else None
            label_info_args = init_args["label_info"] if "label_info" in init_args else None
            native_lstm_compilation_args = init_args["native_lstm"] if "native_lstm" in init_args else None

            self.init_system(
                feature_info_additional_args=feature_info_args,
                frr_additional_args=frame_rate_args,
                label_info_additional_args=label_info_args,
                native_lstm_compilation_args=native_lstm_compilation_args,
            )
            if len(self.cv_corpora) > 1:
                if self.cv_info is None or self.cv_info["segment_list"] is None:
                    assert "cv_info" in init_args, "please set the segment file for the cross validation data"
                    segment_list = init_args["cv_info"]["segment_list"]
                else:
                    segment_list = self.cv_info["segment_list"]
                self._add_merged_cv_corpus(segment_list=segment_list)
        all_corpora = self.train_corpora + self.cv_corpora + self.dev_corpora + self.test_corpora
        for eval_c in self.dev_corpora + self.test_corpora:
            stm_args = self.rasr_init_args.stm_args if self.rasr_init_args.stm_args is not None else {}
            self.create_stm_from_corpus(eval_c, **stm_args)
            self._set_scorer_for_corpus(eval_c)

        for step_idx, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                if step_args is None:
                    step_args = self.rasr_init_args.feature_extraction_args
                    for all_c in all_corpora:
                        self.feature_caches[all_c] = {}
                        self.feature_bundles[all_c] = {}
                        self.feature_flows[all_c] = {}
                if step_args[self.feature_info.feature_type.get()] is not None:
                    step_args[self.feature_info.feature_type.get()]["prefix"] = "features/"
                    self.extract_ivectors = step_args[self.feature_info.feature_type.get()].pop(
                        "extract_ivectors", False
                    )
                    self.extract_features(step_args, corpus_list=all_corpora)
                    if self.extract_ivectors:
                        self.concat_features_with_ivectors_for_feature_flow()

            # -----------Set alignments if needed-------
            # here you might one to align cv with a given aligner
            if step_name.startswith("alignment"):
                # step_args here is a dict that has the keys as corpora
                if self.cv_info is not None:
                    step_args.update(self.cv_info["alignment"])
                for c in step_args.keys():
                    self.alignments[c] = step_args[c]
            # ---------------Step Input ----------
            if step_name.startswith("input"):
                self._run_input_step(step_args)
