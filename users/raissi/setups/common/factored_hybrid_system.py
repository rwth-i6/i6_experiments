__all__ = ["FactoredHybridSystem"]

import copy
import itertools
import sys

from dataclasses import asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict, Union

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

#common modules
from i6_experiments.common.setups.rasr.nn_system import (
    NnSystem
)


from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
    RasrSteps,
    ReturnnRasrDataInput,
)

import i6_experiments.users.raissi.setups.common.encoder.blstm as blstm_setup
import i6_experiments.users.raissi.setups.common.encoder.conformer as conformer_setup
import i6_experiments.users.raissi.setups.common.helpers.network.augment as fh_augmenter
import i6_experiments.users.raissi.common.helpers.train as train_helpers


from i6_experiments.users.raissi.setups.common.helpers.train.specaugment import (
    mask as sa_mask,
    random_mask as sa_random_mask,
    summary as sa_summary,
    transform as sa_transform,
)


#user based modules
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import (
    get_lexicon_args,
    get_tdp_values,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo

from i6_experiments.users.raissi.setups.common.decoder.factored_hybrid_search import (
    FactoredHybridBaseDecoder
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorInfo,
    PosteriorScales,
    SearchParameters
)

from i6_experiments.users.raissi.setups.common.util.hdf import (
    SprintFeatureToHdf
)


#From here to be checked
#get_recog_ctx_args*() functions are imported here
"""
from i6_experiments.users.raissi.experiments.librispeech.search.recognition_args import *
from i6_experiments.users.raissi.setups.common.helpers.estimate_povey_like_prior_fh import *


from i6_experiments.users.raissi.setups.common.helpers.pipeline_data import (
    ContextEnum,
    ContextMapper,
    LabelInfo,
    PipelineStages,
    RasrFeatureToHDF,
    RasrFeatureAndAlignmentToHDF
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

import i6_private.users.raissi.datasets.ukrainian_8khz as uk_8khz_data

from i6_experiments.users.raissi.setups.hykist.util.pipeline_helpers import (
    get_lexicon_args,
    get_tdp_values,
)

from i6_experiments.users.raissi.setups.common.decoder.factored_hybrid_search import (
    FactoredHybridBaseDecoder
)
"""
# -------------------- Init --------------------

Path = tk.setup_path(__package__)


class TrainingCriterion(Enum):
    """The training criterion."""
    viterbi = "viterbi"
    fullsum = "fullsum"
    mmi = "mmi"

    def __str__(self):
        return self.value


class ExtraReturnnCode(TypedDict):
    epilog: str
    prolog: str

class Graphs(TypedDict):
    train: Optional[tk.Path]
    inference: Optional[tk.Path]

class Experiment(TypedDict, total=False):
    extra_returnn_code: ExtraReturnnCode
    name: str
    graph: Graphs
    priors: Optional[PriorInfo]
    prior_job: Optional[returnn.ReturnnRasrComputePriorJobV2]
    returnn_config: Optional[returnn.ReturnnConfig]
    train_job: Optional[returnn.ReturnnRasrTrainingJob]

# -------------------- Systems --------------------
class FactoredHybridBaseSystem(NnSystem):
    """
    this class supports both cart and factored hybrid
    """

    def __init__(
            self,
            returnn_root: Optional[str] = None,
            returnn_python_home: Optional[str] = None,
            returnn_python_exe: Optional[tk.Path] = None,
            rasr_binary_path: Optional[tk.Path] = None,
            rasr_init_args:   RasrInitArgs = None,
            train_data: Dict[str, RasrDataInput] = None,
            dev_data:   Dict[str, RasrDataInput] = None,
            test_data:  Dict[str, RasrDataInput] = None,
            initial_nn_args: Dict = None,
    ):
        super().__init__(
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
        )
        # arguments used for the initialization of the system, they should be set before running the system
        self.rasr_init_args = rasr_init_args
        self.train_data    = train_data
        self.dev_data      = dev_data
        self.test_data     = test_data


        # general modeling approach
        self.label_info = LabelInfo.default_ls()
        self.lexicon_args = get_lexicon_args(norm_pronunciation=False)
        self.tdp_values = get_tdp_values()

        # transcription priors in pickle format
        self.priors = None #these are for transcript priors

        # dataset infomration
        #see if you want to have these or you want to handle this in the function that sets the cv
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


        #data and pipeline related
        self.inputs = {}
        self.experiments: Dict[str, Experiment] = {}

        # train information
        self.nn_feature_type = 'gt' #Gammatones
        # mostly used as placeholders and info to the user about what is necessary
        self.initial_nn_args = {'num_input': None,
                                'partition_epochs': None,
                                "num_epochs": None,
                                "keep_epochs": None,
                                'time_rqmt': 168,
                                'mem_rqmt': 40,
                                'log_verbosity': 3,}

        self.initial_nn_args.update(**initial_nn_args)

        self.partition_epochs = {'train': 6, 'cv': 1}
        self.shuffling_params = {
            "shuffle_data": True,
            "segment_order_sort_by_time_length_chunk_size": 348,
        }
        self.fullsum_log_linear_scales = {"label_posterior_scale": 0.3, "transition_scale": 0.3}

        # extern classes and objects
        self.training_criterion: TrainingCriterion = TrainingCriterion.fullsum
        self.trainers    = {'returnn': returnn.ReturnnTrainingJob, 'rasr-returnn': returnn.ReturnnRasrTrainingJob}
        self.recognizers = {'count_lm': FactoredHybridBaseDecoder}
        self.aligners    = {}
        self.returnn_configs = {}
        self.graphs = {}

        # inference related
        self.native_lstm2_path: Optional[tk.Path] = None

        #keys when you have different dev and test sets
        self.train_key = None  # "train-baseline"
    def set_crp_pairings(self, dev_key, test_key):
        #have a dict of crp_names so that you can refer to your data as you want
        keys = self.corpora.keys()
        if self.train_key is None:
            self.train_key = [k for k in keys if 'train' in k][0]
            print('WARNING: train key was None, it has been set to self.train_key')
        all_names = ['train', 'cvtrain', 'devtrain']
        all_names.extend([n for n in keys if n != self.train_key])
        self.crp_names = dict()
        for k in all_names:
            if 'train' in k:
                crp_n = f'{self.train_key}.{k}'
                self.crp_names[k]  = crp_n
                self._add_feature_and_alignment_for_crp_with_existing_crp(
                    existing_crp_key=self.train_key,
                    new_crp_key=crp_n
                )
        self.crp_names['dev']  = dev_key
        self.crp_names['test'] = test_key



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

    # -------------------- Internal helpers --------------------
    def _init_datasets(
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

    def _add_feature_and_alignment_for_crp_with_existing_crp(self, existing_crp_key, new_crp_key):
        assert self.alignments[existing_crp_key] is not None, f'you need to set the alignment for {existing_crp_key} first'
        #assuming feature flows, caches and buundles are all set similarly
        assert self.feature_flows[existing_crp_key][self.nn_feature_type] is not None, f'you need to set the features for {existing_crp_key} first'
        self.alignments[new_crp_key]      = self.alignments[existing_crp_key]
        self.feature_caches[new_crp_key]  = {self.nn_feature_type: self.feature_caches[existing_crp_key][self.nn_feature_type]}
        self.feature_bundles[new_crp_key] = {self.nn_feature_type: self.feature_bundles[existing_crp_key][self.nn_feature_type]}
        self.feature_flows[new_crp_key]   = {self.nn_feature_type: self.feature_flows[existing_crp_key][self.nn_feature_type]}

    def _set_native_lstm_path(self):
        compile_native_op_job = returnn.CompileNativeOpJob(
            "NativeLstm2",
            returnn_root=returnn_root,
            returnn_python_exe=returnn_python_exe,
            blas_lib=None,
        )
        self.native_lstm2_path = compile_native_op_job.out_op

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

    def _run_input_step(self, step_args):
        for corpus_key, corpus_type in step_args.corpus_type_mapping.items():
            if 'train' in corpus_key:
                self.train_key = corpus_key
            if (
                    corpus_key
                    not in self.train_corpora + self.dev_corpora + self.test_corpora
            ):
                continue
            if 'train' in corpus_key:
                if self.train_key is None:
                    self.train_key = corpus_key
                else:
                    assert self.train_key == corpus_key, f"You already set the train key to be {self.train_key}, you cannot have more than one train key"
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
        print(crp_key)
        for ind, ele in enumerate(tdp_pattern):
            for type in ["*", "silence"]:
                crp.acoustic_model_config["tdp"][type][ele] = tdp_values[type][ind]

        if self.label_info.state_tying == 'cart':
            crp.acoustic_model_config.state_tying.type = self.label_info.state_tying
            assert self.label_info.state_tying_file is not None, 'for cart state tying you need to set state tying file for label_info'
            crp.acoustic_model_config.state_tying.file = self.label_info.state_tying_file
        else:
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

    def _get_segment_file(self, corpus_path, remove_prefix=""):
        return corpus_recipe.SegmentCorpusJob(
            bliss_corpus=corpus_path,
            num_segments=1,
        ).out_single_segment_files[1]

    def _get_merged_corpus_for_train(self, train_corpus, cv_corpus, name="loss-corpus"):
        merged_corpus_job = corpus_recipe.MergeCorporaJob([train_corpus, cv_corpus],
                                                          name=name,
                                                          merge_strategy=corpus_recipe.MergeStrategy.SUBCORPORA)
        return merged_corpus_job.out_merged_corpus

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
            name: str,
    ):
        train_job.add_alias(f"train/nn_{name}")
        tk.register_output(
            f"train/{name}_learning_rate.png",
            train_job.out_plot_lr,
        )

    # -------------------- External helpers --------------------
    def get_epilog_for_train(self, specaug_args=None):
        #this is for FH when one needs to define extern data
        if specaug_args is not None:
            spec_augment_epilog = get_specaugment_epilog(**specaug_args)
        else:
            spec_augment_epilog = None
        return get_epilog_code_dense_label(n_input=self.initial_nn_args["num_input"],
                                           n_contexts=self.label_info.n_contexts,
                                           n_states=self.label_info.n_states_per_phone,
                                           specaugment=spec_augment_epilog)

    def get_model_checkpoint(self, model_job, epoch):
        return model_job.out_checkpoints[epoch]

    def get_model_path(self, model_job, epoch):
        return model_job.out_checkpoints[epoch].ckpt_path

    def delete_multitask_components(self, returnn_config, n_mlps=2):
        config = copy.deepcopy(returnn_config)

        for o in ["left", "right"]:
            if 'linear' in config.config['network'][f'{o}-output']['from']:
                for i in range(n_mlps):
                    for k in ['leftContext', 'triphone']:
                        del config.config['network'][f'linear{i+1}-{k}']
            del config.config['network'][f'{o}-output']

        return config

    def delete_mlps(self, returnn_config, keys=None, source=None):
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

    def set_local_flf_tool_for_decoding(self, path=None):
        self.csp["base"].flf_tool_exe = path

    # --------------------- Init procedure -----------------
    def init_system(self, label_info_additional_args=None):

        if self.native_lstm2_path is None:
            self._set_native_lstm_path()

        if label_info_additional_args is not None:
            for k, v in label_info_additional_args.items():
                setattr(self.label_info, k, v)

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

        assert len(self.train_corpora) < 2, 'you can have only one corpus for training'

    def update_am_setting_for_all_crps(self, train_tdp_type, eval_tdp_type, add_base_allophones=False):
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

    def set_rasr_returnn_input_datas(self, input_key: str, is_cv_separate_from_train=False):
        for k in self.corpora.keys():
            assert self.inputs[k] is not None
            assert self.inputs[k][input_key] is not None

        if is_cv_separate_from_train:
            f = self.prepare_rasr_train_data_with_separate_cv
        else: f = self.prepare_rasr_train_data_with_cv_from_train

        nn_train_data_inputs, \
        nn_cv_data_inputs, \
        nn_devtrain_data_inputs = f(input_key)

        nn_dev_data_inputs = {
            self.crp_names['dev']: self.inputs[self.crp_names['dev']][input_key].as_returnn_rasr_data_input(),
        }
        nn_test_data_inputs = {

        self.crp_names['test']: self.inputs[self.crp_names['test']][input_key].as_returnn_rasr_data_input(),

        }
        self._init_datasets(
            train_data=nn_train_data_inputs,
            cv_data=nn_cv_data_inputs,
            devtrain_data=nn_devtrain_data_inputs,
            dev_data=nn_dev_data_inputs,
            test_data=nn_test_data_inputs)

        label_info_args = {
            'n_states_per_phone': self.label_info.n_states_per_phone,
            'n_contexts': self.label_info.n_contexts}
        self.initial_nn_args.update(label_info_args)

    # ----- data preparation for train-----------------------------------------------------
    def prepare_rasr_train_data_with_cv_from_train(self, input_key: str, cv_num_segments: int=100):
        #from i6_experiments.common.datasets.librispeech.constants import num_segments
        #ToDo: decide how you want to set the number of segments
        print("WARNING: hardcoded number of segments")

        key = train_key if train_key is not None else self.train_key
        train_corpus_path = self.corpora[key].corpus_file

        all_segments = self._get_segment_file(corpus_path=train_corpus_path)

        assert self.train_key in num_segments, 'It seems that you set a wrong train key in inputs step'
        cv_size = cv_num_segments / self.num_segments[self.train_key]

        splitted_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
            all_segments, {"train": 1 - cv_size, "cv": cv_size}
        )
        train_segments = splitted_segments_job.out_segments["train"]
        cv_segments = splitted_segments_job.out_segments["cv"]
        devtrain_segments = text.TailJob(
            train_segments, num_lines=100, zip_output=False
        ).out

        # ******************** NN Init ********************
        nn_train_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(shuffling_parameters=self.shuffling_params)
        nn_train_data.update_crp_with(segment_path=train_segments, concurrent=1)
        nn_train_data_inputs = {self.crp_names['train']: nn_train_data}

        nn_cv_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input()
        nn_cv_data.update_crp_with(segment_path=cv_segments, concurrent=1)
        nn_cv_data_inputs = {self.crp_names['cvtrain']: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input()
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names['devtrain']: nn_devtrain_data}

        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs


    def prepare_rasr_train_data_with_separate_cv(self, input_key, cv_corpus_key='dev-other', configure_rasr_automaton=False):
        train_corpus_key = self.train_key
        self.input_key = input_key

        train_corpus = self.corpora[train_corpus_key].corpus_file
        cv_corpus    = self.corpora[cv_corpus_key].corpus_file

        merged_name = 'loss-corpus'
        merged_corpus = self._get_merged_corpus_for_train(train_corpus=train_corpus,
                                                          cv_corpus=cv_corpus,
                                                          name=merged_name)

        # segments
        train_segments = self._get_segment_file(train_corpus)
        cv_segments    = self._get_segment_file(cv_corpus)
        merged_segments = self._get_segment_file(merged_corpus, remove_prefix=merged_name)

        devtrain_segments = text.TailJob(
            train_segments, num_lines=1000, zip_output=False
        ).out

        if configure_rasr_automaton:
            crp_bw = copy.deepcopy(copy.deepcopy(self.crp[self.train_key]))
            crp_bw.corpus_config.file = merged_corpus
            crp_bw.corpus_config.segment.file = merged_segments
            crp_bw.corpus_config.remove_corpus_name_prefix = f'{merged_name}/'
            self.crp_names['bw'] = f'{self.train_key}.bw'
            self.crp[self.crp_names['bw']] = crp_bw


        nn_train_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input(
            shuffling_parameters=self.shuffling_params,
            returnn_rasr_training_args=ReturnnRasrTrainingArgs(partition_epochs=self.partition_epochs['train']),
        )
        nn_train_data.update_crp_with(corpus_file=train_corpus, segment_path=train_segments, concurrent=1)
        nn_train_data_inputs = {self.crp_names['train']: nn_train_data}

        nn_cv_data = self.inputs[cv_corpus_key][input_key].as_returnn_rasr_data_input(
            returnn_rasr_training_args=ReturnnRasrTrainingArgs(partition_epochs=self.partition_epochs['cv']))
        nn_cv_data.update_crp_with(corpus_file=cv_corpus, segment_path=cv_segments, concurrent=1)
        nn_cv_data_inputs = {self.crp_names['cvtrain']: nn_cv_data}

        nn_devtrain_data = self.inputs[self.train_key][input_key].as_returnn_rasr_data_input()
        nn_devtrain_data.update_crp_with(segment_path=devtrain_segments, concurrent=1)
        nn_devtrain_data_inputs = {self.crp_names['devtrain']: nn_devtrain_data}


        return nn_train_data_inputs, nn_cv_data_inputs, nn_devtrain_data_inputs

    # -------------------------------------------- Training --------------------------------------------------------
    def set_standard_prolog_and_epilog_to_config(self, config: Dict, prolog_additional_str: str=None, epilog_additional_str: str=None):
        #this is not a returnn config, but the dict params
        assert self.initial_nn_args["num_input"] is not None, "set the feature input dimension"
        time_prolog, time_tag_name = train_helpers.returnn_time_tag.get_shared_time_tag()

        config["extern_data"] = {
            "data": {
                "dim": self.initial_nn_args["num_input"],
                "same_dim_tags_as": {"T": returnn.CodeWrapper(time_tag_name)},
            },
            **extern_data.get_extern_data_config(label_info=self.label_info, time_tag_name=time_tag_name),
        }
        #these two are gonna get popped and stored during returnn config object creation
        config["python_prolog"] = {"numpy": "import numpy as np",
                                   "time": time_prolog}
        config["python_epilog"] = {
            "functions": [
                sa_mask,
                sa_random_mask,
                sa_summary,
                sa_transform,
            ],
        }

        if prolog_additional_str is not None:
            config["python_prolog"]["str"] = prolog_additional_str

        if epilog_additional_str is not None:
            config["python_epilog"]["str"] = epilog_additional_str


        return config

    #-------------encoder architectures -------------------------------
    def get_blstm_network(self, **kwargs):
        #this is without any loss and output layers
        network = blstm_setup.blstm_network(**kwargs)
        if self.training_criterion != TrainingCriterion.fullsum:
            network = augment_net_with_label_pops(network, label_info=self.label_info)

        return network

    def get_conformer_network(self, chunking, conf_model_dim, aux_loss_args):
        # this only includes auxilaury losses
        network_builder = conformer_setup.get_best_conformer_network(
            conf_model_dim,
            chunking=chunking,
            focal_loss_factor=aux_loss_args["focal_loss_factor"],
            label_smoothing=aux_loss_args["label_smoothing"],
            num_classes=s.label_info.get_n_of_dense_classes(),
        )
        network = network_builder.network
        if self.training_criterion != TrainingCriterion.fullsum:
            network = augment_net_with_label_pops(network, label_info=s.label_info)
        return network

    #-------------------------------------------------------------------------
    def set_returnn_config_for_experiment(self, key: str, config_dict: Dict):
        assert key in self.experiments.keys()
        assert num_epochs in self.initial_nn_args, "set the number of epochs in the nn args"
        python_prolog = config_dict.pop('python_prolog') if 'python_prolog' in config_dict else None
        python_epilog = config_dict.pop('python_epilog') if 'python_epilog' in config_dict else None

        base_post_config = {
            "cleanup_old_models": {
                "keep_best_n": 3,
                "keep": self.initial_nn_args["keep_epochs"],
            },
        }
        returnn_config = returnn.ReturnnConfig(
            config=config_dict,
            post_config=base_post_config,
            hash_full_python_code=True,
            python_prolog=python_prolog,
            python_epilog=python_epilog,
        )
        self.experiments[key]["returnn_config"] = returnn_config
        self.experiments[key]["extra_returnn_code"]["prolog"] = returnn_config.python_prolog
        self.experiments[key]["extra_returnn_code"]["epilog"] = returnn_config.python_epilog

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

        if self.inputs[self.train_key][self.input_key][self.crp_names['train']].alignments is None:
            returnn_config = fh_augmenter.add_fast_bw_layer(crp=bw_crp,
                                                             returnn_config=returnn_config,
                                                             log_linear_scales=self.fullsum_log_linear_scales,
                                                             )

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

        embed()

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
    def get_hdf_path(self, hdf_key: Optional[str]):
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

        return hdf_job

    # -------------------- Decoding --------------------
    def _compute_returnn_rasr_priors(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: returnn.ReturnnConfig,
        share: float,
        time_rqmt: Optional[int] = None,
    ):
        self.set_graph_for_experiment(key)

        model_checkpoint = self._get_model_checkpoint(self.experiments[key]["train_job"], epoch)

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

        prior_job = returnn.ReturnnRasrComputePriorJobV2(
            train_crp=train_crp,
            dev_crp=dev_crp,
            model_checkpoint=model_checkpoint,
            feature_flow=feature_flow,
            alignment=alignments,
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            mem_rqmt=12,
            time_rqmt=time_rqmt if time_rqmt is not None else 12,
        )

        return prior_job

    def set_mono_priors_returnn_rasr(
        self,
        key: str,
        epoch: int,
        train_corpus_key: str,
        dev_corpus_key: str,
        returnn_config: Optional[returnn.ReturnnConfig] = None,
        output_layer_name: str = "output",
        data_share: float = 0.1,
    ):
        self.set_graph_for_experiment(key)

        name = f"{self.experiments[key]['name']}/e{epoch}"

        if returnn_config is None:
            returnn_config = self.experiments[key]["returnn_config"]
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        config = copy.deepcopy(returnn_config)
        config.config["forward_output_layer"] = output_layer_name

        job = self._compute_returnn_rasr_priors(
            key,
            epoch,
            train_corpus_key=train_corpus_key,
            dev_corpus_key=dev_corpus_key,
            returnn_config=config,
            share=data_share,
        )

        job.add_alias(f"priors/{name}/c")
        tk.register_output(f"priors/{name}/center-state.xml", job.out_prior_xml_file)

        s.experiments[key]['priors'] = [job.out_prior_xml_file]


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

    def run_decoding_for_cart(self, name, corpus, feature_flow, feature_scorer,
                              tdp_scale=1.0, exit_sil=20.0,
                              norm_pron=True, pron_scale=3.0, lm_scale=10.0,
                              beam=18.0, beam_limit=500000, we_pruning=0.8, we_pruning_limit=10000, altas=None,
                              only_lm_opt=True):

        pre_path = 'grid' if (altas is not None and beam < 16.0) else 'decoding'

        search_crp = copy.deepcopy(self.crp[corpus])
        search_crp.acoustic_model_config.tdp.scale = tdp_scale
        search_crp.acoustic_model_config.tdp["silence"]["exit"] = exit_sil

        #lm
        search_crp.language_model_config.scale = lm_scale

        name += f'-{corpus}-beaminfo{beam}-{beam_limit}-{we_pruning}'
        name += f'-lmScale-{lm_scale}'
        if tdp_scale != 1.0:
            name+= f'_tdpscale-{tdp_scale}'
        if exit_sil != 20.0:
            name += f'_exitSil-{tdp_scale}'

        if altas is not None:
            name += f'_altas-{altas}'
        sp = {
            "beam-pruning": beam,
            "beam-pruning-limit": beam_limit,
            "word-end-pruning": we_pruning,
            "word-end-pruning-limit": we_pruning_limit,
        }

        adv_search_extra_config = None
        if altas is not None:
            adv_search_extra_config = rasr.RasrConfig()
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                altas
            )

        modelCombinationConfig = None
        if norm_pron:
            modelCombinationConfig = rasr.RasrConfig()
            modelCombinationConfig.pronunciation_scale = pron_scale

        search = recog.AdvancedTreeSearchJob(
            crp=search_crp,
            feature_flow=feature_flow,
            feature_scorer=feature_scorer,
            search_parameters=sp,
            lm_lookahead=True,
            eval_best_in_lattice=True,
            use_gpu=True,
            rtf=12,
            mem=8,
            model_combination_config=modelCombinationConfig,
            model_combination_post_config=None,
            extra_config=adv_search_extra_config,
            extra_post_config=None,
        )
        search.rqmt["cpu"] = 2
        if corpus == 'russian':
            search.rqmt["time"] = 1

        search.add_alias(f"{pre_path}/recog_{name}")


        lat2ctm_extra_config = rasr.RasrConfig()
        lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = "best"
        lat2ctm = recog.LatticeToCtmJob(
            crp=search_crp,
            lattice_cache=search.out_lattice_bundle,
            parallelize=True,
            fill_empty_segments=True,
            best_path_algo="bellman-ford",
            extra_config=lat2ctm_extra_config,
        )

        sKwrgs = copy.deepcopy(self.scorer_args[corpus])
        sKwrgs["sort_files"] = True
        sKwrgs[self.scorer_hyp_arg[corpus]] = lat2ctm.out_ctm_file
        scorer = self.scorers[corpus](**sKwrgs)

        self.jobs[corpus]["scorer_%s" % name] = scorer
        tk.register_output(f"{pre_path}/{name}.reports", scorer.out_report_dir)

        if beam > 15.0 and altas is None:
            opt = recog.OptimizeAMandLMScaleJob(
                crp=search_crp,
                lattice_cache=search.out_lattice_bundle,
                initial_am_scale=pron_scale,
                initial_lm_scale=lm_scale,
                scorer_cls=recog.ScliteJob,
                scorer_kwargs=sKwrgs,
                opt_only_lm_scale=only_lm_opt,
            )
            tk.register_output(f"optLM/{name}.onlyLmOpt{only_lm_opt}.optlm.txt", opt.out_log_file)

    # -------------------- run setup  --------------------

    def run(self, steps: RasrSteps):
        if "init" in steps.get_step_names_as_list():
            init_args = steps.get_args_via_idx(0)
            self.init_system(label_info_additional_args=init_args)
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
                step_args[self.nn_feature_type]['prefix'] = 'features/'
                for all_c in (
                        self.train_corpora
                        + self.dev_corpora
                        + self.test_corpora
                ):
                    self.feature_caches[all_c] = {}
                    self.feature_bundles[all_c] = {}
                    self.feature_flows[all_c] = {}

                self.extract_features(step_args)
            #-----------Set alignments if needed-------
            # here you might one to align cv with a given aligner
            if step_name.startswith("alignment"):
                #step_args here is a dict that has the keys as corpora
                for c in step_args.keys():
                    self.alignments[c] = step_args[c]
            # ---------------Step Input ----------
            if step_name.startswith("input"):
                if not len(step_args.extract_features):
                    add_feature_to_extract(self.nn_feature_type)

                self.run_input_step(step_args)




