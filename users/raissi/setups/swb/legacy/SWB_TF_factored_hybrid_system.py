__all__ = ["SWBTFFactoredHybridBaseSystem"]

import copy
import dataclasses
import itertools
import sys
from IPython import embed

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

# common modules
from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
)

from i6_experiments.users.berger.network.helpers.conformer_wei import add_initial_conv, add_conformer_stack

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    TrainingCriterion,
    SingleSoftmaxType,
)

from i6_experiments.users.raissi.setups.common.TF_factored_hybrid_system import (
    TFFactoredHybridBaseSystem,
    ExtraReturnnCode,
    Graphs,
    ExtraReturnnCode,
    TFExperiment,
)


import i6_experiments.users.raissi.setups.common.encoder as encoder_archs
import i6_experiments.users.raissi.setups.common.helpers.network as net_helpers
import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers
import i6_experiments.users.raissi.setups.common.helpers.decode as decode_helpers


# user based modules
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import (
    get_lexicon_args,
    get_tdp_values,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)


from i6_experiments.users.raissi.setups.common.helpers.priors import (
    get_returnn_config_for_center_state_prior_estimation,
    get_returnn_config_for_left_context_prior_estimation,
    get_returnn_configs_for_right_context_prior_estimation,
    smoothen_priors,
    JoinRightContextPriorsJob,
    ReshapeCenterStatePriorsJob,
)

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    RasrFeatureScorer,
)

from i6_experiments.users.raissi.setups.swb.legacy.decoder.SWB_factored_hybrid_search import (
    SWBFactoredHybridDecoder,
)

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    BASEFactoredHybridAligner,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorInfo,
    PriorConfig,
    PosteriorScales,
    SearchParameters,
    AlignmentParameters,
)

from i6_experiments.users.raissi.experiments.swb.legacy.data_preparation.legacy_constants_and_paths_swb1 import (
    feature_bundles,
    concurrent,
)


# -------------------- Init --------------------

Path = tk.setup_path(__package__)


# -------------------- Systems --------------------
class SWBTFFactoredHybridBaseSystem(TFFactoredHybridBaseSystem):
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
        initial_nn_args: Dict = None,
    ):
        super().__init__(
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
            rasr_binary_path=rasr_binary_path,
            rasr_init_args=rasr_init_args,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            initial_nn_args=initial_nn_args,
        )

        self.dependencies_path = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies"
        self.recognizers = {"base": SWBFactoredHybridDecoder}
        self.legacy_stm_files = {
            "hub500": tk.Path(f"{self.dependencies_path}/stm-files/hub5e_00.2.stm"),
            "hub501": tk.Path(f"{self.dependencies_path}/stm-files/hub5e_01.2.stm"),
        }
        self.glm_files = dict(zip(["hub500", "hub501"], [tk.Path("/u/corpora/speech/hub5e_00/xml/glm")] * 2))

        self.segments_to_exclude = ["switchboard-1/sw02986A/sw2986A-ms98-a-0013",
                                    "switchboard-1/sw02663A/sw2663A-ms98-a-0022",
                                    "switchboard-1/sw02691A/sw2691A-ms98-a-0017",
                                    "switchboard-1/sw04091A/sw4091A-ms98-a-0063",
                                    "switchboard-1/sw04103A/sw4103A-ms98-a-0022",
                                    "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
                                    "switchboard-1/sw04318A/sw4318A-ms98-a-0024",
                                    'switchboard-1/sw02691A/sw2691A-ms98-a-0017',
                                    'switchboard-1/sw03266B/sw3266B-ms98-a-0055',
                                    'switchboard-1/sw04103A/sw4103A-ms98-a-0022',
                                    'switchboard-1/sw04181A/sw4181A-ms98-a-0036',
                                    'switchboard-1/sw04318A/sw4318A-ms98-a-0024',
                                    'switchboard-1/sw04624A/sw4624A-ms98-a-0055'
                                    "hub5e_00/en_6189a/36",
                                    "hub5e_00/en_4852b/77",
                                    "hub5e_00/en_6189b/66",
                                    ]

        self.cross_validation_info = {
            "pre_path": ("/").join([self.dependencies_path, "cv-from-hub5-00"]),
            "merged_corpus_path": ("/").join(["merged_corpora", "train-dev.corpus.gz"]),
            "merged_corpus_segment": ("/").join(["merged_corpora", "segments"]),
            "cleaned_dev_corpus_path": ("/").join(["zhou-files-dev", "hub5_00.corpus.cleaned.gz"]),
            "cleaned_dev_segment_path": ("/").join(["zhou-files-dev", "segments"]),
            "features_path": ("/").join(["features", "gammatones", "FeatureExtraction.Gammatone.pp9W8m2Z8mHU"]),

        }
        self.prior_transcript_estimates = {'monostate': {'state_prior': ('/').join([self.dependencies_path,
                                                                                    'haotian/monostate/monostate.transcript.prior.pickle']),
                                                         'state_EOW_prior': ('/').join([self.dependencies_path,
                                                                                        'haotian/monostate/monostate.we.transcript.prior.pickle']),
                                                         'speech_forward': .125, 'silence_forward': 0.025},
                                           # 1/8 phoneme
                                           'threepartite': {'state_prior': ('/').join([self.dependencies_path,
                                                                                       'haotian/threepartite/threepartite.transcript.prior.pickle']),
                                                            'state_EOW_prior': ('/').join([self.dependencies_path,
                                                                                           'haotian/threepartite/threepartite.we.transcript.prior.pickle']),
                                                            'speech_forward': .350, 'silence_forward': 0.025}
                                           # 1/9 for 3-state, same amount of silence
                                           }

    # -------------------- External helpers --------------------

    def set_gammatone_features(self):
        for corpus_key in ["train", "hub500", "hub501"]:
            self.feature_bundles[corpus_key] = {self.nn_feature_type: feature_bundles[corpus_key]}
            self.feature_flows[corpus_key] = {
                self.nn_feature_type: features.basic_cache_flow(feature_bundles[corpus_key])
            }
            mapping = {"train": "train", "hub500": "dev", "hub501": "eval"}
            cache_pattern = feature_bundles[corpus_key].get_path().split(".bundle")[0]
            caches = [tk.Path(f"{cache_pattern}.{i}") for i in range(1, concurrent[mapping[corpus_key]] + 1)]
            self.feature_caches[corpus_key] = {self.nn_feature_type: caches}

    def set_stm_and_glm(self):
        for corpus in ["hub500", "hub501"]:
            self.scorer_args[corpus] = {
                "ref": self.legacy_stm_files[corpus],
                "glm": self.glm_files[corpus],
            }

    def prepare_data_with_separate_cv_legacy(self, cv_key="train.cvtrain", bw_key="bw"):
        cv_corpus = ("/").join(
            [self.cross_validation_info["pre_path"], self.cross_validation_info["cleaned_dev_corpus_path"]]
        )
        cv_segment = ("/").join(
            [self.cross_validation_info["pre_path"], self.cross_validation_info["cleaned_dev_segment_path"]]
        )
        cv_feature_path = Path(
            ("/").join(
                [
                    self.cross_validation_info["pre_path"],
                    self.cross_validation_info["features_path"],
                    "output",
                    "gt.cache.bundle",
                ]
            )
        )
        self.cv_input_data[cv_key].update_crp_with(corpus_file=cv_corpus, segment_path=cv_segment, concurrent=1)
        self.feature_flows[cv_key] = self.cv_input_data[cv_key].feature_flow = features.basic_cache_flow(
            cv_feature_path
        )

        if self.training_criterion == TrainingCriterion.VITERBI:
            return

        self.crp[self.crp_names[bw_key]].corpus_config.file = ("/").join(
            [self.cross_validation_info["pre_path"], self.cross_validation_info["merged_corpus_path"]]
        )
        self.crp[self.crp_names[bw_key]].corpus_config.segments.file = ("/").join(
            [self.cross_validation_info["pre_path"], self.cross_validation_info["merged_corpus_segment"]]
        )

