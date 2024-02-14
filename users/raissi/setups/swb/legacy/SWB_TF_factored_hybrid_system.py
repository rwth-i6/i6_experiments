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
        self.stm_files = {
            "hub00": tk.Path(f"{self.dependencies_path}/stm-files/hub5e_00.2.stm"),
            "hub501": tk.Path(f"{self.dependencies_path}/stm-files/hub5e_01.2.stm"),
        }
        self.glm_file = "/u/corpora/speech/hub5e_00/xml/glm"

        for corpus in ["hub00", "hub01"]:
            self.scorer_args[corpus] = {
                "ref": self.stm_files[k],
                "glm": self.glm_file,
            }
        self.set_gammatone_features()

    # -------------------- External helpers --------------------

    def set_gammatone_features(self):
        for corpus_key in ["train", "hub500", "hub501"]:
            self.feature_bundles[corpus_key] = {
                self.nn_feature_type: feature_bundles[corpus_key]
            }
            self.feature_flows[corpus_key] = {
                self.nn_feature_type: features.basic_cache_flow(feature_bundles[corpus_key])
            }

