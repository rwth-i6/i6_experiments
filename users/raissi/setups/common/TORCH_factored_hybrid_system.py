__all__ = ["TORCHFactoredHybridSystem"]

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

# common modules
from i6_experiments.common.setups.rasr.nn_system import NnSystem


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
import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import BASEFactoredHybridSystem

from i6_experiments.users.raissi.setups.common.data.backend import Backend, BackendInfo

# user based modules
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import (
    get_lexicon_args,
    get_tdp_values,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import BASEFactoredHybridSystem

from i6_experiments.users.raissi.setups.common.decoder.config import PriorInfo, PosteriorScales, SearchParameters

from i6_experiments.users.raissi.setups.common.util.hdf import RasrFeaturesToHdf

# -------------------- Init --------------------
Path = tk.setup_path(__package__)

# -------------------- Systems --------------------
class TORCHFactoredHybridSystem(BASEFactoredHybridSystem):
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
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            initial_nn_args=initial_nn_args,
        )

        self.backend_info = BackendInfo(train=Backend.TORCH, decode=Backend.ONNX)
