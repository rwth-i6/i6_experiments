__all__ = ["TEDTFFactoredHybridSystem"]

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

Path = tk.setup_path(__package__)

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipe
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog

import i6_experiments.users.raissi.setups.tedlium.decoder as ted_decoder

# --------------------------------------------------------------------------------
from i6_experiments.common.setups.rasr.util import (
    OggZipHdfDataInput,
    RasrInitArgs,
    RasrDataInput,
)

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import (
    SingleSoftmaxType,
)

from i6_experiments.users.raissi.setups.common.TORCH_factored_hybrid_system import (
    TORCHFactoredHybridSystem,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)

from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    RasrFeatureScorer,
)


from i6_experiments.users.raissi.setups.common.features.taxonomy import FeatureType


class TEDTORCHFactoredHybridSystem(TORCHFactoredHybridSystem):
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
        self.recognizers = {"base": ted_decoder.TEDFactoredHybridDecoder}
        self.feature_info = dataclasses.replace(self.feature_info, feature_type=FeatureType.filterbanks)
