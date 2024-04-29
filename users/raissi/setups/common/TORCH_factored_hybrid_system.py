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
    ReturnnRasrDataInput,
)

import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers
import i6_experiments.users.raissi.setups.common.util.hdf.helpers as hdf_helpers

from i6_experiments.users.raissi.args.system.data import HDFBuilderArgs

from i6_experiments.users.raissi.setups.common.BASE_factored_hybrid_system import BASEFactoredHybridSystem

from i6_experiments.users.raissi.setups.common.data.backend import Backend, BackendInfo
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import (
    TrainingCriterion,
    InputKey
)
from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo
from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorInfo,
    PosteriorScales,
    SearchParameters
)
from i6_experiments.users.raissi.setups.common.util.hdf.helpers import HDFAlignmentData


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
            rasr_init_args=rasr_init_args,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            initial_nn_args=initial_nn_args,
        )

        self.backend_info = BackendInfo(train=Backend.TORCH, decode=Backend.ONNX)
        self.hdf_builder_args = asdict(
            HDFBuilderArgs(returnn_root=self.returnn_root, returnn_python_exe=self.returnn_python_exe)
        )

    def set_hdf_data_for_returnn_training(self, input_data_mapping: Dict = None):
        """
        input_data_mapping is a mapping that has the key that should be used from the {train,cv}_input_data
        as key
        """
        if input_data_mapping is None:
            input_data_mapping = {
                self.crp_names['train']: self.train_input_data,
                self.crp_names['cvtrain']: self.cv_input_data
            }

        if self.training_criterion == TrainingCriterion.VITERBI:
            builder = hdf_helpers.build_feature_alignment_meta_dataset_config
            for k, input_data in input_data_mapping.items():
                alignments = HDFAlignmentData(
                    alignment_cache_bundle=input_data[k].alignments,
                    allophone_file=StoreAllophonesJob(input_data[k].crp).out_allophone_file,
                    state_tying_file=DumpStateTyingJob(input_data[k].crp).out_state_tying,
                )

                input_data['hdf'] = builder(
                    data_inputs=[input_data[k]],
                    feature_info=self.feature_info,
                    alignments=[alignments],
                    **self.hdf_builder_args,
                )

        elif self.training_criterion == TrainingCriterion.FULLSUM:
            builder = hdf_helpers.build_feature_label_meta_dataset_config

        else:
            raise NotImplementedError("Only Viterbi and Fullsum training criteria for data preparation")

