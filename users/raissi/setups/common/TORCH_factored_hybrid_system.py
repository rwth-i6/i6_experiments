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

from i6_experiments.users.raissi.setups.common.util.hdf.hdf import RasrFeaturesToHdf

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



        def get_hdf_data_for_returnn_training(self):

            return

        def returnn_training(
                self,
                experiment_key,
                nn_train_args,
        ):



            """
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
                feature_flow["dev"], alignments["dev"] = self.get_feature_and_alignment_flows_for_training(
                    data=dev_data)

            if self.segments_to_exclude is not None:
                extra_config = (
                    rasr.RasrConfig() if "extra_rasr_config" not in nn_train_args else nn_train_args[
                        "extra_rasr_config"]
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
            )"""


            self._add_output_alias_for_train_job(
                train_job=train_job,
                name=self.experiments[experiment_key]["name"],
            )
            self.experiments[experiment_key]["train_job"] = train_job
            self.set_graph_for_experiment(experiment_key)



