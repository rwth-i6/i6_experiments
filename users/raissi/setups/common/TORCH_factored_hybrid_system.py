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
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import TrainingCriterion, InputKey
from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo
from i6_experiments.users.raissi.setups.common.decoder.config import PriorInfo, PosteriorScales, SearchParameters
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
                self.crp_names["train"]: self.train_input_data,
                self.crp_names["cvtrain"]: self.cv_input_data,
            }

        if self.training_criterion == TrainingCriterion.VITERBI:
            builder = hdf_helpers.build_feature_alignment_meta_dataset_config
            for k, input_data in input_data_mapping.items():
                alignments = HDFAlignmentData(
                    alignment_cache_bundle=input_data[k].alignments,
                    allophone_file=StoreAllophonesJob(input_data[k].crp).out_allophone_file,
                    state_tying_file=DumpStateTyingJob(input_data[k].crp).out_state_tying,
                )

                input_data["hdf"] = builder(
                    data_inputs=[input_data[k]],
                    feature_info=self.feature_info,
                    alignments=[alignments],
                    **self.hdf_builder_args,
                )

        elif self.training_criterion == TrainingCriterion.FULLSUM:
            builder = hdf_helpers.build_feature_label_meta_dataset_config

        else:
            raise NotImplementedError("Only Viterbi and Fullsum training criteria for data preparation")

    def construct_from_net_kwargs(
        self,
        base_config,
        net_kwargs,
        explicit_hash: Optional[str] = None,
        models_commit: str = "8f4c36430fc019faec7d7819c099334f1c170c88",
        post_config: Optional[Dict] = None,
        grad_acc: Optional[int] = None,
        debug=False,
        returnn_commit: Optional[str] = None,
    ):
        base_config = copy.deepcopy(base_config)
        if grad_acc is not None:
            base_config["accum_grad_multiple_step"] = grad_acc
        if returnn_commit is not None:
            base_config["returnn_commit"] = returnn_commit

        model_type = net_kwargs.pop("model_type")
        pytorch_model_import = Import(package + ".pytorch_networks.%s.Model" % model_type)
        pytorch_train_step = Import(package + ".pytorch_networks.%s.train_step" % model_type)
        pytorch_model = PyTorchModel(
            model_class_name=pytorch_model_import.object_name,
            model_kwargs=net_kwargs,
        )
        serializer_objects = [
            pytorch_model_import,
            pytorch_train_step,
            pytorch_model,
        ]
        if recognition:
            pytorch_export = Import(package + ".pytorch_networks.%s.export" % model_type)
            serializer_objects.append(pytorch_export)

            prior_computation = Import(package + ".pytorch_networks.prior.basic.forward_step")
            serializer_objects.append(prior_computation)
            prior_computation = Import(
                package + ".pytorch_networks.prior.prior_callback.ComputePriorCallback", import_as="forward_callback"
            )
            serializer_objects.append(prior_computation)
            base_config["forward_data"] = "train"

        i6_models_repo = CloneGitRepositoryJob(
            url="https://github.com/rwth-i6/i6_models",
            commit=models_commit,
            checkout_folder_name="i6_models",
        ).out_repository
        if models_commit == "8f4c36430fc019faec7d7819c099334f1c170c88":
            i6_models_repo.hash_overwrite = "TEDLIUM2_DEFAULT_I6_MODELS"
        i6_models = ExternalImport(import_path=i6_models_repo)
        serializer_objects.insert(0, i6_models)
        if explicit_hash:
            serializer_objects.append(ExplicitHash(explicit_hash))
        serializer = Collection(
            serializer_objects=serializer_objects,
            make_local_package_copy=not debug,
            packages={
                pytorch_package,
            },
        )

        returnn_config = ReturnnConfig(
            config=base_config,
            post_config=post_config,
            python_epilog=[serializer],
            pprint_kwargs={"sort_dicts": False},
        )

        return returnn_config

    def get_config_with_prolog_and_epilog(
        self,
        config: Dict,
        prolog_additional_str: str = None,
        epilog_additional_str: str = None,
        label_time_tag: str = None,
        add_extern_data_for_fullsum: bool = False,
    ):
        # this is not a returnn config, but the dict params
        assert self.initial_nn_args["num_input"] is not None, "set the feature input dimension"
        config["extern_data"] = {
            "data": {
                "dim": self.initial_nn_args["num_input"],
                "same_dim_tags_as": {"T": returnn.CodeWrapper(self.frame_rate_reduction_ratio_info.time_tag_name)},
            }
        }
        config["python_prolog"] = {
            "numpy": "import numpy as np",
            "time": self.frame_rate_reduction_ratio_info.get_time_tag_prolog_for_returnn_config(),
        }

        if self.training_criterion != TrainingCriterion.FULLSUM or add_extern_data_for_fullsum:
            if self.frame_rate_reduction_ratio_info.factor == 1:
                label_time_tag = self.frame_rate_reduction_ratio_info.time_tag_name
            config["extern_data"].update(
                **net_helpers.extern_data.get_extern_data_config(
                    label_info=self.label_info,
                    time_tag_name=label_time_tag,
                    add_single_state_label=self.frame_rate_reduction_ratio_info.single_state_alignment,
                )
            )

        if prolog_additional_str is not None:
            config["python_prolog"]["str"] = prolog_additional_str

        if epilog_additional_str is not None:
            config["python_epilog"]["str"] = epilog_additional_str

        return config

    def set_returnn_config_for_experiment(
        self,
        key: str,
        config_dict: Dict,
        net_kwargs,
        explicit_hash: Optional[str] = None,
        models_commit: str = "8f4c36430fc019faec7d7819c099334f1c170c88",
        post_config: Optional[Dict] = None,
        grad_acc: Optional[int] = None,
        debug=False,
        returnn_commit: Optional[str] = None,
    ):

        assert key in self.experiments.keys()
        # if legacy code is passed here
        config_dict.pop("use_tensorflow", None)

        if returnn_commit is not None:
            base_config["returnn_commit"] = returnn_commit

        keep_best_n = (
            config_dict.pop("keep_best_n") if "keep_best_n" in config_dict else self.initial_nn_args["keep_best_n"]
        )
        keep_epochs = (
            config_dict.pop("keep_epochs") if "keep_epochs" in config_dict else self.initial_nn_args["keep_epochs"]
        )
        if None in (keep_best_n, keep_epochs):
            assert False, "either keep_epochs or keep_best_n is None, set this in the initial_nn_args"

        python_prolog = config_dict.pop("python_prolog") if "python_prolog" in config_dict else None
        python_epilog = config_dict.pop("python_epilog") if "python_epilog" in config_dict else None

        base_post_config = {
            "backend": "torch",
            "cleanup_old_models": {
                "keep_best_n": keep_best_n,
                "keep": keep_epochs,
            },
        }

        model_type = net_kwargs.pop("model_type")
        pytorch_model_import = Import(package + ".pytorch_networks.%s.Model" % model_type)
        pytorch_train_step = Import(package + ".pytorch_networks.%s.train_step" % model_type)
        pytorch_model = PyTorchModel(
            model_class_name=pytorch_model_import.object_name,
            model_kwargs=net_kwargs,
        )
        serializer_objects = [
            pytorch_model_import,
            pytorch_train_step,
            pytorch_model,
        ]
        if recognition:
            pytorch_export = Import(package + ".pytorch_networks.%s.export" % model_type)
            serializer_objects.append(pytorch_export)

            prior_computation = Import(package + ".pytorch_networks.prior.basic.forward_step")
            serializer_objects.append(prior_computation)
            prior_computation = Import(
                package + ".pytorch_networks.prior.prior_callback.ComputePriorCallback", import_as="forward_callback"
            )
            serializer_objects.append(prior_computation)
            base_config["batch_size"] = 12000
            base_config["forward_data"] = "train"
            del base_config["min_seq_length"]
            if "chunking" in base_config.keys():
                del base_config["chunking"]

        returnn_config = returnn.ReturnnConfig(
            config=config_dict,
            post_config=base_post_config,
            hash_full_python_code=True,
            python_prolog=python_prolog,
            python_epilog=python_epilog,
            sort_config=self.sort_returnn_config,
        )
        self.experiments[key]["returnn_config"] = returnn_config
        self.experiments[key]["extra_returnn_code"]["prolog"] = returnn_config.python_prolog
        self.experiments[key]["extra_returnn_code"]["epilog"] = returnn_config.python_epilog
