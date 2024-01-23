import copy
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Optional

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport, PartialImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection
from .. import PACKAGE

def get_nn_args(num_outputs: int = 9001, num_epochs: int = 250, debug=False, **net_kwargs):
    evaluation_epochs = list(range(num_epochs, num_epochs + 1, 10))

    batch_size = {"classes": 4 * 2000, "data": 4 * 320000}
    chunking = (
        {"classes": 400, "data": 100 * 160},
        {"classes": 50, "data": 50 * 160},
    )
    returnn_configs = get_pytorch_returnn_configs(
        num_inputs=80,
        num_outputs=num_outputs,
        batch_size=batch_size,
        chunking=chunking,
        evaluation_epochs=evaluation_epochs,
        debug=debug,
        num_epochs=num_epochs,
    )

    returnn_recog_configs = get_pytorch_returnn_configs(
        num_inputs=80,
        num_outputs=num_outputs,
        batch_size=batch_size,
        chunking=chunking,
        evaluation_epochs=evaluation_epochs,
        recognition=True,
        debug=debug,
    )

    training_args = ReturnnTrainingJobArgs(
        log_verbosity=5, num_epochs=num_epochs, save_interval=1, keep_epochs=None, time_rqmt=168, mem_rqmt=8, cpu_rqmt=3
    )
    # TODO remove this once merged
    from dataclasses import asdict
    training_args = asdict(training_args)
    del training_args['returnn_python_exe']
    del training_args['returnn_root']

    recognition_args = {
        "dev": {
            "epochs": evaluation_epochs + ["best"],
            "feature_flow_key": "fb",
            "prior_scales": [0.7, 0.9],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0, 7.5, 5.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 15.0,
                "beam-pruning-limit": 10000,
                "word-end-pruning": 0.5,
                "word-end-pruning-limit": 15000,
            },
            "lattice_to_ctm_kwargs": {
                "fill_empty_segments": True,
                "best_path_algo": "bellman-ford",
            },
            "optimize_am_lm_scale": True,
            "rtf": 50,
            "mem": 7,
            "lmgc_mem": 16,
            "cpu": 2,
            "parallelize_conversion": True,
            "needs_features_size": True,
            'nn_prior': True
        },
    }
    test_recognition_args = None

    nn_args = HybridArgs(
        returnn_training_configs=returnn_configs,
        returnn_recognition_configs=returnn_recog_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args


def get_pytorch_returnn_configs(
    num_inputs: int,
    num_outputs: int,
    batch_size: int,
    chunking: str,
    evaluation_epochs: List[int],
    recognition=False,
    debug=False,
    num_epochs=None,
):
    # ******************** blstm base ********************

    base_config = {
        "extern_data": {
            "data": {
                "dim": num_inputs,
                "shape": (None, num_inputs),
                "available_for_inference": True,
            },  # input: 80-dimensional logmel features
            "classes": {
                "dim": num_outputs,
                "shape": (None,),
                "available_for_inference": True,
                "sparse": True,
                "dtype": "int16",
            },
        },
        "behavior_version": 15,
        "batch_size": batch_size,  # {"classes": batch_size, "data": batch_size},
        "chunking": "400:200",
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "gradient_noise": 0.3,  # TODO: this might break setups
        "learning_rates": list(np.linspace(2.5e-5, 3e-4, 50)) + list(np.linspace(3e-4, 2.5e-5, 50)),
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 3,
        "learning_rate_control_relative_error_relative_lr": True,
        # "min_learning_rate": 1e-5,
        # "gradient_clip": 1.0
        "min_seq_length": {"classes": 1},
        "newbob_learning_rate_decay": 0.707,
        "newbob_multi_num_epochs": 40,
        "newbob_multi_update_interval": 1,
    }
    base_post_config = {
        "backend": "torch",
        "log_batch_size": True,
    }
    if not recognition:
        base_post_config["cleanup_old_models"] = {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": evaluation_epochs,
        }

    baseline_config = copy.deepcopy(base_config)
    baseline_config["batch_size"] = 14000
    peak_lr = 7e-4
    if num_epochs is None:
        num_epochs = 250
    learning_rates = (
        list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2))
        + list(np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2))
        + list(np.linspace(peak_lr / 10, 1e-8, 30))
    )
    baseline_config["learning_rates"] = learning_rates

    no_noise_cfg = copy.deepcopy(baseline_config)
    del no_noise_cfg["gradient_noise"]
    no_noise_cfg["behavior_version"] = 19


    # those are hashed
    PACKAGE = __package__
    package = PACKAGE
    pytorch_package = package + ".pytorch_networks"

    def construct_from_net_kwargs(
            base_config,
            net_kwargs,
            explicit_hash: Optional[str] = None,
            models_commit: str = "8f4c36430fc019faec7d7819c099334f1c170c88",
            post_config: Optional[Dict] = None,
            max_seqs: Optional[int] = None,
            grad_acc: Optional[int] = None,
            learning_rate: Optional[float] = None,
            debug=False,
            returnn_commit: Optional[str] = None,
    ):
        base_config = copy.deepcopy(base_config)
        if post_config is None:
            post_config = base_post_config
        if any(x is not None for x in [max_seqs, grad_acc, learning_rate, returnn_commit]):
            if max_seqs is not None:
                base_config["max_seqs"] = max_seqs
            if grad_acc is not None:
                base_config["accum_grad_multiple_step"] = grad_acc
            if learning_rate is not None:
                base_config["learning_rate"] = learning_rate
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
            base_config["batch_size"] = 12000
            base_config["forward_data"] = "train"
            del base_config["min_seq_length"]
            if "chunking" in base_config.keys():
                del base_config["chunking"]

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

    return {
        "baseline": construct_from_net_kwargs(
            baseline_config,
            {
                "model_type": "conformer_baseline",
                "att_heads": 6,
                "conv_kernel_size": 7,
                "pool_1_stride": (3, 1),
                "ff_dim": 1536,
                "upsample_kernel": 3,
                "upsample_stride": 3,
                "upsample_padding": 0,
                "spec_num_time": 20,
                "spec_max_time": 20,
                "spec_num_feat": 5,
                "spec_max_feat": 16,
            },
        ),
        "baseline_old_spec": construct_from_net_kwargs(
            baseline_config,
            {
                "model_type": "conformer_baseline",
                "att_heads": 6,
                "conv_kernel_size": 7,
                "pool_1_stride": (3, 1),
                "ff_dim": 1536,
                "upsample_kernel": 3,
                "upsample_stride": 3,
                "upsample_padding": 0,
                "spec_num_time": 20,
                "spec_max_time": 20,
                "spec_num_feat": 5,
                "spec_max_feat": 16,
                "old_spec": True
            },
        ),
        "baseline_no_noise": construct_from_net_kwargs(
            no_noise_cfg,
            {
                "model_type": "conformer_baseline",
                "att_heads": 6,
                "conv_kernel_size": 7,
                "pool_1_stride": (3, 1),
                "ff_dim": 1536,
                "upsample_kernel": 3,
                "upsample_stride": 3,
                "upsample_padding": 0,
                "spec_num_time": 20,
                "spec_max_time": 20,
                "spec_num_feat": 5,
                "spec_max_feat": 16,
            },
        ),
       "baseline_old_model": construct_from_net_kwargs(
            baseline_config,
            {
                "model_type": "conformer_baseline_old",
                "att_heads": 6,
                "conv_kernel_size": 7,
                "pool_1_stride": (3, 1),
                "ff_dim": 1536,
                "upsample_kernel": 3,
                "upsample_stride": 3,
                "upsample_padding": 0,
                "spec_num_time": 20,
                "spec_max_time": 20,
                "spec_num_feat": 5,
                "spec_max_feat": 16,
            },
        )
    }
