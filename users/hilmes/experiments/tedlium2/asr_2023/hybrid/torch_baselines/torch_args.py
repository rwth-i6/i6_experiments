import copy
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.users.hilmes.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection


# from i6_experiments.common.baselines.tedlium2.default_tools import PACKAGE
from onnxruntime.quantization import CalibrationMethod


def get_nn_args(num_outputs: int = 9001, num_epochs: int = 250, debug=False, **net_kwargs):
    evaluation_epochs = list(range(num_epochs, num_epochs + 1, 10))

    batch_size = 14000
    returnn_configs = get_pytorch_returnn_configs(
        num_inputs=80,
        num_outputs=num_outputs,
        batch_size=batch_size,
        evaluation_epochs=evaluation_epochs,
        debug=debug,
        num_epochs=num_epochs,
    )

    returnn_recog_configs = get_pytorch_returnn_configs(
        num_inputs=80,
        num_outputs=num_outputs,
        batch_size=batch_size,
        evaluation_epochs=evaluation_epochs,
        recognition=True,
        debug=debug,
    )

    training_args = ReturnnTrainingJobArgs(
        log_verbosity=5, num_epochs=num_epochs, save_interval=1, keep_epochs=None, time_rqmt=168, mem_rqmt=8, cpu_rqmt=3
    )

    recognition_args = {
        "dev": {
            "epochs": evaluation_epochs + ["best", "avrg"],
            "feature_flow_key": "fb",
            "prior_scales": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5],
            "pronunciation_scales": [0.0],
            "lm_scales": [20.0, 17.5, 15.0, 12.5, 10.0, 7.5],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 14.0,
                "beam-pruning-limit": 100000,
                "word-end-pruning": 0.5,
                "word-end-pruning-limit": 15000,
            },
            "lattice_to_ctm_kwargs": {"fill_empty_segments": True, "best_path_algo": "bellman-ford",},
            "optimize_am_lm_scale": True,
            "rtf": 50,
            "mem": 6,
            "lmgc_mem": 16,
            "cpu": 2,
            "parallelize_conversion": True,
            "needs_features_size": True,
            "nn_prior": True,
        },
        # "quant": {
        #     "epochs": evaluation_epochs,
        #     "feature_flow_key": "fb",
        #     "prior_scales": [0.9],
        #     "pronunciation_scales": [0.0],
        #     "lm_scales": [10.0],
        #     "lm_lookahead": True,
        #     "lookahead_options": None,
        #     "create_lattice": True,
        #     "eval_single_best": True,
        #     "eval_best_in_lattice": True,
        #     "search_parameters": {
        #         "beam-pruning": 15.0,
        #         "beam-pruning-limit": 10000,
        #         "word-end-pruning": 0.5,
        #         "word-end-pruning-limit": 15000,
        #     },
        #     "lattice_to_ctm_kwargs": {
        #         "fill_empty_segments": True,
        #         "best_path_algo": "bellman-ford",
        #     },
        #     "optimize_am_lm_scale": True,
        #     "rtf": 50,
        #     "mem": 7,
        #     "lmgc_mem": 16,
        #     "cpu": 2,
        #     "parallelize_conversion": True,
        #     "needs_features_size": True,
        #     #"quantize": [10, 15, 25, 100, 250, 500, 750, 1000, 2500, 5000],
        #     "quantize": [5, 10, 25, 100, 500, 1000, 5000, 10000, 50000, 100000],
        #     "quant_modes": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        #     "quant_ops": [["Conv", "MatMul"]],
        #     "quant_sym_modes": [False],
        #     "quant_avg_modes": [False],
        #     "quant_percentiles": [99.999],
        #     "quant_num_bin_ls": [2048],
        #     "training_whitelist": [
        #         "torch_jj_config2",
        #     ],
        # },
    }
    # from .old_torch_recog_args import speed
    # recognition_args.update(speed)
    test_recognition_args = {"dev": {}, "quant-paper": {}}

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
        "behavior_version": 16,
        "batch_size": batch_size,  # {"classes": batch_size, "data": batch_size},
        "gradient_clip": 1.0,
        "min_seq_length": {"classes": 1},
        "optimizer": {"class": "adam", "epsilon": 1e-8},
    }
    base_post_config = {
        "backend": "torch",
        "log_batch_size": True,
    }
    peak_lr = 7e-4
    if num_epochs is None:
        num_epochs = 250
    conformer_base_config = copy.deepcopy(base_config)
    conformer_base_config.update(
        {
            "chunking": "400:200",
            "learning_rates": (
                list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2))
                + list(np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2))
                + list(np.linspace(peak_lr / 10, 1e-8, 30))
            ),
            "newbob_learning_rate_decay": 0.707,
            "newbob_multi_num_epochs": 40,
            "newbob_multi_update_interval": 1,
        }
    )
    if not recognition:
        base_post_config["cleanup_old_models"] = {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": evaluation_epochs,
        }

    conformer_smaller = copy.deepcopy(conformer_base_config)
    conformer_smaller["batch_size"] = 12000

    # those are hashed
    PACKAGE = __package__
    package = PACKAGE
    pytorch_package = package + ".pytorch_networks"

    def construct_from_net_kwargs(
        base_config,
        net_kwargs,
        explicit_hash=None,
        models_commit="75e03f37ac74d3d0c7358d29bb9b71dcec1bf120",
        grad_acc=None,
        random_seed=None,
    ):
        if grad_acc is not None:
            base_config = copy.deepcopy(base_config)
            base_config["accum_grad_multiple_step"] = grad_acc
        if random_seed is not None:
            base_config = copy.deepcopy(base_config)
            base_config["random_seed"] = random_seed
        model_type = net_kwargs.pop("model_type")
        pytorch_model_import = Import(package + ".pytorch_networks.%s.Model" % model_type)
        pytorch_train_step = Import(package + ".pytorch_networks.%s.train_step" % model_type)
        pytorch_model = PyTorchModel(model_class_name=pytorch_model_import.object_name, model_kwargs=net_kwargs,)
        serializer_objects = [
            pytorch_model_import,
            pytorch_train_step,
            pytorch_model,
        ]
        if recognition:
            # pytorch_export = Import(package + ".pytorch_networks.%s.export" % model_type)
            # serializer_objects.append(pytorch_export)
            base_config = copy.deepcopy(base_config)
            prior_computation = Import(package + ".pytorch_networks.prior.basic.forward_step")
            serializer_objects.append(prior_computation)
            prior_computation = Import(
                package + ".pytorch_networks.prior.prior_callback.ComputePriorCallback", import_as="forward_callback"
            )
            serializer_objects.append(prior_computation)
            base_config["max_seqs"] = 7
            base_config["model_outputs"] = {"log_probs": {"dim": num_outputs}}
            base_config["forward_data"] = "train"
            if "min_seq_length" in base_config:
                del base_config["min_seq_length"]

        i6_models_repo = CloneGitRepositoryJob(
            url="https://github.com/rwth-i6/i6_models",
            commit=models_commit,
            checkout_folder_name="i6_models",
            branch="bene_conf_enc" if models_commit == "75e03f37ac74d3d0c7358d29bb9b71dcec1bf120" else None,
        ).out_repository.copy()
        i6_models_repo.hash_overwrite = "TEDLIUM2_DEFAULT_I6_MODELS"
        i6_models = ExternalImport(import_path=i6_models_repo)
        serializer_objects.insert(0, i6_models)
        if explicit_hash:
            serializer_objects.append(ExplicitHash(explicit_hash))
        serializer = Collection(
            serializer_objects=serializer_objects, make_local_package_copy=not debug, packages={pytorch_package,},
        )

        returnn_config = ReturnnConfig(
            config=base_config,
            post_config=base_post_config,
            python_epilog=[serializer],
            pprint_kwargs={"sort_dicts": False},
        )

        return returnn_config

    return {
        "torch_jj_config": construct_from_net_kwargs(
            conformer_base_config,
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
            models_commit="5aa74f878cc0d8d7bbc623a3ced681dcb31955ec",
        ),
        "torch_pos_enc": construct_from_net_kwargs(
            conformer_base_config,
            {
                "model_type": "conformer_pos_enc",
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
                "pos_emb_config": {
                    "learnable_pos_emb": False,
                    "rel_pos_clip": 16,
                    "with_linear_pos": True,
                    "with_pos_bias": True,
                    "separate_pos_emb_per_head": True,
                    "pos_emb_dropout": 0.0,
                },
            },
            models_commit="5aa74f878cc0d8d7bbc623a3ced681dcb31955ec",
        ),
    }
