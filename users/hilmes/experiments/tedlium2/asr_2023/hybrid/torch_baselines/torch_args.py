import copy
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection


from i6_experiments.common.baselines.tedlium2.default_tools import PACKAGE


def get_nn_args(num_outputs: int = 9001, num_epochs: int = 250, debug=False, **net_kwargs):
    evaluation_epochs = list(range(num_epochs, num_epochs + 1, 10))

    batch_size = {"classes": 4 * 2000, "data": 4 * 320000} # change to 14000 probably more
    chunking = ({"classes": 400, "data": 100 * 160}, {"classes": 50, "data": 50 * 160}) # TODO change to 400:200/500:250
    returnn_configs = get_pytorch_returnn_configs(
        num_inputs=80,
        num_outputs=num_outputs,
        batch_size=batch_size,
        chunking=chunking,
        evaluation_epochs=evaluation_epochs,
        debug=debug,
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

    recognition_args = {
        "dev": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "fb",
            "prior_scales": [0.3, 0.5, 0.7],
            "pronunciation_scales": [0.0, 6.0],
            "lm_scales": [20.0, 15.0, 5.0],
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
            "training_whitelist": ["torch_blstm_baseline", "torch_blstm_baseline2"],
        },
        "dev-conformer": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "fb",
            "prior_scales": [0.3, 0.5, 0.7, 0.8, 0.9],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0, 7.5, 5.0],
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
            "needs_features_size": False,
            "training_whitelist": ["torch_conformer_baseline", "torch_conformer_no_cast"],
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
        }
    }
    base_post_config = {
        "backend": "torch",
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
    }
    blstm_base_config = copy.deepcopy(base_config)
    blstm_base_config.update(
        {
            "behavior_version": 15,
            "batch_size": batch_size,  # {"classes": batch_size, "data": batch_size},
            "chunking": "50:25",
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.3,
            "learning_rates": list(np.linspace(2.5e-5, 3e-4, 50)) + list(np.linspace(3e-4, 2.5e-5, 50)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            # "min_learning_rate": 1e-5,
            "min_seq_length": {"classes": 1},
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

    medium_lchunk_config = copy.deepcopy(blstm_base_config)
    medium_lchunk_config["learning_rates"] = list(np.linspace(8e-5, 8e-4, 50)) + list(np.linspace(8e-4, 8e-5, 50))
    medium_lchunk_config["chunking"] = "250:200"
    medium_lchunk_config["gradient_clip"] = 1.0
    medium_lchunk_config["optimizer"] = {"class": "adam", "epsilon": 1e-8}

    wei_lstm_config = copy.deepcopy(base_config)

    wei_lstm_config.update(
        {
            "behavior_version": 15,
            "batch_size": 10000,  # {"classes": batch_size, "data": batch_size},
            "max_seqs": 128,
            "chunking": "64:32",
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "learning_rate": 0.0009,
            "gradient_clip": 0,
            "gradient_noise": 0.1,  # together with l2 and dropout for overfit
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_relative_error_relative_lr": True,
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            # "min_learning_rate": 1e-5,
            "min_seq_length": {"classes": 1},
            "newbob_multi_num_epochs": 5,
            "newbob_multi_update_interval": 1,
            "newbob_learning_rate_decay": 0.9,
        }
    )

    # those are hashed
    package = PACKAGE + ".hybrid"
    pytorch_package = package + ".pytorch_networks"

    def construct_from_net_kwargs(base_config, net_kwargs, explicit_hash=None):
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
        i6_models_repo = CloneGitRepositoryJob(
            url="https://github.com/rwth-i6/i6_models",
            commit="75e03f37ac74d3d0c7358d29bb9b71dcec1bf120",
            checkout_folder_name="i6_models",
            branch="bene_conf_enc",
        ).out_repository
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
            post_config=base_post_config,
            python_epilog=[serializer],
            pprint_kwargs={"sort_dicts": False},
        )

        return returnn_config

    smaller_config = copy.deepcopy(blstm_base_config)
    smaller_config["batch_size"] = {"classes": 2 * 2000, "data": 2 * 320000}
    smaller_config["accum_grad_multiple_step"] = 2

    return {
        "torch_conformer_baseline": construct_from_net_kwargs(medium_lchunk_config, {"model_type": "conformer"}),
        "torch_conformer_no_cast": construct_from_net_kwargs(medium_lchunk_config, {"model_type": "conformer2"}),
        "torch_conformer_new": construct_from_net_kwargs(medium_lchunk_config, {"model_type": "conformer3"}),
        "torch_blstm_paper_baseline": construct_from_net_kwargs(wei_lstm_config, {"model_type": "blstm"}),
        "torch_blstm_baseline": construct_from_net_kwargs(blstm_base_config, {"model_type": "blstm"}),
        "torch_blstm_baseline2": construct_from_net_kwargs(smaller_config, {"model_type": "blstm"}),
    }
