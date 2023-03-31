import copy
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any

from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection


from ..default_tools import PACKAGE


def get_nn_args(num_outputs: int = 12001, num_epochs: int = 250, use_rasr_returnn_training=True, debug=False, **net_kwargs):
    evaluation_epochs  = list(range(num_epochs, num_epochs + 1, 10))

    returnn_configs = get_pytorch_returnn_configs(
        num_inputs=50, num_outputs=num_outputs, batch_size=5000,
        evaluation_epochs=evaluation_epochs, debug=debug,
    )

    returnn_recog_configs = get_pytorch_returnn_configs(
        num_inputs=50, num_outputs=num_outputs, batch_size=5000,
        evaluation_epochs=evaluation_epochs,
        recognition=True, debug=debug,
    )


    training_args = {
        "log_verbosity": 5,
        "num_epochs": num_epochs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 8,
        "cpu_rqmt": 3,
    }

    if use_rasr_returnn_training:
        training_args["num_classes"] = num_outputs
        training_args["use_python_control"] = False
        training_args["partition_epochs"] = {"train": 40, "dev": 20}

    recognition_args = {
        "dev-other": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "gt",
            "prior_scales": [0.3],
            "pronunciation_scales": [6.0],
            "lm_scales": [20.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 12.0,
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
        num_inputs: int, num_outputs: int, batch_size: int, evaluation_epochs: List[int],
        recognition=False, debug=False,
):
    # ******************** blstm base ********************

    base_config = {
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
            #"min_learning_rate": 1e-5,
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


    # those are hashed
    pytorch_package =  PACKAGE + ".pytorch_networks"

    def construct_from_net_kwargs(base_config, net_kwargs, explicit_hash=None):
        model_type = net_kwargs.pop("model_type")
        pytorch_model_import = Import(
            PACKAGE + ".pytorch_networks.%s.Model" % model_type
        )
        pytorch_train_step = Import(
            PACKAGE + ".pytorch_networks.%s.train_step" % model_type
        )
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
            pytorch_export = Import(
                PACKAGE + ".pytorch_networks.%s.export" % model_type
            )
            serializer_objects.append(pytorch_export)
        if explicit_hash:
            serializer_objects.append(ExplicitHash(explicit_hash))
        serializer = Collection(
            serializer_objects=serializer_objects,
            make_local_package_copy=not debug,
            packages={
                pytorch_package,
            },
        )

        blstm_base_returnn_config = ReturnnConfig(
            config=base_config,
            post_config=base_post_config,
            python_epilog=[serializer],
            pprint_kwargs={"sort_dicts": False},
        )

        return blstm_base_returnn_config

    return {
        "blstm_oclr_v1": construct_from_net_kwargs(blstm_base_config, {"model_type": "blstm8x1024"}),
        "blstm_oclr_v2": construct_from_net_kwargs(blstm_base_config, {"model_type": "blstm8x1024_more_specaug"}),
    }
