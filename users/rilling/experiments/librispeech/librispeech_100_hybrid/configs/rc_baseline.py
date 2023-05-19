import copy
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any

from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs

from i6_experiments.common.setups.returnn_common.serialization import (
    DataInitArgs,
    DimInitArgs,
    Collection,
    Network,
    ExternData,
    Import,
    ExplicitHash
)


from ..default_tools import RETURNN_COMMON


def get_nn_args(num_outputs: int = 12001, num_epochs: int = 250, use_rasr_returnn_training=True, **net_kwargs):
    evaluation_epochs  = list(range(num_epochs, num_epochs + 1, 10))

    returnn_configs = get_rc_returnn_configs(
        num_inputs=50, num_outputs=num_outputs, batch_size=5000,
        evaluation_epochs=evaluation_epochs,
    )

    returnn_recog_configs = get_rc_returnn_configs(
        num_inputs=50, num_outputs=num_outputs, batch_size=5000,
        evaluation_epochs=evaluation_epochs,
        recognition=True,
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
            "mem": 8,
            "lmgc_mem": 16,
            "cpu": 4,
            "parallelize_conversion": True,
            "use_epoch_for_compile": True,
            "native_ops": ["NativeLstm2"],
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


def get_default_data_init_args(num_inputs: int, num_outputs: int):
    """
    default for this hybrid model

    :param num_inputs:
    :param num_outputs:
    :return:
    """
    time_dim = DimInitArgs("data_time", dim=None)
    data_feature = DimInitArgs("data_feature", dim=num_inputs, is_feature=True)
    classes_feature = DimInitArgs("classes_feature", dim=num_outputs, is_feature=True)

    return [
        DataInitArgs(name="data", available_for_inference=True, dim_tags=[time_dim, data_feature], sparse_dim=None),
        DataInitArgs(name="classes", available_for_inference=False, dim_tags=[time_dim], sparse_dim=classes_feature)
    ]



def get_rc_returnn_configs(
        num_inputs: int, num_outputs: int, batch_size: int, evaluation_epochs: List[int],
        recognition=False
):
    # ******************** blstm base ********************

    base_config = {
    }
    base_post_config = {
        "use_tensorflow": True,
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

    rc_extern_data = ExternData(extern_data=get_default_data_init_args(num_inputs=num_inputs, num_outputs=num_outputs))

    # those are hashed
    rc_package = "i6_experiments.users.rossenbach.experiments.librispeech.librispeech_100_hybrid.rc_networks"
    rc_construction_code = Import(rc_package + ".default_hybrid_v2.construct_hybrid_network")


    net_kwargs = {
       "train": not recognition,
       "num_layers": 8,
       "size": 1024,
       "dropout": 0.0,
       "specaugment_options": {
           "min_frame_masks": 1,
           "mask_each_n_frames": 100,
           "max_frames_per_mask": 20,
           "min_feature_masks": 1,
           "max_feature_masks": 2,
           "max_features_per_mask": 10
       }
    }

    net_kwargs_focal_loss = copy.deepcopy(net_kwargs)
    net_kwargs_focal_loss["focal_loss_scale"] = 2.0

    def construct_from_net_kwargs(base_config, net_kwargs, explicit_hash=None):
        rc_network = Network(
            net_func_name=rc_construction_code.object_name,
            net_func_map={
                "audio_data": "data",  # name of the constructor parameter vs name of the data object in RETURNN
                "label_data": "classes"
            },  # this is hashed
            net_kwargs=net_kwargs,
        )
        serializer_objects = [
            rc_extern_data,
            rc_construction_code,
            rc_network,
        ]
        if explicit_hash:
            serializer_objects.append(ExplicitHash(explicit_hash))
        serializer = Collection(
            serializer_objects=serializer_objects,
            returnn_common_root=RETURNN_COMMON,
            make_local_package_copy=True,
            packages={
                rc_package,
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
        "blstm_oclr_v1": construct_from_net_kwargs(blstm_base_config, net_kwargs),
        "blstm_oclr_v1_focal_loss": construct_from_net_kwargs(blstm_base_config, net_kwargs_focal_loss),
    }
