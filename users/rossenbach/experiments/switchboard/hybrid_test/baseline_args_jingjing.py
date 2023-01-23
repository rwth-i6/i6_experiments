import copy
import numpy as np
from typing import List

from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs

from i6_experiments.common.setups.returnn_common.serialization import (
    DataInitArgs,
    DimInitArgs,
    Collection,
    Network,
    ExternData,
    Import
)

from ...librispeech.librispeech_100_hybrid.specaugment_clean_legacy import specaug_layer, get_funcs


RECUSRION_LIMIT = """
import sys
sys.setrecursionlimit(3000)
"""

def get_nn_args(num_outputs: int = 9001, num_epochs: int = 500, extra_exps=False):
    evaluation_epochs  = list(np.arange(num_epochs, num_epochs + 1, 10))

    returnn_configs = get_returnn_configs_jingjing(
        num_inputs=40, num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs, extra_exps=extra_exps,
    )

    returnn_recog_configs = get_returnn_configs_jingjing(
        num_inputs=40, num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs,
        recognition=True, extra_exps=extra_exps,
    )


    training_args = {
        "log_verbosity": 5,
        "num_epochs": num_epochs,
        "num_classes": num_outputs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 7,
        "cpu_rqmt": 3,
        "partition_epochs": {"train": 6, "dev": 1},
        "use_python_control": False,
    }
    recognition_args = {
        "hub5e00": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "gt",
            "prior_scales": [0.5, 0.6, 0.7, 0.8],
            "pronunciation_scales": [6.0],
            "lm_scales": [10.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 12.0,
                "beam-pruning-limit": 100000,
                "word-end-pruning": 0.5,
                "word-end-pruning-limit": 10000,
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
            "forward_output_layer": "output",
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


def get_returnn_configs_jingjing(
        num_inputs: int, num_outputs: int, evaluation_epochs: List[int], num_epochs:int = 240,
        recognition=False, extra_exps=False
):
    # ******************** blstm base ********************

    base_config = {
        "extern_data": {
            "data": {"dim": num_inputs},
            "classes": {"dim": num_outputs, "sparse": True},
        },
    }
    base_post_config = {
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",

    }
    if not recognition:
        base_post_config["cleanup_old_models"] = {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": evaluation_epochs,
        }


    from .reduced_dim import network
    network_1 = copy.deepcopy(network)
    network_7 = copy.deepcopy(network)
    network_13 = copy.deepcopy(network)
    network_19= copy.deepcopy(network)
    network_25 = copy.deepcopy(network)
    network_1["source"] = specaug_layer(
        in_layer=["data"],
        min_frame_masks=1,
        max_frames_per_mask=100,
        min_feature_masks=1,
        max_feature_masks=2,
        max_features_per_mask=8,
    )
    network_7["source"] = specaug_layer(
        in_layer=["data"],
        min_frame_masks=2,
        max_frames_per_mask=100,
        min_feature_masks=1,
        max_feature_masks=2,
        max_features_per_mask=8,
    )
    network_13["source"] = specaug_layer(
        in_layer=["data"],
        min_frame_masks=2,
        max_frames_per_mask=100,
        min_feature_masks=1,
        max_feature_masks=3,
        max_features_per_mask=8,
    )
    network_19["source"] = specaug_layer(
        in_layer=["data"],
        min_frame_masks=2,
        max_frames_per_mask=50,
        min_feature_masks=1,
        max_feature_masks=3,
        max_features_per_mask=8,
    )
    network_25["source"] = specaug_layer(
        in_layer=["data"],
        min_frame_masks=2,
        max_frames_per_mask=50,
        min_feature_masks=2,
        max_feature_masks=3,
        max_features_per_mask=8,
    )
    networks = {
        1: network_1,
        7: network_7,
        13: network_13,
        19: network_19,
        25: network_25,
    }
    networks_fast = {
        1: network_1,
        7: network_7,
        13: network_13,
    }


    network_jingjing = copy.deepcopy(network)
    from .legacy_specaug_jingjing import specaug_layer_jingjing, get_funcs_jingjing
    network_jingjing["source"] = specaug_layer_jingjing(in_layer=["data"])

    prolog_new = get_funcs()
    prolog_jingjing = get_funcs_jingjing()
    conformer_base_config = copy.deepcopy(base_config)
    conformer_base_config.update(
        {
            "batch_size": 14000,  # {"classes": batch_size, "data": batch_size},
            # "batching": 'sort_bin_shuffle:.64',
            "chunking": "500:250",
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.0,
            "learning_rates": list(np.linspace(2e-3, 2e-2, num_epochs//2)) + list(np.linspace(8e-4, 1e-7, num_epochs//2)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            #"min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 3,
            "newbob_multi_update_interval": 1,
        }
    )
    conformer_jingjing_config = copy.deepcopy(conformer_base_config)
    conformer_jingjing_config["network"] = network_jingjing

    conformer_fast_config = copy.deepcopy(conformer_base_config)
    conformer_fast_config["chunking"] = "250:200"
    conformer_fast_config["optimizer"] = {"class": "adam", "epsilon": 1e-8}
    conformer_fast_config["learning_rates"] =  list(np.linspace(1e-5, 2e-3, num_epochs//2)) + list(np.linspace(2e-3, 1e-7, num_epochs//2))

    conformer_fast2_config = copy.deepcopy(conformer_base_config)
    conformer_fast2_config["chunking"] = "250:200"


    def make_returnn_config(config, python_prolog, staged_network_dict=None, ):
        if recognition:
            rec_network = copy.deepcopy(network)
            rec_network["output"] = {
                "class": "linear",
                "activation": "log_softmax",
                "from": ["MLP_output"],
                "n_out": 9001
                # "is_output_layer": True,
            }
            config["network"] = rec_network
        return ReturnnConfig(
            config=config,
            post_config=base_post_config,
            staged_network_dict=staged_network_dict if not recognition else None,
            hash_full_python_code=True,
            python_prolog=python_prolog if not recognition else None,
            python_epilog=RECUSRION_LIMIT,
            pprint_kwargs={"sort_dicts": False},
        )
    conformer_base_returnn_config = make_returnn_config(conformer_base_config, staged_network_dict=networks, python_prolog=prolog_new)
    conformer_fast_returnn_config = make_returnn_config(conformer_fast_config, staged_network_dict=networks_fast, python_prolog=prolog_new)
    conformer_fast2_returnn_config = make_returnn_config(conformer_fast2_config, staged_network_dict=networks_fast, python_prolog=prolog_new)
    conformer_jingjing_returnn_config = make_returnn_config(conformer_jingjing_config, staged_network_dict=None, python_prolog=prolog_jingjing)

    return {
        "conformer_base": conformer_base_returnn_config,
        "conformer_fast": conformer_fast_returnn_config,
        "conformer_fast2": conformer_fast2_returnn_config,
        "conformer_jingjing": conformer_jingjing_returnn_config,
    }
