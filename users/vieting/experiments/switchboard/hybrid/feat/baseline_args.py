import copy
import numpy as np
from typing import List, Dict, Any, Optional

from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs
from i6_experiments.common.setups.returnn_common.serialization import (
    DataInitArgs,
    DimInitArgs,
    Collection,
    Network,
    ExternData,
    Import,
)
from .network_helpers.features import GammatoneNetwork, ScfNetwork
from .specaug_jingjing import (
    specaug_layer_jingjing,
    get_funcs_jingjing,
)

RECUSRION_LIMIT = """
import sys
sys.setrecursionlimit(3000)
"""


def get_nn_args(nn_base_args, num_epochs, evaluation_epochs=None, prefix=""):
    evaluation_epochs = evaluation_epochs or [num_epochs]

    returnn_configs = {}
    returnn_recog_configs = {}

    for name, args in nn_base_args.items():
        returnn_config, returnn_recog_config = get_nn_args_single(
            num_epochs=num_epochs, evaluation_epochs=evaluation_epochs, **args)
        returnn_configs[prefix + name] = returnn_config
        returnn_recog_configs[prefix + name] = returnn_recog_config

    training_args = {
        "log_verbosity": 4,
        "num_epochs": num_epochs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 7,
        "cpu_rqmt": 3,
    }
    recognition_args = {
        "hub5e00": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "samples",
            "prior_scales": [0.7, 0.8, 0.9, 1.0],
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


def get_nn_args_single(
    num_outputs: int = 9001, num_epochs: int = 500, evaluation_epochs: Optional[List[int]] = None, extra_exps=False,
    peak_lr=1e-3, feature_args=None, returnn_args=None,
):
    feature_args = feature_args or {"class": "GammatoneNetwork", "sample_rate": 8000}
    feature_network_class = {"GammatoneNetwork": GammatoneNetwork, "ScfNetwork": ScfNetwork}[feature_args.pop("class")]
    feature_net = feature_network_class(**feature_args).get_as_subnetwork()

    returnn_config = get_returnn_config(
        num_inputs=1,
        num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs,
        extra_exps=extra_exps,
        peak_lr=peak_lr,
        num_epochs=num_epochs,
        feature_net=feature_net,
        **(returnn_args or {}),
    )

    returnn_recog_config = get_returnn_config(
        num_inputs=1,
        num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs,
        recognition=True,
        extra_exps=extra_exps,
        peak_lr=peak_lr,
        num_epochs=num_epochs,
        feature_net=feature_net,
    )

    return returnn_config, returnn_recog_config



def fix_network_for_sparse_output(net):
    net = copy.deepcopy(net)
    net.update({
        "classes_int": {"class": "cast", "dtype": "int16", "from": "data:classes"},
        "classes_squeeze": {"class": "squeeze", "axis": "F", "from": "classes_int"},
        "classes_sparse": {
            "class": "reinterpret_data", "from": "classes_squeeze", "set_sparse": True, "set_sparse_dim": 9001},
    })
    for layer in net:
        if net[layer].get("target", None) == "classes":
            net[layer]["target"] = "layer:classes_sparse"
        if net[layer].get("size_base", None) == "data:classes":
            net[layer]["size_base"] = "classes_sparse"
    return net


def get_returnn_config(
    num_inputs: int,
    num_outputs: int,
    evaluation_epochs: List[int],
    peak_lr: float,
    num_epochs: int,
    feature_net: Dict[str, Any],
    batch_size: int = 10000,
    sample_rate: int = 8000,
    recognition: bool = False,
    extra_exps: bool = False,
):
    base_config = {
        "extern_data": {
            "data": {"dim": num_inputs},
            "classes": {"dim": 1, "dtype": "int16"},
            # "classes": {"dim": num_outputs, "sparse": True},  # alignment stored as data with F dim
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

    from .network_helpers.reduced_dim import network
    network = copy.deepcopy(network)
    network["features"] = feature_net
    if recognition:
        for layer in list(network.keys()):
            if "aux" in layer:
                network.pop(layer)
        network["source"] = {"class": "copy", "from": "features"}
    else:
        network["source"] = specaug_layer_jingjing(in_layer=["features"])
        network = fix_network_for_sparse_output(network)

    prolog = get_funcs_jingjing()
    conformer_base_config = copy.deepcopy(base_config)
    conformer_base_config.update(
        {
            "network": network,
            "batch_size": {"classes": batch_size, "data": batch_size * sample_rate // 100},
            "chunking": (
                {"classes": 500, "data": 500 * sample_rate // 100},
                {"classes": 250, "data": 250 * sample_rate // 100},
            ),
            "min_chunk_size": {"classes": 10, "data": 10 * sample_rate // 100},
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.0,
            "learning_rates": list(np.linspace(peak_lr / 10, peak_lr, 100))
            + list(np.linspace(peak_lr, peak_lr / 10, 100))
            + list(np.linspace(peak_lr / 10, 1e-8, 60)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            # "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 3,
            "newbob_multi_update_interval": 1,
        }
    )

    def make_returnn_config(
        config,
        python_prolog,
        staged_network_dict=None,
    ):
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

    conformer_base_returnn_config = make_returnn_config(
        conformer_base_config,
        staged_network_dict=None,
        python_prolog=prolog,
    )

    return conformer_base_returnn_config
