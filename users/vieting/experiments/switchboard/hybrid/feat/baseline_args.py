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
from i6_experiments.users.vieting.models.tf_networks.features import (
    LogMelNetwork,
    GammatoneNetwork,
    ScfNetwork,
    PreemphasisNetwork,
)
from .specaug_jingjing import (
    specaug_layer_jingjing,
    get_funcs_jingjing,
)
from .specaug_sorted import ( 
    specaug_layer_sorted,
    get_funcs_sorted,
)
from .specaug_time import ( 
    specaug_layer_only_time,
    get_funcs_only_time,
)
from .specaug_random import (
    specaug_layer_random,
    get_funcs_random,   
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
    num_outputs: int = 9001, num_epochs: int = 500, evaluation_epochs: Optional[List[int]] = None,
    peak_lr=1e-3, feature_args=None, returnn_args=None,
):
    feature_args = copy.deepcopy(feature_args) or {"class": "GammatoneNetwork", "sample_rate": 8000}
    preemphasis = feature_args.pop("preemphasis", None)
    wave_norm = feature_args.pop("wave_norm", False)
    feature_network_class = {
        "LogMelNetwork": LogMelNetwork,
        "GammatoneNetwork": GammatoneNetwork,
        "ScfNetwork": ScfNetwork,
    }[feature_args.pop("class")]
    feature_net = feature_network_class(**feature_args).get_as_subnetwork()
    if preemphasis:
        for layer in feature_net["subnetwork"]:
            if feature_net["subnetwork"][layer].get("from", "data") == "data":
                feature_net["subnetwork"][layer]["from"] = "preemphasis"
        feature_net["subnetwork"]["preemphasis"] = PreemphasisNetwork(alpha=preemphasis).get_as_subnetwork()
    if wave_norm:
        for layer in feature_net["subnetwork"]:
            if feature_net["subnetwork"][layer].get("from", "data") == "data":
                if feature_net["subnetwork"][layer].get("class", None) != "variable":
                    feature_net["subnetwork"][layer]["from"] = "wave_norm"
        feature_net["subnetwork"]["wave_norm"] = {"axes": "T", "class": "norm", "from": "data"}

    returnn_config = get_returnn_config(
        num_inputs=1,
        num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs,
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
    extra_args: Optional[Dict[str, Any]] = None,
    staged_opts: Optional[Dict[int, Any]] = None,
    enable_specaug: bool = True,
    specaug_mask_sorting: bool = False,
    specaug_after_first_layer: bool = False,
    specaug_time_only: bool = False,
    specaug_shuffled: bool = False,
    mask_divisor: int = None,
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
    network["features"] = copy.deepcopy(feature_net)
    prolog = None
    if recognition:
        for layer in list(network.keys()):
            if "aux" in layer:
                network.pop(layer)
        network["source"] = {"class": "copy", "from": "features"}
    elif enable_specaug:
        if specaug_mask_sorting:
            assert not specaug_shuffled, "shuffling cannot be combined with sorting!"
            assert specaug_after_first_layer, "Sorted specaug is only possible after the first layer!"
            network["features"]["subnetwork"]["specaug"] = specaug_layer_sorted(in_layer=["conv_h_act"], mask_divisor=mask_divisor)
            network["features"]["subnetwork"]["conv_h_split"]["from"] = "specaug"
            network["source"] = {"class": "copy", "from": "features"}
            prolog = get_funcs_sorted()
        else:
            if specaug_after_first_layer:
                assert not specaug_shuffled, "shuffling can only be done on the second layer"
                if specaug_time_only:
                    network["features"]["subnetwork"]["specaug"] = specaug_layer_only_time(in_layer=["conv_h_act"])
                    network["features"]["subnetwork"]["conv_h_split"]["from"] = "specaug"
                    network["source"] = {"class": "copy", "from": "features"}
                    prolog = get_funcs_only_time()
                else:
                    network["features"]["subnetwork"]["specaug"] = specaug_layer_jingjing(in_layer=["conv_h_act"])
                    network["features"]["subnetwork"]["conv_h_split"]["from"] = "specaug"
                    network["source"] = {"class": "copy", "from": "features"}
                    prolog = get_funcs_jingjing()
            else:
                if specaug_time_only:
                    network["source"] = specaug_layer_only_time(in_layer=["features"])
                    prolog = get_funcs_only_time()
                else:
                    if specaug_shuffled:
                        network["source"] = specaug_layer_random(in_layer=["features"])
                        prolog = get_funcs_random()
                    else:
                        network["source"] = specaug_layer_jingjing(in_layer=["features"])
                        prolog = get_funcs_jingjing()

        network = fix_network_for_sparse_output(network)
    else:
        assert (
            not specaug_mask_sorting and 
            not specaug_after_first_layer and 
            not specaug_time_only and 
            not specaug_shuffled
        ), "specaug options are specified, but enable_specaug=False"

        network["source"] = {"class": "copy", "from": "features"}
        network = fix_network_for_sparse_output(network)

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
    conformer_base_config.update(extra_args or {})

    staged_network_dict = None
    if staged_opts is not None and not recognition:
        staged_network_dict = {1: conformer_base_config.pop("network")}
        network_mod = copy.deepcopy(network)
        for epoch, opts in staged_opts.items():
            if opts == "freeze_features":
                network_mod["features"]["trainable"] = False
                staged_network_dict[epoch] = copy.deepcopy(network_mod)
            elif opts == "remove_aux":
                for layer in list(network_mod.keys()):
                    if layer.startswith("aux"):
                        network_mod.pop(layer)
                staged_network_dict[epoch] = copy.deepcopy(network_mod)

    if recognition:
        rec_network = copy.deepcopy(network)
        rec_network["output"] = {
            "class": "linear",
            "activation": "log_softmax",
            "from": ["MLP_output"],
            "n_out": 9001
        }
        conformer_base_config["network"] = rec_network

    return ReturnnConfig(
        config=conformer_base_config,
        post_config=base_post_config,
        staged_network_dict=staged_network_dict,
        hash_full_python_code=True,
        python_prolog=prolog if not recognition else None,
        python_epilog=RECUSRION_LIMIT,
        pprint_kwargs={"sort_dicts": False},
    )
