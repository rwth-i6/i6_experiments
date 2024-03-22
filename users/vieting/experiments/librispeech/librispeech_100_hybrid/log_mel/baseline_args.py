import copy
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Optional
import ipdb
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

from .default_tools import RETURNN_COMMON

RECUSRION_LIMIT = """
import sys
sys.setrecursionlimit(3000)
"""


def get_nn_args(nn_base_args, num_epochs, evaluation_epochs=None, datasets=None, prefix=""):
    evaluation_epochs = evaluation_epochs or [num_epochs]

    returnn_configs = {}
    returnn_recog_configs = {}

    for name, args in nn_base_args.items():
        returnn_config, returnn_recog_config = get_nn_args_single(
            num_epochs=num_epochs, evaluation_epochs=evaluation_epochs, datasets=datasets, **args)
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
    num_outputs: int = 12001, num_epochs: int = 500, evaluation_epochs: Optional[List[int]] = None,
    peak_lr=1e-3, feature_args=None, datasets=None, returnn_args=None,
):
    feature_args = copy.deepcopy(feature_args) or {"class": "LogMelNetwork", "sample_rate": 16000}
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
        datasets=datasets,
        **(returnn_args or {}),
    )


    return returnn_config, returnn_config

def fix_network_for_sparse_output(net):
    net = copy.deepcopy(net)
    net.update({
        "classes_int": {"class": "cast", "dtype": "int16", "from": "data:classes"},
        "classes_squeeze": {"class": "squeeze", "axis": "F", "from": "classes_int"},
        "classes_sparse": {
            "class": "reinterpret_data", "from": "classes_squeeze", "set_sparse": True, "set_sparse_dim": 12001},
    })
    for layer in net:
        if net[layer].get("target", None) == "classes":
            net[layer]["target"] = "layer:classes_sparse"
        if net[layer].get("size_base", None) == "data:classes":
            net[layer]["size_base"] = "classes_sparse"
    return net



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

def get_returnn_config(
    num_inputs: int,
    num_outputs: int,
    evaluation_epochs: List[int],
    peak_lr: float,
    num_epochs: int,
    feature_net: Dict[str, Any],
    batch_size: int = 5000,
    sample_rate: int = 16000,
    recognition: bool = False,
    extra_args: Optional[Dict[str, Any]] = None,
    staged_opts: Optional[Dict[int, Any]] = None,
    enable_specaug: bool = True,
    specaug_mask_sorting: bool = False,
    specaug_after_first_layer: bool = False,
    specaug_time_only: bool = False,
    specaug_shuffled: bool = False,
    mask_divisor: int = None,
    datasets = None,

):
    datasets["train"] = datasets["train"].get_data_dict()
    datasets["cv"] = datasets["cv"].get_data_dict()
    datasets["devtrain"] = datasets["devtrain"].get_data_dict()
    base_config = {
        "extern_data": {
            "data": {"dim": 1},
            "classes": {"dim": 12001, "dtype": "int16", "sparse": True},
            # "classes": {"dim": num_outputs, "sparse": True},  # alignment stored as data with F dim
        },
        **datasets
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
                {"classes": 500, "data": 80000},
                {"classes": 500, "data": 80000},
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
            "n_out": 12001
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


def get_rc_returnn_configs(
        num_inputs: int,
        datasets: Dict[str, Dict],
        num_outputs: int, batch_size: int, evaluation_epochs: List[int],
        recognition=False
):
    # ******************** blstm base ********************
    base_config = {
        **datasets,
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
            "extern_data": {"classes": {"dim": num_outputs,"sparse_dim": num_outputs, "sparse": True, "shape": (None, 1)}, "data": {"dim": 50}},
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
        ipdb.set_trace()
        
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