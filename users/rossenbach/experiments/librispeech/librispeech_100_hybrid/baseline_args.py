import copy
import numpy as np
from typing import List

from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs


def blstm_network(layers, input_layers, dropout=0.1, l2=0.0):
    num_layers = len(layers)
    assert num_layers > 0

    network = {}

    for idx, size in enumerate(layers):
        idx += 1
        for direction, name in [(1, "fwd"), (-1, "bwd")]:
            if idx == 1:
                from_layers = input_layers
            else:
                from_layers = [
                    "lstm_fwd_{}".format(idx - 1),
                    "lstm_bwd_{}".format(idx - 1),
                ]
            network["lstm_{}_{}".format(name, idx)] = {
                "class": "rec",
                "unit": "nativelstm2",
                "direction": direction,
                "n_out": size,
                "dropout": dropout,
                "L2": l2,
                "from": from_layers,
            }

    output_layers = ["lstm_fwd_{}".format(num_layers), "lstm_bwd_{}".format(num_layers)]

    return network, output_layers


def get_nn_args(num_outputs: int = 12001, num_epochs: int = 250):
    evaluation_epochs = list(np.arange(150, num_epochs + 1, 10))

    returnn_configs = get_returnn_configs(
        num_inputs=50,
        num_outputs=num_outputs,
        batch_size=24000,
        evaluation_epochs=evaluation_epochs,
    )

    training_args = {
        "log_verbosity": 4,
        "num_epochs": num_epochs,
        "num_classes": num_outputs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 7,
        "cpu_rqmt": 3,
        "partition_epochs": {"train": 20, "dev": 1},
        "use_python_control": False,
    }
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
                "beam-pruning": 16.0,
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
            "parallelize_conversion": True,
        },
    }
    test_recognition_args = None

    nn_args = HybridArgs(
        returnn_configs=returnn_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args


def get_returnn_configs(
    num_inputs: int,
    num_outputs: int,
    batch_size: int,
    evaluation_epochs: List[int],
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
        "cleanup_old_models": {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": evaluation_epochs,
        },
    }

    network, last_layer = blstm_network([1024] * 8, ["specaug"], dropout=0.1, l2=0.01)
    from .specaugment_clean_legacy import specaug_layer, get_funcs

    network["specaug"] = specaug_layer(["data"])
    network["out_linear"] = {
        "class": "linear",
        "activation": None,
        "from": last_layer,
        "n_out": num_outputs,
    }
    network["output"] = {
        "class": "activation",
        "activation": "softmax",
        "from": ["out_linear"],
        "loss": "ce",
        "target": "classes",
    }
    network["log_output"] = {
        "class": "activation",
        "activation": "log_softmax",
        "from": ["out_linear"],
    }

    blstm_base_config = copy.deepcopy(base_config)
    blstm_base_config.update(
        {
            "batch_size": batch_size,  # {"classes": batch_size, "data": batch_size},
            "chunking": "100:50",
            "optimizer": {"class": "nadam"},
            "optimizer_epsilon": 1e-8,
            "gradient_noise": 0.1,
            "learning_rates": list(np.linspace(3e-4, 8e-4, 10)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 40,
            "newbob_multi_update_interval": 1,
            "network": network,
        }
    )

    blstm_base_returnn_config = ReturnnConfig(
        config=blstm_base_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog=get_funcs(),
        pprint_kwargs={"sort_dicts": False},
    )

    return {
        "blstm_base": blstm_base_returnn_config,
    }
