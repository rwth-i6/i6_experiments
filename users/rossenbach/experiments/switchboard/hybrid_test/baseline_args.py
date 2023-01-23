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


def blstm_network(layers, input_layers, num_outputs, dropout=0.1, l2=0.0):
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

    network["out_linear"] = {
        "class": "linear",
        "activation": None,
        "from": output_layers,
        "n_out": num_outputs,
    }
    network["output"] = {
        "class": "activation",
        "activation": "softmax",
        "from": ["out_linear"],
        "loss": "ce",
        'loss_opts': {'focal_loss_factor': 2.0},
        "target": "classes"
    }
    network["log_output"] = {
        "class": "activation",
        "activation": "log_softmax",
        "from": ["out_linear"],
    }

    return network


def get_nn_args(num_outputs: int = 12001, num_epochs: int = 250, extra_exps=False):
    evaluation_epochs  = list(np.arange(num_epochs, num_epochs + 1, 10))

    returnn_configs = get_returnn_configs(
        num_inputs=40, num_outputs=num_outputs, batch_size=5000,
        evaluation_epochs=evaluation_epochs, extra_exps=extra_exps,
    )

    returnn_recog_configs = get_returnn_configs(
        num_inputs=40, num_outputs=num_outputs, batch_size=5000,
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
        "partition_epochs": {"train": 40, "dev": 20},
        "use_python_control": False,
    }
    recognition_args = {
        "hub5e00": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "gt",
            "prior_scales": [0.5, 0.6, 0.7],
            "pronunciation_scales": [6.0],
            "lm_scales": [10.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 16.0,
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
            "forward_output_layer": "log_output",
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



def get_returnn_configs(
        num_inputs: int, num_outputs: int, batch_size: int, evaluation_epochs: List[int],
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

    networks = {}
    networks["8x1024"] = blstm_network([1024]*8, ["specaug"], num_outputs=num_outputs, dropout=0.0, l2=0.0)
    from ...librispeech.librispeech_100_hybrid.specaugment_clean_legacy import specaug_layer, get_funcs

    for network in networks.values():
        network["specaug"] = specaug_layer(in_layer = ["data"])
        if recognition:
            network["log_output"]["is_output_layer"] = True


    blstm_base_config = copy.deepcopy(base_config)
    blstm_base_config.update(
        {
            "batch_size": batch_size,  # {"classes": batch_size, "data": batch_size},
            "chunking": "50:25",
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.3,
            "learning_rates": list(np.linspace(2.5e-5, 2.5e-4, 10)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            #"min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.707,
            "newbob_multi_num_epochs": 40,
            "newbob_multi_update_interval": 1,
            "network": networks["8x1024"],
        }
    )

    def make_returnn_config(config):
        return ReturnnConfig(
            config=config,
            post_config=base_post_config,
            hash_full_python_code=True,
            python_prolog=get_funcs(),
            pprint_kwargs={"sort_dicts": False},
        )
    blstm_base_returnn_config = make_returnn_config(blstm_base_config)

    if extra_exps:
        oclr_v1_config = copy.deepcopy(blstm_base_config)
        oclr_v1_config["learning_rates"] = list(np.linspace(2.5e-5, 3e-4, 50)) + list(np.linspace(3e-4, 2.5e-5, 50))
        oclr_v1_config["newbob_multi_num_epochs"] = 3
        oclr_v1_nofl_config = copy.deepcopy(oclr_v1_config)
        oclr_v1_nofl_config["network"]["output"]["loss_opts"] = None
        oclr_v2_config = copy.deepcopy(blstm_base_config)
        oclr_v2_config["learning_rates"] = list(np.linspace(2.5e-5, 4e-4, 50)) + list(np.linspace(4e-4, 2.5e-5, 50))
        oclr_v2_config["newbob_multi_num_epochs"] = 3

        return {
            "blstm_oclr_v1": make_returnn_config(oclr_v1_config),
            #"blstm_oclr_v1_nofl": make_returnn_config(oclr_v1_nofl_config),
            #"blstm_oclr_v2": make_returnn_config(oclr_v2_config),
            #"blstm_base": blstm_base_returnn_config,
        }
    else:
        return {
            "blstm_base": blstm_base_returnn_config,
        }


