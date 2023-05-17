import copy
import numpy as np
from typing import List, Dict, Any, Optional

from sisyphus import tk
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
from .fullsum_ctc_raw_samples import make_conformer_fullsum_ctc_model
from .network_helpers.learning_rates import oclr_default_schedule

RECURSION_LIMIT = """
import sys
sys.setrecursionlimit(3000)
"""


def get_nn_args(nn_base_args, num_epochs, evaluation_epochs=None, prefix=""):
    evaluation_epochs = evaluation_epochs or [num_epochs]

    returnn_configs = {}
    returnn_recog_configs = {}

    for name, args in nn_base_args.items():
        returnn_config, returnn_recog_config = get_nn_args_single(
            num_epochs=num_epochs, evaluation_epochs=evaluation_epochs, **copy.deepcopy(args))
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
    lr_args=None, feature_args=None, returnn_args=None,
):
    feature_args = feature_args or {"class": "GammatoneNetwork", "sample_rate": 8000}
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
                feature_net["subnetwork"][layer]["from"] = "wave_norm"
        feature_net["subnetwork"]["wave_norm"] = {"axes": "T", "class": "norm", "from": "data"}

    returnn_config = get_returnn_config(
        num_inputs=1,
        num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs,
        lr_args=lr_args,
        num_epochs=num_epochs,
        feature_net=feature_net,
        **(returnn_args or {}),
    )

    returnn_recog_config = get_returnn_config(
        num_inputs=1,
        num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs,
        recognition=True,
        num_epochs=num_epochs,
        feature_net=feature_net,
        **(returnn_args or {}),
    )

    return returnn_config, returnn_recog_config


def get_returnn_config(
    num_inputs: int,
    num_outputs: int,
    evaluation_epochs: List[int],
    num_epochs: int,
    rasr_binary_path: tk.Path,
    rasr_loss_corpus_path: tk.Path,
    rasr_loss_corpus_segments: tk.Path,
    rasr_loss_lexicon_path: tk.Path,
    datasets: Dict[str, Dict],
    feature_net: Dict[str, Any],
    lr_args: Optional[Dict[str, Any]] = None,
    conformer_type: str = "wei",
    batch_size: int = 10000,
    sample_rate: int = 8000,
    recognition: bool = False,
    extra_args: Optional[Dict[str, Any]] = None,
):
    base_config = {
        "extern_data": {
            "data": {"dim": num_inputs},
        },
        **datasets,
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

    network, prolog = make_conformer_fullsum_ctc_model(
        num_outputs=88,
        output_args={
            "rasr_binary_path": rasr_binary_path,
            "loss_corpus_path": rasr_loss_corpus_path,
            "loss_corpus_segments": rasr_loss_corpus_segments,
            "loss_lexicon_path": rasr_loss_lexicon_path,
            "am_args": {
                "state_tying": "monophone-eow",
                "states_per_phone": 1,
                "tdp_transition": (0.0, 0.0, "infinity", 0.0),
                "tdp_silence": (0.0, 0.0, "infinity", 0.0),
                "phon_history_length": 0,
                "phon_future_length": 0,
                "allophone_file": tk.Path(
                    "/u/vieting/setups/swb/20230406_feat/dependencies/allophones_blank",
                    hash_overwrite="SWB_ALLOPHONE_FILE_WEI_BLANK"
                ),
                "state_tying_file": tk.Path(
                    "/u/vieting/setups/swb/20230406_feat/dependencies/state-tying_blank",
                    hash_overwrite="SWB_STATE_TYING_FILE_WEI_BLANK"
                ),
            },
        },
        conformer_type=conformer_type,
    )
    for layer in list(network.keys()):
        if network[layer]["from"] == "data":
            network[layer]["from"] = "features"
        elif isinstance(network[layer]["from"], list) and "data" in network[layer]["from"]:
            assert len(network[layer]["from"]) == 1
            network[layer]["from"] = "features"
    network["features"] = feature_net
    if recognition:
        for layer in list(network.keys()):
            if "aux" in layer:
                network.pop(layer)
        network["source"] = {"class": "copy", "from": "features"}
    else:
        # network["source"] = specaug_layer_jingjing(in_layer=["features"])
        pass

    conformer_base_config = copy.deepcopy(base_config)
    conformer_base_config.update(
        {
            "network": network,
            "batch_size": {"classes": batch_size, "data": batch_size * sample_rate // 100},
            "max_seqs": 128,
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.0,
            "learning_rates": [] if recognition else oclr_default_schedule(**(lr_args or {})),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 6,
            "newbob_multi_update_interval": 1,
        }
    )
    conformer_base_config.update(extra_args or {})

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
            python_prolog = RECURSION_LIMIT
        else:
            if isinstance(python_prolog, str):
                python_prolog = [RECURSION_LIMIT, python_prolog]
            else:
                assert isinstance(python_prolog, list)
                python_prolog = [RECURSION_LIMIT] + python_prolog
        return ReturnnConfig(
            config=config,
            post_config=base_post_config,
            staged_network_dict=staged_network_dict if not recognition else None,
            hash_full_python_code=True,
            python_prolog=python_prolog,
            pprint_kwargs={"sort_dicts": False},
        )

    conformer_base_returnn_config = make_returnn_config(
        conformer_base_config,
        staged_network_dict=None,
        python_prolog=prolog,
    )

    return conformer_base_returnn_config
