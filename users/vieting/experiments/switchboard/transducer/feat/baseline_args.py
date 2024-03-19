import copy
from typing import List, Dict, Any, Optional, Union

from sisyphus import tk
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs
from i6_experiments.users.vieting.models.tf_networks.features import (
    LogMelNetwork,
    GammatoneNetwork,
    ScfNetwork,
    PreemphasisNetwork,
)
from i6_experiments.users.vieting.jobs.returnn import PeakyAlignmentJob
from ...ctc.feat.network_helpers.perturbation import get_code_for_perturbation
from .helpers.transducer import make_conformer_transducer_model

RECURSION_LIMIT = """
import sys
sys.setrecursionlimit(3000)
"""


def get_nn_args(nn_base_args, num_epochs, evaluation_epochs=None, prefix="", training_args=None):
    evaluation_epochs = evaluation_epochs or [num_epochs]

    returnn_configs = {}
    returnn_recog_configs = {}
    report_args_collection = {}

    for name, args in nn_base_args.items():
        returnn_config, returnn_recog_config, report_args = get_nn_args_single(
            num_epochs=num_epochs, evaluation_epochs=evaluation_epochs, **copy.deepcopy(args))
        returnn_configs[prefix + name] = returnn_config
        returnn_recog_configs[prefix + name] = returnn_recog_config
        report_args_collection[prefix + name] = report_args

    training_args = training_args or {
        "log_verbosity": 4,
        "num_epochs": num_epochs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 16,
        "cpu_rqmt": 3,
    }
    recognition_args = None
    test_recognition_args = None

    nn_args = HybridArgs(
        returnn_training_configs=returnn_configs,
        returnn_recognition_configs=returnn_recog_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args, report_args_collection


def get_nn_args_single(
    num_outputs: int = 88, num_inputs: int = 1, num_epochs: int = 500, evaluation_epochs: Optional[List[int]] = None,
    lr_args=None, feature_args=None, returnn_args=None, report_args=None,
):
    if feature_args is not None:
        preemphasis = feature_args.pop("preemphasis", None)
        wave_norm = feature_args.pop("wave_norm", False)
        wave_cast = feature_args.pop("wave_cast", False)
        feature_network_class = {
            "LogMelNetwork": LogMelNetwork,
            "GammatoneNetwork": GammatoneNetwork,
            "ScfNetwork": ScfNetwork,
        }[feature_args.pop("class")]
        feature_net = feature_network_class(**feature_args).get_as_subnetwork()
        source_layer = "data"
        if wave_cast:
            for layer in feature_net["subnetwork"]:
                if feature_net["subnetwork"][layer].get("from") == source_layer:
                    feature_net["subnetwork"][layer]["from"] = "wave_cast"
            feature_net["subnetwork"]["wave_cast"] = {"class": "cast", "dtype": "float32", "from": source_layer}
            source_layer = "wave_cast"
        if wave_norm:
            for layer in feature_net["subnetwork"]:
                if feature_net["subnetwork"][layer].get("from") == source_layer:
                    feature_net["subnetwork"][layer]["from"] = "wave_norm"
            feature_net["subnetwork"]["wave_norm"] = {"axes": "T", "class": "norm", "from": source_layer}
            source_layer = "wave_norm"
        if preemphasis:
            assert source_layer == "data", "not yet implemented, needs to be fixed in PreemphasisNetwork"
            for layer in feature_net["subnetwork"]:
                if feature_net["subnetwork"][layer].get("from", "data") == source_layer:
                    feature_net["subnetwork"][layer]["from"] = "preemphasis"
            feature_net["subnetwork"]["preemphasis"] = PreemphasisNetwork(alpha=preemphasis).get_as_subnetwork()
            source_layer = "preemphasis"
    else:
        feature_net = None

    returnn_config = get_returnn_config(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs,
        lr_args=lr_args,
        num_epochs=num_epochs,
        feature_net=feature_net,
        **(returnn_args or {}),
    )

    returnn_recog_config = get_returnn_config(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        evaluation_epochs=evaluation_epochs,
        recognition=True,
        num_epochs=num_epochs,
        feature_net=feature_net,
        **(returnn_args or {}),
    )

    report_args = {
        **({
            "features": feature_network_class.__name__,
            "preemphasis": preemphasis,
            "wave_norm": wave_norm,
        } if feature_args is not None else {
            "features": "RasrFeatureCaches",
        }),
        **(report_args or {}),
    }

    return returnn_config, returnn_recog_config, report_args


def get_returnn_config(
    num_inputs: int,
    num_outputs: int,
    evaluation_epochs: List[int],
    num_epochs: int,
    datasets: Dict[str, Dict],
    feature_net: Optional[Dict[str, Any]] = None,
    rasr_loss_args: Optional[Dict[str, tk.Path]] = None,
    lr_args: Optional[Dict[str, Any]] = None,
    conformer_type: str = "wei",
    conformer_args: Optional[Dict] = None,
    specaug_old: Optional[Dict[str, Any]] = None,
    am_args: Optional[Dict[str, Any]] = None,
    batch_size: Union[int, Dict[str, int]] = 10000,
    sample_rate: int = 8000,
    recognition: bool = False,
    extra_args: Optional[Dict[str, Any]] = None,
    staged_opts: Optional[Dict[int, Any]] = None,
    audio_perturbation: bool = False,
    preload_checkpoint: Optional[tk.Path] = None,
):
    if preload_checkpoint is not None and not recognition:
        extra_args["preload_from_files"] = {
            "checkpoint": {
                "filename": preload_checkpoint,
                "ignore_missing": True,
                "init_for_train": True,
            },
        }
    base_config = {
        "extern_data": {
            "data": {"dim": num_inputs},
            "classes": {"dim": num_outputs, "dtype": "int8", "sparse": True},
        },
        **datasets,
    }
    if datasets["train"]["class"] == "MetaDataset":
        alignment_dataset = datasets["train"]["datasets"][datasets["train"]["data_map"]["classes"][0]]
        if isinstance(alignment_dataset["files"][0].creator, PeakyAlignmentJob):
            base_config["extern_data"]["classes"]["dtype"] = "int32" if recognition else "uint16"
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
    am_args = am_args or {
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
    }
    lr_args = lr_args or {}

    network, prolog = make_conformer_transducer_model(
        num_outputs=num_outputs,
        conformer_args=(conformer_args or {}),
        output_args={
            "am_args": am_args,
            **(rasr_loss_args or {}),
        },
        conformer_type=conformer_type,
        specaug_old=specaug_old,
        recognition=recognition,
    )

    if feature_net is not None:
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
        network["source"] = {"class": "copy", "from": "features" if feature_net is not None else "data"}
    else:
        if audio_perturbation:
            prolog += get_code_for_perturbation()
        if "dynamic_learning_rate" in lr_args:
            prolog += [lr_args["dynamic_learning_rate"]]

    if isinstance(batch_size, int) and feature_net is not None:
        # If batch size is int, adapt to waveform. If it is dict, assume it is already correct.
        batch_size = {"classes": batch_size, "data": batch_size * sample_rate // 100}
    conformer_base_config = copy.deepcopy(base_config)
    conformer_base_config.update(
        {
            "network": network,
            "batch_size": batch_size,
            "max_seqs": 128,
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.0,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-6,
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

    return ReturnnConfig(
        config=conformer_base_config,
        post_config=base_post_config,
        staged_network_dict=staged_network_dict,
        hash_full_python_code=True,
        python_prolog=[RECURSION_LIMIT] + prolog,
        pprint_kwargs={"sort_dicts": False},
    )
