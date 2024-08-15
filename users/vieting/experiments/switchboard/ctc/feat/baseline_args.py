import copy
import numpy as np
from typing import List, Dict, Any, Optional, Union

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
from .network_helpers.perturbation import get_code_for_perturbation

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
            num_epochs=num_epochs, evaluation_epochs=evaluation_epochs, **copy.deepcopy(args)
        )
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
    num_outputs: int = 88,
    num_epochs: int = 500,
    evaluation_epochs: Optional[List[int]] = None,
    lr_args=None,
    feature_args=None,
    returnn_args=None,
    report_args=None,
):
    feature_args = feature_args or {"class": "GammatoneNetwork", "sample_rate": 8000}
    preemphasis = feature_args.pop("preemphasis", None)
    preemphasis_first = feature_args.pop("preemphasis_first", False)
    wave_norm = feature_args.pop("wave_norm", False)
    feature_network_class = {
        "LogMelNetwork": LogMelNetwork,
        "GammatoneNetwork": GammatoneNetwork,
        "ScfNetwork": ScfNetwork,
    }[feature_args.pop("class")]
    feature_net = feature_network_class(**feature_args).get_as_subnetwork()
    source_layer = "data"

    if wave_norm == "fix":
        feature_net["subnetwork"]["wave_norm"] = {
            "axes": "T",
            "class": "norm",
            "from": source_layer,
            "trainable": False,
        }
        source_layer = "wave_norm"
    elif wave_norm:
        feature_net["subnetwork"]["wave_norm"] = {"axes": "T", "class": "norm", "from": source_layer}
        source_layer = "wave_norm"
    if preemphasis:
        feature_net["subnetwork"]["preemphasis"] = PreemphasisNetwork(alpha=preemphasis).get_as_subnetwork(
            source=source_layer
        )
        source_layer = "preemphasis"
    if preemphasis_first and wave_norm and preemphasis:
        feature_net["subnetwork"]["preemphasis"]["from"] = feature_net["subnetwork"]["wave_norm"]["from"]
        feature_net["subnetwork"]["wave_norm"]["from"] = "preemphasis"
        source_layer = "wave_norm"
    for layer in feature_net["subnetwork"]:
        if layer not in ["wave_norm", "preemphasis"]:
            layer_config = feature_net["subnetwork"][layer]
            if layer_config.get("class") != "variable" and layer_config.get("from", "data") == "data":
                feature_net["subnetwork"][layer]["from"] = source_layer

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

    report_args = {
        "features": feature_network_class.__name__,
        "preemphasis": preemphasis,
        "wave_norm": wave_norm,
        **(report_args or {}),
    }

    return returnn_config, returnn_recog_config, report_args


def get_returnn_config(
    num_inputs: int,
    num_outputs: int,
    evaluation_epochs: List[int],
    num_epochs: int,
    rasr_binary_path: tk.Path,
    rasr_loss_corpus_path: tk.Path,
    rasr_loss_lexicon_path: tk.Path,
    datasets: Dict[str, Dict],
    feature_net: Dict[str, Any],
    rasr_loss_corpus_segments: Optional[tk.Path] = None,
    rasr_loss_corpus_prefix: Optional[tk.Path] = None,
    lr_args: Optional[Dict[str, Any]] = None,
    conformer_type: str = "wei",
    specaug_old: Optional[Dict[str, Any]] = None,
    specaug_new: Optional[Dict[str, Any]] = None,
    am_args: Optional[Dict[str, Any]] = None,
    batch_size: Union[int, Dict[str, int]] = 10000,
    sample_rate: int = 8000,
    recognition: bool = False,
    extra_args: Optional[Dict[str, Any]] = None,
    staged_opts: Optional[Dict[int, Any]] = None,
    audio_perturbation: bool = False,
    use_multi_proc_dataset: bool = False,
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
    am_args = am_args or {
        "state_tying": "monophone-eow",
        "states_per_phone": 1,
        "tdp_transition": (0.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 0.0, "infinity", 0.0),
        "phon_history_length": 0,
        "phon_future_length": 0,
        "allophone_file": tk.Path(
            "/u/vieting/setups/swb/20230406_feat/dependencies/allophones_blank",
            hash_overwrite="SWB_ALLOPHONE_FILE_WEI_BLANK",
        ),
        "state_tying_file": tk.Path(
            "/u/vieting/setups/swb/20230406_feat/dependencies/state-tying_blank",
            hash_overwrite="SWB_STATE_TYING_FILE_WEI_BLANK",
        ),
    }

    network, prolog = make_conformer_fullsum_ctc_model(
        num_outputs=num_outputs,
        output_args={
            "rasr_binary_path": rasr_binary_path,
            "loss_corpus_path": rasr_loss_corpus_path,
            "loss_corpus_segments": rasr_loss_corpus_segments,
            "loss_corpus_prefix": rasr_loss_corpus_prefix,
            "loss_lexicon_path": rasr_loss_lexicon_path,
            "am_args": am_args,
        },
        conformer_type=conformer_type,
        specaug_old=specaug_old,
        specaug_new=specaug_new,
        recognition=recognition,
        num_epochs=num_epochs,
    )

    if audio_perturbation:
        prolog += get_code_for_perturbation()
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

    if audio_perturbation and recognition:
        # Remove pre-processing from recognition and replace with layers in the network if needed
        datasets["train"]["dataset"]["audio"].pop("pre_process", None)

        feature_net = copy.deepcopy(feature_net)
        audio_perturb_args = extra_args.get("audio_perturb_args", {})
        assert not (
            "preemphasis" in audio_perturb_args and "codecs" in audio_perturb_args
        ), "Not implemented yet, need to think about the order to apply"
        source_layer = "data"
        if "preemphasis" in audio_perturb_args:
            # preemphasis in training is done in perturbation pre-processing, add to network for recognition
            for layer in feature_net["subnetwork"]:
                if feature_net["subnetwork"][layer].get("from", None) == source_layer:
                    feature_net["subnetwork"][layer]["from"] = "preemphasis"
            source_layer = "preemphasis"
            alpha = audio_perturb_args["preemphasis"].get("default")
            feature_net["subnetwork"]["preemphasis"] = PreemphasisNetwork(alpha=alpha).get_as_subnetwork(
                source=source_layer
            )
        if "codecs" in audio_perturb_args:
            # codec in training is applied in perturbation pre-processing, add to network for recognition
            for layer in feature_net["subnetwork"]:
                if feature_net["subnetwork"][layer].get("from", None) == source_layer:
                    feature_net["subnetwork"][layer]["from"] = "codec"
            source_layer = "codec"
            feature_net["subnetwork"]["codec"] = {
                "class": "subnetwork",
                "from": source_layer,
                "subnetwork": {
                    "assert_range": {
                        "class": "eval",
                        "from": "data",
                        "eval": """
                            tf.debugging.assert_less_equal(
                                tf.abs(source(0)),
                                tf.constant(1.0, dtype=source(0).dtype),
                                message="Input values must be in the range [-1.0, 1.0]"
                            )
                        """,
                    },
                    "magnitude": {
                        "class": "eval",
                        "from": "assert_range",
                        "eval": "tf.math.log1p(255.0 * tf.abs(source(0))) / tf.math.log1p(255.0)",
                    },
                    "output": {
                        "class": "eval",
                        "from": "magnitude",
                        "eval": "tf.sign(source(0)) * source(1)",
                        "from": ["assert_range", "magnitude"],
                    },
                },
                "trainable": False,
            }
        extra_args.pop("audio_perturb_args", None)
        extra_args.pop("audio_perturb_runner", None)

    if isinstance(batch_size, int):
        # If batch size is int, adapt to waveform. If it is dict, assume it is already correct.
        batch_size = {"classes": batch_size, "data": batch_size * sample_rate // 100}
    if use_multi_proc_dataset:
        dataset = datasets["train"]["dataset"]["partition_epoch"]
    else:
        dataset = datasets["train"]["partition_epoch"]

    conformer_base_config = copy.deepcopy(base_config)
    conformer_base_config.update(
        {
            "network": network,
            "batch_size": batch_size,
            "max_seqs": 128,
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.0,
            "learning_rates": [] if recognition else oclr_default_schedule(**(lr_args or {})),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": dataset,
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
            elif opts == "unfreeze_features":
                network_mod["features"]["trainable"] = True
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
        python_prolog=RECURSION_LIMIT if recognition else [RECURSION_LIMIT] + prolog,
        pprint_kwargs={"sort_dicts": False},
    )
