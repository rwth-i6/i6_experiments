__all__ = ["get_nn_args"]

import copy
import numpy as np
from typing import Dict

import i6_core.features as features
import i6_core.returnn as returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.luescher.helpers.search_params import get_search_parameters


def get_feature_extraction_args():
    dc_detection = False
    samples_options = {
        "audio_format": "wav",
        "dc_detection": dc_detection,
    }

    mfcc_filter_width = {
        "channels": 80,
        "warping_function": "mel",
        "f_max": 7500,
        "f_min": 100,
    }

    mfcc_filter_width = features.filter_width_from_channels(**mfcc_filter_width)

    mfcc_cepstrum_options = {
        "normalize": False,
        "outputs": 80,  # this is the actual output feature dimension
        "add_epsilon": not dc_detection,  # when there is no dc-detection we can have log(0) otherwise
        "epsilon": 1e-10,
    }

    return {
        "gt": {
            "gt_options": {
                "minfreq": 100,
                "maxfreq": 7500,
                "channels": 50,
                # "warp_freqbreak": 7400,
                "tempint_type": "hanning",
                "tempint_shift": 0.01,
                "tempint_length": 0.025,
                "flush_before_gap": True,
                "do_specint": False,
                "specint_type": "hanning",
                "specint_shift": 4,
                "specint_length": 9,
                "normalize": True,
                "preemphasis": True,
                "legacy_scaling": False,
                "without_samples": False,
                "samples_options": samples_options,
                "normalization_options": {},
                "add_features_output": True,
            }
        },
        "mfcc": {
            "num_deriv": 2,
            "num_features": None,  # 33 (confusing name: # max features, above -> clipped)
            "mfcc_options": {
                "warping_function": "mel",
                "filter_width": mfcc_filter_width,
                "normalize": True,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": samples_options,
                "cepstrum_options": mfcc_cepstrum_options,
                "fft_options": None,
                "add_features_output": True,
            },
        },
    }


def get_nn_args(num_outputs: int = 9001, num_epochs: int = 500):
    returnn_train_configs = get_returnn_configs(
        num_inputs=40,
        num_outputs=num_outputs,
        batch_size=24000,
        num_epochs=num_epochs,
    )
    returnn_recog_configs = get_returnn_configs(
        num_inputs=40,
        num_outputs=num_outputs,
        batch_size=24000,
        num_epochs=num_epochs,
        for_recog=True,
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
            "epochs": list(np.arange(250, num_epochs + 1, 10)),
            "feature_flow_key": "gt",
            "prior_scales": [0.3],
            "pronunciation_scales": [6.0],
            "lm_scales": [20.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": get_search_parameters(),
            "lattice_to_ctm_kwargs": {
                "fill_empty_segments": True,
                "best_path_algo": "bellman-ford",
            },
            "optimize_am_lm_scale": False,
            "rtf": 50,
            "mem": 8,
            "parallelize_conversion": True,
        },
    }
    test_recognition_args = None

    nn_args = rasr_util.HybridArgs(
        returnn_training_configs=returnn_train_configs,
        returnn_recognition_configs=returnn_train_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args


def get_returnn_configs(
    num_inputs: int,
    num_outputs: int,
    batch_size: int,
    num_epochs: int,
    for_recog: bool = False,
):
    # ******************** blstm base ********************

    base_config = {
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "window": 1,
        "update_on_device": True,
        "extern_data": {
            "data": {"dim": num_inputs},
            "classes": {"dim": num_outputs, "sparse": True},
        },
    }
    base_post_config = {
        "cleanup_old_models": {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": returnn.CodeWrapper(f"list(np.arange(10, {num_epochs + 1}, 10))"),
        },
    }

    blstm_base_config = copy.deepcopy(base_config)
    blstm_base_config.update(
        {
            "batch_size": batch_size,  # {"classes": batch_size, "data": batch_size},
            "chunking": "100:50",
            "optimizer": {"class": "nadam"},
            "optimizer_epsilon": 1e-8,
            "gradient_noise": 0.1,
            "learning_rates": returnn.CodeWrapper("list(np.linspace(3e-4, 8e-4, 10))"),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 40,
            "newbob_multi_update_interval": 1,
            "network": {
                "lstm_bwd_1": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": -1,
                    "dropout": 0.1,
                    "from": ["data"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_bwd_2": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": -1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_1", "lstm_bwd_1"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_bwd_3": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": -1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_2", "lstm_bwd_2"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_bwd_4": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": -1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_3", "lstm_bwd_3"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_bwd_5": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": -1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_4", "lstm_bwd_4"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_bwd_6": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": -1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_5", "lstm_bwd_5"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_fwd_1": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": 1,
                    "dropout": 0.1,
                    "from": ["data"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_fwd_2": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": 1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_1", "lstm_bwd_1"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_fwd_3": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": 1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_2", "lstm_bwd_2"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_fwd_4": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": 1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_3", "lstm_bwd_3"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_fwd_5": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": 1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_4", "lstm_bwd_4"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "lstm_fwd_6": {
                    "L2": 0.01,
                    "class": "rec",
                    "direction": 1,
                    "dropout": 0.1,
                    "from": ["lstm_fwd_5", "lstm_bwd_5"],
                    "n_out": 1000,
                    "unit": "nativelstm2",
                },
                "output": {
                    "class": "softmax",
                    "from": ["lstm_fwd_6", "lstm_bwd_6"],
                    "loss": "ce",
                }
                if not for_recog
                else {
                    "class": "linear",
                    "activation": "log_softmax",
                    "from": ["lstm_fwd_6", "lstm_bwd_6"],
                },
            },
        }
    )

    blstm_base_returnn_config = returnn.ReturnnConfig(
        config=blstm_base_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={"numpy": "import numpy as np"},
        pprint_kwargs={"sort_dicts": False},
    )

    return {
        "blstm_base": blstm_base_returnn_config,
    }
