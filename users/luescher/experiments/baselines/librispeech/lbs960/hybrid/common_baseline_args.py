__all__ = ["get_nn_args"]

import copy
import numpy as np

import i6_core.returnn as returnn

import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.luescher.helpers.search_params import get_search_parameters

import returnn_common.nn.encoder.blstm_specaug as net_blstm_specaug
import returnn_common.nn.encoder.blstm_cnn_specaug as net_blstm_cnn_specaug


def get_nn_args(num_outputs: int = 9001, num_epochs: int = 500):
    returnn_configs = get_returnn_configs(
        num_inputs=40, num_outputs=num_outputs, batch_size=24000, num_epochs=num_epochs
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
        returnn_training_configs=returnn_configs,
        returnn_recognition_configs=returnn_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args


def get_returnn_configs(
    num_inputs: int, num_outputs: int, batch_size: int, num_epochs: int
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
                "base_blstm": net_blstm_specaug.make_encoder(
                    src="data",
                    num_layers=4,
                    lstm_dim=512,
                    time_reduction=1,
                    with_specaugment=False,
                    dropout=0.25,
                    l2=0.1,
                    rec_weight_dropout=0.0,
                ),
                "output": {
                    "class": "softmax",
                    "loss": "ce",
                    "dropout": 0.1,
                    "from": ["base_blstm"],
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

    # ******************** blstm cnn ********************

    blstm_cnn_config = copy.deepcopy(base_config)
    blstm_cnn_config.update(
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
                "base_blstm": net_blstm_cnn_specaug.make_encoder(
                    src="data",
                    num_layers=4,
                    lstm_dim=512,
                    time_reduction=1,
                    with_specaugment=False,
                    dropout=0.25,
                    l2=0.1,
                    rec_weight_dropout=0.0,
                ),
                "output": {
                    "class": "softmax",
                    "loss": "ce",
                    "dropout": 0.1,
                    "from": ["base_blstm"],
                },
            },
        }
    )

    blstm_cnn_returnn_config = returnn.ReturnnConfig(
        config=blstm_cnn_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={"numpy": "import numpy as np"},
        pprint_kwargs={"sort_dicts": False},
    )

    # ******************** blstm specaug ********************

    blstm_spec_config = copy.deepcopy(base_config)
    blstm_spec_config.update(
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
                "base_blstm": net_blstm_specaug.make_encoder(
                    src="data",
                    num_layers=4,
                    lstm_dim=512,
                    time_reduction=1,
                    with_specaugment=True,
                    dropout=0.25,
                    l2=0.1,
                    rec_weight_dropout=0.0,
                ),
                "output": {
                    "class": "softmax",
                    "loss": "ce",
                    "dropout": 0.1,
                    "from": ["base_blstm"],
                },
            },
        }
    )

    blstm_spec_returnn_config = returnn.ReturnnConfig(
        config=blstm_spec_config,
        post_config=base_post_config,
        hash_full_python_code=True,
        python_prolog={"numpy": "import numpy as np"},
        pprint_kwargs={"sort_dicts": False},
    )

    return {
        "blstm_base": blstm_base_returnn_config,
        "blstm_cnn": blstm_cnn_returnn_config,
        "blstm_spec": blstm_spec_returnn_config,
    }
