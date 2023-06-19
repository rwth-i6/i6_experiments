from .experiment import get_wei_config
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs
from copy import deepcopy

def get_nn_args(num_epochs=125):

    base_config = get_wei_config()

    returnn_config = ReturnnConfig(config=base_config)
    vanilla_config = deepcopy(returnn_config)
    for key in vanilla_config.config["network"]:
        if "lstm" in key:
            vanilla_config.config["network"][key]["unit"] = "vanillalstm"
    configs = {"wei_config": returnn_config, "vanilla_training": vanilla_config}
    recog_configs = deepcopy(configs)
    recog_configs["wei_config"].config["network"]["output"]["class"] = "log_softmax"
    recog_configs["wei_config"].config["network"]["output"]["class"] = "linear"
    recog_configs["wei_config"].config["network"]["output"]["activation"] = "log_softmax"
    recog_configs["vanilla_training"].config["network"]["output"]["class"] = "log_softmax"
    recog_configs["vanilla_training"].config["network"]["output"]["class"] = "linear"
    recog_configs["vanilla_training"].config["network"]["output"]["activation"] = "log_softmax"

    training_args = ReturnnTrainingJobArgs(
        num_epochs=num_epochs,
        log_verbosity=5,
        save_interval=1,
        keep_epochs=None,
        time_rqmt=168,
        mem_rqmt=8,
        cpu_rqmt=3,
    )

    recognition_args = {
        "dev": {
            "epochs": [num_epochs],
            "feature_flow_key": "fb",
            "prior_scales": [0.3, 0.5, 0.7],
            "pronunciation_scales": [6.0],
            "lm_scales": [20.0, 15.0, 10.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 14.0,
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
            "lmgc_mem": 16,
            "cpu": 4,
            "parallelize_conversion": True,
            "use_epoch_for_compile": True,
            "native_ops": ["NativeLstm2"],
        },
    }

    nn_args = HybridArgs(
        returnn_training_configs=configs,
        returnn_recognition_configs=recog_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=None
    )

    return nn_args
