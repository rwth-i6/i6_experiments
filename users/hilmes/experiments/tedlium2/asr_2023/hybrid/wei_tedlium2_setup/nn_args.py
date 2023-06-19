from .experiment import get_wei_config
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs
from copy import deepcopy

def get_nn_args(num_epochs=125):

    # gets the hardcoded config from existing setup for baseline and comparison
    base_config = get_wei_config()

    returnn_config = ReturnnConfig(config=base_config)
    vanilla_config = deepcopy(returnn_config)

    # Lets look at the diff between native and vanilla
    for key in vanilla_config.config["network"]:
        if "lstm" in key:
            vanilla_config.config["network"][key]["unit"] = "vanillalstm"
    configs = {"wei_config": returnn_config, "vanilla_training": vanilla_config}

    # change softmax to log softmax for hybrid
    recog_configs = deepcopy(configs)
    for config_name in recog_configs:
        recog_configs[config_name].config["network"]["output"]["class"] = "log_softmax"
        recog_configs[config_name].config["network"]["output"]["class"] = "linear"
        recog_configs[config_name].config["network"]["output"]["activation"] = "log_softmax"

    # arguments for ReturnnTraining for now fixed
    training_args = ReturnnTrainingJobArgs(
        num_epochs=num_epochs,
        log_verbosity=5,
        save_interval=1,
        keep_epochs=None,
        time_rqmt=168,
        mem_rqmt=8,
        cpu_rqmt=3,
    )

    # arguments for recognition for now fixed
    # best for now 0.7, 10.0 seems best so try lower
    recognition_args = {
        "dev": {
            "epochs": [num_epochs],
            "feature_flow_key": "fb",
            "prior_scales": [0.3, 0.5, 0.7, 0.8, 0.9],
            "pronunciation_scales": [1.2, 6.0],
            "lm_scales": [20.0, 15.0, 10.0, 9.0, 7.5, 5.0],
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
            "native_ops": ["NativeLstm2"], # this is not required for vanilla lstm but doesn't hurt so leave for now
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
