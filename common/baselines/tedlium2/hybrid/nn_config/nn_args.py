import copy

from .experiment import get_wei_config
from .nn_setup import get_spec_augment_mask_python
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs
from copy import deepcopy


def get_nn_args(num_epochs=125, no_min_seq_len=False):

    # gets the hardcoded config from existing setup for baseline and comparison
    base_config = get_wei_config()
    returnn_config = ReturnnConfig(config=base_config)
    # two variants of spec augment
    spec_augment_args = {
        "max_time_num": 3,
        "max_time": 10,
        "max_feature_num": 5,
        "max_feature": 18,
        "conservatvie_step": 2000,
    }
    specaug = get_spec_augment_mask_python(**spec_augment_args)
    specaug_config = get_wei_config(specaug=True)
    spec_cfg = ReturnnConfig(config=copy.deepcopy(specaug_config), python_epilog=specaug)

    configs = {
        "wei_base_config": returnn_config,
        "wei_specaug_config": spec_cfg,
    }

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

    recognition_args = {
        "dev": {
            "epochs": [num_epochs],
            "feature_flow_key": "fb",
            "prior_scales": [0.7, 0.8, 0.9],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0, 7.5, 5.0],
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
        test_recognition_args=None,
    )

    return nn_args
