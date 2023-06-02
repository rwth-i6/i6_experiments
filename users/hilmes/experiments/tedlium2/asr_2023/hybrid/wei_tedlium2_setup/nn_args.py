from .experiment import get_wei_config
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

def get_nn_args(num_epochs=125):

    base_config = get_wei_config()

    returnn_config = ReturnnConfig(config=base_config)
    configs = {"wei_config": returnn_config}

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
        "dev-other": {
            "epochs": [num_epochs],
            "feature_flow_key": "mfcc",
            "prior_scales": [0.3],
            "pronunciation_scales": [6.0],
            "lm_scales": [20.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 12.0,
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
        },
    }

    nn_args = HybridArgs(
        returnn_training_configs=configs,
        returnn_recognition_configs=configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=None
    )

    return nn_args
