from typing import Dict, Any, Optional, List

import i6_core.returnn as returnn
from i6_core.returnn import CodeWrapper
import i6_experiments.users.berger.args.returnn.learning_rates as learning_rates
import i6_experiments.users.berger.args.returnn.regularization as regularization


def get_base_config() -> Dict[str, Any]:
    return {
        "use_tensorflow": True,
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "batching": "random",
        "window": 1,
        "update_on_device": True,
    }


def get_extern_data_config(
    num_inputs: int,
    num_outputs: int,
    target: str = "classes",
) -> Dict[str, Any]:
    return {
        "extern_data": {
            "data": {"dim": num_inputs},
            target: {"dim": num_outputs, "sparse": True},
        }
    }


def get_base_post_config(
    keep_epochs: Optional[list] = None, **kwargs
) -> Dict[str, Any]:
    post_config = {
        "cleanup_old_models": {
            "keep_last_n": 5,
            "keep_best_n": 5,
        }
    }
    if keep_epochs:
        post_config["cleanup_old_models"]["keep"] = keep_epochs
    return post_config


def get_network_config(network: Dict) -> Dict[str, Dict]:
    return {"network": network}


def get_returnn_config(
    network: Dict,
    *,
    target="classes",
    num_inputs: int,
    num_outputs: int,
    num_epochs: int,
    extern_data_config: bool = True,
    python_prolog: Optional[List] = None,
    extra_python: Optional[List] = None,
    use_chunking: bool = True,
    extra_config: Optional[Dict] = None,
    hash_full_python_code: bool = True,
    **kwargs,
) -> returnn.ReturnnConfig:
    python_prolog = python_prolog or ["import numpy as np"]
    extra_python = extra_python or []
    config_dict = {
        "num_inputs": num_inputs,
        "num_outputs": {target: num_outputs},
        "target": target,
    }
    config_dict.update(get_base_config())
    if extern_data_config:
        config_dict.update(get_extern_data_config(num_inputs, num_outputs, target))
    config_dict.update(get_network_config(network))

    lrate_config, lrate_python = learning_rates.get_learning_rate_config(
        num_epochs=num_epochs, **kwargs
    )
    config_dict.update(lrate_config)
    extra_python += lrate_python

    config_dict.update(regularization.get_base_regularization_config(**kwargs))
    if use_chunking:
        config_dict.update(regularization.get_chunking_config(**kwargs))

    if extra_config:
        config_dict.update(extra_config)

    post_config_dict = {}
    post_config_dict.update(get_base_post_config(**kwargs))

    return returnn.ReturnnConfig(
        config=config_dict,
        post_config=post_config_dict,
        hash_full_python_code=hash_full_python_code,
        python_prolog=python_prolog,
        python_epilog=extra_python,
        pprint_kwargs={"sort_dicts": False},
    )
