from typing import Any, Dict, List, Optional, Union

import i6_core.returnn as returnn
import i6_experiments.users.berger.args.returnn.learning_rates as learning_rates
import i6_experiments.users.berger.args.returnn.regularization as regularization
from i6_experiments.users.berger.recipe.returnn.training import Backend


def get_base_config() -> Dict[str, Any]:
    result = {
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "batching": "random",
        "window": 1,
        "update_on_device": True,
    }
    return result


def get_extern_data_config(
    num_inputs: Optional[int],
    num_outputs: Optional[int],
    extern_data_kwargs: Dict = {},
    extern_target_kwargs: Dict = {},
    target: Optional[str] = "classes",
    **kwargs,
) -> Dict[str, Any]:
    result = {}
    if num_inputs is not None:
        result["data"] = {"dim": num_inputs, **extern_data_kwargs}
    if num_outputs is not None and target is not None:
        result[target] = {"dim": num_outputs, "sparse": True, **extern_target_kwargs}
    return {"extern_data": result}


def get_base_post_config(
    keep_last_n: Optional[int] = None, keep_best_n: Optional[int] = None, keep: Optional[List[int]] = None, **kwargs
) -> Dict[str, Any]:
    if keep_last_n is None and keep_best_n is None and keep is None:
        post_config = {"cleanup_old_models": True}
    else:
        cleanup_opts = {}
        if keep_last_n is not None:
            cleanup_opts["keep_last_n"] = keep_last_n
        if keep_best_n is not None:
            cleanup_opts["keep_best_n"] = keep_best_n
        if keep is not None:
            cleanup_opts["keep"] = keep
        post_config = {"cleanup_old_models": cleanup_opts}
    return post_config


def get_network_config(network: Dict) -> Dict[str, Dict]:
    return {"network": network}


def get_returnn_config(
    network: Optional[Dict] = None,
    *,
    use_base_config: bool = True,
    backend: Backend = Backend.TENSORFLOW,
    target: Optional[str] = "classes",
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    python_prolog: Optional[List] = None,
    extern_data_config: bool = False,
    extra_python: Optional[List] = None,
    use_chunking: bool = True,
    extra_config: Optional[Dict] = None,
    hash_full_python_code: bool = False,
    **kwargs,
) -> returnn.ReturnnConfig:
    python_prolog = ["import numpy as np"] + (python_prolog or [])
    extra_python = extra_python or []
    config_dict: dict[str, Any] = {"target": target}
    if num_inputs is not None:
        config_dict["num_inputs"] = num_inputs
    if num_outputs is not None:
        config_dict["num_outputs"] = {target: num_outputs}
    if extern_data_config:
        config_dict.update(
            get_extern_data_config(num_inputs=num_inputs, num_outputs=num_outputs, target=target, **kwargs)
        )
    if use_base_config:
        config_dict.update(get_base_config())

    if backend == Backend.TENSORFLOW:
        config_dict["use_tensorflow"] = True
    elif backend == Backend.PYTORCH:
        config_dict["backend"] = "torch"
        config_dict["use_lovely_tensors"] = True
    else:
        raise NotImplementedError
    if network:
        config_dict.update(get_network_config(network))

    lrate_config, lrate_python = learning_rates.get_learning_rate_config(**kwargs)
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
