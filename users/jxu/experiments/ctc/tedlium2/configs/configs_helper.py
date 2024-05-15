from typing import Any, Dict, List, Optional, Union
import numpy as np

def get_base_config() -> Dict[str, Any]:
    result = {
        # "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",
        "batching": "random",
        "window": 1,
        "update_on_device": True,
        "backend": "torch",
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3}
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


def get_base_regularization_config(
    batch_size: int = 10000,
    max_seqs: int = 128,
    accum_grad: int = 1,
    grad_noise: Optional[float] = 0.1,
    grad_clip: Optional[float] = None,
    grad_clip_global_norm: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    result = {"batch_size": batch_size, "max_seqs": max_seqs}
    if grad_noise is not None:
        result["gradient_noise"] = grad_noise
    if grad_clip is not None:
        result["gradient_clip"] = grad_clip
    if grad_clip_global_norm is not None:
        result["gradient_clip_global_norm"] = grad_clip_global_norm
    if accum_grad > 1:
        result["accum_grad_multiple_step"] = accum_grad
    return result


def get_base_post_config(keep_last_n: Optional[int] = None, keep_best_n: Optional[int] = None, keep: Optional[List[int]] = None, **kwargs) -> Dict[str, Any]:
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


def get_oclr_config(
    num_epochs: int,
    peak_lr: float = 1e-03,
    cycle_epoch: Optional[int] = None,
    lr_1: Optional[float] = None,
    lr_2: Optional[float] = None,
    final_lr: Optional[float] = None,
    **kwargs,
) -> dict:
    lr_1 = lr_1 or peak_lr / 10
    lr_2 = lr_2 or peak_lr / 10
    final_lr = final_lr or lr_1 / 5
    cycle_epoch = cycle_epoch or (num_epochs * 9) // 20  # 45% of the training
    lr_list = (
        list(np.linspace(lr_1, peak_lr, cycle_epoch, endpoint=False))
        + list(np.linspace(peak_lr, lr_2, cycle_epoch, endpoint=False))
        + list(np.linspace(lr_2, final_lr, num_epochs - 2 * cycle_epoch))
    )

    return {
        "learning_rates": lr_list,
    }