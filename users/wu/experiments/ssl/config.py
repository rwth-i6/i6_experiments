"""RETURNN config assembly for SSL training / forward (no cache manager -- disk-free nodes)."""

import copy
from typing import Any, Dict, Optional

from i6_core.returnn.config import ReturnnConfig

from .serializer import serialize_training, serialize_forward


def get_training_config(
    *,
    train_dataset: Dict[str, Any],
    dev_dataset: Dict[str, Any],
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    train_step_args: Optional[Dict[str, Any]] = None,
    post_config: Optional[Dict[str, Any]] = None,
    python_prolog: Any = None,
    keep_epochs: Optional[Any] = None,
    debug: bool = False,
) -> ReturnnConfig:
    base_post_config = {
        "stop_on_nonfinite_train_score": True,
        "backend": "torch",
        # diagnostics (non-hashed): per-step GPU memory usage and the p2 total grad norm (logged
        # BEFORE clipping, so it is directly comparable to gradient_clip_global_norm and lets us tune
        # the clip threshold from the observed distribution).
        "torch_log_memory_usage": True,
        "log_grad_norm": True,
    }
    # cleanup_old_models stays in the (hashed) config; with keep_epochs=None the dict is byte-identical to
    # the original {keep_last_n:4, keep_best_n:2} so existing runs (e.g. SSL pretraining) do NOT re-hash.
    # Passing keep_epochs adds a "keep" list so those fractional-epoch checkpoints survive cleanup for recog
    # (used by the CTC experiments). RETURNN never deletes epochs listed in cleanup_old_models["keep"].
    cleanup = {"keep_last_n": 4, "keep_best_n": 2}
    if keep_epochs:
        cleanup["keep"] = sorted(set(int(e) for e in keep_epochs))
    base_config = {
        "cleanup_old_models": cleanup,
        "train": copy.deepcopy(train_dataset),
        "dev": copy.deepcopy(dev_dataset),
    }
    config = {**base_config, **copy.deepcopy(config)}
    post_config = {**base_post_config, **copy.deepcopy(post_config or {})}
    serializer = serialize_training(
        network_module=network_module, net_args=net_args, train_step_args=train_step_args, debug=debug
    )
    return ReturnnConfig(
        config=config, post_config=post_config, python_prolog=python_prolog, python_epilog=[serializer]
    )


def get_forward_config(
    *,
    forward_dataset: Dict[str, Any],
    network_module: str,
    config: Dict[str, Any],
    net_args: Dict[str, Any],
    decoder: str,
    decoder_args: Dict[str, Any],
    post_config: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> ReturnnConfig:
    base_post_config = {"backend": "torch"}
    base_config = {"forward_data": copy.deepcopy(forward_dataset)}
    config = {**base_config, **copy.deepcopy(config)}
    post_config = {**base_post_config, **copy.deepcopy(post_config or {})}
    serializer = serialize_forward(
        network_module=network_module,
        net_args=net_args,
        forward_module=decoder,
        forward_init_args=decoder_args,
        debug=debug,
    )
    return ReturnnConfig(config=config, post_config=post_config, python_epilog=[serializer])
