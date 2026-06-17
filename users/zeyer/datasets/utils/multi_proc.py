"""
Helper functions to wrap any dataset in MultiProcDataset.
"""

from __future__ import annotations
from typing import Any, Dict, Optional


def multi_proc_dataset_opts(
    dataset_opts: Dict[str, Any],
    *,
    num_workers: int = 4,
    buffer_size: int = 10,
    sharding_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    wrap

    :param sharding_method: optional MPD sharding mode
        (e.g. ``"dedicated"`` for datasets like ``CombinedDataset`` that don't implement
        ``get_current_seq_order``); ``None`` keeps the MPD default.
    """
    opts: Dict[str, Any] = {
        "class": "MultiProcDataset",
        "dataset": dataset_opts,
        "num_workers": num_workers,
        "buffer_size": buffer_size,
    }
    if sharding_method is not None:
        opts["sharding_method"] = sharding_method
    return opts


def multi_proc_eval_datasets_opts(
    eval_datasets_opts: Dict[str, Dict[str, Any]],
    *,
    num_workers: int = 4,
    buffer_size: int = 10,
    sharding_method: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    wrap
    """
    return {
        key: multi_proc_dataset_opts(
            opts, num_workers=num_workers, buffer_size=buffer_size, sharding_method=sharding_method
        )
        for key, opts in eval_datasets_opts.items()
    }
