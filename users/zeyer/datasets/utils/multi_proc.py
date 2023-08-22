"""
Helper functions to wrap any dataset in MultiProcDataset.
"""

from __future__ import annotations
from typing import Any, Dict


def multi_proc_dataset_opts(
    dataset_opts: Dict[str, Any], *, num_workers: int = 4, buffer_size: int = 10
) -> Dict[str, Any]:
    """
    wrap
    """
    return {
        "class": "MultiProcDataset",
        "dataset": dataset_opts,
        "num_workers": num_workers,
        "buffer_size": buffer_size,
    }


def multi_proc_eval_datasets_opts(
    eval_datasets_opts: Dict[str, Dict[str, Any]], *, num_workers: int = 4, buffer_size: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    wrap
    """
    return {
        key: multi_proc_dataset_opts(opts, num_workers=num_workers, buffer_size=buffer_size)
        for key, opts in eval_datasets_opts.items()
    }
