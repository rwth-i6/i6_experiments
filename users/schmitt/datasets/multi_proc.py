"""
Helper classes around RETURNN datasets for arbitrary data
"""
__all__ = ["MultiProcDataset"]

from sisyphus import tk
from typing import Any, Dict, List, Optional, Union

from i6_experiments.common.setups.returnn.datasets.base import Dataset


def multi_proc_dataset_opts(
    dataset_opts: Dict[str, Any], *, num_workers: int = 4, buffer_size: int = 10, sharding_method: str = "seq_order"
) -> Dict[str, Any]:
    """
    wrap
    """
    d = {
        "class": "MultiProcDataset",
        "dataset": dataset_opts,
        "num_workers": num_workers,
        "buffer_size": buffer_size,
    }

    if sharding_method != "seq_order":
        d["sharding_method"] = sharding_method

    return d



class MultiProcDataset(Dataset):
    """
    Helper class for the MultiProc Dataset
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        buffer_size: int,
        num_workers: int,
        sharding_method: Optional[str] = None,
        additional_options: Optional[Dict[str, Any]] = None,
    ):
        """

        Args:
            dataset:
            buffer_size:
            num_workers:
            additional_options:
        """
        super().__init__(
            additional_options=additional_options,
        )
        self.buffer_size = buffer_size
        self.dataset = dataset
        self.num_workers = num_workers
        self.sharding_method = sharding_method

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """
        d = {
            "class": "MultiProcDataset",
            "dataset": self.dataset.as_returnn_opts(),
            "buffer_size": self.buffer_size,
            "num_workers": self.num_workers,
        }

        if self.sharding_method is not None:
            d["sharding_method"] = self.sharding_method

        return d
