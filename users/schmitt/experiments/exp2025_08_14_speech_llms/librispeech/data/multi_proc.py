"""
Helper classes around RETURNN datasets for arbitrary data
"""
__all__ = ["MultiProcDataset"]

from sisyphus import tk
from typing import Any, Dict, List, Optional, Union

from i6_experiments.common.setups.returnn.datasets.base import Dataset


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

        return d
