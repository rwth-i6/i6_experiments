from typing import List, Optional, Dict, Any, Union, Callable

from sisyphus import Path, tk

from i6_experiments.common.setups.returnn.datasets.base import ControlDataset, Dataset


class CombinedDataset(Dataset):
    def __init__(
            self,
            *,
            datasets: Dict[str, Dataset],
            data_map: Dict,
            seq_ordering: str,
            partition_epoch: int,
    ):
        super().__init__(
            additional_options=None
        )
        self.datasets = datasets
        self.partition_epoch = partition_epoch
        self.data_map = data_map
        self.seq_ordering = seq_ordering

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """

        d = {
            "class": "CombinedDataset",
            "datasets": {k: v.as_returnn_opts() for k, v in self.datasets.items()},
            "data_map": self.data_map,
            "seq_ordering": self.seq_ordering,
            "partition_epoch": self.partition_epoch,
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
