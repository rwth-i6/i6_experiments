from typing import List, Optional, Dict, Any, Union, Callable

from sisyphus import Path, tk

from i6_experiments.common.setups.returnn.datasets.base import ControlDataset, Dataset


class DistributedFilesDataset(Dataset):
    def __init__(
            self,
            *,
            files: List[tk.Path],
            get_subepoch_dataset: Callable,
            partition_epoch: int = 1,
    ):
        super().__init__(
            additional_options=None
        )
        self.files = files
        self.partition_epoch = partition_epoch
        self.get_subepoch_dataset = get_subepoch_dataset

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """

        d = {
            "class": "DistributeFilesDataset",
            "files": self.files,
            "partition_epoch": self.partition_epoch,
            "get_sub_epoch_dataset": self.get_subepoch_dataset,
            "seq_ordering": "random",
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
