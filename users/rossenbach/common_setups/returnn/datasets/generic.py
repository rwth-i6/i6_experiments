__all__ = ["HDFDataset"]

from typing import Any, Dict, List, Optional, Union

from sisyphus import tk

from i6_experiments.users.rossenbach.common_setups.returnn.datasets.base import ControlDataset


class HDFDataset(ControlDataset):
    """
    Helper class for the HDF Dataset
    """

    def __init__(
        self,
        *,
        files: Union[List[tk.Path], tk.Path],
        # super parameters
        partition_epoch: Optional[int] = None,
        seq_list_filter_file: Optional[tk.Path] = None,
        seq_ordering: Optional[str] = None,
        additional_options: Optional[Dict[str, Any]] = None,
    ):
        """
        :param files: file or list of files to hdf files
        :param partition_epoch: partition the data into N parts
        :param seq_list_filter_file: text file (gzip/plain) or pkl containg list of sequence tags to use
        :param seq_ordering: see `https://returnn.readthedocs.io/en/latest/dataset_reference/index.html`_.
        :param additional_options: custom options directly passed to the dataset
        """
        super().__init__(
            partition_epoch=partition_epoch,
            seq_list_filter_file=seq_list_filter_file,
            seq_ordering=seq_ordering,
            additional_options=additional_options,
        )
        self.files = files

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "class": "HDFDataset",
            "files": self.files if isinstance(self.files, list) else [self.files],
            "use_cache_manager": True,
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s"
            % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
