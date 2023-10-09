"""
Helper classes around RETURNN datasets for arbitrary data
"""
__all__ = ["HDFDataset"]

from sisyphus import tk
from typing import Any, Dict, List, Optional, Union

from .base import ControlDataset


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
        segment_file: Optional[tk.Path] = None,
        seq_ordering: Optional[str] = None,
        random_subset: Optional[int] = None,
        additional_options: Optional[Dict[str, Any]] = None,
    ):
        """
        :param files: file or list of files to hdf files
        :param partition_epoch: partition the data into N parts
        :param segment_file: text file (gzip/plain) or pkl containing list of sequence tags to use,
          maps to "seq_list_filter_file" internally.
        :param seq_ordering: see `https://returnn.readthedocs.io/en/latest/dataset_reference/index.html`_.
        :param random_subset: take a random subset of the data, this is typically used for "dev-train", a part
            of the training data which is used to see training scores without data augmentation
        :param additional_options: custom options directly passed to the dataset
        """
        super().__init__(
            partition_epoch=partition_epoch,
            segment_file=segment_file,
            seq_ordering=seq_ordering,
            random_subset=random_subset,
            additional_options=additional_options,
        )
        self.files = files

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """
        d = {
            "class": "HDFDataset",
            "files": self.files if isinstance(self.files, list) else [self.files],
            "use_cache_manager": True,
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
