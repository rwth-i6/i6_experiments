"""
Helper classes around RETURNN datasets
"""
from sisyphus import tk
from typing import *

from i6_experiments.users.hilmes.data.datasets.base import ControlDataset


class OggZipDataset(ControlDataset):
    """
    Represents :class:`OggZipDataset` in RETURNN
    `BlissToOggZipJob` job is used to convert some bliss xml corpus to ogg zip files
    """

    def __init__(
        self,
        *,
        path: Union[List[tk.Path], tk.Path],
        audio_opts: Optional[Dict[str, Any]] = None,
        target_opts: Optional[Dict[str, Any]] = None,
        segment_file: Optional[tk.Path] = None,
        # super parameters
        partition_epoch: Optional[int] = None,
        seq_ordering: Optional[str] = None,
        # super-super parameters
        other_opts: Optional[Dict] = None,
    ):
        """
        :param path: ogg zip files path
        :param audio_opts: used to for feature extraction
        :param target_opts: used to create target labels
        :param partition_epoch: partition the data into N parts
        :param seq_list_filter_file: text file (gzip/plain) or pkl containg list of sequence tags to use
        :param seq_ordering: see `https://returnn.readthedocs.io/en/latest/dataset_reference/index.html`_.
        :param other_opts: custom options directly passed to the dataset
        """
        super().__init__(
            partition_epoch=partition_epoch,
            seq_list_filter_file=None,  # OggZipDataset has custom seq filtering logic
            seq_ordering=seq_ordering,
            other_opts=other_opts,
        )
        self.path = path
        self.audio_opts = audio_opts
        self.target_opts = target_opts
        self.segment_file = segment_file

    def as_returnn_opts(self):
        d = {
            "class": "OggZipDataset",
            "path": self.path[0]
            if isinstance(self.path, list) and len(self.path) == 1
            else self.path,
            "use_cache_manager": True,
            "audio": self.audio_opts,
            "targets": self.target_opts,
            "segment_file": self.segment_file,
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s"
            % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
