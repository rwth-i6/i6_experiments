"""
Helper classes around RETURNN datasets
"""
__all__ = ["OggZipDataset"]

from sisyphus import tk
from typing import *

from i6_experiments.users.rossenbach.common_setups.returnn.datasets.base import ControlDataset


class OggZipDataset(ControlDataset):
    """
    Represents :class:`OggZipDataset` in RETURNN
    `BlissToOggZipJob` job is used to convert some bliss xml corpus to ogg zip files

    Example code for the OggZipDataset:

        train_ogg_zip_dataset = OggZipDataset(
            path=zip_dataset,
            audio_options={
                "window_len": 0.025,
                "step_len": 0.01,
                "num_feature_filters": 40,
                "features": "mfcc",
                "norm_mean": extract_statistics_job.out_mean_file,
                "norm_std_dev": extract_statistics_job.out_std_dev_file
            },
            target_options={
                "class": "BytePairEncoding",
                "bpe_file": returnn_train_bpe_job.out_bpe_codes,
                "vocab_file": returnn_train_bpe_job.out_bpe_vocab,
                "unknown_label": None,
                "seq_postfix": [0],
            },
            segment_file=train_segments,
            partition_epoch=2,
            seq_ordering="laplace:.1000"
        )
    """

    def __init__(
        self,
        *,
        path: Union[List[tk.Path], tk.Path],
        audio_options: Optional[Dict[str, Any]] = None,
        target_options: Optional[Dict[str, Any]] = None,
        segment_file: Optional[tk.Path] = None,
        # super parameters
        partition_epoch: Optional[int] = None,
        seq_ordering: Optional[str] = None,
        random_subset: Optional[int] = None,
        # super-super parameters
        additional_options: Optional[Dict] = None,
    ):
        """
        :param path: ogg zip files path
        :param audio_options: parameters passed to the "audio" field of the dataset, used for the
            "ExtractAudioFeatures" class in RETURNN
        :param target_options: parameters passed to the "targets" field of the dataset, used for the
            initialization of the vocabulary via `Vocabulary.create_vocab" in RETURNN.
        :param partition_epoch: partition the data into N parts
        :param seq_list_filter_file: text file (gzip/plain) or pkl containg list of sequence tags to use
        :param seq_ordering: see `https://returnn.readthedocs.io/en/latest/dataset_reference/index.html`_.
        :param random_subset: take a random subset of the data, this is typically used for "dev-train", a part
            of the training data which is used to see training scores without data augmentation
        :param additional_options: custom options directly passed to the dataset
        """
        super().__init__(
            partition_epoch=partition_epoch,
            seq_list_filter_file=None,  # OggZipDataset has custom seq filtering logic
            seq_ordering=seq_ordering,
            random_subset=random_subset,
            additional_options=additional_options,
        )
        self.path = path
        self.audio_options = audio_options
        self.target_options = target_options
        self.segment_file = segment_file

    def as_returnn_opts(self):
        d = {
            "class": "OggZipDataset",
            "path": self.path[0]
            if isinstance(self.path, list) and len(self.path) == 1
            else self.path,
            "use_cache_manager": True,
            "audio": self.audio_options,
            "targets": self.target_options,
            "segment_file": self.segment_file,
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s"
            % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
