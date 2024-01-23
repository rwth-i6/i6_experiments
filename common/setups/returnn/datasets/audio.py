"""
Dataset helpers for datasets containing audio related data
"""
__all__ = ["OggZipDataset"]

from sisyphus import tk
from typing import Any, Dict, List, Optional, Union

from .base import ControlDataset


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
        files: Union[List[tk.Path], tk.Path],
        audio_options: Optional[Dict[str, Any]] = None,
        target_options: Optional[Dict[str, Any]] = None,
        segment_file: Optional[tk.Path] = None,
        # super parameters
        partition_epoch: Optional[int] = None,
        seq_ordering: Optional[str] = None,
        random_subset: Optional[int] = None,
        # super-super parameters
        additional_options: Optional[Dict[str, Any]] = None,
    ):
        """
        :param files: one or multiple ogg zip file paths, RETURNN parameter name is "path"
        :param audio_options: parameters passed to the "audio" field of the dataset, used for the
            "ExtractAudioFeatures" class in RETURNN
        :param target_options: parameters passed to the "targets" field of the dataset, used for the
            initialization of the vocabulary via `Vocabulary.create_vocab" in RETURNN.
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
            segment_file=None,  # OggZipDataset has custom seq filtering logic
            seq_ordering=seq_ordering,
            random_subset=random_subset,
            additional_options=additional_options,
        )
        self.files = files
        self.audio_options = audio_options
        self.target_options = target_options
        self.segment_file = segment_file

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """
        d = {
            "class": "OggZipDataset",
            "path": self.files[0] if isinstance(self.files, list) and len(self.files) == 1 else self.files,
            "use_cache_manager": True,
            "audio": self.audio_options,
            "targets": self.target_options,
            "segment_file": self.segment_file,
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
