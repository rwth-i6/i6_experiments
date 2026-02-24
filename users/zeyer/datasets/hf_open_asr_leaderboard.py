"""
https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
https://github.com/huggingface/open_asr_leaderboard
"""

from sisyphus import tk
import functools
from typing import Any, Dict, Optional
from functools import cache

from i6_core.datasets.huggingface import TransformAndMapHuggingFaceDatasetJob

from returnn_common.datasets_old_2022_10.interface import VocabConfig, DatasetConfig

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_hf_random_sorted_subset_v2,
    get_hf_dataset_custom_split,
    _hf_dataset_dir_take_first_shard,
)

_alias_prefix = "datasets/"
_names_and_splits = (
    ("gigaspeech", ("test",)),
    ("ami", ("test",)),
    ("earnings22", ("test",)),
    ("librispeech", ("test.clean", "test.other")),
    ("spgispeech", ("test",)),
    ("tedlium", ("test",)),
    ("voxpopuli", ("test",)),
)


class HuggingFaceDataset(DatasetConfig):
    """ """

    def __init__(
        self,
        *,
        hf_data_dir: tk.Path,
        name: str,
        split: str,
        vocab: VocabConfig,
        seq_ordering: str = "sorted_reverse",
        take_first_shard_subset: bool = False,
        take_random_sorted_subset: Optional[int] = None,
        sorting_seq_len_column: str = "duration_ms",
    ):
        self.hf_data_dir = hf_data_dir
        self.name = name
        self.split = split
        self.vocab = vocab
        self.seq_ordering = seq_ordering
        self.take_first_shard_subset = take_first_shard_subset
        self.take_random_sorted_subset = take_random_sorted_subset
        self.sorting_seq_len_column = sorting_seq_len_column

    def get_main_name(self) -> str:
        """
        See `Dataset` definition
        """
        return f"{self.name}.{self.split}"

    def get_default_input(self) -> Optional[str]:
        """
        See `Dataset` definition
        """
        return "audio"

    def get_default_target(self) -> Optional[str]:
        """
        See `Dataset` definition
        """
        return "text"

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        """
        See `Dataset` definition
        """
        return {
            "audio": {"dtype": "float32", "shape": [None]},
            "text": {"dtype": "int32", "shape": [None], "sparse": True, "vocab": self.vocab.get_opts()},
        }

    def get_main_dataset(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """
        if self.take_random_sorted_subset:
            assert not self.take_first_shard_subset
            hf_ds_opts = get_hf_random_sorted_subset_v2(
                path=self.hf_data_dir, split=self.split, take_n=self.take_random_sorted_subset
            )
        else:
            if self.split in ("validation", "test", "test.clean", "test.other"):
                hf_ds_opts = self.hf_data_dir.join_right(self.split)
            elif self.split is not None and "_" in self.split:
                split1, split2 = self.split.split("_", 1)
                assert split1 in ("dev", "test")
                hf_ds_opts = get_hf_dataset_custom_split(self.hf_data_dir.join_right(split1), split2)
            else:
                assert self.split is None, f"invalid split {self.split!r}"
                hf_ds_opts = self.hf_data_dir
            if self.take_first_shard_subset:
                hf_ds_opts = functools.partial(_hf_dataset_dir_take_first_shard, hf_ds_opts)

        d = {
            "class": "HuggingFaceDataset",
            "dataset_opts": hf_ds_opts,
            "use_file_cache": True,
            # {'id': Value(dtype='string', id=None),
            #  'duration': Value(dtype='float32', id=None),
            #  'audio': Audio(sampling_rate=None, mono=True, decode=True, id=None),
            #  'spk_id': Value(dtype='string', id=None),
            #  'sex': Value(dtype='string', id=None),
            #  'text': Value(dtype='string', id=None)}
            "seq_tag_column": "id",
            "sorting_seq_len_column": self.sorting_seq_len_column,
            "cast_columns": {"audio": {"_type": "Audio", "sample_rate": 16_000}},
            # Keep data_format consistent to extern_data_dict.
            "data_format": {
                "audio": {"dtype": "float32", "shape": [None]},
                "text": {"dtype": "int32", "shape": [None], "sparse": True, "vocab": self.vocab.get_opts()},
            },
            "seq_ordering": self.seq_ordering,
        }

        return d


# --------------------------- Dataset functions  -----------------------------------


@cache
def get_peoples_speech_hf_data_dir() -> tk.Path:
    """
    Get the PeoplesSpeech HF dataset as OGG files, with the specified subset (name) and (Ogg) quality.
    """

    __alias_prefix = f"{_alias_prefix}/PeoplesSpeech"
    name = "validation"
    job = TransformAndMapHuggingFaceDatasetJob("MLCommons/peoples_speech", name)
    job.rqmt.update({"cpu": 16, "time": 2, "mem": 48})
    job.add_alias(f"{__alias_prefix}dataset_hf_{name}")
    tk.register_output(f"{__alias_prefix}dataset_hf_{name}", job.out_dir)
    return job.out_dir


@cache
def get_asr_leaderboard_hf_data_dir(name: str) -> tk.Path:
    """
    Get the ASR leaderboard HF datasets, with the specified subset (name).
    """

    __alias_prefix = f"{_alias_prefix}/esb-datasets-test-only"
    job = TransformAndMapHuggingFaceDatasetJob("hf-audio/esb-datasets-test-only-sorted", name)
    job.rqmt.update({"cpu": 16, "time": 2, "mem": 48})
    job.add_alias(f"{__alias_prefix}_dataset_hf_{name}")
    tk.register_output(f"{__alias_prefix}_dataset_hf_{name}", job.out_dir)
    return job.out_dir


def build_peoples_speech_test_datasets(*, vocab: VocabConfig) -> Dict[str, HuggingFaceDataset]:
    hf_data_dir_peoples_speech = get_peoples_speech_hf_data_dir()
    eval_datasets = {
        "ps_validation": HuggingFaceDataset(
            hf_data_dir=hf_data_dir_peoples_speech, name="peoples_speech", split="validation", vocab=vocab
        ),
    }
    return eval_datasets


def build_asr_leaderboard_test_datasets(*, vocab: VocabConfig) -> Dict[str, HuggingFaceDataset]:
    eval_datasets = {}

    for name, splits in _names_and_splits:
        hf_data_dir = get_asr_leaderboard_hf_data_dir(name)
        for split in splits:
            eval_datasets[f"{name}.{split}"] = HuggingFaceDataset(
                hf_data_dir=hf_data_dir,
                name=name,
                split=split,
                vocab=vocab,
                sorting_seq_len_column="audio_length_s",
            )

    return eval_datasets
