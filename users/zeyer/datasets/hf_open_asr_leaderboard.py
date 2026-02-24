"""
https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
https://github.com/huggingface/open_asr_leaderboard
"""

from sisyphus import tk
import functools
from typing import Any, Dict, Optional
from functools import cache

import datasets

from i6_core.datasets.huggingface import TransformAndMapHuggingFaceDatasetJob

from returnn_common.datasets_old_2022_10.interface import VocabConfig, DatasetConfig

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_hf_random_sorted_subset,
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
        split: str,
        vocab: VocabConfig,
        seq_ordering: str = "sorted_reverse",
        take_first_shard_subset: bool = False,
        take_random_sorted_subset: Optional[int] = None,
        take_random_sorted_subset_version: int = 1,
        sorting_seq_len_column: str = "duration_ms",
    ):
        self.hf_data_dir = hf_data_dir
        self.split = split
        self.vocab = vocab
        self.seq_ordering = seq_ordering
        self.take_first_shard_subset = take_first_shard_subset
        self.take_random_sorted_subset = take_random_sorted_subset
        self.take_random_sorted_subset_version = take_random_sorted_subset_version
        self.sorting_seq_len_column = sorting_seq_len_column

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """
        if self.take_random_sorted_subset:
            assert not self.take_first_shard_subset
            hf_ds_opts = {1: get_hf_random_sorted_subset, 2: get_hf_random_sorted_subset_v2}[
                self.take_random_sorted_subset_version
            ](path=self.hf_data_dir, split=self.split, take_n=self.take_random_sorted_subset)
            if self.seq_ordering == "sorted_reverse":
                seq_ordering = "default"
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
                "text": {
                    "dtype": "int32",
                    "shape": [None],
                    "sparse": True,
                    "vocab": self.vocab.get_opts(),
                },
            },
            "seq_ordering": self.seq_ordering,
        }

        return d


# --------------------------- Helper functions  -----------------------------------


def _map_opts(ds: datasets.DatasetDict) -> Dict[str, Any]:
    from datasets import Audio

    features = ds["validation"].features.copy()
    audio_feat = features["audio"]
    assert isinstance(audio_feat, Audio)
    audio_feat.decode = True
    return {"features": features}


# --------------------------- Dataset functions  -----------------------------------


@cache
def get_people_speech_hf_data_dir() -> tk.Path:
    """
    Get the PeoplesSpeech HF dataset as OGG files, with the specified subset (name) and (Ogg) quality.
    """

    __alias_prefix = f"{_alias_prefix}/PeoplesSpeech"
    name = "validation"
    job = TransformAndMapHuggingFaceDatasetJob(
        "MLCommons/peoples_speech",
        name,
        # map_opts=_map_opts,
    )
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
    job = TransformAndMapHuggingFaceDatasetJob(
        "hf-audio/esb-datasets-test-only-sorted",
        name,
    )
    job.rqmt.update({"cpu": 16, "time": 2, "mem": 48})
    job.add_alias(f"{__alias_prefix}_dataset_hf_{name}")
    tk.register_output(f"{__alias_prefix}_dataset_hf_{name}", job.out_dir)
    return job.out_dir


def build_people_speech_test_datasets(*, vocab: VocabConfig) -> Dict[str, HuggingFaceDataset]:
    hf_data_dir_peoples_speech = get_people_speech_hf_data_dir()
    eval_datasets = {
        "ps_validation": HuggingFaceDataset(hf_data_dir=hf_data_dir_peoples_speech, split="validation", vocab=vocab),
    }
    return eval_datasets


def build_asr_leaderboard_test_datasets(*, vocab: VocabConfig) -> Dict[str, HuggingFaceDataset]:
    eval_datasets = {}

    for name, splits in _names_and_splits:
        hf_data_dir = get_asr_leaderboard_hf_data_dir(name)
        for split in splits:
            eval_datasets[f"{name}.{split}"] = HuggingFaceDataset(
                hf_data_dir=hf_data_dir,
                split=split,
                vocab=vocab,
                sorting_seq_len_column="audio_length_s",
            )

    return eval_datasets
