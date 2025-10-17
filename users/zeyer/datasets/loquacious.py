"""
Loquacious dataset

https://github.com/speechbrain/speechbrain/pull/2802/files

# TODO peak_normalization ?
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional, Any, Sequence, Dict, List
import os
import re
import numpy as np
import logging
from functools import partial, cache

from sisyphus import tk, Path

from i6_core.datasets.huggingface import TransformAndMapHuggingFaceDatasetJob, ExtractTextFromHuggingFaceDatasetJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

from i6_experiments.users.zeyer import tools_paths
from i6_experiments.users.zeyer.datasets.utils import multi_proc as mp_ds_utils

from returnn_common.datasets_old_2022_10.interface import VocabConfig, DatasetConfigStatic

from returnn.tensor import batch_dim, Dim

from .utils.spm import SentencePieceModel
from .task import Task, MeasureType, RecogOutput
from .utils.sclite_generic_score import generic_sclite_score_recog_out

if TYPE_CHECKING:
    import datasets


_alias_prefix = "datasets/Loquacious/"

logger = logging.getLogger(__name__)


def py():
    for name in ["small", "medium", "large"]:
        for q in [3, 4]:
            get_loquacious_hf_ogg(name, quality=q)
    get_hf_random_sorted_subset(get_loquacious_hf_ogg("large"), "train", take_n=5_000, alias_name="train_large_q3")
    get_hf_random_sorted_subset(get_loquacious_hf_ogg("large"), "dev", take_n=5_000, alias_name="dev_q3")
    get_train_corpus_text()
    get_hf_text_only()
    get_train_corpus_text("medium")
    get_spm_vocab(dim="10k")


@cache
def get_loquacious_hf_ogg(name: str = "large", *, quality: int = 3) -> Path:
    """
    Get the LoquaciousSet HF dataset as OGG files, with the specified subset (name) and (Ogg) quality.
    This is a DatasetDict with splits "train", "dev", "test".
    """
    ffmpeg_binary = tools_paths.get_ffmpeg_binary()

    job = TransformAndMapHuggingFaceDatasetJob(
        "speechbrain/LoquaciousSet",
        name,
        transform=_transform_rename_columns,
        map_func=partial(_map_func_wav_to_ogg, ffmpeg_binary=ffmpeg_binary, quality_opts=["-q", str(quality)]),
        map_opts=_map_opts,
    )
    job.rqmt.update({"cpu": 16, "time": 24, "mem": 48})
    job.add_alias(f"{_alias_prefix}dataset_hf_{name}_q{quality}_ogg")
    tk.register_output(f"{_alias_prefix}dataset_hf_{name}_q{quality}_ogg", job.out_dir)
    return job.out_dir


@cache
def get_hf_random_sorted_subset(
    path: Path,
    split: str,
    *,
    take_n: int,
    duration_key: str = "duration",
    random_seed: int = 42,
    alias_name: Optional[str] = None,
) -> Path:
    """
    Take some HF dataset path (e.g. via :func:`get_loquacious_hf_ogg`),
    shuffle it, take N seqs, and sort it by (reversed) duration,
    and store it as new HF dataset.
    """
    assert split in ("train", "dev", "test")
    job = TransformAndMapHuggingFaceDatasetJob(
        path,
        load_dataset_opts={"split": split},
        transform=partial(
            _hf_dataset_transform_random_sorted_subset,
            take_n=take_n,
            duration_key=duration_key,
            random_seed=random_seed,
        ),
    )
    if alias_name:
        job.add_alias(f"{_alias_prefix}dataset_hf_{alias_name}_random_sorted_subset_n{take_n}")
        tk.register_output(f"{_alias_prefix}dataset_hf_{alias_name}_random_sorted_subset_n{take_n}", job.out_dir)
    return job.out_dir


def _hf_dataset_transform_random_sorted_subset(
    ds: datasets.Dataset, *, take_n: int, duration_key: str = "duration", random_seed: int = 42
) -> datasets.Dataset:
    import datasets

    assert isinstance(ds, datasets.Dataset), f"expected datasets.Dataset, got {type(ds)} {ds}"
    # like ds.shuffle(...).take(...) but faster and more direct
    generator = np.random.default_rng(random_seed)
    permutation = generator.permutation(take_n)
    ds = ds.select(permutation)
    ds = ds.sort(duration_key, reverse=True)
    return ds


_eval_split_filters = {
    "voxpopuli": re.compile("PLENARY"),
    "commonvoice": re.compile("common_voice"),
    "librispeech": re.compile("^[0-9-]*$"),
    "yodas": re.compile(".wav$"),
}

EvalSubSplits = list(_eval_split_filters.keys())
DevSplits = [f"dev_{k}" for k in EvalSubSplits]
TestSplits = [f"test_{k}" for k in EvalSubSplits]


@cache
def get_hf_dataset_custom_split(path: Path, sub_split: str) -> Path:
    """
    partition the dev/test set into eval subsets
    """
    assert sub_split in EvalSubSplits
    job = TransformAndMapHuggingFaceDatasetJob(
        path,
        transform=partial(_hf_dataset_transform_filter_subset, sub_split=sub_split),
    )
    return job.out_dir


def _hf_dataset_transform_filter_subset(ds: datasets.Dataset, *, sub_split: str) -> datasets.Dataset:
    ds = ds.filter(partial(_hf_dataset_filter_subset, sub_split=sub_split))
    return ds


def _hf_dataset_filter_subset(example: Dict[str, Any], *, sub_split: str) -> bool:
    matches = {k: bool(pat.search(example["id"])) for k, pat in _eval_split_filters.items()}
    assert sum(matches.values()) <= 1, f"ambiguous {example['id']}: {matches}"
    return matches[sub_split]


@cache
def get_hf_text_only(name: str = "large") -> Path:
    """
    Remove the audio part, keep only text.
    """
    job = TransformAndMapHuggingFaceDatasetJob(
        "speechbrain/LoquaciousSet",
        name,
        transform=[_transform_rename_columns, _hf_dataset_remove_audio],
        map_func=_hf_dataset_map_add_num_words,
    )
    job.add_alias(f"{_alias_prefix}dataset_hf_text_only")
    tk.register_output(f"{_alias_prefix}dataset_hf_text_only", job.out_dir)
    return job.out_dir


def _hf_dataset_remove_audio(ds: datasets.DatasetDict) -> datasets.DatasetDict:
    return ds.remove_columns("audio")


def _hf_dataset_map_add_num_words(example: Dict[str, Any]) -> Dict[str, Any]:
    example["num_words"] = len(example["text"].split())
    return example


@cache
def get_train_corpus_text(name: str = "large", *, split: str = "train") -> Path:
    job = ExtractTextFromHuggingFaceDatasetJob("speechbrain/LoquaciousSet", name, split=split, column_name="text")
    job.add_alias(f"{_alias_prefix}{split}_{name}_corpus.txt.extract")
    tk.register_output(f"{_alias_prefix}{split}_{name}_corpus.txt.gz", job.out_text)
    return job.out_text


@cache
def get_spm_vocab(
    *, dim: Union[int, str], model_type: SentencePieceType = SentencePieceType.UNIGRAM
) -> SentencePieceModel:
    """
    Get a SentencePiece model and vocab of given dimension trained on the Loquacious transcriptions.
    """
    dim_str = str(dim)
    if isinstance(dim, str):
        # Not sure if power-of-two or just multiple-of-64, but 10240 has more 2s in it (2048*5) than 10048.
        dim = {"20k": 20_480, "10k": 10_240, "5k": 5_120, "4k": 4_096, "1k": 1_024, "512": 512, "128": 128, "64": 64}[
            dim
        ]
    assert isinstance(dim, int) and dim >= 10

    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    _spm_train_job = TrainSentencePieceJob(
        training_text=get_train_corpus_text(),
        vocab_size=dim,
        model_type=model_type,
        additional_options={
            "split_digits": True,
            "unk_id": 2,  # default is 0
            "bos_id": 1,  # default is 1
            "eos_id": 0,  # default is 2
            "train_extremely_large_corpus": True,
            "shuffle_input_sentence": True,
            "input_sentence_size": 10_000_000,
        },
    )
    _spm_train_job.rqmt.update({"time": 12, "mem": 126})  # needs much more mem, maybe little longer
    _spm_train_job.add_alias(f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train")
    tk.register_output(f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train.model", _spm_train_job.out_model)
    tk.register_output(
        f"{_alias_prefix}vocab/spm_{model_type.value}_{dim_str}_train.vocab",
        ExtractSentencePieceVocabJob(_spm_train_job.out_model).out_vocab,
    )
    spm = SentencePieceModel(
        dim=dim,
        model_file=_spm_train_job.out_model,
        unknown_label="<unk>",
        bos_idx=1,
        eos_idx=0,
    )
    return spm


@cache
def get_loquacious_task_raw(
    *,
    vocab: str,
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_seq_ordering: str = "laplace:.1000",
    multi_proc_dataset: Optional[Dict[str, Any]] = None,
) -> Task:
    """
    Get Loquacious task with raw audio input and text output.
    """
    vocab: VocabConfig = get_vocab_by_str(vocab)

    # See :func:`search_dataset`.
    # We first optionally do :func:`ctc_alignment_to_label_seq` if ``recog_def.output_blank_label`` is set.
    # (SearchCollapseRepeatedLabelsJob, SearchRemoveLabelJob).
    # Then ``recog_post_proc_funcs`` are applied.
    # Then SearchTakeBestJob.
    # Then, for Sclite scoring, there is SearchWordsDummyTimesToCTMJob.
    if isinstance(vocab, SentencePieceModel):
        recog_post_proc_funcs = [_spm_to_words]
    else:
        raise TypeError(f"unhandled vocab type {type(vocab)}")

    hf_data_dir = get_loquacious_hf_ogg()

    train_epoch_split = 25  # so one subepoch is approx 1000h
    train_vocab = vocab.copy(**train_vocab_opts) if train_vocab_opts else None
    train_dataset = _make_hf_dataset_train(
        hf_data_dir=hf_data_dir,
        vocab=vocab,
        train_vocab=train_vocab,
        train_epoch_split=train_epoch_split,
        train_seq_ordering=train_seq_ordering,
        multi_proc_dataset=multi_proc_dataset,
    )
    eval_datasets = {
        "dev": _make_hf_dataset(hf_data_dir=hf_data_dir, split="dev", vocab=vocab),
        "test": _make_hf_dataset(hf_data_dir=hf_data_dir, split="test", vocab=vocab),
    }
    dev_dataset = eval_datasets["dev"]

    task = Task(
        name="loquacious",
        train_dataset=train_dataset,
        train_epoch_split=train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="dev",
        score_recog_output_func=partial(generic_sclite_score_recog_out, post_proc_funcs=recog_post_proc_funcs),
        recog_post_proc_funcs=recog_post_proc_funcs,
    )
    return task


@cache
def get_loquacious_task_raw_v2(
    *,
    vocab: str,
    subset_name: str = "large",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_seq_ordering: str = "laplace:.1000",
    multi_proc: int = 2,
    train_epoch_split: int = 25,  # so one subepoch is approx 1000h
) -> Task:
    """
    v2: multiprocessing by default, subset of dev and devtrain, devtrain not via take_first_shard_subset.
    """
    vocab: VocabConfig = get_vocab_by_str(vocab)

    # See :func:`search_dataset`.
    # We first optionally do :func:`ctc_alignment_to_label_seq` if ``recog_def.output_blank_label`` is set.
    # (SearchCollapseRepeatedLabelsJob, SearchRemoveLabelJob).
    # Then ``recog_post_proc_funcs`` are applied.
    # Then SearchTakeBestJob.
    # Then, for Sclite scoring, there is SearchWordsDummyTimesToCTMJob.
    if isinstance(vocab, SentencePieceModel):
        recog_post_proc_funcs = [_spm_to_words]
    else:
        raise TypeError(f"unhandled vocab type {type(vocab)}")

    hf_data_dir = get_loquacious_hf_ogg(name=subset_name)

    train_vocab = vocab.copy(**train_vocab_opts) if train_vocab_opts else None
    train_dataset = _make_hf_dataset_train_v2(
        hf_data_dir=hf_data_dir,
        vocab=vocab,
        train_vocab=train_vocab,
        train_epoch_split=train_epoch_split,
        train_seq_ordering=train_seq_ordering,
        multi_proc=multi_proc,
    )
    eval_datasets = {
        "dev": _make_hf_dataset(hf_data_dir=hf_data_dir, split="dev", vocab=vocab),
        **{k: _make_hf_dataset(hf_data_dir=hf_data_dir, split=k, vocab=vocab) for k in DevSplits},
        "test": _make_hf_dataset(hf_data_dir=hf_data_dir, split="test", vocab=vocab),
        **{k: _make_hf_dataset(hf_data_dir=hf_data_dir, split=k, vocab=vocab) for k in TestSplits},
    }
    dev_dataset = eval_datasets["dev"]

    task = Task(
        name="loquacious",
        train_dataset=train_dataset,
        train_epoch_split=train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,
        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="dev",
        score_recog_output_func=partial(generic_sclite_score_recog_out, post_proc_funcs=recog_post_proc_funcs),
        recog_post_proc_funcs=recog_post_proc_funcs,
    )
    return task


@cache
def get_loquacious_train_subset_dataset(
    *, vocab: str, num_seqs: int = 100_000, multi_proc: int = 2
) -> DatasetConfigStatic:
    hf_data_dir = get_loquacious_hf_ogg(name="train")
    multi_proc_dataset = {"num_workers": multi_proc} if multi_proc >= 2 else None
    vocab_: VocabConfig = get_vocab_by_str(vocab)

    return _make_hf_dataset(
        hf_data_dir=hf_data_dir,
        split="train",
        vocab=vocab_,
        take_random_sorted_subset=num_seqs,
        multi_proc_dataset=multi_proc_dataset,
    )


def _make_hf_dataset_train(
    *,
    hf_data_dir: Path,
    vocab: VocabConfig,
    train_vocab: Optional[VocabConfig] = None,
    train_epoch_split: Optional[int] = None,
    train_seq_ordering: str = "random",
    multi_proc_dataset: Optional[Dict[str, Any]] = None,
) -> DatasetConfigStatic:
    train_ds = _make_hf_dataset(
        hf_data_dir=hf_data_dir,
        split="train",
        use_distrib_files=True,
        vocab=train_vocab or vocab,
        partition_epoch=train_epoch_split,
        seq_ordering=train_seq_ordering,
        multi_proc_dataset=multi_proc_dataset,
    )
    return DatasetConfigStatic(
        extern_data=train_ds.extern_data,
        default_input=train_ds.default_input,
        default_target=train_ds.default_target,
        train_dataset=train_ds.main_dataset,
        eval_datasets={
            "dev": _make_hf_dataset(
                hf_data_dir=hf_data_dir, split="dev", vocab=vocab, multi_proc_dataset=multi_proc_dataset
            ).main_dataset,
            "devtrain": _make_hf_dataset(
                hf_data_dir=hf_data_dir,
                split="train",
                vocab=vocab,
                take_first_shard_subset=True,
                multi_proc_dataset=multi_proc_dataset,
            ).main_dataset,
        },
        use_deep_copy=True,
    )


def _make_hf_dataset_train_v2(
    *,
    hf_data_dir: Path,
    vocab: VocabConfig,
    train_vocab: Optional[VocabConfig] = None,
    train_epoch_split: Optional[int] = None,
    train_seq_ordering: str,
    multi_proc: int = 2,
    # dev has 7759. take 5000 just for a nicer number.
    eval_take_random_sorted_subset: int = 5000,
) -> DatasetConfigStatic:
    multi_proc_dataset = {"num_workers": multi_proc} if multi_proc >= 2 else None

    train_ds = _make_hf_dataset(
        hf_data_dir=hf_data_dir,
        split="train",
        use_distrib_files=True,
        vocab=train_vocab or vocab,
        partition_epoch=train_epoch_split,
        seq_ordering=train_seq_ordering,
        multi_proc_dataset=multi_proc_dataset,
    )
    return DatasetConfigStatic(
        extern_data=train_ds.extern_data,
        default_input=train_ds.default_input,
        default_target=train_ds.default_target,
        train_dataset=train_ds.main_dataset,
        eval_datasets={
            "dev": _make_hf_dataset(
                hf_data_dir=hf_data_dir,
                split="dev",
                vocab=vocab,
                take_random_sorted_subset=eval_take_random_sorted_subset,
                multi_proc_dataset=multi_proc_dataset,
            ).main_dataset,
            "devtrain": _make_hf_dataset(
                hf_data_dir=hf_data_dir,
                split="train",
                vocab=vocab,
                take_random_sorted_subset=eval_take_random_sorted_subset,
                multi_proc_dataset=multi_proc_dataset,
            ).main_dataset,
        },
        use_deep_copy=True,
    )


def _make_hf_dataset(
    *,
    hf_data_dir: Path,
    split: Optional[str] = None,
    vocab: VocabConfig,
    seq_ordering: str = "sorted_reverse",
    partition_epoch: Optional[int] = None,
    use_distrib_files: bool = False,
    take_first_shard_subset: bool = False,
    take_random_sorted_subset: Optional[int] = None,
    multi_proc_dataset: Optional[Dict[str, Any]] = None,
) -> DatasetConfigStatic:
    vocab_opts = vocab.get_opts()
    extern_data_dict = {
        "audio": {"dtype": "float32", "dim_tags": [batch_dim, Dim(None, name="time")]},
        "text": {
            "dtype": "int32",
            "dim_tags": [batch_dim, Dim(None, name="text_spatial")],
            "sparse": True,
            "vocab": vocab_opts,
        },
    }
    if take_random_sorted_subset:
        assert not take_first_shard_subset
        hf_ds_opts = get_hf_random_sorted_subset(path=hf_data_dir, split=split, take_n=take_random_sorted_subset)
        if seq_ordering == "sorted_reverse":
            seq_ordering = "default"
    else:
        if split in ("train", "dev", "test"):
            hf_ds_opts = hf_data_dir.join_right(split)
        elif "_" in split:
            split1, split2 = split.split("_", 1)
            assert split1 in ("dev", "test")
            hf_ds_opts = get_hf_dataset_custom_split(hf_data_dir.join_right(split1), split2)
        else:
            assert split is None, f"invalid split {split!r}"
            hf_ds_opts = hf_data_dir
        if take_first_shard_subset:
            hf_ds_opts = partial(_hf_dataset_dir_take_first_shard, hf_ds_opts)

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
        "sorting_seq_len_column": "duration",
        "cast_columns": {"audio": {"_type": "Audio", "sample_rate": 16_000}},
        # Keep data_format consistent to extern_data_dict.
        "data_format": {
            "audio": {"dtype": "float32", "shape": [None]},
            "text": {"dtype": "int32", "shape": [None], "sparse": True, "vocab": vocab_opts},
        },
        "seq_ordering": seq_ordering,
    }

    if use_distrib_files:
        assert not take_first_shard_subset
        d["dataset_opts"] = None  # will be set _distribute_files_get_files
        del d["use_file_cache"]  # handled via _distribute_files_get_sub_epoch_dataset
        d = {
            "class": "DistributeFilesDataset",
            "files": partial(_distribute_files_get_files, hf_data_dir=hf_ds_opts),
            "get_sub_epoch_dataset": partial(
                _distribute_files_get_sub_epoch_dataset,
                base_dict=d,
                **({"multi_proc_dataset": multi_proc_dataset} if multi_proc_dataset else {}),
            ),
            "seq_ordering": "random",
        }

    if partition_epoch:
        d["partition_epoch"] = partition_epoch

    if multi_proc_dataset and not use_distrib_files:
        d = mp_ds_utils.multi_proc_dataset_opts(d, **multi_proc_dataset)

    return DatasetConfigStatic(
        main_name=split,
        main_dataset=d,
        extern_data=extern_data_dict,
        default_input="audio",
        default_target="text",
        use_deep_copy=True,
    )


def get_loquacious_text_only_dataset(
    *,
    vocab: str,
    train_epoch_split: Optional[int] = None,
    train_seq_ordering: str = "laplace:.1000",
    multi_proc: int = 2,
    # dev has 7759. take 5000 just for a nicer number.
    eval_take_random_sorted_subset: int = 5000,
) -> DatasetConfigStatic:
    vocab: VocabConfig = get_vocab_by_str(vocab)
    hf_data_dir = get_hf_text_only()
    multi_proc_dataset = {"num_workers": multi_proc} if multi_proc >= 2 else None

    train_ds = _make_hf_dataset_text_only(
        hf_data_dir=hf_data_dir,
        split="train",
        # Don't use_distrib_files, we only have 5 shard files, but we might want partition epoch 25.
        # use_distrib_files=True,
        vocab=vocab,
        partition_epoch=train_epoch_split,
        seq_ordering=train_seq_ordering,
        multi_proc_dataset=multi_proc_dataset,
    )
    return DatasetConfigStatic(
        extern_data=train_ds.extern_data,
        default_input=train_ds.default_input,
        default_target=train_ds.default_target,
        train_dataset=train_ds.main_dataset,
        eval_datasets={
            "dev": _make_hf_dataset_text_only(
                hf_data_dir=hf_data_dir,
                split="dev",
                vocab=vocab,
                take_random_sorted_subset=eval_take_random_sorted_subset,
                multi_proc_dataset=multi_proc_dataset,
            ).main_dataset,
            "devtrain": _make_hf_dataset_text_only(
                hf_data_dir=hf_data_dir,
                split="train",
                vocab=vocab,
                take_random_sorted_subset=eval_take_random_sorted_subset,
                multi_proc_dataset=multi_proc_dataset,
            ).main_dataset,
        },
        use_deep_copy=True,
    )


def _make_hf_dataset_text_only(
    *,
    hf_data_dir: Path,
    split: str,
    vocab: VocabConfig,
    seq_ordering: str = "sorted_reverse",
    partition_epoch: Optional[int] = None,
    use_distrib_files: bool = False,
    take_random_sorted_subset: Optional[int] = None,
    multi_proc_dataset: Optional[Dict[str, Any]] = None,
) -> DatasetConfigStatic:
    vocab_opts = vocab.get_opts()
    extern_data_dict = {
        "text": {
            "dtype": "int32",
            "dim_tags": [batch_dim, Dim(None, name="text_spatial")],
            "sparse": True,
            "vocab": vocab_opts,
        },
    }
    if take_random_sorted_subset:
        hf_ds_opts = get_hf_random_sorted_subset(path=hf_data_dir, split=split, take_n=take_random_sorted_subset)
        if seq_ordering == "sorted_reverse":
            seq_ordering = "default"
    else:
        hf_ds_opts = hf_data_dir.join_right(split)

    d = {
        "class": "HuggingFaceDataset",
        "dataset_opts": hf_ds_opts,
        "use_file_cache": True,
        "seq_tag_column": "id",
        "sorting_seq_len_column": "num_words",
        # Keep data_format consistent to extern_data_dict.
        "data_format": {
            "text": {"dtype": "int32", "shape": [None], "sparse": True, "vocab": vocab_opts},
        },
        "seq_ordering": seq_ordering,
    }

    if use_distrib_files:
        d["dataset_opts"] = None  # will be set _distribute_files_get_files
        del d["use_file_cache"]  # handled via _distribute_files_get_sub_epoch_dataset
        d = {
            "class": "DistributeFilesDataset",
            "files": partial(_distribute_files_get_files, hf_data_dir=hf_ds_opts),
            "get_sub_epoch_dataset": partial(
                _distribute_files_get_sub_epoch_dataset,
                base_dict=d,
                **({"multi_proc_dataset": multi_proc_dataset} if multi_proc_dataset else {}),
            ),
            "seq_ordering": "random",
        }

    if partition_epoch:
        d["partition_epoch"] = partition_epoch

    if multi_proc_dataset and not use_distrib_files:
        d = mp_ds_utils.multi_proc_dataset_opts(d, **multi_proc_dataset)

    return DatasetConfigStatic(
        main_name=split,
        main_dataset=d,
        extern_data=extern_data_dict,
        default_input="text",
        default_target="text",
        use_deep_copy=True,
    )


def _distribute_files_get_files(hf_data_dir: Union[Path, str, os.PathLike]) -> List[Union[Path, str]]:
    from returnn.datasets.huggingface import get_arrow_shard_files_from_hf_dataset_dir

    return get_arrow_shard_files_from_hf_dataset_dir(hf_data_dir)


def _distribute_files_get_sub_epoch_dataset(
    files: List[str], *, base_dict: Dict[str, Any], multi_proc_dataset: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    from returnn.util.file_cache import CachedFile

    d = base_dict.copy()
    d["dataset_opts"] = [CachedFile(fn) for fn in files]

    if multi_proc_dataset is not None:
        d = mp_ds_utils.multi_proc_dataset_opts(d, **multi_proc_dataset)

    return d


def _hf_dataset_dir_take_first_shard(hf_data_dir: Union[Path, str, os.PathLike]) -> List[str]:
    hf_data_dir = os.fspath(hf_data_dir)
    content = os.listdir(hf_data_dir)
    assert "state.json" in content
    assert "dataset_info.json" in content
    content = [fn for fn in content if fn.startswith("data-") and fn.endswith(".arrow")]
    assert content, f"no .arrow files found in {hf_data_dir!r}"
    pat_first = re.compile("^data-(0+)-of-([0-9]+).arrow$")
    content = [pat_first.match(fn) for fn in content]
    content = list(filter(None, content))
    assert len(content) == 1, f"expected exactly one shard file in {hf_data_dir!r}, got {content}"
    return [hf_data_dir + "/" + content[0].group(0)]


@cache
def get_vocab_by_str(vocab: str) -> Union[VocabConfig, SentencePieceModel]:
    """
    Get vocab
    """
    if re.match("^spm[0-9]+.*$", vocab):
        return get_spm_vocab(dim=vocab[len("spm") :], model_type=SentencePieceType.UNIGRAM)
    elif re.match("^spmLm[0-9]+.*$", vocab):
        return get_spm_vocab(dim=vocab[len("spmLm") :], model_type=SentencePieceType.UNIGRAM, train_full=True)
    elif re.match("^spm_bpe[0-9]+.*$", vocab):
        return get_spm_vocab(dim=vocab[len("spm_bpe") :], model_type=SentencePieceType.BPE)
    else:
        raise ValueError(f"invalid vocab {vocab!r}")


def _spm_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    from i6_core.returnn.search import SearchOutputRawReplaceJob

    words = SearchOutputRawReplaceJob(bpe.output, [(" ", ""), ("▁", " ")], output_gzip=True).out_search_results
    return RecogOutput(output=words)


def _transform_rename_columns(ds: datasets.DatasetDict) -> datasets.DatasetDict:
    return ds.rename_columns({"ID": "id", "wav": "audio"})


def _map_func_wav_to_ogg(
    data: Dict[str, Any], *, ffmpeg_binary: Union[str, Path], quality_opts: Sequence[str]
) -> Dict[str, Any]:
    import subprocess

    proc_res = subprocess.run(
        [
            ffmpeg_binary.get_path() if isinstance(ffmpeg_binary, Path) else ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-c:a",
            "libvorbis",
            *quality_opts,
            "-f",
            "ogg",
            "-",
        ],
        input=data["audio"]["bytes"],
        stdout=subprocess.PIPE,
        check=True,
    )
    data["audio"]["bytes"] = proc_res.stdout
    return data


def _map_opts(ds: datasets.DatasetDict) -> Dict[str, Any]:
    from datasets import Audio

    features = ds["train"].features.copy()
    audio_feat = features["audio"]
    assert isinstance(audio_feat, Audio)
    audio_feat.decode = True
    return {"features": features}


class TextNormaliser:
    """
    Used to normalise text with custom rules, and with the Nemo text normalisation tool as base.

    This code is copied from here: https://github.com/speechbrain/speechbrain/pull/2802/files
    And then adapted.

    Authors
    * Titouan Parcollet 2024
    """

    def __init__(self):
        # pip install nemo-text-processing==1.1.0
        from nemo_text_processing.text_normalization.normalize import Normalizer

        self._normaliser = Normalizer(input_case="cased", lang="en")

    def __call__(self, words: str) -> Optional[str]:
        words = self._normaliser.normalize(words)
        words = self._english_specific_preprocess(words)
        return words

    @classmethod
    def _english_specific_preprocess(
        cls, sentence: str, upper_case: bool = True, symbols_limit: int = 4
    ) -> Optional[str]:
        """
        Preprocess English text.
        This function relies on different tools to convert numerals and special symbols.
        This also removes various punctuation and treats it as word boundaries.
        It normalises and retains various apostrophes (’‘´) between letters,
        but not other ones, which are probably quotation marks.
        It capitalises all text.
        This function may error out if new characters show up in the given sentence.

        Note that this does not provide any numeral conversion.
        This must be done beforehand with, for instance the Nemo text processing tool.

        Parameters
        ----------
        sentence : str
            The string to modify.
        upper_case : bool
            Whether to upper case (if True) or lower case (if False) the string.
        symbols_limit : int
            If a sentence contains more than symbols_limit, it will not be normalised and skipped.
            This is because in most case, the pronunciation will not be certain enough.
        Returns
        -------
        str
            The normalised sentence. Returns None if it was not possible to
            normalise the sentence.

        Example
        -------
        >>> norm = TextNormaliser()
        >>> txt = norm._english_specific_preprocess("Over the Rainbow! How are you today? Good + one hundred %")
        >>> print(txt)
        OVER THE RAINBOW HOW ARE YOU TODAY GOOD PLUS ONE HUNDRED PERCENT
        >>> txt = norm._english_specific_preprocess("Over the Rainbow! How are you today? Good + 100 %")
        >>> print(txt)
        None
        """

        # These characters mean we should discard the sentence, because the
        # pronunciation will be too uncertain.
        # if the sentence contains number we simply remove it.
        # This is because we expect the user to provide only text and symbols.
        # Numerals can be converted using the NeMo text processing tool.
        stop_characters = (
            "["
            "áÁàăâåäÄãÃāảạæćčČçÇðéÉèÈêěëęēəğíîÎïīịıłṃńňñóÓòôőõøØōŌœŒřšŠşșȘúÚûūụýžþ"
            # Suggests the sentence is not English but German.
            "öÖßüÜ"
            # All sorts of languages: Greek, Arabic...
            "\u0370-\u1aaf"
            # Chinese/Japanese/Korean.
            "\u4e00-\u9fff"
            # Technical symbols.
            "\u2190-\u23ff"
            # Symbols that could be pronounced in various ways.
            "]"
        )
        if re.search(stop_characters, sentence) is not None:
            return None

        # encoding goes brrrrr
        sentence = cls._clean_text(sentence)

        # These characters mark word boundaries.
        split_character_regex = '[ ",:;!?¡\\.…()\\-—–‑_“”„/«»]'

        # These could all be used as apostrophes in the middle of words.
        # If at the start or end of a word, they will be removed.
        apostrophes_or_quotes = "['`´ʻ‘’]"

        # Just in case Nemo missed it...
        sentence_level_mapping = {
            "&": " and ",
            "+": " plus ",
            "ﬂ": "fl",
            "%": " percent ",
            "=": " equal ",
            "@": " at ",
            "#": " hash ",
            "$": " dollar ",
            "}": "",
            "{": "",
            "\\": "",
            "|": "",
            "[": "",
            "]": "",
            "~": "",
            "^": "",
            "*": "",
            "•": "",
        }

        # Remove sentences that contain too many symbols.
        symbol_list = list(sentence_level_mapping.keys())
        if cls.count_symbols_in_str(sentence, symbol_list) >= symbols_limit:
            return None

        final_characters = set(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

        sentence_mapped = sentence
        if any((source in sentence) for source in sentence_level_mapping):
            for source, target in sentence_level_mapping.items():
                sentence_mapped = sentence_mapped.replace(source, target)

        # Some punctuation that indicates a word boundary.
        words_split = re.split(split_character_regex, sentence_mapped)
        words_quotes = [
            # Use ' as apostrophe.
            # Remove apostrophes at the start and end of words (probably quotes).
            # Word-internal apostrophes, even where rotated, are retained.
            re.sub(apostrophes_or_quotes, "'", word).strip("'")
            for word in words_split
        ]

        # Processing that does not change the length.
        if upper_case:
            words_upper = [word.upper() for word in words_quotes]
        else:
            words_upper = [word.lower() for word in words_quotes]

        words_mapped = [
            # word.translate(character_mapping)
            word
            for word in words_upper
            # Previous processing may have reduced words to nothing.
            # Remove them.
            if word != ""
        ]

        result = " ".join(words_mapped)
        character_set = set(result)

        if not character_set <= final_characters:
            logger.warning("Sentence not properly normalised and removed: " + result)
            return None
        else:
            return result

    @staticmethod
    def _clean_text(text: str) -> str:
        """Some sentences are poorly decoded from people's speech or yodas. This
        removes these char in the text.

        """
        unwanted_char = (
            "\u0159\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n"
            "\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19"
            "\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
            "\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"
            "\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"
            "\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"
            "\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"
            "\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"
            "\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"
            "\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff"
        )
        text = "".join([(" " if n in unwanted_char else n) for n in text if n not in unwanted_char])
        return text

    @staticmethod
    def count_symbols_in_str(sentence: str, symbols: List[str]) -> int:
        """Count the total number of symbols occurring in a string from a list of
        symbols

        Parameters
        ----------
        sentence : str
            The string to check.
        symbols : list
            List of symbols to count.

        Returns
        -------
        int
            The total count

        """
        cpt = 0

        for symbol in symbols:
            cpt += sentence.count(symbol)

        return cpt
