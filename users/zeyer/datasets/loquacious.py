"""
Loquacious dataset
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union, Optional, Any, Sequence, Dict, List
import re
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


def py():
    for name in ["small", "medium", "large"]:
        for q in [3, 4]:
            get_loquacious_hf_ogg(name, quality=q)
    get_train_corpus_text()
    get_train_corpus_text("medium")
    get_spm_vocab(dim="10k")


@cache
def get_loquacious_hf_ogg(name: str = "large", *, quality: int = 3) -> Path:
    ffmpeg_binary = tools_paths.get_ffmpeg_binary()

    job = TransformAndMapHuggingFaceDatasetJob(
        "speechbrain/LoquaciousSet",
        name,
        transform=_transform_rename_columns,
        map_func=partial(_map_func_wav_to_ogg, ffmpeg_binary=ffmpeg_binary, quality_opts=["-q", str(quality)]),
        map_opts=_map_opts,
    )
    job.rqmt.update({"cpu": 32, "time": 24, "mem": 32})
    job.add_alias(_alias_prefix + f"dataset_hf_{name}_q{quality}_ogg")
    tk.register_output(f"{_alias_prefix}dataset_hf_{name}_q{quality}_ogg", job.out_dir)
    return job.out_dir


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

    # TODO peak_normalization ?
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
        score_recog_output_func=generic_sclite_score_recog_out,
        recog_post_proc_funcs=recog_post_proc_funcs,
    )
    return task


# TODO v2:
#   - multiprocdataset?
#   - better devtrain: take random subset of train
#   - better dev: take subset of dev


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


def _make_hf_dataset(
    *,
    hf_data_dir: Path,
    split: str,
    vocab: VocabConfig,
    seq_ordering: str = "sorted_reverse",
    partition_epoch: Optional[int] = None,
    use_distrib_files: bool = False,
    take_first_shard_subset: bool = False,
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
    hf_ds_opts = hf_data_dir.join_right(split)
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
    content = [fn for fn in content if fn.endswith(".arrow")]
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
