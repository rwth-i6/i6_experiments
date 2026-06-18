"""
Offline LibriSpeech datasets for RETURNN, read directly from the local HF parquet cache.

Constraints (this cluster): compute nodes are disk-free and offline. We therefore read the
pre-cached openslr/librispeech_asr parquet shards directly via RETURNN's ``HuggingFaceDataset``
(``datasets.load_dataset("parquet", data_files=...)``), decoding flac to raw 16 kHz float32 on
the fly, with ``use_file_cache=False`` (no node-local staging) and no i6 cache manager (``cf``).

Audio is delivered raw (the model does log-mel). Sequence ordering is "random"/"laplace.."; note
the raw parquet has no duration column, so length-based ordering needs the (optional) ogg prep job
that adds ``duration`` -- for now SSL uses "random" (fine for an initial run).

Split keys (raw HF): train.clean.100 / train.clean.360 / train.other.500,
validation.clean / validation.other, test.clean / test.other.
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from i6_core.returnn.config import CodeWrapper

# raw HF split names
TRAIN_960H = ("train.clean.100", "train.clean.360", "train.other.500")
TRAIN_CLEAN_100 = ("train.clean.100",)
DEV_CLEAN = ("validation.clean",)
DEV_OTHER = ("validation.other",)
DEV_ALL = ("validation.clean", "validation.other")
TEST_CLEAN = ("test.clean",)
TEST_OTHER = ("test.other",)

_HF_HOME = os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home")


@lru_cache()
def _snapshot_all_dir() -> str:
    repo = os.path.join(_HF_HOME, "hub", "datasets--openslr--librispeech_asr")
    refs_main = os.path.join(repo, "refs", "main")
    commit = None
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            commit = f.read().strip()
    if not commit:
        snaps = sorted(os.listdir(os.path.join(repo, "snapshots")))
        commit = snaps[-1] if snaps else None
    assert commit, f"no cached openslr/librispeech_asr snapshot under {repo}"
    all_dir = os.path.join(repo, "snapshots", commit, "all")
    assert os.path.isdir(all_dir), f"missing {all_dir}"
    return all_dir


def parquet_files(src_splits: Sequence[str]) -> List[str]:
    base = _snapshot_all_dir()
    files: List[str] = []
    for split in src_splits:
        matched = sorted(glob.glob(os.path.join(base, split, "*.parquet")))
        assert matched, f"no parquet for split {split!r} under {base}"
        files.extend(matched)
    return files


def audio_hf_dataset(
    src_splits: Sequence[str],
    *,
    seq_ordering: str = "random",
    partition_epoch: Optional[int] = None,
    ddp_seq_shard: bool = False,
) -> Dict[str, Any]:
    """RETURNN HuggingFaceDataset opts delivering raw 16 kHz audio under key 'audio' (no targets).

    ``ddp_seq_shard=True`` (TRAIN only) enables per-rank sequence-level sharding: every rank loads the
    full parquet (one shared HF cache -- the proven path; file-level DistributeFilesDataset sharding
    blew up the parquet->arrow cache with regeneration races) but iterates a DISJOINT contiguous 1/W of
    the shuffled sequence order, so one RETURNN epoch == one pass over the split. Correctness requires
    all ranks to shuffle IDENTICALLY, hence ``random_seed_offset=0`` (overrides RETURNN's per-rank
    ``rank*16127`` default, which would otherwise overlap+omit sequences). ``_shard_index``/``_num_shards``
    read torchrun's global ``RANK``/``WORLD_SIZE`` at config-exec time (CodeWrapper emits the lookups as
    code). Keep ``partition_epoch`` unset/1 so ``current_partition == shard_index`` tiles cleanly.
    Per-epoch reshuffle is preserved (epoch is part of the seed). NEVER set this on dev/eval (each rank
    would score only 1/W of the data). See returnn datasets/basic.py get_seq_order_for_epoch.
    """
    d: Dict[str, Any] = {
        "class": "HuggingFaceDataset",
        "dataset_opts": {
            "path": "parquet",
            "data_files": {"data": parquet_files(src_splits)},
            "split": "data",
        },
        "use_file_cache": False,
        "seq_tag_column": "id",
        "cast_columns": {"audio": {"_type": "Audio", "sampling_rate": 16000}},
        "data_format": {"audio": {"shape": (None,), "dtype": "float32"}},
        "seq_ordering": seq_ordering,
    }
    if partition_epoch is not None:
        d["partition_epoch"] = partition_epoch
    if ddp_seq_shard:
        d["random_seed_offset"] = 0  # equal across ranks -> identical shuffle (disjointness requirement)
        # __import__("os") (not bare ``os``): the config's ``import os`` lives in the epilog, AFTER the
        # train-dict literal, so ``os`` is not yet in scope when this value is evaluated at config-exec.
        d["_num_shards"] = CodeWrapper('int(__import__("os").environ.get("WORLD_SIZE", "1"))')
        d["_shard_index"] = CodeWrapper('int(__import__("os").environ.get("RANK", "0"))')
    return d


def extern_data_audio() -> Dict[str, Any]:
    """extern_data entry for the raw-audio stream (1-D time, raw float32 samples)."""
    return {"audio": {"shape": (None,), "dtype": "float32"}}


def labeled_hf_dataset(
    src_splits: Sequence[str],
    *,
    spm_model,
    seq_ordering: str = "random",
    partition_epoch: Optional[int] = None,
    ddp_seq_shard: bool = False,
) -> Dict[str, Any]:
    """RETURNN HuggingFaceDataset opts with raw audio + on-the-fly SPM-tokenized 'text' targets.

    The transcript string is encoded by RETURNN's SentencePieces vocab at data-loading time (labels
    0..V-1, no EOS for CTC; blank V is added only in the model head). ``ddp_seq_shard`` -> see
    ``audio_hf_dataset`` (TRAIN only)."""
    d = audio_hf_dataset(
        src_splits, seq_ordering=seq_ordering, partition_epoch=partition_epoch, ddp_seq_shard=ddp_seq_shard
    )
    d["data_format"]["text"] = {
        "dtype": "int32",
        "shape": (None,),
        "sparse": True,
        "vocab": {"class": "SentencePieces", "model_file": spm_model, "add_eos": False},
    }
    return d


def extern_data_audio_text(vocab_size: int) -> Dict[str, Any]:
    """extern_data for raw audio + sparse SPM text targets (dim = SPM vocab size, fixed e.g. 5120)."""
    return {
        "audio": {"shape": (None,), "dtype": "float32"},
        "text": {"shape": (None,), "dtype": "int32", "sparse": True, "dim": vocab_size},
    }
