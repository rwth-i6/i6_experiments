"""
BPE vocab for CTC finetuning on LibriSpeech train-clean-100 (offline).

The transcript text was extracted once from the local HF parquet (text column only, no audio) to
``work_data/ls100_text.txt.gz``; subword-nmt is a local clone (``work_data/subword-nmt``) so the
``ReturnnTrainBpeJob`` runs on an offline compute node. Returns a ``BPESettings`` whose
``bpe_vocab_size`` is a tk.Variable (resolved at config-write time, after the BPE job runs).
"""

from __future__ import annotations

from functools import lru_cache

from sisyphus import tk

from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import BPESettings

_WS = "/e/project1/spell/wu24/2026-06-17_ssl"
LS100_TRAIN_TEXT = tk.Path(f"{_WS}/work_data/ls100_text.txt.gz", hash_overwrite="SSL_LS100_TRAIN_TEXT")
SUBWORD_NMT_REPO = tk.Path(f"{_WS}/work_data/subword-nmt", hash_overwrite="SSL_SUBWORD_NMT_REPO")


@lru_cache()
def get_ls100_bpe(bpe_size: int = 128, unk_label: str = "<unk>") -> BPESettings:
    """Train a subword-nmt BPE on train-clean-100 transcripts. `bpe_size` = #merge operations."""
    job = ReturnnTrainBpeJob(
        text_file=LS100_TRAIN_TEXT,
        bpe_size=bpe_size,
        unk_label=unk_label,
        subword_nmt_repo=SUBWORD_NMT_REPO,
    )
    job.add_alias(f"ssl/data/bpe_ls100_{bpe_size}")
    return BPESettings(
        bpe_codes=job.out_bpe_codes,
        bpe_vocab=job.out_bpe_vocab,
        bpe_count_vocab=job.out_bpe_dummy_count_vocab,
        bpe_vocab_size=job.out_vocab_size,
        unk_label=unk_label,
    )
