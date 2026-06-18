"""
SentencePiece (unigram) vocab for CTC, trained offline on the LibriSpeech 960h transcripts.

Mirrors the label unit used by the speech_llm AED/CTC setup (SPM unigram, ~5k). The 960h transcript
text was pre-extracted (text-only, offline) to ``work_data/ls960_text.txt.gz``. `TrainSentencePieceJob`
shells into the `sentencepiece` lib (present in the conda worker env) -> fully offline. vocab_size is
a FIXED constant (5120), so the CTC head dim (vocab+1) is known at graph time (no DelayedBase).
"""

from __future__ import annotations

from functools import lru_cache

from sisyphus import tk

from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_experiments.common.setups.returnn.datastreams.vocabulary import SentencePieceDatastream

_WS = "/e/project1/spell/wu24/2026-06-17_ssl"
LS960_TRAIN_TEXT = tk.Path(f"{_WS}/work_data/ls960_text.txt.gz", hash_overwrite="SSL_LS960_TRAIN_TEXT")

SPM_VOCAB_SIZE = 5120


@lru_cache()
def get_ls960_spm(vocab_size: int = SPM_VOCAB_SIZE):
    """Train a UNIGRAM SentencePiece model on LS960 transcripts. Returns (datastream, spm_model_path)."""
    job = TrainSentencePieceJob(
        training_text=LS960_TRAIN_TEXT,
        vocab_size=vocab_size,
        model_type=SentencePieceType.UNIGRAM,
        character_coverage=1.0,
    )
    job.add_alias(f"ssl/data/spm_ls960_{vocab_size}")
    datastream = SentencePieceDatastream(
        available_for_inference=True, spm_model=job.out_model, vocab_size=vocab_size
    )
    return datastream, job.out_model
