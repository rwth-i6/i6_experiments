from dataclasses import dataclass
from typing import Any, Dict, Optional

from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig
from sisyphus import Path

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.label.sentencepiece.train import SentencePieceType

from ..f8kHz.task import get_asr_task


ALIAS_PREFIX = "datasets/spanish/lm"

LM_TRAIN_DATA = Path("/nas/models/asr/mgunz/2024-07-08--zeyer-setup-apptek/lm_train.bgd.unk.subsampled.100m.txt.gz")


@dataclass
class LmTask:
    pass


class SpanishLmDataset(DatasetConfig):
    def __init__(
        self,
        *,
        vocab: VocabConfig,
        train_vocab: Optional[VocabConfig] = None,
        train_partition_epoch: int,
        train_sort_laplace_num_seqs: int = 1000
    ):
        self.vocab = vocab
        self.train_vocab = train_vocab or vocab
        self.train_partition_epoch = train_partition_epoch
        self.train_sort_laplace_num_seqs = train_sort_laplace_num_seqs


def get_lm_task(
    *,
    spm_dim: int,
    spm_type: SentencePieceType,
    train_partition_epoch: int,
    train_vocab_opts: Optional[Dict[str, Any]] = None
) -> SpanishLmDataset:
    assert spm_dim > 0
    assert train_partition_epoch > 0

    asr_task = get_asr_task()
    acoustic_train_text = CorpusToTxtJob(asr_task.corpora.train, gzip=True).out_txt
