from dataclasses import dataclass

from apptek_asr.users.mgunz.datasets.spanish.lm.task import LM_TRAIN_DATA
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig
from sisyphus import Path, tk

from functools import cache
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from ..f8kHz.task import get_asr_task


ALIAS_PREFIX = "datasets/spanish/lm"
#LM_TRAIN_DATA = Path("/nas/models/asr/mgunz/2024-07-08--zeyer-setup-apptek/lm_train.bgd.unk.subsampled.100m.txt.gz")
@dataclass
class LmTask:
    pass

#
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
