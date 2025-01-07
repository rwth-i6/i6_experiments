from typing import Optional, Callable
from collections import OrderedDict
import textwrap
import copy

from i6_core.returnn.config import ReturnnConfig, CodeWrapper
from i6_core.returnn.training import ReturnnTrainingJob
from i6_experiments.common.setups import serialization

tedlium2_transcription_bpe10k_dataloader_config = OrderedDict(
    train_epoch_split = 1,
    data_files = {
        "train": ["/work/asr4/michel/lm/tedlium/data/tedlium.train.lbs_bpe.txt.gz",
              "/work/asr4/michel/lm/tedlium/data/commoncrawl-9pc.en.bpe.gz",
              "/work/asr4/michel/lm/tedlium/data/europarl-v7-6pc.en.bpe.gz",
              "/work/asr4/michel/lm/tedlium/data/giga-fren-4pc.en.bpe.gz",
              "/work/asr4/michel/lm/tedlium/data/news-18pc.en.bpe.gz",
              "/work/asr4/michel/lm/tedlium/data/news-commentary-v8-9pc.en.bpe.gz",
              "/work/asr4/michel/lm/tedlium/data/yandex-1m-31pc.en.bpe.gz"],
        "cv": ["/work/asr4/michel/lm/tedlium/data/tedlium.test.lbs_bpe.txt.gz"],
        "test": ["/work/asr4/michel/lm/tedlium/data/tedlium.test.lbs_bpe.txt.gz"],
    },
    vocab_file = "/work/asr3/irie/data/librispeech/lm_bpe/trans.bpe.vocab.lm.txt", # upper case
    epoch_split = {"train": CodeWrapper("train_epoch_split"), "cv": 1, "test": 1},
    target = "data",
    update_on_device = True,
)

from i6_experiments.users.phan.train_transcription_lm.train_config import base_sgd_config
tedlium2_transcription_lm_bpe10k_sgd_config = copy.deepcopy(tedlium2_transcription_bpe10k_dataloader_config)
tedlium2_transcription_lm_bpe10k_sgd_config.update(base_sgd_config)
