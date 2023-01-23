from sisyphus import *


class SWBCorpora:
  corpus_paths = {
    "train": Path("/work/asr3/irie/data/switchboard/corpora/train.corpus.gz", cached=True),
    "dev": Path("/work/asr3/irie/data/switchboard/corpora/dev.corpus.gz", cached=True),
    "hub5e_01": Path("/work/asr3/irie/data/switchboard/corpora/hub5e_01.corpus.gz", cached=True),
    "rt03s": Path("/work/asr3/irie/data/switchboard/corpora/rt03s.corpus.gz", cached=True)}
  feature_cache_paths = {
    "train": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.train.bundle", cached=True),
    "dev": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.dev.bundle", cached=True),
    "hub5e_01": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.hub5e_01.bundle", cached=True),
    "rt03s": Path("/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.rt03s.bundle", cached=True)}
  test_stm_paths = {
    "dev": Path("/u/tuske/bin/switchboard/hub5e_00.2.stm"),
    "hub5e_01": Path("/u/tuske/bin/switchboard/hub5e_01.2.stm"),
    "rt03s": Path("/u/tuske/bin/switchboard/rt03s_ctsonly.stm")}
  corpus_mapping = {
    "train": "train", "cv": "train", "devtrain": "train", "dev": "dev", "hub5e_01": "hub5e_01", "rt03s": "rt03s"
  }
  corpus_keys = ("train", "cv", "devtrain", "dev", "hub5e_01", "rt03s")
  train_corpus_keys = ("train", "cv", "devtrain")
  test_corpus_keys = ("dev", "hub5e_01", "rt03s")
