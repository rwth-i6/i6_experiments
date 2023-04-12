from typing import List

dataset_alias = "swb"
allowed_corpus_keys = ["train", "cv", "devtrain", "dev", "hub5e_01", "rt03s"]


def check_corpus_keys(corpus_keys: List[str]):
  for corpus_key in corpus_keys:
    assert corpus_key in allowed_corpus_keys
