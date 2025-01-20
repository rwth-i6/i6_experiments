from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from sisyphus import tk

from ..base import BPEVocabToTextFileConversionJob


def get_bpe_vocab_file(bpe_size: int, add_blank: bool = False) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key="train-other-960", bpe_size=bpe_size)
    return BPEVocabToTextFileConversionJob(
        bpe_vocab_file=bpe_settings.bpe_vocab, extra_tokens=["<blank>"] if add_blank else None
    ).out_vocab_file


def get_default_bpe_target_config(bpe_size: int) -> dict:
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key="train-other-960", bpe_size=bpe_size)
    return {
        "class": "BytePairEncoding",
        "unknown_label": None,
        "bpe_file": bpe_settings.bpe_codes,
        "vocab_file": bpe_settings.bpe_vocab,
    }


def bpe_to_vocab_size(bpe_size: int) -> int:
    return {
        128: 184,
        1000: 1056,
        5000: 5048,
        10000: 10025,
    }.get(bpe_size, bpe_size)
