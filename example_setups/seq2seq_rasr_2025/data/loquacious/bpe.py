from i6_experiments.common.datasets.loquacious.vocab import get_subword_nmt_bpe
from sisyphus import tk

from ..base import BPEVocabToTextFileConversionJob


def get_bpe_vocab_file(bpe_size: int, corpus_key: str, add_blank: bool = False) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe(corpus_key=corpus_key, bpe_size=bpe_size)
    return BPEVocabToTextFileConversionJob(
        bpe_vocab_file=bpe_settings.bpe_vocab, extra_tokens=["<blank>"] if add_blank else None
    ).out_vocab_file


def get_default_bpe_target_config(bpe_size: int, corpus_key: str) -> dict:
    bpe_settings = get_subword_nmt_bpe(corpus_key=corpus_key, bpe_size=bpe_size)
    return {
        "class": "BytePairEncoding",
        "unknown_label": "<unk>",
        "bpe_file": bpe_settings.bpe_codes,
        "vocab_file": bpe_settings.bpe_vocab,
    }


def bpe_to_vocab_size(bpe_size: int) -> int:
    return {
        128: 183,
        5000: 5053,
        10000: 10041,
    }.get(bpe_size, bpe_size)


def vocab_to_bpe_size(vocab_size: int) -> int:
    return {
        183: 128,
        5053: 5000,
        10041: 10000,
    }.get(vocab_size, vocab_size)
