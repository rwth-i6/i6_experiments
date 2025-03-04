from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import BPESettings
from sisyphus import tk

from i6_experiments.common.datasets.switchboard.corpus_train import (
    get_spoken_form_train_bliss_corpus_ldc,
)

from ...tools import subword_nmt_repo
from ..base import BPEVocabToTextFileConversionJob, RemoveWordsFromTranscriptionsJob


def get_bpe_settings(bpe_size: int) -> BPESettings:
    train_corpus_file = get_spoken_form_train_bliss_corpus_ldc()
    train_corpus_file = RemoveWordsFromTranscriptionsJob(
        train_corpus_file, ["[NOISE]", "[LAUGHTER]", "[VOCALIZED-NOISE]"]
    ).out_corpus_file

    to_text_job = CorpusToTxtJob(train_corpus_file)

    train_bpe_job = ReturnnTrainBpeJob(
        text_file=to_text_job.out_txt,
        bpe_size=bpe_size,
        unk_label="<unk>",
        subword_nmt_repo=subword_nmt_repo,
    )

    return BPESettings(
        bpe_codes=train_bpe_job.out_bpe_codes,
        bpe_vocab=train_bpe_job.out_bpe_vocab,
        bpe_count_vocab=train_bpe_job.out_bpe_dummy_count_vocab,
        bpe_vocab_size=train_bpe_job.out_vocab_size,
        unk_label="<unk>",
    )


def get_bpe_vocab_file(bpe_size: int, add_blank: bool = False) -> tk.Path:
    bpe_settings = get_bpe_settings(bpe_size)
    return BPEVocabToTextFileConversionJob(
        bpe_vocab_file=bpe_settings.bpe_vocab, extra_tokens=["<blank>"] if add_blank else None
    ).out_vocab_file


def get_default_bpe_target_config(bpe_size: int) -> dict:
    bpe_settings = get_bpe_settings(bpe_size)
    return {
        "class": "BytePairEncoding",
        "unknown_label": "<unk>",
        "bpe_file": bpe_settings.bpe_codes,
        "vocab_file": bpe_settings.bpe_vocab,
    }


def bpe_to_vocab_size(bpe_size: int) -> int:
    return {
        128: 185,
    }.get(bpe_size, bpe_size)
