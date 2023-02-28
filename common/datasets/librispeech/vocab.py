import os
from functools import lru_cache

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.lm.vocabulary import LmIndexVocabularyFromLexiconJob, LmIndexVocabulary

from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import (
    get_returnn_subword_nmt as _get_returnn_subword_nmt,
    get_bpe_settings as _get_bpe_settings,
    BPESettings,
)

from .corpus import get_bliss_corpus_dict
from .lexicon import get_bliss_lexicon


@lru_cache()
def get_lm_vocab(output_prefix) -> LmIndexVocabulary:
    """
    :param str output_prefix:
    :return: Path to LibriSpeech vocab file (one word per line)
    """
    ls960_text_job = CorpusToTxtJob(bliss_corpus=get_bliss_corpus_dict()["train-other-960"], gzip=True)
    ls960_text_job.add_alias(os.path.join(output_prefix, "ls960_to_text_job"))
    index_vocab_job = LmIndexVocabularyFromLexiconJob(
        bliss_lexicon=get_bliss_lexicon(add_unknown_phoneme_and_mapping=False),
        count_ordering_text=ls960_text_job.out_txt,
    )
    index_vocab_job.add_alias(os.path.join(output_prefix, "lm_index_vocab_from_lexicon_job"))
    return index_vocab_job.out_vocabulary_object


@lru_cache()
def get_subword_nmt_bpe(corpus_key, bpe_size, unk_label="<unk>", output_prefix=""):
    """
    Get the BPE tokens via the subword-nmt fork for a librispeech setup.
    When using the default settings (e.g. bpe_size 2k for train-clean-100 or 10k for train-other-960)
    this will give 100% compatible BPE settings to Albert Zeyers, Kazuki Iries and Mohammad Zeineldeens setups.

    :param str corpus_key: LibriSpeech (sub-)corpus key
    :param int bpe_size: the number of BPE merge operations. This is NOT the resulting vocab size!
    :param str output_prefix: if set registers alias and output path
    :rtype: BPESettings
    """
    if output_prefix:
        output_prefix = os.path.join(output_prefix, "librispeech_%s_bpe_%i" % (corpus_key, bpe_size))

    subword_nmt_repo = _get_returnn_subword_nmt(output_prefix=output_prefix)
    train_corpus = get_bliss_corpus_dict("flac", "corpora")[corpus_key]
    bpe_settings = _get_bpe_settings(
        train_corpus,
        bpe_size=bpe_size,
        unk_label=unk_label,
        output_prefix=output_prefix,
        subword_nmt_repo_path=subword_nmt_repo,
    )
    return bpe_settings
