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


def get_subword_nmt_bpe(
    corpus_key: str, bpe_size: int, unk_label: str = "<unk>", output_prefix: str = ""
) -> BPESettings:
    """
    Get the BPE tokens via the subword-nmt fork for a librispeech setup.
    When using the default settings (e.g. bpe_size 2k for train-clean-100 or 10k for train-other-960)
    this will give 100% compatible BPE settings to Albert Zeyers, Kazuki Iries and Mohammad Zeineldeens setups.

    V2: Uses subword-nmt version corrected for Apptainer related bug, adds hash overwrite for repo

    :param corpus_key: LibriSpeech (sub-)corpus key
    :param bpe_size: the number of BPE merge operations. This is NOT the resulting vocab size!
    :param output_prefix: if set registers alias and output path
    """
    if output_prefix:
        output_prefix = os.path.join(output_prefix, "loquacious_%s_bpe_%i" % (corpus_key, bpe_size))

    subword_nmt_repo = _get_returnn_subword_nmt(
        commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf", output_prefix=output_prefix
    )
    # overwrite hash for future bugfixes, it is unlikely the logic will ever be changed
    subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"
    train_corpus = get_bliss_corpus_dict()[corpus_key]
    bpe_settings = _get_bpe_settings(
        train_corpus,
        bpe_size=bpe_size,
        unk_label=unk_label,
        output_prefix=output_prefix,
        subword_nmt_repo_path=subword_nmt_repo,
    )
    return bpe_settings