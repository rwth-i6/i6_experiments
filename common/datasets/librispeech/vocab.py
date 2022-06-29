import os
from functools import lru_cache

from i6_core.tools.download import DownloadJob

from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import (
    get_returnn_subword_nmt as _get_returnn_subword_nmt,
    get_bpe_settings as _get_bpe_settings,
    BPESettings,
)

from .corpus import get_bliss_corpus_dict


@lru_cache()
def get_lm_vocab(output_prefix="datasets"):
    """
    :param str output_prefix:
    :return: Path to LibriSpeech vocab file (one word per line)
    :rtype: Path
    """
    download_lm_vocab_job = DownloadJob(
        url="https://www.openslr.org/resources/11/librispeech-vocab.txt",
        target_filename="librispeech-vocab.txt",
        checksum="3014e72dffff09cb1a9657f31cfe2e04c1301610a6127a807d1d708b986b5474",
    )
    download_lm_vocab_job.add_alias(
        os.path.join(output_prefix, "LibriSpeech", "download_lm_vocab_job")
    )
    return download_lm_vocab_job.out_file


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
        output_prefix = os.path.join(
            output_prefix, "librispeech_%s_bpe_%i" % (corpus_key, bpe_size)
        )

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
