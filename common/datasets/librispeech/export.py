import os

from sisyphus import tk

from .corpus import get_bliss_corpus_dict, get_ogg_zip_dict
from .language_model import get_arpa_lm_dict
from .lexicon import get_bliss_lexicon, get_g2p_augmented_bliss_lexicon_dict
from .vocab import get_subword_nmt_bpe


def _export_datasets(output_prefix):
    """
    :param str output_prefix:
    """

    # export all bliss corpora
    for audio_format in ["flac", "ogg", "wav"]:
        bliss_corpus_dict = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)
        for name, bliss_corpus in bliss_corpus_dict.items():
            tk.register_output(
                os.path.join(output_prefix, "LibriSpeech", "%s-%s.xml.gz" % (name, audio_format)),
                bliss_corpus,
            )

    # export all ogg zip corpora
    ogg_corpus_dict = get_ogg_zip_dict(output_prefix=output_prefix)
    for name, ogg_corpus in ogg_corpus_dict.items():
        tk.register_output(os.path.join(output_prefix, "LibriSpeech", "%s.ogg.zip" % name), ogg_corpus)


def _export_lm_data(output_prefix):
    """
    :param str output_prefix:
    """
    lm_dict = get_arpa_lm_dict(output_prefix=output_prefix)
    tk.register_output(
        os.path.join(output_prefix, "LibriSpeech", "lm", "3-gram.arpa.gz"),
        lm_dict["3gram"],
    )
    tk.register_output(
        os.path.join(output_prefix, "LibriSpeech", "lm", "4-gram.arpa.gz"),
        lm_dict["4gram"],
    )


def _export_lexicon_and_vocab(output_prefix):
    """
    :param str output_prefix:
    """

    lexicon_output_prefix = os.path.join(output_prefix, "LibriSpeech", "lexicon")

    # folded / without stress marker
    bliss_lexicon = get_bliss_lexicon(output_prefix=output_prefix, use_stress_marker=False)
    tk.register_output(
        os.path.join(lexicon_output_prefix, "librispeech.lexicon.folded.xml.gz"),
        bliss_lexicon,
    )

    g2p_lexicon_dict = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=True, output_prefix=output_prefix)
    for k, lexicon in g2p_lexicon_dict.items():
        tk.register_output(
            os.path.join(lexicon_output_prefix, "%s.lexicon_with_g2p.folded.xml.gz" % k),
            lexicon,
        )

    # with stress marker
    bliss_lexicon = get_bliss_lexicon(output_prefix=output_prefix, use_stress_marker=True)
    tk.register_output(
        os.path.join(lexicon_output_prefix, "librispeech.lexicon.xml.gz"),
        bliss_lexicon,
    )

    g2p_lexicon_dict = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False, output_prefix=output_prefix)
    for k, lexicon in g2p_lexicon_dict.items():
        tk.register_output(
            os.path.join(lexicon_output_prefix, "%s.lexicon_with_g2p.xml.gz" % k),
            lexicon,
        )


def _export_legacy_bpe(output_prefix):
    """
    Export the files for ls-100 2k and ls-960 10k bpe labels

    :param str output_prefix
    """
    lexicon_output_prefix = os.path.join(output_prefix, "LibriSpeech", "bpe")
    ls960_bpe_settings = get_subword_nmt_bpe(corpus_key="train-other-960", bpe_size=10000, output_prefix=output_prefix)
    ls100_bpe_settings = get_subword_nmt_bpe(corpus_key="train-clean-100", bpe_size=2000, output_prefix=output_prefix)
    tk.register_output(
        os.path.join(lexicon_output_prefix, "train-other-960", "bpe_10k.codes"),
        ls960_bpe_settings.bpe_codes,
    )
    tk.register_output(
        os.path.join(lexicon_output_prefix, "train-other-960", "bpe_10k.vocab"),
        ls960_bpe_settings.bpe_vocab,
    )
    tk.register_output(
        os.path.join(lexicon_output_prefix, "train-other-100", "bpe_2k.codes"),
        ls100_bpe_settings.bpe_codes,
    )
    tk.register_output(
        os.path.join(lexicon_output_prefix, "train-other-100", "bpe_2k.vocab"),
        ls100_bpe_settings.bpe_vocab,
    )


def export_all(output_prefix):
    """
    Registers all LibriSpeech related data as output.

    For internal i6 purposes only, as this will create all jobs no matter if they are actually needed:

    physical jobs are located in: `/work/common/asr/librispeech/data/sisyphus_work_dir/`

    :param str output_prefix:
    """
    _export_datasets(output_prefix)
    _export_lm_data(output_prefix)
    _export_lexicon_and_vocab(output_prefix)
    _export_legacy_bpe(output_prefix)
