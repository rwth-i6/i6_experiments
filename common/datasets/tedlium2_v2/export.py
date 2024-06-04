import os

from sisyphus import tk

from .corpus import get_bliss_corpus_dict, get_stm_dict
from .lexicon import get_bliss_lexicon, get_g2p_augmented_bliss_lexicon
from .textual_data import get_text_data_dict

TEDLIUM_PREFIX = "Ted-Lium-2"


def _export_datasets(output_prefix: str = "datasets"):
    """
    exports all datasets for TedLiumV2 with all available audio formats

    :param output_prefix:
    :return:
    """
    for audio_format in ["flac", "ogg", "wav", "nist", "sph"]:
        bliss_corpus_dict = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)
        for name, bliss_corpus in bliss_corpus_dict.items():
            tk.register_output(
                os.path.join(
                    output_prefix,
                    TEDLIUM_PREFIX,
                    "corpus",
                    f"{name}-{audio_format}.xml.gz",
                ),
                bliss_corpus,
            )


def _export_stms(output_prefix: str = "datasets"):
    """
    exports all STMs for TedLiumV2

    :param output_prefix:
    :return:
    """
    stm_dict = get_stm_dict(output_prefix=output_prefix)
    for name, stm_file in stm_dict.items():
        tk.register_output(
            os.path.join(
                output_prefix,
                TEDLIUM_PREFIX,
                "stm",
                f"{name}.txt",
            ),
            stm_file,
        )


def _export_text_data(output_prefix: str = "datasets"):
    """
    exports all the textual data for TedLiumV2 dataset

    :param output_prefix:
    :return:
    """
    txt_data_dict = get_text_data_dict(output_prefix=output_prefix)
    for k, v in txt_data_dict.items():
        tk.register_output(os.path.join(output_prefix, TEDLIUM_PREFIX, "text_data", f"{k}.gz"), v)


def _export_lexicon(output_prefix: str = "datasets"):
    """
    exports the lexicon for TedLiumV2

    :param output_prefix:
    :return:
    """
    lexicon_output_prefix = os.path.join(output_prefix, TEDLIUM_PREFIX, "lexicon")

    bliss_lexicon = get_bliss_lexicon(output_prefix=output_prefix)
    tk.register_output(os.path.join(lexicon_output_prefix, "tedlium2.lexicon.xml.gz"), bliss_lexicon)

    g2p_bliss_lexicon = get_g2p_augmented_bliss_lexicon(
        add_unknown_phoneme_and_mapping=False, output_prefix=output_prefix
    )
    tk.register_output(
        os.path.join(lexicon_output_prefix, "tedlium2.lexicon_with_g2p.xml.gz"),
        g2p_bliss_lexicon,
    )


def export_all(output_prefix: str = "datasets"):
    """
    exports everything for TedLiumV2

    :param output_prefix:
    :return:
    """
    _export_datasets(output_prefix=output_prefix)
    _export_stms(output_prefix=output_prefix)
    _export_text_data(output_prefix=output_prefix)
    _export_lexicon(output_prefix=output_prefix)
