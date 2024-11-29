from sisyphus import tk

from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.users.rossenbach.corpus.generate import CreateBlissFromTextLinesJob


from ..tts.tts_phon import get_lexicon


def create_data_lexicon(prefix: str, lm_text_bliss: tk.Path):
    """

    :param prefix:
    :param lm_text_bliss:
    :return:
    """
    ls960_tts_lexicon = get_lexicon(with_blank=False, corpus_key="train-other-960")

    g2p_augmenter = G2PBasedOovAugmenter(
        original_bliss_lexicon=ls960_tts_lexicon,
        train_lexicon=ls960_tts_lexicon,
        apply_args={"concurrent": 5}
    )
    extended_bliss_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
        bliss_corpus=lm_text_bliss,
        corpus_name="lm_tts_data",
        alias_path=prefix,
        casing="upper",
    )
    return extended_bliss_lexicon


def create_data_lexicon_rasr_style(prefix: str, lm_text_bliss: tk.Path):
    """

    :param prefix:
    :param lm_text_bliss:
    :return:
    """

    ls960_tts_lexicon = get_lexicon(with_blank=False, corpus_key="train-other-960")

    ls960_rasr_lexicon = get_bliss_lexicon(
        use_stress_marker=False, add_unknown_phoneme_and_mapping=False, add_silence=True
    )

    g2p_augmenter = G2PBasedOovAugmenter(
        original_bliss_lexicon=ls960_rasr_lexicon,
        train_lexicon=ls960_tts_lexicon,
        apply_args={"concurrent": 5}
    )
    extended_bliss_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
        bliss_corpus=lm_text_bliss,
        corpus_name="lm_tts_data",
        alias_path=prefix,
        casing="upper",
    )
    return extended_bliss_lexicon


def bliss_from_text(prefix, name, lm_text):
    """

    :param prefix:
    :param name:
    :param lm_text:
    :return:
    """
    bliss_job = CreateBlissFromTextLinesJob(
        text_file=lm_text,
        corpus_name=name,
        sequence_prefix="line"
    )
    bliss_job.add_alias(prefix + "/create_bliss_for_%s" % name)
    return bliss_job.out_corpus
