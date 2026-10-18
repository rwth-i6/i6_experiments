import os
from functools import lru_cache
from sisyphus import tk

from i6_core.lexicon import LexiconFromTextFileJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_core.lib import lexicon
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter

from ..tedlium2.constants import SILENCE_PHONEME, UNKNOWN_PHONEME
from .corpus import get_bliss_corpus_dict
from .download import download_data_dict


@lru_cache()
def _get_special_lemma_lexicon(
    add_unknown_phoneme_and_mapping: bool = False,
    add_silence: bool = True,
) -> lexicon.Lexicon:
    """
    creates the special lemma used in RASR

    :param add_unknown_phoneme_and_mapping: adds [unknown] as label with [UNK] as phoneme and <unk> as LM token
    :param add_silence: adds [silence] label with [SILENCE] phoneme,
        use False for CTC/RNN-T setups without silence modelling.
    :return:
    """
    lex = lexicon.Lexicon()
    if add_silence:
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[silence]", ""],
                phon=[SILENCE_PHONEME],
                synt=[],
                special="silence",
                eval=[[]],
            )
        )
    if add_unknown_phoneme_and_mapping:
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[unknown]"],
                phon=[UNKNOWN_PHONEME],
                synt=["<unk>"],
                special="unknown",
                eval=[[]],
            )
        )
    else:
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[unknown]"],
                synt=["<unk>"],
                special="unknown",
                eval=[[]],
            )
        )

    lex.add_lemma(
        lexicon.Lemma(
            orth=["[sentence-begin]"],
            synt=["<s>"],
            special="sentence-begin",
            eval=[[]],
        )
    )
    lex.add_lemma(
        lexicon.Lemma(
            orth=["[sentence-end]"],
            synt=["</s>"],
            special="sentence-end",
            eval=[[]],
        )
    )
    if add_silence:
        lex.add_phoneme(SILENCE_PHONEME, variation="none")
    if add_unknown_phoneme_and_mapping:
        lex.add_phoneme(UNKNOWN_PHONEME, variation="none")

    return lex


@lru_cache()
def _get_raw_bliss_lexicon(
    output_prefix: str,
) -> tk.Path:
    """
    downloads the vocabulary file from the TedLiumV2 dataset and creates a bliss lexicon

    :param output_prefix:
    :return:
    """
    vocab = download_data_dict(output_prefix=output_prefix).vocab

    convert_lexicon_job = LexiconFromTextFileJob(
        text_file=vocab,
        compressed=True,
    )
    convert_lexicon_job.add_alias(os.path.join(output_prefix, "convert_text_to_bliss_lexicon_job"))

    return convert_lexicon_job.out_bliss_lexicon


@lru_cache()
def get_bliss_lexicon(
    add_unknown_phoneme_and_mapping: bool = True,
    add_silence: bool = True,
    output_prefix: str = "datasets",
) -> tk.Path:
    """
    merges the lexicon with special RASR tokens with the lexicon created from the downloaded TedLiumV2 vocabulary

    :param add_unknown_phoneme_and_mapping: add an unknown phoneme and mapping unknown phoneme:lemma
    :param add_silence: include silence lemma and phoneme
    :param output_prefix:
    :return:
    """
    static_lexicon = _get_special_lemma_lexicon(add_unknown_phoneme_and_mapping, add_silence)
    static_lexicon_job = WriteLexiconJob(static_lexicon, sort_phonemes=True, sort_lemmata=False)
    static_lexicon_job.add_alias(os.path.join(output_prefix, "static_lexicon_job"))

    raw_tedlium2_lexicon = _get_raw_bliss_lexicon(output_prefix=output_prefix)

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            raw_tedlium2_lexicon,
        ],
        sort_phonemes=True,
        sort_lemmata=True,
        compressed=True,
    )
    merge_lexicon_job.add_alias(os.path.join(output_prefix, "merge_lexicon_job"))

    return merge_lexicon_job.out_bliss_lexicon


@lru_cache()
def get_g2p_augmented_bliss_lexicon(
    add_unknown_phoneme_and_mapping: bool = False,
    add_silence: bool = True,
    audio_format: str = "wav",
    output_prefix: str = "datasets",
) -> tk.Path:
    """
    augment the kernel lexicon with unknown words from the training corpus

    :param add_unknown_phoneme_and_mapping: add an unknown phoneme and mapping unknown phoneme:lemma
    :param add_silence: include silence lemma and phoneme
    :param audio_format: options: wav, ogg, flac, sph, nist. nist (NIST sphere format) and sph are the same.
    :param output_prefix:
    :return:
    """
    original_bliss_lexicon = get_bliss_lexicon(
        add_unknown_phoneme_and_mapping, add_silence=add_silence, output_prefix=output_prefix
    )
    corpus_name = "train"
    bliss_corpus = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)[corpus_name]

    g2p_augmenter = G2PBasedOovAugmenter(
        original_bliss_lexicon=original_bliss_lexicon,
        train_lexicon=original_bliss_lexicon,
    )
    augmented_bliss_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
        bliss_corpus=bliss_corpus,
        corpus_name=corpus_name,
        alias_path=os.path.join(output_prefix, "g2p"),
        casing="lower",
    )

    return augmented_bliss_lexicon
