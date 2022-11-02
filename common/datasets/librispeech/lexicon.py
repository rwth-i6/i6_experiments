import os
from functools import lru_cache

from i6_core.lexicon import LexiconFromTextFileJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_core.lib import lexicon
from i6_core.tools.download import DownloadJob
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter


@lru_cache()
def _get_special_lemma_lexicon(add_unknown_phoneme_and_mapping=True):
    """
    Generate the special lemmas for LibriSpeech

    Librispeech uses silence, sentence begin/end and unknown, but no other special tokens.

    :param bool add_unknown_phoneme_and_mapping: add [UNKNOWN] as phoneme, otherwise add only the lemma without it
    :return: the lexicon with special lemmas and phonemes
    :rtype: lexicon.Lexicon
    """
    lex = lexicon.Lexicon()
    lex.add_lemma(
        lexicon.Lemma(
            orth=["[SILENCE]", ""],
            phon=["[SILENCE]"],
            synt=[],
            special="silence",
            eval=[[]],
        )
    )
    lex.add_lemma(
        lexicon.Lemma(orth=["[SENTENCE-BEGIN]"], synt=["<s>"], special="sentence-begin")
    )
    lex.add_lemma(
        lexicon.Lemma(orth=["[SENTENCE-END]"], synt=["</s>"], special="sentence-end")
    )
    if add_unknown_phoneme_and_mapping:
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[UNKNOWN]"],
                phon=["[UNKNOWN]"],
                synt=["<UNK>"],
                special="unknown",
            )
        )
    else:
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[UNKNOWN]"],
                synt=["<UNK>"],
                special="unknown",
            )
        )

    lex.add_phoneme("[SILENCE]", variation="none")
    if add_unknown_phoneme_and_mapping:
        lex.add_phoneme("[UNKNOWN]", variation="none")
    return lex


@lru_cache()
def _get_raw_bliss_lexicon(
    use_stress_marker,
    alias_path,
):
    """
    helper function to download and convert the official lexicon from OpenSLR
    here: https://www.openslr.org/resources/11/

    No special lemmas are added

    :param bool use_stress_marker:
    :param str alias_path: alias_path already including the lexicon path passed from another function
    :return: Path to LibriSpeech bliss lexicon
    :rtype: Path
    """
    download_lexicon_job = DownloadJob(
        url="https://www.openslr.org/resources/11/librispeech-lexicon.txt",
        target_filename="librispeech-lexicon.txt",
        checksum="d722bc29908cd338ae738edd70f61826a6fca29aaa704a9493f0006773f79d71",
    )

    text_lexicon = download_lexicon_job.out_file
    if not use_stress_marker:
        from i6_core.text import PipelineJob

        eliminate_stress_job = PipelineJob(
            text_lexicon, ["sed 's/[0-9]//g'"], zip_output=False, mini_task=True
        )
        eliminate_stress_job.add_alias(os.path.join(alias_path, "remove_stress_marker"))
        text_lexicon = eliminate_stress_job.out

    convert_lexicon_job = LexiconFromTextFileJob(
        text_file=text_lexicon, compressed=True
    )

    download_lexicon_job.add_alias(os.path.join(alias_path, "download_lexicon_job"))
    convert_lexicon_job.add_alias(
        os.path.join(alias_path, "convert_text_to_bliss_lexicon_job")
    )

    return convert_lexicon_job.out_bliss_lexicon


@lru_cache()
def get_bliss_lexicon(
    use_stress_marker=False,
    add_unknown_phoneme_and_mapping=True,
    output_prefix="datasets",
):
    """
    Create the full LibriSpeech bliss lexicon based on the static lexicon
    with special lemmas and the converted official lexicon from OpenSLR
    here: https://www.openslr.org/resources/11/

    The phoneme inventory is ordered alphabetically, with the special phonemes for silence and unknown at the end,
    while the special lemmas come first. This way the result resembles the "legacy" lexicon closely, and all
    "special" entries are at one position.
    Librispeech standard phoneme inventorory contains also phonemes with stress. By not including these variants,
    the phoneme inventory is reduced from to 42 phonemes.

    :param bool use_stress_marker:
    :param bool add_unknown_phoneme_and_mapping: add [UNKNOWN] as phoneme, otherwise add only the lemma without it
    :param str output_prefix:
    :return: Path to LibriSpeech bliss lexicon
    :rtype: Path
    """
    alias_path = os.path.join(
        output_prefix,
        "LibriSpeech",
        "%s_lexicon_%s"
        % (
            "regular" if use_stress_marker else "folded",
            "with_unknown" if add_unknown_phoneme_and_mapping else "without_unknown",
        ),
    )

    static_lexicon = _get_special_lemma_lexicon(
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping
    )
    static_lexicon_job = WriteLexiconJob(
        static_lexicon, sort_phonemes=True, sort_lemmata=False
    )

    raw_librispeech_lexicon = _get_raw_bliss_lexicon(
        use_stress_marker=use_stress_marker, alias_path=alias_path
    )

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            raw_librispeech_lexicon,
        ],
        sort_phonemes=True,
        sort_lemmata=False,
        compressed=True,
    )
    static_lexicon_job.add_alias(os.path.join(alias_path, "static_lexicon_job"))
    merge_lexicon_job.add_alias(os.path.join(alias_path, "merge_lexicon_job"))

    return merge_lexicon_job.out_bliss_lexicon


@lru_cache()
def get_g2p_augmented_bliss_lexicon_dict(
    use_stress_marker=False,
    add_unknown_phoneme_and_mapping=True,
    output_prefix="datasets",
):
    """
    Given the original LibriSpeech bliss lexicon, it is possible to estimate the pronunciation for
    out of vocabulary (OOV) words for each of the LibriSpeech training corpora. Here, we create a dictionary
    that has different train corpora as keys and the corresponding g2p augmented bliss lexicon as values

    :param bool use_stress_marker: uses phoneme symbols with stress markers
    :param bool add_unknown_phoneme_and_mapping: add [UNKNOWN] as phoneme, otherwise add only the lemma without it
    :param str output_prefix:
    :param g2p_train_args:
    :param g2p_apply_args:
    :return: dictionary of Paths to augmented bliss_lexicon
    :rtype: dict[str, Path]
    """
    alias_path = os.path.join(
        output_prefix,
        "LibriSpeech",
        "%s_lexicon%s"
        % (
            "regular" if use_stress_marker else "folded",
            "_with_unk" if add_unknown_phoneme_and_mapping else "",
        ),
    )
    augmented_bliss_lexica = {}

    original_bliss_lexicon = get_bliss_lexicon(
        use_stress_marker=use_stress_marker,
        output_prefix=output_prefix,
        add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
    )
    current_bliss_lexicon = original_bliss_lexicon

    bliss_corpus_dict = get_bliss_corpus_dict(output_prefix=output_prefix)
    for corpus_name, bliss_corpus in sorted(bliss_corpus_dict.items()):
        if "train" in corpus_name:
            if corpus_name in ["train-clean-460", "train-other-960"]:
                augmented_bliss_lexica[corpus_name] = current_bliss_lexicon
            else:
                g2p_augmenter = G2PBasedOovAugmenter(
                    original_bliss_lexicon=current_bliss_lexicon,
                    train_lexicon=original_bliss_lexicon,
                )
                current_bliss_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
                    bliss_corpus=bliss_corpus,
                    corpus_name=corpus_name,
                    alias_path=alias_path,
                    casing="upper",
                )
                augmented_bliss_lexica[corpus_name] = current_bliss_lexicon

    return augmented_bliss_lexica
