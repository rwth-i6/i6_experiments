import os
from functools import lru_cache
from sisyphus import tk

from i6_core.lexicon import LexiconFromTextFileJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_core.lib import lexicon
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter

from .constants import SILENCE, PHONEMES
from .corpus import get_bliss_corpus_dict
from .download import download_data_dict


@lru_cache()
def _get_special_lemma_lexicon() -> lexicon.Lexicon:
    lex = lexicon.Lexicon()
    lex.add_lemma(
        lexicon.Lemma(
            orth=["[silence]", ""],
            phon=[SILENCE],
            synt=[],
            special="silence",
            eval=[[]],
        )
    )
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
    lex.add_phoneme(SILENCE, variation="none")

    return lex


@lru_cache()
def _get_raw_bliss_lexicon(
    output_prefix: str,
) -> tk.Path:
    vocab = download_data_dict(output_prefix=output_prefix)["vocab"]

    convert_lexicon_job = LexiconFromTextFileJob(
        text_file=vocab,
        compressed=True,
    )
    convert_lexicon_job.add_alias(
        os.path.join(output_prefix, "convert_text_to_bliss_lexicon_job")
    )

    return convert_lexicon_job.out_bliss_lexicon


@lru_cache()
def get_bliss_lexicon(
    output_prefix="datasets",
) -> tk.Path:
    static_lexicon = _get_special_lemma_lexicon()
    static_lexicon_job = WriteLexiconJob(
        static_lexicon, sort_phonemes=True, sort_lemmata=False
    )
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
    output_prefix="datasets",
) -> tk.Path:
    original_bliss_lexicon = get_bliss_lexicon(output_prefix=output_prefix)
    corpus_name = "train"
    bliss_corpus = get_bliss_corpus_dict(output_prefix=output_prefix)[corpus_name]

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
