"""
Functions related to lexicon creation
"""
__all__ = ["get_special_lemma_lexicon", "get_text_lexicon", "get_bliss_lexicon"]
from sisyphus import tk

import os

from i6_core.datasets.switchboard import (
    DownloadSwitchboardTranscriptionAndDictJob,
    CreateSwitchboardLexiconTextFileJob,
)
from i6_core.lexicon import LexiconFromTextFileJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_core.lib import lexicon
from i6_core.text.processing import PipelineJob

from .constants import SUBDIR_PREFIX


def get_special_lemma_lexicon() -> lexicon.Lexicon:
    """
    Generate the special phonemes/lemmas for Switchboard
    """
    lex = lexicon.Lexicon()

    tags = ["[SILENCE]", "[NOISE]", "[VOCALIZED-NOISE]", "[LAUGHTER]"]
    tag_to_phon = {
        "[SILENCE]": "[SILENCE]",
        "[NOISE]": "[NOISE]",
        "[VOCALIZED-NOISE]": "[VOCALIZEDNOISE]",
        "[LAUGHTER]": "[LAUGHTER]",
    }
    for tag in tags:
        lex.add_phoneme(tag_to_phon[tag], variation="none")

    # add non-special lemmas
    for tag in tags[1:]:  # silence is considered below
        lex.add_lemma(
            lexicon.Lemma(
                orth=[tag],
                phon=[tag_to_phon[tag]],
                synt=[],
                eval=[[]],
            )
        )

    # create special lemmas
    lex.add_lemma(
        lexicon.Lemma(
            orth=["[SENTENCE-END]"], synt=["</s>"], special="sentence-boundary"
        )
    )

    lex.add_lemma(
        lexicon.Lemma(
            orth=["[sentence-begin]"],
            synt=["<s>"],
            eval=[[]],
            special="sentence-begin",
        )
    )

    lex.add_lemma(
        lexicon.Lemma(
            orth=["[sentence-end]"], synt=["</s>"], eval=[[]], special="sentence-end"
        )
    )

    lex.add_lemma(
        lexicon.Lemma(
            orth=["[SILENCE]", ""],
            phon=["[SILENCE]"],
            synt=[],
            eval=[[]],
            special="silence",
        )
    )

    lex.add_lemma(
        lexicon.Lemma(orth=["[UNKNOWN]"], synt=["<unk>"], eval=[[]], special="unknown")
    )

    return lex


def get_text_lexicon(subdir_prefix: str = SUBDIR_PREFIX):
    """
    :param subdir_prefix:
    """
    raw_lexicon_dir = DownloadSwitchboardTranscriptionAndDictJob()

    # apply preprocessing on words to be consistent with the training corpus
    mapped_raw_lexicon_file_job = CreateSwitchboardLexiconTextFileJob(
        raw_lexicon_dir.out_raw_dict
    )
    mapped_raw_lexicon_file_job.add_alias(
        os.path.join(subdir_prefix, "create_lexicon_text_file_job")
    )
    lowercase_raw_lexicon_file_job = PipelineJob(
        mapped_raw_lexicon_file_job.out_dict,
        ["tr '[:upper:]' '[:lower:]'", "sort -u"],
        zip_output=False,
        mini_task=True,
    )
    lowercase_raw_lexicon_file_job.add_alias(
        os.path.join(subdir_prefix, "lowercase_lexicon_text_file_job")
    )

    return lowercase_raw_lexicon_file_job.out


def get_bliss_lexicon(subdir_prefix: str = SUBDIR_PREFIX) -> tk.Path:
    """
    Creates Switchboard bliss xml lexicon. The special lemmas are added via `get_special_lemma_lexicon` function.
    The original raw dictionary is downloaded from here:
    http://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz

    :param subdir_prefix: alias prefix name
    :return: Path to switchboard bliss lexicon
    """
    static_lexicon = get_special_lemma_lexicon()
    static_lexicon_job = WriteLexiconJob(
        static_lexicon=static_lexicon, sort_phonemes=True, sort_lemmata=False
    )
    static_lexicon_job.add_alias(
        os.path.join(subdir_prefix, "write_special_lemma_lexicon_job")
    )

    mapped_raw_lexicon_file = get_text_lexicon(subdir_prefix=subdir_prefix)

    bliss_lexicon = LexiconFromTextFileJob(mapped_raw_lexicon_file, compressed=True)
    bliss_lexicon.add_alias(os.path.join(subdir_prefix, "create_lexicon_job"))

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            bliss_lexicon.out_bliss_lexicon,
        ],
        sort_phonemes=True,
        sort_lemmata=False,
        compressed=True,
    )
    merge_lexicon_job.add_alias(os.path.join(subdir_prefix, "merge_lexicon_job"))

    return merge_lexicon_job.out_bliss_lexicon
