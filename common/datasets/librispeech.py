"""
This file contains helper functions for the (common) pipeline steps needed to use the LibriSpeech corpus.
It will download and convert the corpus parts that are used in later steps.
(and ONLY those, no unneeded corpus jobs will be registered as output)

The corpora can be accessed in 3 ways:
 - as bliss xml with a specific audio format: get_bliss_corpus_dict
 - as meta.System.CorpusObject with a specific format and duration set: get_corpus_object_dict
 - as ogg zip file (containing .oggs): get_ogg_zip_dict

All corpus functions return a dict with the following keys:
- "dev-clean"
- "dev-other"
- "test-clean"
- "test-other"
- "train-clean-100"
- "train-clean-360"
- "train-clean-460"
- "train-other-500"
- "train-other-960"

Available language models can be accessed with ``get_arpa_lm_dict``:
 - "3gram" for the non-pruned 3-gram LM
 - "4gram" for the non-pruned 4-gram LM

If you want to use other subsets (especially with .ogg zips),
please consider to use segment lists to avoid creating new corpus files.

All alias and output paths will be under: ``<subdir_prefix>/LibriSpeech/....``

For i6-users: physical jobs generated via the "export" functions
are located in: `/work/common/asr/librispeech/data/sisyphus_work_dir/`
"""
import os

from sisyphus import tk

from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_core.corpus.transform import MergeCorporaJob, MergeStrategy
from i6_core.datasets.librispeech import *
from i6_core.lib import lexicon
from i6_core.lexicon.conversion import LexiconFromTextFileJob
from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_core.meta.system import CorpusObject
from i6_core.tools.download import DownloadJob


durations = {
    "train-clean-100": 100.6,
    "train-clean-360": 363.6,
    "train-other-500": 496.7,
    "dev-clean": 5.4,
    "dev-other": 5.3,
    "test-clean": 5.4,
    "test-other": 5.1,
}

durations["train-clean-460"] = (
    durations["train-clean-100"] + durations["train-clean-360"]
)
durations["train-other-960"] = (
    durations["train-clean-460"] + durations["train-other-500"]
)


def get_bliss_corpus_dict(audio_format="flac", subdir_prefix=""):
    """
    Download and create a bliss corpus for each of the LibriSpeech training corpora and test sets,
    and return all corpora as a single corpus dict.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str subdir_prefix:
    :return: A corpus dict with the following entries:
        - 'dev-clean'
        - 'dev-other'
        - 'test-clean'
        - 'test-other'
        - 'train-clean-100'
        - 'train-clean-360'
        - 'train-clean-460'
        - 'train-other-500'
        - 'train-other-960'
    :rtype: dict[str, Path]
    """
    assert audio_format in ["flac", "ogg", "wav"]

    subdir_prefix = os.path.join(subdir_prefix, "LibriSpeech")

    download_metadata_job = DownloadLibriSpeechMetadataJob()
    download_metadata_job.add_alias(
        os.path.join(subdir_prefix, "download", "metadata_job")
    )

    def _get_corpus(corpus_name):
        download_corpus_job = DownloadLibriSpeechCorpusJob(corpus_key=corpus_name)
        create_bliss_corpus_job = LibriSpeechCreateBlissCorpusJob(
            corpus_folder=download_corpus_job.out_corpus_folder,
            speaker_metadata=download_metadata_job.out_speakers,
        )
        download_corpus_job.add_alias(
            os.path.join(subdir_prefix, "download_job", corpus_name)
        )
        create_bliss_corpus_job.add_alias(
            os.path.join(subdir_prefix, "create_bliss_job", corpus_name)
        )
        return create_bliss_corpus_job.out_corpus

    corpus_names = [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]

    bliss_corpus_dict = {
        corpus_name: _get_corpus(corpus_name) for corpus_name in corpus_names
    }

    audio_format_options = {
        "wav": {
            "output_format": "wav",
            "codec": "pcm_s16le",
        },
        "ogg": {"output_format": "ogg", "codec": "libvorbis"},
    }

    if audio_format != "flac":
        converted_bliss_corpus_dict = {}
        for corpus_name, flac_corpus in bliss_corpus_dict.items():
            bliss_change_encoding_job = BlissChangeEncodingJob(
                corpus_file=flac_corpus,
                sample_rate=16000,
                **audio_format_options[audio_format]
            )
            bliss_change_encoding_job.add_alias(
                os.path.join(
                    subdir_prefix,
                    "%s_conversion" % audio_format,
                    corpus_name,
                )
            )
            converted_bliss_corpus_dict[
                corpus_name
            ] = bliss_change_encoding_job.out_corpus
    else:
        converted_bliss_corpus_dict = bliss_corpus_dict

    def _merge_corpora(corpora, name):
        merge_job = MergeCorporaJob(
            bliss_corpora=corpora, name=name, merge_strategy=MergeStrategy.FLAT
        )
        merge_job.add_alias(os.path.join(subdir_prefix, "%s_merge_job" % audio_format, name))
        return merge_job.out_merged_corpus

    converted_bliss_corpus_dict["train-clean-460"] = _merge_corpora(
        corpora=[
            converted_bliss_corpus_dict["train-clean-100"],
            converted_bliss_corpus_dict["train-clean-360"],
        ],
        name="train-clean-460",
    )

    converted_bliss_corpus_dict["train-other-960"] = _merge_corpora(
        corpora=[
            converted_bliss_corpus_dict["train-clean-100"],
            converted_bliss_corpus_dict["train-clean-360"],
            converted_bliss_corpus_dict["train-other-500"],
        ],
        name="train-other-960",
    )

    return converted_bliss_corpus_dict


def get_corpus_object_dict(audio_format="flac", subdir_prefix=""):
    """
    Download and create a bliss corpus for each of the LibriSpeech training corpora and test sets,
    and return all corpora as a dict of CorpusObjects.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str subdir_prefix:
    :return: A corpus dict with the following entries:
        - 'dev-clean'
        - 'dev-other'
        - 'test-clean'
        - 'test-other'
        - 'train-clean-100'
        - 'train-clean-360'
        - 'train-clean-460'
        - 'train-other-500'
        - 'train-other-960'
    :rtype: dict[str, CorpusObject]
    """
    bliss_corpus_dict = get_bliss_corpus_dict(
        audio_format=audio_format, subdir_prefix=subdir_prefix
    )

    corpus_object_dict = {}

    for corpus_name, bliss_corpus in bliss_corpus_dict.items():
        corpus_object = CorpusObject()
        corpus_object.corpus_file = bliss_corpus
        corpus_object.audio_format = audio_format
        corpus_object.audio_dir = None
        corpus_object.duration = durations[corpus_name]

        corpus_object_dict[corpus_name] = corpus_object

    return corpus_object_dict


def get_ogg_zip_dict(subdir_prefix=""):
    """
    Get a dictionary containing the paths to the ogg_zip for each corpus part.

    No outputs will be registered.

    :param str subdir_prefix:
    :return: dictionary with ogg zip paths for:
        - 'dev-clean'
        - 'dev-other'
        - 'test-clean'
        - 'test-other'
        - 'train-clean-100'
        - 'train-clean-360'
        - 'train-clean-460'
        - 'train-other-500'
        - 'train-other-960'
    :rtype: dict[str, Path]
    """
    from i6_core.returnn.oggzip import BlissToOggZipJob

    ogg_zip_dict = {}
    bliss_corpus_dict = get_bliss_corpus_dict(
        audio_format="ogg", subdir_prefix=subdir_prefix
    )
    for name, bliss_corpus in bliss_corpus_dict.items():
        ogg_zip_job = BlissToOggZipJob(bliss_corpus, no_conversion=True)
        ogg_zip_job.add_alias(
            os.path.join(subdir_prefix, "LibriSpeech", "%s_ogg_zip_job" % name)
        )
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    return ogg_zip_dict


def get_arpa_lm_dict(subdir_prefix=""):
    """
    Download the ARPA language models from OpenSLR,
    valid keys are: "3gram" and "4gram".

    :param str subdir_prefix:
    :return: A dictionary with Paths to the arpa lm files
    :rtype: dict[str, Path]
    """
    lm_dict = {}

    download_arpa_4gram_lm_job = DownloadJob(
        url="https://www.openslr.org/resources/11/4-gram.arpa.gz",
        target_filename="4-gram.arpa.gz",
        checksum="f2b2d1507637ddf459d3579159f7e8099ed7d77452ff1059aeeeaea33d274613",
    )
    lm_dict["4gram"] = download_arpa_4gram_lm_job.out_file

    download_arpa_3gram_lm_job = DownloadJob(
        url="https://www.openslr.org/resources/11/3-gram.arpa.gz",
        target_filename="3-gram.arpa.gz",
        checksum="263649573475c2991d3e755eb4e690c9d2656f2b3283a1eb589e1e4e174bf874",
    )
    lm_dict["3gram"] = download_arpa_3gram_lm_job.out_file

    lm_prefix = os.path.join(subdir_prefix, "LibriSpeech", "lm")
    download_arpa_3gram_lm_job.add_alias(os.path.join(lm_prefix, "download_3gram_lm_job"))
    download_arpa_4gram_lm_job.add_alias(os.path.join(lm_prefix, "download_4gram_lm_job"))

    return lm_dict


def get_special_lemma_lexicon():
    """
    Generate the special lemmas for LibriSpeech

    Librispeech uses silence, sentence begin/end and unknown, but no other special tokens.

    :return: the lexicon with special lemmas and phonemes
    :rtype: lexicon.Lexicon
    """
    lex = lexicon.Lexicon()
    lex.add_lemma(
        lexicon.Lemma(
            orth=["[SILENCE]", ""],
            phon=["[SILENCE]"],
            synt=[[]],
            special="silence",
            eval=[[]],
        )
    )
    lex.add_lemma(
        lexicon.Lemma(
            orth=["[SENTENCE-BEGIN]"], synt=[["<s>"]], special="sentence-begin"
        )
    )
    lex.add_lemma(
        lexicon.Lemma(orth=["[SENTENCE-END]"], synt=[["</s>"]], special="sentence-end")
    )
    lex.add_lemma(
        lexicon.Lemma(
            orth=["[UNKNOWN]"],
            phon=["[UNKNOWN]"],
            synt=[["<UNK>"]],
            special="unknown",
        )
    )

    lex.add_phoneme("[SILENCE]", variation="none")
    lex.add_phoneme("[UNKNOWN]", variation="none")
    return lex


def get_bliss_lexicon(subdir_prefix="", use_stress_marker=False):
    """
    Create the full LibriSpeech bliss lexicon based on the static lexicon
    with special lemmas and the converted official lexicon from OpenSLR
    here: https://www.openslr.org/resources/11/

    The phoneme inventory is ordered alphabetically, with the special phonemes for silence and unknown at the end,
    while the special lemmas come first. This way the result resembles the "legacy" lexicon closely, and all
    "special" entries are at one position.

    :param str subdir_prefix:
    :param bool use_stress_marker:
    :return: Path to LibriSpeech bliss lexicon
    :rtype: Path
    """
    static_lexicon = get_special_lemma_lexicon()
    static_lexicon_job = WriteLexiconJob(
        static_lexicon, sort_phonemes=True, sort_lemmata=False
    )

    download_lexicon_job = DownloadJob(
        url="https://www.openslr.org/resources/11/librispeech-lexicon.txt",
        target_filename="librispeech-lexicon.txt",
        checksum="d722bc29908cd338ae738edd70f61826a6fca29aaa704a9493f0006773f79d71",
    )

    text_lexicon = download_lexicon_job.out_file
    if use_stress_marker:
        from i6_core.text import PipelineJob
        eliminate_stress_job = PipelineJob(
            text_lexicon,
            ["sed 's/[0-9]//g'"],
            zip_output=False,
            mini_task=True
        )
        text_lexicon = eliminate_stress_job.out


    convert_lexicon_job = LexiconFromTextFileJob(
        text_file=text_lexicon, compressed=True
    )

    merge_lexicon_job = MergeLexiconJob(
        bliss_lexica=[
            static_lexicon_job.out_bliss_lexicon,
            convert_lexicon_job.out_bliss_lexicon,
        ],
        sort_phonemes=True,
        sort_lemmata=False,
        compressed=True,
    )
    alias_path = os.path.join(subdir_prefix, "LibriSpeech", "lexicon")
    static_lexicon_job.add_alias(os.path.join(alias_path, "static_lexicon_job"))
    download_lexicon_job.add_alias(os.path.join(alias_path, "download_lexicon_job"))
    convert_lexicon_job.add_alias(
        os.path.join(alias_path, "convert_text_to_bliss_lexicon_job")
    )
    merge_lexicon_job.add_alias(os.path.join(alias_path, "merge_lexicon_job"))

    return merge_lexicon_job.out_bliss_lexicon


def get_lm_vocab(subdir_prefix=""):
    """
    :param str subdir_prefix:
    :return: Path to LibriSpeech vocab file (one word per line)
    :rtype: Path
    """
    download_lm_vocab_job = DownloadJob(
        url="https://www.openslr.org/resources/11/librispeech-vocab.txt",
        target_filename="librispeech-vocab.txt",
        checksum="3014e72dffff09cb1a9657f31cfe2e04c1301610a6127a807d1d708b986b5474",
    )
    download_lm_vocab_job.add_alias(
        os.path.join(subdir_prefix, "LibriSpeech", "download_lm_vocab_job")
    )
    return download_lm_vocab_job.out_file


def _export_datasets(subdir_prefix):
    """
    :param str subdir_prefix:
    """

    # export all bliss corpora
    for audio_format in ["flac", "ogg", "wav"]:
        bliss_corpus_dict = get_bliss_corpus_dict(
            audio_format=audio_format, subdir_prefix=subdir_prefix
        )
        for name, bliss_corpus in bliss_corpus_dict.items():
            tk.register_output(
                os.path.join(
                    subdir_prefix, "LibriSpeech", "%s-%s.xml.gz" % (name, audio_format)
                ),
                bliss_corpus,
            )

    # export all ogg zip corpora
    ogg_corpus_dict = get_ogg_zip_dict(subdir_prefix=subdir_prefix)
    for name, ogg_corpus in ogg_corpus_dict.items():
        tk.register_output(
            os.path.join(subdir_prefix, "LibriSpeech", "%s.ogg.zip" % name), ogg_corpus
        )


def _export_lm_data(subdir_prefix):
    """
    :param str subdir_prefix:
    """
    lm_dict = get_arpa_lm_dict(subdir_prefix=subdir_prefix)
    tk.register_output(
        os.path.join(subdir_prefix, "LibriSpeech", "lm", "3-gram.arpa.gz"),
        lm_dict["3gram"],
    )
    tk.register_output(
        os.path.join(subdir_prefix, "LibriSpeech", "lm", "4-gram.arpa.gz"),
        lm_dict["4gram"],
    )


def _export_lexicon_and_vocab(subdir_prefix):
    """
    :param str subdir_prefix:
    """
    bliss_lexicon = get_bliss_lexicon(
        subdir_prefix=subdir_prefix,
        use_stress_marker=True
    )
    tk.register_output(
        os.path.join(subdir_prefix, "LibriSpeech", "librispeech.lexicon.folded.xml.gz"),
        bliss_lexicon,
    )

    bliss_lexicon = get_bliss_lexicon(
        subdir_prefix=subdir_prefix,
        use_stress_marker=False
    )
    tk.register_output(
        os.path.join(subdir_prefix, "LibriSpeech", "librispeech.lexicon.xml.gz"),
        bliss_lexicon,
    )


def export_all(subdir_prefix):
    """
    Registers all LibriSpeech related data as output.

    For internal i6 purposes only:

    physical jobs are located in: `/work/common/asr/librispeech/data/sisyphus_work_dir/`

    :param str subdir_prefix:
    """
    _export_datasets(subdir_prefix)
    _export_lm_data(subdir_prefix)
    _export_lexicon_and_vocab(subdir_prefix)
