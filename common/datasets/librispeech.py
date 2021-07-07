"""
This file contains helper functions for the (common) pipeline steps needed to use the LibriSpeech corpus.
It will download and convert the corpus parts that are used in later steps.
(and ONLY those, no unneeded corpus jobs will be registered as output)

The corpora can be accessed in 3 ways:
 - as bliss xml with a specific audio format: get_bliss_corpus_dict
 - as meta.System.CorpusObject with a specific format and duration set: get_corpus_object_dict
 - as ogg zip file (containing .oggs): get_ogg_zip_dict

All functions return a dict with the following keys:
- 'dev-clean'
- 'dev-other'
- 'test-clean'
- 'test-other'
- 'train-clean-100'
- 'train-clean-360'
- 'train-clean-460'
- 'train-other-500'
- 'train-other-960'

If you want to use other subsets (especially with .ogg zips),
please consider to use segment lists to avoid creating new corpus files.

For i6-users: physical jobs are located in: `/work/common/asr/librispeech/data/sisyphus_work_dir/`
"""
import os

from sisyphus import tk

from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_core.corpus.transform import MergeCorporaJob, MergeStrategy
from i6_core.datasets.librispeech import *
from i6_core.meta.system import CorpusObject


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


def get_bliss_corpus_dict(audio_format="flac", create_alias_with_prefix=None):
    """
    Download and create a bliss corpus for each of the LibriSpeech training corpora and test sets,
    and return all corpora as a single corpus dict.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str|None create_alias_with_prefix:
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

    create_alias_with_prefix = (
        os.path.join(create_alias_with_prefix, "LibriSpeech")
        if create_alias_with_prefix
        else None
    )

    download_metadata_job = DownloadLibriSpeechMetadataJob()
    if create_alias_with_prefix:
        download_metadata_job.add_alias(
            os.path.join(create_alias_with_prefix, "download", "metadata_job")
        )

    def _get_corpus(corpus_name):
        download_corpus_job = DownloadLibriSpeechCorpusJob(corpus_key=corpus_name)
        create_bliss_corpus_job = LibriSpeechCreateBlissCorpusJob(
            corpus_folder=download_corpus_job.out_corpus_folder,
            speaker_metadata=download_metadata_job.out_speakers,
        )
        if create_alias_with_prefix:
            download_corpus_job.add_alias(
                os.path.join(create_alias_with_prefix, "download_job", corpus_name)
            )
            create_bliss_corpus_job.add_alias(
                os.path.join(create_alias_with_prefix, "create_bliss_job", corpus_name)
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
            if create_alias_with_prefix:
                bliss_change_encoding_job.add_alias(
                    os.path.join(
                        create_alias_with_prefix,
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
        if create_alias_with_prefix:
            merge_job.add_alias(
                os.path.join(
                    create_alias_with_prefix, "%s_merge_job" % audio_format, name
                )
            )
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


def get_corpus_object_dict(audio_format="flac", create_alias_with_prefix=None):
    """
    Download and create a bliss corpus for each of the LibriSpeech training corpora and test sets,
    and return all corpora as a dict of CorpusObjects.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str|None create_alias_with_prefix:
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
        audio_format=audio_format, create_alias_with_prefix=create_alias_with_prefix
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


def get_ogg_zip_dict(create_alias_with_prefix=None):
    """
    Get a dictionary containing the paths to the ogg_zip for each corpus part.

    No outputs will be registered.

    :param str|None create_alias_with_path_prefix:
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
        audio_format="ogg", create_alias_with_prefix=create_alias_with_prefix
    )
    for name, bliss_corpus in bliss_corpus_dict.items():
        ogg_zip_job = BlissToOggZipJob(bliss_corpus, no_conversion=True)
        if create_alias_with_prefix:
            ogg_zip_job.add_alias(
                os.path.join(
                    create_alias_with_prefix, "LibriSpeech", "%s_ogg_zip_job" % name
                )
            )
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    return ogg_zip_dict


def export_all_datasets(path_prefix):
    """
    export all datasets to path_prefix/LibriSpeech/<corpus_name>.xml.gz

    For i6-users: physical jobs are located in: `/work/common/asr/librispeech/data/sisyphus_work_dir/`

    :param str path_prefix:
    :return:
    """

    # export all bliss corpora
    for audio_format in ["flac", "ogg", "wav"]:
        bliss_corpus_dict = get_bliss_corpus_dict(
            audio_format=audio_format, create_alias_with_prefix=path_prefix
        )
        for name, bliss_corpus in bliss_corpus_dict.items():
            tk.register_output(
                os.path.join(
                    path_prefix, "LibriSpeech", "%s-%s.xml.gz" % (name, audio_format)
                ),
                bliss_corpus,
            )

    # export all ogg zip corpora
    ogg_corpus_dict = get_ogg_zip_dict(create_alias_with_prefix=path_prefix)
    for name, ogg_corpus in ogg_corpus_dict.items():
        tk.register_output(
            os.path.join(path_prefix, "LibriSpeech", "%s.ogg.zip" % name), ogg_corpus
        )
