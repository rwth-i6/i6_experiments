"""
Ref: /u/maximilian.azevedo/experiments/streaming_experiments/recipe/i6_experiments/users/azevedo/experiments/librispeech/ctc_rnnt_standalone_2025/librispeech_long
"""

import os
from functools import lru_cache
from typing import Optional

from sisyphus import tk

from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_core.corpus.convert import CorpusToTxtJob, CorpusReplaceOrthFromTxtJob
from i6_core.datasets.librispeech import *
from i6_core.meta.system import CorpusObject
from i6_core.text import PipelineJob

from i6_experiments.common.datasets.librispeech.constants import durations

from .librispeech_long import DownloadLibriSpeechLongCorpusJob, LibriSpeechLongCreateBlissCorpusJob


@lru_cache()
def get_bliss_corpus_dict(audio_format="flac", output_prefix="datasets"):
    """
    Download and create a bliss corpus for each of the LibriSpeech-Long test sets,
    and return all corpora as a single corpus dict.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str output_prefix:
    :return: A corpus dict with the following entries:
        - 'dev-clean'
        - 'dev-other'
        - 'test-clean'
        - 'test-other'
    :rtype: dict[str, tk.Path]
    """
    assert audio_format in ["flac", "ogg", "wav"]

    output_prefix = os.path.join(output_prefix, "LibriSpeechLong")

    download_metadata_job = DownloadLibriSpeechMetadataJob()
    download_metadata_job.add_alias(os.path.join(output_prefix, "download", "metadata_job"))

    download_corpus_job = DownloadLibriSpeechLongCorpusJob()
    download_corpus_job.add_alias(os.path.join(output_prefix, "download", "tests"))

    corpus_names = [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]

    def _get_corpus(corpus_name):
        create_bliss_corpus_job = LibriSpeechLongCreateBlissCorpusJob(
            corpus_folder=download_corpus_job.out_corpus_folders[corpus_name],
            speaker_metadata=download_metadata_job.out_speakers,
        )

        # remove dots
        bliss_text = CorpusToTxtJob(create_bliss_corpus_job.out_corpus).out_txt
        corpus_removed_dots = PipelineJob(text_files=bliss_text, pipeline=["sed 's/\.//g'"], mini_task=True).out
        bliss_corpus = CorpusReplaceOrthFromTxtJob(
            bliss_corpus=create_bliss_corpus_job.out_corpus, text_file=corpus_removed_dots
        )
        bliss_corpus.add_alias(os.path.join(output_prefix, "create_bliss", corpus_name))

        return bliss_corpus.out_corpus

    bliss_corpus_dict = {corpus_name: _get_corpus(corpus_name) for corpus_name in corpus_names}

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
                **audio_format_options[audio_format],
            )
            bliss_change_encoding_job.add_alias(
                os.path.join(
                    output_prefix,
                    "%s_conversion" % audio_format,
                    corpus_name,
                )
            )
            converted_bliss_corpus_dict[corpus_name] = bliss_change_encoding_job.out_corpus
    else:
        converted_bliss_corpus_dict = bliss_corpus_dict

    return converted_bliss_corpus_dict


@lru_cache()
def get_corpus_object_dict(audio_format="flac", output_prefix="datasets"):
    """
    Download and create a bliss corpus for each of the LibriSpeech-Long test sets,
    and return all corpora as a dict of CorpusObjects.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str output_prefix:
    :return: A corpus dict with the following entries:
        - 'dev-clean'
        - 'dev-other'
        - 'test-clean'
        - 'test-other'
    :rtype: dict[str, CorpusObject]
    """
    bliss_corpus_dict = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)

    corpus_object_dict = {}

    for corpus_name, bliss_corpus in bliss_corpus_dict.items():
        corpus_object = CorpusObject()
        corpus_object.corpus_file = bliss_corpus
        corpus_object.audio_format = audio_format
        corpus_object.audio_dir = None
        corpus_object.duration = durations[corpus_name]

        corpus_object_dict[corpus_name] = corpus_object

    return corpus_object_dict


@lru_cache()
def get_ogg_zip_dict(
    output_prefix: str = "datasets",
    returnn_python_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
):
    """
    Get a dictionary containing the paths to the ogg_zip for each corpus part.

    No outputs will be registered.

    :param str output_prefix:
    :param
    :return: dictionary with ogg zip paths for:
        - 'dev-clean'
        - 'dev-other'
        - 'test-clean'
        - 'test-other'
    :rtype: dict[str, tk.Path]
    """
    from i6_core.returnn.oggzip import BlissToOggZipJob

    ogg_zip_dict = {}
    bliss_corpus_dict = get_bliss_corpus_dict(audio_format="ogg", output_prefix=output_prefix)
    for name, bliss_corpus in bliss_corpus_dict.items():
        ogg_zip_job = BlissToOggZipJob(
            bliss_corpus,
            no_conversion=True,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
        )
        ogg_zip_job.add_alias(os.path.join(output_prefix, "LibriSpeechLong", "%s_ogg_zip_job" % name))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    return ogg_zip_dict
