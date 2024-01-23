import os
from functools import lru_cache
from typing import Dict, Optional, Any

from sisyphus import tk

from i6_core.audio.encoding import BlissChangeEncodingJob

from i6_core.corpus import SegmentCorpusJob
from i6_core.meta import CorpusObject

from .constants import DURATIONS
from .download import download_data_dict


@lru_cache()
def get_bliss_corpus_dict(audio_format: str = "wav", output_prefix: str = "datasets") -> Dict[str, tk.Path]:
    """
    creates a dictionary of all corpora in the TedLiumV2 dataset in the bliss xml format

    :param audio_format: options: wav, ogg, flac, sph, nist. nist (NIST sphere format) and sph are the same.
    :param output_prefix:
    :return:
    """
    assert audio_format in ["flac", "ogg", "wav", "sph", "nist"]

    output_prefix = os.path.join(output_prefix, "Ted-Lium-2")

    bliss_corpus_dict = download_data_dict(output_prefix=output_prefix).bliss_nist

    audio_format_options = {
        "wav": {
            "output_format": "wav",
            "codec": "pcm_s16le",
            "recover_duration": False,
        },
        "ogg": {"output_format": "ogg", "codec": "libvorbis", "recover_duration": False},
        "flac": {"output_format": "flac", "codec": "flac"},
    }

    converted_bliss_corpus_dict = {}
    if audio_format not in ["sph", "nist"]:
        for corpus_name, sph_corpus in bliss_corpus_dict.items():
            bliss_change_encoding_job = BlissChangeEncodingJob(
                corpus_file=sph_corpus,
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
def get_corpus_object_dict(audio_format: str = "flac", output_prefix: str = "datasets") -> Dict[str, CorpusObject]:
    """
    creates a dict of all corpora in the TedLiumV2 dataset as a `meta.CorpusObject`

    :param audio_format: options: wav, ogg, flac, sph, nist. nist (NIST sphere format) and sph are the same.
    :param output_prefix:
    :return:
    """
    bliss_corpus_dict = get_bliss_corpus_dict(audio_format=audio_format, output_prefix=output_prefix)

    corpus_object_dict = {}

    for corpus_name, bliss_corpus in bliss_corpus_dict.items():
        corpus_object = CorpusObject()
        corpus_object.corpus_file = bliss_corpus
        corpus_object.audio_format = audio_format
        corpus_object.audio_dir = None
        corpus_object.duration = DURATIONS[corpus_name]

        corpus_object_dict[corpus_name] = corpus_object

    return corpus_object_dict


@lru_cache()
def get_stm_dict(output_prefix: str = "datasets") -> Dict[str, tk.Path]:
    """
    fetches the STM files for TedLiumV2 dataset

    :param output_prefix:
    :return:
    """
    return download_data_dict(output_prefix=output_prefix).stm


def get_ogg_zip_dict(
    output_prefix: str = "datasets",
    returnn_python_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
    extra_args: Optional[Dict[str, Any]] = None,
):
    """
    Get a dictionary containing the paths to the ogg_zip for each corpus part.

    No outputs will be registered.

    :param str output_prefix:
    :param returnn_python_exe:
    :param returnn_root:
    :param extra_args: extra args for ogg zip job, like e.g. feature cache
    :return: dictionary with ogg zip paths
    :rtype: dict[str, tk.Path]
    """
    from i6_core.returnn.oggzip import BlissToOggZipJob

    ogg_zip_dict = {}
    bliss_corpus_dict = get_bliss_corpus_dict(audio_format="wav", output_prefix=output_prefix)
    for name, bliss_corpus in bliss_corpus_dict.items():
        segments = None
        if name == "train":
            segments = SegmentCorpusJob(bliss_corpus, 20).out_segment_path
        ogg_zip_job = BlissToOggZipJob(
            bliss_corpus,
            no_conversion=False,  # cannot be used here, because then each segment gets full recording audio
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            segments=segments,
            **(extra_args[name] or {}),
        )
        if segments is not None:
            ogg_zip_job.rqmt = {"mem": 8, "time": 2}
        ogg_zip_job.add_alias(os.path.join(output_prefix, "Ted-Lium-2", "%s_ogg_zip_job" % name))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    return ogg_zip_dict
