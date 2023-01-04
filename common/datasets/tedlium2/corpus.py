import os
from functools import lru_cache
from typing import Dict, Optional

from sisyphus import tk

from i6_core.audio.encoding import BlissChangeEncodingJob

from i6_core.meta import CorpusObject

from .constants import DURATIONS
from .download import download_data_dict


@lru_cache()
def get_bliss_corpus_dict(
    audio_format: str = "wav", output_prefix: str = "datasets"
) -> Dict[str, tk.Path]:
    """
    :param audio_format: options: wav, ogg, flac, sph, nist. nist (NIST sphere format) and sph are the same.
    :param output_prefix:
    :return:
    """
    assert audio_format in ["flac", "ogg", "wav", "sph", "nist"]

    output_prefix = os.path.join(output_prefix, "Ted-Lium-2")

    bliss_corpus_dict = download_data_dict(output_prefix=output_prefix)["bliss_nist"]

    audio_format_options = {
        "wav": {
            "output_format": "wav",
            "codec": "pcm_s16le",
        },
        "ogg": {"output_format": "ogg", "codec": "libvorbis"},
        "flac": {"output_format": "flac", "codec": "flac"},
    }

    converted_bliss_corpus_dict = {}
    if audio_format != "sph" or audio_format != "nist":
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
            converted_bliss_corpus_dict[
                corpus_name
            ] = bliss_change_encoding_job.out_corpus
        else:
            converted_bliss_corpus_dict = bliss_corpus_dict

    return converted_bliss_corpus_dict


@lru_cache()
def get_corpus_object_dict(
    audio_format: str = "flac", output_prefix: str = "datasets"
) -> Dict[str, CorpusObject]:
    bliss_corpus_dict = get_bliss_corpus_dict(
        audio_format=audio_format, output_prefix=output_prefix
    )

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
    return download_data_dict(output_prefix=output_prefix)["stm"]
