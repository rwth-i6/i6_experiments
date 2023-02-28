import os
from functools import lru_cache

from sisyphus import tk

from i6_core.tools.download import DownloadJob


@lru_cache()
def get_arpa_lm_dict(output_prefix="datasets"):
    """
    Download the ARPA language models from OpenSLR,
    valid keys are: "3gram" and "4gram".

    :param str output_prefix:
    :return: A dictionary with Paths to the arpa lm files
    :rtype: dict[str, tk.Path]
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

    lm_prefix = os.path.join(output_prefix, "LibriSpeech", "lm")
    download_arpa_3gram_lm_job.add_alias(os.path.join(lm_prefix, "download_3gram_lm_job"))
    download_arpa_4gram_lm_job.add_alias(os.path.join(lm_prefix, "download_4gram_lm_job"))

    return lm_dict


@lru_cache
def get_librispeech_normalized_lm_data(output_prefix="datasets") -> tk.Path:
    """
    Download the official normalized LM data for LibriSpeech

    :param output_prefix:
    :return: gzipped text file containing the LM training data
    """
    download_job = DownloadJob(url="https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz")
    download_job.add_alias(os.path.join(output_prefix, "LibriSpeech", "lm", "download_lm_data"))
    return download_job.out_file
