import os

from sisyphus import tk

from i6_core.datasets.ljspeech import DownloadLJSpeechCorpusJob, LJSpeechCreateBlissCorpusJob
from i6_core.audio.encoding import BlissChangeEncodingJob


def get_22khz_bliss_corpus(create_alias_with_prefix=None):
    """
    Download LJSpeech and create the bliss corpus file

    Keep the original sampling rate of 22050 Hz

    :param str create_alias_with_prefix:
    :return: path to LJSpeech bliss corpus
    :rtype: Path
    """
    download_ljspeech_job = DownloadLJSpeechCorpusJob()
    bliss_corpus_job = LJSpeechCreateBlissCorpusJob(
        metadata=download_ljspeech_job.out_metadata,
        audio_folder=download_ljspeech_job.out_audio_folder,
    )
    if create_alias_with_prefix:
        download_ljspeech_job.add_alias(
            os.path.join(create_alias_with_prefix, "LJSpeech", "download_job")
        )
        bliss_corpus_job.add_alias(
            os.path.join(create_alias_with_prefix, "LJSpeech", "create_bliss_job")
        )
    return bliss_corpus_job.out_bliss_corpus


def get_16khz_bliss_corpus(create_alias_with_prefix=None):
    """
    Download LJSpeech and create the bliss corpus file

    Resamples the corpus to 16 kHz before returning

    :param str create_alias_with_prefix:
    :return: path to LJSpeech bliss corpus
    :rtype: Path
    """
    ljspeech_22khz_corpus = get_22khz_bliss_corpus(create_alias_with_prefix)
    sample_rate_conversion_job = BlissChangeEncodingJob(
        corpus_file=ljspeech_22khz_corpus,
        output_format="wav",
        sample_rate=16000
    )
    if create_alias_with_prefix:
        sample_rate_conversion_job.add_alias(
            os.path.join(create_alias_with_prefix, "LJSpeech", "convert_16khz_job")
        )
    return sample_rate_conversion_job.out_corpus


def export(path_prefix):
    """
    :param str path_prefix:
    """
    ljspeech_22khz_bliss_corpus = get_22khz_bliss_corpus(create_alias_with_prefix=path_prefix)
    ljspeech_16khz_bliss_corpus = get_16khz_bliss_corpus(create_alias_with_prefix=path_prefix)
    tk.register_output(
        os.path.join(path_prefix, "LJSpeech", "ljspeech_22khz.xml.gz",), ljspeech_22khz_bliss_corpus
    )
    tk.register_output(
        os.path.join(path_prefix, "LJSpeech", "ljspeech_16khz.xml.gz",), ljspeech_16khz_bliss_corpus
    )


