import os

from sisyphus import tk

from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_core.datasets.ljspeech import (
    DownloadLJSpeechCorpusJob,
    LJSpeechCreateBlissCorpusJob,
)
from i6_core.meta.system import CorpusObject


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
        download_ljspeech_job.add_alias(os.path.join(create_alias_with_prefix, "LJSpeech", "download_job"))
        bliss_corpus_job.add_alias(os.path.join(create_alias_with_prefix, "LJSpeech", "create_bliss_job"))
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
        corpus_file=ljspeech_22khz_corpus, output_format="wav", sample_rate=16000
    )
    if create_alias_with_prefix:
        sample_rate_conversion_job.add_alias(os.path.join(create_alias_with_prefix, "LJSpeech", "convert_16khz_job"))
    return sample_rate_conversion_job.out_corpus


def get_22khz_corpus_object(create_alias_with_prefix=None):
    """
    :param create_alias_with_prefix:
    :return:
    :rtype: CorpusObject
    """
    corpus_object = CorpusObject()
    corpus_object.corpus_file = get_22khz_bliss_corpus(create_alias_with_prefix)
    corpus_object.audio_format = "wav"
    corpus_object.audio_dir = None
    corpus_object.duration = 24.0

    return corpus_object


def get_16khz_corpus_object(create_alias_with_prefix=None):
    """
    :param create_alias_with_prefix:
    :return:
    :rtype: CorpusObject
    """
    corpus_object = CorpusObject()
    corpus_object.corpus_file = get_16khz_bliss_corpus(create_alias_with_prefix)
    corpus_object.audio_format = "wav"
    corpus_object.audio_dir = None
    corpus_object.duration = 24.0

    return corpus_object


def get_g2p(create_alias_with_prefix=None):
    """
    Create a default Sequitur-G2P model based on the CMU dict

    This G2P model can onle be used with "uppercase" data

    :param create_alias_with_prefix:
    :return: a sequitur model trained on the uppercased CMU lexicon
    :rtype: Path
    """
    from i6_core.lexicon.cmu import DownloadCMUDictJob
    from i6_core.text import PipelineJob

    download_cmu_lexicon_job = DownloadCMUDictJob()
    cmu_lexicon_lowercase = download_cmu_lexicon_job.out_cmu_lexicon
    convert_cmu_lexicon_uppercase_job = PipelineJob(
        cmu_lexicon_lowercase,
        ["tr '[:lower:]' '[:upper:]'"],
        zip_output=False,
        mini_task=True,
    )
    cmu_lexicon_uppercase = convert_cmu_lexicon_uppercase_job.out

    from i6_core.g2p.train import TrainG2PModelJob

    train_cmu_default_sequitur = TrainG2PModelJob(cmu_lexicon_uppercase)
    sequitur_cmu_uppercase_model = train_cmu_default_sequitur.out_best_model

    if create_alias_with_prefix:
        path_prefix = os.path.join(create_alias_with_prefix, "LJSpeech", "G2P")
        download_cmu_lexicon_job.add_alias(os.path.join(path_prefix, "download_cmu"))
        convert_cmu_lexicon_uppercase_job.add_alias(os.path.join(path_prefix, "convert_cmu"))
        train_cmu_default_sequitur.add_alias(os.path.join(path_prefix, "train_sequitur"))

    return sequitur_cmu_uppercase_model


def export(path_prefix):
    """
    :param str path_prefix:
    """
    ljspeech_22khz_bliss_corpus = get_22khz_bliss_corpus(create_alias_with_prefix=path_prefix)
    ljspeech_16khz_bliss_corpus = get_16khz_bliss_corpus(create_alias_with_prefix=path_prefix)
    ljspeech_sequitur_model = get_g2p(create_alias_with_prefix=path_prefix)

    tk.register_output(
        os.path.join(
            path_prefix,
            "LJSpeech",
            "ljspeech_22khz.xml.gz",
        ),
        ljspeech_22khz_bliss_corpus,
    )
    tk.register_output(
        os.path.join(
            path_prefix,
            "LJSpeech",
            "ljspeech_16khz.xml.gz",
        ),
        ljspeech_16khz_bliss_corpus,
    )
    tk.register_output(
        os.path.join(path_prefix, "LJSpeech", "ljspeech_sequitur_g2p.model"),
        ljspeech_sequitur_model,
    )
