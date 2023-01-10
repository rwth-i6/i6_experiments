"""
Functions related to speaker and training corpus creation
"""
__all__ = [
    "get_train_bliss_corpus_ldc",
    "get_train_bliss_corpus_i6_legacy",
    "get_train_corpus_object_ldc",
    "get_train_corpus_object_i6_legacy",
]

import os
from sisyphus import tk

from i6_core.tools.download import DownloadJob
from i6_core.datasets.switchboard import (
    DownloadSwitchboardTranscriptionAndDictJob,
    CreateSwitchboardBlissCorpusJob,
    SwitchboardSphereToWaveJob,
    DownloadSwitchboardSpeakersStatsJob,
    CreateSwitchboardSpeakersListJob,
    CreateLDCSwitchboardSpeakerListJob,
)

from i6_experiments.common.datasets.util import CorpusObject

from .constants import SUBDIR_PREFIX, durations
from .paths import SWITCHBOARD1_PATH, SWITCHBOARD1_LEGACY_PATH


def get_train_bliss_corpus(
    audio_dir: tk.Path,
    skip_empty_ldc_file: bool = False,
    subdir_prefix: str = SUBDIR_PREFIX,
) -> tk.Path:
    """
    Returns Switchboard training bliss corpus

    :param audio_dir: path for audio data
    :param subdir_prefix: alias name prefix
    :param skip_empty_ldc_file: exclude sw2167B which is empty in the official LDC download
    :return: Path to switchboard training corpus
    """
    swb_trans_and_dict = DownloadSwitchboardTranscriptionAndDictJob()
    swb_trans_and_dict.add_alias(
        os.path.join(subdir_prefix, "download_trans_and_dict_job")
    )

    speakers_list = get_speakers_list_legacy(subdir_prefix=subdir_prefix)
    corpus = CreateSwitchboardBlissCorpusJob(
        audio_dir=audio_dir,
        trans_dir=swb_trans_and_dict.out_trans_dir,
        speakers_list_file=speakers_list,
        skip_empty_ldc_file=skip_empty_ldc_file,
    )
    corpus.add_alias(
        os.path.join(
            subdir_prefix,
            "create_train_corpus_job_skip_empty_%s" % str(skip_empty_ldc_file),
        )
    )

    return corpus.out_corpus


def get_train_bliss_corpus_ldc(subdir_prefix: str = SUBDIR_PREFIX) -> tk.Path:
    """
    Switchboard-1 training corpus based on the original LDC file
    Uses i6-custom text processing, and the transcriptions are fully
    identical to the "i6-legacy" version.

    :param subdir_prefix:
    :return:
    """
    swb_trans_and_dict = DownloadSwitchboardTranscriptionAndDictJob()
    swb_trans_and_dict.add_alias(
        os.path.join(subdir_prefix, "download_trans_and_dict_job")
    )

    audio_dir = SwitchboardSphereToWaveJob(
        sph_audio_folder=SWITCHBOARD1_PATH,
    ).out_wave_audio_folder

    speaker_list_file = get_speakers_list_ldc(subdir_prefix)

    corpus = CreateSwitchboardBlissCorpusJob(
        audio_dir=audio_dir,
        trans_dir=swb_trans_and_dict.out_trans_dir,
        speakers_list_file=speaker_list_file,
        skip_empty_ldc_file=True,
        lowercase=True,
    )
    corpus.add_alias(os.path.join(subdir_prefix, "create_train_corpus_job"))

    return corpus.out_corpus


def get_train_bliss_corpus_i6_legacy(subdir_prefix: str = SUBDIR_PREFIX) -> tk.Path:
    """
    i6-internal Switchboard-1 training corpus

    :param subdir_prefix:
    :return:
    """
    subdir_prefix = os.path.join(subdir_prefix, "Switchboard-i6-legacy")
    train_bliss_corpus = get_train_bliss_corpus(
        SWITCHBOARD1_LEGACY_PATH, subdir_prefix=subdir_prefix
    )
    return train_bliss_corpus


def get_train_corpus_object_ldc(subdir_prefix: str = SUBDIR_PREFIX):
    """
    :param subdir_prefix:
    :return:
    """
    return CorpusObject(
        corpus_file=get_train_bliss_corpus_ldc(subdir_prefix=subdir_prefix),
        audio_format="wav",
        duration=durations["train"]
    )


def get_train_corpus_object_i6_legacy(
    subdir_prefix: str = SUBDIR_PREFIX,
) -> CorpusObject:
    """
    :param subdir_prefix:
    :return:
    """
    return CorpusObject(
        corpus_file= get_train_bliss_corpus_i6_legacy(subdir_prefix=subdir_prefix),
        audio_format="wav",
        duration=311.78
    )


def get_speakers_list_legacy(subdir_prefix: str = SUBDIR_PREFIX) -> tk.Path:
    """
    Returns speakers list for the legacy setup

    :param str subdir_prefix: alias name prefix
    :return: Path to switchboard recording to speakers mapping list
    :rtype: tk.Path
    """
    speakers_stats = DownloadSwitchboardSpeakersStatsJob()
    speakers_stats.add_alias(os.path.join(subdir_prefix, "download_speakers_stats_job"))

    speakers_list = CreateSwitchboardSpeakersListJob(speakers_stats.out_file)
    speakers_list.add_alias(os.path.join(subdir_prefix, "create_speakers_list_job"))

    return speakers_list.out_speakers_list


def get_speakers_list_ldc(subdir_prefix: str = SUBDIR_PREFIX) -> tk.Path:
    """
    Returns speaker list computed from the LDC-Switchboard documentation files

    :param str subdir_prefix: alias name prefix
    :return: Path to switchboard recording to speakers mapping list
    """
    caller_tab_job = DownloadJob(
        url="https://catalog.ldc.upenn.edu/docs/LDC97S62/caller_tab.csv",
        checksum="c0c73336973403e95f47f347a2b5c43e344f94408f8782aa8812cfb2a7688442",
    )
    caller_tab_job.add_alias(os.path.join(subdir_prefix, "download_caller_tab"))
    caller_tab = caller_tab_job.out_file
    # overwrite the hash so that a change in the URL does not break the setup
    caller_tab.hash_overwrite = "LDC-swbd1-release2-caller_tab.csv"

    conv_tab_job = DownloadJob(
        url="https://catalog.ldc.upenn.edu/docs/LDC97S62/conv_tab.csv",
        checksum="029b32739c05e2589564de84a38a65458959c375a2e829457df82b5b0db71165",
    )
    conv_tab_job.add_alias(os.path.join(subdir_prefix, "download_conv_tab"))
    conv_tab = conv_tab_job.out_file
    conv_tab.hash_overwrite = "LDC-swbd1-release2-conv_tab.csv"

    speaker_list_job = CreateLDCSwitchboardSpeakerListJob(
        caller_tab_file=caller_tab, conv_tab_file=conv_tab
    )
    speaker_list_job.add_alias(os.path.join(subdir_prefix, "create_ldc_speaker_list"))
    return speaker_list_job.out_speakers_list
