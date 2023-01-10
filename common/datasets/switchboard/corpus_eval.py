"""
All functions and classes related to switchboard evaluation sets
"""
__all__ = ["SwitchboardEvalDataset", "get_hub5e00", "get_hub5e01", "get_rt03s"]
from functools import lru_cache
from dataclasses import dataclass
from sisyphus import tk

from i6_core.datasets.switchboard import (
    SwitchboardSphereToWaveJob,
    CreateHub5e00CorpusJob,
    CreateHub5e01CorpusJob,
    CreateRT03sCTSCorpusJob,
)
from i6_experiments.common.datasets.switchboard.constants import durations
from i6_experiments.common.datasets.switchboard.paths import (
    HUB5E00_SPH_PATH,
    HUB5E00_TRANSCRIPT_PATH,
    HUB5E01_PATH,
    RT03S_PATH,
)
from i6_experiments.common.datasets.util import CorpusObject


@dataclass(frozen=True)
class SwitchboardEvalDataset:
    """
    A dataclass helper to unify the objects belonging to a single evaluation set from Switchboard
    """

    bliss_corpus: tk.Path
    stm: tk.Path
    glm: tk.Path


@lru_cache()
def get_hub5e00() -> SwitchboardEvalDataset:
    """
    :return: hub5e00 eval dataset
    """
    hub5e00_wav_audio = SwitchboardSphereToWaveJob(
        sph_audio_folder=HUB5E00_SPH_PATH
    ).out_wave_audio_folder
    hub5e00_job = CreateHub5e00CorpusJob(
        wav_audio_folder=hub5e00_wav_audio,
        hub5_transcription_folder=HUB5E00_TRANSCRIPT_PATH,
    )
    return SwitchboardEvalDataset(
        bliss_corpus=hub5e00_job.out_bliss_corpus,
        stm=hub5e00_job.out_stm,
        glm=hub5e00_job.out_glm,
    )


@lru_cache()
def get_hub5e00_corpus_object() -> CorpusObject:
    """
    :return: hub5e00 corpus object
    """
    hub5e00 = get_hub5e00()
    return CorpusObject(
        corpus_file=hub5e00.bliss_corpus,
        audio_format="wav",
        duration=durations["hub5e00"],
    )


@lru_cache()
def get_hub5e01() -> SwitchboardEvalDataset:
    """
    :return: hub5e_01 eval dataset
    """
    hub5e01_wav_audio = SwitchboardSphereToWaveJob(
        sph_audio_folder=HUB5E01_PATH
    ).out_wave_audio_folder
    hub5e01_job = CreateHub5e01CorpusJob(
        wav_audio_folder=hub5e01_wav_audio, hub5e01_folder=HUB5E01_PATH
    )
    glm = get_hub5e00().glm  # same glm as for hub5e_00
    return SwitchboardEvalDataset(
        bliss_corpus=hub5e01_job.out_bliss_corpus, stm=hub5e01_job.out_stm, glm=glm
    )


@lru_cache()
def get_hub5e01_corpus_object() -> CorpusObject:
    """
    :return: hub5e01 corpus object
    """
    hub5e01 = get_hub5e01()
    return CorpusObject(
        corpus_file=hub5e01.bliss_corpus,
        audio_format="wav",
        duration=durations["hub5e01"],
    )


@lru_cache
def get_rt03s() -> SwitchboardEvalDataset:
    """
    Note: rt03s includes segments with empty transcriptions.

    :return: rt03s eval dataset
    """
    rt03s_wav_audio = SwitchboardSphereToWaveJob(
        sph_audio_folder=RT03S_PATH.join_right("data/audio/eval03/english/cts")
    ).out_wave_audio_folder
    rt03s_job = CreateRT03sCTSCorpusJob(
        wav_audio_folder=rt03s_wav_audio,
        rt03_folder=RT03S_PATH,
    )
    return SwitchboardEvalDataset(
        bliss_corpus=rt03s_job.out_bliss_corpus,
        stm=rt03s_job.out_stm,
        glm=rt03s_job.out_glm,
    )


@lru_cache()
def get_rt03s_corpus_object() -> CorpusObject:
    """
    :return: rt03s corpus object
    """
    rt03s = get_rt03s()
    return CorpusObject(
        corpus_file=rt03s.bliss_corpus, audio_format="wav", duration=durations["rt03s"]
    )
